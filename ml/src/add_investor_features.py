"""
수급 피처 추가: 외국인/기관 순매수 데이터

네이버 금융에서 투자자별 매매동향 크롤링
"""

import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import time
import pickle
from pathlib import Path
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

# 경로 설정
ML_DIR = Path(__file__).parent.parent
DATA_PATH = ML_DIR / "data" / "training_data_v2.csv"
OUTPUT_PATH = ML_DIR / "data" / "training_data_v3.csv"
CACHE_DIR = ML_DIR / ".cache" / "investor"

CACHE_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
}


def load_cache(ticker):
    """캐시 로드"""
    path = CACHE_DIR / f"{ticker}.pkl"
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


def save_cache(ticker, data):
    """캐시 저장"""
    path = CACHE_DIR / f"{ticker}.pkl"
    with open(path, "wb") as f:
        pickle.dump(data, f)


def parse_number(text):
    """숫자 파싱"""
    if not text:
        return None
    text = text.replace(',', '').replace('+', '').strip()
    try:
        return int(text)
    except:
        return None


def fetch_investor_data(ticker, pages=3):
    """네이버 금융에서 외국인/기관 순매수 데이터 크롤링"""

    # 캐시 확인
    cached = load_cache(ticker)
    if cached is not None:
        return cached

    all_data = []

    for page in range(1, pages + 1):
        url = f'https://finance.naver.com/item/frgn.naver?code={ticker}&page={page}'

        try:
            resp = requests.get(url, headers=HEADERS, timeout=10)
            soup = BeautifulSoup(resp.text, 'html.parser')

            # type2 테이블 중 데이터 테이블 찾기
            tables = soup.select('table.type2')

            for table in tables:
                rows = table.select('tr')

                # 헤더 확인
                header = rows[0].get_text() if rows else ""
                if '날짜' not in header and '외국인' not in header:
                    continue

                # 데이터 행 파싱
                for row in rows[2:]:  # 헤더 2줄 스킵
                    cols = row.select('td')
                    if len(cols) < 7:
                        continue

                    date_text = cols[0].get_text(strip=True)
                    if '.' not in date_text:
                        continue

                    try:
                        # 날짜 파싱 (2026.01.30 형식)
                        date = pd.to_datetime(date_text, format='%Y.%m.%d')

                        # 순매수 데이터 (5번째: 기관, 6번째: 외국인)
                        inst_net = parse_number(cols[5].get_text(strip=True))
                        foreign_net = parse_number(cols[6].get_text(strip=True))

                        if inst_net is not None and foreign_net is not None:
                            all_data.append({
                                'date': date,
                                'inst_net': inst_net,
                                'foreign_net': foreign_net
                            })
                    except:
                        continue

            time.sleep(0.1)

        except Exception as e:
            continue

    if all_data:
        df = pd.DataFrame(all_data)
        df = df.sort_values('date').drop_duplicates('date')
        save_cache(ticker, df)
        return df

    return None


def calculate_investor_features(ticker, breakout_date, lookback=5):
    """
    특정 날짜 기준 수급 피처 계산

    피처:
    - foreign_net_5d: 5일간 외국인 순매수 합계
    - inst_net_5d: 5일간 기관 순매수 합계
    - foreign_net_avg: 5일 평균 외국인 순매수
    - inst_net_avg: 5일 평균 기관 순매수
    - foreign_trend: 외국인 순매수 추세 (최근 vs 이전)
    - inst_trend: 기관 순매수 추세
    """
    features = {
        'foreign_net_5d': 0,
        'inst_net_5d': 0,
        'foreign_net_avg': 0,
        'inst_net_avg': 0,
        'foreign_trend': 0,
        'inst_trend': 0
    }

    investor_df = fetch_investor_data(ticker)
    if investor_df is None or investor_df.empty:
        return features

    # 돌파일 기준 이전 데이터 필터링
    try:
        breakout_dt = pd.to_datetime(breakout_date)
        mask = investor_df['date'] <= breakout_dt
        recent_data = investor_df[mask].tail(lookback * 2)  # 추세 계산용으로 2배

        if len(recent_data) < lookback:
            return features

        # 최근 5일 데이터
        last_5 = recent_data.tail(lookback)

        # 5일간 합계
        features['foreign_net_5d'] = last_5['foreign_net'].sum()
        features['inst_net_5d'] = last_5['inst_net'].sum()

        # 5일 평균
        features['foreign_net_avg'] = last_5['foreign_net'].mean()
        features['inst_net_avg'] = last_5['inst_net'].mean()

        # 추세 (최근 5일 vs 이전 5일)
        if len(recent_data) >= lookback * 2:
            prev_5 = recent_data.head(lookback)

            prev_foreign_avg = prev_5['foreign_net'].mean()
            prev_inst_avg = prev_5['inst_net'].mean()

            # 추세: 증가하면 양수, 감소하면 음수
            if prev_foreign_avg != 0:
                features['foreign_trend'] = (features['foreign_net_avg'] - prev_foreign_avg) / abs(prev_foreign_avg)

            if prev_inst_avg != 0:
                features['inst_trend'] = (features['inst_net_avg'] - prev_inst_avg) / abs(prev_inst_avg)

    except Exception as e:
        pass

    return features


def process_single_row(args):
    """단일 행 처리 (병렬 처리용)"""
    idx, row = args
    ticker = row['ticker']
    breakout_date = row['breakout_date']

    features = calculate_investor_features(ticker, breakout_date)
    return idx, features


def main():
    print("=" * 60)
    print("수급 피처 추가")
    print("=" * 60)

    # 데이터 로드
    print("\n[1/4] 기존 데이터 로드...")
    df = pd.read_csv(DATA_PATH)
    print(f"  샘플 수: {len(df)}")

    # 고유 종목 수
    unique_tickers = df['ticker'].unique()
    print(f"  고유 종목 수: {len(unique_tickers)}")

    # 새 피처 컬럼 초기화
    new_features = ['foreign_net_5d', 'inst_net_5d', 'foreign_net_avg',
                    'inst_net_avg', 'foreign_trend', 'inst_trend']
    for feat in new_features:
        df[feat] = 0.0

    # 수급 데이터 크롤링 및 피처 계산
    print("\n[2/4] 수급 데이터 크롤링 중...")
    print("  (캐시된 데이터는 재사용)")

    total = len(df)
    success = 0

    # 순차 처리 (API 제한 고려)
    for idx, row in df.iterrows():
        if (idx + 1) % 100 == 0 or idx == total - 1:
            print(f"  진행: {idx+1}/{total} ({(idx+1)/total*100:.1f}%), 성공: {success}")

        ticker = row['ticker']
        breakout_date = row['breakout_date']

        features = calculate_investor_features(ticker, breakout_date)

        # 피처 업데이트
        for feat_name, feat_value in features.items():
            df.at[idx, feat_name] = feat_value

        if features['foreign_net_5d'] != 0 or features['inst_net_5d'] != 0:
            success += 1

        # API 제한 방지
        if idx % 50 == 0:
            time.sleep(0.5)

    print(f"\n  수급 데이터 수집 완료: {success}/{total} ({success/total*100:.1f}%)")

    # 피처 정규화 (스케일 조정)
    print("\n[3/4] 피처 정규화...")

    # 거래량 스케일 조정 (만 단위로)
    for col in ['foreign_net_5d', 'inst_net_5d', 'foreign_net_avg', 'inst_net_avg']:
        df[col] = df[col] / 10000  # 만 단위로 변환

    # 추세는 -1 ~ 1 범위로 클리핑
    for col in ['foreign_trend', 'inst_trend']:
        df[col] = df[col].clip(-2, 2)

    # 저장
    print("\n[4/4] 저장 중...")
    df.to_csv(OUTPUT_PATH, index=False, encoding='utf-8-sig')
    print(f"  저장 완료: {OUTPUT_PATH}")

    # 결과 요약
    print("\n" + "=" * 60)
    print("수급 피처 요약")
    print("=" * 60)

    for feat in new_features:
        non_zero = (df[feat] != 0).sum()
        mean_val = df[feat].mean()
        print(f"  {feat:20}: 비영값 {non_zero}개 ({non_zero/total*100:.1f}%), 평균 {mean_val:.2f}")

    # 타겟과 상관관계
    print("\n" + "=" * 60)
    print("타겟(max_gain_20d)과 상관관계")
    print("=" * 60)

    for feat in new_features:
        corr = df[feat].corr(df['max_gain_20d'])
        bar = "#" * int(abs(corr) * 50) if not pd.isna(corr) else ""
        sign = "+" if corr >= 0 else "-"
        print(f"  {feat:20}: {sign}{abs(corr):.4f} {bar}")

    return df


if __name__ == "__main__":
    start_time = datetime.now()
    df = main()
    elapsed = datetime.now() - start_time
    print(f"\n완료! (소요시간: {elapsed})")
