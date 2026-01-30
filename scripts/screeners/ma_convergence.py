"""
이평선 수렴 스크리너

이동평균선이 수렴하면서 정배열을 유지하는 종목을 발굴합니다.
상승 추세에서 조정 구간에 진입한 종목을 찾는 기법입니다.

[대상]
    관리종목, 거래정지, ETF, ETN, 스팩 제외

[조건]
    A. 거래량: 100,000주 이상
    B. 신고가: [주봉] 10봉전 기준, 30봉 이내 100봉 신고가
    C. 이동평균배열: [주봉] 10 > 20 > 60 정배열
    D. 이격도 ①: [일] 20이평 vs 60이평: 5% 이내
    E. 이격도 ②: [일] 20이평 vs 120이평: 5% 이내
    F. 이격도 ③: [일] 60이평 vs 120이평: 5% 이내
    G. 현재 위치: [일] 종가 vs 20이평: 5% 이내

    조합: A AND B AND C AND D AND E AND F AND G
"""

import FinanceDataReader as fdr
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
from typing import Optional, List, Dict
from dataclasses import dataclass, asdict

# 상수
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DATA_PATH = os.path.join(ROOT_DIR, 'data')

# 스크리닝 조건
MIN_VOLUME = 100000          # 최소 거래량 (주)
MA_GAP_THRESHOLD = 5.0       # 이격도 허용 범위 (%)
NEW_HIGH_LOOKBACK = 100      # 신고가 기준 봉 수 (주봉)
NEW_HIGH_WITHIN = 30         # 신고가 발생 기간 (주봉)
NEW_HIGH_AFTER = 10          # 신고가 이후 최소 봉 수 (주봉)


@dataclass
class MAConvergenceResult:
    """이평선 수렴 스크리너 결과"""
    ticker: str
    name: str
    current_price: int
    volume: int                 # 거래량
    ma10w: int                  # 10주선
    ma20w: int                  # 20주선
    ma60w: int                  # 60주선
    ma20d: int                  # 20일선
    ma60d: int                  # 60일선
    ma120d: int                 # 120일선
    gap_20_60: float            # 20일 vs 60일 이격도
    gap_20_120: float           # 20일 vs 120일 이격도
    gap_60_120: float           # 60일 vs 120일 이격도
    gap_price_20: float         # 종가 vs 20일 이격도
    weeks_since_high: int       # 신고가 이후 주 수
    market_cap: int             # 시가총액 (억원)
    updated_at: str

    def to_dict(self) -> dict:
        return asdict(self)


def get_stock_list() -> pd.DataFrame:
    """종목 리스트 가져오기 (관리종목, ETF 등 제외)"""
    print("종목 리스트 조회 중...")

    try:
        # 코스피 + 코스닥
        kospi = fdr.StockListing('KOSPI')
        kosdaq = fdr.StockListing('KOSDAQ')

        stocks = pd.concat([kospi, kosdaq], ignore_index=True)

        # 시가총액 컬럼 처리
        if 'Marcap' in stocks.columns:
            stocks['MarketCap'] = stocks['Marcap'] / 100000000
        elif 'MarketCap' in stocks.columns:
            stocks['MarketCap'] = stocks['MarketCap'] / 100000000

        # 필터링: ETF, ETN, 스팩, 리츠 등 제외
        exclude_keywords = ['ETF', 'ETN', '스팩', 'SPAC', '리츠', 'REITs', '인버스', '레버리지']
        for keyword in exclude_keywords:
            stocks = stocks[~stocks['Name'].str.contains(keyword, case=False, na=False)]

        # 우선주 제외 (종목코드 끝이 0이 아닌 것)
        stocks = stocks[stocks['Code'].str[-1] == '0']

        # 시가총액 1000억 이상
        stocks = stocks[stocks['MarketCap'] >= 1000]

        print(f"  시가총액 1000억 이상: {len(stocks)}개 종목")
        return stocks

    except Exception as e:
        print(f"  종목 리스트 조회 실패: {e}")
        return pd.DataFrame()


def get_daily_data(ticker: str, days: int = 200) -> Optional[pd.DataFrame]:
    """일봉 데이터 조회"""
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days + 50)

        df = fdr.DataReader(ticker, start_date.strftime('%Y-%m-%d'))
        if df is not None and len(df) >= 120:
            return df
    except:
        pass
    return None


def get_weekly_data(ticker: str, weeks: int = 150) -> Optional[pd.DataFrame]:
    """주봉 데이터 조회"""
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(weeks=weeks + 20)

        df = fdr.DataReader(ticker, start_date.strftime('%Y-%m-%d'))
        if df is None or len(df) < 100:
            return None

        # 주봉으로 변환
        df_weekly = df.resample('W-FRI').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()

        if len(df_weekly) >= 100:
            return df_weekly
    except:
        pass
    return None


def check_new_high_condition(weekly_df: pd.DataFrame) -> tuple[bool, int]:
    """
    신고가 조건 체크: 10봉전 기준, 30봉 이내 100봉 신고가

    Returns:
        (조건 충족 여부, 신고가 이후 주 수)
    """
    if len(weekly_df) < 100:
        return False, 0

    close = weekly_df['Close'].values
    high = weekly_df['High'].values

    # 최근 100봉 고가
    high_100 = high[-100:].max()

    # 10봉전 ~ 30봉전 구간에서 100봉 신고가가 있는지 확인
    for i in range(10, min(31, len(weekly_df))):
        idx = -i
        period_high = high[max(-100, idx-100):idx+1].max()
        if high[idx] >= period_high * 0.99:  # 1% 오차 허용
            return True, i

    return False, 0


def check_ma_alignment(weekly_df: pd.DataFrame) -> bool:
    """
    이동평균 정배열 체크: [주봉] 10 > 20 > 60
    """
    if len(weekly_df) < 60:
        return False

    close = weekly_df['Close']
    ma10 = close.rolling(10).mean().iloc[-1]
    ma20 = close.rolling(20).mean().iloc[-1]
    ma60 = close.rolling(60).mean().iloc[-1]

    return ma10 > ma20 > ma60


def calculate_gap(value1: float, value2: float) -> float:
    """두 값 사이의 이격도 계산 (%)"""
    if value2 == 0:
        return float('inf')
    return abs(value1 - value2) / value2 * 100


def screen_ma_convergence() -> List[Dict]:
    """이평선 수렴 스크리너 실행"""
    print("\n[이평선 수렴 스크리너] 분석 시작...")

    # 1. 종목 리스트 가져오기
    stocks = get_stock_list()
    if stocks.empty:
        print("  종목 조회 실패")
        return []

    results = []
    stats = {
        'total': len(stocks),
        'no_data': 0,
        'volume_fail': 0,
        'new_high_fail': 0,
        'ma_align_fail': 0,
        'gap_fail': 0,
        'passed': 0
    }

    print(f"  {len(stocks)}개 종목 스캔...")

    for idx, (_, row) in enumerate(stocks.iterrows()):
        ticker = row.get('Code', row.get('Symbol', ''))
        name = row.get('Name', '')
        market_cap = row.get('MarketCap', 0)

        if not ticker:
            continue

        # 진행률 표시
        if (idx + 1) % 100 == 0:
            print(f"  진행 중... {idx + 1}/{len(stocks)}")

        # 일봉 데이터 조회
        daily_df = get_daily_data(ticker)
        if daily_df is None:
            stats['no_data'] += 1
            continue

        # A. 거래량 체크 (최근 거래량)
        recent_volume = daily_df['Volume'].iloc[-1]
        if recent_volume < MIN_VOLUME:
            stats['volume_fail'] += 1
            continue

        # 주봉 데이터 조회
        weekly_df = get_weekly_data(ticker)
        if weekly_df is None:
            stats['no_data'] += 1
            continue

        # B. 신고가 조건 체크
        new_high_ok, weeks_since = check_new_high_condition(weekly_df)
        if not new_high_ok:
            stats['new_high_fail'] += 1
            continue

        # C. 이동평균 정배열 체크 (주봉)
        if not check_ma_alignment(weekly_df):
            stats['ma_align_fail'] += 1
            continue

        # 일봉 이동평균 계산
        close_d = daily_df['Close']
        ma20d = close_d.rolling(20).mean().iloc[-1]
        ma60d = close_d.rolling(60).mean().iloc[-1]
        ma120d = close_d.rolling(120).mean().iloc[-1]
        current_price = close_d.iloc[-1]

        if pd.isna(ma120d) or ma120d <= 0:
            stats['no_data'] += 1
            continue

        # D, E, F, G. 이격도 체크
        gap_20_60 = calculate_gap(ma20d, ma60d)
        gap_20_120 = calculate_gap(ma20d, ma120d)
        gap_60_120 = calculate_gap(ma60d, ma120d)
        gap_price_20 = calculate_gap(current_price, ma20d)

        if (gap_20_60 > MA_GAP_THRESHOLD or
            gap_20_120 > MA_GAP_THRESHOLD or
            gap_60_120 > MA_GAP_THRESHOLD or
            gap_price_20 > MA_GAP_THRESHOLD):
            stats['gap_fail'] += 1
            continue

        # 주봉 이동평균
        close_w = weekly_df['Close']
        ma10w = close_w.rolling(10).mean().iloc[-1]
        ma20w = close_w.rolling(20).mean().iloc[-1]
        ma60w = close_w.rolling(60).mean().iloc[-1]

        # 모든 조건 통과
        stats['passed'] += 1

        result = MAConvergenceResult(
            ticker=ticker,
            name=name,
            current_price=int(current_price),
            volume=int(recent_volume),
            ma10w=int(ma10w),
            ma20w=int(ma20w),
            ma60w=int(ma60w),
            ma20d=int(ma20d),
            ma60d=int(ma60d),
            ma120d=int(ma120d),
            gap_20_60=round(gap_20_60, 2),
            gap_20_120=round(gap_20_120, 2),
            gap_60_120=round(gap_60_120, 2),
            gap_price_20=round(gap_price_20, 2),
            weeks_since_high=weeks_since,
            market_cap=int(market_cap),
            updated_at=datetime.now().isoformat()
        )
        results.append(result.to_dict())

    # 시가총액 순 정렬
    results.sort(key=lambda x: x['market_cap'], reverse=True)

    print(f"\n├─ 총 종목: {stats['total']}개")
    print(f"├─ 데이터 없음: {stats['no_data']}개")
    print(f"├─ 거래량 < 10만주: {stats['volume_fail']}개 탈락")
    print(f"├─ 신고가 조건 미충족: {stats['new_high_fail']}개 탈락")
    print(f"├─ 정배열 미충족: {stats['ma_align_fail']}개 탈락")
    print(f"├─ 이격도 > 5%: {stats['gap_fail']}개 탈락")
    print(f"[완료] 이평선 수렴: {len(results)}개")

    return results


def save_results(results: List[Dict], filename: str = 'ma_convergence.json'):
    """결과를 JSON으로 저장"""
    os.makedirs(DATA_PATH, exist_ok=True)

    output = {
        'meta': {
            'updated_at': datetime.now().isoformat(),
            'total_count': len(results),
            'conditions': {
                'min_volume': f'{MIN_VOLUME:,}주',
                'ma_gap_threshold': f'{MA_GAP_THRESHOLD}%',
                'weekly_ma_alignment': '10 > 20 > 60',
                'new_high': f'{NEW_HIGH_LOOKBACK}봉 내 신고가'
            }
        },
        'data': results
    }

    filepath = os.path.join(DATA_PATH, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"  저장: {filepath}")


def main():
    print("=" * 60)
    print("이평선 수렴 스크리너")
    print("=" * 60)

    # 스크리닝 실행
    results = screen_ma_convergence()

    # 결과 저장
    save_results(results)

    # 결과 출력
    print("\n" + "=" * 90)
    print("이평선 수렴 스크리닝 결과")
    print("=" * 90)
    print(f"{'순위':<4} {'종목명':<14} {'현재가':>10} {'20/60':>8} {'20/120':>8} {'60/120':>8} {'종가/20':>8}")
    print("-" * 90)

    for i, r in enumerate(results[:20]):
        print(f"{i+1:<4} {r['name']:<14} {r['current_price']:>10,}원 {r['gap_20_60']:>7.1f}% {r['gap_20_120']:>7.1f}% {r['gap_60_120']:>7.1f}% {r['gap_price_20']:>7.1f}%")

    print(f"\n총 {len(results)}개 종목")


if __name__ == '__main__':
    main()
