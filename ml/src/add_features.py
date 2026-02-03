"""
피처 추가: 시장 레짐, 시장 MA200 거리

기존 volume_surge는 이미 원본에 있음 (유지)
추가 피처:
1. market_regime - 시장 레짐 (지수 > 200MA → 1)
2. market_ma200_dist - 지수의 200MA 대비 거리 (%)
"""

import pandas as pd
import numpy as np
import time
import pickle
from pathlib import Path
from datetime import datetime
import FinanceDataReader as fdr

# 경로 설정
ML_DIR = Path(__file__).parent.parent
DATA_PATH = ML_DIR / "data" / "training_data.csv"
OUTPUT_PATH = ML_DIR / "data" / "training_data_v2.csv"
CACHE_DIR = ML_DIR / ".cache"


def load_cache(name):
    """캐시 로드"""
    path = CACHE_DIR / f"{name}.pkl"
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


def save_cache(name, data):
    """캐시 저장"""
    path = CACHE_DIR / f"{name}.pkl"
    with open(path, "wb") as f:
        pickle.dump(data, f)


def get_market_index_with_ma200():
    """시장 지수 + 200MA 데이터"""
    print("  시장 지수 데이터 로드...")

    result = {}
    for market, symbol in [('KOSPI', 'KS11'), ('KOSDAQ', 'KQ11')]:
        cache_name = f"index_{market}_ma200"
        cached = load_cache(cache_name)

        if cached is not None:
            result[market] = cached
            continue

        # 새로 다운로드
        df = fdr.DataReader(symbol, '2020-01-01', '2025-06-30')
        time.sleep(0.1)

        if df.empty:
            continue

        df = df.rename(columns={"Close": "종가"})

        # 200일 이동평균 계산
        df['MA200'] = df['종가'].rolling(window=200).mean()
        df['regime'] = (df['종가'] > df['MA200']).astype(int)
        df['ma200_dist'] = (df['종가'] / df['MA200'] - 1) * 100

        save_cache(cache_name, df)
        result[market] = df

    return result


def add_features_to_data():
    """기존 데이터에 피처 추가"""
    print("=" * 60)
    print("피처 추가 시작")
    print("=" * 60)

    # 기존 데이터 로드
    print("\n[1/3] 기존 데이터 로드...")
    df = pd.read_csv(DATA_PATH)
    print(f"  → {len(df)}개 샘플")
    print(f"  → volume_surge 있음: 평균 {df['volume_surge'].mean():.2f}")

    # 시장 지수 데이터
    print("\n[2/3] 시장 지수 데이터 준비...")
    market_data = get_market_index_with_ma200()
    for market, data in market_data.items():
        print(f"  → {market}: {len(data)}일, MA200 시작: {data['MA200'].first_valid_index()}")

    # 새 피처 컬럼 초기화
    df['market_regime'] = 0
    df['market_ma200_dist'] = 0.0

    # 피처 계산
    print("\n[3/3] 피처 계산 중...")
    total = len(df)

    for idx, row in df.iterrows():
        if (idx + 1) % 1000 == 0 or idx == total - 1:
            print(f"  진행: {idx+1}/{total} ({(idx+1)/total*100:.1f}%)")

        market_type = row['market_type']
        date_str = row['breakout_date']

        # 날짜 변환
        try:
            date = pd.Timestamp(date_str)
        except:
            continue

        # 시장 데이터 가져오기
        market_df = market_data.get(market_type)
        if market_df is None:
            continue

        # 해당 날짜 또는 가장 가까운 이전 날짜 찾기
        try:
            if date in market_df.index:
                target_date = date
            else:
                # 이전 날짜 중 가장 가까운 것
                earlier_dates = market_df.index[market_df.index <= date]
                if len(earlier_dates) == 0:
                    continue
                target_date = earlier_dates[-1]

            # market_regime
            regime = market_df.loc[target_date, 'regime']
            if pd.notna(regime):
                df.at[idx, 'market_regime'] = int(regime)

            # market_ma200_dist
            ma200_dist = market_df.loc[target_date, 'ma200_dist']
            if pd.notna(ma200_dist):
                df.at[idx, 'market_ma200_dist'] = round(ma200_dist, 4)

        except Exception as e:
            continue

    # 저장
    df.to_csv(OUTPUT_PATH, index=False, encoding='utf-8-sig')
    print(f"\n저장 완료: {OUTPUT_PATH}")

    # 결과 요약
    print("\n" + "=" * 60)
    print("피처 요약")
    print("=" * 60)

    print(f"\n기존 피처:")
    print(f"  volume_surge      : 평균 {df['volume_surge'].mean():.2f}, 중앙값 {df['volume_surge'].median():.2f}")

    print(f"\n추가 피처:")
    print(f"  market_regime     : 상승장 {df['market_regime'].sum()}개 ({df['market_regime'].mean()*100:.1f}%)")
    print(f"  market_ma200_dist : 평균 {df['market_ma200_dist'].mean():.2f}%")

    # 타겟과 상관관계
    print("\n" + "=" * 60)
    print("타겟(max_gain_20d)과 상관관계")
    print("=" * 60)

    feature_cols = [
        'ma20_deviation', 'breakout_pct', 'rs_vs_market', 'pct_above_52w_low',
        'volume_surge', 'atr_ratio', 'volume_dry_up', 'volatility_contraction',
        'close_strength', 'base_length', 'days_since_ath', 'ma200_slope',
        'liquidity', 'market_return', 'market_regime', 'market_ma200_dist'
    ]

    correlations = []
    for col in feature_cols:
        if col in df.columns:
            corr = df[col].corr(df['max_gain_20d'])
            correlations.append((col, corr))

    # 절대값 기준 정렬
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("\n피처별 상관관계 (절대값 순):")
    print("-" * 50)
    for col, corr in correlations:
        bar = "#" * int(abs(corr) * 50)
        sign = "+" if corr >= 0 else "-"
        print(f"  {col:22} : {sign}{abs(corr):.4f} {bar}")

    return df


def main():
    start_time = datetime.now()
    df = add_features_to_data()
    elapsed = datetime.now() - start_time
    print(f"\n완료! (소요시간: {elapsed})")


if __name__ == "__main__":
    main()
