"""
박스권 돌파 학습 데이터 수집 스크립트

2019-01-01 ~ 2025-06-30 기간 동안
박스권 돌파 조건에 걸린 종목들의 데이터를 수집합니다.

조건 (run_screeners.py와 동일):
- 60거래일 박스권 (변동폭 25% 이내)
- 종가 > 저항선 × 1.015 (1.5% 돌파)

타겟:
- 20거래일 최고 수익률
- 15% 이상 성공 여부 (이진 분류)

피처: 15개
"""

import pandas as pd
import numpy as np
import FinanceDataReader as fdr
from datetime import datetime, timedelta
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 설정
# ============================================================================
START_DATE = '2019-01-01'
END_DATE = '2025-06-30'

# 박스권 조건 (run_screeners.py와 동일)
BOX_PERIOD = 60
MAX_BOX_RANGE = 25.0
BREAKOUT_THRESHOLD = 1.015  # 1.5% 돌파

# 타겟 변수
TARGET_DAYS = 20
SUCCESS_THRESHOLD = 15.0  # 15% 이상

# 경로
OUTPUT_DIR = Path(__file__).parent / "data"
OUTPUT_DIR.mkdir(exist_ok=True)
OUTPUT_FILE = OUTPUT_DIR / "box_breakout_history.csv"

# 진행 상황 저장
CHECKPOINT_FILE = OUTPUT_DIR / "checkpoint_box_breakout.txt"
BATCH_SIZE = 100  # 100개 종목마다 저장

# KOSPI 데이터 (시장 수익률 계산용)
kospi_data = None


# ============================================================================
# 데이터 로드
# ============================================================================

def load_kospi_data():
    """KOSPI 지수 데이터 로드"""
    global kospi_data
    print("[*] KOSPI 데이터 로딩 중...")
    kospi_data = fdr.DataReader('KS11', START_DATE, END_DATE)
    print(f"[OK] KOSPI 데이터 로드 완료: {len(kospi_data)}일")


def get_stock_list():
    """KRX 전체 종목 리스트"""
    print("[*] 종목 리스트 로딩 중...")
    stocks = fdr.StockListing('KRX')

    # 시가총액 1000억 이상만 (노이즈 제거)
    stocks = stocks[stocks['Marcap'] >= 100_000_000_000].copy()

    print(f"[OK] 종목 리스트 로드 완료: {len(stocks)}개")
    return stocks


# ============================================================================
# 박스권 돌파 체크
# ============================================================================

def check_box_breakout(df, idx):
    """
    특정 인덱스에서 박스권 돌파 조건 확인

    Args:
        df: OHLCV 데이터
        idx: 체크할 인덱스

    Returns:
        (True, resistance, box_low) or (False, None, None)
    """
    # 최소 데이터 필요
    if idx < BOX_PERIOD:
        return False, None, None

    # 박스권 데이터 (idx-60 ~ idx-1)
    box_data = df.iloc[idx - BOX_PERIOD:idx]

    if len(box_data) < BOX_PERIOD:
        return False, None, None

    box_high = box_data['Close'].max()
    box_low = box_data['Close'].min()

    if box_low <= 0:
        return False, None, None

    # 1. 박스권 조건: 변동폭 25% 이내
    box_range_pct = (box_high - box_low) / box_low * 100
    if box_range_pct > MAX_BOX_RANGE:
        return False, None, None

    # 2. 돌파 조건: 종가 > 저항선 * 1.015
    current_close = df.iloc[idx]['Close']
    if current_close <= box_high * BREAKOUT_THRESHOLD:
        return False, None, None

    return True, box_high, box_low


# ============================================================================
# 타겟 변수 계산
# ============================================================================

def calculate_targets(df, breakout_idx):
    """
    타겟 변수 계산

    Args:
        df: OHLCV 데이터
        breakout_idx: 돌파일 인덱스

    Returns:
        (max_gain_20d, success) or (None, None)
    """
    # 20거래일 후 데이터 필요
    if breakout_idx + TARGET_DAYS >= len(df):
        return None, None

    breakout_close = df.iloc[breakout_idx]['Close']

    # 돌파일 다음날부터 20거래일 동안의 최고가
    future_data = df.iloc[breakout_idx + 1:breakout_idx + 1 + TARGET_DAYS]

    if len(future_data) < TARGET_DAYS:
        return None, None

    max_high = future_data['High'].max()
    max_gain_20d = (max_high / breakout_close - 1) * 100

    # 성공 여부: 15% 이상
    success = 1 if max_gain_20d >= SUCCESS_THRESHOLD else 0

    return round(max_gain_20d, 2), success


# ============================================================================
# 피처 계산
# ============================================================================

def calculate_features(df, breakout_idx, resistance, support):
    """
    15개 피처 계산

    모든 피처는 돌파일 기준으로 계산 (미래 정보 누설 방지)
    """
    try:
        close = df['Close']
        high = df['High']
        low = df['Low']
        volume = df['Volume']

        breakout_close = close.iloc[breakout_idx]
        breakout_high = high.iloc[breakout_idx]
        breakout_low = low.iloc[breakout_idx]
        breakout_open = df.iloc[breakout_idx]['Open']

        features = {}

        # 1. box_range_pct - 박스권 크기
        features['box_range_pct'] = (resistance - support) / support * 100

        # 2. breakout_strength - 돌파 강도
        features['breakout_strength'] = (breakout_close / resistance - 1) * 100

        # 3. volume_surge - 거래량 급증
        if breakout_idx >= 20:
            vol_20d_avg = volume.iloc[breakout_idx - 20:breakout_idx].mean()
            features['volume_surge'] = volume.iloc[breakout_idx] / vol_20d_avg if vol_20d_avg > 0 else 1.0
        else:
            features['volume_surge'] = 1.0

        # 4. volume_dry_up - 박스권 내 거래량 감소
        if breakout_idx >= 40:
            vol_recent = volume.iloc[breakout_idx - 10:breakout_idx].mean()
            vol_past = volume.iloc[breakout_idx - 40:breakout_idx - 10].mean()
            features['volume_dry_up'] = vol_recent / vol_past if vol_past > 0 else 1.0
        else:
            features['volume_dry_up'] = 1.0

        # 5. close_strength - 종가 강도
        if breakout_high != breakout_low:
            features['close_strength'] = (breakout_close - breakout_low) / (breakout_high - breakout_low)
        else:
            features['close_strength'] = 0.5

        # 6. volatility_contraction - 변동성 수축
        if breakout_idx >= 60:
            recent_close = close.iloc[breakout_idx - 20:breakout_idx]
            past_close = close.iloc[breakout_idx - 60:breakout_idx - 20]

            recent_vol = recent_close.std() / recent_close.mean() if recent_close.mean() > 0 else 0
            past_vol = past_close.std() / past_close.mean() if past_close.mean() > 0 else 0

            features['volatility_contraction'] = recent_vol / past_vol if past_vol > 0 else 1.0
        else:
            features['volatility_contraction'] = 1.0

        # 7. ma20_deviation - 20일선 이격도
        if breakout_idx >= 20:
            ma20 = close.iloc[breakout_idx - 20:breakout_idx].mean()
            features['ma20_deviation'] = (breakout_close / ma20 - 1) * 100 if ma20 > 0 else 0
        else:
            features['ma20_deviation'] = 0

        # 8. breakout_gap - 돌파일 갭
        if breakout_idx >= 1:
            prev_close = close.iloc[breakout_idx - 1]
            features['breakout_gap'] = (breakout_open / prev_close - 1) * 100 if prev_close > 0 else 0
        else:
            features['breakout_gap'] = 0

        # 9. ma200_slope - 200일선 기울기
        if breakout_idx >= 220:
            ma200_now = close.iloc[breakout_idx - 200:breakout_idx].mean()
            ma200_20d_ago = close.iloc[breakout_idx - 220:breakout_idx - 20].mean()
            features['ma200_slope'] = (ma200_now / ma200_20d_ago - 1) * 100 if ma200_20d_ago > 0 else 0
        else:
            features['ma200_slope'] = 0

        # 10. pct_above_52w_low - 52주 저점 대비
        lookback_start = max(0, breakout_idx - 250)
        low_52w = low.iloc[lookback_start:breakout_idx].min()
        features['pct_above_52w_low'] = (breakout_close / low_52w - 1) * 100 if low_52w > 0 else 0

        # 11. days_since_ath - 최고가 이후 경과일
        high_52w_idx = high.iloc[lookback_start:breakout_idx].idxmax()
        breakout_date = df.index[breakout_idx]
        features['days_since_ath'] = (breakout_date - high_52w_idx).days

        # 12. rs_vs_market - 시장 대비 상대강도
        # 13. market_return - 시장 수익률
        if kospi_data is not None and breakout_idx >= 20:
            try:
                breakout_date = df.index[breakout_idx]
                kospi_idx = kospi_data.index.get_indexer([breakout_date], method='nearest')[0]

                if kospi_idx >= 20:
                    market_return = (kospi_data['Close'].iloc[kospi_idx] /
                                   kospi_data['Close'].iloc[kospi_idx - 20] - 1) * 100
                    stock_return = (breakout_close / close.iloc[breakout_idx - 20] - 1) * 100

                    features['market_return'] = market_return
                    features['rs_vs_market'] = stock_return - market_return
                else:
                    features['market_return'] = 0
                    features['rs_vs_market'] = 0
            except:
                features['market_return'] = 0
                features['rs_vs_market'] = 0
        else:
            features['market_return'] = 0
            features['rs_vs_market'] = 0

        # 14. atr_ratio - ATR 비율
        if breakout_idx >= 14:
            tr_values = []
            for j in range(breakout_idx - 14, breakout_idx):
                h = high.iloc[j]
                l = low.iloc[j]
                c_prev = close.iloc[j - 1] if j > 0 else close.iloc[j]

                tr = max(h - l, abs(h - c_prev), abs(l - c_prev))
                tr_values.append(tr)

            atr_14 = np.mean(tr_values)
            features['atr_ratio'] = atr_14 / breakout_close * 100 if breakout_close > 0 else 0
        else:
            features['atr_ratio'] = 0

        # 15. liquidity - 유동성 (5일 평균 거래대금)
        if breakout_idx >= 5:
            liquidity = (close.iloc[breakout_idx - 5:breakout_idx] *
                        volume.iloc[breakout_idx - 5:breakout_idx]).mean()
            features['liquidity'] = int(liquidity)
        else:
            features['liquidity'] = 0

        # 반올림
        for key in features:
            if key != 'liquidity' and isinstance(features[key], float):
                features[key] = round(features[key], 4)

        return features

    except Exception as e:
        return None


# ============================================================================
# 개별 종목 분석
# ============================================================================

def analyze_stock(ticker, name, market_cap):
    """
    개별 종목의 박스권 돌파 이력 수집

    Returns:
        List[dict]: 돌파 이력 리스트
    """
    results = []

    try:
        # 데이터 로드 (여유있게)
        df = fdr.DataReader(ticker, START_DATE, END_DATE)

        if df is None or len(df) < BOX_PERIOD + TARGET_DAYS + 50:
            return results

        # 각 거래일마다 박스권 돌파 체크
        # BOX_PERIOD부터 시작, 마지막 TARGET_DAYS는 타겟 계산 불가
        for idx in range(BOX_PERIOD, len(df) - TARGET_DAYS):

            # 박스권 돌파 체크
            is_breakout, resistance, support = check_box_breakout(df, idx)

            if not is_breakout:
                continue

            # 타겟 변수 계산
            max_gain, success = calculate_targets(df, idx)

            if max_gain is None:
                continue

            # 피처 계산
            features = calculate_features(df, idx, resistance, support)

            if features is None:
                continue

            # 데이터 저장
            breakout_date = df.index[idx]
            breakout_close = df.iloc[idx]['Close']

            result = {
                'ticker': ticker,
                'name': name,
                'market_cap': int(market_cap),
                'breakout_date': breakout_date.strftime('%Y-%m-%d'),
                'resistance': int(resistance),
                'support': int(support),
                'breakout_close': int(breakout_close),
                'max_gain_20d': max_gain,
                'success': success,
                **features
            }

            results.append(result)

    except Exception as e:
        pass

    return results


# ============================================================================
# 메인 수집 함수
# ============================================================================

def collect_data():
    """메인 데이터 수집 함수"""

    print("=" * 80)
    print("[START] 박스권 돌파 학습 데이터 수집 시작")
    print("=" * 80)
    print(f"기간: {START_DATE} ~ {END_DATE}")
    print(f"조건: 60일 박스권 (변동폭 25% 이내) + 1.5% 돌파")
    print(f"타겟: 20거래일 최고 수익률, 15% 이상 성공 여부")
    print()

    # KOSPI 데이터 로드
    load_kospi_data()

    # 종목 리스트
    stocks = get_stock_list()

    # 체크포인트 확인
    start_idx = 0
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE, 'r') as f:
            start_idx = int(f.read().strip())
        print(f"[CHECKPOINT] 체크포인트 발견: {start_idx}번째 종목부터 재개")

    all_results = []

    # 기존 데이터 로드 (있으면)
    if OUTPUT_FILE.exists() and start_idx > 0:
        existing_df = pd.read_csv(OUTPUT_FILE)
        all_results = existing_df.to_dict('records')
        print(f"[OK] 기존 데이터 로드: {len(all_results)}개 샘플")

    total_stocks = len(stocks)

    print(f"\n[RUN] 데이터 수집 시작: {total_stocks}개 종목")
    print(f"배치 크기: {BATCH_SIZE}개마다 저장\n")

    # 종목별 분석
    for idx, (_, row) in enumerate(stocks.iterrows(), 1):

        if idx <= start_idx:
            continue

        ticker = row['Code']
        name = row['Name']
        market_cap = row['Marcap']

        # 진행 상황 표시
        if idx % 10 == 0:
            print(f"[{idx}/{total_stocks}] {name} ({ticker}) 분석 중... "
                  f"현재 샘플: {len(all_results)}개")

        # 종목 분석
        stock_results = analyze_stock(ticker, name, market_cap)

        if stock_results:
            all_results.extend(stock_results)
            print(f"  [OK] {name}: {len(stock_results)}개 돌파 발견")

        # 배치 저장
        if idx % BATCH_SIZE == 0:
            df_batch = pd.DataFrame(all_results)
            df_batch.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')

            # 체크포인트 저장
            with open(CHECKPOINT_FILE, 'w') as f:
                f.write(str(idx))

            print(f"\n[SAVE] 배치 저장 완료: {len(all_results)}개 샘플\n")

    # 최종 저장
    if all_results:
        df_final = pd.DataFrame(all_results)
        df_final.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')

        print("\n" + "=" * 80)
        print("[OK] 데이터 수집 완료!")
        print("=" * 80)
        print(f"[DATA] 총 샘플 수: {len(all_results):,}개")
        print(f"[FILE] 저장 위치: {OUTPUT_FILE}")
        print()

        # 통계 출력
        print("[STATS] 데이터 통계:")
        print(f"  - 성공 샘플 (15%+ 상승): {df_final['success'].sum():,}개 ({df_final['success'].mean()*100:.1f}%)")
        print(f"  - 평균 최고 수익률: {df_final['max_gain_20d'].mean():.2f}%")
        print(f"  - 중앙값 최고 수익률: {df_final['max_gain_20d'].median():.2f}%")
        print(f"  - 최대 수익률: {df_final['max_gain_20d'].max():.2f}%")
        print()

        # 체크포인트 삭제
        if CHECKPOINT_FILE.exists():
            CHECKPOINT_FILE.unlink()

    else:
        print("\n[WARN]  수집된 데이터가 없습니다.")


# ============================================================================
# 실행
# ============================================================================

if __name__ == "__main__":
    collect_data()
