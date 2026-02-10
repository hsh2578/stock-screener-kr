"""
52주 신고가 근접 ML 학습 데이터 수집

이벤트: 종가가 52주 고가(High 기준)의 5% 이내에 진입 (미돌파)
라벨:
  - broke_through_10d: 10거래일 내 52주 고가 돌파 여부 (0/1)
  - max_gain_20d: 20거래일 내 최고 수익률 (%)
  - days_to_breakout: 돌파까지 걸린 일수 (-1=미돌파)

피처: 13개 (gap_pct, resistance_tests, days_since_high, momentum_5d,
       volume_ratio, volume_trend, volatility_contraction, atr_pct,
       ma_alignment, trend_strength, rs_vs_market, market_return,
       pct_from_52w_low)

기간: 2020-01-01 ~ 2025-06-30
필터: 시총 1000억+, 고가 기록 후 20일+ 경과
중복: 동일 종목 10거래일 간격

출력: ml/data/near_high_training_data.csv
"""

import os
import sys
import time
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List
from dataclasses import dataclass, asdict

import pandas as pd
import numpy as np
import FinanceDataReader as fdr

# ============================================================================
# 설정
# ============================================================================
PROJECT_ROOT = Path(__file__).parent.parent.parent  # stock-screener-kr
ML_DIR = Path(__file__).parent.parent  # ml
CACHE_DIR = ML_DIR / ".cache"
OUTPUT_DIR = ML_DIR / "data"

CACHE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 데이터 수집 기간
START_DATE = "2020-01-01"
END_DATE = "2025-06-30"
DATA_START_DATE = "2018-10-01"  # 52주(250일) + MA200 + 여유

# 필터 조건
MIN_MARKET_CAP = 1000 * 100000000  # 1000억원
HIGH_52W_PERIOD = 250
MIN_DAYS_SINCE_HIGH = 20  # 저항선 확립
NEAR_HIGH_THRESHOLD = 5.0  # 근접 기준 (%) — 5%로 수집, gap_pct가 3%/5% 차이 학습
MIN_EVENT_INTERVAL = 10  # 동일 종목 중복 방지 (거래일)
LOOKAHEAD_BREAKOUT = 10  # 돌파 판정 기간
LOOKAHEAD_GAIN = 20  # 수익률 계산 기간

# 제외 키워드
EXCLUDE_KEYWORDS = ["스팩", "ETF", "ETN", "리츠", "인프라", "SPAC", "기업인수"]

REQUEST_DELAY = 0.1


# ============================================================================
# 데이터 클래스
# ============================================================================
@dataclass
class NearHighEvent:
    """52주 신고가 근접 이벤트"""
    # 메타 (7)
    ticker: str
    name: str
    event_date: str
    high_52w: float
    event_price: float
    market_type: str
    market_cap: float

    # 피처 (13)
    gap_pct: float
    resistance_tests: float
    days_since_high: float
    momentum_5d: float
    volume_ratio: float
    volume_trend: float
    volatility_contraction: float
    atr_pct: float
    ma_alignment: float
    trend_strength: float
    rs_vs_market: float
    market_return: float
    pct_from_52w_low: float

    # 라벨 (3)
    broke_through_10d: int
    max_gain_20d: float
    days_to_breakout: int


# ============================================================================
# 캐시 함수
# ============================================================================
def get_cache_path(name: str) -> Path:
    return CACHE_DIR / f"{name}.pkl"


def load_cache(name: str):
    path = get_cache_path(name)
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


def save_cache(name: str, data):
    path = get_cache_path(name)
    with open(path, "wb") as f:
        pickle.dump(data, f)


# ============================================================================
# 데이터 수집
# ============================================================================
def get_stock_list() -> pd.DataFrame:
    """시총 1000억+ 종목 리스트"""
    cache_name = "near_high_stock_list"
    cached = load_cache(cache_name)
    if cached is not None:
        print(f"  [캐시] 종목 리스트: {len(cached)}개")
        return cached

    print("  종목 리스트 조회 중...")
    result = []

    for market in ["KOSPI", "KOSDAQ"]:
        try:
            df = fdr.StockListing(market)
            for _, row in df.iterrows():
                ticker = row["Code"]
                name = row["Name"]
                marcap = row.get("Marcap", 0)

                if any(kw in str(name) for kw in EXCLUDE_KEYWORDS):
                    continue
                if marcap < MIN_MARKET_CAP:
                    continue

                result.append({
                    "ticker": ticker,
                    "name": name,
                    "market_type": market,
                    "market_cap": marcap / 100000000  # 억원
                })
        except Exception as e:
            print(f"  [ERROR] {market} 조회 실패: {e}")

    df_result = pd.DataFrame(result)
    save_cache(cache_name, df_result)
    print(f"  → {len(df_result)}개 종목")
    return df_result


def get_ohlcv(ticker: str) -> pd.DataFrame:
    """개별 종목 OHLCV"""
    cache_name = f"near_high_ohlcv_{ticker}"
    cached = load_cache(cache_name)
    if cached is not None:
        return cached

    try:
        df = fdr.DataReader(ticker, DATA_START_DATE, END_DATE)
        time.sleep(REQUEST_DELAY)
        if not df.empty:
            save_cache(cache_name, df)
        return df
    except Exception:
        return pd.DataFrame()


def get_market_index(market_type: str) -> pd.DataFrame:
    """시장 지수"""
    cache_name = f"near_high_index_{market_type}"
    cached = load_cache(cache_name)
    if cached is not None:
        return cached

    try:
        symbol = "KS11" if market_type == "KOSPI" else "KQ11"
        df = fdr.DataReader(symbol, DATA_START_DATE, END_DATE)
        time.sleep(REQUEST_DELAY)
        if not df.empty:
            save_cache(cache_name, df)
        return df
    except Exception as e:
        print(f"  [ERROR] {market_type} 지수 조회 실패: {e}")
        return pd.DataFrame()


# ============================================================================
# 피처 계산 (13개)
# ============================================================================
def calculate_features(
    df: pd.DataFrame,
    idx: int,
    high_52w: float,
    high_52w_pos: int,
    market_index: pd.DataFrame,
) -> Optional[Dict]:
    """근접 이벤트 시점의 13개 피처 계산"""
    try:
        if idx < 220:  # MA200 + 20일 여유
            return None

        close = df['Close']
        high = df['High']
        low = df['Low']
        volume = df['Volume']

        event_price = close.iloc[idx]
        if event_price <= 0:
            return None

        # === 1. gap_pct: 괴리율 (-3% ~ 0%) ===
        gap_pct = (event_price / high_52w - 1) * 100

        # === 2. resistance_tests: 최근 60일 중 고가 3% 이내 종가 일수 ===
        test_start = max(0, idx - 60)
        test_count = 0
        for d in range(test_start, idx):
            d_gap = (high_52w - close.iloc[d]) / high_52w * 100
            if 0 < d_gap <= NEAR_HIGH_THRESHOLD:
                test_count += 1
        resistance_tests = test_count

        # === 3. days_since_high: 52주 고가 후 경과 거래일 ===
        days_since_high = idx - high_52w_pos

        # === 4. momentum_5d: 5거래일 수익률 ===
        if idx >= 5:
            momentum_5d = (event_price / close.iloc[idx - 5] - 1) * 100
        else:
            momentum_5d = 0

        # === 5. volume_ratio: 당일 거래량 / 20일 평균 ===
        if idx >= 21:
            avg_vol_20 = volume.iloc[idx - 20:idx].mean()
            volume_ratio = volume.iloc[idx] / avg_vol_20 if avg_vol_20 > 0 else 1
        else:
            volume_ratio = 1

        # === 6. volume_trend: 5일 평균 / 20일 평균 거래량 ===
        if idx >= 20:
            vol_5d = volume.iloc[idx - 4:idx + 1].mean()
            vol_20d = volume.iloc[idx - 19:idx + 1].mean()
            volume_trend = vol_5d / vol_20d if vol_20d > 0 else 1
        else:
            volume_trend = 1

        # === 7. volatility_contraction: 10일 수익률std / 60일 수익률std ===
        if idx >= 61:
            returns = close.pct_change()
            std_10d = returns.iloc[idx - 9:idx + 1].std()
            std_60d = returns.iloc[idx - 59:idx + 1].std()
            volatility_contraction = std_10d / std_60d if std_60d > 0 else 1
        else:
            volatility_contraction = 1

        # === 8. atr_pct: ATR(14) / 종가 × 100 ===
        if idx >= 15:
            h = high.iloc[idx - 13:idx + 1].values
            l = low.iloc[idx - 13:idx + 1].values
            c_prev = close.iloc[idx - 14:idx].values

            tr1 = h - l
            tr2 = np.abs(h - c_prev)
            tr3 = np.abs(l - c_prev)
            tr = np.maximum(np.maximum(tr1, tr2), tr3)
            atr_14 = tr.mean()
            atr_pct = atr_14 / event_price * 100
        else:
            atr_pct = 0

        # === 9. ma_alignment: 정배열 점수 (0~4) ===
        score = 0
        ma_vals = {}
        for period in [5, 20, 60, 120, 200]:
            if idx >= period:
                ma_vals[period] = close.iloc[idx - period + 1:idx + 1].mean()
            else:
                ma_vals[period] = None

        if ma_vals[5] and ma_vals[20] and ma_vals[5] > ma_vals[20]:
            score += 1
        if ma_vals[20] and ma_vals[60] and ma_vals[20] > ma_vals[60]:
            score += 1
        if ma_vals[60] and ma_vals[120] and ma_vals[60] > ma_vals[120]:
            score += 1
        if ma_vals[120] and ma_vals[200] and ma_vals[120] > ma_vals[200]:
            score += 1
        ma_alignment = score

        # === 10. trend_strength: 60일선 기울기 ===
        if idx >= 80:
            ma60_now = close.iloc[idx - 59:idx + 1].mean()
            ma60_20ago = close.iloc[idx - 79:idx - 19].mean()
            trend_strength = (ma60_now / ma60_20ago - 1) * 100 if ma60_20ago > 0 else 0
        else:
            trend_strength = 0

        # === 11, 12. rs_vs_market, market_return ===
        rs_vs_market = 0
        market_return = 0
        stock_return_20d = 0

        if idx >= 20:
            stock_return_20d = (event_price / close.iloc[idx - 20] - 1) * 100

        if not market_index.empty and idx >= 20:
            try:
                current_date = df.index[idx]
                market_idx = market_index.index.get_indexer([current_date], method='nearest')[0]
                if market_idx >= 20:
                    m_close = market_index['Close']
                    market_return = (m_close.iloc[market_idx] / m_close.iloc[market_idx - 20] - 1) * 100
                    rs_vs_market = stock_return_20d - market_return
            except Exception:
                pass

        # === 13. pct_from_52w_low ===
        low_52w = low.iloc[max(0, idx - HIGH_52W_PERIOD):idx + 1].min()
        pct_from_52w_low = (event_price / low_52w - 1) * 100 if low_52w > 0 else 0

        return {
            "gap_pct": round(gap_pct, 4),
            "resistance_tests": resistance_tests,
            "days_since_high": days_since_high,
            "momentum_5d": round(momentum_5d, 4),
            "volume_ratio": round(volume_ratio, 4),
            "volume_trend": round(volume_trend, 4),
            "volatility_contraction": round(volatility_contraction, 4),
            "atr_pct": round(atr_pct, 4),
            "ma_alignment": ma_alignment,
            "trend_strength": round(trend_strength, 4),
            "rs_vs_market": round(rs_vs_market, 4),
            "market_return": round(market_return, 4),
            "pct_from_52w_low": round(pct_from_52w_low, 4),
        }

    except Exception as e:
        return None


# ============================================================================
# 라벨 계산
# ============================================================================
def calculate_labels(
    df: pd.DataFrame,
    idx: int,
    high_52w: float,
) -> Optional[Dict]:
    """라벨 3개 계산"""
    try:
        close = df['Close']
        high = df['High']
        event_price = close.iloc[idx]

        # 미래 데이터 확인
        if idx + LOOKAHEAD_GAIN >= len(df):
            return None

        # (1) 10거래일 내 돌파 여부 (High 기준)
        broke_through_10d = 0
        days_to_breakout = -1
        for d in range(1, LOOKAHEAD_BREAKOUT + 1):
            if idx + d >= len(df):
                break
            if high.iloc[idx + d] > high_52w:
                broke_through_10d = 1
                days_to_breakout = d
                break

        # (2) 20거래일 내 최고 수익률
        future_high = high.iloc[idx + 1:idx + LOOKAHEAD_GAIN + 1]
        if future_high.empty:
            return None
        max_gain_20d = (future_high.max() / event_price - 1) * 100

        return {
            "broke_through_10d": broke_through_10d,
            "max_gain_20d": round(max_gain_20d, 2),
            "days_to_breakout": days_to_breakout,
        }

    except Exception:
        return None


# ============================================================================
# 메인 수집
# ============================================================================
def collect_near_high_events() -> List[NearHighEvent]:
    """52주 신고가 근접 이벤트 수집"""
    all_events = []

    # 1. 종목 리스트
    print("\n[1/4] 종목 리스트 조회...")
    stock_list = get_stock_list()
    if stock_list.empty:
        print("  [ERROR] 종목 리스트 비어있음")
        return []

    # 2. 시장 지수 (KOSPI만 사용 — 추론 시 run_screeners.py도 KOSPI만 사용)
    print("\n[2/4] 시장 지수 조회...")
    kospi_index = get_market_index("KOSPI")
    print(f"  KOSPI: {len(kospi_index)}일 (전 종목 공통 사용)")

    # 3. 종목별 스캔
    total = len(stock_list)
    print(f"\n[3/4] 근접 이벤트 스캔 ({total}개 종목)...")

    start_dt = pd.Timestamp(START_DATE)
    end_dt = pd.Timestamp(END_DATE)

    for i, row in stock_list.iterrows():
        ticker = row["ticker"]
        name = row["name"]
        market_type = row["market_type"]
        market_cap = row["market_cap"]

        progress = i + 1
        if progress % 50 == 0 or progress == total:
            print(f"  진행: {progress}/{total} ({progress/total*100:.1f}%) - 이벤트: {len(all_events)}개")

        # OHLCV
        df = get_ohlcv(ticker)
        if df.empty or len(df) < HIGH_52W_PERIOD + LOOKAHEAD_GAIN + 50:
            continue

        # 컬럼 정리 (FDR은 Open/High/Low/Close/Volume)
        if 'Close' not in df.columns:
            continue

        market_index = kospi_index  # 전 종목 KOSPI 사용 (추론 코드와 일치)

        close = df['Close']
        high = df['High']
        last_event_idx = -MIN_EVENT_INTERVAL

        for idx in range(HIGH_52W_PERIOD, len(df) - LOOKAHEAD_GAIN):
            current_date = df.index[idx]

            # 날짜 범위
            if current_date < start_dt or current_date > end_dt:
                continue

            # 중복 방지
            if idx - last_event_idx < MIN_EVENT_INTERVAL:
                continue

            # 52주 고가 (High 기준, 직전 250일)
            high_window = high.iloc[idx - HIGH_52W_PERIOD:idx]
            high_52w = high_window.max()
            if pd.isna(high_52w) or high_52w <= 0:
                continue

            # 고가 기록 위치
            high_52w_pos_rel = high_window.values.argmax()
            high_52w_pos_abs = (idx - HIGH_52W_PERIOD) + high_52w_pos_rel

            # 저항선 확립: 20일+ 경과
            days_elapsed = idx - high_52w_pos_abs
            if days_elapsed < MIN_DAYS_SINCE_HIGH:
                continue

            # 근접 조건: 3% 이내, 미돌파
            event_price = close.iloc[idx]
            gap = (high_52w - event_price) / high_52w * 100
            if not (0 < gap <= NEAR_HIGH_THRESHOLD):
                continue

            # 피처 계산
            features = calculate_features(df, idx, high_52w, high_52w_pos_abs, market_index)
            if features is None:
                continue

            # 라벨 계산
            labels = calculate_labels(df, idx, high_52w)
            if labels is None:
                continue

            # 이벤트 생성
            event = NearHighEvent(
                ticker=ticker,
                name=name,
                event_date=current_date.strftime("%Y-%m-%d"),
                high_52w=round(high_52w, 0),
                event_price=round(event_price, 0),
                market_type=market_type,
                market_cap=round(market_cap, 0),
                **features,
                **labels,
            )

            all_events.append(event)
            last_event_idx = idx

    print(f"\n  → 총 {len(all_events)}개 근접 이벤트")
    return all_events


def save_to_csv(events: List[NearHighEvent]):
    """CSV 저장"""
    if not events:
        print("[ERROR] 저장할 이벤트 없음")
        return

    df = pd.DataFrame([asdict(e) for e in events])

    # 컬럼 순서
    columns = [
        # 메타 (7)
        "ticker", "name", "event_date", "high_52w", "event_price", "market_type", "market_cap",
        # 피처 (13)
        "gap_pct", "resistance_tests", "days_since_high", "momentum_5d",
        "volume_ratio", "volume_trend", "volatility_contraction", "atr_pct",
        "ma_alignment", "trend_strength", "rs_vs_market", "market_return",
        "pct_from_52w_low",
        # 라벨 (3)
        "broke_through_10d", "max_gain_20d", "days_to_breakout",
    ]

    df = df[columns]
    df = df.sort_values("event_date").reset_index(drop=True)

    output_path = OUTPUT_DIR / "near_high_training_data.csv"
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    # 통계
    total = len(df)
    positive = df['broke_through_10d'].sum()
    pos_rate = positive / total * 100

    print(f"\n[4/4] 저장 완료: {output_path}")
    print(f"  → 총 {total}개 샘플")
    print(f"  → 돌파 성공 (label=1): {positive}개 ({pos_rate:.1f}%)")
    print(f"  → 돌파 실패 (label=0): {total - positive}개 ({100 - pos_rate:.1f}%)")
    print(f"  → 평균 20일 최고수익률: {df['max_gain_20d'].mean():.2f}%")
    print(f"  → 기간: {df['event_date'].min()} ~ {df['event_date'].max()}")

    # 피처 기술통계
    print(f"\n  === 피처 기술통계 ===")
    feature_cols = columns[7:20]  # 13개 피처
    for col in feature_cols:
        print(f"  {col:25s}: mean={df[col].mean():8.2f}, std={df[col].std():8.2f}, "
              f"min={df[col].min():8.2f}, max={df[col].max():8.2f}")


def main():
    print("=" * 60)
    print("52주 신고가 근접 ML 학습 데이터 수집")
    print("=" * 60)
    print(f"기간: {START_DATE} ~ {END_DATE}")
    print(f"시총: {int(MIN_MARKET_CAP / 100000000)}억 이상")
    print(f"근접 기준: {NEAR_HIGH_THRESHOLD}% 이내")
    print(f"중복 간격: {MIN_EVENT_INTERVAL}거래일")
    print(f"라벨: 10거래일 돌파 여부 + 20거래일 최고 수익률")
    print(f"피처: 13개")
    print("=" * 60)

    start_time = datetime.now()

    events = collect_near_high_events()
    save_to_csv(events)

    elapsed = datetime.now() - start_time
    print(f"\n완료! (소요시간: {elapsed})")


if __name__ == "__main__":
    main()
