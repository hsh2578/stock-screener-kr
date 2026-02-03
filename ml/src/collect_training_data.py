"""
52주 신고가 돌파 성공률 예측 모델 - 학습 데이터 수집

기간: 2022.01 ~ 2025.06
필터: 시가총액 1000억 이상
라벨: 돌파 당일 종가 대비 20거래일 내 10% 상승 여부

출력: ml/data/training_data.csv (25개 컬럼)
"""

import os
import sys
import time
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List, Tuple
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
START_DATE = "2022-01-01"
END_DATE = "2025-06-30"
DATA_START_DATE = "2021-01-01"  # 52주 계산을 위해 1년 전부터

# 필터 조건
MIN_MARKET_CAP = 1000 * 100000000  # 1000억원 (원 단위)
HIGH_52W_PERIOD = 250  # 52주 = 약 250거래일
LOOKAHEAD_DAYS = 20    # 라벨 계산용
MIN_BREAKOUT_INTERVAL = 20  # 동일 종목 중복 돌파 최소 간격

# 제외 키워드
EXCLUDE_KEYWORDS = ["스팩", "ETF", "ETN", "리츠", "인프라", "SPAC", "기업인수"]

# API 딜레이
REQUEST_DELAY = 0.1


# ============================================================================
# 데이터 클래스
# ============================================================================
@dataclass
class BreakoutEvent:
    """52주 신고가 돌파 이벤트"""
    # 메타 정보 (7개)
    ticker: str
    name: str
    breakout_date: str
    high_52w: float
    breakout_price: float
    market_type: str
    market_cap: float

    # A. 돌파 품질 (3개)
    breakout_pct: float
    volume_surge: float
    close_strength: float

    # B. 베이스 품질 (3개)
    base_length: float
    volatility_contraction: float
    volume_dry_up: float

    # C. 추세 상태 (2개)
    ma200_slope: float
    pct_above_52w_low: float

    # D. 상대강도 (2개)
    rs_vs_market: float
    market_return: float

    # E. 리스크 (2개)
    ma20_deviation: float
    liquidity: float

    # F. 저항선/변동성 (2개)
    days_since_ath: float
    atr_ratio: float

    # 라벨 (4개)
    max_gain_20d: float
    max_drawdown_20d: float
    gain_at_day_20: float
    label: int


# ============================================================================
# 캐시 함수
# ============================================================================
def get_cache_path(name: str) -> Path:
    """캐시 파일 경로"""
    return CACHE_DIR / f"{name}.pkl"


def load_cache(name: str):
    """캐시 로드"""
    path = get_cache_path(name)
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


def save_cache(name: str, data):
    """캐시 저장"""
    path = get_cache_path(name)
    with open(path, "wb") as f:
        pickle.dump(data, f)


# ============================================================================
# 데이터 수집 함수
# ============================================================================
def get_stock_list() -> pd.DataFrame:
    """전체 종목 리스트 조회 (FinanceDataReader 사용)"""
    cache_name = "stock_list_fdr"
    cached = load_cache(cache_name)
    if cached is not None:
        print(f"  [캐시] 종목 리스트 로드: {len(cached)}개")
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

                # 제외 키워드 체크
                if any(kw in str(name) for kw in EXCLUDE_KEYWORDS):
                    continue

                # 시총 필터
                if marcap < MIN_MARKET_CAP:
                    continue

                result.append({
                    "ticker": ticker,
                    "name": name,
                    "market_type": market,
                    "market_cap": marcap / 100000000  # 억원 단위
                })

        except Exception as e:
            print(f"  [ERROR] {market} 종목 조회 실패: {e}")

    df_result = pd.DataFrame(result)
    save_cache(cache_name, df_result)
    print(f"  → {len(df_result)}개 종목 (시총 1000억 이상)")

    return df_result


def get_ohlcv(ticker: str) -> pd.DataFrame:
    """개별 종목 OHLCV 데이터"""
    cache_name = f"ohlcv_{ticker}"
    cached = load_cache(cache_name)
    if cached is not None:
        return cached

    try:
        df = fdr.DataReader(ticker, DATA_START_DATE, END_DATE)
        time.sleep(REQUEST_DELAY)

        if not df.empty:
            # 컬럼명 통일
            df = df.rename(columns={
                "Open": "시가",
                "High": "고가",
                "Low": "저가",
                "Close": "종가",
                "Volume": "거래량"
            })
            save_cache(cache_name, df)

        return df
    except Exception as e:
        return pd.DataFrame()


def get_market_index(market_type: str) -> pd.DataFrame:
    """시장 지수 데이터 (KOSPI/KOSDAQ)"""
    cache_name = f"index_{market_type}_fdr"
    cached = load_cache(cache_name)
    if cached is not None:
        return cached

    try:
        symbol = "KS11" if market_type == "KOSPI" else "KQ11"
        df = fdr.DataReader(symbol, DATA_START_DATE, END_DATE)
        time.sleep(REQUEST_DELAY)

        if not df.empty:
            df = df.rename(columns={"Close": "종가"})
            save_cache(cache_name, df)

        return df
    except Exception as e:
        print(f"  [ERROR] {market_type} 지수 조회 실패: {e}")
        return pd.DataFrame()


# ============================================================================
# 피처 계산 함수
# ============================================================================
def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """ATR (Average True Range) 계산"""
    high = df["고가"]
    low = df["저가"]
    close = df["종가"]

    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()

    return atr


def find_52w_high_date(df: pd.DataFrame, breakout_idx: int) -> int:
    """직전 52주 고가가 발생한 날짜의 인덱스"""
    start_idx = max(0, breakout_idx - HIGH_52W_PERIOD)
    close = df["종가"].iloc[start_idx:breakout_idx]

    if close.empty:
        return breakout_idx - 1

    max_pos = close.values.argmax()
    return start_idx + max_pos


def find_ath_date(df: pd.DataFrame, breakout_idx: int) -> int:
    """역사적 신고가(ATH)가 발생한 날짜의 인덱스"""
    close = df["종가"].iloc[:breakout_idx]

    if close.empty:
        return 0

    max_pos = close.values.argmax()
    return max_pos


def calculate_features(
    df: pd.DataFrame,
    idx: int,
    market_cap: float,
    market_index: pd.DataFrame,
    market_type: str
) -> Optional[Dict]:
    """돌파 시점의 14개 피처 계산"""
    try:
        if idx < 220:  # 최소 220일 데이터 필요 (MA200 + 20일 slope)
            return None

        close = df["종가"]
        high = df["고가"]
        low = df["저가"]
        volume = df["거래량"]

        breakout_price = close.iloc[idx]

        # =====================
        # A. 돌파 품질 (3개)
        # =====================

        # 1. breakout_pct: (종가 - 직전52주고가) / 직전52주고가
        high_52w = close.iloc[max(0, idx-HIGH_52W_PERIOD):idx].max()
        breakout_pct = (breakout_price - high_52w) / high_52w * 100

        # 2. volume_surge: 당일거래량 / 50일평균거래량
        vol_50d_avg = volume.iloc[max(0, idx-50):idx].mean()
        volume_surge = volume.iloc[idx] / vol_50d_avg if vol_50d_avg > 0 else 0

        # 3. close_strength: (종가 - 저가) / (고가 - 저가)
        day_range = high.iloc[idx] - low.iloc[idx]
        close_strength = (close.iloc[idx] - low.iloc[idx]) / day_range if day_range > 0 else 0.5

        # =====================
        # B. 베이스 품질 (3개)
        # =====================

        # 4. base_length: 직전 52주 고가 이후 경과일
        high_52w_idx = find_52w_high_date(df, idx)
        base_length = idx - high_52w_idx

        # 5. volatility_contraction: 최근5일 변동폭 / 20일 변동폭
        range_5d = (high.iloc[idx-4:idx+1].max() - low.iloc[idx-4:idx+1].min()) / close.iloc[idx]
        range_20d = (high.iloc[idx-19:idx+1].max() - low.iloc[idx-19:idx+1].min()) / close.iloc[idx]
        volatility_contraction = range_5d / range_20d if range_20d > 0 else 1

        # 6. volume_dry_up: 최근10일 평균거래량 / 50일 평균거래량
        vol_10d_avg = volume.iloc[idx-9:idx+1].mean()
        volume_dry_up = vol_10d_avg / vol_50d_avg if vol_50d_avg > 0 else 1

        # =====================
        # C. 추세 상태 (2개)
        # =====================

        # 7. ma200_slope: (200MA - 200MA[20일전]) / 200MA[20일전]
        ma200_now = close.iloc[idx-199:idx+1].mean()
        ma200_20d_ago = close.iloc[idx-219:idx-19].mean()
        ma200_slope = (ma200_now - ma200_20d_ago) / ma200_20d_ago if ma200_20d_ago > 0 else 0

        # 8. pct_above_52w_low: (종가 - 52주저가) / 52주저가
        low_52w = close.iloc[max(0, idx-HIGH_52W_PERIOD):idx+1].min()
        pct_above_52w_low = (breakout_price - low_52w) / low_52w * 100 if low_52w > 0 else 0

        # =====================
        # D. 상대강도 (2개)
        # =====================

        # 종목 20일 수익률
        stock_return_20d = (close.iloc[idx] / close.iloc[idx-20] - 1) * 100 if idx >= 20 else 0

        # 시장 20일 수익률
        market_return = 0
        rs_vs_market = stock_return_20d

        if not market_index.empty:
            try:
                current_date = df.index[idx]
                if current_date in market_index.index:
                    market_idx = market_index.index.get_loc(current_date)
                    if market_idx >= 20:
                        market_close = market_index["종가"]
                        market_return = (market_close.iloc[market_idx] / market_close.iloc[market_idx-20] - 1) * 100
                        rs_vs_market = stock_return_20d - market_return
            except:
                pass

        # =====================
        # E. 리스크 (2개)
        # =====================

        # 11. ma20_deviation: 종가 / 20일선 - 1
        ma20 = close.iloc[idx-19:idx+1].mean()
        ma20_deviation = (breakout_price / ma20 - 1) if ma20 > 0 else 0

        # 12. liquidity: 5일 평균 거래대금
        turnover = (close * volume).iloc[idx-4:idx+1]
        liquidity = turnover.mean()

        # =====================
        # F. 저항선/변동성 (2개)
        # =====================

        # 13. days_since_ath: 역사적 신고가 이후 경과일
        ath_idx = find_ath_date(df, idx)
        days_since_ath = idx - ath_idx

        # 14. atr_ratio: ATR(14) / 종가
        atr = calculate_atr(df, 14)
        atr_ratio = atr.iloc[idx] / breakout_price if breakout_price > 0 else 0

        return {
            "high_52w": round(high_52w, 0),
            "breakout_price": round(breakout_price, 0),
            "breakout_pct": round(breakout_pct, 4),
            "volume_surge": round(volume_surge, 4),
            "close_strength": round(close_strength, 4),
            "base_length": base_length,
            "volatility_contraction": round(volatility_contraction, 4),
            "volume_dry_up": round(volume_dry_up, 4),
            "ma200_slope": round(ma200_slope, 6),
            "pct_above_52w_low": round(pct_above_52w_low, 4),
            "rs_vs_market": round(rs_vs_market, 4),
            "market_return": round(market_return, 4),
            "ma20_deviation": round(ma20_deviation, 4),
            "liquidity": round(liquidity, 0),
            "days_since_ath": days_since_ath,
            "atr_ratio": round(atr_ratio, 6),
            "market_cap": round(market_cap, 0)
        }

    except Exception as e:
        return None


def calculate_labels(df: pd.DataFrame, idx: int) -> Optional[Dict]:
    """라벨 계산: 돌파 당일 종가 대비 20거래일 수익률"""
    try:
        close = df["종가"]
        breakout_price = close.iloc[idx]

        # 20거래일 후까지 데이터 확인
        end_idx = min(idx + LOOKAHEAD_DAYS, len(df) - 1)

        if end_idx <= idx:
            return None

        future_close = close.iloc[idx+1:end_idx+1]

        if future_close.empty:
            return None

        # 최대 수익률
        max_price = future_close.max()
        max_gain_20d = (max_price - breakout_price) / breakout_price * 100

        # 최대 하락률
        min_price = future_close.min()
        max_drawdown_20d = (min_price - breakout_price) / breakout_price * 100

        # 20일 후 수익률
        if len(future_close) >= LOOKAHEAD_DAYS:
            price_at_20 = future_close.iloc[LOOKAHEAD_DAYS - 1]
        else:
            price_at_20 = future_close.iloc[-1]
        gain_at_day_20 = (price_at_20 - breakout_price) / breakout_price * 100

        # 라벨
        label = 1 if max_gain_20d >= 10 else 0

        return {
            "max_gain_20d": round(max_gain_20d, 2),
            "max_drawdown_20d": round(max_drawdown_20d, 2),
            "gain_at_day_20": round(gain_at_day_20, 2),
            "label": label
        }

    except:
        return None


# ============================================================================
# 메인 수집 함수
# ============================================================================
def collect_breakout_events() -> List[BreakoutEvent]:
    """52주 신고가 돌파 이벤트 수집"""
    all_events = []

    # 1. 종목 리스트
    print("\n[1/4] 종목 리스트 조회...")
    stock_list = get_stock_list()

    if stock_list.empty:
        print("  [ERROR] 종목 리스트가 비어있습니다.")
        return []

    # 2. 시장 지수 데이터
    print("\n[2/4] 시장 지수 데이터 조회...")
    kospi_index = get_market_index("KOSPI")
    kosdaq_index = get_market_index("KOSDAQ")
    print(f"  KOSPI: {len(kospi_index)}일, KOSDAQ: {len(kosdaq_index)}일")

    # 3. 종목별 스캔
    print(f"\n[3/4] 돌파 이벤트 스캔 중... ({len(stock_list)}개 종목)")

    start_dt = pd.Timestamp(START_DATE)
    end_dt = pd.Timestamp(END_DATE)

    for i, row in stock_list.iterrows():
        ticker = row["ticker"]
        name = row["name"]
        market_type = row["market_type"]
        market_cap = row["market_cap"]

        # 진행률
        progress = i + 1
        if progress % 50 == 0 or progress == len(stock_list):
            print(f"  진행: {progress}/{len(stock_list)} ({progress/len(stock_list)*100:.1f}%) - 이벤트: {len(all_events)}개")

        # OHLCV 데이터
        df = get_ohlcv(ticker)
        if df.empty or len(df) < HIGH_52W_PERIOD + LOOKAHEAD_DAYS:
            continue

        # 시장 지수
        market_index = kospi_index if market_type == "KOSPI" else kosdaq_index

        # 돌파 스캔
        close = df["종가"]
        last_breakout_idx = -MIN_BREAKOUT_INTERVAL

        for idx in range(HIGH_52W_PERIOD, len(df) - LOOKAHEAD_DAYS):
            current_date = df.index[idx]

            # 날짜 범위 체크
            if current_date < start_dt or current_date > end_dt:
                continue

            # 중복 방지
            if idx - last_breakout_idx < MIN_BREAKOUT_INTERVAL:
                continue

            # 52주 신고가
            high_52w = close.iloc[idx-HIGH_52W_PERIOD:idx].max()
            current_price = close.iloc[idx]

            # 돌파 조건
            if current_price > high_52w:
                # 피처 계산
                features = calculate_features(df, idx, market_cap, market_index, market_type)
                if features is None:
                    continue

                # 라벨 계산
                labels = calculate_labels(df, idx)
                if labels is None:
                    continue

                # 이벤트 생성
                event = BreakoutEvent(
                    ticker=ticker,
                    name=name,
                    breakout_date=current_date.strftime("%Y-%m-%d"),
                    market_type=market_type,
                    **features,
                    **labels
                )

                all_events.append(event)
                last_breakout_idx = idx

    print(f"\n  → 총 {len(all_events)}개 돌파 이벤트 발견")

    return all_events


def save_to_csv(events: List[BreakoutEvent]):
    """CSV 저장"""
    if not events:
        print("[ERROR] 저장할 이벤트가 없습니다.")
        return

    # DataFrame 변환
    df = pd.DataFrame([asdict(e) for e in events])

    # 컬럼 순서 정렬
    columns = [
        # 메타 (7개)
        "ticker", "name", "breakout_date", "high_52w", "breakout_price", "market_type", "market_cap",
        # A. 돌파 품질 (3개)
        "breakout_pct", "volume_surge", "close_strength",
        # B. 베이스 품질 (3개)
        "base_length", "volatility_contraction", "volume_dry_up",
        # C. 추세 상태 (2개)
        "ma200_slope", "pct_above_52w_low",
        # D. 상대강도 (2개)
        "rs_vs_market", "market_return",
        # E. 리스크 (2개)
        "ma20_deviation", "liquidity",
        # F. 저항선/변동성 (2개)
        "days_since_ath", "atr_ratio",
        # 라벨 (4개)
        "max_gain_20d", "max_drawdown_20d", "gain_at_day_20", "label"
    ]

    df = df[columns]

    # 날짜순 정렬
    df = df.sort_values("breakout_date").reset_index(drop=True)

    # 저장
    output_path = OUTPUT_DIR / "training_data.csv"
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"\n[4/4] 저장 완료: {output_path}")
    print(f"  → 총 {len(df)}개 샘플")
    print(f"  → 성공 (label=1): {df['label'].sum()}개 ({df['label'].mean()*100:.1f}%)")
    print(f"  → 실패 (label=0): {(df['label']==0).sum()}개")
    print(f"\n  → 기간: {df['breakout_date'].min()} ~ {df['breakout_date'].max()}")


def main():
    """메인 실행"""
    print("=" * 60)
    print("52주 신고가 돌파 ML 학습 데이터 수집")
    print("=" * 60)
    print(f"기간: {START_DATE} ~ {END_DATE}")
    print(f"시총: {int(MIN_MARKET_CAP/100000000)}억 이상")
    print(f"라벨: 돌파 종가 대비 20거래일 내 10% 상승")
    print("=" * 60)

    start_time = datetime.now()

    # 이벤트 수집
    events = collect_breakout_events()

    # CSV 저장
    save_to_csv(events)

    elapsed = datetime.now() - start_time
    print(f"\n완료! (소요시간: {elapsed})")


if __name__ == "__main__":
    main()
