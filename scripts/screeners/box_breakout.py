"""
박스권 돌파 스크리너 - 거래량 동반 (Box Breakout with Volume)

박스권 상단 돌파 + 거래량 폭발 종목을 포착합니다. (1차 매수 타점)
거래량 없는 돌파는 가짜 돌파일 가능성이 높으므로 거래량 조건을 포함합니다.

조건:
    - 시가총액: 1,000억 원 이상
    - 사전 조건: 40거래일 이상 박스권 (종가 변동폭 20% 이내)
    - 돌파 조건: 종가 > 저항선 × 1.02
    - 거래량: 당일 거래량 ≥ 20일 평균 × 2배
    - 이평선: 돌파 시점에 주가가 150일선 위에 위치
"""
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from utils.logger import setup_logger

logger = setup_logger()

# ============================================================================
# 상수 정의
# ============================================================================
BOX_PERIOD = 40
MAX_RANGE_PERCENT = 20.0
BREAKOUT_THRESHOLD = 1.02
VOLUME_MULTIPLIER = 2.0
MA_PERIOD = 150
VOLUME_AVG_PERIOD = 20
MIN_MARKET_CAP = 1000
MAX_DAYS_SINCE_BREAKOUT = 5  # 돌파 후 최대 경과일


@dataclass
class BoxBreakoutResult:
    """박스권 돌파 (거래량 동반) 결과"""
    ticker: str
    name: str
    price: int
    resistance: int
    breakout_date: str
    volume_ratio: float
    ma150_diff: float  # 현재가 vs 150일선 차이 (%)
    updated_at: str

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "name": self.name,
            "price": self.price,
            "resistance": self.resistance,
            "breakout_date": self.breakout_date,
            "volume_ratio": round(self.volume_ratio, 2),
            "ma150_diff": round(self.ma150_diff, 2),
            "updated_at": self.updated_at
        }


def validate_data(data: pd.DataFrame, required_days: int) -> bool:
    """데이터 유효성 검증"""
    if len(data) < required_days:
        return False

    required_cols = ["종가", "거래량"]
    for col in required_cols:
        if col not in data.columns:
            return False
        if data[col].tail(required_days).isna().any():
            return False
        if np.isinf(data[col].tail(required_days)).any():
            return False

    if (data["종가"].tail(required_days) <= 0).any():
        return False

    return True


def calculate_ma(close: pd.Series, period: int) -> pd.Series:
    """이동평균선 계산"""
    return close.rolling(window=period, min_periods=period).mean()


def check_box_range(data: pd.DataFrame, end_idx: int) -> Optional[tuple[int, float]]:
    """박스권 확인 및 저항선 반환"""
    start_idx = end_idx - BOX_PERIOD
    if start_idx < 0:
        return None

    box_data = data.iloc[start_idx:end_idx]
    if len(box_data) < BOX_PERIOD:
        return None

    close = box_data["종가"]
    if close.isna().any() or (close <= 0).any():
        return None

    box_high = close.max()
    box_low = close.min()

    if box_low <= 0:
        return None

    range_percent = (box_high - box_low) / box_low * 100
    if range_percent > MAX_RANGE_PERCENT:
        return None

    return int(box_high), range_percent


def find_volume_breakout(
    data: pd.DataFrame,
    ticker: str
) -> Optional[tuple[int, str, float, float]]:
    """
    거래량 동반 돌파일을 찾습니다.

    Returns:
        (저항선, 돌파일, 거래량비율, 150일선대비%) 또는 None
    """
    required_days = BOX_PERIOD + MA_PERIOD
    if len(data) < required_days:
        return None

    close = data["종가"]
    volume = data["거래량"]

    # 150일 이동평균
    ma150 = calculate_ma(close, MA_PERIOD)

    # 20일 평균 거래량
    vol_avg = volume.rolling(window=VOLUME_AVG_PERIOD, min_periods=VOLUME_AVG_PERIOD).mean()

    # 최근 MAX_DAYS_SINCE_BREAKOUT 내에서 돌파 탐색
    for days_ago in range(0, MAX_DAYS_SINCE_BREAKOUT + 1):
        idx = len(data) - 1 - days_ago
        if idx < BOX_PERIOD + MA_PERIOD:
            continue

        # 박스권 확인 (돌파일 직전까지)
        box_result = check_box_range(data, idx)
        if box_result is None:
            continue

        resistance, _ = box_result

        # 돌파 조건: 종가 > 저항선 × 1.02
        breakout_close = close.iloc[idx]
        if pd.isna(breakout_close) or breakout_close <= resistance * BREAKOUT_THRESHOLD:
            continue

        # 거래량 조건: 당일 거래량 ≥ 20일 평균 × 2배
        day_volume = volume.iloc[idx]
        avg_vol = vol_avg.iloc[idx - 1] if idx > 0 else vol_avg.iloc[idx]

        if pd.isna(day_volume) or pd.isna(avg_vol) or avg_vol <= 0:
            continue

        volume_ratio = day_volume / avg_vol
        if volume_ratio < VOLUME_MULTIPLIER:
            continue

        # 이평선 조건: 150일선 위에 위치
        ma150_value = ma150.iloc[idx]
        if pd.isna(ma150_value) or ma150_value <= 0:
            continue

        if breakout_close <= ma150_value:
            continue

        ma150_diff = (breakout_close / ma150_value - 1) * 100

        # 돌파일 이전에 이미 돌파한 적이 없는지 확인
        pre_period = close.iloc[idx - BOX_PERIOD:idx]
        if (pre_period > resistance * BREAKOUT_THRESHOLD).any():
            continue

        breakout_date = data.index[idx]
        if isinstance(breakout_date, pd.Timestamp):
            breakout_date_str = breakout_date.strftime("%Y-%m-%d")
        else:
            breakout_date_str = str(breakout_date)

        return resistance, breakout_date_str, volume_ratio, ma150_diff

    return None


def analyze_box_breakout(
    data: pd.DataFrame,
    ticker: str,
    name: str
) -> Optional[BoxBreakoutResult]:
    """개별 종목 분석"""
    required_days = BOX_PERIOD + MA_PERIOD
    if not validate_data(data, required_days):
        return None

    result = find_volume_breakout(data, ticker)
    if result is None:
        return None

    resistance, breakout_date, volume_ratio, ma150_diff = result
    current_price = int(data["종가"].iloc[-1])

    return BoxBreakoutResult(
        ticker=ticker,
        name=name,
        price=current_price,
        resistance=resistance,
        breakout_date=breakout_date,
        volume_ratio=volume_ratio,
        ma150_diff=ma150_diff,
        updated_at=datetime.now().isoformat()
    )


def screen_box_breakout(
    stock_data: dict[str, pd.DataFrame],
    stock_info: pd.DataFrame
) -> list[dict]:
    """박스권 돌파 (거래량 동반) 스크리너 실행"""
    results = []
    total = len(stock_data)
    passed_cap = 0
    passed_data = 0

    logger.info(f"[박스권 돌파(거래량)] 시작: {total}개 종목")

    for ticker, data in stock_data.items():
        stock_row = stock_info[stock_info["ticker"] == ticker]
        if stock_row.empty:
            continue

        name = stock_row.iloc[0].get("name", "")
        market_cap = stock_row.iloc[0].get("market_cap", 0)

        if market_cap < MIN_MARKET_CAP:
            continue
        passed_cap += 1

        required_days = BOX_PERIOD + MA_PERIOD
        if not validate_data(data, required_days):
            continue
        passed_data += 1

        result = analyze_box_breakout(data, ticker, name)
        if result:
            results.append(result)

    logger.info(
        f"[박스권 돌파(거래량)] 완료: "
        f"전체 {total}개 → 시총 {passed_cap}개 → "
        f"데이터 {passed_data}개 → 돌파 {len(results)}개"
    )

    results.sort(key=lambda x: x.volume_ratio, reverse=True)
    return [r.to_dict() for r in results]
