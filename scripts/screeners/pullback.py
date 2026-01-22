"""
풀백 스크리너 (Pullback Screener)

돌파 후 저항선으로 되돌아오는 종목을 포착합니다. (2차 매수 타점)
거래량이 감소하며 되돌아오는 것이 건전한 풀백의 신호입니다.

조건:
    - 시가총액: 1,000억 원 이상
    - 사전 조건: 거래량 동반 돌파 발생 이력
    - 풀백 기간: 돌파 후 10거래일 이내
    - 풀백 조건: 현재 주가가 저항선 ± 2% 영역으로 되돌아옴
    - 거래량: 풀백 구간 평균 거래량 < 돌파일 거래량 × 50%
    - 이평선: 150일선 위 유지
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
VOLUME_MULTIPLIER = 2.0  # 돌파 시 필요 거래량 배수
MA_PERIOD = 150
VOLUME_AVG_PERIOD = 20
MIN_MARKET_CAP = 1000
PULLBACK_ZONE = 0.02  # 저항선 ± 2%
MAX_PULLBACK_DAYS = 10  # 돌파 후 최대 풀백 기간
VOLUME_DECREASE_THRESHOLD = 0.5  # 풀백 거래량 < 돌파 거래량 × 50%


@dataclass
class PullbackResult:
    """풀백 스크리너 결과"""
    ticker: str
    name: str
    price: int
    breakout_date: str
    resistance: int
    pullback_date: str
    volume_decrease_rate: float  # 거래량 감소율 (%)
    updated_at: str

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "name": self.name,
            "price": self.price,
            "breakout_date": self.breakout_date,
            "resistance": self.resistance,
            "pullback_date": self.pullback_date,
            "volume_decrease_rate": round(self.volume_decrease_rate, 2),
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


def check_box_range(data: pd.DataFrame, end_idx: int) -> Optional[int]:
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

    return int(box_high)


def find_pullback(
    data: pd.DataFrame,
    ticker: str
) -> Optional[tuple[str, int, str, float]]:
    """
    풀백 종목을 찾습니다.

    Returns:
        (돌파일, 저항선, 풀백일, 거래량감소율) 또는 None
    """
    required_days = BOX_PERIOD + MA_PERIOD + MAX_PULLBACK_DAYS
    if len(data) < required_days:
        return None

    close = data["종가"]
    volume = data["거래량"]

    ma150 = calculate_ma(close, MA_PERIOD)
    vol_avg = volume.rolling(window=VOLUME_AVG_PERIOD, min_periods=VOLUME_AVG_PERIOD).mean()

    # 돌파일 탐색 (최근 2~MAX_PULLBACK_DAYS일 전에 돌파가 있어야 함)
    for days_since_breakout in range(2, MAX_PULLBACK_DAYS + 1):
        breakout_idx = len(data) - 1 - days_since_breakout

        if breakout_idx < BOX_PERIOD + MA_PERIOD:
            continue

        # 박스권 확인
        resistance = check_box_range(data, breakout_idx)
        if resistance is None:
            continue

        # 돌파 조건 확인
        breakout_close = close.iloc[breakout_idx]
        if pd.isna(breakout_close) or breakout_close <= resistance * BREAKOUT_THRESHOLD:
            continue

        # 돌파일 거래량 조건
        breakout_volume = volume.iloc[breakout_idx]
        avg_vol_before = vol_avg.iloc[breakout_idx - 1] if breakout_idx > 0 else vol_avg.iloc[breakout_idx]

        if pd.isna(breakout_volume) or pd.isna(avg_vol_before) or avg_vol_before <= 0:
            continue

        if breakout_volume < avg_vol_before * VOLUME_MULTIPLIER:
            continue

        # 돌파일 150일선 위 확인
        ma150_at_breakout = ma150.iloc[breakout_idx]
        if pd.isna(ma150_at_breakout) or breakout_close <= ma150_at_breakout:
            continue

        # 이전에 돌파한 적이 없는지 확인
        pre_period = close.iloc[breakout_idx - BOX_PERIOD:breakout_idx]
        if (pre_period > resistance * BREAKOUT_THRESHOLD).any():
            continue

        # === 풀백 조건 확인 ===
        current_price = close.iloc[-1]
        current_ma150 = ma150.iloc[-1]

        # 현재가가 150일선 위인지 확인
        if pd.isna(current_ma150) or current_price <= current_ma150:
            continue

        # 풀백 영역 확인: 저항선 ± 2%
        resistance_lower = resistance * (1 - PULLBACK_ZONE)
        resistance_upper = resistance * (1 + PULLBACK_ZONE)

        if not (resistance_lower <= current_price <= resistance_upper):
            continue

        # 풀백 구간 평균 거래량 계산
        pullback_start_idx = breakout_idx + 1
        pullback_volumes = volume.iloc[pullback_start_idx:]

        if len(pullback_volumes) == 0:
            continue

        pullback_avg_volume = pullback_volumes.mean()

        if pd.isna(pullback_avg_volume):
            continue

        # 거래량 감소 조건: 풀백 거래량 < 돌파 거래량 × 50%
        if pullback_avg_volume >= breakout_volume * VOLUME_DECREASE_THRESHOLD:
            continue

        volume_decrease_rate = (1 - pullback_avg_volume / breakout_volume) * 100

        # 돌파일, 풀백일 문자열 변환
        breakout_date = data.index[breakout_idx]
        pullback_date = data.index[-1]

        if isinstance(breakout_date, pd.Timestamp):
            breakout_date_str = breakout_date.strftime("%Y-%m-%d")
        else:
            breakout_date_str = str(breakout_date)

        if isinstance(pullback_date, pd.Timestamp):
            pullback_date_str = pullback_date.strftime("%Y-%m-%d")
        else:
            pullback_date_str = str(pullback_date)

        return breakout_date_str, resistance, pullback_date_str, volume_decrease_rate

    return None


def analyze_pullback(
    data: pd.DataFrame,
    ticker: str,
    name: str
) -> Optional[PullbackResult]:
    """개별 종목 분석"""
    required_days = BOX_PERIOD + MA_PERIOD + MAX_PULLBACK_DAYS
    if not validate_data(data, required_days):
        return None

    result = find_pullback(data, ticker)
    if result is None:
        return None

    breakout_date, resistance, pullback_date, volume_decrease_rate = result
    current_price = int(data["종가"].iloc[-1])

    return PullbackResult(
        ticker=ticker,
        name=name,
        price=current_price,
        breakout_date=breakout_date,
        resistance=resistance,
        pullback_date=pullback_date,
        volume_decrease_rate=volume_decrease_rate,
        updated_at=datetime.now().isoformat()
    )


def screen_pullback(
    stock_data: dict[str, pd.DataFrame],
    stock_info: pd.DataFrame
) -> list[dict]:
    """풀백 스크리너 실행"""
    results = []
    total = len(stock_data)
    passed_cap = 0
    passed_data = 0

    logger.info(f"[풀백 스크리너] 시작: {total}개 종목")

    for ticker, data in stock_data.items():
        stock_row = stock_info[stock_info["ticker"] == ticker]
        if stock_row.empty:
            continue

        name = stock_row.iloc[0].get("name", "")
        market_cap = stock_row.iloc[0].get("market_cap", 0)

        if market_cap < MIN_MARKET_CAP:
            continue
        passed_cap += 1

        required_days = BOX_PERIOD + MA_PERIOD + MAX_PULLBACK_DAYS
        if not validate_data(data, required_days):
            continue
        passed_data += 1

        result = analyze_pullback(data, ticker, name)
        if result:
            results.append(result)

    logger.info(
        f"[풀백 스크리너] 완료: "
        f"전체 {total}개 → 시총 {passed_cap}개 → "
        f"데이터 {passed_data}개 → 풀백 {len(results)}개"
    )

    results.sort(key=lambda x: x.volume_decrease_rate, reverse=True)
    return [r.to_dict() for r in results]
