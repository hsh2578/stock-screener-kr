"""
52주 신고가 돌파 스크리너 (52-Week High Breakout Screener)

52주 신고가를 돌파한 종목을 포착합니다.
신고가 돌파는 강력한 추세 전환/지속 신호로 해석됩니다.

조건:
    - 시가총액: 1,000억 원 이상
    - 52주 신고가: 최근 250거래일 중 최고 종가
    - 돌파: 종가 > 52주 신고가
    - 기간: 돌파 후 8거래일 이내
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
HIGH_52W_PERIOD = 250  # 52주 = 약 250거래일
MAX_DAYS_SINCE_BREAKOUT = 8  # 돌파 후 최대 거래일
MIN_MARKET_CAP = 1000  # 시가총액 1000억 이상


@dataclass
class NewHigh52wResult:
    """52주 신고가 돌파 스크리너 결과"""
    ticker: str
    name: str
    price: int
    change_rate: float
    high_52w: int
    breakout_date: str
    days_since: int
    above_high_percent: float
    market_cap: int
    updated_at: str

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "name": self.name,
            "price": int(self.price),
            "change_rate": round(float(self.change_rate), 2),
            "high_52w": int(self.high_52w),
            "breakout_date": self.breakout_date,
            "days_since": int(self.days_since),
            "above_high_percent": round(float(self.above_high_percent), 2),
            "market_cap": int(self.market_cap),
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


def find_breakout_info(df: pd.DataFrame) -> Optional[tuple[int, str, int]]:
    """
    52주 신고가 돌파 정보를 찾습니다.

    단순 로직:
    1. 9일 전 기준 52주 신고가 계산
    2. 8일 전부터 오늘까지 언제 처음 돌파했는지 확인
    3. 돌파 후 며칠 지났는지 반환

    Returns:
        (52주신고가, 돌파일, 경과일) 또는 None
    """
    total_len = len(df)
    required_len = HIGH_52W_PERIOD + MAX_DAYS_SINCE_BREAKOUT + 1

    if total_len < required_len:
        return None

    # 9일 전 기준으로 52주 신고가 계산 (돌파 검사 시작일 전날 기준)
    base_idx = total_len - 1 - MAX_DAYS_SINCE_BREAKOUT - 1  # 9일 전
    high_52w_prices = df["종가"].iloc[base_idx - HIGH_52W_PERIOD:base_idx]

    if high_52w_prices.empty:
        return None

    high_52w = int(high_52w_prices.max())

    if pd.isna(high_52w) or high_52w <= 0:
        return None

    # 8일 전부터 오늘까지 돌파일 찾기
    for days_ago in range(MAX_DAYS_SINCE_BREAKOUT, -1, -1):
        idx = total_len - 1 - days_ago
        close = df["종가"].iloc[idx]

        if close > high_52w:
            # 첫 돌파일 발견
            breakout_date = df.index[idx]
            return high_52w, breakout_date.strftime("%Y-%m-%d"), days_ago

    return None


def check_new_high_52w(
    data: pd.DataFrame,
    ticker: str
) -> Optional[tuple[int, str, int, float, float]]:
    """
    52주 신고가 돌파 여부를 확인합니다.

    Returns:
        (52주신고가, 돌파일, 경과일, 신고가대비상승률, 등락률) 또는 None
    """
    required_days = HIGH_52W_PERIOD + MAX_DAYS_SINCE_BREAKOUT + 2
    if len(data) < required_days:
        return None

    # 돌파 정보 찾기
    breakout_result = find_breakout_info(data)
    if breakout_result is None:
        return None

    high_52w, breakout_date, days_since = breakout_result

    # 현재가 및 등락률 계산
    current_close = data["종가"].iloc[-1]
    prev_close = data["종가"].iloc[-2]

    if pd.isna(current_close) or pd.isna(prev_close) or prev_close <= 0:
        return None

    # 현재가가 돌파 시점의 신고가 위에 있는지 확인
    if current_close <= high_52w:
        return None

    change_rate = (current_close - prev_close) / prev_close * 100
    above_high_percent = (current_close - high_52w) / high_52w * 100

    return high_52w, breakout_date, days_since, above_high_percent, change_rate


def analyze_new_high_52w(
    data: pd.DataFrame,
    ticker: str,
    name: str,
    market_cap: int
) -> Optional[NewHigh52wResult]:
    """개별 종목 분석"""
    required_days = HIGH_52W_PERIOD + MAX_DAYS_SINCE_BREAKOUT + 2
    if not validate_data(data, required_days):
        return None

    result = check_new_high_52w(data, ticker)
    if result is None:
        return None

    high_52w, breakout_date, days_since, above_high_percent, change_rate = result
    current_price = int(data["종가"].iloc[-1])

    return NewHigh52wResult(
        ticker=ticker,
        name=name,
        price=current_price,
        change_rate=change_rate,
        high_52w=high_52w,
        breakout_date=breakout_date,
        days_since=days_since,
        above_high_percent=above_high_percent,
        market_cap=market_cap,
        updated_at=datetime.now().isoformat()
    )


def screen_new_high_52w(
    stock_data: dict[str, pd.DataFrame],
    stock_info: pd.DataFrame
) -> list[dict]:
    """
    52주 신고가 돌파 스크리너 실행

    Args:
        stock_data: OHLCV 데이터
        stock_info: 종목 정보
    """
    results = []
    total = len(stock_data)
    passed_cap = 0
    passed_data = 0

    logger.info(f"[52주 신고가] 시작: {total}개 종목")

    for ticker, data in stock_data.items():
        stock_row = stock_info[stock_info["ticker"] == ticker]
        if stock_row.empty:
            continue

        name = stock_row.iloc[0].get("name", "")
        market_cap = stock_row.iloc[0].get("market_cap", 0)

        if market_cap < MIN_MARKET_CAP:
            continue
        passed_cap += 1

        required_days = HIGH_52W_PERIOD + MAX_DAYS_SINCE_BREAKOUT + 2
        if not validate_data(data, required_days):
            continue
        passed_data += 1

        result = analyze_new_high_52w(data, ticker, name, market_cap)
        if result:
            results.append(result)

    logger.info(
        f"[52주 신고가] 완료: "
        f"전체 {total}개 → 시총 {passed_cap}개 → "
        f"데이터 {passed_data}개 → 돌파 {len(results)}개"
    )

    # 신고가 대비 상승률 내림차순 정렬
    results.sort(key=lambda x: x.above_high_percent, reverse=True)
    return [r.to_dict() for r in results]
