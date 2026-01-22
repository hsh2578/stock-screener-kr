"""
박스권 돌파 스크리너 - 거래량 무관 (Box Breakout Simple Screener)

박스권 돌파 후 10거래일 이내 종목을 포착합니다.
거래량 조건 없이 순수하게 가격 돌파만 확인합니다.

조건:
    - 시가총액: 1,000억 원 이상
    - 사전 조건: 40거래일 이상 박스권 (종가 변동폭 20% 이내)
    - 저항선: 박스권 내 종가 최고가
    - 돌파 조건: 종가 > 저항선 × 1.02 (상단 +2% 초과)
    - 경과 기간: 돌파일로부터 10거래일 이내

사용법:
    >>> from screeners.box_breakout_simple import screen_box_breakout_simple
    >>> results = screen_box_breakout_simple(stock_data_dict, market_cap_df)
"""
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# 프로젝트 경로 설정
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from utils.logger import setup_logger

logger = setup_logger()

# ============================================================================
# 상수 정의
# ============================================================================
BOX_PERIOD = 40  # 박스권 판단 기간 (거래일)
MAX_RANGE_PERCENT = 20.0  # 최대 허용 변동폭 (%)
BREAKOUT_THRESHOLD = 1.02  # 돌파 기준 (저항선 +2%)
MAX_DAYS_SINCE_BREAKOUT = 10  # 돌파 후 최대 경과일
MIN_MARKET_CAP = 1000  # 최소 시가총액 (억원)
LOOKBACK_PERIOD = 60  # 돌파 탐색 기간 (최근 60일 내에서 돌파 확인)


@dataclass
class BoxBreakoutResult:
    """박스권 돌파 스크리너 결과 데이터 클래스"""
    ticker: str
    name: str
    price: int
    change_rate: float
    resistance: int
    breakout_date: str
    days_since: int
    current_vs_resistance: float  # 현재가 / 저항선 비율 (%)
    updated_at: str

    def to_dict(self) -> dict:
        """딕셔너리로 변환"""
        return {
            "ticker": self.ticker,
            "name": self.name,
            "price": self.price,
            "change_rate": round(self.change_rate, 2),
            "resistance": self.resistance,
            "breakout_date": self.breakout_date,
            "days_since": self.days_since,
            "current_vs_resistance": round(self.current_vs_resistance, 2),
            "updated_at": self.updated_at
        }


def validate_data(data: pd.DataFrame, required_days: int) -> bool:
    """
    데이터 유효성을 검증합니다.

    Args:
        data: OHLCV 데이터프레임
        required_days: 필요한 최소 거래일 수

    Returns:
        True: 데이터가 유효함
        False: 데이터 부족 또는 NaN 존재
    """
    if len(data) < required_days:
        return False

    if "종가" not in data.columns:
        return False

    close = data["종가"].tail(required_days)

    # NaN 확인
    if close.isna().any():
        return False

    # Inf 확인
    if np.isinf(close).any():
        return False

    # 0 이하 값 확인
    if (close <= 0).any():
        return False

    return True


def check_box_range(data: pd.DataFrame, end_idx: int) -> Optional[tuple[int, float]]:
    """
    특정 시점 이전의 박스권 여부를 확인합니다.

    Args:
        data: OHLCV 데이터프레임
        end_idx: 박스권 종료 인덱스 (돌파 직전일)

    Returns:
        (저항선, 변동폭%) 튜플 또는 None

    Example:
        >>> result = check_box_range(df, -11)  # 11일 전까지의 박스권
        >>> if result:
        ...     resistance, range_pct = result
    """
    # 박스권 기간 데이터 추출
    start_idx = end_idx - BOX_PERIOD
    if start_idx < 0:
        return None

    box_data = data.iloc[start_idx:end_idx]

    if len(box_data) < BOX_PERIOD:
        return None

    close = box_data["종가"]

    # NaN/Inf 확인
    if close.isna().any() or np.isinf(close).any():
        return None

    box_high = close.max()
    box_low = close.min()

    if box_low <= 0:
        return None

    range_percent = (box_high - box_low) / box_low * 100

    # 변동폭 20% 이내인지 확인
    if range_percent > MAX_RANGE_PERCENT:
        return None

    return int(box_high), range_percent


def find_breakout_date(
    data: pd.DataFrame,
    ticker: str
) -> Optional[tuple[int, str, int]]:
    """
    박스권 돌파일을 찾습니다.

    최근 LOOKBACK_PERIOD(60일) 내에서 박스권 돌파가 발생했는지 확인합니다.
    가장 최근의 유효한 돌파일을 반환합니다.

    Args:
        data: OHLCV 데이터프레임 (최소 BOX_PERIOD + LOOKBACK_PERIOD 일 필요)
        ticker: 종목코드 (로깅용)

    Returns:
        (저항선, 돌파일, 경과일수) 튜플 또는 None

    Example:
        >>> result = find_breakout_date(df, "005930")
        >>> if result:
        ...     resistance, breakout_date, days_since = result
    """
    required_days = BOX_PERIOD + LOOKBACK_PERIOD

    if len(data) < required_days:
        logger.debug(f"[{ticker}] 데이터 부족: {len(data)}일 < {required_days}일")
        return None

    close = data["종가"]

    # 최근 LOOKBACK_PERIOD 내에서 돌파 탐색 (최근일부터 역순으로)
    for days_ago in range(1, MAX_DAYS_SINCE_BREAKOUT + 1):
        # 돌파 후보일의 인덱스
        breakout_idx = len(data) - days_ago

        if breakout_idx < BOX_PERIOD:
            continue

        # 돌파 전날까지의 박스권 확인
        box_result = check_box_range(data, breakout_idx)

        if box_result is None:
            continue

        resistance, range_pct = box_result

        # 돌파 조건 확인: 종가 > 저항선 × 1.02
        breakout_close = close.iloc[breakout_idx]

        if pd.isna(breakout_close) or np.isinf(breakout_close):
            continue

        if breakout_close > resistance * BREAKOUT_THRESHOLD:
            # 돌파일 이전에는 돌파하지 않았는지 확인 (최초 돌파인지)
            pre_breakout_period = close.iloc[breakout_idx - BOX_PERIOD:breakout_idx]
            if (pre_breakout_period > resistance * BREAKOUT_THRESHOLD).any():
                # 이미 이전에 돌파한 적이 있음 -> 이 돌파는 무효
                continue

            # 유효한 돌파 발견
            breakout_date = data.index[breakout_idx]

            if isinstance(breakout_date, pd.Timestamp):
                breakout_date_str = breakout_date.strftime("%Y-%m-%d")
            else:
                breakout_date_str = str(breakout_date)

            logger.debug(
                f"[{ticker}] 돌파 발견: 저항선 {resistance:,}원, "
                f"돌파일 {breakout_date_str}, {days_ago}일 경과"
            )

            return resistance, breakout_date_str, days_ago

    return None


def analyze_box_breakout(
    data: pd.DataFrame,
    ticker: str,
    name: str
) -> Optional[BoxBreakoutResult]:
    """
    개별 종목의 박스권 돌파 여부를 분석합니다.

    Args:
        data: OHLCV 데이터프레임 (pykrx 형식)
        ticker: 종목코드
        name: 종목명

    Returns:
        BoxBreakoutResult: 돌파 조건 충족 시
        None: 조건 미충족 또는 데이터 오류
    """
    # 1. 데이터 유효성 검증
    required_days = BOX_PERIOD + LOOKBACK_PERIOD
    if not validate_data(data, required_days):
        return None

    # 2. 돌파일 탐색
    breakout_result = find_breakout_date(data, ticker)

    if breakout_result is None:
        return None

    resistance, breakout_date_str, days_since = breakout_result

    # 3. 현재가 및 등락률 계산
    close = data["종가"]
    current_price = int(close.iloc[-1])
    prev_price = close.iloc[-2] if len(close) > 1 else current_price

    if prev_price > 0 and not pd.isna(prev_price):
        change_rate = (current_price - prev_price) / prev_price * 100
    else:
        change_rate = 0.0

    # 4. 현재가 vs 저항선 비율
    if resistance > 0:
        current_vs_resistance = (current_price / resistance - 1) * 100
    else:
        current_vs_resistance = 0.0

    # 5. 결과 생성
    return BoxBreakoutResult(
        ticker=ticker,
        name=name,
        price=current_price,
        change_rate=change_rate,
        resistance=resistance,
        breakout_date=breakout_date_str,
        days_since=days_since,
        current_vs_resistance=current_vs_resistance,
        updated_at=datetime.now().isoformat()
    )


def screen_box_breakout_simple(
    stock_data: dict[str, pd.DataFrame],
    stock_info: pd.DataFrame
) -> list[dict]:
    """
    박스권 돌파 스크리너를 실행합니다 (거래량 무관).

    Args:
        stock_data: {ticker: DataFrame} 형태의 OHLCV 데이터
        stock_info: 종목 정보 DataFrame (ticker, name, market_cap 컬럼 필수)

    Returns:
        박스권 돌파 조건 충족 종목 리스트 (딕셔너리 형태)

    Example:
        >>> results = screen_box_breakout_simple(stock_data_dict, market_cap_df)
        >>> print(f"돌파 종목: {len(results)}개")
    """
    results: list[BoxBreakoutResult] = []
    total_count = len(stock_data)
    passed_market_cap = 0
    passed_data_validation = 0

    logger.info(f"[박스권 돌파 스크리너] 시작: {total_count}개 종목")

    required_days = BOX_PERIOD + LOOKBACK_PERIOD

    for ticker, data in stock_data.items():
        # 종목 정보 조회
        stock_row = stock_info[stock_info["ticker"] == ticker]
        if stock_row.empty:
            continue

        name = stock_row.iloc[0].get("name", "")
        market_cap = stock_row.iloc[0].get("market_cap", 0)

        # 1. 시가총액 필터 (1,000억 이상)
        if market_cap < MIN_MARKET_CAP:
            continue
        passed_market_cap += 1

        # 2. 데이터 유효성 확인
        if not validate_data(data, required_days):
            continue
        passed_data_validation += 1

        # 3. 박스권 돌파 분석
        result = analyze_box_breakout(data, ticker, name)
        if result:
            results.append(result)

    logger.info(
        f"[박스권 돌파 스크리너] 완료: "
        f"전체 {total_count}개 → "
        f"시가총액 {passed_market_cap}개 → "
        f"데이터검증 {passed_data_validation}개 → "
        f"돌파 {len(results)}개"
    )

    # 돌파 후 경과일이 적은 순으로 정렬 (최근 돌파 우선)
    results.sort(key=lambda x: x.days_since)

    return [r.to_dict() for r in results]
