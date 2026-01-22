"""
박스권 스크리너 (Box Range Screener)

2개월(40거래일) 이상 횡보 중인 종목을 발굴합니다.
박스권 횡보 종목은 돌파 시 큰 상승을 기대할 수 있어 선매수 기회로 활용됩니다.

조건:
    - 시가총액: 1,000억 원 이상
    - 박스 기간: 40 거래일 이상
    - 박스 범위: 종가 기준 변동폭 20% 이내
    - 형태 검증: V자 반등이나 역V자 하락 제외

사용법:
    >>> from screeners.box_range import screen_box_range
    >>> results = screen_box_range(stock_data_dict, market_cap_df)
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
MIN_MARKET_CAP = 1000  # 최소 시가총액 (억원)
HALF_PERIOD = BOX_PERIOD // 2  # 전반부/후반부 구분 기간


@dataclass
class BoxRangeResult:
    """박스권 스크리너 결과 데이터 클래스"""
    ticker: str
    name: str
    price: int
    change_rate: float
    box_high: int
    box_low: int
    range_percent: float
    days: int
    updated_at: str

    def to_dict(self) -> dict:
        """딕셔너리로 변환"""
        return {
            "ticker": self.ticker,
            "name": self.name,
            "price": self.price,
            "change_rate": round(self.change_rate, 2),
            "box_high": self.box_high,
            "box_low": self.box_low,
            "range_percent": round(self.range_percent, 2),
            "days": self.days,
            "updated_at": self.updated_at
        }


def validate_data(data: pd.DataFrame, required_days: int = BOX_PERIOD) -> bool:
    """
    데이터 유효성을 검증합니다.

    Args:
        data: OHLCV 데이터프레임
        required_days: 필요한 최소 거래일 수

    Returns:
        True: 데이터가 유효함
        False: 데이터 부족 또는 NaN 존재

    Example:
        >>> if not validate_data(df, 40):
        ...     logger.warning("데이터 부족")
    """
    # 데이터 개수 확인
    if len(data) < required_days:
        return False

    # 최근 required_days 데이터 추출
    recent_data = data.tail(required_days)

    # 필수 컬럼 확인
    if "종가" not in recent_data.columns:
        return False

    # NaN 확인 (종가 기준)
    if recent_data["종가"].isna().any():
        return False

    # Inf 확인
    if np.isinf(recent_data["종가"]).any():
        return False

    # 0 이하 값 확인 (비정상 데이터)
    if (recent_data["종가"] <= 0).any():
        return False

    return True


def check_box_shape(data: pd.DataFrame) -> bool:
    """
    박스권 형태를 검증합니다.

    V자 반등이나 역V자 하락은 박스권이 아닙니다.
    전반부(20일)와 후반부(20일)의 고/저점이 전체 박스 범위 안에 있어야 합니다.

    Args:
        data: 최근 40거래일 OHLCV 데이터프레임 (종가 컬럼 필수)

    Returns:
        True: 올바른 박스권 형태
        False: V자 또는 역V자 형태

    Example:
        >>> recent_40d = df.tail(40)
        >>> is_valid_box = check_box_shape(recent_40d)
    """
    if len(data) < BOX_PERIOD:
        return False

    close = data["종가"].values

    # 전체 범위
    total_high = np.max(close)
    total_low = np.min(close)

    # 전반부 (첫 20일)
    first_half = close[:HALF_PERIOD]
    first_high = np.max(first_half)
    first_low = np.min(first_half)

    # 후반부 (마지막 20일)
    second_half = close[HALF_PERIOD:]
    second_high = np.max(second_half)
    second_low = np.min(second_half)

    # 형태 검증: 전반부/후반부 모두 전체 범위 내에 있어야 함
    # (이미 부분집합이므로 항상 참이지만, 추가 검증 로직 적용)

    # V자 검증: 전반부에서 하락 후 후반부에서 상승
    # -> 전반부 고점이 후반부 저점보다 현저히 높고, 후반부 고점이 전체 고점인 경우
    if first_high == total_high and second_low < first_low:
        # 전반부가 고점이고 후반부가 크게 하락한 경우 -> 역V자
        range_pct = (total_high - total_low) / total_low * 100 if total_low > 0 else 0
        if range_pct > 10:  # 10% 이상 차이나면 역V자로 판단
            first_half_range = (first_high - first_low) / first_low * 100 if first_low > 0 else 0
            second_half_range = (second_high - second_low) / second_low * 100 if second_low > 0 else 0
            # 후반부가 전반부보다 현저히 낮은 레벨이면 역V자
            if second_high < first_low:
                return False

    # 역V자 검증: 전반부에서 상승 후 후반부에서 하락
    if second_high == total_high and first_low < second_low:
        range_pct = (total_high - total_low) / total_low * 100 if total_low > 0 else 0
        if range_pct > 10:
            if first_high < second_low:
                return False

    # 추가 검증: 전반부와 후반부의 중심값이 크게 다르면 횡보가 아님
    first_mid = (first_high + first_low) / 2
    second_mid = (second_high + second_low) / 2
    mid_diff_pct = abs(first_mid - second_mid) / min(first_mid, second_mid) * 100

    if mid_diff_pct > 15:  # 중심값 차이가 15% 이상이면 횡보 아님
        return False

    return True


def analyze_box_range(
    data: pd.DataFrame,
    ticker: str,
    name: str
) -> Optional[BoxRangeResult]:
    """
    개별 종목의 박스권 여부를 분석합니다.

    Args:
        data: OHLCV 데이터프레임 (pykrx 형식)
        ticker: 종목코드
        name: 종목명

    Returns:
        BoxRangeResult: 박스권 조건 충족 시
        None: 조건 미충족 또는 데이터 오류

    Example:
        >>> result = analyze_box_range(df, "005930", "삼성전자")
        >>> if result:
        ...     print(f"박스권: {result.range_percent}%")
    """
    # 1. 데이터 유효성 검증
    if not validate_data(data, BOX_PERIOD):
        logger.debug(f"[{ticker}] 데이터 검증 실패")
        return None

    # 2. 최근 40거래일 데이터 추출
    recent_data = data.tail(BOX_PERIOD).copy()
    close_prices = recent_data["종가"]

    # 3. 박스 범위 계산 (종가 기준)
    box_high = int(close_prices.max())
    box_low = int(close_prices.min())

    # 0으로 나누기 방지
    if box_low <= 0:
        logger.debug(f"[{ticker}] 저가 0 이하")
        return None

    range_percent = (box_high - box_low) / box_low * 100

    # 4. 변동폭 조건 확인 (20% 이내)
    if range_percent > MAX_RANGE_PERCENT:
        logger.debug(f"[{ticker}] 변동폭 초과: {range_percent:.2f}%")
        return None

    # 5. 박스권 형태 검증
    if not check_box_shape(recent_data):
        logger.debug(f"[{ticker}] 박스권 형태 검증 실패")
        return None

    # 6. 현재가 및 등락률 계산
    current_price = int(close_prices.iloc[-1])
    prev_price = close_prices.iloc[-2] if len(close_prices) > 1 else current_price

    if prev_price > 0:
        change_rate = (current_price - prev_price) / prev_price * 100
    else:
        change_rate = 0.0

    # 7. 결과 생성
    return BoxRangeResult(
        ticker=ticker,
        name=name,
        price=current_price,
        change_rate=change_rate,
        box_high=box_high,
        box_low=box_low,
        range_percent=range_percent,
        days=BOX_PERIOD,
        updated_at=datetime.now().isoformat()
    )


def screen_box_range(
    stock_data: dict[str, pd.DataFrame],
    stock_info: pd.DataFrame
) -> list[dict]:
    """
    박스권 스크리너를 실행합니다.

    Args:
        stock_data: {ticker: DataFrame} 형태의 OHLCV 데이터
        stock_info: 종목 정보 DataFrame (ticker, name, market_cap 컬럼 필수)

    Returns:
        박스권 조건 충족 종목 리스트 (딕셔너리 형태)

    Example:
        >>> results = screen_box_range(stock_data_dict, market_cap_df)
        >>> print(f"박스권 종목: {len(results)}개")
    """
    results: list[BoxRangeResult] = []
    total_count = len(stock_data)
    passed_market_cap = 0
    passed_data_validation = 0

    logger.info(f"[박스권 스크리너] 시작: {total_count}개 종목")

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
        if not validate_data(data, BOX_PERIOD):
            continue
        passed_data_validation += 1

        # 3. 박스권 분석
        result = analyze_box_range(data, ticker, name)
        if result:
            results.append(result)

    logger.info(
        f"[박스권 스크리너] 완료: "
        f"전체 {total_count}개 → "
        f"시가총액 {passed_market_cap}개 → "
        f"데이터검증 {passed_data_validation}개 → "
        f"박스권 {len(results)}개"
    )

    # 변동폭이 작은 순으로 정렬 (더 타이트한 박스권 우선)
    results.sort(key=lambda x: x.range_percent)

    return [r.to_dict() for r in results]
