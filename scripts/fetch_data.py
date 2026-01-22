"""
KRX 주식 데이터 수집 모듈

pykrx 라이브러리를 사용하여 한국거래소(KRX) 데이터를 수집합니다.
실제 투자에 사용되므로 데이터 정확성과 안정적인 예외 처리를 최우선으로 합니다.

주요 기능:
    - check_market_open(): 오늘 장 운영 여부 확인
    - get_stock_list(): 전 종목 리스트 (필터링 적용)
    - get_stock_data(): 개별 종목 OHLCV 데이터 (수정주가)
    - get_market_cap_list(): 시가총액 1,000억 이상 종목
    - get_volume_rank(): 거래대금 상위 종목
"""
import time
from datetime import datetime, timedelta
from functools import wraps
from typing import Callable

import pandas as pd
from pykrx import stock

from utils.logger import setup_logger

logger = setup_logger()

# ============================================================================
# 상수 정의
# ============================================================================
SAMSUNG_TICKER = "005930"  # 삼성전자 (시장 운영 확인용)
DEFAULT_RETRY_COUNT = 3
DEFAULT_RETRY_DELAY = 1.0  # 초
REQUEST_DELAY = 0.05  # 요청 간 딜레이 (초)

# 제외할 종목명 키워드
EXCLUDE_KEYWORDS = ["스팩", "ETF", "ETN", "리츠", "인프라"]


# ============================================================================
# 유틸리티 함수
# ============================================================================
def get_today_str() -> str:
    """오늘 날짜를 YYYYMMDD 형식 문자열로 반환합니다."""
    return datetime.now().strftime("%Y%m%d")


def get_date_str(days_ago: int = 0) -> str:
    """
    지정된 일수 전의 날짜를 YYYYMMDD 형식으로 반환합니다.

    Args:
        days_ago: 오늘로부터 며칠 전 (기본 0 = 오늘)

    Returns:
        YYYYMMDD 형식 문자열
    """
    target_date = datetime.now() - timedelta(days=days_ago)
    return target_date.strftime("%Y%m%d")


def retry_on_failure(
    max_retries: int = DEFAULT_RETRY_COUNT,
    base_delay: float = DEFAULT_RETRY_DELAY,
    exceptions: tuple = (Exception,)
) -> Callable:
    """
    실패 시 지수 백오프로 재시도하는 데코레이터.

    Args:
        max_retries: 최대 재시도 횟수 (기본 3회)
        base_delay: 기본 대기 시간 (초, 기본 1초)
        exceptions: 재시도할 예외 타입들

    Returns:
        데코레이터 함수

    Example:
        >>> @retry_on_failure(max_retries=3, base_delay=1.0)
        ... def fetch_data():
        ...     pass
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)  # 지수 백오프: 1초, 2초, 4초
                        logger.warning(
                            f"{func.__name__}() 재시도 {attempt + 1}/{max_retries} "
                            f"({delay:.1f}초 대기) - 오류: {e}"
                        )
                        time.sleep(delay)
                    else:
                        logger.error(
                            f"{func.__name__}() 최종 실패 ({max_retries}회 시도) - 오류: {e}"
                        )

            raise last_exception

        return wrapper
    return decorator


# ============================================================================
# 핵심 데이터 수집 함수
# ============================================================================
def check_market_open() -> bool:
    """
    오늘 주식시장이 열렸는지 확인합니다.

    삼성전자(005930) 일봉 데이터를 조회하여 오늘 데이터가 있으면 개장일로 판단합니다.
    공휴일, 주말 등 휴장일에는 False를 반환합니다.

    Returns:
        True: 오늘 장이 열린 경우 (거래 데이터 존재)
        False: 오늘 장이 열리지 않은 경우 (휴장일)

    Example:
        >>> if not check_market_open():
        ...     print("Market is closed today.")
        ...     sys.exit(0)
    """
    today = get_today_str()

    try:
        df = stock.get_market_ohlcv(today, today, SAMSUNG_TICKER)

        if df.empty:
            logger.info(f"휴장일 확인: {today} - 삼성전자 데이터 없음")
            return False

        # 거래량이 0인 경우도 휴장으로 간주
        if df["거래량"].iloc[0] == 0:
            logger.info(f"휴장일 확인: {today} - 거래량 0")
            return False

        logger.info(f"개장일 확인: {today}")
        return True

    except Exception as e:
        logger.warning(f"시장 상태 확인 실패: {e} - 휴장일로 간주")
        return False


def get_stock_list(market: str = "ALL") -> pd.DataFrame:
    """
    코스피/코스닥 전 종목 티커를 수집하고 필터링합니다.

    필터링 조건:
        1. 시가총액 0원인 종목 제외
        2. 최근 5일 거래량이 모두 0인 종목 제외
        3. 종목명에 "스팩", "ETF", "ETN" 등 포함된 종목 제외

    Args:
        market: "KOSPI", "KOSDAQ", 또는 "ALL" (기본값)

    Returns:
        DataFrame with columns: ['ticker', 'name', 'market']

    Raises:
        ValueError: 유효하지 않은 market 값인 경우

    Example:
        >>> stocks = get_stock_list()
        >>> print(f"총 {len(stocks)}개 종목")
    """
    valid_markets = ["KOSPI", "KOSDAQ", "ALL"]
    if market not in valid_markets:
        raise ValueError(f"market은 {valid_markets} 중 하나여야 합니다: {market}")

    today = get_today_str()
    five_days_ago = get_date_str(7)  # 주말 고려하여 7일

    result_list = []

    # 마켓별 종목 수집
    markets_to_fetch = ["KOSPI", "KOSDAQ"] if market == "ALL" else [market]

    for mkt in markets_to_fetch:
        logger.info(f"{mkt} 종목 리스트 수집 중...")

        try:
            tickers = stock.get_market_ticker_list(today, market=mkt)
        except Exception as e:
            logger.error(f"{mkt} 종목 리스트 조회 실패: {e}")
            continue

        # 시가총액 데이터 조회
        try:
            market_cap_df = stock.get_market_cap(today, market=mkt)
        except Exception as e:
            logger.error(f"{mkt} 시가총액 조회 실패: {e}")
            market_cap_df = pd.DataFrame()

        for ticker in tickers:
            time.sleep(REQUEST_DELAY)

            try:
                name = stock.get_market_ticker_name(ticker)
            except Exception:
                name = ""

            # 1. 종목명 키워드 필터링
            if any(keyword in name for keyword in EXCLUDE_KEYWORDS):
                logger.debug(f"제외 (키워드): {ticker} {name}")
                continue

            # 2. 시가총액 0원 필터링
            if not market_cap_df.empty and ticker in market_cap_df.index:
                market_cap = market_cap_df.loc[ticker, "시가총액"]
                if market_cap == 0:
                    logger.debug(f"제외 (시가총액 0): {ticker} {name}")
                    continue

            # 3. 최근 5일 거래량 확인
            try:
                ohlcv = stock.get_market_ohlcv(five_days_ago, today, ticker)
                if not ohlcv.empty:
                    recent_volume = ohlcv["거래량"].tail(5)
                    if (recent_volume == 0).all():
                        logger.debug(f"제외 (거래량 0): {ticker} {name}")
                        continue
            except Exception:
                pass  # 조회 실패 시 일단 포함

            result_list.append({
                "ticker": ticker,
                "name": name,
                "market": mkt
            })

    result_df = pd.DataFrame(result_list)
    logger.info(f"종목 리스트 수집 완료: 총 {len(result_df)}개")

    return result_df


@retry_on_failure(max_retries=3, base_delay=1.0)
def get_stock_data(ticker: str, days: int = 250) -> pd.DataFrame:
    """
    개별 종목의 일봉 OHLCV 데이터를 조회합니다.

    **중요**: 수정주가(adjusted=True)를 사용하여 액면분할, 배당락 등이 반영된
    데이터를 반환합니다. 기술적 분석에 적합한 연속적인 가격 데이터입니다.

    Args:
        ticker: 종목코드 (6자리 문자열, 예: "005930")
        days: 조회 기간 (거래일 기준, 기본 250일 ≈ 1년)

    Returns:
        DataFrame with columns: ['시가', '고가', '저가', '종가', '거래량']
        인덱스: DatetimeIndex

    Raises:
        ValueError: ticker가 유효하지 않은 경우
        Exception: API 호출 실패 (3회 재시도 후)

    Example:
        >>> df = get_stock_data("005930", days=250)
        >>> print(df.tail())
    """
    if not ticker or len(ticker) != 6:
        raise ValueError(f"유효하지 않은 종목코드: {ticker}")

    # 충분한 기간 확보를 위해 캘린더일 기준으로 계산 (거래일 약 250일 = 캘린더 약 365일)
    calendar_days = int(days * 1.5)
    end_date = get_today_str()
    start_date = get_date_str(calendar_days)

    time.sleep(REQUEST_DELAY)

    # pykrx의 get_market_ohlcv는 기본적으로 수정주가를 반환
    df = stock.get_market_ohlcv(start_date, end_date, ticker, adjusted=True)

    if df.empty:
        logger.warning(f"데이터 없음: {ticker}")
        return pd.DataFrame()

    # 최소 데이터 개수 확인 (200일 이상 권장)
    if len(df) < 200:
        logger.warning(f"데이터 부족: {ticker} ({len(df)}일)")

    # 필요한 컬럼만 선택 (등락률, 거래대금 제외)
    columns = ["시가", "고가", "저가", "종가", "거래량"]
    df = df[columns]

    return df


def get_market_cap_list(min_cap_billion: int = 1000) -> pd.DataFrame:
    """
    시가총액 기준 이상의 종목 리스트를 조회합니다.

    Args:
        min_cap_billion: 최소 시가총액 (억원 단위, 기본 1,000억원)

    Returns:
        DataFrame with columns: ['ticker', 'name', 'market_cap', 'market']
        market_cap은 억원 단위

    Example:
        >>> large_caps = get_market_cap_list(min_cap_billion=1000)
        >>> print(f"시총 1000억 이상: {len(large_caps)}개")
    """
    today = get_today_str()
    min_cap_won = min_cap_billion * 100_000_000  # 억원 -> 원

    result_list = []

    for mkt in ["KOSPI", "KOSDAQ"]:
        logger.info(f"{mkt} 시가총액 조회 중...")

        try:
            df = stock.get_market_cap(today, market=mkt)
        except Exception as e:
            logger.error(f"{mkt} 시가총액 조회 실패: {e}")
            continue

        # 시가총액 필터링
        filtered = df[df["시가총액"] >= min_cap_won]

        for ticker in filtered.index:
            try:
                name = stock.get_market_ticker_name(ticker)
            except Exception:
                name = ""

            # ETF, 스팩 등 제외
            if any(keyword in name for keyword in EXCLUDE_KEYWORDS):
                continue

            market_cap_billion = int(filtered.loc[ticker, "시가총액"] / 100_000_000)

            result_list.append({
                "ticker": ticker,
                "name": name,
                "market_cap": market_cap_billion,
                "market": mkt
            })

            time.sleep(REQUEST_DELAY)

    result_df = pd.DataFrame(result_list)

    # 시가총액 내림차순 정렬
    result_df = result_df.sort_values("market_cap", ascending=False).reset_index(drop=True)

    logger.info(f"시가총액 {min_cap_billion}억 이상 종목: {len(result_df)}개")

    return result_df


def get_volume_rank(date: str = None, top_n: int = 50) -> pd.DataFrame:
    """
    해당일 거래대금 상위 종목을 조회합니다.

    Args:
        date: 조회 날짜 (YYYYMMDD 형식, 기본값 오늘)
        top_n: 상위 몇 개 (기본 50개)

    Returns:
        DataFrame with columns: ['ticker', 'name', 'close', 'volume', 'trading_value', 'market']
        trading_value는 억원 단위

    Example:
        >>> top_volume = get_volume_rank(top_n=30)
        >>> print(top_volume.head(10))
    """
    if date is None:
        date = get_today_str()

    result_list = []

    for mkt in ["KOSPI", "KOSDAQ"]:
        logger.info(f"{mkt} 거래대금 조회 중...")

        try:
            # 거래대금 포함된 시가총액 데이터 조회
            df = stock.get_market_cap(date, market=mkt)
        except Exception as e:
            logger.error(f"{mkt} 거래대금 조회 실패: {e}")
            continue

        if "거래대금" not in df.columns:
            logger.warning(f"{mkt} 거래대금 컬럼 없음")
            continue

        for ticker in df.index:
            try:
                name = stock.get_market_ticker_name(ticker)
            except Exception:
                name = ""

            # ETF, 스팩 등 제외
            if any(keyword in name for keyword in EXCLUDE_KEYWORDS):
                continue

            try:
                ohlcv = stock.get_market_ohlcv(date, date, ticker)
                close = int(ohlcv["종가"].iloc[0]) if not ohlcv.empty else 0
                volume = int(ohlcv["거래량"].iloc[0]) if not ohlcv.empty else 0
            except Exception:
                close = 0
                volume = 0

            trading_value_billion = int(df.loc[ticker, "거래대금"] / 100_000_000)

            result_list.append({
                "ticker": ticker,
                "name": name,
                "close": close,
                "volume": volume,
                "trading_value": trading_value_billion,
                "market": mkt
            })

            time.sleep(REQUEST_DELAY)

    result_df = pd.DataFrame(result_list)

    # 거래대금 내림차순 정렬 후 상위 N개
    result_df = result_df.sort_values("trading_value", ascending=False).head(top_n)
    result_df = result_df.reset_index(drop=True)

    logger.info(f"거래대금 상위 {top_n}개 종목 조회 완료")

    return result_df


def get_fundamental_data(date: str = None) -> pd.DataFrame:
    """
    전 종목의 기본적 지표(PER, PBR, 배당수익률 등)를 조회합니다.

    Args:
        date: 조회 날짜 (YYYYMMDD 형식, 기본값 오늘)

    Returns:
        DataFrame with columns: ['ticker', 'name', 'per', 'pbr', 'dividend_yield', 'market']

    Example:
        >>> fundamentals = get_fundamental_data()
        >>> low_per = fundamentals[fundamentals['per'] < 10]
    """
    if date is None:
        date = get_today_str()

    result_list = []

    for mkt in ["KOSPI", "KOSDAQ"]:
        logger.info(f"{mkt} 기본적 지표 조회 중...")

        try:
            df = stock.get_market_fundamental(date, market=mkt)
        except Exception as e:
            logger.error(f"{mkt} 기본적 지표 조회 실패: {e}")
            continue

        for ticker in df.index:
            try:
                name = stock.get_market_ticker_name(ticker)
            except Exception:
                name = ""

            # ETF, 스팩 등 제외
            if any(keyword in name for keyword in EXCLUDE_KEYWORDS):
                continue

            result_list.append({
                "ticker": ticker,
                "name": name,
                "per": df.loc[ticker, "PER"] if "PER" in df.columns else None,
                "pbr": df.loc[ticker, "PBR"] if "PBR" in df.columns else None,
                "dividend_yield": df.loc[ticker, "DIV"] if "DIV" in df.columns else None,
                "market": mkt
            })

            time.sleep(REQUEST_DELAY)

    result_df = pd.DataFrame(result_list)
    logger.info(f"기본적 지표 조회 완료: {len(result_df)}개 종목")

    return result_df
