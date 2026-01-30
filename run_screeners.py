"""
퀀트 수준 주식 스크리너 실행 스크립트
FinanceDataReader를 사용하여 정교한 조건으로 종목을 발굴합니다.

스크리너 목록:
    1. 박스권 횡보: 60거래일 이상 횡보 (ATR(60)×5, 적응적 터치, 거래량 5%↓)
    2. 박스권 돌파 (거래량 동반): 완전한 박스권 조건 + 돌파 + 2배 거래량 + 150일선 위
    3. 박스권 돌파 (거래량 무관): 완전한 박스권 조건 + 저항선 돌파 후 10일 이내
    4. 풀백: 돌파 후 저항선으로 되돌아온 종목
    5. 거래량 폭발: 당일 거래량 6배 이상
    6. 거래량 급감: 급등 후 거래량 고갈 종목
"""
import FinanceDataReader as fdr
import pandas as pd
import numpy as np
import json
import pickle
import sys
import logging
from datetime import datetime, timedelta
import os
import time
import threading

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, List, Dict, Any
from tqdm import tqdm

# scipy.signal for vectorized pivot detection (optional, fallback to loop if not available)
try:
    from scipy.signal import argrelextrema
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# ============================================================================
# 데이터 캐시 (스레드 안전)
# ============================================================================
_DATA_CACHE: Dict[str, pd.DataFrame] = {}
_CACHE_LOCK = threading.Lock()  # 캐시 접근 동기화용 Lock
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.cache')
FORCE_DOWNLOAD = '--fresh' in sys.argv


@contextmanager
def cache_manager():
    """
    캐시 생명주기 관리를 위한 컨텍스트 매니저.
    사용 후 명시적으로 캐시를 정리하여 메모리 누수 방지.
    """
    global _DATA_CACHE
    try:
        yield _DATA_CACHE
    finally:
        clear_cache()


def clear_cache() -> None:
    """전역 캐시를 명시적으로 정리합니다."""
    global _DATA_CACHE
    with _CACHE_LOCK:
        _DATA_CACHE.clear()
    import gc
    gc.collect()

# ============================================================================
# 상수 정의
# ============================================================================
# 스크립트 위치 기준 상대 경로 (로컬/GitHub Actions 모두 호환)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, 'data')

# 박스권 스크리너 설정
BOX_PERIOD = 60  # 박스권 판단 기간 (약 3개월, 전체 스크리너 공통)
MIN_MARKET_CAP = 1000  # 최소 시가총액 (억원)
MAX_BOX_RANGE_PERCENT = 25.0  # 최대 허용 변동폭 (%)
ATR_PERIOD = 60  # ATR 계산 기간 (박스 기간과 동일)
ATR_MULTIPLE_MAX = 6  # ATR 배수 최대
ATR_TOUCH_MULTIPLE = 1.5  # ATR 기반 터치 허용범위 배수
PIVOT_WINDOW = 5  # 피벗 포인트 검출 윈도우
MIN_TOUCHES = 2  # 최소 터치 횟수
MAX_SLOPE_PERCENT = 0.05  # 최대 일평균 기울기 (%)
VOLUME_DECREASE_THRESHOLD = 0.95  # 거래량 감소 임계값 (후반 < 전반 × 0.95, 5% 감소)
BREAKOUT_WINDOW = 11  # 돌파 확인 윈도우 (10거래일 이내 = 오늘 포함 11행)

# ============================================================================
# 유틸리티 함수
# ============================================================================

def quick_range_check(df: pd.DataFrame, period: int, max_range: float) -> bool:
    """종가 기준 빠른 변동폭 사전 필터. 통과 시 True 반환."""
    last_close = df['Close'].iloc[-period:]
    if last_close.min() <= 0:
        return False
    quick_range = (last_close.max() - last_close.min()) / last_close.min() * 100
    return quick_range <= max_range


def calculate_atr(df: pd.DataFrame, period: int = 60) -> float:
    """
    ATR(Average True Range)을 계산합니다.

    ATR은 변동성 지표로, 박스권 범위를 동적으로 설정하는 데 사용됩니다.
    저변동 종목은 좁은 박스, 고변동 종목은 넓은 박스를 허용합니다.

    Args:
        df: OHLCV 데이터프레임 (High, Low, Close 컬럼 필수)
        period: ATR 계산 기간 (기본 60일)

    Returns:
        ATR 값 (비율, 0.01 = 1%)

    Formula:
        True Range = max(고가-저가, |고가-전일종가|, |저가-전일종가|)
        ATR = 60일간 True Range의 평균 / 종가

    Example:
        >>> atr = calculate_atr(df, 60)
        >>> print(f"ATR: {atr*100:.2f}%")
    """
    if len(df) < period + 1:
        return 0.0

    recent = df.tail(period + 1)

    high = recent['High'].values
    low = recent['Low'].values
    close = recent['Close'].values

    # True Range 벡터화 계산
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)

    current_close = close[-1]
    if current_close <= 0:
        return 0.0

    return float(np.mean(tr) / current_close)


def find_pivot_lows(prices: np.ndarray, n: int = 5) -> List[Tuple[int, float]]:
    """
    로컬 저점(Pivot Low)을 찾습니다.

    피벗 저점은 앞뒤 n일보다 낮은 가격을 가진 날입니다.
    지지선 확인에 사용됩니다.

    Args:
        prices: 종가 배열
        n: 피벗 검출 윈도우 (앞뒤 n일, 기본 5일)

    Returns:
        [(인덱스, 가격), ...] 형태의 피벗 저점 리스트

    Example:
        >>> pivots = find_pivot_lows(close_prices, n=5)
        >>> print(f"저점 {len(pivots)}개 발견")
    """
    if len(prices) < 2 * n + 1:
        return []

    # scipy가 사용 가능하면 벡터화된 argrelextrema 사용 (약 3-5배 빠름)
    if SCIPY_AVAILABLE:
        # argrelextrema는 order=n으로 앞뒤 n개와 비교
        indices = argrelextrema(prices, np.less_equal, order=n)[0]
        return [(int(i), float(prices[i])) for i in indices]

    # 폴백: 기존 Python 루프 방식
    pivots = []
    for i in range(n, len(prices) - n):
        window = prices[i-n:i+n+1]
        if prices[i] == np.min(window):
            pivots.append((i, prices[i]))
    return pivots


def find_pivot_highs(prices: np.ndarray, n: int = 5) -> List[Tuple[int, float]]:
    """
    로컬 고점(Pivot High)을 찾습니다.

    피벗 고점은 앞뒤 n일보다 높은 가격을 가진 날입니다.
    저항선 확인에 사용됩니다.

    Args:
        prices: 종가 배열
        n: 피벗 검출 윈도우 (앞뒤 n일, 기본 5일)

    Returns:
        [(인덱스, 가격), ...] 형태의 피벗 고점 리스트

    Example:
        >>> pivots = find_pivot_highs(close_prices, n=5)
        >>> print(f"고점 {len(pivots)}개 발견")
    """
    if len(prices) < 2 * n + 1:
        return []

    # scipy가 사용 가능하면 벡터화된 argrelextrema 사용 (약 3-5배 빠름)
    if SCIPY_AVAILABLE:
        # argrelextrema는 order=n으로 앞뒤 n개와 비교
        indices = argrelextrema(prices, np.greater_equal, order=n)[0]
        return [(int(i), float(prices[i])) for i in indices]

    # 폴백: 기존 Python 루프 방식
    pivots = []
    for i in range(n, len(prices) - n):
        window = prices[i-n:i+n+1]
        if prices[i] == np.max(window):
            pivots.append((i, prices[i]))
    return pivots


def count_touches_near_level(pivots: List[Tuple[int, float]], level: float, tolerance: float = 0.03) -> int:
    """
    특정 레벨 근처의 피벗 포인트 개수를 셉니다.

    지지선/저항선 확인에 사용됩니다.
    level ± tolerance 범위 안에 있는 피벗 개수를 반환합니다.

    Args:
        pivots: [(인덱스, 가격), ...] 형태의 피벗 리스트
        level: 기준 레벨 (가격)
        tolerance: 허용 오차 비율 (기본 3%)

    Returns:
        터치 횟수

    Example:
        >>> touches = count_touches_near_level(pivot_lows, box_low, 0.03)
        >>> if touches >= 2:
        ...     print("지지선 확인됨")
    """
    if level <= 0:
        return 0

    lower_bound = level * (1 - tolerance)
    upper_bound = level * (1 + tolerance)

    count = 0
    for _, price in pivots:
        if lower_bound <= price <= upper_bound:
            count += 1

    return count


def calculate_linear_slope(prices: np.ndarray) -> float:
    """
    선형회귀로 일평균 기울기를 계산합니다.

    가격의 추세를 판단하는 데 사용됩니다.
    기울기가 작으면 횡보, 크면 상승/하락 추세입니다.

    Args:
        prices: 종가 배열

    Returns:
        일평균 기울기 (% 단위)
        양수: 상승 추세
        음수: 하락 추세
        0 근처: 횡보

    Example:
        >>> slope = calculate_linear_slope(close_prices)
        >>> if abs(slope) <= 0.05:
        ...     print("횡보 중")
    """
    if len(prices) < 2:
        return 0.0

    n = len(prices)
    mean_price = np.mean(prices)
    if mean_price <= 0:
        return 0.0

    # NumPy 기반 선형회귀 (scipy보다 3-5배 빠름)
    x = np.arange(n)
    x_mean = (n - 1) / 2.0
    numerator = np.dot(x - x_mean, prices - mean_price)
    denominator = n * (n * n - 1) / 12.0

    slope = numerator / denominator
    slope_percent = (slope / mean_price) * 100

    return slope_percent


def check_volume_decrease(volumes: np.ndarray, period: int = 60, threshold: float = 0.9) -> Tuple[bool, float]:
    """
    거래량 감소 여부를 확인합니다.

    전반부 vs 후반부 평균 거래량을 비교합니다.
    후반부가 전반부 × threshold 미만이면 감소로 판단합니다.

    Args:
        volumes: 거래량 배열
        period: 분석 기간 (기본 60일)
        threshold: 감소 임계값 (기본 0.9 = 10% 이상 감소 필요)

    Returns:
        (감소 여부, 감소율 %)

    Example:
        >>> is_decreasing, rate = check_volume_decrease(volumes, 60, 0.9)
        >>> if is_decreasing:
        ...     print(f"거래량 {rate:.1f}% 감소")
    """
    if len(volumes) < period:
        return False, 0.0

    recent = volumes[-period:]
    half = period // 2

    first_half_avg = np.mean(recent[:half])
    second_half_avg = np.mean(recent[half:])

    if first_half_avg <= 0:
        return False, 0.0

    # 거래량 감소 확인: 후반부 < 전반부 × threshold
    is_decreasing = second_half_avg < first_half_avg * threshold
    decrease_rate = ((first_half_avg - second_half_avg) / first_half_avg) * 100

    return is_decreasing, decrease_rate


def calculate_actual_box_days(df: pd.DataFrame, box_high: float, box_low: float, min_period: int = 60, tolerance: float = 0.03) -> int:
    """
    실제 박스권 기간을 계산합니다.

    최근 min_period 일의 박스 범위를 기준으로 더 과거로 거슬러 올라가며
    종가가 박스 범위 내에 있는 동안의 총 기간을 계산합니다.

    Args:
        df: OHLCV 데이터프레임
        box_high: 박스 상단 가격
        box_low: 박스 하단 가격
        min_period: 최소 박스 기간 (기본 60일)
        tolerance: 박스 범위 확장 허용 오차 (ATR 기반)

    Returns:
        실제 박스권 기간 (일)
    """
    if len(df) < min_period:
        return min_period

    close_prices = df['Close'].values

    extended_high = box_high * (1 + tolerance)
    extended_low = box_low * (1 - tolerance)

    # 최근 min_period 일은 이미 박스권으로 확인됨
    total_days = min_period

    # 그 이전 날짜들을 역순으로 확인
    for i in range(len(close_prices) - min_period - 1, -1, -1):
        price = close_prices[i]
        # 가격이 확장된 박스 범위 내에 있으면 기간 추가
        if extended_low <= price <= extended_high:
            total_days += 1
        else:
            # 박스 범위를 벗어나면 중단
            break

    return total_days


def is_box_range(df: pd.DataFrame, period: int = 60) -> Tuple[bool, Dict[str, Any]]:
    """
    박스권 여부를 7가지 조건으로 판단합니다.

    조건:
        ① 데이터 검증: period일 데이터 온전함 (NaN 없음)
        ② 박스 기간: period 거래일 이상
        ③ 변동폭: 박스 범위 ≤ ATR(60) × 6 AND 박스 범위 ≤ 25%
        ④ 저점 터치: 박스 하단 ±ATR×1.5 영역에 로컬 저점 2개 이상
        ⑤ 고점 터치: 박스 상단 ±ATR×1.5 영역에 로컬 고점 2개 이상
        ⑥ 추세 필터: |선형회귀 기울기| ≤ 0.05% (일평균)
        ⑦ 거래량 감소: 후반부 평균 < 전반부 평균 × 0.95

    Args:
        df: OHLCV 데이터프레임
        period: 박스 판단 기간 (기본 60일)

    Returns:
        (박스권 여부, 상세 데이터 딕셔너리)

    Example:
        >>> is_box, data = is_box_range(df, 60)
        >>> if is_box:
        ...     print(f"박스권 확인: {data['range_percent']:.2f}%")
    """
    result_data = {
        'is_box': False,
        'box_high': 0,
        'box_low': 0,
        'range_percent': 0,
        'atr': 0,
        'atr_multiple': 0,
        'support_touches': 0,
        'resistance_touches': 0,
        'slope_daily': 0,
        'volume_decrease_rate': 0,
        'actual_days': period,  # 실제 횡보 기간 (기본값: 분석 기간)
        'failed_reason': ''
    }

    # ① 데이터 검증
    if len(df) < period:
        result_data['failed_reason'] = 'data_insufficient'
        return False, result_data

    recent = df.tail(period).copy()

    if recent['Close'].isna().any():
        result_data['failed_reason'] = 'nan_exists'
        return False, result_data

    close_prices = recent['Close'].values
    volumes = recent['Volume'].values

    # ② 박스 범위 계산
    box_high = float(np.max(close_prices))
    box_low = float(np.min(close_prices))

    if box_low <= 0:
        result_data['failed_reason'] = 'invalid_price'
        return False, result_data

    range_percent = (box_high - box_low) / box_low * 100

    result_data['box_high'] = int(box_high)
    result_data['box_low'] = int(box_low)
    result_data['range_percent'] = round(range_percent, 2)

    # ③ ATR 기반 변동폭 검사 (ATR(60) 사용 - 박스 기간 변동성 반영)
    atr = calculate_atr(df, ATR_PERIOD)
    atr_multiple = range_percent / (atr * 100) if atr > 0 else 999.0

    result_data['atr'] = round(atr * 100, 2)  # % 단위
    result_data['atr_multiple'] = round(atr_multiple, 2)

    # 조건: 박스 범위 ≤ ATR(60) × 6 AND 박스 범위 ≤ 25%
    if range_percent > MAX_BOX_RANGE_PERCENT:
        result_data['failed_reason'] = 'range_too_wide'
        return False, result_data

    if atr_multiple > ATR_MULTIPLE_MAX:
        result_data['failed_reason'] = 'atr_multiple_exceeded'
        return False, result_data

    # 적응적 터치 허용범위: ATR × 1.5 (저변동 종목은 좁게, 고변동 종목은 넓게)
    touch_tolerance = atr * ATR_TOUCH_MULTIPLE

    # ④ 저점 터치 확인
    pivot_lows = find_pivot_lows(close_prices, PIVOT_WINDOW)
    support_touches = count_touches_near_level(pivot_lows, box_low, touch_tolerance)
    result_data['support_touches'] = support_touches

    if support_touches < MIN_TOUCHES:
        result_data['failed_reason'] = 'support_touches_insufficient'
        return False, result_data

    # ⑤ 고점 터치 확인
    pivot_highs = find_pivot_highs(close_prices, PIVOT_WINDOW)
    resistance_touches = count_touches_near_level(pivot_highs, box_high, touch_tolerance)
    result_data['resistance_touches'] = resistance_touches

    if resistance_touches < MIN_TOUCHES:
        result_data['failed_reason'] = 'resistance_touches_insufficient'
        return False, result_data

    # ⑥ 추세 필터 (선형회귀)
    slope = calculate_linear_slope(close_prices)
    result_data['slope_daily'] = round(slope, 4)

    if abs(slope) > MAX_SLOPE_PERCENT:
        result_data['failed_reason'] = 'trending'
        return False, result_data

    # ⑦ 거래량 감소 확인 (후반부 < 전반부 × 0.95)
    is_vol_decreasing, vol_decrease_rate = check_volume_decrease(volumes, period, VOLUME_DECREASE_THRESHOLD)
    result_data['volume_decrease_rate'] = round(vol_decrease_rate, 2)

    if not is_vol_decreasing:
        result_data['failed_reason'] = 'volume_not_decreasing'
        return False, result_data

    # 모든 조건 통과
    result_data['is_box'] = True
    result_data['failed_reason'] = ''

    # 실제 횡보 기간 계산 (60일 이상일 수 있음)
    actual_days = calculate_actual_box_days(df, box_high, box_low, period, touch_tolerance)
    result_data['actual_days'] = actual_days

    return True, result_data


# ============================================================================
# 데이터 수집 함수
# ============================================================================

def get_stock_list() -> pd.DataFrame:
    """
    KOSPI + KOSDAQ 종목 리스트를 가져옵니다.

    시가총액 1,000억 이상 종목만 필터링합니다.

    Returns:
        DataFrame with columns: Code, Name, MarketCap, Market
    """
    print("종목 리스트 조회 중...")

    # KOSPI 종목
    kospi = fdr.StockListing('KOSPI')
    kospi['Market'] = 'KOSPI'

    # KOSDAQ 종목
    kosdaq = fdr.StockListing('KOSDAQ')
    kosdaq['Market'] = 'KOSDAQ'

    # 합치기
    stocks = pd.concat([kospi, kosdaq], ignore_index=True)

    # 시가총액 컬럼 처리
    if 'Marcap' in stocks.columns:
        stocks['MarketCap'] = stocks['Marcap'] / 100000000  # 원 -> 억원
    elif 'MarketCap' in stocks.columns:
        stocks['MarketCap'] = stocks['MarketCap'] / 100000000
    else:
        stocks['MarketCap'] = MIN_MARKET_CAP + 1

    # 시가총액 필터링
    stocks = stocks[stocks['MarketCap'] >= MIN_MARKET_CAP].copy()

    print(f"  시가총액 {MIN_MARKET_CAP}억 이상: {len(stocks)}개 종목")

    return stocks


def get_ohlcv(ticker: str, days: int = 200) -> Optional[pd.DataFrame]:
    """
    종목의 OHLCV 데이터를 가져옵니다.
    캐시가 있으면 캐시에서 반환하고, 없으면 API 호출합니다.
    스레드 안전한 캐시 접근을 보장합니다.

    Args:
        ticker: 종목코드
        days: 조회 기간 (거래일)

    Returns:
        OHLCV DataFrame or None
    """
    global _DATA_CACHE

    # 캐시에 있으면 캐시에서 반환 (Lock으로 동기화)
    with _CACHE_LOCK:
        if ticker in _DATA_CACHE:
            df = _DATA_CACHE[ticker]
            if df is not None and len(df) > 0:
                return df.tail(days) if len(df) >= days else df
            return None

    # 캐시에 없으면 API 호출 (Lock 밖에서 수행하여 병렬성 유지)
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days + 50)
        df = fdr.DataReader(ticker, start_date.strftime('%Y-%m-%d'))
        with _CACHE_LOCK:
            _DATA_CACHE[ticker] = df  # 캐시에 저장
        return df.tail(days) if len(df) >= days else df
    except Exception as e:
        logging.warning(f"Failed to fetch data for {ticker}: {e}")
        with _CACHE_LOCK:
            _DATA_CACHE[ticker] = None  # 실패도 캐시 (재시도 방지)
        return None


def _download_single_stock(args: Tuple[str, str], max_retries: int = 3) -> Tuple[str, Optional[pd.DataFrame]]:
    """
    단일 종목 데이터 다운로드 (병렬 처리용, 지수 백오프 재시도 포함)

    Args:
        args: (ticker, start_str) 튜플
        max_retries: 최대 재시도 횟수 (기본 3회)

    Returns:
        (ticker, DataFrame or None) 튜플
    """
    ticker, start_str = args
    base_delay = 0.5  # 기본 대기 시간 (초)

    for attempt in range(max_retries):
        try:
            df = fdr.DataReader(ticker, start_str)

            # 빈 데이터도 재시도 대상으로 처리
            if df is not None and len(df) > 0:
                return (ticker, df)

            # 빈 데이터인 경우도 재시도 (마지막 시도가 아닌 경우)
            if attempt < max_retries - 1:
                # 지수 백오프: base_delay * 2^attempt (0.5, 1.0, 2.0초)
                delay = base_delay * (2 ** attempt)
                time.sleep(delay)
                continue

        except Exception:
            if attempt < max_retries - 1:
                # 지수 백오프 적용
                delay = base_delay * (2 ** attempt)
                time.sleep(delay)
                continue

    return (ticker, None)


def _get_cache_path() -> str:
    """오늘 날짜 기준 캐시 파일 경로 반환"""
    today = datetime.now().strftime('%Y-%m-%d')
    return os.path.join(CACHE_DIR, f'stock_data_{today}.pkl')


def _load_cache() -> bool:
    """디스크 캐시에서 데이터 로딩. 성공 시 True 반환. 스레드 안전."""
    global _DATA_CACHE
    cache_path = _get_cache_path()

    if FORCE_DOWNLOAD:
        print("\n[캐시] --fresh 옵션: 강제 다운로드")
        return False

    if not os.path.exists(cache_path):
        return False

    try:
        print(f"\n[캐시] 오늘자 캐시 로딩: {os.path.basename(cache_path)}")
        with open(cache_path, 'rb') as f:
            loaded_data = pickle.load(f)
        with _CACHE_LOCK:
            _DATA_CACHE = loaded_data
        valid = sum(1 for v in _DATA_CACHE.values() if v is not None)
        print(f"  완료: {valid}개 종목 로딩됨 (캐시 사용)")
        return True
    except Exception as e:
        print(f"  캐시 로딩 실패: {e}")
        return False


def _save_cache() -> None:
    """현재 데이터를 디스크 캐시로 저장. 스레드 안전."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = _get_cache_path()

    # 이전 날짜 캐시 삭제
    for f in os.listdir(CACHE_DIR):
        if f.startswith('stock_data_') and f.endswith('.pkl'):
            old_path = os.path.join(CACHE_DIR, f)
            if old_path != cache_path:
                try:
                    os.remove(old_path)
                except OSError:
                    pass  # 삭제 실패는 무시

    try:
        # Lock 내에서 캐시 복사 후 저장 (Lock 보유 시간 최소화)
        with _CACHE_LOCK:
            cache_copy = _DATA_CACHE.copy()
        with open(cache_path, 'wb') as f:
            pickle.dump(cache_copy, f, protocol=pickle.HIGHEST_PROTOCOL)
        size_mb = os.path.getsize(cache_path) / (1024 * 1024)
        print(f"  캐시 저장: {size_mb:.0f}MB ({os.path.basename(cache_path)})")
    except Exception as e:
        print(f"  캐시 저장 실패: {e}")


def preload_all_data(stocks: pd.DataFrame, days: int = 200, max_workers: int = 60) -> None:
    """
    모든 종목의 데이터를 병렬로 다운로드하여 캐시합니다.
    오늘자 디스크 캐시가 있으면 다운로드 없이 즉시 로딩합니다.
    스레드 안전한 캐시 접근을 보장합니다.

    Args:
        stocks: 종목 리스트 DataFrame
        days: 조회 기간 (거래일)
        max_workers: 동시 다운로드 스레드 수
    """
    global _DATA_CACHE

    # 디스크 캐시 확인
    if _load_cache():
        return

    with _CACHE_LOCK:
        _DATA_CACHE = {}

    end_date = datetime.now()
    start_date = end_date - timedelta(days=days + 50)
    start_str = start_date.strftime('%Y-%m-%d')

    # 티커 목록 추출
    tickers = []
    for _, row in stocks.iterrows():
        ticker = row.get('Code', row.get('Symbol', ''))
        if ticker:
            tickers.append(ticker)

    print(f"\n[데이터 다운로드] {len(tickers)}개 종목 (스레드 {max_workers}개)...")

    success_count = 0
    fail_count = 0

    # 병렬 다운로드
    download_args = [(ticker, start_str) for ticker in tickers]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_download_single_stock, arg): arg[0] for arg in download_args}

        for future in tqdm(as_completed(futures), total=len(futures), desc="데이터 로딩"):
            ticker, df = future.result()
            # 스레드 안전한 캐시 업데이트
            with _CACHE_LOCK:
                _DATA_CACHE[ticker] = df
            if df is not None:
                success_count += 1
            else:
                fail_count += 1

    print(f"  완료: 성공 {success_count}개, 실패 {fail_count}개")

    # 디스크에 캐시 저장
    _save_cache()


# ============================================================================
# 박스권 스크리너
# ============================================================================

def screen_box_range(stocks: pd.DataFrame) -> List[Dict]:
    """
    박스권 스크리너: 60거래일 이상 진짜 횡보 중인 종목 발굴

    7가지 조건을 모두 충족해야 합니다:
        ① 시가총액 1,000억 원 이상
        ② 60일 데이터 검증 (NaN 없음)
        ③ 변동폭 ≤ ATR(60) × 6 AND ≤ 25%
        ④ 저점 터치 2회 이상 (±ATR×1.5 적응적 허용범위)
        ⑤ 고점 터치 2회 이상 (±ATR×1.5 적응적 허용범위)
        ⑥ 추세 필터: |기울기| ≤ 0.05%
        ⑦ 거래량 감소: 후반 30일 < 전반 30일 × 0.95

    Args:
        stocks: 종목 리스트 DataFrame

    Returns:
        박스권 종목 리스트
    """
    print("\n[박스권 스크리너 시작]")
    print(f"전체 종목: {len(stocks)}개")

    results = []

    # 필터링 통계
    stats = {
        'total': len(stocks),
        'market_cap': 0,
        'data_valid': 0,
        'range_atr': 0,
        'support_touch': 0,
        'resistance_touch': 0,
        'trend': 0,
        'volume': 0
    }

    start_time = time.time()

    for idx, row in stocks.iterrows():
        ticker = row.get('Code', row.get('Symbol', ''))
        name = row.get('Name', '')
        market_cap = row.get('MarketCap', 0)

        if not ticker:
            continue

        # ① 시가총액 필터
        if market_cap < MIN_MARKET_CAP:
            continue
        stats['market_cap'] += 1

        # 데이터 조회 (150일선 계산을 위해 더 긴 기간 필요)
        df = get_ohlcv(ticker, 200)
        if df is None or len(df) < BOX_PERIOD:
            continue
        stats['data_valid'] += 1

        # 박스권 판별
        is_box, box_data = is_box_range(df, BOX_PERIOD)

        # 통계 업데이트
        failed = box_data['failed_reason']
        if failed == '':
            pass
        elif failed in ['range_too_wide', 'atr_multiple_exceeded']:
            stats['range_atr'] += 1
            continue
        elif failed == 'support_touches_insufficient':
            stats['range_atr'] += 1
            stats['support_touch'] += 1
            continue
        elif failed == 'resistance_touches_insufficient':
            stats['range_atr'] += 1
            stats['support_touch'] += 1
            stats['resistance_touch'] += 1
            continue
        elif failed == 'trending':
            stats['range_atr'] += 1
            stats['support_touch'] += 1
            stats['resistance_touch'] += 1
            stats['trend'] += 1
            continue
        elif failed == 'volume_not_decreasing':
            stats['range_atr'] += 1
            stats['support_touch'] += 1
            stats['resistance_touch'] += 1
            stats['trend'] += 1
            stats['volume'] += 1
            continue
        else:
            continue

        if not is_box:
            continue

        # 모든 조건 통과
        stats['range_atr'] += 1
        stats['support_touch'] += 1
        stats['resistance_touch'] += 1
        stats['trend'] += 1
        stats['volume'] += 1

        # 현재가 및 등락률
        close_prices = df['Close']
        current_price = int(close_prices.iloc[-1])
        prev_price = float(close_prices.iloc[-2]) if len(close_prices) > 1 else current_price
        change_rate = (current_price - prev_price) / prev_price * 100 if prev_price > 0 else 0

        # 150일선 계산
        ma150 = None
        above_ma150 = False
        if len(close_prices) >= 150:
            ma150 = int(close_prices.rolling(150).mean().iloc[-1])
            above_ma150 = bool(current_price > ma150)

        results.append({
            'ticker': ticker,
            'name': name,
            'price': current_price,
            'change_rate': round(change_rate, 2),
            'market_cap': int(market_cap),  # 시가총액 (억원)
            'box_high': box_data['box_high'],
            'box_low': box_data['box_low'],
            'range_percent': box_data['range_percent'],
            'atr': box_data['atr'],
            'atr_multiple': box_data['atr_multiple'],
            'support_touches': box_data['support_touches'],
            'resistance_touches': box_data['resistance_touches'],
            'slope_daily': box_data['slope_daily'],
            'volume_decrease_rate': box_data['volume_decrease_rate'],
            'days': box_data.get('actual_days', BOX_PERIOD),  # 실제 횡보 기간 사용
            'ma150': ma150,  # 150일 이동평균
            'above_ma150': above_ma150,  # 150일선 위 여부
            'updated_at': datetime.now().isoformat()
        })

    elapsed = time.time() - start_time

    # 변동폭이 작은 순으로 정렬
    results.sort(key=lambda x: x['range_percent'])

    # 통계 출력
    print(f"├─ 시총 {MIN_MARKET_CAP}억+ 필터: {stats['total']} → {stats['market_cap']}개")
    print(f"├─ 데이터 검증: {stats['market_cap']} → {stats['data_valid']}개")
    print(f"├─ 변동폭(ATR) 필터: {stats['data_valid']} → {stats['range_atr']}개")
    print(f"├─ 저점 터치 필터: {stats['range_atr']} → {stats['support_touch']}개")
    print(f"├─ 고점 터치 필터: {stats['support_touch']} → {stats['resistance_touch']}개")
    print(f"├─ 추세 필터: {stats['resistance_touch']} → {stats['trend']}개")
    print(f"└─ 거래량 감소 필터: {stats['trend']} → {stats['volume']}개")
    print(f"[완료] 박스권 종목: {len(results)}개 (소요시간: {elapsed:.1f}초)")

    return results


# ============================================================================
# 박스권 돌파 스크리너 (거래량 동반)
# ============================================================================

def screen_box_breakout(stocks: pd.DataFrame) -> List[Dict]:
    """
    박스권 돌파 스크리너 (거래량 동반)

    조건:
        - 사전 조건: 수정된 박스권 조건 만족 (이전에 박스권이었던 종목)
        - 저항선: 박스 상단
        - 돌파 조건: 당일 종가 > 저항선 × 1.015
        - 거래량 조건: 당일 거래량 ≥ 20일 평균 × 2배
        - 이평선 조건: 150일선 위

    Args:
        stocks: 종목 리스트 DataFrame

    Returns:
        돌파 종목 리스트
    """
    print("\n[박스권 돌파 (거래량 동반) 스크리너 시작]")
    results = []

    stats = {'total': len(stocks), 'box_history': 0, 'breakout': 0, 'volume': 0, 'ma150': 0}

    for idx, row in stocks.iterrows():
        ticker = row.get('Code', row.get('Symbol', ''))
        name = row.get('Name', '')
        market_cap = row.get('MarketCap', 0)

        if not ticker:
            continue

        df = get_ohlcv(ticker, 200)
        if df is None or len(df) < 160:
            continue

        # 박스권 + 돌파 확인 (합집합 방식)
        breakout_day = None
        breakout_idx = None
        box_high = None
        box_low = None

        for box_end in range(BREAKOUT_WINDOW, 0, -1):
            box_start = BOX_PERIOD * 2 + box_end
            if len(df) < box_start:
                continue
            box_period_df = df.iloc[-box_start:-box_end]
            if len(box_period_df) < BOX_PERIOD + 1:
                continue

            # 빠른 사전 필터
            if not quick_range_check(box_period_df, BOX_PERIOD, MAX_BOX_RANGE_PERCENT):
                continue

            is_box, box_data = is_box_range(box_period_df, BOX_PERIOD)
            if not is_box:
                continue

            candidate_days = df.iloc[-box_end:]
            for i, (date, row_data) in enumerate(candidate_days.iterrows()):
                if row_data['Close'] > box_data['box_high'] * 1.015:
                    breakout_day = date
                    breakout_idx = df.index.get_loc(date)
                    box_high = box_data['box_high']
                    box_low = box_data['box_low']
                    break
            if breakout_day is not None:
                break

        if breakout_day is None:
            continue

        stats['box_history'] += 1
        stats['breakout'] += 1

        # 거래량 조건: 돌파일 거래량 ≥ 20일 평균 × 2배
        if breakout_idx < 20:
            continue

        avg_volume = df['Volume'].iloc[breakout_idx-20:breakout_idx].mean()
        breakout_volume = df['Volume'].iloc[breakout_idx]

        if avg_volume <= 0:
            continue

        volume_ratio = breakout_volume / avg_volume
        if volume_ratio < 2:
            continue

        stats['volume'] += 1

        # 150일선 조건
        ma150 = df['Close'].rolling(150).mean()
        if pd.isna(ma150.iloc[-1]):
            continue

        current_price = df['Close'].iloc[-1]
        if current_price <= ma150.iloc[-1]:
            continue

        stats['ma150'] += 1

        # 등락률
        prev_price = float(df['Close'].iloc[-2])
        change_rate = (current_price - prev_price) / prev_price * 100 if prev_price > 0 else 0

        # 저항선 대비 상승률
        breakout_pct = (current_price - box_high) / box_high * 100

        results.append({
            'ticker': ticker,
            'name': name,
            'price': int(current_price),
            'change_rate': round(change_rate, 2),
            'breakout_date': breakout_day.strftime('%Y-%m-%d'),
            'breakout_price': int(box_high),
            'breakout_pct': round(breakout_pct, 2),
            'volume_ratio': round(volume_ratio, 1),
            'ma150': int(ma150.iloc[-1]),
            'above_ma150': bool(current_price > ma150.iloc[-1]),
            'market_cap': int(market_cap),
            'updated_at': datetime.now().isoformat()
        })

    # 거래량 배수 높은 순 정렬
    results.sort(key=lambda x: x['volume_ratio'], reverse=True)

    print(f"├─ 박스권 이력: {stats['total']} → {stats['box_history']}개")
    print(f"├─ 돌파 확인: {stats['box_history']} → {stats['breakout']}개")
    print(f"├─ 거래량 2배+: {stats['breakout']} → {stats['volume']}개")
    print(f"└─ 150일선 위: {stats['volume']} → {stats['ma150']}개")
    print(f"[완료] 박스권 돌파 종목: {len(results)}개")

    return results


# ============================================================================
# 박스권 돌파 스크리너 (거래량 무관)
# ============================================================================

def screen_box_breakout_simple(stocks: pd.DataFrame) -> List[Dict]:
    """
    박스권 돌파 스크리너 (거래량 무관)

    조건:
        - 사전 조건: 박스권 전체 조건 만족 (60일 기간)
        - 저항선: 박스 상단
        - 돌파 조건: 종가 > 저항선 × 1.015 (상단 +1.5% 초과)
        - 경과 기간: 돌파일로부터 10 거래일 이내
        - 거래량/이평선 조건: 없음

    Args:
        stocks: 종목 리스트 DataFrame

    Returns:
        돌파 종목 리스트
    """
    print("\n[박스권 돌파 (거래량 무관) 스크리너 시작]")
    results = []

    for idx, row in stocks.iterrows():
        ticker = row.get('Code', row.get('Symbol', ''))
        name = row.get('Name', '')
        market_cap = row.get('MarketCap', 0)

        if not ticker:
            continue

        df = get_ohlcv(ticker, 200)
        if df is None or len(df) < BOX_PERIOD * 2 + BREAKOUT_WINDOW:
            continue

        # 박스권 + 돌파 확인 (합집합 방식)
        breakout_day = None
        days_since = 0
        resistance = None

        for box_end in range(BREAKOUT_WINDOW, 0, -1):
            box_start = BOX_PERIOD * 2 + box_end
            if len(df) < box_start:
                continue
            box_period_df = df.iloc[-box_start:-box_end]
            if len(box_period_df) < BOX_PERIOD + 1:
                continue

            # 빠른 사전 필터
            if not quick_range_check(box_period_df, BOX_PERIOD, MAX_BOX_RANGE_PERCENT):
                continue

            is_box, box_data = is_box_range(box_period_df, BOX_PERIOD)
            if not is_box:
                continue

            candidate_days = df.iloc[-box_end:]
            for i, (date, row_data) in enumerate(candidate_days.iterrows()):
                if row_data['Close'] > box_data['box_high'] * 1.015:
                    breakout_day = date
                    days_since = len(candidate_days) - i - 1
                    resistance = box_data['box_high']
                    break
            if breakout_day is not None:
                break

        if breakout_day is None:
            continue

        current_price = df['Close'].iloc[-1]
        prev_price = float(df['Close'].iloc[-2])
        change_rate = (current_price - prev_price) / prev_price * 100 if prev_price > 0 else 0

        # 저항선 대비 상승률
        breakout_pct = (current_price - resistance) / resistance * 100

        # 150일선 계산
        ma150 = None
        above_ma150 = False
        if len(df) >= 150:
            ma150 = int(df['Close'].rolling(150).mean().iloc[-1])
            above_ma150 = bool(current_price > ma150)

        results.append({
            'ticker': ticker,
            'name': name,
            'price': int(current_price),
            'change_rate': round(change_rate, 2),
            'breakout_date': breakout_day.strftime('%Y-%m-%d'),
            'resistance': int(resistance),
            'days_since_breakout': days_since,
            'breakout_pct': round(breakout_pct, 2),
            'ma150': ma150,
            'above_ma150': above_ma150,
            'market_cap': int(market_cap),
            'updated_at': datetime.now().isoformat()
        })

    # 최근 돌파 순 정렬
    results.sort(key=lambda x: x['days_since_breakout'])

    print(f"[완료] 박스권 돌파 (무관) 종목: {len(results)}개")

    return results


# ============================================================================
# 풀백 스크리너
# ============================================================================

# 풀백 설정
PULLBACK_LOOKBACK = 20         # 돌파 탐색 기간 (거래일)
PULLBACK_MIN_DAYS = 3          # 풀백 최소 경과일
PULLBACK_TOLERANCE = 0.05      # 풀백 허용범위 (±5%)
PULLBACK_VOLUME_DECREASE = 50  # 거래량 감소 임계값 (%)
PULLBACK_RECENT_DAYS = 5       # 최근 거래량 평균 기간

def screen_pullback(stocks: pd.DataFrame) -> List[Dict]:
    """
    풀백 스크리너: 돌파 후 저항선으로 되돌아온 종목

    조건:
        - 사전 조건: 박스권 돌파 (거래량 동반) 조건 충족한 돌파 발생 이력
        - 풀백 기간: 돌파 후 10 거래일 이내
        - 풀백 조건: 현재 주가가 저항선 ± 5% 영역으로 되돌아옴
        - 거래량 조건: 풀백 구간 평균 거래량 < 돌파일 거래량 × 50%
        - 이평선 조건: 150일선 위 유지

    Args:
        stocks: 종목 리스트 DataFrame

    Returns:
        풀백 종목 리스트
    """
    print("\n[풀백 스크리너 시작]")
    results = []

    for idx, row in stocks.iterrows():
        ticker = row.get('Code', row.get('Symbol', ''))
        name = row.get('Name', '')
        market_cap = row.get('MarketCap', 0)

        if not ticker:
            continue

        df = get_ohlcv(ticker, 200)
        if df is None or len(df) < 160:
            continue

        # 박스권 구간 확인 (140~20일 전 데이터, 마지막 60일을 박스로 분석)
        box_period_df = df.iloc[-(BOX_PERIOD*2+20):-20].copy()
        if len(box_period_df) < BOX_PERIOD + 1:
            continue

        # 박스권이었는지 체크 (전체 조건 적용)
        is_box, box_data = is_box_range(box_period_df, BOX_PERIOD)
        if not is_box:
            continue

        resistance = box_data['box_high']

        # 돌파 기간 내 거래량 2배 이상이었는지 확인
        breakout_period = df.iloc[-PULLBACK_LOOKBACK:-PULLBACK_MIN_DAYS]
        breakout_day = None
        breakout_idx = None
        breakout_volume = 0

        for i, (date, row_data) in enumerate(breakout_period.iterrows()):
            idx_in_df = df.index.get_loc(date)
            if idx_in_df < 20:
                continue

            # 돌파 조건
            if row_data['Close'] > resistance * 1.015:
                # 거래량 조건
                avg_vol = df['Volume'].iloc[idx_in_df-20:idx_in_df].mean()
                if avg_vol > 0 and row_data['Volume'] >= avg_vol * 2:
                    breakout_day = date
                    breakout_idx = idx_in_df
                    breakout_volume = row_data['Volume']
                    break

        if breakout_day is None:
            continue

        # 150일선 조건
        ma150 = df['Close'].rolling(150).mean()
        if pd.isna(ma150.iloc[-1]):
            continue

        current_price = float(df['Close'].iloc[-1])
        if current_price <= ma150.iloc[-1]:
            continue

        # 풀백 조건: 현재가가 저항선 허용범위 이내
        pullback_lower = resistance * (1 - PULLBACK_TOLERANCE)
        pullback_upper = resistance * (1 + PULLBACK_TOLERANCE)

        if not (pullback_lower <= current_price <= pullback_upper):
            continue

        # 거래량 감소 조건
        recent_avg_volume = df['Volume'].iloc[-PULLBACK_RECENT_DAYS:].mean()
        if breakout_volume > 0:
            volume_decrease = (1 - recent_avg_volume / breakout_volume) * 100
            if volume_decrease < PULLBACK_VOLUME_DECREASE:
                continue
        else:
            continue

        prev_price = float(df['Close'].iloc[-2])
        change_rate = (current_price - prev_price) / prev_price * 100 if prev_price > 0 else 0

        pullback_ratio = abs((current_price - resistance) / resistance * 100)

        results.append({
            'ticker': ticker,
            'name': name,
            'price': int(current_price),
            'change_rate': round(change_rate, 2),
            'breakout_date': breakout_day.strftime('%Y-%m-%d'),
            'resistance': int(resistance),
            'pullback_ratio': round(pullback_ratio, 2),
            'volume_decrease': int(volume_decrease),
            'ma150': int(ma150.iloc[-1]),
            'above_ma150': bool(current_price > ma150.iloc[-1]),
            'market_cap': int(market_cap),
            'updated_at': datetime.now().isoformat()
        })

    # 풀백 비율이 작은 순 정렬 (저항선에 가까울수록 좋음)
    results.sort(key=lambda x: x['pullback_ratio'])

    print(f"[완료] 풀백 종목: {len(results)}개")

    return results


# ============================================================================
# 거래량 폭발 스크리너
# ============================================================================

def screen_volume_explosion(stocks: pd.DataFrame) -> List[Dict]:
    """
    거래량 폭발 스크리너: 당일 거래량이 40일 평균의 6배 이상 + 6% 이상 장대양봉

    Args:
        stocks: 종목 리스트 DataFrame

    Returns:
        거래량 폭발 종목 리스트
    """
    print("\n[거래량 폭발 스크리너 시작]")
    results = []

    for idx, row in stocks.iterrows():
        ticker = row.get('Code', row.get('Symbol', ''))
        name = row.get('Name', '')
        market_cap = row.get('MarketCap', 0)

        if not ticker:
            continue

        df = get_ohlcv(ticker, 200)  # 150일선 계산을 위해 200일 필요
        if df is None or len(df) < 41:
            continue

        # 40일 평균 거래량
        avg_volume = df['Volume'].iloc[-41:-1].mean()
        today_volume = df['Volume'].iloc[-1]

        if avg_volume <= 0:
            continue

        volume_ratio = today_volume / avg_volume

        # 6배 이상
        if volume_ratio < 6:
            continue

        current_price = int(df['Close'].iloc[-1])
        prev_price = float(df['Close'].iloc[-2])
        change_rate = (current_price - prev_price) / prev_price * 100 if prev_price > 0 else 0

        # 6% 이상 장대양봉
        if change_rate < 6:
            continue

        # 150일선 계산
        ma150 = None
        above_ma150 = False
        if len(df) >= 150:
            ma150 = int(df['Close'].rolling(150).mean().iloc[-1])
            above_ma150 = bool(current_price > ma150)

        results.append({
            'ticker': ticker,
            'name': name,
            'price': current_price,
            'change_rate': round(change_rate, 2),
            'volume_ratio': round(volume_ratio, 1),
            'volume_rank': 0,
            'ma150': ma150,
            'above_ma150': above_ma150,
            'market_cap': int(market_cap),
            'updated_at': datetime.now().isoformat()
        })

    # 거래량 배수 높은 순 정렬
    results.sort(key=lambda x: x['volume_ratio'], reverse=True)

    # 순위 설정
    for i, r in enumerate(results):
        r['volume_rank'] = i + 1

    print(f"[완료] 거래량 폭발 종목: {len(results)}개")

    return results


# ============================================================================
# 거래량 급감 스크리너
# ============================================================================

# 거래량 급감 설정
DRYUP_LOOKBACK = 20            # 급등일 탐색 기간 (거래일)
DRYUP_MIN_CHANGE = 8           # 최소 급등 변화율 (%)
DRYUP_GAP_LIMIT = 0.97         # 갭다운 제한 (전일종가 대비)
DRYUP_MAX_SHADOW = 0.3         # 최대 윗꼬리 비율
DRYUP_VOLUME_MULTIPLE = 4      # 급등일 최소 거래량 배수
DRYUP_MIN_DAYS = 3             # 눌림목 최소 경과일 (급등 후)
DRYUP_MAX_DAYS = 8             # 눌림목 최대 경과일 (급등 후)
DRYUP_DISPLAY_DAYS = 8         # 포착 후 표시 유지 기간
DRYUP_PRICE_VS_OPEN = 0.98     # 현재가 vs 급등봉 시가
DRYUP_PRICE_VS_HIGH = 0.90     # 현재가 vs 급등 고점
DRYUP_RECENT_DAYS = 3          # 최근 거래량 평균 기간
DRYUP_VOLUME_DECREASE = 60     # 거래량 감소율 임계값 (%)

def screen_volume_dry_up(stocks: pd.DataFrame) -> List[Dict]:
    """
    거래량 급감 눌림목 스크리너

    [급등봉 조건]
        1. 종가 > 전일종가 × 1.08 (8% 이상 양봉)
        2. 거래량 > 20일 평균 × 4배
        3. (고가 - 종가) / (고가 - 저가) < 0.3 (윗꼬리 30% 이하)
        4. 종가 > 20일선
        5. 시가 >= 전일종가 × 0.97 (갭다운 -3% 이상 제외)

    [눌림목 조건] (급등 후 3~8일)
        1. 현재가 > 급등봉 시가 × 0.98 (가격 유지)
        2. 현재가 > 급등 고점 × 0.90 (조정폭 -10% 이내)
        3. 현재가 > 급등 전일종가 (급등 전 가격 위)
        4. 현재가 > 20일선 (추세 유지)
        5. 최근 3일 평균 거래량 < 급등일 × 0.4 (60% 감소)

    [표시 조건]
        - 조건 포착 후 8거래일까지 계속 표시 (추이 관찰용)
    """
    print("\n[거래량 급감 눌림목 스크리너 시작]")
    results = []

    # 탐색 범위 확대: 포착 후 DISPLAY_DAYS까지 표시하기 위해
    max_search_days = DRYUP_MAX_DAYS + DRYUP_DISPLAY_DAYS

    for idx, row in stocks.iterrows():
        ticker = row.get('Code', row.get('Symbol', ''))
        name = row.get('Name', '')
        market_cap = row.get('MarketCap', 0)

        if not ticker:
            continue

        df = get_ohlcv(ticker, 200)
        if df is None or len(df) < 30:
            continue

        # 급등일 후보 찾기 (더 넓은 범위에서 탐색)
        found_result = None

        for days_ago in range(DRYUP_MIN_DAYS, max_search_days + 1):
            if days_ago >= len(df) - DRYUP_LOOKBACK - 1:
                continue

            explosion_idx = len(df) - 1 - days_ago

            # 급등봉 조건 검증
            prev_close = df['Close'].iloc[explosion_idx - 1]
            curr_close = df['Close'].iloc[explosion_idx]
            curr_open = df['Open'].iloc[explosion_idx]
            curr_high = df['High'].iloc[explosion_idx]
            curr_low = df['Low'].iloc[explosion_idx]
            curr_volume = df['Volume'].iloc[explosion_idx]

            if prev_close <= 0:
                continue

            change = (curr_close - prev_close) / prev_close * 100

            # 조건1: 최소 급등폭 이상 양봉
            if change < DRYUP_MIN_CHANGE:
                continue

            # 조건5: 시가 >= 전일종가 × 갭다운 제한
            if curr_open < prev_close * DRYUP_GAP_LIMIT:
                continue

            # 조건3: 윗꼬리 제한
            candle_range = curr_high - curr_low
            if candle_range <= 0:
                continue
            upper_shadow_ratio = (curr_high - curr_close) / candle_range
            if upper_shadow_ratio >= DRYUP_MAX_SHADOW:
                continue

            # 조건2: 거래량 > 20일 평균 × N배
            if explosion_idx < DRYUP_LOOKBACK:
                continue
            avg_volume_20d = df['Volume'].iloc[explosion_idx - DRYUP_LOOKBACK:explosion_idx].mean()
            if avg_volume_20d <= 0:
                continue
            volume_ratio = curr_volume / avg_volume_20d
            if volume_ratio < DRYUP_VOLUME_MULTIPLE:
                continue

            # 조건4: 종가 > 20일선
            ma20_at_explosion = df['Close'].iloc[explosion_idx - (DRYUP_LOOKBACK - 1):explosion_idx + 1].mean()
            if curr_close <= ma20_at_explosion:
                continue

            # 급등봉 조건 통과 - 눌림목 조건 확인
            explosion_day = df.index[explosion_idx]
            explosion_volume = curr_volume

            # 급감 조건 확인: 급등 후 3~8일 사이에 급감이 발생했는지 찾기
            for detect_offset in range(DRYUP_MIN_DAYS, DRYUP_MAX_DAYS + 1):
                detect_idx = explosion_idx + detect_offset

                if detect_idx >= len(df):
                    continue

                # 포착일까지의 거래량 평균 (최근 3일)
                vol_start = max(explosion_idx + 1, detect_idx - DRYUP_RECENT_DAYS + 1)
                post_volumes = df['Volume'].iloc[vol_start:detect_idx + 1]
                if len(post_volumes) < 1:
                    continue

                recent_avg_volume = post_volumes.mean()
                if pd.isna(recent_avg_volume) or explosion_volume <= 0:
                    continue

                # 급감 조건: 폭발일 대비 60% 이상 감소
                volume_decrease_rate = (1 - recent_avg_volume / explosion_volume) * 100
                if volume_decrease_rate < DRYUP_VOLUME_DECREASE:
                    continue

                # 포착일 가격 조건 확인
                detect_price = df['Close'].iloc[detect_idx]

                # 눌림목 조건1: 포착일 종가 > 급등봉 시가
                if detect_price <= curr_open * DRYUP_PRICE_VS_OPEN:
                    continue

                # 눌림목 조건2: 포착일 종가 > 급등 고점
                if detect_price <= curr_high * DRYUP_PRICE_VS_HIGH:
                    continue

                # 눌림목 조건3: 포착일 종가 > 급등 전일종가
                if detect_price <= prev_close:
                    continue

                # 눌림목 조건4: 포착일 종가 > 20일선
                ma20_at_detect = df['Close'].iloc[detect_idx - (DRYUP_LOOKBACK - 1):detect_idx + 1].mean()
                if detect_price <= ma20_at_detect:
                    continue

                # 모든 조건 통과 - 포착일 확정
                detected_date = df.index[detect_idx]
                days_since_detected = len(df) - 1 - detect_idx

                # 포착 후 8거래일 이내만 표시
                if days_since_detected <= DRYUP_DISPLAY_DAYS:
                    found_result = {
                        'explosion_day': explosion_day,
                        'explosion_change': change,
                        'explosion_volume_ratio': volume_ratio,
                        'explosion_open': curr_open,
                        'explosion_high': curr_high,
                        'explosion_prev_close': prev_close,
                        'explosion_volume': explosion_volume,
                        'detected_date': detected_date,
                        'days_since_detected': days_since_detected,
                        'volume_decrease_rate': volume_decrease_rate
                    }
                    break

            if found_result:
                break

        if not found_result:
            continue

        # 현재 가격 조건도 확인 (현재도 조건 유지 중인지)
        current_price = float(df['Close'].iloc[-1])
        ma20_current = df['Close'].iloc[-DRYUP_LOOKBACK:].mean()

        # 현재가 조건 체크 (너무 많이 빠지면 제외)
        if current_price <= found_result['explosion_prev_close'] * 0.95:
            continue

        # 150일선 계산
        ma150 = None
        above_ma150 = False
        if len(df) >= 150:
            ma150 = int(df['Close'].rolling(150).mean().iloc[-1])
            above_ma150 = bool(current_price > ma150)

        # 등락률
        prev_price = float(df['Close'].iloc[-2])
        change_rate = (current_price - prev_price) / prev_price * 100 if prev_price > 0 else 0

        results.append({
            'ticker': ticker,
            'name': name,
            'price': int(current_price),
            'change_rate': round(change_rate, 2),
            'explosion_date': found_result['explosion_day'].strftime('%Y-%m-%d'),
            'explosion_change_rate': round(found_result['explosion_change'], 1),
            'explosion_volume_ratio': round(found_result['explosion_volume_ratio'], 1),
            'detected_date': found_result['detected_date'].strftime('%Y-%m-%d'),
            'days_since_detected': found_result['days_since_detected'],
            'volume_decrease_rate': int(found_result['volume_decrease_rate']),
            'volume_rank': 0,
            'ma150': ma150,
            'above_ma150': above_ma150,
            'market_cap': int(market_cap),
            'updated_at': datetime.now().isoformat()
        })

    # 포착 경과일 오름차순, 같으면 거래량 감소율 높은 순
    results.sort(key=lambda x: (x['days_since_detected'], -x['volume_decrease_rate']))

    for i, r in enumerate(results):
        r['volume_rank'] = i + 1

    print(f"[완료] 거래량 급감 눌림목: {len(results)}개")

    return results


# ============================================================================
# 7. 낙폭과대 반등 스크리너 (52주 고가 대비 -40% + 바닥 상승)
# ============================================================================

FALLEN_DROP_THRESHOLD = 0.60  # 52주 고가 대비 60% 이하 (40% 이상 하락)
FALLEN_52W_PERIOD = 250       # 52주 ≈ 250거래일
FALLEN_LOW_PERIOD = 20        # 저가 비교 기간


def screen_fallen_rebound(stocks: pd.DataFrame) -> List[Dict]:
    """
    낙폭과대 반등 스크리너

    52주 고가 대비 40% 이상 하락했으나 바닥이 높아지는 종목을 발굴합니다.

    [조건]
        A. 현재가 < 52주 고가 × 60% (40% 이상 하락)
        C. 최근 20일 저가 > 이전 20일 저가 (바닥 상승 = Higher Low)
    """
    print("\n[낙폭과대 반등] 분석 시작...")

    # PER/PBR 데이터 로드 (financial_data.json)
    financial_data = {}
    financial_path = os.path.join(DATA_PATH, 'financial_data.json')
    if os.path.exists(financial_path):
        try:
            with open(financial_path, 'r', encoding='utf-8') as f:
                fin_json = json.load(f)
                financial_data = fin_json.get('data', {})
        except Exception as e:
            logging.warning(f"Failed to load financial data from {financial_path}: {e}")

    results = []
    required_days = FALLEN_52W_PERIOD + 10

    for _, row in stocks.iterrows():
        ticker = row.get('Code', row.get('Symbol', ''))
        name = row.get('Name', '')
        market_cap = row.get('MarketCap', 0)

        # PER/PBR from financial_data
        fin_info = financial_data.get(ticker, {})
        metrics = fin_info.get('metrics', {})
        per_list = metrics.get('per', [])
        pbr_list = metrics.get('pbr', [])
        per = per_list[0] if per_list else None
        pbr = pbr_list[0] if pbr_list else None

        if not ticker:
            continue

        df = get_ohlcv(ticker, required_days)
        if df is None or len(df) < FALLEN_52W_PERIOD:
            continue

        current_price = float(df['Close'].iloc[-1])

        # 52주 고가 (High 기준)
        high_52w = float(df['High'].iloc[-FALLEN_52W_PERIOD:].max())

        if high_52w <= 0:
            continue

        # A. 52주 고가 대비 40% 이상 하락
        drop_ratio = current_price / high_52w
        if drop_ratio > FALLEN_DROP_THRESHOLD:
            continue

        drop_percent = (1 - drop_ratio) * 100  # 하락률 (양수)

        # C. 바닥 상승 (Higher Low)
        # 이전 20일 저가 (21~40일 전)
        if len(df) < 41:
            continue
        prev_20d_low = float(df['Low'].iloc[-40:-20].min())
        # 최근 20일 저가 (1~20일 전)
        recent_20d_low = float(df['Low'].iloc[-20:].min())

        # 유효성 검사
        if prev_20d_low <= 0 or recent_20d_low <= 0:
            continue

        if recent_20d_low <= prev_20d_low:
            continue

        # 바닥 상승률
        low_rise_percent = (recent_20d_low - prev_20d_low) / prev_20d_low * 100

        # 150일선 계산
        ma150 = None
        above_ma150 = False
        if len(df) >= 150:
            ma150 = int(df['Close'].rolling(150).mean().iloc[-1])
            above_ma150 = bool(current_price > ma150)

        # 등락률
        prev_price = float(df['Close'].iloc[-2])
        change_rate = (current_price - prev_price) / prev_price * 100 if prev_price > 0 else 0

        results.append({
            'ticker': ticker,
            'name': name,
            'price': int(current_price),
            'change_rate': round(change_rate, 2),
            'high_52w': int(high_52w),
            'drop_percent': round(drop_percent, 1),
            'prev_low': int(prev_20d_low),
            'recent_low': int(recent_20d_low),
            'low_rise_percent': round(low_rise_percent, 1),
            'per': round(per, 1) if per and not pd.isna(per) else None,
            'pbr': round(pbr, 2) if pbr and not pd.isna(pbr) else None,
            'ma150': ma150,
            'above_ma150': above_ma150,
            'market_cap': int(market_cap),
            'updated_at': datetime.now().isoformat()
        })

    # 낙폭 큰 순서로 정렬
    results.sort(key=lambda x: x['drop_percent'], reverse=True)

    print(f"[완료] 낙폭과대 반등: {len(results)}개")

    return results


# ============================================================================
# 8. 52주 신고가 돌파 스크리너
# ============================================================================

HIGH_52W_PERIOD = 250  # 52주 ≈ 250거래일
MAX_DAYS_SINCE_BREAKOUT = 8  # 돌파 후 최대 거래일

def screen_new_high_52w(stocks: pd.DataFrame) -> List[Dict]:
    """
    52주 신고가를 돌파한 종목을 발굴합니다.

    [52주 고가 기준]
        - High(고가) 기준 250거래일 최고가

    [돌파 조건]
        1. 종가 > 52주 고가 (High 기준)
        2. 돌파일 거래량 ≥ 20일 평균 × 1.5
        3. 현재가 > 150일선
        4. 150일선 우상향 (현재 150MA > 20일 전 150MA)
        5. 돌파 후 8거래일 이내
        6. 현재가 > 52주 고가 유지
    """
    print("\n[52주 신고가 돌파] 분석 시작...")

    results = []
    required_days = HIGH_52W_PERIOD + MAX_DAYS_SINCE_BREAKOUT + 2
    MA150_PERIOD = 150
    MA150_SLOPE_DAYS = 20  # 150MA 우상향 판단 기간

    for _, row in stocks.iterrows():
        ticker = row.get('Code', row.get('Symbol', ''))
        name = row.get('Name', '')
        market_cap = row.get('MarketCap', 0)
        if not isinstance(market_cap, (int, float)):
            market_cap = 0

        if market_cap < MIN_MARKET_CAP:
            continue

        df = get_ohlcv(ticker, required_days + 50)
        if df is None or len(df) < required_days:
            continue

        close = df['Close']
        high = df['High']
        volume = df['Volume']

        if close.isna().sum() > 10:
            continue

        total_len = len(df)
        if total_len < MA150_PERIOD + MA150_SLOPE_DAYS:
            continue

        # --- 추세 필터: 150일선 위 + 우상향 ---
        ma150_now = close.iloc[-MA150_PERIOD:].mean()
        ma150_20ago = close.iloc[-(MA150_PERIOD + MA150_SLOPE_DAYS):-MA150_SLOPE_DAYS].mean()
        current_close = close.iloc[-1]

        if current_close <= ma150_now:
            continue
        if ma150_now <= ma150_20ago:
            continue

        # --- 52주 신고가 계산 (High 기준) ---
        base_idx = total_len - 1 - MAX_DAYS_SINCE_BREAKOUT - 1
        if base_idx < HIGH_52W_PERIOD:
            continue

        high_52w_prices = high.iloc[base_idx - HIGH_52W_PERIOD:base_idx]
        if high_52w_prices.empty:
            continue

        high_52w = high_52w_prices.max()
        if pd.isna(high_52w) or high_52w <= 0:
            continue

        # --- 돌파일 찾기 (8일 전부터 오늘까지) ---
        breakout_date = None
        days_since = None
        breakout_idx = None
        for days_ago in range(MAX_DAYS_SINCE_BREAKOUT, -1, -1):
            idx = total_len - 1 - days_ago
            if idx < 0:
                continue
            if close.iloc[idx] > high_52w:
                breakout_date = df.index[idx]
                days_since = days_ago
                breakout_idx = idx
                break

        if breakout_date is None:
            continue

        # --- 거래량 확인: 돌파일 거래량 ≥ 20일 평균 × 1.5 ---
        if breakout_idx < 20:
            continue
        avg_vol_20 = volume.iloc[breakout_idx - 20:breakout_idx].mean()
        breakout_vol = volume.iloc[breakout_idx]

        if avg_vol_20 <= 0 or breakout_vol < avg_vol_20 * 1.5:
            continue

        # --- 현재가가 52주 신고가 위에 있어야 함 ---
        if current_close <= high_52w:
            continue

        prev_close = close.iloc[-2] if len(close) >= 2 else current_close
        change_rate = (current_close - prev_close) / prev_close * 100 if prev_close > 0 else 0
        above_high_percent = (current_close - high_52w) / high_52w * 100
        vol_ratio = round(breakout_vol / avg_vol_20, 1) if avg_vol_20 > 0 else 0

        results.append({
            'ticker': ticker,
            'name': name,
            'price': int(current_close),
            'change_rate': round(change_rate, 2),
            'high_52w': int(high_52w),
            'breakout_date': breakout_date.strftime('%Y-%m-%d'),
            'days_since': days_since,
            'above_high_percent': round(above_high_percent, 2),
            'volume_ratio': vol_ratio,
            'market_cap': int(market_cap),
            'updated_at': datetime.now().isoformat()
        })

    # 신고가 대비 상승률 내림차순 정렬
    results.sort(key=lambda x: x['above_high_percent'], reverse=True)
    print(f"[완료] 52주 신고가 돌파: {len(results)}개")
    return results


# ============================================================================
# 8. 업종별 4단계 스크리너 (네이버 증권 업종 기준, 와인스테인 4단계)
# ============================================================================

SECTOR_MA_PERIOD = 150
SECTOR_SLOPE_PERIOD = 20
SECTOR_SLOPE_THRESHOLD = 1.0
SECTOR_MIN_STOCKS = 10  # 최소 종목 수

# 네이버 증권 업종 목록 (업종코드, 업종명)
NAVER_SECTOR_LIST = [
    (278, "반도체"), (267, "IT서비스"), (287, "소프트웨어"),
    (272, "화학"), (270, "자동차부품"), (261, "제약"),
    (282, "전자장비"), (279, "건설"), (268, "식품"),
    (274, "섬유/의류"), (299, "기계"), (283, "전기제품"),
    (269, "디스플레이장비"), (286, "생물공학"), (304, "철강"),
    (285, "방송/엔터"), (263, "게임"), (294, "통신장비"),
    (292, "핸드셋"), (281, "건강관리장비"), (266, "화장품"),
    (284, "우주항공/국방"), (280, "부동산"), (289, "건축자재"),
    (322, "비철금속"), (291, "조선"), (306, "전기장비"),
    (273, "자동차"), (276, "복합기업"), (295, "에너지장비"),
    (311, "포장재"), (290, "교육서비스"), (324, "상업서비스"),
    (277, "창업투자"), (301, "은행"), (321, "증권"),
    (313, "석유/가스"), (316, "건강관리서비스"), (271, "레저장비"),
    (317, "호텔/레저"), (326, "항공화물/물류"), (293, "컴퓨터/주변기기"),
    (298, "가정용기기"), (262, "생명과학도구"), (310, "광고"),
    (300, "양방향미디어"), (308, "인터넷소매"),
    (315, "손해보험"), (312, "가스유틸리티"),
]


def _fetch_sector_stocks(sector_no: int) -> List[str]:
    """네이버 증권에서 업종 구성 종목 코드를 가져옵니다."""
    import requests
    from bs4 import BeautifulSoup

    url = f'https://finance.naver.com/sise/sise_group_detail.naver?type=upjong&no={sector_no}'
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        resp.encoding = 'euc-kr'
        soup = BeautifulSoup(resp.text, 'html.parser')
        table = soup.find('table', {'class': 'type_5'})
        if not table:
            return []
        codes = []
        for link in table.find_all('a', href=True):
            if 'code=' in link['href']:
                code = link['href'].split('code=')[1]
                if code and len(code) == 6:
                    codes.append(code)
        return codes
    except Exception as e:
        logging.warning(f"Failed to fetch sector stocks from {url}: {e}")
        return []


def _calc_sector_index(stock_codes: List[str]) -> Optional[pd.Series]:
    """캐시된 주가 데이터로 업종 등락률 기반 지수를 계산합니다."""
    global _DATA_CACHE
    required_days = SECTOR_MA_PERIOD + SECTOR_SLOPE_PERIOD * 2 + 10

    # 캐시에서 종목 데이터 수집
    close_list = []
    for code in stock_codes:
        if code in _DATA_CACHE and _DATA_CACHE[code] is not None:
            df = _DATA_CACHE[code]
            if len(df) >= required_days:
                close = df['Close'].iloc[-required_days:]
                if close.isna().sum() < 10:
                    close_list.append(close.pct_change())

    if len(close_list) < SECTOR_MIN_STOCKS:
        return None

    # 등락률 평균 → 누적 지수
    returns_df = pd.concat(close_list, axis=1)
    avg_returns = returns_df.mean(axis=1).fillna(0)
    sector_index = (1 + avg_returns).cumprod() * 1000
    return sector_index


def screen_sector_stage() -> List[Dict]:
    """업종별 4단계 판별 스크리너 (네이버 증권 업종 기준, 와인스테인 Stage Analysis)"""
    print("\n[업종별 4단계] 분석 시작...")

    results = []

    # 업종별 종목 목록 병렬 스크래핑
    from concurrent.futures import ThreadPoolExecutor
    sector_stocks = {}

    def fetch_one(item):
        no, name = item
        codes = _fetch_sector_stocks(no)
        return (no, name, codes)

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = list(executor.map(fetch_one, NAVER_SECTOR_LIST))

    for no, name, codes in futures:
        if len(codes) >= SECTOR_MIN_STOCKS:
            sector_stocks[(no, name)] = codes

    print(f"  업종 {len(sector_stocks)}개 로드 (종목 {SECTOR_MIN_STOCKS}개 이상)")

    for (sector_no, sector_name), codes in sector_stocks.items():
        sector_index = _calc_sector_index(codes)
        if sector_index is None or len(sector_index) < SECTOR_MA_PERIOD + SECTOR_SLOPE_PERIOD * 2:
            continue

        close = sector_index.dropna()
        if len(close) < SECTOR_MA_PERIOD + SECTOR_SLOPE_PERIOD * 2:
            continue

        # 150일 이동평균
        ma150_series = close.rolling(window=SECTOR_MA_PERIOD, min_periods=SECTOR_MA_PERIOD).mean()
        current_price = float(close.iloc[-1])
        ma150 = float(ma150_series.iloc[-1])

        if pd.isna(ma150) or ma150 <= 0:
            continue

        # 기울기 계산 (20일간 150MA 변화율)
        current_ma = ma150_series.iloc[-1]
        past_ma = ma150_series.iloc[-SECTOR_SLOPE_PERIOD]
        if pd.isna(current_ma) or pd.isna(past_ma) or past_ma <= 0:
            continue
        slope = (current_ma - past_ma) / past_ma * 100

        # 이전 기울기 (3단계 판별용)
        prev_slope = None
        if len(ma150_series) > SECTOR_SLOPE_PERIOD * 2:
            prev_current = ma150_series.iloc[-SECTOR_SLOPE_PERIOD]
            prev_past = ma150_series.iloc[-SECTOR_SLOPE_PERIOD * 2]
            if not pd.isna(prev_current) and not pd.isna(prev_past) and prev_past > 0:
                prev_slope = (prev_current - prev_past) / prev_past * 100

        # 와인스테인 4단계 판별
        if slope < -SECTOR_SLOPE_THRESHOLD:
            stage, stage_name = 4, "쇠퇴"
        elif slope > SECTOR_SLOPE_THRESHOLD and current_price > ma150:
            stage, stage_name = 2, "상승"
        elif prev_slope is not None and prev_slope > SECTOR_SLOPE_THRESHOLD and abs(slope) <= SECTOR_SLOPE_THRESHOLD and current_price > ma150:
            stage, stage_name = 3, "최정상"
        else:
            stage, stage_name = 1, "기초"


        # 3개월 수익률
        return_3m = 0.0
        if len(close) >= 60:
            past_price = float(close.iloc[-60])
            if past_price > 0:
                return_3m = (current_price - past_price) / past_price * 100

        results.append({
            'sector_name': sector_name,
            'stage': stage,
            'stage_name': stage_name,
            'ma150_slope': round(slope, 2),
            'return_3m': round(return_3m, 2),
            'is_overheated': return_3m > 50,
            'current_price': round(current_price, 2),
            'ma150': round(ma150, 2),
            'stock_count': len(codes),
            'updated_at': datetime.now().isoformat()
        })

    # 2단계 우선, 그 다음 3개월 수익률 순
    results.sort(key=lambda x: (-x['stage'] if x['stage'] == 2 else x['stage'], -x['return_3m']))
    print(f"[완료] 업종별 4단계: {len(results)}개")
    return results


# ============================================================================
# 결과 저장 및 차트 데이터 생성
# ============================================================================

def save_results(results: List[Dict], filename: str, screened_from: int) -> None:
    """결과를 JSON 파일로 저장합니다."""
    now = datetime.now().isoformat()

    output = {
        'meta': {
            'lastUpdated': now,
            'totalCount': len(results),
            'screened_from': screened_from
        },
        'data': results[:300]  # 최대 300개 저장
    }

    filepath = os.path.join(DATA_PATH, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"  저장: {filename} ({len(results)}개)")


def generate_chart_data(all_results: List[List[Dict]]) -> None:
    """스크리너 결과에 포함된 모든 종목의 차트 데이터를 생성합니다."""
    print("\n[차트 데이터 생성] 실행 중...")

    # 모든 결과에서 고유 종목 추출
    tickers = set()
    for results in all_results:
        for r in results:
            tickers.add((r['ticker'], r['name']))

    print(f"  총 {len(tickers)}개 종목 차트 데이터 생성")

    chart_data = {}
    for ticker, name in tickers:
        try:
            df = get_ohlcv(ticker, 200)
            if df is None or len(df) < 10:
                continue

            # NaN 값이 있는 행 제거
            df = df.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'])
            if len(df) < 10:
                continue

            data = []
            for idx, row in df.iterrows():
                data.append({
                    'date': idx.strftime('%Y-%m-%d'),
                    'open': int(row['Open']),
                    'high': int(row['High']),
                    'low': int(row['Low']),
                    'close': int(row['Close']),
                    'volume': int(row['Volume'])
                })
            chart_data[ticker] = data
        except Exception:
            continue

    filepath = os.path.join(DATA_PATH, 'chart_data.json')
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(chart_data, f, ensure_ascii=False)

    print(f"  저장: chart_data.json ({len(chart_data)}개 종목)")


# ============================================================================
# 메인 실행
# ============================================================================

def main():
    print("=" * 70)
    print("퀀트 주식 스크리너 v2.0")
    print(f"실행 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    total_start = time.time()

    # 종목 리스트 조회
    stocks = get_stock_list()
    total_stocks = len(stocks)

    # 샘플링 (테스트용 - 전체 분석하려면 주석처리)
    # stocks = stocks.head(200)  # 테스트용

    print(f"\n분석 대상: {len(stocks)}개 종목 (전체 {total_stocks}개 중)")

    # 데이터 사전 로딩 (한 번만 다운로드하고 모든 스크리너에서 재사용)
    # 52주 신고가 스크리너: 260거래일 필요 → 500 캘린더일 ≈ 345거래일 확보
    preload_all_data(stocks, days=500)

    try:
        # 1. 박스권 스크리너 (퀀트 수준)
        box_range_results = screen_box_range(stocks)
        save_results(box_range_results, 'box_range.json', len(stocks))

        # 2. 박스권 돌파 (거래량 동반)
        box_breakout_results = screen_box_breakout(stocks)
        save_results(box_breakout_results, 'box_breakout.json', len(stocks))

        # 3. 박스권 돌파 (거래량 무관)
        box_breakout_simple_results = screen_box_breakout_simple(stocks)
        save_results(box_breakout_simple_results, 'box_breakout_simple.json', len(stocks))

        # 4. 풀백 스크리너
        pullback_results = screen_pullback(stocks)
        save_results(pullback_results, 'pullback.json', len(stocks))

        # 5. 거래량 폭발 스크리너
        vol_explosion_results = screen_volume_explosion(stocks)
        save_results(vol_explosion_results, 'volume_explosion.json', len(stocks))

        # 6. 거래량 급감 스크리너
        vol_dry_up_results = screen_volume_dry_up(stocks)
        save_results(vol_dry_up_results, 'volume_dry_up.json', len(stocks))

        # 7. 낙폭과대 반등 스크리너
        fallen_rebound_results = screen_fallen_rebound(stocks)
        save_results(fallen_rebound_results, 'fallen_rebound.json', len(stocks))

        # 8. 52주 신고가 돌파 스크리너
        new_high_results = screen_new_high_52w(stocks)
        save_results(new_high_results, 'new_high_52w.json', len(stocks))

        # 8. 업종별 4단계 스크리너
        sector_stage_results = screen_sector_stage()
        save_results(sector_stage_results, 'sector_stage.json', len(NAVER_SECTOR_LIST))

        # 9. 바닥 탈출 스크리너 (pykrx 사용)
        from scripts.screeners.bottom_breakout import screen_bottom_breakout
        bottom_breakout_results = screen_bottom_breakout()
        save_results(bottom_breakout_results, 'bottom_breakout.json', len(stocks))

        # 차트 데이터 생성
        all_results = [
            box_range_results,
            box_breakout_results,
            box_breakout_simple_results,
            pullback_results,
            vol_explosion_results,
            vol_dry_up_results,
            fallen_rebound_results,
            new_high_results,
            bottom_breakout_results
        ]
        generate_chart_data(all_results)

        total_elapsed = time.time() - total_start

        print("\n" + "=" * 70)
        print("스크리너 결과 요약")
        print("=" * 70)
        print(f"  박스권 횡보: {len(box_range_results)}개")
        print(f"  박스권 돌파 (거래량): {len(box_breakout_results)}개")
        print(f"  박스권 돌파 (무관): {len(box_breakout_simple_results)}개")
        print(f"  풀백: {len(pullback_results)}개")
        print(f"  거래량 폭발: {len(vol_explosion_results)}개")
        print(f"  거래량 급감: {len(vol_dry_up_results)}개")
        print(f"  낙폭과대 반등: {len(fallen_rebound_results)}개")
        print(f"  52주 신고가: {len(new_high_results)}개")
        print(f"  업종별 4단계: {len(sector_stage_results)}개")
        print(f"  바닥 탈출: {len(bottom_breakout_results)}개")
        print(f"\n총 실행 시간: {total_elapsed:.1f}초 ({total_elapsed/60:.1f}분)")
        print("=" * 70)

    finally:
        # 메모리 누수 방지를 위한 캐시 명시적 정리
        print("\n[정리] 캐시 메모리 해제 중...")
        clear_cache()
        print("  캐시 정리 완료")


if __name__ == '__main__':
    main()
