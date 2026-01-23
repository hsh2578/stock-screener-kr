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
from datetime import datetime, timedelta
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, List, Dict, Any
from scipy import stats
from tqdm import tqdm

# ============================================================================
# 데이터 캐시 (전역)
# ============================================================================
_DATA_CACHE: Dict[str, pd.DataFrame] = {}

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

# ============================================================================
# 유틸리티 함수
# ============================================================================

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

    recent = df.tail(period + 1).copy()

    high = recent['High'].values
    low = recent['Low'].values
    close = recent['Close'].values

    # True Range 계산 (첫 번째 행 제외)
    tr_list = []
    for i in range(1, len(recent)):
        tr1 = high[i] - low[i]  # 당일 고가 - 저가
        tr2 = abs(high[i] - close[i-1])  # 당일 고가 - 전일 종가
        tr3 = abs(low[i] - close[i-1])  # 당일 저가 - 전일 종가
        tr_list.append(max(tr1, tr2, tr3))

    # ATR = TR의 평균 / 현재 종가 (비율로 변환)
    current_close = close[-1]
    if current_close <= 0:
        return 0.0

    atr = np.mean(tr_list) / current_close
    return atr


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
    pivots = []

    for i in range(n, len(prices) - n):
        # 현재 가격이 앞뒤 n일 중 가장 낮은지 확인
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
    pivots = []

    for i in range(n, len(prices) - n):
        # 현재 가격이 앞뒤 n일 중 가장 높은지 확인
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

    # x: 일수 (0, 1, 2, ..., n-1)
    x = np.arange(len(prices))

    # 선형회귀
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, prices)

    # 일평균 변화율 (%)로 변환
    mean_price = np.mean(prices)
    if mean_price <= 0:
        return 0.0

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
    atr_multiple = range_percent / (atr * 100) if atr > 0 else float('inf')

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

    Args:
        ticker: 종목코드
        days: 조회 기간 (거래일)

    Returns:
        OHLCV DataFrame or None
    """
    global _DATA_CACHE

    # 캐시에 있으면 캐시에서 반환
    if ticker in _DATA_CACHE:
        df = _DATA_CACHE[ticker]
        if df is not None and len(df) > 0:
            return df.tail(days) if len(df) >= days else df
        return None

    # 캐시에 없으면 API 호출
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days + 50)
        df = fdr.DataReader(ticker, start_date.strftime('%Y-%m-%d'))
        _DATA_CACHE[ticker] = df  # 캐시에 저장
        return df.tail(days) if len(df) >= days else df
    except:
        _DATA_CACHE[ticker] = None  # 실패도 캐시 (재시도 방지)
        return None


def _download_single_stock(args: Tuple[str, str]) -> Tuple[str, Optional[pd.DataFrame]]:
    """단일 종목 데이터 다운로드 (병렬 처리용)"""
    ticker, start_str = args
    try:
        df = fdr.DataReader(ticker, start_str)
        if df is not None and len(df) > 0:
            return (ticker, df)
        return (ticker, None)
    except:
        return (ticker, None)


def preload_all_data(stocks: pd.DataFrame, days: int = 200, max_workers: int = 20) -> None:
    """
    모든 종목의 데이터를 병렬로 다운로드하여 캐시합니다.
    ThreadPoolExecutor를 사용하여 동시에 여러 종목을 다운로드합니다.

    Args:
        stocks: 종목 리스트 DataFrame
        days: 조회 기간 (거래일)
        max_workers: 동시 다운로드 스레드 수
    """
    global _DATA_CACHE
    _DATA_CACHE = {}  # 캐시 초기화

    end_date = datetime.now()
    start_date = end_date - timedelta(days=days + 50)
    start_str = start_date.strftime('%Y-%m-%d')

    # 티커 목록 추출
    tickers = []
    for _, row in stocks.iterrows():
        ticker = row.get('Code', row.get('Symbol', ''))
        if ticker:
            tickers.append(ticker)

    print(f"\n[데이터 사전 로딩] {len(tickers)}개 종목 (스레드 {max_workers}개)...")

    success_count = 0
    fail_count = 0

    # 병렬 다운로드
    download_args = [(ticker, start_str) for ticker in tickers]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_download_single_stock, arg): arg[0] for arg in download_args}

        for future in tqdm(as_completed(futures), total=len(futures), desc="데이터 로딩"):
            ticker, df = future.result()
            _DATA_CACHE[ticker] = df
            if df is not None:
                success_count += 1
            else:
                fail_count += 1

    print(f"  완료: 성공 {success_count}개, 실패 {fail_count}개")


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
        - 돌파 조건: 당일 종가 > 저항선 × 1.02
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

        # 이전 박스권 확인 (130~10일 전 데이터, 마지막 60일을 박스로 분석)
        box_period_df = df.iloc[-(BOX_PERIOD*2+10):-10].copy()
        if len(box_period_df) < BOX_PERIOD + 1:
            continue

        # 박스권이었는지 체크 (전체 조건 적용)
        is_box, box_data = is_box_range(box_period_df, BOX_PERIOD)
        if not is_box:
            continue

        box_high = box_data['box_high']
        box_low = box_data['box_low']

        stats['box_history'] += 1

        # 최근 10일 내 돌파 확인
        recent_10d = df.iloc[-10:]
        breakout_day = None
        breakout_idx = None

        for i, (date, row_data) in enumerate(recent_10d.iterrows()):
            # 저항선 +2% 돌파
            if row_data['Close'] > box_high * 1.02:
                breakout_day = date
                breakout_idx = df.index.get_loc(date)
                break

        if breakout_day is None:
            continue

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

        results.append({
            'ticker': ticker,
            'name': name,
            'price': int(current_price),
            'change_rate': round(change_rate, 2),
            'breakout_date': breakout_day.strftime('%Y-%m-%d'),
            'breakout_price': int(box_high),
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
        - 돌파 조건: 종가 > 저항선 × 1.02 (상단 +2% 초과)
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
        if df is None or len(df) < BOX_PERIOD * 2 + 10:
            continue

        # 이전 박스권 구간 (130~10일 전 데이터, 마지막 60일을 박스로 분석)
        box_period_df = df.iloc[-(BOX_PERIOD*2+10):-10].copy()
        if len(box_period_df) < BOX_PERIOD + 1:
            continue

        # 박스권이었는지 체크 (전체 조건 적용)
        is_box, box_data = is_box_range(box_period_df, BOX_PERIOD)
        if not is_box:
            continue

        # 저항선 = 박스 상단
        resistance = box_data['box_high']

        # 최근 10일 내 돌파 확인
        recent_10d = df.iloc[-10:]
        breakout_day = None
        days_since = 0

        for i, (date, row_data) in enumerate(recent_10d.iterrows()):
            if row_data['Close'] > resistance * 1.02:  # +2% 돌파
                breakout_day = date
                days_since = len(recent_10d) - i - 1
                break

        if breakout_day is None:
            continue

        current_price = df['Close'].iloc[-1]
        prev_price = float(df['Close'].iloc[-2])
        change_rate = (current_price - prev_price) / prev_price * 100 if prev_price > 0 else 0

        # 현재가 / 저항선 비율
        current_vs_resistance = (current_price / resistance) * 100

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
            'current_vs_resistance': round(current_vs_resistance, 2),
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

        # 10~3일 전에 돌파 + 거래량 2배 이상이었는지 확인
        breakout_period = df.iloc[-20:-3]
        breakout_day = None
        breakout_idx = None
        breakout_volume = 0

        for i, (date, row_data) in enumerate(breakout_period.iterrows()):
            idx_in_df = df.index.get_loc(date)
            if idx_in_df < 20:
                continue

            # 돌파 조건
            if row_data['Close'] > resistance * 1.02:
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

        # 풀백 조건: 현재가가 저항선 ±5% 이내
        pullback_lower = resistance * 0.95
        pullback_upper = resistance * 1.05

        if not (pullback_lower <= current_price <= pullback_upper):
            continue

        # 거래량 감소 조건: 풀백 구간 평균 < 돌파일 × 50%
        recent_avg_volume = df['Volume'].iloc[-5:].mean()
        if breakout_volume > 0:
            volume_decrease = (1 - recent_avg_volume / breakout_volume) * 100
            if volume_decrease < 50:
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

def screen_volume_dry_up(stocks: pd.DataFrame) -> List[Dict]:
    """
    거래량 급감 스크리너: 20일 평균 3배 이상 거래량 + 8% 이상 장대양봉 후 10거래일 이내 거래량 고갈 종목

    Args:
        stocks: 종목 리스트 DataFrame

    Returns:
        거래량 급감 종목 리스트
    """
    print("\n[거래량 급감 스크리너 시작]")
    results = []

    for idx, row in stocks.iterrows():
        ticker = row.get('Code', row.get('Symbol', ''))
        name = row.get('Name', '')
        market_cap = row.get('MarketCap', 0)

        if not ticker:
            continue

        df = get_ohlcv(ticker, 200)  # 150일선 계산을 위해 200일 필요
        if df is None or len(df) < 30:
            continue

        # 최근 20일 내 급등일 찾기 (20일 평균 거래량 3배 이상 + 8% 이상 장대양봉)
        recent_20d = df.iloc[-20:]
        explosion_day = None
        explosion_change = 0
        explosion_volume_ratio = 0
        days_since_explosion = 0

        for i in range(1, len(recent_20d)):
            prev_close = recent_20d['Close'].iloc[i-1]
            curr_close = recent_20d['Close'].iloc[i]
            curr_volume = recent_20d['Volume'].iloc[i]

            if prev_close <= 0:
                continue

            change = (curr_close - prev_close) / prev_close * 100

            # 해당 일자 기준 직전 20일 평균 거래량 계산
            day_idx = df.index.get_loc(recent_20d.index[i])
            if day_idx < 20:
                continue
            avg_volume_20d = df['Volume'].iloc[day_idx-20:day_idx].mean()

            if avg_volume_20d <= 0:
                continue

            volume_ratio = curr_volume / avg_volume_20d

            # 8% 이상 장대양봉 + 거래량 3배 이상
            if change >= 8 and volume_ratio >= 3:
                explosion_day = recent_20d.index[i]
                explosion_change = change
                explosion_volume_ratio = volume_ratio
                days_since_explosion = len(recent_20d) - i - 1
                break

        if explosion_day is None:
            continue

        # 급등 후 10거래일 이내만 필터
        if days_since_explosion > 10:
            continue

        # 급등일 이후 거래량 감소 확인
        explosion_idx = df.index.get_loc(explosion_day)
        if explosion_idx >= len(df) - 3:
            continue

        explosion_volume = df['Volume'].iloc[explosion_idx]
        recent_avg_volume = df['Volume'].iloc[-3:].mean()

        if explosion_volume <= 0:
            continue

        volume_decrease_rate = (1 - recent_avg_volume / explosion_volume) * 100

        if volume_decrease_rate < 50:
            continue

        current_price = int(df['Close'].iloc[-1])

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
            'explosion_date': explosion_day.strftime('%Y-%m-%d'),
            'explosion_change_rate': round(explosion_change, 1),
            'explosion_volume_ratio': round(explosion_volume_ratio, 1),
            'days_since_explosion': days_since_explosion,
            'volume_decrease_rate': int(volume_decrease_rate),
            'volume_rank': 0,
            'ma150': ma150,
            'above_ma150': above_ma150,
            'market_cap': int(market_cap),
            'updated_at': datetime.now().isoformat()
        })

    # 거래량 감소율 높은 순 정렬
    results.sort(key=lambda x: x['volume_decrease_rate'], reverse=True)

    for i, r in enumerate(results):
        r['volume_rank'] = i + 1

    print(f"[완료] 거래량 급감 종목: {len(results)}개")

    return results


# ============================================================================
# 7. 52주 신고가 돌파 스크리너
# ============================================================================

HIGH_52W_PERIOD = 250  # 52주 ≈ 250거래일
MAX_DAYS_SINCE_BREAKOUT = 8  # 돌파 후 최대 거래일

def screen_new_high_52w(stocks: pd.DataFrame) -> List[Dict]:
    """52주 신고가를 돌파한 종목을 발굴합니다."""
    print("\n[52주 신고가 돌파] 분석 시작...")

    results = []
    required_days = HIGH_52W_PERIOD + MAX_DAYS_SINCE_BREAKOUT + 2

    for _, row in stocks.iterrows():
        ticker = row.get('Code', row.get('Symbol', ''))
        name = row.get('Name', '')
        market_cap = row.get('Marcap', 0)
        if isinstance(market_cap, (int, float)):
            market_cap_억 = market_cap / 1e8
        else:
            market_cap_억 = 0

        if market_cap_억 < MIN_MARKET_CAP:
            continue

        df = get_ohlcv(ticker, required_days + 50)
        if df is None or len(df) < required_days:
            continue

        close = df['Close']
        if close.isna().any():
            close = close.dropna()
            if len(close) < required_days:
                continue

        total_len = len(df)

        # 9일 전 기준 52주 신고가 계산
        base_idx = total_len - 1 - MAX_DAYS_SINCE_BREAKOUT - 1
        if base_idx < HIGH_52W_PERIOD:
            continue

        high_52w_prices = close.iloc[base_idx - HIGH_52W_PERIOD:base_idx]
        if high_52w_prices.empty:
            continue

        high_52w = high_52w_prices.max()
        if pd.isna(high_52w) or high_52w <= 0:
            continue

        # 8일 전부터 오늘까지 돌파일 찾기
        breakout_date = None
        days_since = None
        for days_ago in range(MAX_DAYS_SINCE_BREAKOUT, -1, -1):
            idx = total_len - 1 - days_ago
            if idx < 0:
                continue
            c = close.iloc[idx]
            if c > high_52w:
                breakout_date = df.index[idx]
                days_since = days_ago
                break

        if breakout_date is None:
            continue

        # 현재가가 52주 신고가 위에 있어야 함
        current_close = close.iloc[-1]
        if current_close <= high_52w:
            continue

        prev_close = close.iloc[-2] if len(close) >= 2 else current_close
        change_rate = (current_close - prev_close) / prev_close * 100 if prev_close > 0 else 0
        above_high_percent = (current_close - high_52w) / high_52w * 100

        results.append({
            'ticker': ticker,
            'name': name,
            'price': int(current_close),
            'change_rate': round(change_rate, 2),
            'high_52w': int(high_52w),
            'breakout_date': breakout_date.strftime('%Y-%m-%d'),
            'days_since': days_since,
            'above_high_percent': round(above_high_percent, 2),
            'market_cap': int(market_cap_억),
            'updated_at': datetime.now().isoformat()
        })

    # 신고가 대비 상승률 내림차순 정렬
    results.sort(key=lambda x: x['above_high_percent'], reverse=True)
    print(f"[완료] 52주 신고가 돌파: {len(results)}개")
    return results


# ============================================================================
# 8. 업종별 4단계 스크리너
# ============================================================================

SECTOR_MA_PERIOD = 150
SECTOR_SLOPE_PERIOD = 20
SECTOR_SLOPE_THRESHOLD = 2.0

# 업종 ETF 목록 (FinanceDataReader 사용)
SECTOR_ETF_LIST = [
    ("KS11", "KOSPI"),
    ("KQ11", "KOSDAQ"),
    ("091160", "KODEX IT"),
    ("091170", "KODEX 반도체"),
    ("091180", "KODEX 자동차"),
    ("117700", "KODEX 건설"),
    ("117460", "KODEX 에너지화학"),
    ("140710", "KODEX 헬스케어"),
    ("091220", "KODEX 철강"),
    ("102780", "KODEX 조선"),
    ("140700", "KODEX 보험"),
    ("102970", "KODEX 증권"),
    ("091230", "KODEX 은행"),
    ("266360", "KODEX 2차전지"),
    ("385510", "KODEX AI반도체핵심장비"),
    ("396500", "KODEX 미디어엔터"),
    ("381170", "KODEX 기계장비"),
]


def screen_sector_stage() -> List[Dict]:
    """업종별 4단계 판별 스크리너를 실행합니다. (섹터 ETF 기반)"""
    print("\n[업종별 4단계] 분석 시작...")

    results = []
    start_date = (datetime.now() - timedelta(days=450)).strftime("%Y-%m-%d")

    for ticker, sector_name in SECTOR_ETF_LIST:
        try:
            df = fdr.DataReader(ticker, start_date)
            if df is None or df.empty:
                print(f"  [업종] {sector_name}({ticker}): 데이터 없음")
                continue

            close = df['Close'].dropna()
            if len(close) < SECTOR_MA_PERIOD + SECTOR_SLOPE_PERIOD:
                print(f"  [업종] {sector_name}({ticker}): 데이터 부족 ({len(close)}일)")
                continue

            # 150일 이동평균
            ma150_series = close.rolling(window=SECTOR_MA_PERIOD, min_periods=SECTOR_MA_PERIOD).mean()
            current_price = float(close.iloc[-1])
            ma150 = float(ma150_series.iloc[-1])

            if pd.isna(ma150) or ma150 <= 0:
                continue

            # 기울기 계산
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

            # 단계 판별
            if slope < -SECTOR_SLOPE_THRESHOLD:
                stage, stage_name = 4, "쇠퇴"
            elif slope > SECTOR_SLOPE_THRESHOLD and current_price > ma150:
                stage, stage_name = 2, "상승"
            elif prev_slope is not None and prev_slope > SECTOR_SLOPE_THRESHOLD and abs(slope) <= SECTOR_SLOPE_THRESHOLD:
                stage, stage_name = 3, "최정상"
            else:
                stage, stage_name = 1, "기초"

            # 3개월 수익률
            return_3m = 0.0
            if len(close) >= 60:
                past_price = float(close.iloc[-60])
                if past_price > 0:
                    return_3m = (current_price - past_price) / past_price * 100

            # 6개월 과열 여부
            is_overheated = False
            if len(close) >= 120:
                monthly_positive = []
                for i in range(6):
                    s_idx = -(i + 1) * 20 - 1
                    e_idx = -i * 20 - 1 if i > 0 else -1
                    if abs(s_idx) <= len(close):
                        s_p = float(close.iloc[s_idx])
                        e_p = float(close.iloc[e_idx]) if e_idx != -1 else float(close.iloc[-1])
                        if s_p > 0:
                            monthly_positive.append((e_p - s_p) / s_p > 0)
                is_overheated = len(monthly_positive) == 6 and all(monthly_positive)

            results.append({
                'sector_name': sector_name,
                'stage': stage,
                'stage_name': stage_name,
                'ma150_slope': round(slope, 2),
                'return_3m': round(return_3m, 2),
                'is_overheated': is_overheated,
                'current_price': round(current_price, 2),
                'ma150': round(ma150, 2),
                'updated_at': datetime.now().isoformat()
            })
        except Exception as e:
            print(f"  [업종] {sector_name}({ticker}) 오류: {e}")
            continue

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

    # 7. 52주 신고가 돌파 스크리너
    new_high_results = screen_new_high_52w(stocks)
    save_results(new_high_results, 'new_high_52w.json', len(stocks))

    # 8. 업종별 4단계 스크리너
    sector_stage_results = screen_sector_stage()
    save_results(sector_stage_results, 'sector_stage.json', len(SECTOR_ETF_LIST))

    # 차트 데이터 생성
    all_results = [
        box_range_results,
        box_breakout_results,
        box_breakout_simple_results,
        pullback_results,
        vol_explosion_results,
        vol_dry_up_results,
        new_high_results
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
    print(f"  52주 신고가: {len(new_high_results)}개")
    print(f"  업종별 4단계: {len(sector_stage_results)}개")
    print(f"\n총 실행 시간: {total_elapsed:.1f}초 ({total_elapsed/60:.1f}분)")
    print("=" * 70)


if __name__ == '__main__':
    main()
