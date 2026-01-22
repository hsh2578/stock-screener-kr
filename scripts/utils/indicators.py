"""
기술적 지표 계산 유틸리티 모듈

모든 계산은 종가(close) 기준으로 수행됩니다.
pykrx DataFrame의 컬럼명 규칙:
    - 종가: '종가'
    - 고가: '고가'
    - 저가: '저가'
    - 시가: '시가'
    - 거래량: '거래량'
"""
import numpy as np
import pandas as pd


def calculate_sma(data: pd.DataFrame, period: int) -> pd.Series:
    """
    단순이동평균선(SMA)을 계산합니다.

    Args:
        data: OHLCV 데이터프레임 (pykrx 형식, '종가' 컬럼 필수)
        period: 이동평균 기간 (일반적으로 5, 20, 40, 60, 120, 150, 200일)

    Returns:
        이동평균값 Series (인덱스는 원본과 동일)

    Raises:
        KeyError: '종가' 컬럼이 없는 경우
        ValueError: period가 0 이하인 경우

    Example:
        >>> df = get_stock_data("005930", days=250)
        >>> sma_20 = calculate_sma(df, 20)
        >>> sma_60 = calculate_sma(df, 60)
    """
    if period <= 0:
        raise ValueError(f"period는 양수여야 합니다: {period}")

    if "종가" not in data.columns:
        raise KeyError("DataFrame에 '종가' 컬럼이 필요합니다")

    return data["종가"].rolling(window=period, min_periods=period).mean()


def calculate_ema(data: pd.DataFrame, period: int) -> pd.Series:
    """
    지수이동평균선(EMA)을 계산합니다.

    최근 가격에 더 높은 가중치를 부여하여 추세 변화에 민감합니다.

    Args:
        data: OHLCV 데이터프레임 (pykrx 형식)
        period: EMA 기간

    Returns:
        EMA 값 Series

    Example:
        >>> ema_12 = calculate_ema(df, 12)
        >>> ema_26 = calculate_ema(df, 26)
    """
    if period <= 0:
        raise ValueError(f"period는 양수여야 합니다: {period}")

    if "종가" not in data.columns:
        raise KeyError("DataFrame에 '종가' 컬럼이 필요합니다")

    return data["종가"].ewm(span=period, adjust=False).mean()


def calculate_atr(data: pd.DataFrame, period: int = 20) -> pd.Series:
    """
    ATR(Average True Range)을 계산합니다.

    ATR은 변동성 지표로, 손절가 설정이나 포지션 사이징에 활용됩니다.
    True Range = max(고가-저가, |고가-전일종가|, |저가-전일종가|)

    Args:
        data: OHLCV 데이터프레임 (pykrx 형식, '고가', '저가', '종가' 컬럼 필수)
        period: ATR 계산 기간 (기본 20일)

    Returns:
        ATR 값 Series

    Raises:
        KeyError: 필수 컬럼이 없는 경우

    Example:
        >>> atr = calculate_atr(df, 20)
        >>> stop_loss = current_price - (2 * atr.iloc[-1])
    """
    required_columns = ["고가", "저가", "종가"]
    missing = [col for col in required_columns if col not in data.columns]
    if missing:
        raise KeyError(f"DataFrame에 필수 컬럼이 없습니다: {missing}")

    if period <= 0:
        raise ValueError(f"period는 양수여야 합니다: {period}")

    high = data["고가"]
    low = data["저가"]
    close = data["종가"]
    prev_close = close.shift(1)

    # True Range 계산
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # ATR = True Range의 이동평균
    atr = true_range.rolling(window=period, min_periods=period).mean()

    return atr


def calculate_volume_avg(data: pd.DataFrame, period: int = 20) -> pd.Series:
    """
    평균 거래량을 계산합니다.

    Args:
        data: OHLCV 데이터프레임 (pykrx 형식, '거래량' 컬럼 필수)
        period: 평균 계산 기간 (기본 20일)

    Returns:
        평균 거래량 Series

    Example:
        >>> avg_vol = calculate_volume_avg(df, 20)
        >>> volume_ratio = df["거래량"].iloc[-1] / avg_vol.iloc[-1]
    """
    if "거래량" not in data.columns:
        raise KeyError("DataFrame에 '거래량' 컬럼이 필요합니다")

    if period <= 0:
        raise ValueError(f"period는 양수여야 합니다: {period}")

    return data["거래량"].rolling(window=period, min_periods=period).mean()


def calculate_rsi(data: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    RSI(Relative Strength Index)를 계산합니다.

    RSI는 과매수(70 이상)/과매도(30 이하) 판단에 사용됩니다.

    Args:
        data: OHLCV 데이터프레임 (pykrx 형식)
        period: RSI 계산 기간 (기본 14일)

    Returns:
        RSI 값 Series (0~100 범위)

    Example:
        >>> rsi = calculate_rsi(df, 14)
        >>> is_oversold = rsi.iloc[-1] < 30
    """
    if "종가" not in data.columns:
        raise KeyError("DataFrame에 '종가' 컬럼이 필요합니다")

    close = data["종가"]
    delta = close.diff()

    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)

    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    # 0으로 나누기 방지
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    return rsi


def calculate_bollinger_bands(
    data: pd.DataFrame,
    period: int = 20,
    std_multiplier: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    볼린저 밴드를 계산합니다.

    Args:
        data: OHLCV 데이터프레임 (pykrx 형식)
        period: 이동평균 기간 (기본 20일)
        std_multiplier: 표준편차 배수 (기본 2.0)

    Returns:
        (상단밴드, 중간밴드, 하단밴드) 튜플

    Example:
        >>> upper, middle, lower = calculate_bollinger_bands(df)
        >>> bandwidth = (upper - lower) / middle * 100
    """
    if "종가" not in data.columns:
        raise KeyError("DataFrame에 '종가' 컬럼이 필요합니다")

    close = data["종가"]
    middle = close.rolling(window=period, min_periods=period).mean()
    std = close.rolling(window=period, min_periods=period).std()

    upper = middle + (std * std_multiplier)
    lower = middle - (std * std_multiplier)

    return upper, middle, lower


def calculate_macd(
    data: pd.DataFrame,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    MACD(Moving Average Convergence Divergence)를 계산합니다.

    Args:
        data: OHLCV 데이터프레임 (pykrx 형식)
        fast_period: 빠른 EMA 기간 (기본 12일)
        slow_period: 느린 EMA 기간 (기본 26일)
        signal_period: 시그널선 기간 (기본 9일)

    Returns:
        (MACD선, 시그널선, 히스토그램) 튜플

    Example:
        >>> macd, signal, hist = calculate_macd(df)
        >>> golden_cross = (macd.iloc[-2] < signal.iloc[-2]) and (macd.iloc[-1] > signal.iloc[-1])
    """
    if "종가" not in data.columns:
        raise KeyError("DataFrame에 '종가' 컬럼이 필요합니다")

    close = data["종가"]

    fast_ema = close.ewm(span=fast_period, adjust=False).mean()
    slow_ema = close.ewm(span=slow_period, adjust=False).mean()

    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram


def calculate_returns(data: pd.DataFrame, periods: list[int]) -> dict[int, float | None]:
    """
    기간별 수익률을 계산합니다.

    Args:
        data: OHLCV 데이터프레임 (pykrx 형식)
        periods: 계산할 기간 리스트 (예: [5, 20, 60, 120])

    Returns:
        {기간: 수익률(%)} 딕셔너리, 데이터 부족 시 None

    Example:
        >>> returns = calculate_returns(df, [5, 20, 60, 120])
        >>> print(f"20일 수익률: {returns[20]:.2f}%")
    """
    if "종가" not in data.columns:
        raise KeyError("DataFrame에 '종가' 컬럼이 필요합니다")

    close = data["종가"]
    current_price = close.iloc[-1]
    results: dict[int, float | None] = {}

    for period in periods:
        if len(close) > period:
            past_price = close.iloc[-(period + 1)]
            if past_price > 0:
                results[period] = ((current_price - past_price) / past_price) * 100
            else:
                results[period] = None
        else:
            results[period] = None

    return results


def calculate_volatility(data: pd.DataFrame, period: int = 20) -> float | None:
    """
    연환산 변동성을 계산합니다.

    Args:
        data: OHLCV 데이터프레임 (pykrx 형식)
        period: 변동성 계산 기간 (기본 20일)

    Returns:
        연환산 변동성(%), 데이터 부족 시 None

    Example:
        >>> volatility = calculate_volatility(df, 20)
        >>> print(f"20일 변동성: {volatility:.2f}%")
    """
    if "종가" not in data.columns:
        raise KeyError("DataFrame에 '종가' 컬럼이 필요합니다")

    close = data["종가"]

    if len(close) < period + 1:
        return None

    returns = close.pct_change().dropna().tail(period)
    return float(returns.std() * np.sqrt(252) * 100)


def calculate_52week_high_low(data: pd.DataFrame) -> tuple[float, float]:
    """
    52주(약 250거래일) 최고가/최저가를 계산합니다.

    Args:
        data: OHLCV 데이터프레임 (pykrx 형식, 최소 250일 데이터 권장)

    Returns:
        (52주 최고가, 52주 최저가) 튜플

    Example:
        >>> high_52w, low_52w = calculate_52week_high_low(df)
        >>> from_high = (current_price - high_52w) / high_52w * 100
    """
    required = ["고가", "저가"]
    missing = [col for col in required if col not in data.columns]
    if missing:
        raise KeyError(f"DataFrame에 필수 컬럼이 없습니다: {missing}")

    # 최근 250거래일 (약 52주)
    recent_data = data.tail(250)

    high_52w = float(recent_data["고가"].max())
    low_52w = float(recent_data["저가"].min())

    return high_52w, low_52w
