"""
바닥 탈출 스크리너 (Bottom Breakout Screener)

충분히 하락한 후 바닥을 다지고 상승 전환하는 종목 포착
150일선 기준으로 바닥 탈출 신호 포착

[1단계: 필수 조건] 3개 모두 충족
    1. 충분한 낙폭: 52주 고점 대비 -30% ~ -75%
    2. 신저가 미갱신: 최근 10일간 52주 신저가 없음
    3. 최소 거래대금: 5일 평균 거래대금 1억 이상

[2단계: 탈출 신호] 6개 항목, 12점 만점, 5점 이상
    4. 150일선 근접: 종가가 150일선 대비 -5%~15% 범위 (2점)
    5. 150일선 돌파: 아래→위 돌파 + 거래량 1.5배 + 양봉 (3점)
    6. 저점 상승: 최근 20일 저점 > 이전 20일 저점 (2점)
    7. 기울기 상승: 150일선 20일 기울기 > 0 (2점)
    8. 거래량 급증: 5일 평균 > 20일 평균 × 1.5 (2점)
    9. MACD 골든크로스: MACD가 Signal 상향 돌파 (1점)

[필터]
    150일선 대비 -5%~15% 범위 외 종목 제외

[등급]
    A급: 9점 이상 (강력 신호)
    B급: 5~8점 (유력 신호)

[신호 유지]
    최근 10거래일 내 5점 이상 달성한 종목 모두 표시
"""

import pandas as pd
import numpy as np
import FinanceDataReader as fdr
from datetime import datetime, timedelta
import time
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, asdict


# ============================================================================
# 상수 정의
# ============================================================================
MIN_MARKET_CAP = 1000  # 최소 시가총액 (억원)
DATA_DAYS = 400  # 데이터 조회 기간 (캘린더일 기준, 거래일 252일 확보 위해)
HIGH_52W_PERIOD = 252  # 52주 거래일
SLOPE_PERIOD = 20  # 기울기 계산 기간 (20일)
MA150_PERIOD = 150  # 150일 이동평균
SIGNAL_LOOKBACK = 10  # 신호 탐색 기간 (10거래일)
MIN_SCORE = 6  # 최소 점수
MAX_RETRIES = 3  # API 재시도 횟수

# 스팩/리츠/ETF 필터 키워드
EXCLUDE_KEYWORDS = ['스팩', 'SPAC', '리츠', 'REIT', 'ETF', 'ETN', '인버스', '레버리지']


@dataclass
class BottomBreakoutResult:
    """바닥 탈출 스크리너 결과"""
    ticker: str
    name: str
    grade: str  # A 또는 B
    score: int
    signal_date: str
    days_since_signal: int  # 경과일
    current_price: int
    change_since_signal: float  # 신호일 대비 등락률
    drop_from_high: float  # 52주 고점 대비 낙폭
    ma150: int
    above_ma150: bool
    market_cap: int
    # 점수 상세
    score_near_ma150: int
    score_ma_breakout: int  # 150일선 돌파
    score_higher_low: int  # 저점 상승
    score_slope: int  # 기울기 상승
    score_volume: int
    score_macd: int
    ma150_gap: float  # 150일선 대비 이격도
    updated_at: str

    def to_dict(self) -> dict:
        return asdict(self)


def get_stock_data_with_retry(ticker: str, days: int = DATA_DAYS) -> Optional[pd.DataFrame]:
    """FinanceDataReader로 주가 데이터 조회 (재시도 로직 포함)"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days + 100)  # 충분한 여유 확보

    for attempt in range(MAX_RETRIES):
        try:
            df = fdr.DataReader(ticker, start_date.strftime('%Y-%m-%d'))
            if df is not None and len(df) > 0:
                df = df.reset_index()
                # 컬럼명 통일
                df.columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'change']
                return df.tail(days) if len(df) >= days else df
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(0.5 * (2 ** attempt))  # 지수 백오프
                continue
    return None


def calculate_moving_average(prices: pd.Series, period: int) -> pd.Series:
    """이동평균 계산"""
    return prices.rolling(window=period, min_periods=period).mean()


def check_slope_positive(ma_series: pd.Series, days: int = 20) -> bool:
    """
    기울기 상승 체크: 20일 전 대비 현재 MA가 높음 (기울기 > 0)
    """
    if len(ma_series) < days + 1:
        return False

    ma_now = ma_series.iloc[-1]
    ma_days_ago = ma_series.iloc[-days-1]

    if pd.isna(ma_now) or pd.isna(ma_days_ago):
        return False

    return ma_now > ma_days_ago


def check_higher_low(low: pd.Series) -> bool:
    """
    저점 상승 체크: 최근 20일 저점 > 이전 20일 저점
    """
    if len(low) < 40:
        return False

    recent_20d_low = low.iloc[-20:].min()
    prev_20d_low = low.iloc[-40:-20].min()

    return recent_20d_low > prev_20d_low


def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series]:
    """MACD 계산"""
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line


def check_macd_golden_cross(macd: pd.Series, signal: pd.Series) -> bool:
    """MACD 골든크로스 체크: 최근 5일 내 MACD가 Signal을 상향 돌파"""
    if len(macd) < 6 or len(signal) < 6:
        return False

    for i in range(1, 6):  # 최근 5일 체크
        idx = -i
        prev_idx = idx - 1

        if pd.isna(macd.iloc[idx]) or pd.isna(signal.iloc[idx]):
            continue
        if pd.isna(macd.iloc[prev_idx]) or pd.isna(signal.iloc[prev_idx]):
            continue

        # 이전에는 MACD < Signal, 현재는 MACD > Signal
        if macd.iloc[prev_idx] <= signal.iloc[prev_idx] and macd.iloc[idx] > signal.iloc[idx]:
            return True

    return False


def is_excluded_stock(name: str) -> bool:
    """스팩/리츠/ETF 등 제외 종목 체크"""
    return any(keyword in name for keyword in EXCLUDE_KEYWORDS)


def check_stage1(df: pd.DataFrame) -> Tuple[bool, float]:
    """
    1단계: 낙폭과대 + 바닥 확인 + 유동성 필터

    Returns:
        (통과 여부, 낙폭률)
    """
    if len(df) < HIGH_52W_PERIOD:
        return False, 0.0

    close = df['close']
    high = df['high']
    low = df['low']
    # volume 대신 거래대금 계산 (종가 × 거래량)
    trading_value = df['close'] * df['volume']

    # 조건 1: 52주 고점 대비 -30% ~ -75% 하락
    high_52w = high.iloc[-HIGH_52W_PERIOD:].max()
    if high_52w == 0 or pd.isna(high_52w):
        return False, 0.0

    current_price = close.iloc[-1]
    drop_rate = (current_price - high_52w) / high_52w * 100

    if drop_rate > -30 or drop_rate < -75:
        return False, drop_rate

    # 조건 2: 최근 10일간 52주 신저가 없음 (오늘 제외하고 52주 저점 계산)
    if len(low) < HIGH_52W_PERIOD + 1:
        return False, drop_rate

    low_52w = low.iloc[-HIGH_52W_PERIOD-1:-1].min()  # 오늘 제외
    recent_10d_low = low.iloc[-10:].min()  # 최근 10일 (오늘 포함)

    if recent_10d_low <= low_52w:
        return False, drop_rate

    # 조건 3: 5일 평균 거래대금 1억 이상
    avg_value_5d = trading_value.iloc[-5:].mean()
    if avg_value_5d < 100000000:  # 1억
        return False, drop_rate

    return True, drop_rate


def calculate_score(df: pd.DataFrame, days_ago: int = 0) -> Tuple[int, Dict[str, int], float]:
    """
    특정 기준일에 대한 탈출 신호 점수 계산

    Args:
        df: 전체 데이터프레임
        days_ago: 0=오늘, 1=1일전, ... 9=9일전

    Returns:
        (총점수, 점수상세 딕셔너리, 150일선 이격도)
    """
    score_detail = {
        'near_ma150': 0,
        'ma_breakout': 0,
        'higher_low': 0,
        'slope': 0,
        'volume': 0,
        'macd': 0
    }

    if len(df) < HIGH_52W_PERIOD + 10:
        return 0, score_detail, 0.0

    # 기준일까지의 데이터만 사용
    if days_ago > 0:
        data = df.iloc[:-days_ago].copy()
    else:
        data = df.copy()

    if len(data) < HIGH_52W_PERIOD:
        return 0, score_detail, 0.0

    close = data['close']
    high = data['high']
    low = data['low']
    volume = data['volume']
    open_price = data['open']

    # 이동평균 계산
    ma150 = calculate_moving_average(close, MA150_PERIOD)

    total_score = 0
    ma150_gap = 0.0

    # 150일선 이격도 계산
    if len(ma150) >= MA150_PERIOD and not pd.isna(ma150.iloc[-1]) and ma150.iloc[-1] > 0:
        ma150_gap = (close.iloc[-1] - ma150.iloc[-1]) / ma150.iloc[-1] * 100

    # 조건 4: 150일선 근접 - 종가가 150일선 대비 -5%~15% 범위 (2점)
    if len(ma150) >= MA150_PERIOD and not pd.isna(ma150.iloc[-1]):
        if -5 <= ma150_gap <= 15:
            score_detail['near_ma150'] = 2
            total_score += 2

    # 조건 5: 150일선 돌파 - 아래→위 돌파 + 거래량 1.5배 + 양봉 (3점)
    if len(data) >= MA150_PERIOD + 1:
        vol_20d_avg = volume.iloc[-20:].mean() if len(volume) >= 20 else 0

        # 최근 5일 내 돌파 체크
        for i in range(min(5, len(data) - MA150_PERIOD)):
            idx = -(i + 1)
            prev_idx = idx - 1

            if abs(prev_idx) >= len(data):
                continue

            curr_close = close.iloc[idx]
            curr_open = open_price.iloc[idx]
            prev_close = close.iloc[prev_idx]
            curr_vol = volume.iloc[idx]

            # 양봉 체크 (종가 > 시가)
            is_bullish = curr_close > curr_open

            if not is_bullish:
                continue

            # 150일선 돌파 체크 (아래→위)
            if len(ma150) >= MA150_PERIOD and not pd.isna(ma150.iloc[idx]) and not pd.isna(ma150.iloc[prev_idx]):
                ma150_curr = ma150.iloc[idx]
                ma150_prev = ma150.iloc[prev_idx]
                if prev_close <= ma150_prev and curr_close > ma150_curr and curr_vol > vol_20d_avg * 1.5:
                    score_detail['ma_breakout'] = 3
                    total_score += 3
                    break

    # 조건 6: 저점 상승 - 최근 20일 저점 > 이전 20일 저점 (2점)
    if len(low) >= 40:
        if check_higher_low(low):
            score_detail['higher_low'] = 2
            total_score += 2

    # 조건 7: 기울기 상승 - 150일선 20일 기울기 > 0 (2점)
    if len(ma150) >= MA150_PERIOD + SLOPE_PERIOD:
        if check_slope_positive(ma150, days=SLOPE_PERIOD):
            score_detail['slope'] = 2
            total_score += 2

    # 조건 8: 거래량 급증 (2점) - 5일 평균 > 20일 평균 × 1.5
    if len(volume) >= 20:
        vol_5d = volume.iloc[-5:].mean()
        vol_20d = volume.iloc[-20:].mean()
        if vol_20d > 0 and vol_5d > vol_20d * 1.5:
            score_detail['volume'] = 2
            total_score += 2

    # 조건 9: MACD 골든크로스 (1점)
    if len(close) >= 35:  # MACD 계산에 충분한 데이터
        macd, signal = calculate_macd(close)
        if check_macd_golden_cross(macd, signal):
            score_detail['macd'] = 1
            total_score += 1

    return total_score, score_detail, ma150_gap


def screen_bottom_breakout(stocks: pd.DataFrame = None) -> List[Dict]:
    """
    바닥 탈출 스크리너 메인 함수

    Args:
        stocks: 종목 리스트 DataFrame (None이면 내부에서 조회)

    Returns:
        스크리닝 결과 리스트
    """
    print("\n[바닥 탈출 스크리너] 분석 시작...")

    # 종목 리스트 조회
    if stocks is None:
        print("  종목 리스트 조회 중...")
        kospi = fdr.StockListing('KOSPI')
        kospi['Market'] = 'KOSPI'
        kosdaq = fdr.StockListing('KOSDAQ')
        kosdaq['Market'] = 'KOSDAQ'
        stocks = pd.concat([kospi, kosdaq], ignore_index=True)

        # 시가총액 처리
        if 'Marcap' in stocks.columns:
            stocks['MarketCap'] = stocks['Marcap'] / 100000000  # 원 -> 억원
        elif 'MarketCap' in stocks.columns:
            stocks['MarketCap'] = stocks['MarketCap'] / 100000000
        else:
            stocks['MarketCap'] = MIN_MARKET_CAP + 1

        # 시가총액 필터링
        stocks = stocks[stocks['MarketCap'] >= MIN_MARKET_CAP].copy()
        print(f"  시총 {MIN_MARKET_CAP}억 이상: {len(stocks)}개 종목")

    results = []
    stats = {
        'total': len(stocks),
        'market_cap': 0,
        'excluded': 0,
        'stage1': 0,
        'stage2': 0
    }

    print(f"  총 {len(stocks)}개 종목 스캔...")

    for idx, row in stocks.iterrows():
        ticker = row.get('Code', row.get('Symbol', ''))
        name = row.get('Name', '')
        market_cap = row.get('MarketCap', 0)

        if not ticker:
            continue

        # 시가총액 필터
        if market_cap < MIN_MARKET_CAP:
            continue
        stats['market_cap'] += 1

        # 스팩/리츠 필터
        if is_excluded_stock(name):
            stats['excluded'] += 1
            continue

        # 진행률 표시
        if stats['market_cap'] % 200 == 0:
            print(f"  진행 중... {stats['market_cap']}/{stats['total']}")

        try:
            # 데이터 로드
            df = get_stock_data_with_retry(ticker, DATA_DAYS)
            if df is None or len(df) < HIGH_52W_PERIOD:
                continue

            # 거래량 0인 종목 제외 (거래정지)
            if df['volume'].iloc[-5:].sum() == 0:
                continue

            # 1단계: 낙폭과대 + 바닥 확인 + 유동성 필터
            passed_stage1, drop_rate = check_stage1(df)
            if not passed_stage1:
                continue
            stats['stage1'] += 1

            # 2단계: 최근 10거래일 내 신호 체크
            for days_ago in range(SIGNAL_LOOKBACK):
                score, score_detail, ma150_gap = calculate_score(df, days_ago)

                # 150일선 대비 -5%~15% 범위 외 종목 제외
                if ma150_gap < -5 or ma150_gap > 15:
                    continue

                if score >= MIN_SCORE:
                    stats['stage2'] += 1

                    # 신호 발생일 정보
                    signal_idx = -(days_ago + 1) if days_ago > 0 else -1
                    signal_date = df['date'].iloc[signal_idx]
                    signal_close = df['close'].iloc[signal_idx]
                    current_price = int(df['close'].iloc[-1])

                    # 신호일 대비 등락률
                    change_since = (current_price - signal_close) / signal_close * 100

                    # 52주 고점 대비 낙폭률
                    high_52w = df['high'].iloc[-HIGH_52W_PERIOD:].max()
                    drop_from_high = (current_price - high_52w) / high_52w * 100

                    # 150일선
                    ma150 = calculate_moving_average(df['close'], MA150_PERIOD)
                    ma150_val = int(ma150.iloc[-1]) if not pd.isna(ma150.iloc[-1]) else 0
                    above_ma150 = current_price > ma150_val if ma150_val > 0 else False

                    # 등급 판정
                    grade = 'A' if score >= 9 else 'B'

                    # 날짜 포맷
                    if hasattr(signal_date, 'strftime'):
                        signal_date_str = signal_date.strftime('%Y-%m-%d')
                    elif isinstance(signal_date, pd.Timestamp):
                        signal_date_str = signal_date.strftime('%Y-%m-%d')
                    else:
                        signal_date_str = str(signal_date)[:10]

                    result = BottomBreakoutResult(
                        ticker=ticker,
                        name=name,
                        grade=grade,
                        score=score,
                        signal_date=signal_date_str,
                        days_since_signal=days_ago,
                        current_price=current_price,
                        change_since_signal=round(change_since, 2),
                        drop_from_high=round(drop_from_high, 2),
                        ma150=ma150_val,
                        above_ma150=above_ma150,
                        market_cap=int(market_cap),
                        score_near_ma150=score_detail['near_ma150'],
                        score_ma_breakout=score_detail['ma_breakout'],
                        score_higher_low=score_detail['higher_low'],
                        score_slope=score_detail['slope'],
                        score_volume=score_detail['volume'],
                        score_macd=score_detail['macd'],
                        ma150_gap=round(ma150_gap, 1),
                        updated_at=datetime.now().isoformat()
                    )
                    results.append(result.to_dict())
                    break  # 첫 번째 신호만 사용

        except Exception as e:
            continue

    # 점수 내림차순, 경과일 오름차순 정렬
    results.sort(key=lambda x: (-x['score'], x['days_since_signal']))

    print(f"├─ 시총 {MIN_MARKET_CAP}억+ 필터: {stats['total']} → {stats['market_cap']}개")
    print(f"├─ 스팩/리츠 제외: {stats['excluded']}개 제외")
    print(f"├─ 1단계 통과 (낙폭/바닥/유동성): {stats['stage1']}개")
    print(f"├─ 150일선 이격 -5%~15% + 5점 이상: {stats['stage2']}개")
    print(f"[완료] 바닥 탈출 종목: {len(results)}개")

    return results


if __name__ == "__main__":
    results = screen_bottom_breakout()

    print(f"\n총 {len(results)}개 종목 발견\n")
    print("등급 | 종목명       | 점수 | 경과일 | 150일선이격 | 52주고점대비 | 현재가")
    print("-" * 80)

    for r in results[:30]:
        grade_str = "⭐A" if r['grade'] == 'A' else "  B"
        days_str = "오늘" if r['days_since_signal'] == 0 else f"{r['days_since_signal']}일전"
        print(f"{grade_str} | {r['name'][:8]:8s} | {r['score']:2d}점 | {days_str:5s} | {r['ma150_gap']:+6.1f}% | {r['drop_from_high']:6.1f}% | {r['current_price']:,d}")
