"""
업종별 4단계 판별 스크리너 (Sector Stage Screener)

업종 단위 추세를 파악하여 상승 업종에서 종목을 선정하는 데 활용합니다.
Weinstein 4단계 이론을 업종 지수에 적용합니다.

단계 정의:
    - 1단계 (기초): 150일선 기울기 ±2% 이내, 지수 횡보
    - 2단계 (상승): 기울기 +2% 초과, 지수가 150일선 위
    - 3단계 (최정상): 기울기가 우상향에서 평평으로 전환
    - 4단계 (쇠퇴): 기울기 -2% 미만

추가 지표:
    - 3개월 상승률
    - 6개월 과열 경고 (6개월 연속 상승 시)
"""
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from pykrx import stock

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from utils.logger import setup_logger

logger = setup_logger()

# ============================================================================
# 상수 정의
# ============================================================================
MA_PERIOD = 150
SLOPE_PERIOD = 20  # 기울기 측정 기간
SLOPE_FLAT_THRESHOLD = 2.0  # 평평 기준 (±2%)
SLOPE_UP_THRESHOLD = 2.0  # 상승 기준 (+2% 초과)
SLOPE_DOWN_THRESHOLD = -2.0  # 하락 기준 (-2% 미만)
RETURN_3M_DAYS = 60  # 3개월 ≈ 60거래일
RETURN_6M_DAYS = 120  # 6개월 ≈ 120거래일
REQUEST_DELAY = 0.1


@dataclass
class SectorStageResult:
    """업종별 4단계 판별 결과"""
    sector_name: str
    stage: int  # 1, 2, 3, 4
    stage_name: str  # 기초, 상승, 최정상, 쇠퇴
    ma150_slope: float  # 기울기 (%)
    return_3m: float  # 3개월 수익률 (%)
    is_overheated: bool  # 6개월 과열 경고
    current_price: float  # 현재 지수
    ma150: float  # 150일선
    updated_at: str

    def to_dict(self) -> dict:
        return {
            "sector_name": self.sector_name,
            "stage": self.stage,
            "stage_name": self.stage_name,
            "ma150_slope": round(self.ma150_slope, 2),
            "return_3m": round(self.return_3m, 2),
            "is_overheated": self.is_overheated,
            "current_price": round(self.current_price, 2),
            "ma150": round(self.ma150, 2),
            "updated_at": self.updated_at
        }


def get_sector_list() -> list[tuple[str, str]]:
    """
    KRX 업종 지수 목록을 반환합니다.

    Returns:
        [(업종코드, 업종명), ...] 리스트
    """
    # KOSPI 업종 지수
    kospi_sectors = [
        ("1001", "KOSPI"),
        ("1002", "KOSPI 대형주"),
        ("1003", "KOSPI 중형주"),
        ("1004", "KOSPI 소형주"),
        ("1005", "음식료품"),
        ("1006", "섬유의복"),
        ("1007", "종이목재"),
        ("1008", "화학"),
        ("1009", "의약품"),
        ("1010", "비금속광물"),
        ("1011", "철강금속"),
        ("1012", "기계"),
        ("1013", "전기전자"),
        ("1014", "의료정밀"),
        ("1015", "운수장비"),
        ("1016", "유통업"),
        ("1017", "전기가스업"),
        ("1018", "건설업"),
        ("1019", "운수창고업"),
        ("1020", "통신업"),
        ("1021", "금융업"),
        ("1022", "은행"),
        ("1024", "증권"),
        ("1025", "보험"),
        ("1026", "서비스업"),
        ("1027", "제조업"),
    ]

    # KOSDAQ 업종 지수
    kosdaq_sectors = [
        ("2001", "KOSDAQ"),
        ("2024", "KOSDAQ IT"),
    ]

    return kospi_sectors + kosdaq_sectors


def get_sector_ohlcv(sector_code: str, days: int = 200) -> Optional[pd.DataFrame]:
    """
    업종 지수의 OHLCV 데이터를 조회합니다.

    Args:
        sector_code: 업종 코드
        days: 조회 기간 (거래일)

    Returns:
        OHLCV DataFrame 또는 None
    """
    end_date = datetime.now().strftime("%Y%m%d")
    start_date = (datetime.now() - timedelta(days=int(days * 1.5))).strftime("%Y%m%d")

    try:
        time.sleep(REQUEST_DELAY)
        df = stock.get_index_ohlcv(start_date, end_date, sector_code)

        if df.empty:
            return None

        # 컬럼명 정리 (pykrx 업종지수는 컬럼명이 다를 수 있음)
        if "종가" not in df.columns and "close" in df.columns.str.lower():
            df = df.rename(columns=lambda x: x.lower())
            df = df.rename(columns={"close": "종가", "open": "시가", "high": "고가", "low": "저가", "volume": "거래량"})

        return df

    except Exception as e:
        logger.debug(f"업종 지수 조회 실패 ({sector_code}): {e}")
        return None


def calculate_slope(series: pd.Series, period: int) -> Optional[float]:
    """
    이동평균선의 기울기를 계산합니다 (% 변화율).

    Args:
        series: 이동평균선 시리즈
        period: 기울기 측정 기간

    Returns:
        기울기 (%) 또는 None
    """
    if len(series) < period:
        return None

    current = series.iloc[-1]
    past = series.iloc[-period]

    if pd.isna(current) or pd.isna(past) or past <= 0:
        return None

    slope = (current - past) / past * 100
    return slope


def determine_stage(
    current_price: float,
    ma150: float,
    slope: float,
    prev_slope: Optional[float] = None
) -> tuple[int, str]:
    """
    현재 단계를 판별합니다.

    Args:
        current_price: 현재 지수
        ma150: 150일 이동평균
        slope: 현재 기울기 (%)
        prev_slope: 이전 기울기 (%, 3단계 판별용)

    Returns:
        (단계 번호, 단계명)
    """
    # 4단계: 기울기가 -2% 미만
    if slope < SLOPE_DOWN_THRESHOLD:
        return 4, "쇠퇴"

    # 2단계: 기울기 +2% 초과 + 지수가 150일선 위
    if slope > SLOPE_UP_THRESHOLD and current_price > ma150:
        return 2, "상승"

    # 3단계: 우상향에서 평평으로 전환
    if prev_slope is not None:
        if prev_slope > SLOPE_UP_THRESHOLD and -SLOPE_FLAT_THRESHOLD <= slope <= SLOPE_FLAT_THRESHOLD:
            return 3, "최정상"

    # 1단계: 기울기 ±2% 이내 (평평)
    if -SLOPE_FLAT_THRESHOLD <= slope <= SLOPE_FLAT_THRESHOLD:
        return 1, "기초"

    # 기본값
    return 1, "기초"


def check_overheated(data: pd.DataFrame, months: int = 6) -> bool:
    """
    6개월 연속 상승 여부를 확인합니다.

    Args:
        data: OHLCV 데이터
        months: 확인 기간 (월)

    Returns:
        과열 여부
    """
    if len(data) < RETURN_6M_DAYS:
        return False

    close = data["종가"]

    # 월별 수익률 확인 (약 20거래일 = 1개월)
    monthly_returns = []
    for i in range(months):
        start_idx = -(i + 1) * 20 - 1
        end_idx = -i * 20 - 1 if i > 0 else -1

        if abs(start_idx) > len(close):
            return False

        start_price = close.iloc[start_idx]
        end_price = close.iloc[end_idx] if end_idx != -1 else close.iloc[-1]

        if pd.isna(start_price) or pd.isna(end_price) or start_price <= 0:
            return False

        monthly_return = (end_price - start_price) / start_price * 100
        monthly_returns.append(monthly_return > 0)

    # 6개월 연속 상승
    return all(monthly_returns)


def analyze_sector(
    sector_code: str,
    sector_name: str
) -> Optional[SectorStageResult]:
    """
    개별 업종 분석

    Args:
        sector_code: 업종 코드
        sector_name: 업종명

    Returns:
        SectorStageResult 또는 None
    """
    # 데이터 조회
    data = get_sector_ohlcv(sector_code, days=MA_PERIOD + RETURN_6M_DAYS)

    if data is None or len(data) < MA_PERIOD + SLOPE_PERIOD:
        return None

    close = data["종가"]

    # NaN 확인
    if close.isna().any():
        close = close.dropna()
        if len(close) < MA_PERIOD + SLOPE_PERIOD:
            return None

    # 150일 이동평균
    ma150_series = close.rolling(window=MA_PERIOD, min_periods=MA_PERIOD).mean()

    current_price = close.iloc[-1]
    ma150 = ma150_series.iloc[-1]

    if pd.isna(ma150) or ma150 <= 0:
        return None

    # 기울기 계산
    slope = calculate_slope(ma150_series, SLOPE_PERIOD)
    if slope is None:
        return None

    # 이전 기울기 (3단계 판별용)
    prev_slope = None
    if len(ma150_series) > SLOPE_PERIOD * 2:
        prev_ma150 = ma150_series.iloc[:-SLOPE_PERIOD]
        prev_slope = calculate_slope(prev_ma150, SLOPE_PERIOD)

    # 단계 판별
    stage, stage_name = determine_stage(current_price, ma150, slope, prev_slope)

    # 3개월 수익률
    if len(close) >= RETURN_3M_DAYS:
        past_price = close.iloc[-RETURN_3M_DAYS]
        if not pd.isna(past_price) and past_price > 0:
            return_3m = (current_price - past_price) / past_price * 100
        else:
            return_3m = 0.0
    else:
        return_3m = 0.0

    # 과열 여부
    is_overheated = check_overheated(data)

    return SectorStageResult(
        sector_name=sector_name,
        stage=stage,
        stage_name=stage_name,
        ma150_slope=slope,
        return_3m=return_3m,
        is_overheated=is_overheated,
        current_price=current_price,
        ma150=ma150,
        updated_at=datetime.now().isoformat()
    )


def screen_sector_stage() -> list[dict]:
    """
    업종별 4단계 판별 스크리너 실행

    Returns:
        업종별 분석 결과 리스트
    """
    results = []
    sectors = get_sector_list()

    logger.info(f"[업종 4단계] 시작: {len(sectors)}개 업종")

    for sector_code, sector_name in sectors:
        try:
            result = analyze_sector(sector_code, sector_name)
            if result:
                results.append(result)
                logger.debug(f"[{sector_name}] {result.stage}단계 ({result.stage_name})")
        except Exception as e:
            logger.warning(f"업종 분석 실패 ({sector_name}): {e}")
            continue

    # 단계별 그룹화하여 로깅
    stage_counts = {}
    for r in results:
        stage_counts[r.stage] = stage_counts.get(r.stage, 0) + 1

    logger.info(
        f"[업종 4단계] 완료: "
        f"1단계 {stage_counts.get(1, 0)}개, "
        f"2단계 {stage_counts.get(2, 0)}개, "
        f"3단계 {stage_counts.get(3, 0)}개, "
        f"4단계 {stage_counts.get(4, 0)}개"
    )

    # 2단계 우선, 그 다음 3개월 수익률 순
    results.sort(key=lambda x: (-x.stage if x.stage == 2 else x.stage, -x.return_3m))

    return [r.to_dict() for r in results]
