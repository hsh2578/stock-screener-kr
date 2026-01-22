"""
거래량 급감 스크리너 (Volume Dry Up Screener)

급등 후 매도세 고갈 종목을 포착합니다. (세력 보유 + 눌림목)
거래량 폭발 후 거래량이 급감하면 매도 물량이 소진된 것으로 해석합니다.

조건:
    - 시가총액: 1,000억 원 이상
    - 거래량 폭발: 특정일 거래량 ≥ 20일 평균 × 4배
    - 주가 조건: 폭발일 종가 ≥ 전일 대비 +8% 이상 상승
    - 거래대금: 폭발일 기준 50위 이내 (별도 체크)
    - 거래량 급감: 이후 거래량 ≤ 폭발일 거래량 × 40%
    - 경과 기간: 폭발일로부터 2거래일 이상 경과
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
VOLUME_EXPLOSION_MULTIPLIER = 4.0  # 거래량 폭발 기준 (20일 평균 × 4배)
PRICE_CHANGE_THRESHOLD = 8.0  # 상승률 기준 (%)
VOLUME_DRY_THRESHOLD = 0.4  # 건조 기준 (폭발 거래량 × 40%)
MIN_DAYS_AFTER = 2  # 최소 경과일
MAX_DAYS_AFTER = 10  # 최대 탐색 기간
VOLUME_AVG_PERIOD = 20
MIN_MARKET_CAP = 1000
TOP_VOLUME_RANK = 50  # 거래대금 상위 순위


@dataclass
class VolumeDryUpResult:
    """거래량 급감 스크리너 결과"""
    ticker: str
    name: str
    price: int
    explosion_date: str
    explosion_change_rate: float
    volume_decrease_rate: float  # 거래량 감소율 (%)
    volume_rank: int  # 폭발일 거래대금 순위 (추정)
    updated_at: str

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "name": self.name,
            "price": self.price,
            "explosion_date": self.explosion_date,
            "explosion_change_rate": round(self.explosion_change_rate, 2),
            "volume_decrease_rate": round(self.volume_decrease_rate, 2),
            "volume_rank": self.volume_rank,
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


def find_volume_dry_up(
    data: pd.DataFrame,
    ticker: str,
    volume_rank_data: Optional[dict] = None
) -> Optional[tuple[str, float, float, int]]:
    """
    거래량 급감 종목을 찾습니다.

    Args:
        data: OHLCV 데이터
        ticker: 종목코드
        volume_rank_data: 날짜별 거래대금 순위 데이터 (선택)

    Returns:
        (폭발일, 상승률, 거래량감소율, 순위) 또는 None
    """
    required_days = VOLUME_AVG_PERIOD + MAX_DAYS_AFTER + 5
    if len(data) < required_days:
        return None

    close = data["종가"]
    volume = data["거래량"]

    vol_avg = volume.rolling(window=VOLUME_AVG_PERIOD, min_periods=VOLUME_AVG_PERIOD).mean()

    # 폭발일 탐색 (MIN_DAYS_AFTER ~ MAX_DAYS_AFTER 전)
    for days_ago in range(MIN_DAYS_AFTER, MAX_DAYS_AFTER + 1):
        explosion_idx = len(data) - 1 - days_ago

        if explosion_idx < VOLUME_AVG_PERIOD + 1:
            continue

        # 폭발일 데이터
        explosion_close = close.iloc[explosion_idx]
        prev_close = close.iloc[explosion_idx - 1]
        explosion_volume = volume.iloc[explosion_idx]
        avg_vol = vol_avg.iloc[explosion_idx - 1]

        if pd.isna(explosion_close) or pd.isna(prev_close) or prev_close <= 0:
            continue
        if pd.isna(explosion_volume) or pd.isna(avg_vol) or avg_vol <= 0:
            continue

        # 거래량 폭발 조건: 20일 평균 × 4배 이상
        if explosion_volume < avg_vol * VOLUME_EXPLOSION_MULTIPLIER:
            continue

        # 주가 상승 조건: +8% 이상
        change_rate = (explosion_close - prev_close) / prev_close * 100
        if change_rate < PRICE_CHANGE_THRESHOLD:
            continue

        # 이후 거래량 급감 확인
        post_volumes = volume.iloc[explosion_idx + 1:]
        if len(post_volumes) < MIN_DAYS_AFTER:
            continue

        # 최근 거래량이 폭발일 대비 40% 이하인지 확인
        recent_avg_volume = post_volumes.tail(min(3, len(post_volumes))).mean()

        if pd.isna(recent_avg_volume):
            continue

        if recent_avg_volume > explosion_volume * VOLUME_DRY_THRESHOLD:
            continue

        volume_decrease_rate = (1 - recent_avg_volume / explosion_volume) * 100

        # 거래대금 순위 (데이터 있으면 사용, 없으면 추정)
        volume_rank = TOP_VOLUME_RANK  # 기본값 (검증 필요 시 외부 데이터 활용)

        explosion_date = data.index[explosion_idx]
        if isinstance(explosion_date, pd.Timestamp):
            explosion_date_str = explosion_date.strftime("%Y-%m-%d")
        else:
            explosion_date_str = str(explosion_date)

        return explosion_date_str, change_rate, volume_decrease_rate, volume_rank

    return None


def analyze_volume_dry_up(
    data: pd.DataFrame,
    ticker: str,
    name: str,
    volume_rank_data: Optional[dict] = None
) -> Optional[VolumeDryUpResult]:
    """개별 종목 분석"""
    required_days = VOLUME_AVG_PERIOD + MAX_DAYS_AFTER + 5
    if not validate_data(data, required_days):
        return None

    result = find_volume_dry_up(data, ticker, volume_rank_data)
    if result is None:
        return None

    explosion_date, change_rate, volume_decrease_rate, volume_rank = result
    current_price = int(data["종가"].iloc[-1])

    return VolumeDryUpResult(
        ticker=ticker,
        name=name,
        price=current_price,
        explosion_date=explosion_date,
        explosion_change_rate=change_rate,
        volume_decrease_rate=volume_decrease_rate,
        volume_rank=volume_rank,
        updated_at=datetime.now().isoformat()
    )


def screen_volume_dry_up(
    stock_data: dict[str, pd.DataFrame],
    stock_info: pd.DataFrame,
    volume_rank_data: Optional[dict] = None
) -> list[dict]:
    """거래량 급감 스크리너 실행"""
    results = []
    total = len(stock_data)
    passed_cap = 0
    passed_data = 0

    logger.info(f"[거래량 급감] 시작: {total}개 종목")

    for ticker, data in stock_data.items():
        stock_row = stock_info[stock_info["ticker"] == ticker]
        if stock_row.empty:
            continue

        name = stock_row.iloc[0].get("name", "")
        market_cap = stock_row.iloc[0].get("market_cap", 0)

        if market_cap < MIN_MARKET_CAP:
            continue
        passed_cap += 1

        required_days = VOLUME_AVG_PERIOD + MAX_DAYS_AFTER + 5
        if not validate_data(data, required_days):
            continue
        passed_data += 1

        result = analyze_volume_dry_up(data, ticker, name, volume_rank_data)
        if result:
            results.append(result)

    logger.info(
        f"[거래량 급감] 완료: "
        f"전체 {total}개 → 시총 {passed_cap}개 → "
        f"데이터 {passed_data}개 → 급감 {len(results)}개"
    )

    results.sort(key=lambda x: x.volume_decrease_rate, reverse=True)
    return [r.to_dict() for r in results]
