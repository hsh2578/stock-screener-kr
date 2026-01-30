"""
거래량 급감 스크리너 (Volume Dry Up Screener)

급등 후 매도세 고갈 종목을 포착합니다. (세력 보유 + 눌림목)
거래량 폭발 후 거래량이 급감하면 매도 물량이 소진된 것으로 해석합니다.

조건:
    - 시가총액: 1,000억 원 이상
    - 거래량 폭발: 특정일 거래량 ≥ 20일 평균 × 4배
    - 주가 조건: 폭발일 종가 ≥ 전일 대비 +8% 이상 상승
    - 거래량 급감: 이후 거래량 ≤ 폭발일 거래량 × 40%
    - 급감 확인: 폭발일로부터 3~8거래일 사이
    - 표시 유지: 포착 후 8거래일까지 계속 표시 (추이 관찰용)
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
MIN_DAYS_AFTER = 3  # 급감 확인 최소 경과일
MAX_DAYS_AFTER = 8  # 급감 확인 최대 경과일
DISPLAY_DAYS = 8  # 포착 후 표시 유지 기간
VOLUME_AVG_PERIOD = 20
MIN_MARKET_CAP = 1000


@dataclass
class VolumeDryUpResult:
    """거래량 급감 스크리너 결과"""
    ticker: str
    name: str
    price: int
    explosion_date: str          # 거래량 폭발일
    detected_date: str           # 조건 포착일 (급감 확인일)
    days_since_detected: int     # 포착 후 경과일
    explosion_change_rate: float # 폭발일 상승률
    volume_decrease_rate: float  # 거래량 감소율 (%)
    updated_at: str

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "name": self.name,
            "price": self.price,
            "explosion_date": self.explosion_date,
            "detected_date": self.detected_date,
            "days_since_detected": self.days_since_detected,
            "explosion_change_rate": round(self.explosion_change_rate, 2),
            "volume_decrease_rate": round(self.volume_decrease_rate, 2),
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
) -> Optional[tuple[str, str, int, float, float]]:
    """
    거래량 급감 종목을 찾습니다.
    포착 후 8거래일까지 표시를 유지합니다.

    Args:
        data: OHLCV 데이터
        ticker: 종목코드

    Returns:
        (폭발일, 포착일, 포착후경과일, 상승률, 거래량감소율) 또는 None
    """
    required_days = VOLUME_AVG_PERIOD + MAX_DAYS_AFTER + DISPLAY_DAYS + 5
    if len(data) < required_days:
        return None

    close = data["종가"]
    volume = data["거래량"]

    vol_avg = volume.rolling(window=VOLUME_AVG_PERIOD, min_periods=VOLUME_AVG_PERIOD).mean()

    # 폭발일 탐색 범위 확대 (포착 후 DISPLAY_DAYS까지 표시하기 위해)
    # 폭발일이 (MIN_DAYS_AFTER + DISPLAY_DAYS) ~ (MAX_DAYS_AFTER + DISPLAY_DAYS) 전까지 탐색
    max_search_days = MAX_DAYS_AFTER + DISPLAY_DAYS

    for days_ago in range(MIN_DAYS_AFTER, max_search_days + 1):
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

        # 급감 조건 확인 (폭발일 + 3~8일 사이에 급감이 발생했는지)
        # 급감 발생일(포착일) 찾기
        for detect_offset in range(MIN_DAYS_AFTER, MAX_DAYS_AFTER + 1):
            detect_idx = explosion_idx + detect_offset

            if detect_idx >= len(data):
                continue

            # 포착일까지의 거래량 평균
            post_volumes = volume.iloc[explosion_idx + 1:detect_idx + 1]
            if len(post_volumes) < MIN_DAYS_AFTER:
                continue

            recent_avg_volume = post_volumes.tail(min(3, len(post_volumes))).mean()

            if pd.isna(recent_avg_volume):
                continue

            # 급감 조건: 폭발일 대비 40% 이하
            if recent_avg_volume <= explosion_volume * VOLUME_DRY_THRESHOLD:
                volume_decrease_rate = (1 - recent_avg_volume / explosion_volume) * 100

                # 폭발일 날짜
                explosion_date = data.index[explosion_idx]
                if isinstance(explosion_date, pd.Timestamp):
                    explosion_date_str = explosion_date.strftime("%Y-%m-%d")
                else:
                    explosion_date_str = str(explosion_date)

                # 포착일 날짜
                detected_date = data.index[detect_idx]
                if isinstance(detected_date, pd.Timestamp):
                    detected_date_str = detected_date.strftime("%Y-%m-%d")
                else:
                    detected_date_str = str(detected_date)

                # 포착 후 경과일 (오늘 기준)
                days_since_detected = len(data) - 1 - detect_idx

                # 포착 후 8거래일 이내만 표시
                if days_since_detected <= DISPLAY_DAYS:
                    return explosion_date_str, detected_date_str, days_since_detected, change_rate, volume_decrease_rate

    return None


def analyze_volume_dry_up(
    data: pd.DataFrame,
    ticker: str,
    name: str,
) -> Optional[VolumeDryUpResult]:
    """개별 종목 분석"""
    required_days = VOLUME_AVG_PERIOD + MAX_DAYS_AFTER + DISPLAY_DAYS + 5
    if not validate_data(data, required_days):
        return None

    result = find_volume_dry_up(data, ticker)
    if result is None:
        return None

    explosion_date, detected_date, days_since_detected, change_rate, volume_decrease_rate = result
    current_price = int(data["종가"].iloc[-1])

    return VolumeDryUpResult(
        ticker=ticker,
        name=name,
        price=current_price,
        explosion_date=explosion_date,
        detected_date=detected_date,
        days_since_detected=days_since_detected,
        explosion_change_rate=change_rate,
        volume_decrease_rate=volume_decrease_rate,
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

        required_days = VOLUME_AVG_PERIOD + MAX_DAYS_AFTER + DISPLAY_DAYS + 5
        if not validate_data(data, required_days):
            continue
        passed_data += 1

        result = analyze_volume_dry_up(data, ticker, name)
        if result:
            results.append(result)

    logger.info(
        f"[거래량 급감] 완료: "
        f"전체 {total}개 → 시총 {passed_cap}개 → "
        f"데이터 {passed_data}개 → 급감 {len(results)}개"
    )

    # 포착일 기준 정렬 (최신순), 같은 날이면 거래량 감소율 순
    results.sort(key=lambda x: (-x.days_since_detected, -x.volume_decrease_rate), reverse=True)
    return [r.to_dict() for r in results]
