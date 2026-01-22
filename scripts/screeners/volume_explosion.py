"""
거래량 폭발 스크리너 (Volume Explosion Screener)

당일 거래량 폭발 + 급등 종목을 포착합니다.
세력의 본격적인 매집 또는 급등 시작 신호로 해석할 수 있습니다.

조건:
    - 시가총액: 1,000억 원 이상
    - 거래량: 당일 거래량 ≥ 20일 평균 × 6배
    - 주가: 당일 종가 ≥ 전일 대비 +8% 이상 상승
    - 거래대금: 당일 기준 50위 이내 (별도 체크)
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
VOLUME_EXPLOSION_MULTIPLIER = 6.0  # 거래량 폭발 기준 (20일 평균 × 6배)
PRICE_CHANGE_THRESHOLD = 8.0  # 상승률 기준 (%)
VOLUME_AVG_PERIOD = 20
MIN_MARKET_CAP = 1000
TOP_VOLUME_RANK = 50


@dataclass
class VolumeExplosionResult:
    """거래량 폭발 스크리너 결과"""
    ticker: str
    name: str
    price: int
    change_rate: float
    volume_ratio: float
    volume_rank: int
    updated_at: str

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "name": self.name,
            "price": self.price,
            "change_rate": round(self.change_rate, 2),
            "volume_ratio": round(self.volume_ratio, 2),
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


def check_volume_explosion(
    data: pd.DataFrame,
    ticker: str
) -> Optional[tuple[float, float]]:
    """
    당일 거래량 폭발 여부를 확인합니다.

    Returns:
        (등락률, 거래량비율) 또는 None
    """
    required_days = VOLUME_AVG_PERIOD + 2
    if len(data) < required_days:
        return None

    close = data["종가"]
    volume = data["거래량"]

    # 당일 데이터
    current_close = close.iloc[-1]
    prev_close = close.iloc[-2]
    current_volume = volume.iloc[-1]

    if pd.isna(current_close) or pd.isna(prev_close) or prev_close <= 0:
        return None
    if pd.isna(current_volume):
        return None

    # 20일 평균 거래량 (당일 제외)
    vol_avg = volume.iloc[-(VOLUME_AVG_PERIOD + 1):-1].mean()

    if pd.isna(vol_avg) or vol_avg <= 0:
        return None

    # 등락률 계산
    change_rate = (current_close - prev_close) / prev_close * 100

    # 거래량 비율 계산
    volume_ratio = current_volume / vol_avg

    # 조건 확인
    # 1. 상승률 8% 이상
    if change_rate < PRICE_CHANGE_THRESHOLD:
        return None

    # 2. 거래량 6배 이상
    if volume_ratio < VOLUME_EXPLOSION_MULTIPLIER:
        return None

    return change_rate, volume_ratio


def analyze_volume_explosion(
    data: pd.DataFrame,
    ticker: str,
    name: str,
    volume_rank: int = TOP_VOLUME_RANK
) -> Optional[VolumeExplosionResult]:
    """개별 종목 분석"""
    required_days = VOLUME_AVG_PERIOD + 2
    if not validate_data(data, required_days):
        return None

    result = check_volume_explosion(data, ticker)
    if result is None:
        return None

    change_rate, volume_ratio = result
    current_price = int(data["종가"].iloc[-1])

    return VolumeExplosionResult(
        ticker=ticker,
        name=name,
        price=current_price,
        change_rate=change_rate,
        volume_ratio=volume_ratio,
        volume_rank=volume_rank,
        updated_at=datetime.now().isoformat()
    )


def screen_volume_explosion(
    stock_data: dict[str, pd.DataFrame],
    stock_info: pd.DataFrame,
    volume_rank_list: Optional[list[str]] = None
) -> list[dict]:
    """
    거래량 폭발 스크리너 실행

    Args:
        stock_data: OHLCV 데이터
        stock_info: 종목 정보
        volume_rank_list: 거래대금 상위 50개 종목 리스트 (선택)
    """
    results = []
    total = len(stock_data)
    passed_cap = 0
    passed_data = 0

    logger.info(f"[거래량 폭발] 시작: {total}개 종목")

    # 거래대금 순위 매핑
    volume_rank_map = {}
    if volume_rank_list:
        for rank, ticker in enumerate(volume_rank_list[:TOP_VOLUME_RANK], 1):
            volume_rank_map[ticker] = rank

    for ticker, data in stock_data.items():
        stock_row = stock_info[stock_info["ticker"] == ticker]
        if stock_row.empty:
            continue

        name = stock_row.iloc[0].get("name", "")
        market_cap = stock_row.iloc[0].get("market_cap", 0)

        if market_cap < MIN_MARKET_CAP:
            continue
        passed_cap += 1

        required_days = VOLUME_AVG_PERIOD + 2
        if not validate_data(data, required_days):
            continue
        passed_data += 1

        # 거래대금 순위 확인
        volume_rank = volume_rank_map.get(ticker, TOP_VOLUME_RANK + 1)

        result = analyze_volume_explosion(data, ticker, name, volume_rank)
        if result:
            # 거래대금 50위 이내 필터 (데이터 있을 때만)
            if volume_rank_list and result.volume_rank > TOP_VOLUME_RANK:
                continue
            results.append(result)

    logger.info(
        f"[거래량 폭발] 완료: "
        f"전체 {total}개 → 시총 {passed_cap}개 → "
        f"데이터 {passed_data}개 → 폭발 {len(results)}개"
    )

    # 거래량 비율 내림차순 정렬
    results.sort(key=lambda x: x.volume_ratio, reverse=True)
    return [r.to_dict() for r in results]
