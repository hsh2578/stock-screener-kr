"""
차트 데이터 생성 스크립트

스크리너 결과에 포함된 종목들만 차트 데이터를 생성합니다.
웹사이트 용량 최적화를 위해 선별된 종목만 포함합니다.

사용법:
    python scripts/generate_chart_data.py
"""
import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

from tqdm import tqdm
from pykrx import stock

# 프로젝트 루트를 sys.path에 추가
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from utils.logger import setup_logger

logger = setup_logger()

# ============================================================================
# 상수 정의
# ============================================================================
DATA_DIR = PROJECT_ROOT / "data"
CHART_DAYS = 250  # 약 1년 (거래일 기준)
REQUEST_DELAY = 0.05  # API 요청 간 딜레이 (초)
MAX_RETRIES = 3

# 스크리너 결과 파일 목록
SCREENER_FILES = [
    "box_range.json",
    "box_breakout.json",
    "box_breakout_simple.json",
    "pullback.json",
    "volume_dry_up.json",
    "volume_explosion.json",
    "sector_stage.json",
]


def collect_tickers_from_screeners() -> set[str]:
    """
    모든 스크리너 결과 파일에서 종목 코드를 수집합니다.

    Returns:
        중복 제거된 종목 코드 집합
    """
    tickers = set()

    for filename in SCREENER_FILES:
        filepath = DATA_DIR / filename
        if not filepath.exists():
            logger.debug(f"파일 없음: {filename}")
            continue

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                result = json.load(f)

            # data 배열에서 ticker 추출
            data = result.get("data", [])
            for item in data:
                ticker = item.get("ticker")
                if ticker:
                    tickers.add(ticker)

            logger.debug(f"{filename}: {len(data)}개 종목")

        except Exception as e:
            logger.warning(f"파일 읽기 실패 ({filename}): {e}")
            continue

    return tickers


def get_chart_data(ticker: str, days: int = CHART_DAYS) -> list[dict]:
    """
    개별 종목의 차트 데이터(OHLCV)를 조회합니다.

    Args:
        ticker: 종목 코드
        days: 조회 기간 (거래일)

    Returns:
        차트 데이터 리스트 또는 빈 리스트
    """
    end_date = datetime.now().strftime("%Y%m%d")
    start_date = (datetime.now() - timedelta(days=int(days * 1.5))).strftime("%Y%m%d")

    for attempt in range(MAX_RETRIES):
        try:
            time.sleep(REQUEST_DELAY)

            # 수정주가 사용 (액면분할, 배당락 등 반영)
            df = stock.get_market_ohlcv(start_date, end_date, ticker, adjusted=True)

            if df.empty:
                return []

            # 최근 N거래일만 사용
            df = df.tail(days)

            # 차트 데이터 형식으로 변환
            chart_data = []
            for date_idx, row in df.iterrows():
                # 날짜 형식 변환
                if hasattr(date_idx, "strftime"):
                    date_str = date_idx.strftime("%Y-%m-%d")
                else:
                    date_str = str(date_idx)

                # 소수점 제거 (정수로 저장하여 용량 최적화)
                chart_data.append({
                    "date": date_str,
                    "open": int(row["시가"]),
                    "high": int(row["고가"]),
                    "low": int(row["저가"]),
                    "close": int(row["종가"]),
                    "volume": int(row["거래량"]),
                })

            return chart_data

        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                wait_time = (attempt + 1) * 2
                logger.debug(f"재시도 ({ticker}): {attempt + 1}/{MAX_RETRIES}, {wait_time}초 대기")
                time.sleep(wait_time)
            else:
                logger.debug(f"차트 데이터 조회 실패 ({ticker}): {e}")
                return []

    return []


def generate_chart_data() -> dict[str, list[dict]]:
    """
    스크리너 결과에 포함된 종목들의 차트 데이터를 생성합니다.

    Returns:
        {ticker: [차트데이터], ...} 딕셔너리
    """
    # 1. 스크리너 결과에서 종목 수집
    tickers = collect_tickers_from_screeners()

    if not tickers:
        logger.warning("스크리너 결과에 종목이 없습니다.")
        return {}

    logger.info(f"차트 데이터 생성 중... {len(tickers)}개 종목")

    # 2. 각 종목의 차트 데이터 수집
    chart_data = {}
    failed_count = 0

    for ticker in tqdm(sorted(tickers), desc="차트 데이터 수집", unit="종목", ncols=80):
        data = get_chart_data(ticker, CHART_DAYS)
        if data:
            chart_data[ticker] = data
        else:
            failed_count += 1

    if failed_count > 0:
        logger.warning(f"차트 데이터 수집 실패: {failed_count}개 종목")

    logger.info(f"차트 데이터 수집 완료: {len(chart_data)}개 종목")

    return chart_data


def save_chart_data(chart_data: dict[str, list[dict]]) -> None:
    """
    차트 데이터를 JSON 파일로 저장합니다.

    Args:
        chart_data: {ticker: [차트데이터], ...}
    """
    output_path = DATA_DIR / "chart_data.json"

    # 용량 최적화: indent 없이 저장
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chart_data, f, ensure_ascii=False, separators=(",", ":"))

    # 파일 크기 확인
    file_size = output_path.stat().st_size
    if file_size < 1024:
        size_str = f"{file_size} bytes"
    elif file_size < 1024 * 1024:
        size_str = f"{file_size / 1024:.1f} KB"
    else:
        size_str = f"{file_size / (1024 * 1024):.1f} MB"

    logger.info(f"저장 완료: {output_path} ({size_str})")


def main() -> int:
    """
    메인 실행 함수.

    Returns:
        0: 정상 종료
        1: 에러 발생
    """
    logger.info("=" * 60)
    logger.info("차트 데이터 생성 시작")
    logger.info("=" * 60)

    try:
        # 차트 데이터 생성
        chart_data = generate_chart_data()

        if not chart_data:
            logger.warning("생성된 차트 데이터가 없습니다.")
            return 0

        # 저장
        save_chart_data(chart_data)

        logger.info("=" * 60)
        logger.info("차트 데이터 생성 완료")
        logger.info("=" * 60)

        return 0

    except Exception as e:
        logger.error(f"차트 데이터 생성 실패: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
