"""
국내 주식 스크리너 메인 실행 스크립트

이 스크립트는 GitHub Actions에서 매일 자동 실행되며,
다음 작업을 순차적으로 수행합니다:

    1. 시장 개장 여부 확인
    2. 종목 리스트 수집 (시가총액 1,000억 이상)
    3. OHLCV 데이터 수집
    4. 8개 스크리너 순차 실행
    5. 결과 JSON 파일 저장

사용법:
    python scripts/main.py

환경 변수:
    TZ: Asia/Seoul (GitHub Actions에서 설정)
"""
import json
import sys
import time
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

# 프로젝트 루트를 sys.path에 추가
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from fetch_data import (
    check_market_open,
    get_stock_data,
    get_market_cap_list,
    get_volume_rank,
)
from screeners.box_range import screen_box_range
from screeners.box_breakout import screen_box_breakout
from screeners.box_breakout_simple import screen_box_breakout_simple
from screeners.pullback import screen_pullback
from screeners.volume_dry_up import screen_volume_dry_up
from screeners.volume_explosion import screen_volume_explosion
from screeners.sector_stage import screen_sector_stage
from screeners.new_high_52w import screen_new_high_52w
from generate_chart_data import generate_chart_data, save_chart_data
from utils.logger import setup_logger, timer

# 디렉토리 설정
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

logger = setup_logger()


def save_json(filename: str, data: dict) -> None:
    """데이터를 JSON 파일로 저장합니다."""
    output_path = DATA_DIR / f"{filename}.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    logger.info(f"저장 완료: {output_path}")


def collect_stock_data(stock_list: list[dict], days: int = 200) -> dict:
    """
    종목 리스트의 OHLCV 데이터를 수집합니다.

    Args:
        stock_list: [{'ticker': '005930', 'name': '삼성전자', ...}, ...]
        days: 수집 기간 (거래일, 기본 200일 - 150일선 계산에 충분)

    Returns:
        {ticker: DataFrame, ...} 딕셔너리
    """
    stock_data = {}
    failed_count = 0

    logger.info(f"OHLCV 데이터 수집 시작: {len(stock_list)}개 종목, {days}거래일")

    for item in tqdm(stock_list, desc="OHLCV 수집", unit="종목", ncols=80):
        ticker = item["ticker"]
        name = item["name"]

        try:
            df = get_stock_data(ticker, days=days)
            if not df.empty and len(df) >= 40:
                stock_data[ticker] = df
            else:
                failed_count += 1
        except Exception as e:
            logger.debug(f"데이터 수집 실패: {ticker} ({name}) - {e}")
            failed_count += 1

    if failed_count > 0:
        logger.warning(f"수집 실패/부족: {failed_count}개 종목")

    logger.info(f"OHLCV 데이터 수집 완료: {len(stock_data)}개 성공")

    return stock_data


def run_screener_safe(name: str, screener_func, *args, **kwargs) -> list[dict]:
    """
    스크리너를 안전하게 실행합니다 (에러 발생 시 스킵).

    Args:
        name: 스크리너 이름
        screener_func: 스크리너 함수
        *args, **kwargs: 스크리너 인자

    Returns:
        스크리너 결과 리스트 (에러 시 빈 리스트)
    """
    try:
        return screener_func(*args, **kwargs)
    except Exception as e:
        logger.error(f"[{name}] 스크리너 실행 실패: {e}", exc_info=True)
        return []


def run_all_screeners(
    stock_data: dict,
    stock_info,
    volume_rank_list: list[str]
) -> dict[str, list[dict]]:
    """
    모든 스크리너를 순차적으로 실행합니다.

    Args:
        stock_data: {ticker: DataFrame} 형태의 OHLCV 데이터
        stock_info: 종목 정보 DataFrame
        volume_rank_list: 거래대금 상위 종목 리스트

    Returns:
        {스크리너명: 결과리스트} 딕셔너리
    """
    logger.info("=" * 60)
    logger.info("스크리너 실행 (8개)")
    logger.info("=" * 60)

    total_stocks = len(stock_data)
    results = {}
    screener_times = {}

    # ========================================================================
    # 1. 박스권 횡보 스크리너
    # ========================================================================
    start = time.perf_counter()
    with timer("1. 박스권 횡보", logger):
        box_range_results = run_screener_safe(
            "박스권 횡보", screen_box_range, stock_data, stock_info
        )
        results["box_range"] = box_range_results

        save_json("box_range", {
            "meta": {
                "type": "box-range",
                "name": "박스권 횡보",
                "description": "40거래일 이상 종가 변동폭 20% 이내 횡보 종목",
                "lastUpdated": datetime.now().isoformat(),
                "totalCount": len(box_range_results),
                "screened_from": total_stocks
            },
            "data": box_range_results
        })
    screener_times["박스권 횡보"] = time.perf_counter() - start

    # ========================================================================
    # 2. 박스권 돌파 (거래량 동반)
    # ========================================================================
    start = time.perf_counter()
    with timer("2. 박스권 돌파 (거래량)", logger):
        box_breakout_results = run_screener_safe(
            "박스권 돌파(거래량)", screen_box_breakout, stock_data, stock_info
        )
        results["box_breakout"] = box_breakout_results

        save_json("box_breakout", {
            "meta": {
                "type": "box-breakout",
                "name": "박스권 돌파 (거래량 동반)",
                "description": "박스권 돌파 + 거래량 2배 + 150일선 위",
                "lastUpdated": datetime.now().isoformat(),
                "totalCount": len(box_breakout_results),
                "screened_from": total_stocks
            },
            "data": box_breakout_results
        })
    screener_times["박스권 돌파(거래량)"] = time.perf_counter() - start

    # ========================================================================
    # 3. 풀백 (돌파 후 눌림목)
    # ========================================================================
    start = time.perf_counter()
    with timer("3. 풀백 (눌림목)", logger):
        pullback_results = run_screener_safe(
            "풀백", screen_pullback, stock_data, stock_info
        )
        results["pullback"] = pullback_results

        save_json("pullback", {
            "meta": {
                "type": "pullback",
                "name": "풀백 (돌파 후 눌림목)",
                "description": "돌파 후 저항선으로 되돌아온 종목",
                "lastUpdated": datetime.now().isoformat(),
                "totalCount": len(pullback_results),
                "screened_from": total_stocks
            },
            "data": pullback_results
        })
    screener_times["풀백"] = time.perf_counter() - start

    # ========================================================================
    # 4. 거래량 급감 (폭발 후 건조)
    # ========================================================================
    start = time.perf_counter()
    with timer("4. 거래량 급감", logger):
        volume_dry_results = run_screener_safe(
            "거래량 급감", screen_volume_dry_up, stock_data, stock_info
        )
        results["volume_dry_up"] = volume_dry_results

        save_json("volume_dry_up", {
            "meta": {
                "type": "volume-dry-up",
                "name": "거래량 급감 (폭발 후 건조)",
                "description": "급등 후 매도세 고갈 종목",
                "lastUpdated": datetime.now().isoformat(),
                "totalCount": len(volume_dry_results),
                "screened_from": total_stocks
            },
            "data": volume_dry_results
        })
    screener_times["거래량 급감"] = time.perf_counter() - start

    # ========================================================================
    # 5. 거래량 폭발 (당일 급등)
    # ========================================================================
    start = time.perf_counter()
    with timer("5. 거래량 폭발", logger):
        volume_exp_results = run_screener_safe(
            "거래량 폭발", screen_volume_explosion, stock_data, stock_info, volume_rank_list
        )
        results["volume_explosion"] = volume_exp_results

        save_json("volume_explosion", {
            "meta": {
                "type": "volume-explosion",
                "name": "거래량 폭발 (당일 급등)",
                "description": "당일 거래량 6배 + 8% 이상 상승",
                "lastUpdated": datetime.now().isoformat(),
                "totalCount": len(volume_exp_results),
                "screened_from": total_stocks
            },
            "data": volume_exp_results
        })
    screener_times["거래량 폭발"] = time.perf_counter() - start

    # ========================================================================
    # 6. 업종별 4단계 판별
    # ========================================================================
    start = time.perf_counter()
    with timer("6. 업종 4단계", logger):
        sector_results = run_screener_safe("업종 4단계", screen_sector_stage)
        results["sector_stage"] = sector_results

        save_json("sector_stage", {
            "meta": {
                "type": "sector-stage",
                "name": "업종별 4단계 판별",
                "description": "Weinstein 4단계 업종 분석",
                "lastUpdated": datetime.now().isoformat(),
                "totalCount": len(sector_results)
            },
            "data": sector_results
        })
    screener_times["업종 4단계"] = time.perf_counter() - start

    # ========================================================================
    # 7. 박스권 돌파 (거래량 무관)
    # ========================================================================
    start = time.perf_counter()
    with timer("7. 박스권 돌파 (단순)", logger):
        breakout_simple_results = run_screener_safe(
            "박스권 돌파(단순)", screen_box_breakout_simple, stock_data, stock_info
        )
        results["box_breakout_simple"] = breakout_simple_results

        save_json("box_breakout_simple", {
            "meta": {
                "type": "box-breakout-simple",
                "name": "박스권 돌파 (거래량 무관)",
                "description": "박스권 상단 +2% 돌파 후 10거래일 이내",
                "lastUpdated": datetime.now().isoformat(),
                "totalCount": len(breakout_simple_results),
                "screened_from": total_stocks
            },
            "data": breakout_simple_results
        })
    screener_times["박스권 돌파(단순)"] = time.perf_counter() - start

    # ========================================================================
    # 8. 52주 신고가 돌파
    # ========================================================================
    start = time.perf_counter()
    with timer("8. 52주 신고가", logger):
        new_high_results = run_screener_safe(
            "52주 신고가", screen_new_high_52w, stock_data, stock_info
        )
        results["new_high_52w"] = new_high_results

        save_json("new_high_52w", {
            "meta": {
                "type": "new-high-52w",
                "name": "52주 신고가 돌파",
                "description": "52주 신고가 돌파 후 8거래일 이내 종목",
                "lastUpdated": datetime.now().isoformat(),
                "totalCount": len(new_high_results),
                "screened_from": total_stocks
            },
            "data": new_high_results
        })
    screener_times["52주 신고가"] = time.perf_counter() - start

    # ========================================================================
    # 결과 요약
    # ========================================================================
    logger.info("=" * 60)
    logger.info("스크리너 결과 요약")
    logger.info("=" * 60)
    logger.info(f"  분석 대상: {total_stocks}개 종목")
    logger.info("-" * 40)

    for name, result_list in results.items():
        count = len(result_list)
        elapsed = screener_times.get(name.replace("_", " ").title(), 0)
        logger.info(f"  {name}: {count}개")

    logger.info("-" * 40)
    logger.info("스크리너별 소요 시간:")
    for name, elapsed in screener_times.items():
        logger.info(f"  {name}: {elapsed:.2f}초")

    return results


def main() -> int:
    """
    메인 실행 함수.

    Returns:
        0: 정상 종료
        1: 에러 발생
    """
    start_time = time.perf_counter()

    logger.info("=" * 60)
    logger.info("국내 주식 스크리너 시작")
    logger.info(f"실행 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)

    # ========================================================================
    # 1. 시장 개장 여부 확인
    # ========================================================================
    with timer("시장 상태 확인", logger):
        is_open = check_market_open()

    if not is_open:
        logger.info("Market is closed today.")
        logger.info("=" * 60)
        logger.info("스크리너 종료 (휴장일)")
        logger.info("=" * 60)
        return 0

    # ========================================================================
    # 2. 시가총액 1,000억 이상 종목 리스트 수집
    # ========================================================================
    with timer("종목 리스트 수집 (시총 1,000억 이상)", logger):
        stock_info = get_market_cap_list(min_cap_billion=1000)

        if stock_info.empty:
            logger.error("종목 리스트 수집 실패")
            return 1

        stock_list = stock_info.to_dict(orient="records")

        save_json("stock_list", {
            "meta": {
                "lastUpdated": datetime.now().isoformat(),
                "totalCount": len(stock_list),
                "filter": "시가총액 1,000억원 이상",
                "markets": {
                    "KOSPI": len(stock_info[stock_info["market"] == "KOSPI"]),
                    "KOSDAQ": len(stock_info[stock_info["market"] == "KOSDAQ"])
                }
            },
            "data": stock_list
        })

        logger.info(f"종목 리스트: {len(stock_list)}개")

    # ========================================================================
    # 3. 거래대금 상위 50개 종목 조회
    # ========================================================================
    with timer("거래대금 상위 종목 조회", logger):
        try:
            volume_rank_df = get_volume_rank(top_n=50)
            volume_rank_list = volume_rank_df["ticker"].tolist()
        except Exception as e:
            logger.warning(f"거래대금 순위 조회 실패: {e}")
            volume_rank_list = []

    # ========================================================================
    # 4. OHLCV 데이터 수집
    # ========================================================================
    with timer("OHLCV 데이터 수집", logger):
        # 52주 신고가 스크리너에 260일(250+8+2) 필요
        stock_data = collect_stock_data(stock_list, days=270)

        if not stock_data:
            logger.error("OHLCV 데이터 수집 실패")
            return 1

    # ========================================================================
    # 5. 스크리너 실행
    # ========================================================================
    run_all_screeners(stock_data, stock_info, volume_rank_list)

    # ========================================================================
    # 6. 차트 데이터 생성 (스크리너에 걸린 종목만)
    # ========================================================================
    with timer("차트 데이터 생성", logger):
        try:
            chart_data = generate_chart_data()
            if chart_data:
                save_chart_data(chart_data)
        except Exception as e:
            logger.error(f"차트 데이터 생성 실패: {e}", exc_info=True)

    # ========================================================================
    # 7. 완료
    # ========================================================================
    elapsed_time = time.perf_counter() - start_time

    logger.info("=" * 60)
    logger.info("국내 주식 스크리너 완료")
    logger.info(f"총 실행 시간: {elapsed_time:.2f}초 ({elapsed_time / 60:.1f}분)")
    logger.info("=" * 60)

    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("사용자에 의해 중단됨")
        sys.exit(130)
    except Exception as e:
        logger.critical(f"예상치 못한 에러: {e}", exc_info=True)
        sys.exit(1)
