"""
PER/PBR 경량 업데이트 스크립트

스크리너 결과에 포함된 종목들의 PER/PBR만 네이버 금융에서 빠르게 갱신합니다.
전체 크롤링(90분) 대신, 스크리너 결과 종목만 병렬로 업데이트합니다.

사용법:
    python scripts/update_per_pbr.py
"""
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from naver_finance import fetch_naver_per_pbr

# ============================================================================
# 상수 정의
# ============================================================================
DATA_DIR = PROJECT_ROOT / "data"
FINANCIAL_DATA_FILE = DATA_DIR / "financial_data.json"
MAX_WORKERS = 8  # 병렬 스레드 수

# 모든 스크리너 결과 파일
SCREENER_FILES = [
    "box_range.json",
    "box_breakout.json",
    "box_breakout_simple.json",
    "pullback.json",
    "volume_dry_up.json",
    "volume_explosion.json",
    "sector_stage.json",
    "new_high_52w.json",
    "fallen_rebound.json",
    "bottom_breakout.json",
    "ma_convergence.json",
    "value_stocks.json",
    "ma60w_quality.json",
]


def collect_tickers_from_screeners() -> set:
    """모든 스크리너 결과에서 종목 코드를 수집합니다."""
    tickers = set()

    for filename in SCREENER_FILES:
        filepath = DATA_DIR / filename
        if not filepath.exists():
            continue

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                result = json.load(f)

            data = result.get("data", [])
            if isinstance(data, list):
                for item in data:
                    ticker = item.get("ticker")
                    if ticker:
                        tickers.add(ticker)
            elif isinstance(data, dict):
                tickers.update(data.keys())
        except Exception:
            continue

    return tickers


def fetch_one(ticker: str) -> tuple:
    """단일 종목 PER/PBR 조회. (ticker, result_dict or None) 반환."""
    try:
        per_pbr = fetch_naver_per_pbr(ticker)
        return (ticker, per_pbr if per_pbr else None)
    except Exception:
        return (ticker, None)


def load_financial_data() -> dict:
    """기존 financial_data.json을 로드합니다."""
    if not FINANCIAL_DATA_FILE.exists():
        return {"meta": {}, "data": {}}

    try:
        with open(FINANCIAL_DATA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"meta": {}, "data": {}}


def save_financial_data(fin_data: dict) -> None:
    """financial_data.json을 저장합니다."""
    DATA_DIR.mkdir(exist_ok=True)
    with open(FINANCIAL_DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(fin_data, f, ensure_ascii=False, indent=2)


def main() -> int:
    print("=" * 60)
    print("PER/PBR 경량 업데이트 시작")
    print("=" * 60)

    # 1. 스크리너 결과에서 종목 수집
    tickers = collect_tickers_from_screeners()
    if not tickers:
        print("스크리너 결과에 종목이 없습니다.")
        return 0

    ticker_list = sorted(tickers)
    print(f"대상 종목: {len(ticker_list)}개 (스레드: {MAX_WORKERS}개)")

    # 2. 기존 데이터 로드
    fin_data = load_financial_data()
    data = fin_data.get("data", {})

    # 3. 병렬 PER/PBR 업데이트
    start_time = time.time()
    updated = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(fetch_one, t): t for t in ticker_list}

        for future in as_completed(futures):
            ticker, per_pbr = future.result()

            if per_pbr:
                if ticker in data:
                    if "metrics" not in data[ticker]:
                        data[ticker]["metrics"] = {}
                    if "per" in per_pbr:
                        data[ticker]["metrics"]["per"] = per_pbr["per"]
                    if "pbr" in per_pbr:
                        data[ticker]["metrics"]["pbr"] = per_pbr["pbr"]
                else:
                    data[ticker] = {
                        "code": ticker,
                        "crawled_at": datetime.now().isoformat(),
                        "metrics": per_pbr,
                    }
                updated += 1
            else:
                failed += 1

    elapsed = time.time() - start_time

    # 4. 메타데이터 업데이트 및 저장
    fin_data["data"] = data
    fin_data["meta"]["updated_at"] = datetime.now().isoformat()
    fin_data["meta"]["total_count"] = len(data)
    fin_data["meta"]["per_pbr_updated_at"] = datetime.now().isoformat()

    save_financial_data(fin_data)

    print(f"\n업데이트 완료: {updated}개 성공, {failed}개 실패 ({elapsed:.1f}초)")
    print(f"저장: {FINANCIAL_DATA_FILE}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
