"""
PER/PBR 경량 업데이트 스크립트

스크리너 결과에 포함된 종목들의 PER/PBR만 네이버 금융에서 빠르게 갱신합니다.
전체 크롤링(90분) 대신, 스크리너 결과 종목만 업데이트하여 평일에도 실행 가능합니다.

사용법:
    python scripts/update_per_pbr.py
"""
import json
import sys
import time
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

# 프로젝트 루트를 sys.path에 추가
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from naver_finance import fetch_naver_per_pbr

# ============================================================================
# 상수 정의
# ============================================================================
DATA_DIR = PROJECT_ROOT / "data"
FINANCIAL_DATA_FILE = DATA_DIR / "financial_data.json"
REQUEST_DELAY = 0.15  # 네이버 금융 요청 간 딜레이 (초)

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
                # financial_data.json 등 dict 형태
                tickers.update(data.keys())
        except Exception:
            continue

    return tickers


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

    print(f"대상 종목: {len(tickers)}개")

    # 2. 기존 데이터 로드
    fin_data = load_financial_data()
    data = fin_data.get("data", {})

    # 3. PER/PBR 업데이트
    updated = 0
    failed = 0

    for ticker in tqdm(sorted(tickers), desc="PER/PBR 업데이트", unit="종목", ncols=80):
        try:
            time.sleep(REQUEST_DELAY)
            per_pbr = fetch_naver_per_pbr(ticker)

            if per_pbr:
                # 기존 데이터가 있으면 PER/PBR만 갱신
                if ticker in data:
                    if "metrics" not in data[ticker]:
                        data[ticker]["metrics"] = {}
                    if "per" in per_pbr:
                        data[ticker]["metrics"]["per"] = per_pbr["per"]
                    if "pbr" in per_pbr:
                        data[ticker]["metrics"]["pbr"] = per_pbr["pbr"]
                else:
                    # 새로운 종목이면 기본 구조 생성
                    data[ticker] = {
                        "code": ticker,
                        "crawled_at": datetime.now().isoformat(),
                        "metrics": per_pbr,
                    }
                updated += 1
        except Exception:
            failed += 1

    # 4. 메타데이터 업데이트 및 저장
    fin_data["data"] = data
    fin_data["meta"]["updated_at"] = datetime.now().isoformat()
    fin_data["meta"]["total_count"] = len(data)
    fin_data["meta"]["per_pbr_updated_at"] = datetime.now().isoformat()

    save_financial_data(fin_data)

    print(f"\n업데이트 완료: {updated}개 성공, {failed}개 실패")
    print(f"저장: {FINANCIAL_DATA_FILE}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
