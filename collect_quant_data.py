"""
FnGuide 재무데이터 공유 수집기

4개 퀀트 스크리너가 공유하는 FnGuide 데이터를 1회 수집하여 pickle 캐시로 저장합니다.
개별 스크리너가 각각 ~6분씩 크롤링하던 것을 1회(~6분)로 통합합니다.

출력: .cache/fnguide_YYYY-MM-DD.pkl
사용: test_value_screener.py, magic_formula.py, multi_factor.py, ma60w_quality.py

사용법:
    python collect_quant_data.py
"""

import os
import sys
import glob
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# 프로젝트 루트 설정
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from scripts.krx_data import get_stock_master
from scripts.fnguide_data import get_financial_data, save_fnguide_cache

# ============================================================================
# 상수
# ============================================================================

CACHE_DIR = os.path.join(SCRIPT_DIR, '.cache')
MIN_MARKET_CAP = 1000  # 시가총액 1,000억 이상
MAX_WORKERS = 3  # FnGuide 병렬 요청 수 (각 종목당 2페이지 동시 요청 → 실제 최대 6 concurrent)
MAX_CACHE_DAYS = 2  # 최근 N일 캐시만 유지


# ============================================================================
# 메인 로직
# ============================================================================

def get_cache_filepath() -> str:
    """오늘 날짜 기반 캐시 파일 경로"""
    today = datetime.now().strftime('%Y-%m-%d')
    return os.path.join(CACHE_DIR, f'fnguide_{today}.pkl')


def cleanup_old_caches():
    """오래된 FnGuide 캐시 파일 정리 (최근 N일만 유지)"""
    pkl_files = sorted(
        glob.glob(os.path.join(CACHE_DIR, 'fnguide_*.pkl')),
        reverse=True
    )
    for old_file in pkl_files[MAX_CACHE_DAYS:]:
        try:
            os.remove(old_file)
            print(f"  이전 캐시 삭제: {os.path.basename(old_file)}")
        except Exception:
            pass


def collect_fnguide_data():
    """FnGuide 재무데이터 1회 수집 → 캐시 저장"""
    print("=" * 60)
    print("FnGuide 재무데이터 공유 수집기")
    print(f"실행 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    start_time = time.time()

    # 1. KRX 종목 마스터 수집
    print("\n[1/3] KRX 종목 마스터 수집...")
    master = get_stock_master()

    # 2. 필터링: 시총 1000억+, 보통주, 스팩/리츠 제외
    print("\n[2/3] 종목 필터링...")
    filtered = master[
        (master['시가총액'] >= MIN_MARKET_CAP) &
        (master['is_common'] == True) &
        (master['is_spac'] == False) &
        (master['is_reit'] == False)
    ].copy()
    print(f"  대상 종목: {len(filtered)}개 (시총 {MIN_MARKET_CAP}억+, 보통주, 스팩/리츠 제외)")

    # 3. FnGuide 재무데이터 병렬 수집
    print(f"\n[3/3] FnGuide 재무데이터 수집 ({len(filtered)}개 종목, {MAX_WORKERS}스레드)...")

    codes = filtered['종목코드'].tolist()
    results = {}
    success = 0
    failed = 0

    def _fetch_one(code):
        return code, get_financial_data(code)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(_fetch_one, code): code for code in codes}

        for future in as_completed(futures):
            try:
                code, data = future.result()
                if data:
                    results[code] = data
                    success += 1
                else:
                    failed += 1
            except Exception:
                failed += 1

            done = success + failed
            if done % 100 == 0 or done == len(codes):
                elapsed = time.time() - start_time
                print(f"  진행: {done}/{len(codes)} (성공: {success}, 실패: {failed}, {elapsed:.0f}초)")

    print(f"\n수집 완료: {success}개 성공, {failed}개 실패")

    # 캐시 저장
    cache_file = get_cache_filepath()
    save_fnguide_cache(results, cache_file)

    # 이전 캐시 정리
    cleanup_old_caches()

    elapsed = time.time() - start_time
    print(f"\n총 실행 시간: {elapsed:.1f}초 ({elapsed/60:.1f}분)")
    print("=" * 60)

    return results


if __name__ == '__main__':
    collect_fnguide_data()
