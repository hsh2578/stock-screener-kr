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
import json
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# 프로젝트 루트 설정
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from scripts.krx_data import get_stock_master
from scripts.fnguide_data import get_financial_data, save_fnguide_cache, SCHEMA_VERSION

# ============================================================================
# 상수
# ============================================================================

CACHE_DIR = os.path.join(SCRIPT_DIR, '.cache')
MIN_MARKET_CAP = 1000  # 시가총액 1,000억 이상
MAX_WORKERS = 5  # FnGuide 병렬 종목 수 (종목당 7요청은 순차)
MAX_CACHE_DAYS = 2  # 최근 N일 캐시만 유지
CACHE_REFRESH_DAYS = 7  # 종목별 수집 시각(fetched_at)이 N일 지나면 재수집
MIN_FAILED_TO_ABORT = 10
MIN_SUCCESS_RATE = 0.5  # 수집 성공률이 이보다 낮으면 실패로 종료 (FnGuide 개편 등 감지)
STALE_FALLBACK_DAYS = 30  # 수집 실패 시 이전 값을 쓰는 최대 기간

# 재무 히스토리: 신규 API는 연간 3년·분기 4개만 주므로 받은 값을 git에 쌓아 둔다
# (actions/cache 는 7일 미사용 시 삭제되므로 영구 저장소로 부적합)
HISTORY_FILE = os.path.join(SCRIPT_DIR, 'history', 'fin_history.json')
KEEP_PERIODS = {'annual': 5, 'quarter': 8}  # 연간 최대 5년, 분기 8개(=TTM 전년 대비 가능)


def _keep_latest(series: dict, n: int) -> dict:
    return dict(sorted(series.items())[-n:])


def sanitize_annual(annual: dict, trusted=()) -> int:
    """
    지배주주순이익 > 당기순이익 × 1.01 (둘 다 양수)인 연도는 과거 크롤링에서 연결/별도가 섞인 값으로 보고
    그 연도의 지배주주순이익·EPS 를 지운다 (제자리 수정). 지운 연도 수를 반환.
    trusted: 이번에 신규 API 로 받은 연도 — 비지배지분 적자로 지배 > 전체가 실제로 성립하므로 건드리지 않는다
    (예: BGF 2023 지배 778 / 전체 691 / 비지배 -87).
    ponytail: 히스토리 전용 연도에서 비지배 적자로 성립하는 드문 경우도 함께 지워진다.
    """
    ctrl = annual.get('net_income_controlling', {})
    total = annual.get('net_income', {})
    bad = [p for p, v in ctrl.items()
           if p not in trusted and v is not None and total.get(p) is not None
           and v > 0 and total[p] > 0 and v > total[p] * 1.01]
    for p in bad:
        del ctrl[p]
        annual.get('eps', {}).pop(p, None)
    return len(bad)


def apply_history(data: dict, hist: dict) -> None:
    """
    새로 받은 값과 히스토리를 합친다 (같은 기간은 새 값 우선).
    합친 결과를 data 와 hist 양쪽에 반영하고 기간 수를 제한한다.
    """
    trusted = set(data['annual'].get('net_income_controlling', {}))  # 신규 API 값 (병합 전)
    sections = [('annual', data['annual'], KEEP_PERIODS['annual']),
                ('quarter', data['quarter'], KEEP_PERIODS['quarter'])]
    for section, target, n in sections:
        h = hist.setdefault(section, {})
        for key in set(target) | set(h):
            merged = _keep_latest({**h.get(key, {}), **target.get(key, {})}, n)
            target[key] = h[key] = merged
    sanitize_annual(data['annual'], trusted)  # data/hist 가 같은 dict 를 공유하므로 양쪽에 반영됨
    h_cf = hist.setdefault('cashflow', {})
    for key in set(data['cashflow']) | set(h_cf):
        n = KEEP_PERIODS['quarter' if key.endswith('_quarter') else 'annual']
        merged = _keep_latest({**h_cf.get(key, {}), **data['cashflow'].get(key, {})}, n)
        data['cashflow'][key] = h_cf[key] = merged


def seed_history(hist: dict, prev: dict) -> None:
    """
    히스토리에 없는 연간 기간을 이전 캐시(prev, 이관 전 5년치 포함)에서 채운다 (히스토리 값 우선).
    히스토리 항목이 없는 종목이 신규 API(3년)만으로 4~5년차를 잃지 않게 apply_history 전에 호출.
    """
    h = hist.setdefault('annual', {})
    for key, series in ((prev or {}).get('annual') or {}).items():
        h[key] = {**series, **h.get(key, {})}


def load_history() -> dict:
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, encoding='utf-8') as f:
            return json.load(f)
    return {}


def save_history(history: dict) -> None:
    os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
    rounded = {code: {sec: {k: {p: round(v, 2) for p, v in s.items()} for k, s in d.items()}
                      for sec, d in h.items()}
               for code, h in history.items()}
    # 종목당 한 줄 → git diff 가 바뀐 종목만 보이도록
    lines = (json.dumps(code) + ':' + json.dumps(rounded[code], sort_keys=True, ensure_ascii=False,
                                                 separators=(',', ':'))
             for code in sorted(rounded))
    with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
        f.write('{\n' + ',\n'.join(lines) + '\n}\n')


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

    # 3. 이전 캐시에서 재사용 가능한 종목 분리
    codes = filtered['종목코드'].tolist()
    results = {}
    reused = 0

    # 이전 캐시 로드 (어제~2일 전)
    prev_cache = {}
    prev_cache_date = None
    pkl_files = sorted(glob.glob(os.path.join(CACHE_DIR, 'fnguide_*.pkl')), reverse=True)
    today_str = datetime.now().strftime('%Y-%m-%d')
    for pkl_file in pkl_files:
        basename = os.path.basename(pkl_file)
        # 오늘 캐시는 건너뜀 (지금 새로 만들 거니까)
        if today_str in basename:
            continue
        try:
            import pickle
            with open(pkl_file, 'rb') as f:
                prev_cache = pickle.load(f)
            # 파일명에서 날짜 추출
            prev_cache_date = basename.replace('fnguide_', '').replace('.pkl', '')
            print(f"  이전 캐시 로드: {basename} ({len(prev_cache)}개 종목)")
            break
        except Exception:
            continue

    # 증분 수집: 종목별 수집 시각이 7일 이내면 재사용, 아니면 재수집
    # (파일 날짜로 판단하면 매일 이어 저장되는 캐시가 영원히 '신선'해 보여 재수집이 안 됨)
    # 종목마다 0~3일 엇갈리게 해서 전 종목이 같은 날 한꺼번에 재수집되지 않도록 함
    codes_to_fetch = []
    for code in codes:
        cached = prev_cache.get(code)
        refresh_sec = (CACHE_REFRESH_DAYS + sum(map(ord, code)) % 4) * 86400
        fresh = time.time() - cached.get('fetched_at', 0) <= refresh_sec if cached else False
        if fresh and cached.get('schema') == SCHEMA_VERSION:
            results[code] = cached
            reused += 1
        else:
            codes_to_fetch.append(code)
    print(f"  캐시 재사용: {reused}개, 수집 필요: {len(codes_to_fetch)}개 (이전 캐시: {prev_cache_date or '없음'})")

    # 4. FnGuide 재무데이터 병렬 수집 (신규/갱신 대상만)
    print(f"\n[3/3] FnGuide 재무데이터 수집 ({len(codes_to_fetch)}개 종목, {MAX_WORKERS}스레드)...")

    success = 0
    failed = 0

    history = load_history()
    print(f"  재무 히스토리: {len(history)}개 종목")

    if codes_to_fetch:
        def _fetch_one(code):
            return code, get_financial_data(code)

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(_fetch_one, code): code for code in codes_to_fetch}

            for future in as_completed(futures):
                try:
                    code, data = future.result()
                    if data:
                        hist = history.setdefault(code, {})
                        seed_history(hist, prev_cache.get(code))
                        apply_history(data, hist)
                        results[code] = data
                        success += 1
                    else:
                        failed += 1
                        # 일시 실패 시 최근(30일 내) 값만 유지. fetched_at 그대로 → 다음 실행에 재시도
                        prev = prev_cache.get(code)
                        if prev and time.time() - prev.get('fetched_at', 0) <= STALE_FALLBACK_DAYS * 86400:
                            results[code] = prev
                except Exception:
                    failed += 1

                done = success + failed
                if done % 100 == 0 or done == len(codes_to_fetch):
                    elapsed = time.time() - start_time
                    print(f"  진행: {done}/{len(codes_to_fetch)} (성공: {success}, 실패: {failed}, {elapsed:.0f}초)")

    print(f"\n수집 완료: 재사용 {reused}개 + 신규 {success}개 성공, {failed}개 실패 (총 {len(results)}개)")

    # 캐시 저장
    cache_file = get_cache_filepath()
    save_fnguide_cache(results, cache_file)
    save_history(history)

    # 이전 캐시 정리
    cleanup_old_caches()

    elapsed = time.time() - start_time
    print(f"\n총 실행 시간: {elapsed:.1f}초 ({elapsed/60:.1f}분)")
    print("=" * 60)

    # 표본이 몇 개뿐이면 신규상장 등 정상 실패로도 비율이 깨진다 → 실패가 일정 수 이상일 때만 판정
    # ponytail: 고정 하한, 소량 수집에서 API 깨짐은 다음 대량 재수집 때 잡힌다
    if failed >= MIN_FAILED_TO_ABORT and success / len(codes_to_fetch) < MIN_SUCCESS_RATE:
        sys.exit(f"FnGuide 수집 성공률 {success}/{len(codes_to_fetch)} < {MIN_SUCCESS_RATE:.0%} "
                 f"— API 변경 여부 확인 필요 (scripts/fnguide_data.py)")

    return results


if __name__ == '__main__':
    collect_fnguide_data()
