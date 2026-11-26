"""
투자 대가 스크리너 4종 (Buffett / Ackman / Lynch / Graham)

대가의 투자 원칙을 재무 조건으로 옮긴 스크리너. 지표 정의 (손익은 전부 TTM = 최근 4분기 누적):
  PER = 시총 / 지배순이익(TTM), PBR = 시총 / 지배자본
  ROE·ROA = 전체 순이익(TTM) / (최근 분기 + 1년 전) 평균 지배자본·자산
  EPS 성장률 = 지배순이익 TTM / 1년 전 TTM − 1,  PEG = PER / EPS 성장률
  배당수익률 = 최근 보고서의 주당배당금 / 전년 말 종가
현재 분기 재무는 FnGuide 공유 캐시, 1년 전·10년 시계열과 배당은 DART(dartlab 미러, scripts/dart_data.py).
대상은 다른 재무 스크리너와 같이 시총 1,000억 이상 보통주.

사용법:
    python guru_screeners.py
"""

import glob
import json
import os
import sys
import time
from datetime import datetime
from functools import lru_cache

import pandas as pd

# 프로젝트 루트 설정
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from scripts.krx_data import get_stock_master
from scripts.fnguide_data import (
    get_financial_data,
    get_current_ratio,
    get_debt_ratio,
    get_roic,
    get_net_current_assets,
    get_fcf,
    get_inventory_ratio,
    load_fnguide_cache,
)
from scripts.dart_data import (
    batch, dart_series, dart_dps, return_on, ttm_growth, all_quarters_profitable, avg3_growth,
)
from scripts.wics_data import WICS_CACHE_FILE

# ============================================================================
# 상수
# ============================================================================

DATA_PATH = os.path.join(SCRIPT_DIR, 'data')
MIN_MARKET_CAP = 1000  # 시가총액 1,000억 이상
DAILY_DIR = os.path.join(SCRIPT_DIR, '.cache', 'daily')  # 날짜별 전종목 일봉 (scripts/ohlcv_store.py)
TECH_WICS_PREFIX = 'G45'  # WICS IT 섹터 (소프트웨어·하드웨어·반도체·디스플레이) — 그레이엄 '기술 기업 제외'


# ============================================================================
# 공통 유틸리티
# ============================================================================

def _load_data():
    """KRX 마스터 + FnGuide 캐시 로드"""
    print("\n[1/2] KRX 종목 마스터 수집...")
    master = get_stock_master()

    # 기본 필터링 (보통주, 시총 1000억+, 스팩/리츠 제외)
    filtered = master[
        (master['시가총액'] >= MIN_MARKET_CAP) &
        (master['is_common'] == True) &
        (master['is_spac'] == False) &
        (master['is_reit'] == False)
    ].copy()
    print(f"  기본 필터링: {len(filtered)}개 종목")

    print("\n[2/2] FnGuide 캐시 로드...")
    fnguide_cache = load_fnguide_cache()
    if fnguide_cache:
        print(f"  캐시: {len(fnguide_cache)}개 종목")
    else:
        print("  캐시 없음 -개별 수집 모드 (느림)")
        fnguide_cache = {}

    return filtered, fnguide_cache


def _get_fin_data(code, fnguide_cache):
    """캐시 우선 → fallback 개별 수집 (결과를 공유 캐시에 넣어 4개 스크리너가 재수집하지 않게)"""
    if code not in fnguide_cache:
        fnguide_cache[code] = get_financial_data(code)
    return fnguide_cache[code]


def _save_result(filename, type_id, name, description, conditions, data_list):
    """결과 JSON 저장"""
    os.makedirs(DATA_PATH, exist_ok=True)
    filepath = os.path.join(DATA_PATH, filename)

    output = {
        'meta': {
            'type': type_id,
            'name': name,
            'description': description,
            'lastUpdated': datetime.now().isoformat(),
            'totalCount': len(data_list),
            'conditions': conditions,
        },
        'data': data_list,
    }

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"  저장: {filepath} ({len(data_list)}개 종목)")
    return filepath


def _print_result(title, data_list, columns):
    """결과 테이블 출력"""
    if not data_list:
        print(f"\n{title}: 결과 없음")
        return

    print(f"\n{'=' * 80}")
    print(f"{title} ({len(data_list)}개 종목)")
    print(f"{'=' * 80}")

    # 헤더
    header = ''.join(f'{col["label"]:>{col["width"]}}' for col in columns)
    print(header)
    print('-' * 80)

    for item in data_list[:20]:  # 상위 20개만 출력
        row = ''
        for col in columns:
            val = item.get(col['key'])
            if val is None:
                row += f'{"-":>{col["width"]}}'
            elif isinstance(val, float):
                fmt = col.get('fmt', '.1f')
                row += f'{val:>{col["width"]}{fmt}}'
            elif isinstance(val, int):
                row += f'{val:>{col["width"]}}'
            else:
                row += f'{str(val):>{col["width"]}}'
        print(row)


def _base(row, fin):
    """
    시세 + FnGuide 최근 분기로 구하는 지표. 계산할 수 없는 값은 None.
    per/pbr: 지배주주 기준 (별도 재무제표 회사는 전체 = 지배). 적자·자본잠식은 None.
    debt_ratio: 부채총계/자본총계 %,  lt_debt_ratio: 비유동부채/지배자본 %,  current_ratio: %
    """
    cap = row.get('시가총액', 0)
    bs = fin.get('balance', {})
    ni = fin.get('net_income_controlling_ttm')
    if ni is None:
        ni = fin.get('net_income_ttm')
    eq = bs.get('controlling_equity')
    if eq is None:
        eq = bs.get('total_equity')
    cr = get_current_ratio(fin)
    inv = get_inventory_ratio(fin)
    if inv is None and bs.get('current_assets') is not None and (fin.get('revenue_ttm') or 0) > 0:
        inv = 0.0  # 일반 기업 양식인데 재고 계정이 없음 = 재고 없음
    ncl = bs.get('non_current_liabilities')
    return {
        'ticker': row['종목코드'],
        'name': row['종목명'],
        'current_price': int(row.get('종가', 0) or 0),
        'market_cap': round(cap, 0),
        'per': cap / ni if ni and ni > 0 else None,
        'pbr': cap / eq if eq and eq > 0 else None,
        'current_ratio': cr * 100 if cr is not None else None,
        'debt_ratio': get_debt_ratio(fin),
        'lt_debt_ratio': ncl / eq * 100 if ncl is not None and eq and eq > 0 else None,
        'fcf': get_fcf(fin),
        'inventory_ratio': inv,
    }


def _lt(v, limit):
    return v is not None and v < limit


def _gt(v, limit):
    return v is not None and v > limit


def _candidates(filtered, fnguide_cache, keep):
    """FnGuide 지표만으로 거를 수 있는 조건(keep)을 통과한 종목의 [(지표, fin)] — DART 내려받기 대상을 줄인다."""
    out = []
    for _, row in filtered.iterrows():
        fin = _get_fin_data(row['종목코드'], fnguide_cache)
        if not fin:
            continue
        m = _base(row, fin)
        if keep(m, fin):
            out.append((m, fin))
    return out


@lru_cache(maxsize=None)
def _year_end_close(year):
    """그 해 마지막 거래일 종가 {종목코드: 종가}. 일봉 저장소에 12월 파일이 없으면 None."""
    files = sorted(glob.glob(os.path.join(DAILY_DIR, f'{year}12??.parquet')))
    return pd.read_parquet(files[-1], columns=['Close'])['Close'] if files else None


def _dividend_yield(code, fin, dps):
    """
    배당수익률(%) = 최근 보고서 주당배당금 / 전년 말 종가.
    현재가로 나누면 주가가 급락한 종목의 배당 매력이 과대평가되므로 배당 기준 연도의 종가를 쓴다.
    DART 배당 자료나 연말 종가가 없으면 FnGuide 헤더 배당수익률(현재가 기준)로 대체.
    ponytail: 12월 말 종가 고정 — 3·6·9월 결산 법인은 결산월 종가와 다르다
    """
    d = dps.get(code)
    if d:
        closes = _year_end_close(d[1])
        if closes is not None and closes.get(code, 0) > 0:
            return d[0] / closes[code] * 100
    return fin.get('header', {}).get('dividend_yield')


_DIGITS = {'per': 1, 'pbr': 2, 'roe': 1, 'roa': 1, 'roic': 1, 'current_ratio': 2, 'debt_ratio': 1,
           'lt_debt_ratio': 1, 'fcf': 0, 'eps_growth': 1, 'dividend_yield': 2, 'peg': 2,
           'inventory_ratio': 2, 'per_x_pbr': 2, 'net_current_assets': 0, 'eps_growth_10y': 1,
           'revenue_ttm': 0}


def _round(m):
    """출력용 반올림 (조건 판정은 반올림 전 값으로 끝낸 뒤 호출). None 은 그대로."""
    return {k: (round(v, _DIGITS[k]) if k in _DIGITS and v is not None else v) for k, v in m.items()}


# ============================================================================
# 워렌 버핏 -내재가치 전략
# ============================================================================

def screen_buffett(filtered, fnguide_cache):
    """
    워렌 버핏: 자본 효율이 높고 빚이 적으며 이익이 늘고 있는 기업을 비싸지 않게.
    ROE > 15%, 장기부채비율 < 100%, 유동비율 > 1.5, FCF > 0, PER < 17, PBR < 1.5, 부채비율 < 150%, EPS 성장률 > 10%
    """
    print("\n" + "=" * 60)
    print("워렌 버핏 -내재가치 전략")
    print("=" * 60)

    cands = _candidates(filtered, fnguide_cache, lambda m, fin: (
        _lt(m['per'], 17) and _lt(m['pbr'], 1.5) and _gt(m['current_ratio'], 150)
        and _lt(m['debt_ratio'], 150) and _lt(m['lt_debt_ratio'], 100) and _gt(m['fcf'], 0)))
    series = batch(dart_series, [m['ticker'] for m, _ in cands])
    print(f"  재무 조건 통과 {len(cands)}개 → DART 시계열로 ROE·EPS 성장률 확인")

    results = []
    for m, fin in cands:
        s = series[m['ticker']]
        m['roe'], m['eps_growth'] = return_on(s, 'eq'), ttm_growth(s)
        if not (_gt(m['roe'], 15) and _gt(m['eps_growth'], 10)):
            continue
        m['current_ratio'] /= 100  # HTML 은 배수로 표시
        m['dividend_yield'] = fin.get('header', {}).get('dividend_yield')
        results.append(_round(m))

    results.sort(key=lambda x: x['fcf'], reverse=True)  # FCF 큰 순

    _save_result('buffett.json', 'buffett', '워렌 버핏 전략',
                 'ROE·저부채·FCF·저평가·이익 성장',
                 ['ROE > 15%', '장기부채비율 < 100%', '유동비율 > 1.5', 'FCF > 0', 'PER < 17', 'PBR < 1.5',
                  '부채비율 < 150%', 'EPS 성장률(TTM 전년 대비) > 10%'],
                 results)

    _print_result('워렌 버핏 전략', results, [
        {'key': 'name', 'label': '종목명', 'width': 14},
        {'key': 'roe', 'label': 'ROE%', 'width': 8, 'fmt': '.1f'},
        {'key': 'per', 'label': 'PER', 'width': 8, 'fmt': '.1f'},
        {'key': 'pbr', 'label': 'PBR', 'width': 7, 'fmt': '.2f'},
        {'key': 'debt_ratio', 'label': '부채비율', 'width': 8, 'fmt': '.1f'},
        {'key': 'fcf', 'label': 'FCF', 'width': 10, 'fmt': '.0f'},
        {'key': 'eps_growth', 'label': 'EPS성장', 'width': 9, 'fmt': '.1f'},
    ])

    return results


# ============================================================================
# 빌 애크먼 -주가 재평가 전략
# ============================================================================

def screen_ackman(filtered, fnguide_cache):
    """
    빌 애크먼: 차입을 쓰면서도 투하자본 수익률이 높고 현금을 버는, 싸게 거래되는 기업.
    ROIC > 13%, PER < 20, PBR < 2, 부채비율 > 150% (차입을 쓰는 기업), FCF > 0, 배당수익률 > 2%
    ROIC = 영업이익(TTM) / 유형 투하자본 (그린블라트 방식, 마법공식과 같은 정의).
    """
    print("\n" + "=" * 60)
    print("빌 애크먼 -주가 재평가 전략")
    print("=" * 60)

    cands = _candidates(filtered, fnguide_cache, lambda m, fin: (
        _lt(m['per'], 20) and _lt(m['pbr'], 2) and _gt(m['debt_ratio'], 150) and _gt(m['fcf'], 0)))
    codes = [m['ticker'] for m, _ in cands]
    dps, series = batch(dart_dps, codes), batch(dart_series, codes)

    results = []
    for m, fin in cands:
        m['dividend_yield'] = _dividend_yield(m['ticker'], fin, dps)
        m['roic'] = get_roic(fin)
        if not (_gt(m['dividend_yield'], 2) and _gt(m['roic'], 13)):
            continue
        m['roe'] = return_on(series[m['ticker']], 'eq')  # 표시용
        results.append(_round(m))

    results.sort(key=lambda x: x['roic'], reverse=True)

    _save_result('ackman.json', 'ackman', '빌 애크먼 전략',
                 '고ROIC·저평가·차입 활용·FCF·배당',
                 ['ROIC > 13%', 'PER < 20', 'PBR < 2', '부채비율 > 150%', 'FCF > 0', '배당수익률 > 2%'],
                 results)

    _print_result('빌 애크먼 전략', results, [
        {'key': 'name', 'label': '종목명', 'width': 14},
        {'key': 'roic', 'label': 'ROIC%', 'width': 8, 'fmt': '.1f'},
        {'key': 'per', 'label': 'PER', 'width': 8, 'fmt': '.1f'},
        {'key': 'pbr', 'label': 'PBR', 'width': 7, 'fmt': '.2f'},
        {'key': 'debt_ratio', 'label': '부채비율', 'width': 8, 'fmt': '.1f'},
        {'key': 'dividend_yield', 'label': '배당률%', 'width': 8, 'fmt': '.2f'},
    ])

    return results


# ============================================================================
# 피터 린치 -성장주 전략
# ============================================================================

def screen_lynch(filtered, fnguide_cache):
    """
    피터 린치: 이익 성장에 비해 싸고(PEG), 재고가 쌓이지 않으며, 배당을 주는 기업.
    PER < 30, 0 < PEG < 1.8, 재고/매출 < 5%, 부채비율 < 150%, ROE > 5%, ROA > 1%, 배당수익률 > 3%
    """
    print("\n" + "=" * 60)
    print("피터 린치 -성장주 전략")
    print("=" * 60)

    cands = _candidates(filtered, fnguide_cache, lambda m, fin: (
        _lt(m['per'], 30) and _lt(m['inventory_ratio'], 5) and _lt(m['debt_ratio'], 150)))
    dps = batch(dart_dps, [m['ticker'] for m, _ in cands])
    for m, fin in cands:
        m['dividend_yield'] = _dividend_yield(m['ticker'], fin, dps)
    cands = [(m, fin) for m, fin in cands if _gt(m['dividend_yield'], 3)]
    series = batch(dart_series, [m['ticker'] for m, _ in cands])
    print(f"  재무·배당 조건 통과 {len(cands)}개 → DART 시계열로 ROE·ROA·PEG 확인")

    results = []
    for m, fin in cands:
        s = series[m['ticker']]
        m['roe'], m['roa'], m['eps_growth'] = return_on(s, 'eq'), return_on(s, 'assets'), ttm_growth(s)
        m['peg'] = m['per'] / m['eps_growth'] if _gt(m['eps_growth'], 0) else None
        if not (_gt(m['roe'], 5) and _gt(m['roa'], 1) and _lt(m['peg'], 1.8)):
            continue
        results.append(_round(m))

    results.sort(key=lambda x: x['peg'])  # PEG 낮은 순

    _save_result('lynch.json', 'lynch', '피터 린치 전략',
                 'PEG·저재고·저부채·수익성·배당',
                 ['PER < 30', '0 < PEG < 1.8 (PEG = PER / EPS 성장률)', '재고/매출 < 5%', '부채비율 < 150%',
                  'ROE > 5%', 'ROA > 1%', '배당수익률 > 3%'],
                 results)

    _print_result('피터 린치 전략', results, [
        {'key': 'name', 'label': '종목명', 'width': 14},
        {'key': 'peg', 'label': 'PEG', 'width': 7, 'fmt': '.2f'},
        {'key': 'per', 'label': 'PER', 'width': 8, 'fmt': '.1f'},
        {'key': 'eps_growth', 'label': 'EPS성장', 'width': 9, 'fmt': '.1f'},
        {'key': 'roe', 'label': 'ROE%', 'width': 8, 'fmt': '.1f'},
        {'key': 'dividend_yield', 'label': '배당률%', 'width': 8, 'fmt': '.2f'},
    ])

    return results


# ============================================================================
# 벤저민 그레이엄 -가치투자 전략
# ============================================================================

def _tech_codes():
    """WICS IT 섹터 종목코드. 섹터 캐시가 없으면 빈 집합 (제외 없이 진행)."""
    if not os.path.exists(WICS_CACHE_FILE):
        print("  주의: WICS 섹터 캐시 없음 — 기술 기업 제외 조건을 적용하지 못함")
        return set()
    df = pd.read_csv(WICS_CACHE_FILE, dtype=str, encoding='utf-8-sig')
    return set(df.loc[df['WICS_Code'].str.startswith(TECH_WICS_PREFIX), 'Code'])


def screen_graham(filtered, fnguide_cache):
    """
    벤저민 그레이엄 방어적 투자자: 규모·재무 건전성·꾸준한 이익·장기 성장·저평가.
    기술 기업 제외, 매출 > 1,000억, 유동비율 > 200%, 순유동자산 > 장기부채, EPS 10년 성장률 > 30%,
    최근 5년 흑자, PER < 15, PBR×PER < 22
    - 장기부채 = 장기차입금 + 사채
    - 최근 5년 흑자 = 최근 20개 분기 순이익이 전부 흑자 (연간 합계가 흑자여도 적자 분기가 있으면 제외 — 이익의 꾸준함을 본다)
    - EPS 10년 성장률 = 그레이엄 원전 방식: 10년 구간의 처음 3년 평균 대비 최근 3년 평균 지배순이익 증가율
      (끝점 비교는 시작·끝 해의 일시적 호황·불황에 좌우되므로 3년 평균을 쓴다)
    """
    print("\n" + "=" * 60)
    print("벤저민 그레이엄 -가치투자 전략")
    print("=" * 60)

    tech = _tech_codes()

    def keep(m, fin):
        if m['ticker'] in tech or not _gt(fin.get('revenue_ttm'), 1000):
            return False
        if not (_gt(m['current_ratio'], 200) and _lt(m['per'], 15) and m['pbr'] and m['per'] * m['pbr'] < 22):
            return False
        bs = fin.get('balance', {})
        nca = get_net_current_assets(fin)
        if nca is None or nca <= (bs.get('long_term_debt') or 0) + (bs.get('bonds') or 0):
            return False
        m['net_current_assets'] = nca
        m['revenue_ttm'] = fin.get('revenue_ttm')
        return True

    cands = _candidates(filtered, fnguide_cache, keep)
    series = batch(dart_series, [m['ticker'] for m, _ in cands])
    print(f"  재무 조건 통과 {len(cands)}개 → DART 분기 흑자·10년 이익 확인")

    results = []
    for m, fin in cands:
        s = series[m['ticker']]
        m['eps_growth_10y'] = avg3_growth(s, 'ni', 10)
        if not (all_quarters_profitable(s, 20) and _gt(m['eps_growth_10y'], 30)):
            continue
        m['eps_growth'] = ttm_growth(s)  # 표시용 (TTM 전년 대비)
        m['per_x_pbr'] = m['per'] * m['pbr']
        m['current_ratio'] /= 100  # HTML 은 배수로 표시
        m['dividend_yield'] = fin.get('header', {}).get('dividend_yield')
        results.append(_round(m))

    results.sort(key=lambda x: x['per_x_pbr'])  # PER×PBR 낮은 순

    _save_result('graham.json', 'graham', '벤저민 그레이엄 전략',
                 '규모·재무 건전성·꾸준한 이익·장기 성장·저평가',
                 ['기술 기업 제외 (WICS IT)', '매출 > 1,000억', '유동비율 > 200%', '순유동자산 > 장기부채',
                  'EPS 10년 성장률 > 30% (처음 3년 평균 → 최근 3년 평균)', '최근 5년(20개 분기) 연속 흑자',
                  'PER < 15', 'PBR×PER < 22'],
                 results)

    _print_result('벤저민 그레이엄 전략', results, [
        {'key': 'name', 'label': '종목명', 'width': 14},
        {'key': 'per', 'label': 'PER', 'width': 8, 'fmt': '.1f'},
        {'key': 'pbr', 'label': 'PBR', 'width': 7, 'fmt': '.2f'},
        {'key': 'per_x_pbr', 'label': 'PxB', 'width': 7, 'fmt': '.2f'},
        {'key': 'current_ratio', 'label': '유동비율', 'width': 8, 'fmt': '.2f'},
        {'key': 'eps_growth_10y', 'label': '10년성장', 'width': 9, 'fmt': '.1f'},
    ])

    return results


# ============================================================================
# 메인
# ============================================================================

def main():
    print("=" * 60)
    print("투자 대가 스크리너 4종")
    print(f"실행 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    start_time = time.time()

    # 데이터 로드
    filtered, fnguide_cache = _load_data()

    # 4개 전략 실행
    buffett_results = screen_buffett(filtered, fnguide_cache)
    ackman_results = screen_ackman(filtered, fnguide_cache)
    lynch_results = screen_lynch(filtered, fnguide_cache)
    graham_results = screen_graham(filtered, fnguide_cache)

    elapsed = time.time() - start_time

    # 요약
    print(f"\n{'=' * 60}")
    print(f"실행 완료 ({elapsed:.1f}초 / {elapsed/60:.1f}분)")
    print(f"{'=' * 60}")
    print(f"  버핏:   {len(buffett_results)}개 종목")
    print(f"  애크먼: {len(ackman_results)}개 종목")
    print(f"  린치:   {len(lynch_results)}개 종목")
    print(f"  그레이엄: {len(graham_results)}개 종목")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
