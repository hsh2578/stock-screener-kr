"""
투자 대가 스크리너 4종 (Buffett / Ackman / Lynch / Graham)

4명의 투자 대가 전략을 기반으로 종목을 스크리닝합니다.
FnGuide 공유 캐시 + KRX 마스터 인프라 위에 구축.

사용법:
    python guru_screeners.py
"""

import json
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd

# 프로젝트 루트 설정
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from scripts.krx_data import get_stock_master
from scripts.fnguide_data import (
    get_financial_data,
    get_ebit,
    get_roe,
    get_per_from_data,
    get_pbr_from_data,
    get_current_ratio,
    get_debt_ratio,
    get_roa,
    get_roic,
    get_net_current_assets,
    get_fcf,
    get_cagr,
    get_pegy,
    load_fnguide_cache,
)

# ============================================================================
# 상수
# ============================================================================

DATA_PATH = os.path.join(SCRIPT_DIR, 'data')
MIN_MARKET_CAP = 1000  # 시가총액 1,000억 이상
ACKMAN_TOP_N = 30  # 애크먼: FCF 수익률 상위 N개만 출력


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


def _save_result(filename, type_id, name, description, conditions, data_list, total_count=None):
    """결과 JSON 저장 (total_count: 전체 통과 수, data_list 가 상위 N개로 잘린 경우)"""
    os.makedirs(DATA_PATH, exist_ok=True)
    filepath = os.path.join(DATA_PATH, filename)

    output = {
        'meta': {
            'type': type_id,
            'name': name,
            'description': description,
            'lastUpdated': datetime.now().isoformat(),
            'totalCount': len(data_list) if total_count is None else total_count,
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


def _annual_values(fin, key):
    """연간 시계열 [(연도, 값)] 오래된→최근 순, 결측 제외"""
    return sorted((int(p[:4]), v) for p, v in fin.get('annual', {}).get(key, {}).items()
                  if v is not None and not np.isnan(v))


def _per_3y(market_cap, fin):
    """3년 평균 이익 기준 PER = 시총 / 최근 3개 연간 (지배)순이익 평균. 평균 이익 0 이하면 None."""
    ni = _annual_values(fin, 'net_income_controlling') or _annual_values(fin, 'net_income')
    if len(ni) < 3:
        return None
    avg = sum(v for _, v in ni[-3:]) / 3
    return market_cap / avg if avg > 0 else None


def _graham_eps_growth(fin):
    """
    그레이엄 이익 성장 (현명한 투자자 14장: 10년간 ⅓ 이상 ≈ 연 3% → 5년 데이터에 적용).
    최근 2개 연도 EPS 평균 vs 가장 오래된 2개 연도 EPS 평균. 연간 EPS 4개 이상 필요.
    Returns: (실제 증가율 %, 요구 증가율 %) 또는 None
    """
    eps = _annual_values(fin, 'eps')
    if len(eps) < 4:
        return None
    old = (eps[0][1] + eps[1][1]) / 2
    new = (eps[-2][1] + eps[-1][1]) / 2
    if old <= 0:
        return None
    span = eps[-1][0] - eps[0][0] - 1  # 두 평균 구간 중심 사이 간격 (년)
    return (new / old - 1) * 100, (1.03 ** span - 1) * 100


def _buffett_quality(fin):
    """
    버핏 이익 일관성 (Validea Patient Investor / Buffettology, 가용 연간 최대 5년).
    - 연간 ROE 전부 > 10% 그리고 평균 >= 15% (3년 이상)
    - 연간 EPS 전부 > 0, 전년 대비 감소는 최대 1회 (3년 이상)
    Returns: 연간 ROE 평균 또는 None (탈락)
    """
    roe = [v for _, v in _annual_values(fin, 'roe')]
    eps = [v for _, v in _annual_values(fin, 'eps')]
    if len(roe) < 3 or len(eps) < 3:
        return None
    roe_avg = sum(roe) / len(roe)
    if min(roe) <= 10 or roe_avg < 15:
        return None
    if min(eps) <= 0 or sum(b < a for a, b in zip(eps, eps[1:])) > 1:
        return None
    return roe_avg


def _net_debt_to_ebit(fin):
    """(이자부채 - 현금) / EBIT. 순현금이면 음수. 이자부채 계정이 없거나 EBIT <= 0 이면 None."""
    bs = fin.get('balance', {})
    debt = bs.get('total_debt')
    ebit = get_ebit(fin)
    if debt is None or ebit is None or ebit <= 0:
        return None
    return (debt - (bs.get('cash') or 0)) / ebit


# ============================================================================
# 워렌 버핏 -내재가치 전략
# ============================================================================

def screen_buffett(filtered, fnguide_cache):
    """
    워렌 버핏 전략 (Validea "Patient Investor" / Buffettology 기준, 10년 → 가용 연간 최대 5년으로 축소)

    조건:
    - 연간 ROE 전부 > 10%, 평균 >= 15% (3년 이상)
    - 연간 EPS 전부 > 0, 전년 대비 감소 최대 1회 (3년 이상)
    - 장기부채(장기차입금+사채) <= 순이익(TTM) × 5 (무차입 통과)
    - FCF(영업CF - CAPEX, TTM) > 0
    - 이익수익률(1/PER)은 표시만 (CI에 국채금리 데이터 없음)
    - 시총 1000억+
    """
    print("\n" + "=" * 60)
    print("워렌 버핏 -내재가치 전략")
    print("=" * 60)

    results = []
    for _, row in filtered.iterrows():
        code = row['종목코드']
        name = row['종목명']
        market_cap = row.get('시가총액', 0)

        fin = _get_fin_data(code, fnguide_cache)
        if not fin:
            continue

        roe_avg = _buffett_quality(fin)
        if roe_avg is None:
            continue

        bs = fin.get('balance', {})
        long_debt = (bs.get('long_term_debt') or 0) + (bs.get('bonds') or 0)
        ni_ttm = fin.get('net_income_ttm')
        if ni_ttm is None or long_debt > 5 * ni_ttm:
            continue

        fcf = get_fcf(fin)
        if fcf is None or fcf <= 0:
            continue

        # 표시용 지표
        per = get_per_from_data(market_cap, fin)
        pbr = get_pbr_from_data(market_cap, fin)
        cr = get_current_ratio(fin)
        dr = get_debt_ratio(fin)
        div_yield = fin.get('header', {}).get('dividend_yield')

        results.append({
            'ticker': code,
            'name': name,
            'current_price': int(row.get('종가', 0) or 0),
            'market_cap': round(market_cap, 0),
            'roe': round(roe_avg, 1),  # 연간 ROE 평균
            'per': round(per, 1) if per is not None else None,
            'pbr': round(pbr, 2) if pbr is not None else None,
            'earnings_yield': round(100 / per, 2) if per else None,
            'current_ratio': round(cr, 2) if cr is not None else None,
            'debt_ratio': round(dr, 1) if dr is not None else None,
            'long_term_debt': round(long_debt, 0),
            'fcf': round(fcf, 0),
            'eps_growth': get_cagr(fin, 'eps', 5),
            'dividend_yield': div_yield,
        })

    # 연간 ROE 평균 내림차순 정렬
    results.sort(key=lambda x: x['roe'], reverse=True)

    _save_result('buffett.json', 'buffett', '워렌 버핏 전략',
                 'ROE·이익 일관성·저부채·FCF 기반 내재가치 전략 (Validea Patient Investor, 5년 데이터로 축소)',
                 ['연간 ROE 전부 > 10% & 평균 >= 15% (3년+)', '연간 EPS 전부 > 0, 감소 최대 1회 (3년+)',
                  '장기부채 <= 순이익(TTM) × 5', 'FCF > 0'],
                 results)

    _print_result('워렌 버핏 전략', results, [
        {'key': 'name', 'label': '종목명', 'width': 14},
        {'key': 'roe', 'label': 'ROE평균', 'width': 8, 'fmt': '.1f'},
        {'key': 'per', 'label': 'PER', 'width': 8, 'fmt': '.1f'},
        {'key': 'earnings_yield', 'label': '이익수익률', 'width': 9, 'fmt': '.2f'},
        {'key': 'long_term_debt', 'label': '장기부채', 'width': 10, 'fmt': '.0f'},
        {'key': 'fcf', 'label': 'FCF', 'width': 10, 'fmt': '.0f'},
        {'key': 'eps_growth', 'label': 'EPS CAGR', 'width': 9, 'fmt': '.1f'},
    ])

    return results


# ============================================================================
# 빌 애크먼 -주가 재평가 전략
# ============================================================================

def screen_ackman(filtered, fnguide_cache):
    """
    애크먼 투자 원칙(단순·예측 가능한 고수익 사업, 강한 FCF, 보수적 재무)에서 착안한 자체 규칙.
    애크먼이 공개한 정량 스크린이 아님.

    조건:
    - 유형 투하자본 >= 총자산 × 10% (자산이 가벼워 ROIC 가 수백 %로 튀는 종목 제외)
    - ROIC > 15% (EBIT / 유형 투하자본, 그린블라트 방식)
    - ROE > 12%
    - PER < 15
    - FCF > 0
    - 순차입금 / EBIT < 2 (순현금 통과)
    정렬: FCF 수익률(FCF / 시총) 내림차순, 상위 ACKMAN_TOP_N 개만 출력 (meta.totalCount = 전체 통과 수)
    """
    print("\n" + "=" * 60)
    print("빌 애크먼 -주가 재평가 전략")
    print("=" * 60)

    results = []
    for _, row in filtered.iterrows():
        code = row['종목코드']
        name = row['종목명']
        market_cap = row.get('시가총액', 0)

        fin = _get_fin_data(code, fnguide_cache)
        if not fin:
            continue

        roic = get_roic(fin)  # 유형 IC < 총자산 10% 이면 None (get_invested_capital)
        if roic is None or roic <= 15:
            continue

        roe = get_roe(fin)
        if roe is None or roe <= 12:
            continue

        per = get_per_from_data(market_cap, fin)
        if per is None or per <= 0 or per > 15:
            continue

        fcf = get_fcf(fin)
        if fcf is None or fcf <= 0:
            continue

        nd_ebit = _net_debt_to_ebit(fin)
        if nd_ebit is None or nd_ebit >= 2:
            continue

        pbr = get_pbr_from_data(market_cap, fin)
        dr = get_debt_ratio(fin)
        div_yield = fin.get('header', {}).get('dividend_yield')

        results.append({
            'ticker': code,
            'name': name,
            'current_price': int(row.get('종가', 0) or 0),
            'market_cap': round(market_cap, 0),
            'roic': round(roic, 1),
            'per': round(per, 1),
            'pbr': round(pbr, 2) if pbr is not None else None,
            'debt_ratio': round(dr, 1) if dr is not None else None,
            'net_debt_to_ebit': round(nd_ebit, 2),
            'fcf': round(fcf, 0),
            'fcf_yield': round(fcf / market_cap * 100, 2),
            'dividend_yield': div_yield,
            'roe': round(roe, 1),
        })

    # FCF 수익률 내림차순 정렬 → 상위 N개
    results.sort(key=lambda x: x['fcf_yield'], reverse=True)
    total = len(results)
    results = results[:ACKMAN_TOP_N]

    _save_result('ackman.json', 'ackman', '빌 애크먼 전략',
                 '애크먼 투자 원칙에서 착안한 자체 규칙 (고ROIC·FCF·보수적 차입), FCF 수익률 상위 30',
                 ['유형 투하자본 >= 총자산 10%', 'ROIC(유형 투하자본) > 15%', 'ROE > 12%', 'PER < 15',
                  'FCF > 0', '순차입금/EBIT < 2'],
                 results, total_count=total)

    _print_result('빌 애크먼 전략', results, [
        {'key': 'name', 'label': '종목명', 'width': 14},
        {'key': 'roic', 'label': 'ROIC%', 'width': 8, 'fmt': '.1f'},
        {'key': 'roe', 'label': 'ROE%', 'width': 8, 'fmt': '.1f'},
        {'key': 'per', 'label': 'PER', 'width': 8, 'fmt': '.1f'},
        {'key': 'net_debt_to_ebit', 'label': '순차입/EBIT', 'width': 10, 'fmt': '.2f'},
        {'key': 'fcf_yield', 'label': 'FCF수익률', 'width': 9, 'fmt': '.2f'},
    ])

    return results


# ============================================================================
# 피터 린치 -성장주 전략
# ============================================================================

def screen_lynch(filtered, fnguide_cache):
    """
    피터 린치 전략: 합리적 가격의 성장주 (GARP, AAII Lynch 스크린의 PEGY 기준)

    조건:
    - 5 <= PER < 25 (PER 5 미만은 일회성 이익 신호로 제외)
    - 영업이익(TTM) > 0
    - PEGY = PER / (EPS 5년 CAGR + 배당수익률) <= 0.5, 0 < CAGR <= 50%
    - 이자부채(단기차입금+유동성장기부채+장기차입금+사채) / 자본총계 < 80%
    재고/매출 추이는 분기 재고 시계열을 저장하지 않아 판단할 수 없으므로 조건에서 뺐다.
    """
    print("\n" + "=" * 60)
    print("피터 린치 -성장주 전략")
    print("=" * 60)

    results = []
    for _, row in filtered.iterrows():
        code = row['종목코드']
        name = row['종목명']
        market_cap = row.get('시가총액', 0)

        fin = _get_fin_data(code, fnguide_cache)
        if not fin:
            continue

        per = get_per_from_data(market_cap, fin)
        if per is None or per < 5 or per > 25:
            continue

        op_ttm = fin.get('operating_income_ttm')
        if op_ttm is None or op_ttm <= 0:
            continue

        pegy = get_pegy(market_cap, fin)
        if pegy is None or pegy > 0.5:
            continue

        bs = fin.get('balance', {})
        debt, eq = bs.get('total_debt'), bs.get('total_equity')
        if debt is None or eq is None or eq <= 0:
            continue
        debt_to_equity = debt / eq * 100
        if debt_to_equity >= 80:
            continue

        pbr = get_pbr_from_data(market_cap, fin)
        roe = get_roe(fin)
        roa = get_roa(fin)
        div_yield = fin.get('header', {}).get('dividend_yield')

        results.append({
            'ticker': code,
            'name': name,
            'current_price': int(row.get('종가', 0) or 0),
            'market_cap': round(market_cap, 0),
            'per': round(per, 1),
            'peg': round(pegy, 2),  # HTML 'PEG' 컬럼 호환 (값은 PEGY)
            'pegy': round(pegy, 2),
            'eps_growth': get_cagr(fin, 'eps', 5),
            'roe': round(roe, 1) if roe is not None else None,
            'roa': round(roa, 1) if roa is not None else None,
            'pbr': round(pbr, 2) if pbr is not None else None,
            'debt_ratio': round(debt_to_equity, 1),  # 이자부채/자본
            'dividend_yield': div_yield,
        })

    # PEGY 오름차순 정렬 (낮을수록 좋음)
    results.sort(key=lambda x: x['pegy'])

    _save_result('lynch.json', 'lynch', '피터 린치 전략',
                 'PEGY·저부채 기반 합리적 성장주 전략 (AAII Lynch 스크린 기준)',
                 ['5 <= PER < 25', '영업이익(TTM) > 0', 'PEGY = PER/(EPS 5년 CAGR+배당수익률) <= 0.5',
                  '0 < EPS CAGR <= 50%', '이자부채/자본 < 80%'],
                 results)

    _print_result('피터 린치 전략', results, [
        {'key': 'name', 'label': '종목명', 'width': 14},
        {'key': 'pegy', 'label': 'PEGY', 'width': 7, 'fmt': '.2f'},
        {'key': 'per', 'label': 'PER', 'width': 8, 'fmt': '.1f'},
        {'key': 'eps_growth', 'label': 'EPS CAGR', 'width': 9, 'fmt': '.1f'},
        {'key': 'debt_ratio', 'label': '차입/자본', 'width': 9, 'fmt': '.1f'},
        {'key': 'dividend_yield', 'label': '배당률%', 'width': 8, 'fmt': '.2f'},
    ])

    return results


# ============================================================================
# 벤저민 그레이엄 -가치투자 전략
# ============================================================================

def screen_graham(filtered, fnguide_cache):
    """
    벤저민 그레이엄 방어적 투자자 기준 (『현명한 투자자』 14장, 10~20년 조건을 가용 연간 최대 5년으로 축소)

    조건:
    - 매출(TTM) > 1000억 (충분한 규모)
    - 유동비율 >= 2, 장기부채(장기차입금+사채) <= 순유동자산 (재무 건전성)
    - 가용 연간 전 기간 흑자 (순이익 + 영업이익 모두 > 0)
    - 배당 지급 (배당수익률 > 0 으로 근사)
    - 이익 성장: 최근 2년 EPS 평균 >= 최초 2년 EPS 평균 × 1.03^(간격) (연간 EPS 4개 이상)
    - PER(3년 평균 이익) <= 15
    - PBR <= 1.5 또는 PER(3년) × PBR <= 22.5
    """
    print("\n" + "=" * 60)
    print("벤저민 그레이엄 -가치투자 전략")
    print("=" * 60)

    results = []
    for _, row in filtered.iterrows():
        code = row['종목코드']
        name = row['종목명']
        market_cap = row.get('시가총액', 0)

        fin = _get_fin_data(code, fnguide_cache)
        if not fin:
            continue

        # 매출 > 1000억
        rev_ttm = fin.get('revenue_ttm')
        if rev_ttm is None or rev_ttm < 1000:
            continue

        # 유동비율 >= 2 (200%)
        cr = get_current_ratio(fin)
        if cr is None or cr < 2:
            continue

        # 장기부채 <= 순유동자산
        nca = get_net_current_assets(fin)
        bs = fin.get('balance', {})
        long_debt = (bs.get('long_term_debt') or 0) + (bs.get('bonds') or 0)
        if nca is None or long_debt > nca:
            continue

        # 전 기간 흑자 (순이익 + 영업이익 모두 > 0)
        ni = _annual_values(fin, 'net_income')
        oi = _annual_values(fin, 'operating_income')
        if len(ni) < 2 or not oi or min(v for _, v in ni + oi) <= 0:
            continue

        # 배당 지급
        div_yield = fin.get('header', {}).get('dividend_yield')
        if not div_yield or div_yield <= 0:
            continue

        # 이익 성장 (연 3% 환산)
        growth = _graham_eps_growth(fin)
        if growth is None or growth[0] < growth[1]:
            continue

        # PER(3년 평균 이익) <= 15
        per3 = _per_3y(market_cap, fin)
        if per3 is None or per3 > 15:
            continue

        # PBR <= 1.5 또는 PER × PBR <= 22.5
        pbr = get_pbr_from_data(market_cap, fin)
        if pbr is None or pbr <= 0 or (pbr > 1.5 and per3 * pbr > 22.5):
            continue

        dr = get_debt_ratio(fin)

        results.append({
            'ticker': code,
            'name': name,
            'current_price': int(row.get('종가', 0) or 0),
            'market_cap': round(market_cap, 0),
            'per': round(per3, 1),  # 3년 평균 이익 기준
            'pbr': round(pbr, 2),
            'per_x_pbr': round(per3 * pbr, 2),
            'current_ratio': round(cr, 2),
            'net_current_assets': round(nca, 0),
            'eps_growth': round(growth[0], 1),  # 최근 2년 vs 최초 2년 EPS 평균 증가율
            'eps_growth_required': round(growth[1], 1),
            'debt_ratio': round(dr, 1) if dr is not None else None,
            'dividend_yield': div_yield,
            'revenue_ttm': round(rev_ttm, 0),
        })

    # PER×PBR 오름차순 (낮을수록 좋음)
    results.sort(key=lambda x: x['per_x_pbr'])

    _save_result('graham.json', 'graham', '벤저민 그레이엄 전략',
                 '그레이엄 『현명한 투자자』 14장 방어적 투자자 기준 (5년 데이터로 축소)',
                 ['매출 > 1000억', '유동비율 >= 200%', '장기부채 <= 순유동자산',
                  '전 기간 흑자(순이익+영업이익)', '배당 지급(배당수익률 > 0)',
                  '최근 2년 EPS 평균 >= 최초 2년 평균 × 1.03^간격',
                  'PER(3년 평균 이익) <= 15', 'PBR <= 1.5 또는 PER×PBR <= 22.5'],
                 results)

    _print_result('벤저민 그레이엄 전략', results, [
        {'key': 'name', 'label': '종목명', 'width': 14},
        {'key': 'per', 'label': 'PER3y', 'width': 8, 'fmt': '.1f'},
        {'key': 'pbr', 'label': 'PBR', 'width': 7, 'fmt': '.2f'},
        {'key': 'per_x_pbr', 'label': 'PxB', 'width': 7, 'fmt': '.2f'},
        {'key': 'current_ratio', 'label': '유동비율', 'width': 8, 'fmt': '.2f'},
        {'key': 'eps_growth', 'label': 'EPS증가', 'width': 8, 'fmt': '.1f'},
        {'key': 'debt_ratio', 'label': '부채비율', 'width': 8, 'fmt': '.1f'},
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
