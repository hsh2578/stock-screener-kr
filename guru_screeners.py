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
    get_invested_capital,
    get_ev,
    get_roe,
    get_per_from_data,
    get_pbr_from_data,
    get_current_ratio,
    get_debt_ratio,
    get_long_term_debt_ratio,
    get_roa,
    get_roic,
    get_net_current_assets,
    get_fcf,
    get_inventory_ratio,
    get_peg,
    get_growth_rates_with_ttm,
    get_annual_growth_rates,
    load_fnguide_cache,
)

# ============================================================================
# 상수
# ============================================================================

DATA_PATH = os.path.join(SCRIPT_DIR, 'data')
MIN_MARKET_CAP = 1000  # 시가총액 1,000억 이상


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
    """캐시 우선 → fallback 개별 수집"""
    fin = fnguide_cache.get(code)
    if not fin:
        fin = get_financial_data(code)
    return fin


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


# ============================================================================
# 워렌 버핏 -내재가치 전략
# ============================================================================

def screen_buffett(filtered, fnguide_cache):
    """
    워렌 버핏 전략: 경쟁우위를 가진 우량 기업을 합리적 가격에 매수

    조건:
    - ROE > 15%
    - 장기부채비율 < 100%
    - 유동비율 > 1.5
    - FCF > 0
    - PER < 17
    - PBR < 1.5
    - 부채비율 < 150%
    - EPS 성장률 > 10% (3년 평균)
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

        roe = get_roe(fin)
        if roe is None or roe <= 15:
            continue

        per = get_per_from_data(market_cap, fin)
        if per is None or per <= 0 or per > 17:
            continue

        pbr = get_pbr_from_data(market_cap, fin)
        if pbr is None or pbr <= 0 or pbr > 1.5:
            continue

        cr = get_current_ratio(fin)
        if cr is None or cr < 1.5:
            continue

        dr = get_debt_ratio(fin)
        if dr is None or dr > 150:
            continue

        ltdr = get_long_term_debt_ratio(fin)
        if ltdr is not None and ltdr > 100:
            continue

        fcf = get_fcf(fin)
        if fcf is None or fcf <= 0:
            continue

        # EPS 성장률 (3년 평균 > 10%)
        growth = get_growth_rates_with_ttm(fin, 'net_income', years=3)
        valid_growth = [g for g in growth if g is not None]
        avg_growth = sum(valid_growth) / len(valid_growth) if valid_growth else None
        if avg_growth is None or avg_growth <= 10:
            continue

        # 헤더에서 배당수익률 가져오기
        header = fin.get('header', {})
        div_yield = header.get('dividend_yield')

        results.append({
            'ticker': code,
            'name': name,
            'current_price': int(row.get('종가', 0) or 0),
            'market_cap': round(market_cap, 0),
            'roe': round(roe, 1),
            'per': round(per, 1),
            'pbr': round(pbr, 2),
            'current_ratio': round(cr, 2),
            'debt_ratio': round(dr, 1),
            'long_term_debt_ratio': round(ltdr, 1) if ltdr is not None else None,
            'fcf': round(fcf, 0),
            'eps_growth': round(avg_growth, 1),
            'dividend_yield': div_yield,
        })

    # ROE 내림차순 정렬
    results.sort(key=lambda x: x['roe'], reverse=True)

    _save_result('buffett.json', 'buffett', '워렌 버핏 전략',
                 'ROE·FCF·부채비율 기반 내재가치 전략',
                 ['ROE > 15%', 'PER < 17', 'PBR < 1.5', '유동비율 > 1.5',
                  '부채비율 < 150%', 'FCF > 0', 'EPS성장률 > 10%'],
                 results)

    _print_result('워렌 버핏 전략', results, [
        {'key': 'name', 'label': '종목명', 'width': 14},
        {'key': 'roe', 'label': 'ROE%', 'width': 8, 'fmt': '.1f'},
        {'key': 'per', 'label': 'PER', 'width': 8, 'fmt': '.1f'},
        {'key': 'pbr', 'label': 'PBR', 'width': 7, 'fmt': '.2f'},
        {'key': 'current_ratio', 'label': '유동비율', 'width': 8, 'fmt': '.2f'},
        {'key': 'debt_ratio', 'label': '부채비율', 'width': 8, 'fmt': '.1f'},
        {'key': 'fcf', 'label': 'FCF', 'width': 10, 'fmt': '.0f'},
        {'key': 'eps_growth', 'label': 'EPS성장', 'width': 8, 'fmt': '.1f'},
    ])

    return results


# ============================================================================
# 빌 애크먼 -주가 재평가 전략
# ============================================================================

def screen_ackman(filtered, fnguide_cache):
    """
    빌 애크먼 전략: 높은 ROIC + 저평가 + 배당으로 주가 재평가 기대

    조건:
    - ROIC > 15%
    - ROE > 12%
    - PER < 15
    - PBR < 2
    - 부채비율 < 150%
    - FCF > 0
    - 배당수익률 > 3%
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

        roic = get_roic(fin)
        if roic is None or roic <= 15:
            continue

        roe = get_roe(fin)
        if roe is None or roe <= 12:
            continue

        per = get_per_from_data(market_cap, fin)
        if per is None or per <= 0 or per > 15:
            continue

        pbr = get_pbr_from_data(market_cap, fin)
        if pbr is None or pbr <= 0 or pbr > 2:
            continue

        dr = get_debt_ratio(fin)
        if dr is None or dr > 150:
            continue

        fcf = get_fcf(fin)
        if fcf is None or fcf <= 0:
            continue

        # 배당수익률 (FnGuide 헤더)
        header = fin.get('header', {})
        div_yield = header.get('dividend_yield')
        if div_yield is None or div_yield < 3:
            continue

        results.append({
            'ticker': code,
            'name': name,
            'current_price': int(row.get('종가', 0) or 0),
            'market_cap': round(market_cap, 0),
            'roic': round(roic, 1),
            'per': round(per, 1),
            'pbr': round(pbr, 2),
            'debt_ratio': round(dr, 1),
            'fcf': round(fcf, 0),
            'dividend_yield': round(div_yield, 2),
            'roe': round(roe, 1) if roe is not None else None,
        })

    # ROIC 내림차순 정렬
    results.sort(key=lambda x: x['roic'], reverse=True)

    _save_result('ackman.json', 'ackman', '빌 애크먼 전략',
                 'ROIC·배당·FCF 기반 주가 재평가 전략',
                 ['ROIC > 15%', 'ROE > 12%', 'PER < 15', 'PBR < 2',
                  '부채비율 < 150%', 'FCF > 0', '배당수익률 > 3%'],
                 results)

    _print_result('빌 애크먼 전략', results, [
        {'key': 'name', 'label': '종목명', 'width': 14},
        {'key': 'roic', 'label': 'ROIC%', 'width': 8, 'fmt': '.1f'},
        {'key': 'per', 'label': 'PER', 'width': 8, 'fmt': '.1f'},
        {'key': 'pbr', 'label': 'PBR', 'width': 7, 'fmt': '.2f'},
        {'key': 'debt_ratio', 'label': '부채비율', 'width': 8, 'fmt': '.1f'},
        {'key': 'fcf', 'label': 'FCF', 'width': 10, 'fmt': '.0f'},
        {'key': 'dividend_yield', 'label': '배당률%', 'width': 8, 'fmt': '.2f'},
    ])

    return results


# ============================================================================
# 피터 린치 -성장주 전략
# ============================================================================

def screen_lynch(filtered, fnguide_cache):
    """
    피터 린치 전략: 합리적 가격의 성장주 (GARP)

    조건:
    - PER < 25
    - 0 < PEG < 1.8
    - 부채비율 < 150%
    - ROE > 5%
    - ROA > 1%
    - 배당수익률 > 3%
    - 재고/매출 < 5%
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
        if per is None or per <= 0 or per > 25:
            continue

        peg = get_peg(market_cap, fin)
        if peg is None or peg <= 0 or peg > 1.8:
            continue

        dr = get_debt_ratio(fin)
        if dr is None or dr > 150:
            continue

        roe = get_roe(fin)
        if roe is None or roe < 5:
            continue

        roa = get_roa(fin)
        if roa is None or roa < 1:
            continue

        # 배당수익률 > 3% (필수)
        header = fin.get('header', {})
        div_yield = header.get('dividend_yield')
        if div_yield is None or div_yield < 3:
            continue

        # 재고/매출 비율 < 5% (필수)
        inv_ratio = get_inventory_ratio(fin)
        if inv_ratio is None or inv_ratio > 5:
            continue

        pbr = get_pbr_from_data(market_cap, fin)

        results.append({
            'ticker': code,
            'name': name,
            'current_price': int(row.get('종가', 0) or 0),
            'market_cap': round(market_cap, 0),
            'per': round(per, 1),
            'peg': round(peg, 2),
            'roe': round(roe, 1),
            'roa': round(roa, 1),
            'pbr': round(pbr, 2) if pbr is not None else None,
            'debt_ratio': round(dr, 1),
            'dividend_yield': round(div_yield, 2) if div_yield is not None else None,
            'inventory_ratio': round(inv_ratio, 1) if inv_ratio is not None else None,
        })

    # PEG 오름차순 정렬 (낮을수록 좋음)
    results.sort(key=lambda x: x['peg'])

    _save_result('lynch.json', 'lynch', '피터 린치 전략',
                 'PEG·ROE·부채비율 기반 합리적 성장주 전략',
                 ['PER < 25', '0 < PEG < 1.8', '부채비율 < 150%',
                  'ROE > 5%', 'ROA > 1%', '배당수익률 > 3%',
                  '재고/매출 < 5%'],
                 results)

    _print_result('피터 린치 전략', results, [
        {'key': 'name', 'label': '종목명', 'width': 14},
        {'key': 'peg', 'label': 'PEG', 'width': 7, 'fmt': '.2f'},
        {'key': 'per', 'label': 'PER', 'width': 8, 'fmt': '.1f'},
        {'key': 'roe', 'label': 'ROE%', 'width': 8, 'fmt': '.1f'},
        {'key': 'roa', 'label': 'ROA%', 'width': 8, 'fmt': '.1f'},
        {'key': 'debt_ratio', 'label': '부채비율', 'width': 8, 'fmt': '.1f'},
        {'key': 'dividend_yield', 'label': '배당률%', 'width': 8, 'fmt': '.2f'},
    ])

    return results


# ============================================================================
# 벤저민 그레이엄 -가치투자 전략
# ============================================================================

def screen_graham(filtered, fnguide_cache):
    """
    벤저민 그레이엄 전략: 안전마진 중심의 보수적 가치투자

    조건:
    - 매출 > 1000억
    - 유동비율 > 200% (= 유동자산/유동부채 > 2)
    - 순유동자산 > 장기부채
    - EPS 누적 성장률 > 30% (가용 연간 데이터 기준, ~3-4년)
    - 전 기간 흑자 (순이익 + 영업이익 모두 > 0, ~4년)
    - PER < 15
    - PBR × PER < 22
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

        # 유동비율 > 2 (200%)
        cr = get_current_ratio(fin)
        if cr is None or cr < 2:
            continue

        # 순유동자산 > 장기부채
        nca = get_net_current_assets(fin)
        bs = fin.get('balance', {})
        long_debt = (bs.get('long_term_debt', 0) or 0) + (bs.get('bonds', 0) or 0)
        if nca is None or nca <= long_debt:
            continue

        # 전 기간 흑자 (순이익 + 영업이익 모두 > 0) + 누적 성장률 > 30%
        annual_ni = fin.get('annual', {}).get('net_income', {})
        sorted_annual = sorted(annual_ni.items(), key=lambda x: x[0], reverse=True)
        annual_values = [v for _, v in sorted_annual]

        if len(annual_values) < 2:
            continue

        # 순이익 모두 흑자
        if not all(v is not None and not np.isnan(v) and v > 0 for v in annual_values):
            continue

        # 영업이익 모두 흑자
        annual_oi = fin.get('annual', {}).get('operating_income', {})
        oi_values = [v for _, v in sorted(annual_oi.items(), key=lambda x: x[0], reverse=True)]
        if not oi_values or not all(v is not None and not np.isnan(v) and v > 0 for v in oi_values):
            continue

        # 누적 성장률 (가용 연간 데이터 기준)
        oldest = annual_values[-1] if len(annual_values) >= 2 else None
        newest = annual_values[0]
        if oldest is not None and oldest > 0:
            cumulative_growth = ((newest / oldest) - 1) * 100
        else:
            cumulative_growth = None

        if cumulative_growth is None or cumulative_growth < 30:
            continue

        # PER < 15
        per = get_per_from_data(market_cap, fin)
        if per is None or per <= 0 or per > 15:
            continue

        # PBR * PER < 22
        pbr = get_pbr_from_data(market_cap, fin)
        if pbr is None or pbr <= 0:
            continue
        if per * pbr > 22:
            continue

        dr = get_debt_ratio(fin)
        header = fin.get('header', {})
        div_yield = header.get('dividend_yield')

        results.append({
            'ticker': code,
            'name': name,
            'current_price': int(row.get('종가', 0) or 0),
            'market_cap': round(market_cap, 0),
            'per': round(per, 1),
            'pbr': round(pbr, 2),
            'per_x_pbr': round(per * pbr, 2),
            'current_ratio': round(cr, 2),
            'net_current_assets': round(nca, 0),
            'cumulative_growth': round(cumulative_growth, 1),
            'debt_ratio': round(dr, 1) if dr is not None else None,
            'dividend_yield': div_yield,
            'revenue_ttm': round(rev_ttm, 0),
        })

    # PER×PBR 오름차순 (낮을수록 좋음)
    results.sort(key=lambda x: x['per_x_pbr'])

    _save_result('graham.json', 'graham', '벤저민 그레이엄 전략',
                 '유동비율·PER×PBR·흑자 기반 보수적 가치투자',
                 ['PER < 15', 'PBR×PER < 22', '유동비율 > 200%',
                  '순유동자산 > 장기부채', '전 기간 흑자(순이익+영업이익)',
                  'EPS 누적 성장 > 30%', '매출 > 1000억'],
                 results)

    _print_result('벤저민 그레이엄 전략', results, [
        {'key': 'name', 'label': '종목명', 'width': 14},
        {'key': 'per', 'label': 'PER', 'width': 8, 'fmt': '.1f'},
        {'key': 'pbr', 'label': 'PBR', 'width': 7, 'fmt': '.2f'},
        {'key': 'per_x_pbr', 'label': 'PxB', 'width': 7, 'fmt': '.2f'},
        {'key': 'current_ratio', 'label': '유동비율', 'width': 8, 'fmt': '.2f'},
        {'key': 'cumulative_growth', 'label': '5Y성장', 'width': 8, 'fmt': '.1f'},
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
