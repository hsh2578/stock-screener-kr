"""
저평가 우량주 스크리너 (TTM 버전)
TTM(Trailing Twelve Months) 기반으로 최근 12개월 실적을 분석합니다.

FnGuide 공유 캐시를 사용하여 개별 크롤링 없이 빠르게 실행합니다.
캐시가 없으면 fallback으로 개별 수집합니다.
"""

import json
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

# 프로젝트 루트 설정
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from scripts.krx_data import get_stock_master
from scripts.fnguide_data import (
    load_fnguide_cache,
    get_financial_data,
    get_growth_rates_with_ttm,
    get_operating_margins,
    get_per_from_data,
    get_pbr_from_data,
)

# 상수
DATA_PATH = os.path.join(SCRIPT_DIR, 'data')
FINANCIAL_DATA_FILE = os.path.join(DATA_PATH, 'financial_data.json')
TREASURY_RATE = 3.4  # 국채금리 (호환성 유지)
MIN_MARKET_CAP = 1000  # 시가총액 1,000억 이상


def save_financial_data(fnguide_cache: Dict[str, Dict], master: 'pd.DataFrame' = None) -> None:
    """
    FnGuide 캐시 데이터를 financial_data.json으로 변환 저장.
    PER/PBR 포함 — HTML에서 모든 스크리너의 PER/PBR 표시에 사용.
    """
    os.makedirs(DATA_PATH, exist_ok=True)

    # KRX 마스터에서 시가총액 조회용 딕셔너리
    market_cap_map = {}
    if master is not None:
        for _, row in master.iterrows():
            code = row.get('종목코드', '')
            cap = row.get('시가총액', 0)
            if code and cap:
                market_cap_map[code] = cap

    converted = {}
    for code, data in fnguide_cache.items():
        metrics = {}

        # PER/PBR (시가총액 + FnGuide TTM 데이터 기반 계산)
        market_cap = market_cap_map.get(code, 0)
        if market_cap > 0:
            per = get_per_from_data(market_cap, data)
            pbr = get_pbr_from_data(market_cap, data)
            if per is not None:
                metrics['per'] = [round(per, 2)]
            if pbr is not None:
                metrics['pbr'] = [round(pbr, 2)]

        # 영업이익률
        margins = get_operating_margins(data)
        if margins:
            metrics['operating_margin'] = margins

        # 영업이익 성장률
        op_growth = get_growth_rates_with_ttm(data, 'operating_income')
        if op_growth:
            metrics['op_profit_growth_rate'] = op_growth

        converted[code] = {
            'code': code,
            'metrics': metrics,
        }

    output = {
        'meta': {
            'updated_at': datetime.now().isoformat(),
            'total_count': len(converted),
            'treasury_rate': TREASURY_RATE,
        },
        'data': converted,
    }

    with open(FINANCIAL_DATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    per_count = sum(1 for v in converted.values() if 'per' in v['metrics'])
    print(f"financial_data.json 저장: {len(converted)}개 종목 (PER/PBR: {per_count}개)")


def pass_first_filter_ttm(fin_data: Dict, market_cap: float) -> Tuple[bool, Dict[str, Any]]:
    """
    TTM 기반 1차 필터 (6개 조건)

    조건:
    1. PER: 3 < PER < 30
    2. 매출액 성장률: 3년 평균 > 10%
    3. 영업이익률: 5년 평균 > 10%
    4. 영업이익 성장률: 5년 평균 > 10%
    5. EPS(순이익) 성장률: 5년 평균 > 10%
    6. 순이익 증가율: 20% < 5년 평균 < 50%
    """
    results = {
        'per_check': {'value': None, 'pass': False, 'condition': '3 < PER < 30'},
        'revenue_growth': {'value': None, 'pass': False, 'condition': '매출성장률 3년평균 > 10%'},
        'operating_margin_avg': {'value': None, 'pass': False, 'condition': '영업이익률 5년평균 > 10%'},
        'operating_profit_growth': {'value': None, 'pass': False, 'condition': '영업이익성장률 5년평균 > 10%'},
        'eps_growth': {'value': None, 'pass': False, 'condition': '순이익성장률 5년평균 > 10%'},
        'net_income_growth': {'value': None, 'pass': False, 'condition': '순이익증가율 20~50%'},
    }

    # 1. PER: 3 < PER < 30
    per = get_per_from_data(market_cap, fin_data)
    if per is not None:
        results['per_check']['value'] = round(per, 2)
        results['per_check']['pass'] = 3 < per < 30

    # 2. 매출액 성장률: 3년 평균 > 10%
    rev_growth = get_growth_rates_with_ttm(fin_data, 'revenue', years=5)
    if rev_growth:
        valid_rates = [r for r in rev_growth[:3] if r is not None]
        if valid_rates:
            avg = sum(valid_rates) / len(valid_rates)
            results['revenue_growth']['value'] = round(avg, 2)
            results['revenue_growth']['pass'] = avg > 10

    # 3. 영업이익률: 5년 평균 > 10%
    margins = get_operating_margins(fin_data, years=5)
    if margins:
        valid_margins = [m for m in margins[:5] if m is not None]
        if valid_margins:
            avg = sum(valid_margins) / len(valid_margins)
            results['operating_margin_avg']['value'] = round(avg, 2)
            results['operating_margin_avg']['pass'] = avg > 10

    # 4. 영업이익 성장률: 5년 평균 > 10%
    op_growth = get_growth_rates_with_ttm(fin_data, 'operating_income', years=5)
    if op_growth:
        valid_rates = [r for r in op_growth[:5] if r is not None]
        if valid_rates:
            avg = sum(valid_rates) / len(valid_rates)
            results['operating_profit_growth']['value'] = round(avg, 2)
            results['operating_profit_growth']['pass'] = avg > 10

    # 5. EPS(순이익) 성장률: 5년 평균 > 10% (순이익 성장률로 대체)
    ni_growth = get_growth_rates_with_ttm(fin_data, 'net_income', years=5)
    if ni_growth:
        valid_rates = [r for r in ni_growth[:5] if r is not None]
        if valid_rates:
            avg = sum(valid_rates) / len(valid_rates)
            results['eps_growth']['value'] = round(avg, 2)
            results['eps_growth']['pass'] = avg > 10

    # 6. 순이익 증가율: 20% < 5년 평균 < 50%
    if ni_growth:
        valid_rates = [r for r in ni_growth[:5] if r is not None]
        if valid_rates:
            avg = sum(valid_rates) / len(valid_rates)
            results['net_income_growth']['value'] = round(avg, 2)
            results['net_income_growth']['pass'] = 20 < avg < 50

    # 통과 여부: 6개 조건 모두 충족
    pass_count = sum(1 for r in results.values() if r['pass'])
    all_pass = pass_count == 6

    return all_pass, results


def save_results(results: list, filename: str):
    """결과를 JSON으로 저장"""
    os.makedirs(DATA_PATH, exist_ok=True)

    output = {
        'meta': {
            'updated_at': datetime.now().isoformat(),
            'total_count': len(results),
            'treasury_rate': TREASURY_RATE,
            'data_type': 'TTM (FnGuide)'
        },
        'data': results
    }

    filepath = os.path.join(DATA_PATH, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"저장: {filepath}")


def main():
    print("=" * 60)
    print("저평가 우량주 스크리너 (TTM 버전)")
    print(f"실행 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    start_time = time.time()

    # 1. KRX 종목 마스터 수집
    print("\n[1/4] KRX 종목 마스터 수집...")
    master = get_stock_master()

    # 필터링: 시총 1000억+, 보통주, 스팩/리츠 제외
    filtered = master[
        (master['시가총액'] >= MIN_MARKET_CAP) &
        (master['is_common'] == True) &
        (master['is_spac'] == False) &
        (master['is_reit'] == False)
    ].copy()

    print(f"  필터링 후: {len(filtered)}개 종목")

    # 2. FnGuide 재무데이터 로드 (캐시 → fallback 개별 수집)
    print("\n[2/4] FnGuide 재무데이터 로드...")
    fnguide_cache = load_fnguide_cache()
    if fnguide_cache:
        print(f"  캐시 사용: {len(fnguide_cache)}개 종목")
    else:
        print("  캐시 없음 — 개별 수집 모드 (느림)")
        fnguide_cache = {}

    # 캐시에 없는 종목 개별 수집
    missing_codes = [
        row['종목코드'] for _, row in filtered.iterrows()
        if row['종목코드'] not in fnguide_cache
    ]
    if missing_codes:
        print(f"  캐시 미포함 {len(missing_codes)}개 종목 개별 수집 중...")
        for i, code in enumerate(missing_codes):
            data = get_financial_data(code)
            if data:
                fnguide_cache[code] = data
            if (i + 1) % 100 == 0:
                print(f"    진행: {i+1}/{len(missing_codes)}")

    # 3. financial_data.json 저장 (PER/PBR 포함)
    print("\n[3/4] financial_data.json 저장 (PER/PBR 포함)...")
    save_financial_data(fnguide_cache, master)

    # 4. 스크리닝 (TTM 6개 조건 모두 충족)
    print("\n[4/4] TTM 스크리닝 시작 (6개 조건 모두 충족)...")

    results = []
    screened = 0

    for _, row in filtered.iterrows():
        code = row['종목코드']
        name = row['종목명']
        market_cap = row.get('시가총액', 0)
        close = row.get('종가', 0)

        fin_data = fnguide_cache.get(code)
        if not fin_data:
            continue

        # TTM 데이터 존재 확인
        if not fin_data.get('revenue_ttm') and not fin_data.get('operating_income_ttm'):
            continue

        # TTM 1차 필터 적용
        passed, first_filter_results = pass_first_filter_ttm(fin_data, market_cap)
        screened += 1

        if not passed:
            continue

        # 지표 추출
        per = get_per_from_data(market_cap, fin_data)
        pbr = get_pbr_from_data(market_cap, fin_data)

        # TTM 순이익
        net_income_ttm = fin_data.get('net_income_ttm')

        results.append({
            'ticker': code,
            'name': name,
            'current_price': int(close) if close else 0,
            'market_cap': round(market_cap, 0),
            'per': round(per, 2) if per else None,
            'pbr': round(pbr, 2) if pbr else None,
            'revenue_growth_3y': first_filter_results.get('revenue_growth', {}).get('value'),
            'op_margin_avg': first_filter_results.get('operating_margin_avg', {}).get('value'),
            'op_growth_5y': first_filter_results.get('operating_profit_growth', {}).get('value'),
            'eps_growth_5y': first_filter_results.get('eps_growth', {}).get('value'),
            'net_income_growth_5y': first_filter_results.get('net_income_growth', {}).get('value'),
            'net_income_ttm': round(net_income_ttm, 0) if net_income_ttm else None,
            'data_year': 'TTM'
        })

    # 시가총액 순 정렬
    results.sort(key=lambda x: x['market_cap'], reverse=True)

    # 결과 저장
    save_results(results, 'value_stocks.json')

    elapsed = time.time() - start_time

    # 결과 출력
    print("\n" + "=" * 80)
    print(f"저평가 우량주 스크리닝 결과 (6개 조건 모두 충족, TTM 기준)")
    print("=" * 80)
    print(f"{'순위':<4} {'종목명':<12} {'시가총액':>10} {'PER':>6} {'영업이익률':>10}")
    print("-" * 80)

    for i, r in enumerate(results[:20]):
        mc = f"{r['market_cap']:,.0f}억"
        per = f"{r['per']:.1f}" if r['per'] else "-"
        op_margin = f"{r['op_margin_avg']:.1f}%" if r['op_margin_avg'] else "-"
        print(f"{i+1:<4} {r['name']:<12} {mc:>10} {per:>6} {op_margin:>10}")

    print(f"\n총 {len(results)}개 종목 스크리닝 완료 ({screened}개 분석)")
    print(f"총 실행 시간: {elapsed:.1f}초 ({elapsed/60:.1f}분)")


if __name__ == '__main__':
    main()
