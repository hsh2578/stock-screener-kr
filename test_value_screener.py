"""
저평가 우량주 스크리너 (TTM 버전)
TTM(Trailing Twelve Months) 기반으로 최근 12개월 실적을 분석합니다.
"""

import FinanceDataReader as fdr
import json
import os
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from naver_finance import (
    fetch_financial_data_ttm,
    pass_first_filter_ttm,
    save_financial_data,
    TREASURY_RATE
)
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, 'data')


def get_top_stocks(n: int = 20, min_market_cap: int = 2000) -> list:
    """시가총액 상위 N개 종목 가져오기"""
    print(f"종목 리스트 조회 중...")

    # KOSPI + KOSDAQ
    kospi = fdr.StockListing('KOSPI')
    kospi['Market'] = 'KOSPI'
    kosdaq = fdr.StockListing('KOSDAQ')
    kosdaq['Market'] = 'KOSDAQ'

    stocks = kospi._append(kosdaq, ignore_index=True)

    # 시가총액 계산
    if 'Marcap' in stocks.columns:
        stocks['MarketCap'] = stocks['Marcap'] / 100000000  # 원 -> 억원

    # 필터링 및 정렬
    stocks = stocks[stocks['MarketCap'] >= min_market_cap].copy()
    stocks = stocks.sort_values('MarketCap', ascending=False)

    print(f"  시총 {min_market_cap}억 이상: {len(stocks)}개 종목")
    print(f"  상위 {n}개 선택")

    top = stocks.head(n)

    return [
        {'Code': row['Code'], 'Name': row['Name'], 'MarketCap': row['MarketCap']}
        for _, row in top.iterrows()
    ]


def fetch_single_stock_ttm(stock: dict) -> tuple:
    """단일 종목 TTM 크롤링 (병렬 처리용)"""
    code = stock['Code']
    name = stock['Name']

    data = fetch_financial_data_ttm(code)
    if data and data.get('metrics'):
        data['name'] = name
        data['market_cap'] = stock['MarketCap']
        return code, data
    return code, None


def crawl_stocks_ttm(stocks: list, max_workers: int = 10) -> dict:
    """종목들의 TTM 재무데이터 병렬 크롤링"""
    print(f"\nTTM 재무데이터 병렬 크롤링 시작 ({len(stocks)}개 종목, {max_workers}개 스레드)")

    results = {}
    success = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_single_stock_ttm, stock): stock for stock in stocks}

        for i, future in enumerate(as_completed(futures)):
            code, data = future.result()
            if data:
                results[code] = data
                success += 1
            else:
                failed += 1

            # 진행 상황 출력 (100개마다)
            if (i + 1) % 100 == 0 or (i + 1) == len(stocks):
                print(f"  진행: {i+1}/{len(stocks)} (성공: {success}, 실패: {failed})")

    print(f"\n크롤링 완료: 성공 {success}개, 실패 {failed}개")
    return results


def save_results(results: list, filename: str):
    """결과를 JSON으로 저장"""
    os.makedirs(DATA_PATH, exist_ok=True)

    output = {
        'meta': {
            'updated_at': datetime.now().isoformat(),
            'total_count': len(results),
            'treasury_rate': TREASURY_RATE,
            'data_type': 'TTM'
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
    print("=" * 60)

    # 1. 시가총액 1000억 이상 전체 종목 가져오기
    stocks = get_top_stocks(n=9999, min_market_cap=1000)

    # 2. TTM 데이터 크롤링
    financial_data = crawl_stocks_ttm(stocks)

    # 3. 재무데이터 저장
    save_financial_data(financial_data)

    # 4. 스크리닝 (TTM 6개 조건 모두 충족)
    print("\nTTM 스크리닝 시작 (6개 조건 모두 충족)...")

    results = []
    for stock in stocks:
        code = stock['Code']
        if code not in financial_data:
            continue

        fin_data = financial_data[code]
        metrics = fin_data.get('metrics', {})

        if not metrics:
            continue

        # TTM 1차 필터 적용
        passed, first_filter_results = pass_first_filter_ttm(fin_data)
        if not passed:
            continue

        # 지표 추출
        per = metrics.get('per', [None])[0] if metrics.get('per') else None
        pbr = metrics.get('pbr', [None])[0] if metrics.get('pbr') else None
        quarters = fin_data.get('quarters', [])

        results.append({
            'ticker': code,
            'name': fin_data.get('name', stock['Name']),
            'market_cap': round(stock['MarketCap'], 0),
            'per': round(per, 2) if per else None,
            'pbr': round(pbr, 2) if pbr else None,
            'revenue_growth_3y': first_filter_results.get('revenue_growth', {}).get('value'),
            'op_margin_avg': first_filter_results.get('operating_margin_avg', {}).get('value'),
            'op_growth_5y': first_filter_results.get('operating_profit_growth', {}).get('value'),
            'eps_growth_5y': first_filter_results.get('eps_growth', {}).get('value'),
            'net_income_ttm': first_filter_results.get('net_income_positive', {}).get('value'),
            'quarters': quarters[-1] if quarters else None  # 가장 최근 분기
        })

    # 시가총액 순 정렬
    results.sort(key=lambda x: x['market_cap'], reverse=True)

    # 5. 결과 저장
    save_results(results, 'value_stocks.json')

    # 6. 결과 출력
    print("\n" + "=" * 80)
    print("TTM 스크리닝 결과 (6개 조건 모두 충족)")
    print("=" * 80)
    print(f"{'순위':<4} {'종목명':<12} {'시가총액':>10} {'PER':>6} {'최근분기':>10} {'영업이익률':>10}")
    print("-" * 80)

    for i, r in enumerate(results[:20]):
        mc = f"{r['market_cap']:,.0f}억"
        per = f"{r['per']:.1f}" if r['per'] else "-"
        quarter = r.get('quarters', '-') or '-'
        op_margin = f"{r['op_margin_avg']:.1f}%" if r['op_margin_avg'] else "-"
        print(f"{i+1:<4} {r['name']:<12} {mc:>10} {per:>6} {quarter:>10} {op_margin:>10}")

    print(f"\n총 {len(results)}개 종목 스크리닝 완료 (TTM 기준)")


if __name__ == '__main__':
    main()
