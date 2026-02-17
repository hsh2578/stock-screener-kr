"""
마법공식 스크리너 (Magic Formula - 조엘 그린블라트)

딱 2개 지표만 사용하는 단순하고 강력한 전략:
1. 이익수익률 (Earnings Yield) = EBIT / EV
2. 투하자본수익률 (ROC) = EBIT / Invested Capital

각 지표별 순위를 매기고, 순위 합산으로 종합 순위를 산출합니다.

사용법:
    python magic_formula.py
"""

import json
import os
import sys
import time
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

# 프로젝트 루트 설정
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from scripts.krx_data import get_stock_master, is_financial_stock
from scripts.fnguide_data import (
    get_financial_data,
    get_ebit,
    get_invested_capital,
    get_ev,
    get_per_from_data,
    get_pbr_from_data,
    load_fnguide_cache,
)

# ============================================================================
# 상수
# ============================================================================

DATA_PATH = os.path.join(SCRIPT_DIR, 'data')
OUTPUT_FILE = os.path.join(DATA_PATH, 'magic_formula.json')

MIN_MARKET_CAP = 1000  # 시가총액 1,000억 이상
TOP_N = 20  # 상위 20개


# ============================================================================
# 메인 로직
# ============================================================================

def run_magic_formula():
    """마법공식 스크리너 실행"""
    print("=" * 60)
    print("마법공식 스크리너 (조엘 그린블라트)")
    print(f"실행 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    start_time = time.time()

    # 1. KRX 종목 마스터 수집
    print("\n[1/4] KRX 종목 마스터 수집...")
    master = get_stock_master()

    # 2. 필터링
    print("\n[2/4] 종목 필터링...")
    filtered = master[
        (master['시가총액'] >= MIN_MARKET_CAP) &
        (master['is_common'] == True) &
        (master['is_spac'] == False) &
        (master['is_reit'] == False)
    ].copy()

    # 금융주 제외
    if '업종' in filtered.columns:
        filtered = filtered[
            ~filtered.apply(
                lambda row: is_financial_stock(
                    str(row.get('업종', '')),
                    str(row.get('종목명', ''))
                ),
                axis=1
            )
        ]

    print(f"  필터링 후: {len(filtered)}개 종목 (금융주/우선주/스팩/리츠 제외)")

    # 3. FnGuide 재무데이터 (캐시 → fallback 개별 수집)
    print(f"\n[3/4] FnGuide 재무데이터 로드 ({len(filtered)}개 종목)...")

    fnguide_cache = load_fnguide_cache()
    if fnguide_cache:
        print(f"  캐시 사용: {len(fnguide_cache)}개 종목")
    else:
        print("  캐시 없음 — 개별 수집 모드 (느림)")
        fnguide_cache = {}

    results = []
    success = 0
    skipped = 0

    for _, row in filtered.iterrows():
        code = row['종목코드']
        name = row['종목명']
        market_cap = row.get('시가총액', 0)
        current_price = row.get('종가', 0) or 0

        fin_data = fnguide_cache.get(code)
        if not fin_data:
            fin_data = get_financial_data(code)
        if not fin_data:
            skipped += 1
            continue

        ebit = get_ebit(fin_data)
        if ebit is None or ebit <= 0:
            skipped += 1
            continue

        invested_capital = get_invested_capital(fin_data)
        if invested_capital is None or invested_capital <= 0:
            skipped += 1
            continue

        ev = get_ev(market_cap, fin_data)
        if ev is None or ev <= 0:
            skipped += 1
            continue

        earnings_yield = (ebit / ev) * 100
        roc = (ebit / invested_capital) * 100

        per = get_per_from_data(market_cap, fin_data)
        pbr = get_pbr_from_data(market_cap, fin_data)

        results.append({
            'ticker': code,
            'name': name,
            'current_price': int(current_price),
            'per': round(per, 1) if per is not None else None,
            'pbr': round(pbr, 2) if pbr is not None else None,
            'earnings_yield': round(earnings_yield, 2),
            'roc': round(roc, 2),
            'ebit': round(ebit, 0),
            'market_cap': round(market_cap, 0),
            'ev': round(ev, 0),
            'invested_capital': round(invested_capital, 0),
        })
        success += 1

        done = success + skipped
        if done % 200 == 0 or done == len(filtered):
            print(f"  진행: {done}/{len(filtered)} (성공: {success}, 스킵: {skipped})")

    print(f"  완료: {success}개 성공, {skipped}개 스킵")

    if not results:
        print("결과가 없습니다.")
        return

    # 4. 순위 계산
    print(f"\n[4/4] 순위 계산 ({len(results)}개 종목)...")
    df = pd.DataFrame(results)

    # 이익수익률 순위 (높을수록 좋음 → 순위 1)
    df['ey_rank'] = df['earnings_yield'].rank(ascending=False, method='min')
    # ROC 순위 (높을수록 좋음 → 순위 1)
    df['roc_rank'] = df['roc'].rank(ascending=False, method='min')
    # 종합 순위 (합산, 낮을수록 좋음)
    df['combined_rank'] = df['ey_rank'] + df['roc_rank']

    # 종합 순위 정렬
    df = df.sort_values('combined_rank').reset_index(drop=True)

    # Top N 선정
    top = df.head(TOP_N)

    # 결과 JSON 생성
    output_data = []
    for i, row in top.iterrows():
        output_data.append({
            'rank': i + 1,
            'ticker': row['ticker'],
            'name': row['name'],
            'current_price': int(row['current_price']),
            'per': row['per'] if pd.notna(row.get('per')) else None,
            'pbr': row['pbr'] if pd.notna(row.get('pbr')) else None,
            'combined_rank': int(row['combined_rank']),
            'ey_rank': int(row['ey_rank']),
            'roc_rank': int(row['roc_rank']),
            'earnings_yield': row['earnings_yield'],
            'roc': row['roc'],
            'ebit': row['ebit'],
            'market_cap': row['market_cap'],
            'ev': row['ev'],
        })

    # 저장
    os.makedirs(DATA_PATH, exist_ok=True)
    output = {
        'meta': {
            'type': 'magic-formula',
            'name': '마법공식 (그린블라트)',
            'description': '이익수익률 + 투하자본수익률 순위 합산',
            'lastUpdated': datetime.now().isoformat(),
            'totalCount': len(output_data),
            'screened_from': len(df),
        },
        'data': output_data,
    }

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    elapsed = time.time() - start_time

    # 결과 출력
    print(f"\n{'=' * 80}")
    print(f"마법공식 Top {TOP_N}")
    print(f"{'=' * 80}")
    print(f"{'순위':<4} {'종목명':<12} {'종합':>6} {'EY순위':>6} {'ROC순위':>6} {'EY%':>8} {'ROC%':>8}")
    print(f"{'-' * 70}")

    for item in output_data:
        print(
            f"{item['rank']:<4} {item['name']:<12} "
            f"{item['combined_rank']:>6} {item['ey_rank']:>6} {item['roc_rank']:>6} "
            f"{item['earnings_yield']:>7.1f}% {item['roc']:>7.1f}%"
        )

    print(f"\n저장: {OUTPUT_FILE}")
    print(f"총 실행 시간: {elapsed:.1f}초 ({elapsed/60:.1f}분)")
    print("=" * 60)


if __name__ == '__main__':
    run_magic_formula()
