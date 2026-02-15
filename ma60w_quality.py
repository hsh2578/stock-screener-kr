"""
60주선 우량주 스크리너

60주선(약 300일) 근처에서 지지받는 우량주를 발굴합니다.
FnGuide 공유 캐시 + OHLCV 캐시를 활용합니다.

[대상]
    코스피 200 종목

[조건]
    A. 60주선 이격도: 100% ~ 110% (60주선 위 0~10% 구간)
    B. 종가 >= 60주 지수이평 (이중 확인)
    C. 영업이익률: 최근 결산 10% 이상 (수익성)
    D. 영업이익 성장률: 최근 3년 평균 10% 이상 (성장성)

[실행]
    value-screener.yml에서 다른 퀀트 스크리너와 함께 실행
"""

import json
import os
import pickle
import sys
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, asdict

import pandas as pd
import numpy as np

# 프로젝트 루트 설정
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from scripts.krx_data import get_stock_master
from scripts.fnguide_data import (
    load_fnguide_cache,
    get_financial_data,
    get_operating_margins,
    get_growth_rates_with_ttm,
)

# 상수
DATA_PATH = os.path.join(SCRIPT_DIR, 'data')
CACHE_PATH = os.path.join(SCRIPT_DIR, '.cache')

MA60W_PERIOD = 300  # 60주 x 5일 = 300 거래일
EMA_PERIOD = 300    # 지수이동평균 기간

# 스크리닝 조건
MIN_GAP_PERCENT = 0.0   # 60주선 대비 최소 이격도 (%)
MAX_GAP_PERCENT = 10.0  # 60주선 대비 최대 이격도 (%)
MIN_OP_MARGIN = 10.0    # 최소 영업이익률 (%)
MIN_OP_GROWTH = 10.0    # 최소 영업이익 성장률 (%)


@dataclass
class MA60WQualityResult:
    """60주선 우량주 스크리너 결과"""
    ticker: str
    name: str
    current_price: int
    ma60w: int              # 60주선 (300일 지수이평)
    ma60w_gap: float        # 60주선 이격도 (%)
    op_margin: float        # 영업이익률
    op_growth_3y: float     # 영업이익 성장률 3년
    market_cap: int         # 시가총액 (억원)
    sector: str             # 업종
    updated_at: str

    def to_dict(self) -> dict:
        return asdict(self)


def _load_ohlcv_cache() -> Dict[str, pd.DataFrame]:
    """OHLCV 캐시에서 데이터 로드 (.cache/stock_data_*.pkl)"""
    cache_data = {}
    if not os.path.exists(CACHE_PATH):
        return cache_data

    pkl_files = sorted(
        [f for f in os.listdir(CACHE_PATH) if f.startswith('stock_data_') and f.endswith('.pkl')],
        reverse=True
    )

    for pkl_file in pkl_files:
        try:
            filepath = os.path.join(CACHE_PATH, pkl_file)
            with open(filepath, 'rb') as f:
                cache_data = pickle.load(f)
            print(f"  OHLCV 캐시 로드: {pkl_file} ({len(cache_data)}개 종목)")
            break
        except Exception:
            continue

    return cache_data


def _get_ohlcv(ticker: str, ohlcv_cache: Dict, days: int = 400) -> Optional[pd.DataFrame]:
    """OHLCV 데이터 조회 (캐시 → FDR fallback)"""
    if ticker in ohlcv_cache:
        df = ohlcv_cache[ticker]
        if df is not None and len(df) >= MA60W_PERIOD:
            return df.tail(days)

    # fallback: 개별 FDR 요청
    try:
        import FinanceDataReader as fdr
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days + 100)
        df = fdr.DataReader(ticker, start_date.strftime('%Y-%m-%d'))
        if df is not None and len(df) >= MA60W_PERIOD:
            return df
    except Exception:
        pass

    return None


def calculate_ema(prices: pd.Series, period: int) -> pd.Series:
    """지수이동평균 계산"""
    return prices.ewm(span=period, adjust=False).mean()


def extract_financial_metrics_from_fnguide(fin_data: Dict) -> Tuple[Optional[float], Optional[float]]:
    """
    FnGuide 데이터에서 영업이익률, 영업이익 성장률 추출

    Returns:
        (영업이익률, 영업이익 성장률 3년)
    """
    # 영업이익률 (최신 = TTM or 최신연간)
    margins = get_operating_margins(fin_data, years=5)
    op_margin = None
    if margins:
        valid = [m for m in margins if m is not None]
        if valid:
            op_margin = valid[0]

    # 영업이익 성장률 3년 평균
    op_growth = get_growth_rates_with_ttm(fin_data, 'operating_income', years=5)
    op_growth_3y = None
    if op_growth and len(op_growth) >= 1:
        valid = [g for g in op_growth[:3] if g is not None]
        if valid:
            op_growth_3y = sum(valid) / len(valid)

    return op_margin, op_growth_3y


def screen_ma60w_quality() -> List[Dict]:
    """60주선 우량주 스크리너 실행"""
    print("\n[60주선 우량주 스크리너] 분석 시작...")

    # 1. KRX 종목 마스터 → 코스피 상위 200개
    print("  종목 리스트 수집...")
    master = get_stock_master()
    kospi = master[master['시장구분'] == 'KOSPI'].copy()
    kospi = kospi.sort_values('시가총액', ascending=False).head(200)
    print(f"  코스피 상위 200개 종목 선택")

    # 2. FnGuide 캐시 로드
    print("  FnGuide 캐시 로드...")
    fnguide_cache = load_fnguide_cache()
    if fnguide_cache:
        print(f"  캐시 사용: {len(fnguide_cache)}개 종목")
    else:
        print("  캐시 없음 — 개별 수집 fallback")
        fnguide_cache = {}

    # 3. OHLCV 캐시 로드
    print("  OHLCV 캐시 로드...")
    ohlcv_cache = _load_ohlcv_cache()
    if not ohlcv_cache:
        print("  OHLCV 캐시 없음 — 개별 다운로드 모드")

    results = []
    stats = {
        'total': len(kospi),
        'no_price': 0,
        'no_financial': 0,
        'gap_fail': 0,
        'op_margin_fail': 0,
        'op_growth_fail': 0,
        'passed': 0
    }

    print(f"  {len(kospi)}개 종목 스캔...")

    for _, row in kospi.iterrows():
        ticker = row['종목코드']
        name = row['종목명']
        market_cap = row.get('시가총액', 0)
        sector = row.get('업종', '')

        if not ticker:
            continue

        # 주가 데이터 조회 (OHLCV 캐시 → FDR fallback)
        df = _get_ohlcv(ticker, ohlcv_cache, 400)
        if df is None or len(df) < MA60W_PERIOD:
            stats['no_price'] += 1
            continue

        # 60주 지수이동평균 계산
        close = df['Close']
        ema60w = calculate_ema(close, EMA_PERIOD)

        current_price = close.iloc[-1]
        ma60w_val = ema60w.iloc[-1]

        if pd.isna(ma60w_val) or ma60w_val <= 0:
            stats['no_price'] += 1
            continue

        # A & B: 60주선 이격도 100%~110% (0~10% 위)
        ma60w_gap = (current_price - ma60w_val) / ma60w_val * 100

        if ma60w_gap < MIN_GAP_PERCENT or ma60w_gap > MAX_GAP_PERCENT:
            stats['gap_fail'] += 1
            continue

        # 재무 데이터 확인 (FnGuide 캐시 → fallback 개별 수집)
        fin_data = fnguide_cache.get(ticker)
        if not fin_data:
            fin_data = get_financial_data(ticker)
        if not fin_data:
            stats['no_financial'] += 1
            continue

        op_margin, op_growth_3y = extract_financial_metrics_from_fnguide(fin_data)

        # C: 영업이익률 10% 이상
        if op_margin is None or op_margin < MIN_OP_MARGIN:
            stats['op_margin_fail'] += 1
            continue

        # D: 영업이익 성장률 3년 평균 10% 이상
        if op_growth_3y is None or op_growth_3y < MIN_OP_GROWTH:
            stats['op_growth_fail'] += 1
            continue

        # 모든 조건 통과
        stats['passed'] += 1

        result = MA60WQualityResult(
            ticker=ticker,
            name=name,
            current_price=int(current_price),
            ma60w=int(ma60w_val),
            ma60w_gap=round(ma60w_gap, 2),
            op_margin=round(op_margin, 1),
            op_growth_3y=round(op_growth_3y, 1),
            market_cap=int(market_cap),
            sector=sector if sector else '',
            updated_at=datetime.now().isoformat()
        )
        results.append(result.to_dict())

    # 시가총액 순 정렬
    results.sort(key=lambda x: x['market_cap'], reverse=True)

    print(f"\n  |- 총 종목: {stats['total']}개")
    print(f"  |- 주가 데이터 없음: {stats['no_price']}개")
    print(f"  |- 60주선 이격도 필터 (0~10%): {stats['gap_fail']}개 탈락")
    print(f"  |- 재무 데이터 없음: {stats['no_financial']}개")
    print(f"  |- 영업이익률 < 10%: {stats['op_margin_fail']}개 탈락")
    print(f"  |- 영업이익 성장률 < 10%: {stats['op_growth_fail']}개 탈락")
    print(f"  [완료] 60주선 우량주: {len(results)}개")

    return results


def save_results(results: List[Dict], filename: str = 'ma60w_quality.json'):
    """결과를 JSON으로 저장"""
    os.makedirs(DATA_PATH, exist_ok=True)

    output = {
        'meta': {
            'updated_at': datetime.now().isoformat(),
            'total_count': len(results),
            'conditions': {
                'ma60w_gap': f'{MIN_GAP_PERCENT}% ~ {MAX_GAP_PERCENT}%',
                'min_op_margin': f'{MIN_OP_MARGIN}%',
                'min_op_growth': f'{MIN_OP_GROWTH}%'
            }
        },
        'data': results
    }

    filepath = os.path.join(DATA_PATH, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"  저장: {filepath}")


def main():
    print("=" * 60)
    print("60주선 우량주 스크리너")
    print(f"실행 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    start_time = time.time()

    # 스크리닝 실행
    results = screen_ma60w_quality()

    # 결과 저장
    save_results(results)

    elapsed = time.time() - start_time

    # 결과 출력
    print("\n" + "=" * 70)
    print("60주선 우량주 스크리닝 결과")
    print("=" * 70)
    print(f"{'순위':<4} {'종목명':<12} {'현재가':>10} {'60주선이격':>10} {'영업이익률':>10} {'성장률':>8}")
    print("-" * 70)

    for i, r in enumerate(results[:20]):
        print(f"{i+1:<4} {r['name']:<12} {r['current_price']:>10,}원 {r['ma60w_gap']:>9.1f}% {r['op_margin']:>9.1f}% {r['op_growth_3y']:>7.1f}%")

    print(f"\n총 {len(results)}개 종목")
    print(f"총 실행 시간: {elapsed:.1f}초 ({elapsed/60:.1f}분)")


if __name__ == '__main__':
    main()
