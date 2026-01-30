"""
60주선 우량주 스크리너

60주선(약 300일) 근처에서 지지받는 우량주를 발굴합니다.
저평가 우량주 스크리너의 재무 데이터를 활용합니다.

[대상]
    코스피 200 종목

[조건]
    A. 60주선 이격도: 100% ~ 110% (60주선 위 0~10% 구간)
    B. 종가 ≥ 60주 지수이평 (이중 확인)
    C. 영업이익률: 최근 결산 10% 이상 (수익성)
    D. 영업이익 성장률: 최근 3년 평균 10% 이상 (성장성)

[실행]
    토요일 저평가 우량주 스크리너와 함께 실행
"""

import FinanceDataReader as fdr
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, asdict

# 상수
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, 'data')
FINANCIAL_DATA_PATH = os.path.join(SCRIPT_DIR, 'data', 'financial_data.json')

MA60W_PERIOD = 300  # 60주 × 5일 = 300 거래일
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


def get_kospi200_stocks() -> pd.DataFrame:
    """코스피 200 종목 리스트 가져오기"""
    print("코스피 200 종목 조회 중...")

    try:
        # 코스피 전체에서 시가총액 상위 200개 선택 (코스피200 근사)
        kospi = fdr.StockListing('KOSPI')

        if 'Marcap' in kospi.columns:
            kospi['MarketCap'] = kospi['Marcap'] / 100000000  # 원 -> 억원
        elif 'MarketCap' in kospi.columns:
            kospi['MarketCap'] = kospi['MarketCap'] / 100000000

        # 시가총액 상위 200개
        kospi = kospi.sort_values('MarketCap', ascending=False).head(200)

        print(f"  코스피 상위 200개 종목 선택")
        return kospi

    except Exception as e:
        print(f"  코스피 200 조회 실패: {e}")
        return pd.DataFrame()


def load_financial_data() -> Dict:
    """저평가 우량주 스크리너의 재무 데이터 로드"""
    if not os.path.exists(FINANCIAL_DATA_PATH):
        print(f"  재무 데이터 없음: {FINANCIAL_DATA_PATH}")
        return {}

    try:
        with open(FINANCIAL_DATA_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # data 구조: {"meta": {...}, "data": {"종목코드": {...}, ...}}
        fin_data = data.get('data', {})
        print(f"  재무 데이터 로드: {len(fin_data)}개 종목")
        return fin_data
    except Exception as e:
        print(f"  재무 데이터 로드 실패: {e}")
        return {}


def calculate_ema(prices: pd.Series, period: int) -> pd.Series:
    """지수이동평균 계산"""
    return prices.ewm(span=period, adjust=False).mean()


def get_stock_price_data(ticker: str, days: int = 400) -> Optional[pd.DataFrame]:
    """주가 데이터 조회"""
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days + 100)

        df = fdr.DataReader(ticker, start_date.strftime('%Y-%m-%d'))
        if df is not None and len(df) >= MA60W_PERIOD:
            return df
    except:
        pass
    return None


def extract_financial_metrics(fin_data: Dict) -> Tuple[Optional[float], Optional[float]]:
    """
    재무 데이터에서 영업이익률, 영업이익 성장률 추출

    Returns:
        (영업이익률, 영업이익 성장률 3년)
    """
    metrics = fin_data.get('metrics', {})

    # 영업이익률 (최근)
    op_margin_list = metrics.get('operating_margin', [])
    op_margin = None
    if op_margin_list:
        valid_margins = [m for m in op_margin_list if m is not None]
        if valid_margins:
            op_margin = valid_margins[0]

    # 영업이익 성장률 3년 평균
    op_growth_list = metrics.get('op_profit_growth_rate', [])
    op_growth_3y = None
    if op_growth_list and len(op_growth_list) >= 3:
        valid_growth = [g for g in op_growth_list[:3] if g is not None]
        if valid_growth:
            op_growth_3y = sum(valid_growth) / len(valid_growth)
    elif op_growth_list and len(op_growth_list) >= 1:
        valid_growth = [g for g in op_growth_list if g is not None]
        if valid_growth:
            op_growth_3y = valid_growth[0]

    return op_margin, op_growth_3y


def screen_ma60w_quality() -> List[Dict]:
    """60주선 우량주 스크리너 실행"""
    print("\n[60주선 우량주 스크리너] 분석 시작...")

    # 1. 코스피 200 종목 가져오기
    stocks = get_kospi200_stocks()
    if stocks.empty:
        print("  종목 조회 실패")
        return []

    # 2. 재무 데이터 로드
    financial_data = load_financial_data()

    results = []
    stats = {
        'total': len(stocks),
        'no_price': 0,
        'no_financial': 0,
        'gap_fail': 0,
        'op_margin_fail': 0,
        'op_growth_fail': 0,
        'passed': 0
    }

    print(f"  {len(stocks)}개 종목 스캔...")

    for idx, row in stocks.iterrows():
        ticker = row.get('Code', row.get('Symbol', ''))
        name = row.get('Name', '')
        market_cap = row.get('MarketCap', 0)
        sector = row.get('Sector', row.get('Industry', ''))

        if not ticker:
            continue

        # 진행률 표시
        if (stats['no_price'] + stats['no_financial'] + stats['gap_fail'] +
            stats['op_margin_fail'] + stats['op_growth_fail'] + stats['passed'] + 1) % 50 == 0:
            print(f"  진행 중... {stats['no_price'] + stats['no_financial'] + stats['gap_fail'] + stats['op_margin_fail'] + stats['op_growth_fail'] + stats['passed'] + 1}/{len(stocks)}")

        # 주가 데이터 조회
        df = get_stock_price_data(ticker)
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

        # 재무 데이터 확인
        if ticker not in financial_data:
            stats['no_financial'] += 1
            continue

        fin_data = financial_data[ticker]
        op_margin, op_growth_3y = extract_financial_metrics(fin_data)

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

    print(f"\n├─ 총 종목: {stats['total']}개")
    print(f"├─ 주가 데이터 없음: {stats['no_price']}개")
    print(f"├─ 60주선 이격도 필터 (0~10%): {stats['gap_fail']}개 탈락")
    print(f"├─ 재무 데이터 없음: {stats['no_financial']}개")
    print(f"├─ 영업이익률 < 10%: {stats['op_margin_fail']}개 탈락")
    print(f"├─ 영업이익 성장률 < 10%: {stats['op_growth_fail']}개 탈락")
    print(f"[완료] 60주선 우량주: {len(results)}개")

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
    print("=" * 60)

    # 스크리닝 실행
    results = screen_ma60w_quality()

    # 결과 저장
    save_results(results)

    # 결과 출력
    print("\n" + "=" * 70)
    print("60주선 우량주 스크리닝 결과")
    print("=" * 70)
    print(f"{'순위':<4} {'종목명':<12} {'현재가':>10} {'60주선이격':>10} {'영업이익률':>10} {'성장률':>8}")
    print("-" * 70)

    for i, r in enumerate(results[:20]):
        print(f"{i+1:<4} {r['name']:<12} {r['current_price']:>10,}원 {r['ma60w_gap']:>9.1f}% {r['op_margin']:>9.1f}% {r['op_growth_3y']:>7.1f}%")

    print(f"\n총 {len(results)}개 종목")


if __name__ == '__main__':
    main()
