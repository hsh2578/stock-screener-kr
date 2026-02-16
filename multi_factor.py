"""
멀티팩터 스크리너 (Quality + Value + Momentum)

3대 팩터를 결합한 고급 퀀트 전략:
- Quality: ROE, GPA, CFO/자산
- Value: PER, PBR, PSR, PCR, 배당수익률
- Momentum: 12개월 수익률, K-Ratio

전처리: Winsorizing + 섹터 중립화 (Z-Score)

사용법:
    python multi_factor.py
"""

import json
import os
import pickle
import sys
import time
from datetime import datetime, timedelta
from typing import Optional, Dict

import numpy as np
import pandas as pd
from scipy.stats import zscore, linregress

# 프로젝트 루트 설정
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from scripts.krx_data import get_stock_master, get_sector_data, merge_sector_info
from scripts.fnguide_data import (
    get_financial_data,
    get_roe,
    get_gpa,
    get_cfo_ratio,
    get_psr,
    get_pcr,
    load_fnguide_cache,
)

# OHLCV 캐시 활용을 위한 import
try:
    import FinanceDataReader as fdr
    FDR_AVAILABLE = True
except ImportError:
    FDR_AVAILABLE = False

# ============================================================================
# 상수
# ============================================================================

DATA_PATH = os.path.join(SCRIPT_DIR, 'data')
CACHE_PATH = os.path.join(SCRIPT_DIR, '.cache')
OUTPUT_FILE = os.path.join(DATA_PATH, 'multi_factor.json')

MIN_MARKET_CAP = 1000  # 시가총액 1,000억 이상
TOP_N = 20  # 상위 20개

# 팩터 가중치
WEIGHTS = {
    'quality': 0.33,
    'value': 0.33,
    'momentum': 0.33,
}


# ============================================================================
# Momentum 계산
# ============================================================================

def _load_ohlcv_cache() -> Dict[str, pd.DataFrame]:
    """OHLCV 캐시에서 데이터 로드"""
    cache_data = {}
    if not os.path.exists(CACHE_PATH):
        return cache_data

    # 최신 캐시 파일 찾기
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


def _get_ohlcv(ticker: str, cache: Dict, days: int = 300) -> Optional[pd.DataFrame]:
    """OHLCV 데이터 조회 (캐시 → FDR fallback)"""
    if ticker in cache:
        df = cache[ticker]
        if df is not None and len(df) > 0:
            return df.tail(days)

    if FDR_AVAILABLE:
        try:
            start = (datetime.now() - timedelta(days=days + 50)).strftime('%Y-%m-%d')
            df = fdr.DataReader(ticker, start)
            if df is not None and len(df) > 0:
                return df
        except Exception:
            pass

    return None


def calc_return_12m(df: pd.DataFrame) -> Optional[float]:
    """12개월 수익률 계산"""
    if df is None or len(df) < 200:
        return None

    try:
        current = df['Close'].iloc[-1]
        past = df['Close'].iloc[-252] if len(df) >= 252 else df['Close'].iloc[0]
        if past > 0:
            return ((current / past) - 1) * 100
    except Exception:
        pass

    return None


def calc_k_ratio(df: pd.DataFrame) -> Optional[float]:
    """
    K-Ratio 계산: 누적수익률 선형회귀 기울기 / 표준오차
    높을수록 안정적인 상승 추세
    """
    if df is None or len(df) < 200:
        return None

    try:
        # 누적 수익률
        returns = df['Close'].pct_change().dropna()
        cum_returns = (1 + returns).cumprod()

        if len(cum_returns) < 100:
            return None

        # 최근 252일 (1년)
        cum_recent = cum_returns.tail(252).values
        x = np.arange(len(cum_recent))

        # 선형 회귀
        slope, _, _, _, stderr = linregress(x, np.log(cum_recent + 1e-10))

        if stderr > 0:
            return round(slope / stderr, 4)
    except Exception:
        pass

    return None


# ============================================================================
# 전처리 파이프라인
# ============================================================================

def winsorize(series: pd.Series, lower_pct: float = 0.01, upper_pct: float = 0.99) -> pd.Series:
    """Winsorizing: 상하위 극단값 제거"""
    lower = series.quantile(lower_pct)
    upper = series.quantile(upper_pct)
    return series.clip(lower, upper)


def preprocess_factor(df: pd.DataFrame, factor_col: str, sector_col: str,
                      ascending: bool = True) -> pd.DataFrame:
    """
    팩터 전처리: Winsorize → Rank → 섹터별 Z-Score

    Args:
        df: 데이터프레임
        factor_col: 팩터 컬럼명
        sector_col: 섹터 컬럼명
        ascending: True면 값이 높을수록 좋음
    """
    df = df.copy()
    col = factor_col

    # NaN 제외
    mask = df[col].notna()
    if mask.sum() < 10:
        df[f'{col}_zscore'] = np.nan
        return df

    # 1. Winsorizing (상하위 1%)
    df.loc[mask, col] = winsorize(df.loc[mask, col])

    # 2. Rank 변환
    df[f'{col}_rank'] = df[col].rank(ascending=ascending, na_option='keep')

    # 3. 섹터별 Z-Score
    def sector_zscore(group):
        if len(group) < 3:
            return pd.Series(0, index=group.index)
        vals = group.values
        valid = ~np.isnan(vals)
        if valid.sum() < 3:
            return pd.Series(0, index=group.index)
        result = np.full(len(vals), np.nan)
        result[valid] = zscore(vals[valid], nan_policy='omit')
        return pd.Series(result, index=group.index)

    df[f'{col}_zscore'] = df.groupby(sector_col)[f'{col}_rank'].transform(sector_zscore)

    # NaN을 0으로 (섹터 내 종목 부족 등)
    df[f'{col}_zscore'] = df[f'{col}_zscore'].fillna(0)

    return df


# ============================================================================
# 메인 로직
# ============================================================================

def run_multi_factor():
    """멀티팩터 스크리너 실행"""
    print("=" * 60)
    print("멀티팩터 스크리너 (Quality + Value + Momentum)")
    print(f"실행 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    start_time = time.time()

    # 1. KRX 종목 마스터 수집
    print("\n[1/6] KRX 종목 마스터 수집...")
    master = get_stock_master()

    # 필터링: 시총 1000억+, 보통주, 스팩/리츠 제외
    filtered = master[
        (master['시가총액'] >= MIN_MARKET_CAP) &
        (master['is_common'] == True) &
        (master['is_spac'] == False) &
        (master['is_reit'] == False)
    ].copy()

    print(f"  필터링 후: {len(filtered)}개 종목")

    # 2. 섹터 데이터 수집
    print("\n[2/6] WiseIndex 섹터 데이터 수집...")
    try:
        sector_df = get_sector_data()
        filtered = merge_sector_info(filtered, sector_df)
    except Exception as e:
        print(f"  섹터 수집 실패 ({e}), 업종 컬럼 사용")
        if '업종' in filtered.columns:
            filtered['섹터명'] = filtered['업종']
        else:
            filtered['섹터명'] = '기타'

    # 3. OHLCV 캐시 로드 + Momentum 계산
    print("\n[3/6] Momentum 계산 (OHLCV 캐시)...")
    ohlcv_cache = _load_ohlcv_cache()

    momentum_data = {}
    for _, row in filtered.iterrows():
        code = row['종목코드']
        df_ohlcv = _get_ohlcv(code, ohlcv_cache, days=300)
        if df_ohlcv is not None:
            ret_12m = calc_return_12m(df_ohlcv)
            k_ratio = calc_k_ratio(df_ohlcv)
            momentum_data[code] = {
                'return_12m': ret_12m,
                'k_ratio': k_ratio,
            }

    print(f"  Momentum 계산: {len(momentum_data)}개 종목")

    # 4. FnGuide 재무데이터 (캐시 → fallback 개별 수집)
    print(f"\n[4/6] FnGuide 재무데이터 로드 ({len(filtered)}개 종목)...")

    fnguide_cache = load_fnguide_cache()
    if fnguide_cache:
        print(f"  캐시 사용: {len(fnguide_cache)}개 종목")
    else:
        print("  캐시 없음 — 개별 수집 모드 (느림)")
        fnguide_cache = {}

    records = []
    success = 0
    skipped = 0

    for _, row in filtered.iterrows():
        code = row['종목코드']
        name = row['종목명']
        market_cap = row.get('시가총액', 0)
        current_price = row.get('종가', 0) or 0
        sector = row.get('섹터명', '기타')

        fin_data = fnguide_cache.get(code)
        if not fin_data:
            fin_data = get_financial_data(code)

        record = {
            'ticker': code,
            'name': name,
            'sector': sector,
            'current_price': int(current_price),
            'market_cap': round(market_cap, 0),
            'per': row.get('PER'),
            'pbr': row.get('PBR'),
            'div_yield': row.get('배당수익률'),
        }

        has_fin = False
        if fin_data:
            record['roe'] = get_roe(fin_data)
            record['gpa'] = get_gpa(fin_data)
            record['cfo'] = get_cfo_ratio(fin_data)
            record['psr'] = get_psr(market_cap, fin_data)
            record['pcr'] = get_pcr(market_cap, fin_data)

            if record['per'] is None or (isinstance(record['per'], float) and np.isnan(record['per'])):
                ni_ttm = fin_data.get('net_income_ttm')
                if ni_ttm and ni_ttm > 0 and market_cap > 0:
                    record['per'] = round(market_cap / ni_ttm, 2)

            if record['pbr'] is None or (isinstance(record['pbr'], float) and np.isnan(record['pbr'])):
                eq = fin_data.get('balance', {}).get('total_equity')
                if eq and eq > 0 and market_cap > 0:
                    record['pbr'] = round(market_cap / eq, 2)

            if record['div_yield'] is None or (isinstance(record['div_yield'], float) and np.isnan(record['div_yield'])):
                header_div = fin_data.get('header', {}).get('dividend_yield')
                if header_div is not None and not (isinstance(header_div, float) and np.isnan(header_div)):
                    record['div_yield'] = round(header_div, 2)

            has_fin = True

        # Momentum
        if code in momentum_data:
            record['return_12m'] = momentum_data[code]['return_12m']
            record['k_ratio'] = momentum_data[code]['k_ratio']

        if has_fin:
            success += 1
        else:
            skipped += 1
        records.append(record)

        done = success + skipped
        if done % 200 == 0 or done == len(filtered):
            print(f"  진행: {done}/{len(filtered)} (재무: {success}, 스킵: {skipped})")

    print(f"  완료: {success}개 재무데이터")

    # DataFrame 변환
    df = pd.DataFrame(records)

    # 유효 데이터 필터
    # 조건1: 각 팩터 그룹에 최소 1개 이상 유효
    # 조건2: 전체 10개 지표 중 최소 7개 이상 유효
    all_indicators = ['roe', 'gpa', 'cfo', 'per', 'pbr', 'psr', 'pcr', 'div_yield', 'return_12m', 'k_ratio']
    has_quality = df[['roe', 'gpa', 'cfo']].notna().any(axis=1)
    has_value = df[['per', 'pbr', 'psr', 'pcr', 'div_yield']].notna().any(axis=1)
    has_momentum = df[['return_12m', 'k_ratio']].notna().any(axis=1)
    valid_count = df[all_indicators].notna().sum(axis=1)

    before_count = len(df)
    df = df[has_quality & has_value & has_momentum & (valid_count >= 7)].copy()

    print(f"\n  유효 데이터: {len(df)}개 종목 (필터 전: {before_count}, 제외: {before_count - len(df)})")

    if len(df) < 20:
        print("유효 데이터가 부족합니다.")
        return

    # 5. 팩터 전처리 (Winsorize + 섹터 Z-Score)
    print("\n[5/6] 팩터 전처리...")
    sector_col = 'sector'

    # Quality (높을수록 좋음)
    for col in ['roe', 'gpa', 'cfo']:
        df = preprocess_factor(df, col, sector_col, ascending=True)

    # Value (낮을수록 좋음 → ascending=False로 rank 반전)
    for col in ['per', 'pbr', 'psr', 'pcr']:
        df = preprocess_factor(df, col, sector_col, ascending=False)

    # 배당수익률 (높을수록 좋음)
    df = preprocess_factor(df, 'div_yield', sector_col, ascending=True)

    # Momentum (높을수록 좋음)
    for col in ['return_12m', 'k_ratio']:
        df = preprocess_factor(df, col, sector_col, ascending=True)

    # 6. 팩터 결합
    print("\n[6/6] 팩터 결합 및 최종 순위...")

    # Step 4: 팩터 내부 Z-Score 합산
    q_cols = [f'{c}_zscore' for c in ['roe', 'gpa', 'cfo']]
    df['quality_raw'] = df[q_cols].sum(axis=1, skipna=True, min_count=1)

    v_cols = [f'{c}_zscore' for c in ['per', 'pbr', 'psr', 'pcr', 'div_yield']]
    df['value_raw'] = df[v_cols].sum(axis=1, skipna=True, min_count=1)

    m_cols = [f'{c}_zscore' for c in ['return_12m', 'k_ratio']]
    df['momentum_raw'] = df[m_cols].sum(axis=1, skipna=True, min_count=1)

    # Step 5: 분포 재조정 (Rescaling)
    # Q(3개), V(5개), M(2개) 지표 수가 달라 점수 범위가 다름
    # → 각 팩터 합산값에 다시 Z-Score 적용하여 동일 분포로 맞춤
    for col in ['quality_raw', 'value_raw', 'momentum_raw']:
        score_col = col.replace('_raw', '_score')
        valid = df[col].notna()
        if valid.sum() >= 3:
            df.loc[valid, score_col] = zscore(df.loc[valid, col].values, nan_policy='omit')
            df[score_col] = df[score_col].fillna(0)
        else:
            df[score_col] = 0

    # Step 6: 최종 가중합
    df['total_score'] = (
        df['quality_score'] * WEIGHTS['quality'] +
        df['value_score'] * WEIGHTS['value'] +
        df['momentum_score'] * WEIGHTS['momentum']
    )

    # NaN 처리 (일부 팩터 없는 종목)
    df['total_score'] = df['total_score'].fillna(df['total_score'].min() - 1)

    # 정렬
    df = df.sort_values('total_score', ascending=False).reset_index(drop=True)

    # Top N 선정
    top = df.head(TOP_N)

    # 결과 JSON 생성
    output_data = []
    for i, row in top.iterrows():
        output_data.append({
            'rank': i + 1,
            'ticker': row['ticker'],
            'name': row['name'],
            'sector': row['sector'],
            'current_price': int(row.get('current_price', 0)),
            'total_score': round(row['total_score'], 3),
            'quality_score': round(row['quality_score'], 3) if pd.notna(row['quality_score']) else None,
            'value_score': round(row['value_score'], 3) if pd.notna(row['value_score']) else None,
            'momentum_score': round(row['momentum_score'], 3) if pd.notna(row['momentum_score']) else None,
            'roe': round(row['roe'], 1) if pd.notna(row.get('roe')) else None,
            'gpa': round(row['gpa'], 3) if pd.notna(row.get('gpa')) else None,
            'cfo': round(row['cfo'], 3) if pd.notna(row.get('cfo')) else None,
            'per': round(row['per'], 1) if pd.notna(row.get('per')) else None,
            'pbr': round(row['pbr'], 2) if pd.notna(row.get('pbr')) else None,
            'psr': round(row['psr'], 2) if pd.notna(row.get('psr')) else None,
            'pcr': round(row['pcr'], 1) if pd.notna(row.get('pcr')) else None,
            'div_yield': round(row['div_yield'], 1) if pd.notna(row.get('div_yield')) else None,
            'return_12m': round(row['return_12m'], 1) if pd.notna(row.get('return_12m')) else None,
            'k_ratio': round(row['k_ratio'], 3) if pd.notna(row.get('k_ratio')) else None,
            'market_cap': row['market_cap'],
        })

    # 저장
    os.makedirs(DATA_PATH, exist_ok=True)
    output = {
        'meta': {
            'type': 'multi-factor',
            'name': '멀티팩터 (Q+V+M)',
            'description': 'Quality + Value + Momentum 섹터 중립화 멀티팩터',
            'lastUpdated': datetime.now().isoformat(),
            'totalCount': len(output_data),
            'screened_from': len(df),
            'weights': WEIGHTS,
        },
        'data': output_data,
    }

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    elapsed = time.time() - start_time

    # 결과 출력
    print(f"\n{'=' * 90}")
    print(f"멀티팩터 Top {TOP_N}")
    print(f"{'=' * 90}")
    print(f"{'순위':<4} {'종목명':<12} {'섹터':<8} {'총점':>7} {'Q점수':>7} {'V점수':>7} {'M점수':>7} {'ROE':>6} {'PER':>6}")
    print(f"{'-' * 90}")

    for item in output_data:
        q = f"{item['quality_score']:.2f}" if item['quality_score'] else '-'
        v = f"{item['value_score']:.2f}" if item['value_score'] else '-'
        m = f"{item['momentum_score']:.2f}" if item['momentum_score'] else '-'
        roe = f"{item['roe']:.1f}" if item['roe'] else '-'
        per = f"{item['per']:.1f}" if item['per'] else '-'
        print(
            f"{item['rank']:<4} {item['name']:<12} {item['sector']:<8} "
            f"{item['total_score']:>7.3f} {q:>7} {v:>7} {m:>7} {roe:>6} {per:>6}"
        )

    print(f"\n저장: {OUTPUT_FILE}")
    print(f"총 실행 시간: {elapsed:.1f}초 ({elapsed/60:.1f}분)")
    print("=" * 60)


if __name__ == '__main__':
    run_multi_factor()
