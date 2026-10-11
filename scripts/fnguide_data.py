"""
FnGuide 재무제표 수집 모듈 (wcomp.fnguide.com JSON API)

2026-07 FnGuide 개편으로 comp.fnguide.com/SVO2/ASP/SVD_*.asp 가 폐기되어
신규 사이트의 JSON 엔드포인트로 이관했다.

수집 데이터:
- 연간/분기 손익계산서: 매출액, 매출총이익, 영업이익(EBIT), 당기순이익, 지배주주순이익
- 연간/분기 재무상태표: 총자산, 자본총계, 지배주주지분, 유동자산, 유동부채, 비유동자산, 유형자산, 차입금(유동성장기부채 포함), 현금, 재고자산
- 연간/분기 현금흐름표: 영업활동현금흐름, 투자활동현금흐름
- Snapshot 페이지: PER, 12M PER, 업종 PER, PBR, 배당수익률(헤더), 연간 EPS·ROE

주의: 신규 API의 연간 데이터는 최근 3개년뿐이다. 그 이전 연도는
collect_quant_data.py 가 이전 캐시에서 이어붙인다.

사용법:
    from scripts.fnguide_data import get_financial_data
    data = get_financial_data('005930')
"""

import json
import os
import re
import time
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd
import requests

# ============================================================================
# 상수
# ============================================================================

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Accept-Language': 'ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7',
}

FNGUIDE_BASE = 'https://wcomp.fnguide.com/CompanyInfo'
SCHEMA_VERSION = 3  # 수집 필드가 바뀌면 올림 → 이전 캐시는 재사용하지 않고 재수집 (2: capex, ytd / 3: 유형자산, 유동성장기부채, 연간 ROE)
FNGUIDE_SNAPSHOT_URL = f'{FNGUIDE_BASE}/Snapshot'

# 유형 투하자본 / 총자산 하한 (마법공식·애크먼 공통). 미만이면 지주사·결제대행처럼
# 자산이 금융성이라 ROIC 가 수백~수만 %로 튀는 종목 → 제외
MIN_IC_TO_ASSETS = 0.1

# 손익계산서 항목 매핑 (FnGuide 계정명 → 코드 키)
INCOME_ITEMS = {
    '매출액': 'revenue',
    '매출총이익': 'gross_profit',
    '영업이익': 'operating_income',
    '당기순이익': 'net_income',
    '(지배주주지분)당기순이익': 'net_income_controlling',
    '법인세비용': 'tax_expense',
    '금융원가': 'finance_cost',
}

# 재무상태표 항목 매핑 (key: 영문키, value: 검색할 계정명 리스트)
BALANCE_SEARCH = {
    'total_assets': ['자산총계', '자산'],  # 금융업은 '자산'/'자본'
    'total_equity': ['자본총계', '자본'],
    'controlling_equity': ['지배주주지분'],
    'current_assets': ['유동자산'],
    'current_liabilities': ['유동부채'],
    'non_current_assets': ['비유동자산'],
    'non_current_liabilities': ['비유동부채'],
    'cash': ['현금및현금성자산'],
    'short_term_debt': ['단기차입금'],
    'current_portion_ltd': ['유동성장기부채'],
    'long_term_debt': ['장기차입금'],
    'bonds': ['사채'],
    'inventory': ['재고자산'],
    'ppe': ['유형자산'],
}

# 현금흐름표 항목 매핑
CASHFLOW_ITEMS = {
    '영업활동으로인한현금흐름': 'cfo',
}

CASHFLOW_INVEST_ITEMS = ['투자활동으로인한현금흐름']
CAPEX_ITEMS = ['유형자산의증가', '무형자산의증가']

# Snapshot 페이지 헤더 지표 ID 매핑 (corp_group2 영역)
HEADER_METRIC_IDS = {
    'h_per': 'per',
    'h_12m': 'per_12m',
    'h_u_per': 'sector_per',
    'h_pbr': 'pbr',
    'h_rate': 'dividend_yield',
}


# ============================================================================
# 파싱 함수 (네트워크 없음 — test_fnguide_parse.py 에서 점검)
# ============================================================================

def _dataset_to_df(dataset: Dict, quarterly: bool) -> pd.DataFrame:
    """
    FnGuide JSON dataset → DataFrame (index=계정명, columns='YYYY/MM' 오래된→최근 순).

    - '전년동기', '전년동기대비(%)' 컬럼은 버린다.
    - 연간(quarterly=False)은 '(최근분기)' 처럼 괄호 표시가 붙은 누적 컬럼도 버린다.
      분기 표에서는 '(최근분기)' 가 실제 최신 분기이므로 유지한다.
    - 같은 이름의 계정(하위 '기타' 등)은 첫 행(상위 계정)만 남긴다.
    """
    cols = {}
    for h in dataset.get('header') or []:
        yymm = str(h.get('YYMM', ''))
        m = re.match(r'(\d{4})/(\d{2})', yymm)
        if not m or '전년' in yymm or (not quarterly and '(' in yymm):
            continue
        cols[h['CD']] = f'{m.group(1)}/{m.group(2)}'

    rows = {}
    for r in dataset.get('data') or []:
        name = re.sub(r'\s+', '', str(r.get('NAME', '')))
        if name and name not in rows:
            rows[name] = {col: r.get(cd) for cd, col in cols.items()}

    df = pd.DataFrame.from_dict(rows, orient='index', columns=list(cols.values()))
    return df.apply(pd.to_numeric, errors='coerce')


def _latest_value(series: pd.Series) -> Optional[float]:
    """가장 최근(오른쪽) 기간의 유효값."""
    valid = series.dropna()
    return float(valid.iloc[-1]) if len(valid) else None


def _parse_header_metrics(html: str) -> Dict[str, float]:
    """Snapshot 헤더에서 PER, 12M PER, 업종 PER, PBR, 배당수익률 추출."""
    metrics = {}
    pattern = r'id="(h_\w+)"[^>]*>.*?</button>\s*</li>\s*<li>\s*([^<]*)</li>'
    for match in re.finditer(pattern, html, re.DOTALL):
        key = HEADER_METRIC_IDS.get(match.group(1))
        value_str = match.group(2).replace('%', '').replace(',', '').strip()
        if key:
            try:
                metrics[key] = float(value_str)
            except ValueError:
                pass
    return metrics


def _parse_snapshot_eps(html: str) -> Dict[str, float]:
    """Snapshot 의 확정 연간 EPS."""
    return _parse_snapshot_row(html, 'EPS')


def _parse_snapshot_row(html: str, row_name: str) -> Dict[str, float]:
    """Snapshot 의 snpFinancial(Financial Highlight)에서 row_name 행의 확정 연간값만 추출 (추정치·분기 제외)."""
    m = re.search(r'snpFinancial\s*:\s*', html)
    if not m:
        return {}
    try:
        obj, _ = json.JSONDecoder().raw_decode(html[m.end():])
    except ValueError:
        return {}
    annual_cds = {h['CD']: h['YYMM'] for h in obj.get('header', [])
                  if h.get('EP_CHK') is None and re.fullmatch(r'\d{4}/\d{2}', str(h.get('YYMM', '')))}
    for r in obj.get('data', []):
        if str(r.get('NAME', '')).strip() == row_name:
            out = {}
            for cd, yymm in annual_cds.items():
                try:
                    out[yymm] = float(r.get(cd))
                except (TypeError, ValueError):
                    pass
            return out
    return {}


def _extract_item(df: pd.DataFrame, item_names: list) -> pd.Series:
    """
    DataFrame에서 항목을 추출합니다.
    여러 가능한 이름을 시도합니다.
    부분 일치 시 가장 짧은 인덱스를 선택 (더 구체적인 매칭).
    """
    if df is None or df.empty:
        return pd.Series(dtype=float)

    for name in item_names:
        # 정확히 일치
        if name in df.index:
            return df.loc[name]

        # 부분 일치 (가장 짧은 매칭 = 가장 구체적)
        matches = [idx for idx in df.index if name in str(idx)]
        if matches:
            # '비지배' vs '지배' 같은 오매칭 방지: name이 idx의 시작과 일치하는 것 우선
            starts_with = [idx for idx in matches if str(idx).startswith(name)]
            if starts_with:
                return df.loc[min(starts_with, key=len)]
            return df.loc[min(matches, key=len)]

    return pd.Series(dtype=float)


# ============================================================================
# 데이터 수집 함수
# ============================================================================

def _fetch_dataset(endpoint: str, code: str, freq: str, consol: str) -> Dict:
    resp = requests.get(
        f'{FNGUIDE_BASE}/{endpoint}',
        params={'cmp_cd': code, 'freq_typ': freq, 'consol_typ': consol},
        headers=HEADERS, timeout=20,
    )
    resp.raise_for_status()
    return resp.json().get('dataset') or {}


def _parse_ytd(dataset: Dict) -> Dict[str, Dict[str, float]]:
    """
    연간 손익표의 '(최근분기)'·'(전년동기)' 누적 컬럼 → {en_key: {'cur', 'prev'}}.
    예: 2026/06 누적 vs 2025/06 누적. 최근 실적 꺾임 판단용.
    """
    cds = {}
    for h in dataset.get('header') or []:
        yymm = str(h.get('YYMM', ''))
        if '최근' in yymm:
            cds['cur'] = h['CD']
        elif '전년동기' in yymm and '%' not in yymm:
            cds['prev'] = h['CD']
    if len(cds) < 2:
        return {}
    out = {}
    for r in dataset.get('data') or []:
        name = re.sub(r'\s+', '', str(r.get('NAME', '')))
        for kr_name, en_key in INCOME_ITEMS.items():
            if en_key not in out and (name == kr_name or (kr_name == '매출액' and name.startswith('매출액'))):
                try:
                    out[en_key] = {k: float(r.get(cd)) for k, cd in cds.items()}
                except (TypeError, ValueError):
                    pass
    return out


def _fetch_statements(code: str) -> Optional[Dict[str, Any]]:
    """
    연결(C) 우선, 연결 연간 손익이 비어 있을 때만 별도(P)로 6개 표를 받는다.
    요청 오류는 None (연결 종목을 일시 오류로 별도 기준으로 받아 섞이지 않게 — 호출부가 재시도/이전 값 유지).
    """
    for consol in ('C', 'P'):
        try:
            y_income = _fetch_dataset('getFinIncome', code, 'Y', consol)
            tables = {'Y_income': _dataset_to_df(y_income, False), 'ytd': _parse_ytd(y_income)}
            if tables['Y_income'].dropna(how='all').empty:
                continue  # 이 기준으로는 데이터 없음 → 나머지 5개 요청 생략
            tables['Q_income'] = _dataset_to_df(_fetch_dataset('getFinIncome', code, 'Q', consol), True)
            for name, endpoint in (('balance', 'getFinBalance'), ('cashflow', 'getFinCashFlow')):
                for freq, quarterly in (('Y', False), ('Q', True)):
                    tables[f'{freq}_{name}'] = _dataset_to_df(
                        _fetch_dataset(endpoint, code, freq, consol), quarterly)
            return tables
        except (requests.RequestException, ValueError):
            return None
    return None


def get_financial_data(code: str, retry: int = 2) -> Optional[Dict[str, Any]]:
    """
    FnGuide에서 종목의 재무데이터를 수집합니다.

    Returns:
        {
            'code': '005930',
            'fetched_at': epoch 초,
            'header': {'per':..., 'pbr':..., 'dividend_yield':...},
            'annual': {'revenue': {'2025/12': 값, ...}, 'eps': {...}, ...},
            'quarter': {'revenue': {'2026/06': 값, ...}, ...},   # 개별 분기값
            'balance': {...},       # 최근 분기 시점값 (EV, PBR 등)
            'balance_avg': {...},   # 최근 4분기 평균 (ROE, GPA 등)
            'cashflow': {'cfo_annual': {...}, 'cfo_quarter': {...}, ...},
            'revenue_ttm': ..., 'cfo_ttm': ..., ...
        }
    """
    for attempt in range(retry):
        try:
            tables = _fetch_statements(code)
            if tables is None:
                return None

            snap = requests.get(
                FNGUIDE_SNAPSHOT_URL,
                params={'c_id': 'AA', 'menu_type': '01', 'cmp_cd': code},
                headers=HEADERS, timeout=20,
            )
            snap.raise_for_status()

            result = {
                'code': code,
                'schema': SCHEMA_VERSION,
                'fetched_at': time.time(),
                'header': _parse_header_metrics(snap.text),
                'annual': {},
                'quarter': {},
                'balance': {},
                'balance_avg': {},
                'cashflow': {},
            }

            # 1. 손익계산서
            for period, df in (('annual', tables['Y_income']), ('quarter', tables['Q_income'])):
                for kr_name, en_key in INCOME_ITEMS.items():
                    series = _extract_item(df, [kr_name]).dropna()
                    if not series.empty:
                        result[period][en_key] = series.to_dict()

            for key, row_name in (('eps', 'EPS'), ('roe', 'ROE')):
                values = _parse_snapshot_row(snap.text, row_name)
                if values:
                    result['annual'][key] = values

            # 2. 재무상태표 (분기 우선, 없으면 연간). 컬럼은 오래된→최근 순
            bs = tables['Q_balance'] if not tables['Q_balance'].dropna(how='all').empty else tables['Y_balance']
            for en_key, kr_names in BALANCE_SEARCH.items():
                series = _extract_item(bs, kr_names).dropna()
                if not series.empty:
                    result['balance'][en_key] = _latest_value(series)
                    result['balance_avg'][en_key] = float(series.tail(4).mean())

            # 이자부채: 항목이 하나라도 있거나 일반 기업 양식(유동자산 있음)이면 합산 (무차입 = 0).
            # 금융업은 계정 체계가 달라 total_debt 를 만들지 않는다 (없음 ≠ 0)
            debt_keys = ('short_term_debt', 'current_portion_ltd', 'long_term_debt', 'bonds')
            if 'current_assets' in result['balance'] or any(k in result['balance'] for k in debt_keys):
                result['balance']['total_debt'] = sum(result['balance'].get(k) or 0 for k in debt_keys)

            # 3. 현금흐름표
            for suffix, df in (('annual', tables['Y_cashflow']), ('quarter', tables['Q_cashflow'])):
                cfo = _extract_item(df, list(CASHFLOW_ITEMS)).dropna()
                if not cfo.empty:
                    result['cashflow'][f'cfo_{suffix}'] = cfo.to_dict()
                inv = _extract_item(df, CASHFLOW_INVEST_ITEMS).dropna()
                if not inv.empty:
                    result['cashflow'][f'invest_cf_{suffix}'] = inv.to_dict()
                # CAPEX = 유형자산의증가 + 무형자산의증가 (FnGuide는 유출을 양수로 표기)
                capex_rows = [_extract_item(df, [n]) for n in CAPEX_ITEMS]
                capex_rows = [s for s in capex_rows if not s.empty]
                if capex_rows:
                    capex = pd.concat(capex_rows, axis=1).sum(axis=1, min_count=1).dropna()
                    if not capex.empty:
                        result['cashflow'][f'capex_{suffix}'] = capex.to_dict()

            result['ytd'] = tables['ytd']
            _calculate_ttm(result)
            return result

        except Exception:
            if attempt < retry - 1:
                time.sleep(1)

    return None


def _ttm(quarter: Dict[str, float], annual: Dict[str, float]) -> Optional[float]:
    """최근 4분기 합. 4분기가 다 없으면 최신 연간값."""
    recent_4 = [v for _, v in sorted(quarter.items(), reverse=True)[:4]
                if v is not None and not np.isnan(v)]
    if len(recent_4) == 4:
        return sum(recent_4)
    for _, v in sorted(annual.items(), reverse=True):
        if v is not None and not np.isnan(v):
            return v
    return None


def _calculate_ttm(data: Dict[str, Any]) -> None:
    """TTM (Trailing Twelve Months): 손익 항목 + 영업CF·투자CF·CAPEX."""
    for key in ['revenue', 'operating_income', 'net_income', 'gross_profit',
                'net_income_controlling', 'tax_expense', 'finance_cost']:
        value = _ttm(data['quarter'].get(key, {}), data['annual'].get(key, {}))
        if value is not None:
            data[f'{key}_ttm'] = value

    cf = data['cashflow']
    for key, out in (('cfo', 'cfo_ttm'), ('invest_cf', 'invest_cf_ttm'), ('capex', 'capex_ttm')):
        value = _ttm(cf.get(f'{key}_quarter', {}), cf.get(f'{key}_annual', {}))
        if value is not None:
            data[out] = value


# ============================================================================
# 배치 수집
# ============================================================================

def get_financial_data_batch(codes: List[str], delay: float = 0.3) -> Dict[str, Dict]:
    """
    여러 종목의 재무데이터를 순차 수집합니다.

    Args:
        codes: 종목코드 리스트
        delay: 요청 간 대기시간 (FnGuide 속도 제한으로 병렬 처리 불가)

    Returns:
        {code: financial_data} 딕셔너리
    """
    results = {}
    total = len(codes)

    print(f"FnGuide 재무데이터 수집 시작 ({total}개 종목)")

    for i, code in enumerate(codes):
        data = get_financial_data(code)
        if data:
            results[code] = data

        if (i + 1) % 50 == 0 or (i + 1) == total:
            print(f"  진행: {i + 1}/{total} (수집: {len(results)}개)")

        if delay > 0:
            time.sleep(delay)

    print(f"수집 완료: {len(results)}/{total}개")
    return results


# ============================================================================
# 헬퍼 함수 (스크리너에서 사용)
# ============================================================================

def get_ebit(data: Dict) -> Optional[float]:
    """
    EBIT = 영업이익 (Operating Income, TTM)
    K-IFRS '영업이익'이 그린블라트의 EBIT과 동일.
    NI+Tax+FC는 비영업수익(금융수익, 자산매각 등)을 포함하므로 사용하지 않음.
    """
    op = data.get('operating_income_ttm')
    if op is not None:
        return op

    # fallback: NI + Tax + Finance Cost (영업이익 데이터 없을 때만)
    ni = data.get('net_income_ttm')
    tax = data.get('tax_expense_ttm')
    fin_cost = data.get('finance_cost_ttm')

    if ni is not None and tax is not None and fin_cost is not None:
        return ni + tax + fin_cost

    return None


def get_invested_capital(data: Dict) -> Optional[float]:
    """
    유형 투하자본 (그린블라트) = 순운전자본 + 순고정자산 (최신 분기 B/S)
    순운전자본 = max(0, (유동자산 - 현금) - (유동부채 - 단기차입금 - 유동성장기부채))
    순고정자산 = 유형자산
    총자산 × MIN_IC_TO_ASSETS 미만(지주사·결제대행 등 금융성 자산 기업)이거나 총자산이 없으면 None.
    """
    bs = data.get('balance', {})
    ca = bs.get('current_assets')
    cl = bs.get('current_liabilities')
    ppe = bs.get('ppe')
    ta = bs.get('total_assets')
    if ca is None or cl is None or ppe is None or not ta:
        return None
    nwc = max(0, (ca - (bs.get('cash') or 0))
              - (cl - (bs.get('short_term_debt') or 0) - (bs.get('current_portion_ltd') or 0)))
    ic = nwc + ppe
    return ic if ic >= ta * MIN_IC_TO_ASSETS else None


def get_ev(market_cap: float, data: Dict) -> Optional[float]:
    """
    기업가치 (EV) = 시가총액 + 이자부채 - 여유자금 (그린블라트 excess cash)
    이자부채 = 단기차입금 + 유동성장기부채 + 장기차입금 + 사채 (balance['total_debt'])
    여유자금 = max(0, 현금 - max(0, 유동부채 - 유동자산 + 현금))
      → 운전자본에 묶인 현금(결제대행 예수금, 광고대행 미지급금 등)은 빼지 않는다.
    이자부채 계정이 없는 금융업은 None. 0 이하가 될 수 있음 (호출부에서 거름).
    """
    bs = data.get('balance', {})
    debt = bs.get('total_debt')
    if debt is None:
        return None
    cash = bs.get('cash') or 0
    cl = bs.get('current_liabilities') or 0
    ca = bs.get('current_assets') or 0
    excess_cash = max(0, cash - max(0, cl - ca + cash))
    return market_cap + debt - excess_cash


def get_roe(data: Dict) -> Optional[float]:
    """ROE = 지배주주순이익(TTM) / 지배주주지분(4분기 평균)"""
    ni = data.get('net_income_controlling_ttm') or data.get('net_income_ttm')
    bs = data.get('balance_avg') or data.get('balance', {})
    eq = bs.get('controlling_equity') or bs.get('total_equity')

    if ni is not None and eq is not None and eq > 0:
        return (ni / eq) * 100
    return None


def get_gpa(data: Dict) -> Optional[float]:
    """GPA = 매출총이익(TTM) / 총자산(4분기 평균)"""
    gp = data.get('gross_profit_ttm')
    bs = data.get('balance_avg') or data.get('balance', {})
    ta = bs.get('total_assets')

    if gp is not None and ta is not None and ta > 0:
        return gp / ta
    return None


def get_cfo_ratio(data: Dict) -> Optional[float]:
    """CFO = 영업활동현금흐름(TTM) / 총자산(4분기 평균)"""
    cfo = data.get('cfo_ttm')
    bs = data.get('balance_avg') or data.get('balance', {})
    ta = bs.get('total_assets')

    if cfo is not None and ta is not None and ta > 0:
        return cfo / ta
    return None


def get_psr(market_cap: float, data: Dict) -> Optional[float]:
    """PSR = 시가총액 / 매출액(TTM)"""
    rev = data.get('revenue_ttm')
    if rev is not None and rev > 0:
        return market_cap / rev
    return None


def get_pcr(market_cap: float, data: Dict) -> Optional[float]:
    """PCR = 시가총액 / 영업현금흐름(TTM)"""
    cfo = data.get('cfo_ttm')
    if cfo is not None and cfo > 0:
        return market_cap / cfo
    return None


def get_current_ratio(data: Dict) -> Optional[float]:
    """유동비율 = 유동자산 / 유동부채 (배수)"""
    bs = data.get('balance', {})
    ca = bs.get('current_assets')
    cl = bs.get('current_liabilities')
    if ca is not None and cl is not None and cl > 0:
        return ca / cl
    return None


def get_debt_ratio(data: Dict) -> Optional[float]:
    """부채비율 = (유동부채+비유동부채) / 자본총계 × 100"""
    bs = data.get('balance', {})
    cl = bs.get('current_liabilities', 0) or 0
    ncl = bs.get('non_current_liabilities', 0) or 0
    eq = bs.get('total_equity')
    if eq is not None and eq > 0:
        return ((cl + ncl) / eq) * 100
    return None


def get_long_term_debt_ratio(data: Dict) -> Optional[float]:
    """장기부채비율 = (장기차입금+사채) / 자본총계 × 100"""
    bs = data.get('balance', {})
    long_debt = bs.get('long_term_debt', 0) or 0
    bonds = bs.get('bonds', 0) or 0
    eq = bs.get('total_equity')
    if eq is not None and eq > 0:
        return ((long_debt + bonds) / eq) * 100  # 무차입 = 0.0 (제외 대상 아님)
    return None


def get_roa(data: Dict) -> Optional[float]:
    """ROA = 순이익(TTM) / 총자산(4분기 평균) × 100"""
    ni = data.get('net_income_ttm')
    bs = data.get('balance_avg') or data.get('balance', {})
    ta = bs.get('total_assets')
    if ni is not None and ta is not None and ta > 0:
        return (ni / ta) * 100
    return None


def get_roic(data: Dict) -> Optional[float]:
    """ROIC = EBIT / 투하자본 × 100"""
    ebit = get_ebit(data)
    ic = get_invested_capital(data)
    if ebit is not None and ic is not None and ic > 0:
        return (ebit / ic) * 100
    return None


def get_net_current_assets(data: Dict) -> Optional[float]:
    """순유동자산 = 유동자산 - 유동부채 (억원)"""
    bs = data.get('balance', {})
    ca = bs.get('current_assets')
    cl = bs.get('current_liabilities')
    if ca is not None and cl is not None:
        return ca - cl
    return None


def get_fcf(data: Dict) -> Optional[float]:
    """
    FCF = 영업활동현금흐름(TTM) - CAPEX(TTM), CAPEX = 유형자산의증가 + 무형자산의증가.
    (투자활동CF 전체를 빼면 단기금융상품 매매까지 섞여 FCF가 왜곡됨)
    """
    cfo = data.get('cfo_ttm')
    capex = data.get('capex_ttm')
    if cfo is not None and capex is not None:
        return cfo - capex
    return None


def get_cagr(data: Dict, key: str, years: int = 5) -> Optional[float]:
    """
    연간 CAGR(%) = (최근 연간 / N년 전 연간)^(1/N) - 1, 최근 years년 구간 사용.
    양 끝값이 모두 양수일 때만 계산 (적자 구간은 성장률이 정의되지 않음).
    """
    annual = sorted((p, v) for p, v in data.get('annual', {}).get(key, {}).items()
                    if re.fullmatch(r'\d{4}/\d{2}', p) and v is not None and not np.isnan(v))
    if annual:
        last_year = int(annual[-1][0][:4])
        annual = [(p, v) for p, v in annual if int(p[:4]) >= last_year - years]  # 최근 years년 구간
    if len(annual) < 2:
        return None
    first, last = annual[0][1], annual[-1][1]
    if first <= 0 or last <= 0:
        return None
    n = int(annual[-1][0][:4]) - int(annual[0][0][:4])
    if n <= 0:
        return None
    return round(((last / first) ** (1 / n) - 1) * 100, 2)


def get_ytd_growth(data: Dict, key: str) -> Optional[float]:
    """
    최근 누적(예: 2026 상반기) 전년동기 대비 증가율(%). 전년동기가 0 이하면 None.
    최신 분기가 결산월이면 FnGuide 연간표에 누적 컬럼이 없다 (12월 결산은 ~3~5월) →
    최근 2개 연간값의 YoY 로 대체 (직전 연도가 0 이하면 None).
    """
    ytd = data.get('ytd', {}).get(key)
    if ytd:
        cur, prev = ytd.get('cur'), ytd.get('prev')
    else:
        annual = sorted((p, v) for p, v in data.get('annual', {}).get(key, {}).items()
                        if v is not None and not np.isnan(v))
        if len(annual) < 2:
            return None
        prev, cur = annual[-2][1], annual[-1][1]
    if prev is None or prev <= 0 or cur is None:
        return None
    return round((cur / prev - 1) * 100, 2)


def get_inventory_ratio(data: Dict) -> Optional[float]:
    """재고/매출 비율 = 재고자산(4Q평균) / 매출액(TTM) × 100"""
    bs = data.get('balance_avg') or data.get('balance', {})
    inv = bs.get('inventory')
    rev = data.get('revenue_ttm')
    if inv is not None and rev is not None and rev > 0:
        return (inv / rev) * 100
    return None


def get_pegy(market_cap: float, data: Dict) -> Optional[float]:
    """
    PEGY (린치) = PER / (EPS 성장률 + 배당수익률)
    성장률 = 연간 EPS CAGR(최대 5년), 0 < g <= 50 일 때만 (과도한 성장률은 지속 불가로 봄).
    """
    per = get_per_from_data(market_cap, data)
    g = get_cagr(data, 'eps', 5)
    if per is None or per <= 0 or g is None or not 0 < g <= 50:
        return None
    y = data.get('header', {}).get('dividend_yield') or 0
    return per / (g + y)


# ============================================================================
# 캐시 저장/로드
# ============================================================================

def save_fnguide_cache(results: Dict[str, Dict], filepath: str) -> None:
    """FnGuide 수집 결과를 pickle 캐시로 저장"""
    import pickle
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
    size_mb = os.path.getsize(filepath) / (1024 * 1024)
    print(f"캐시 저장: {filepath} ({len(results)}개 종목, {size_mb:.1f}MB)")


def load_fnguide_cache(filepath: str = None) -> Optional[Dict[str, Dict]]:
    """
    FnGuide pickle 캐시 로드.
    filepath이 None이면 .cache/ 에서 최신 fnguide_*.pkl 파일을 찾음.
    """
    import pickle
    import glob as glob_mod

    if filepath and os.path.exists(filepath):
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            print(f"캐시 로드: {filepath} ({len(data)}개 종목)")
            return data
        except Exception as e:
            print(f"캐시 로드 실패: {e}")
            return None

    # filepath이 없으면 .cache/에서 최신 파일 탐색
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cache_dir = os.path.join(script_dir, '.cache')
    if not os.path.exists(cache_dir):
        return None

    pkl_files = sorted(
        glob_mod.glob(os.path.join(cache_dir, 'fnguide_*.pkl')),
        reverse=True
    )
    for pkl_file in pkl_files:
        try:
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)
            print(f"캐시 로드: {pkl_file} ({len(data)}개 종목)")
            return data
        except Exception:
            continue

    return None


# ============================================================================
# 성장률/마진 헬퍼 (test_value_screener.py 등에서 사용)
# ============================================================================

def get_growth_rates_with_ttm(data: Dict, key: str, years: int = 5) -> List[Optional[float]]:
    """
    TTM 값 + 연간 데이터로 YoY 성장률 시리즈를 구성.

    반환: [TTM/최신연간, 최신연간/전년, 전년/전전년, ...]

    Args:
        data: get_financial_data() 결과
        key: 'revenue', 'operating_income', 'net_income' 등
        years: 최대 성장률 개수
    """
    ttm = data.get(f'{key}_ttm')
    annual = data.get('annual', {}).get(key, {})

    # 연도 역순 정렬 (최신 먼저)
    sorted_annual = sorted(annual.items(), key=lambda x: x[0], reverse=True)
    annual_values = [v for _, v in sorted_annual]

    # TTM + 연간 값 결합
    if ttm is not None:
        values = [ttm] + annual_values
    else:
        values = annual_values

    # YoY 성장률 계산
    rates = []
    for i in range(min(years, len(values) - 1)):
        curr, prev = values[i], values[i + 1]
        if curr is not None and prev is not None and prev != 0 and not np.isnan(curr) and not np.isnan(prev):
            rates.append(round(((curr / prev) - 1) * 100, 2))
        else:
            rates.append(None)

    return rates


def get_annual_growth_rates(data: Dict, key: str, years: int = 5) -> List[Optional[float]]:
    """
    연간 데이터만으로 YoY 성장률 계산 (TTM 미포함).

    반환: [최신연간/전년, 전년/전전년, ...]
    """
    annual = data.get('annual', {}).get(key, {})
    sorted_annual = sorted(annual.items(), key=lambda x: x[0], reverse=True)
    values = [v for _, v in sorted_annual]

    rates = []
    for i in range(min(years, len(values) - 1)):
        curr, prev = values[i], values[i + 1]
        if curr is not None and prev is not None and prev != 0 and not np.isnan(curr) and not np.isnan(prev):
            rates.append(round(((curr / prev) - 1) * 100, 2))
        else:
            rates.append(None)

    return rates


def get_operating_margins(data: Dict, years: int = 5) -> List[Optional[float]]:
    """
    영업이익률 시리즈 반환 (TTM + 연간).

    반환: [TTM 영업이익률, 최신연간, 전년, ...]
    """
    margins = []

    # TTM 영업이익률
    rev_ttm = data.get('revenue_ttm')
    op_ttm = data.get('operating_income_ttm')
    if rev_ttm and op_ttm and rev_ttm != 0:
        margins.append(round((op_ttm / rev_ttm) * 100, 2))

    # 연간 영업이익률
    annual_rev = data.get('annual', {}).get('revenue', {})
    annual_op = data.get('annual', {}).get('operating_income', {})

    sorted_years = sorted(annual_rev.keys(), reverse=True)
    for year in sorted_years[:years]:
        rev = annual_rev.get(year)
        op = annual_op.get(year)
        if rev and op and rev != 0:
            margins.append(round((op / rev) * 100, 2))
        else:
            margins.append(None)

    return margins[:years + 1]  # TTM + years개


def get_per_from_data(market_cap: float, data: Dict) -> Optional[float]:
    """PER: FnGuide 헤더값 우선, 없으면 시가총액/순이익TTM으로 계산"""
    header_per = data.get('header', {}).get('per')
    if header_per is not None and header_per > 0:
        return round(header_per, 2)
    ni = data.get('net_income_controlling_ttm') or data.get('net_income_ttm')
    if ni is not None and ni > 0 and market_cap > 0:
        return round(market_cap / ni, 2)
    return None


def get_pbr_from_data(market_cap: float, data: Dict) -> Optional[float]:
    """PBR: FnGuide 헤더값 우선, 없으면 시가총액/자본총계로 계산"""
    header_pbr = data.get('header', {}).get('pbr')
    if header_pbr is not None and header_pbr > 0:
        return round(header_pbr, 2)
    bs = data.get('balance', {})
    eq = bs.get('controlling_equity') or bs.get('total_equity')
    if eq is not None and eq > 0 and market_cap > 0:
        return round(market_cap / eq, 2)
    return None


if __name__ == '__main__':
    # 테스트: 삼성전자
    print("=" * 60)
    print("FnGuide 재무데이터 테스트: 삼성전자 (005930)")
    print("=" * 60)

    data = get_financial_data('005930')
    if data:
        print(f"\n연간 손익계산서:")
        for key, values in data.get('annual', {}).items():
            print(f"  {key}: {values}")

        print(f"\nTTM 데이터:")
        for key in ['revenue_ttm', 'operating_income_ttm', 'net_income_ttm',
                     'gross_profit_ttm', 'tax_expense_ttm', 'finance_cost_ttm', 'cfo_ttm']:
            print(f"  {key}: {data.get(key)}")
        print(f"\nEBIT (영업이익 TTM): {get_ebit(data)}")

        print(f"\n재무상태표:")
        for key, value in data.get('balance', {}).items():
            print(f"  {key}: {value}")

        print(f"\n투하자본: {get_invested_capital(data)}")
        print(f"ROE: {get_roe(data)}")
        print(f"GPA: {get_gpa(data)}")
        print(f"CFO ratio: {get_cfo_ratio(data)}")
    else:
        print("데이터 수집 실패")
