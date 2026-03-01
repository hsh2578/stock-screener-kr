"""
FnGuide 재무제표 수집 모듈 (pd.read_html 기반)

comp.fnguide.com에서 재무제표를 수집합니다.
기존 naver_finance.py의 BeautifulSoup 파싱을 pd.read_html()로 대체합니다.

수집 데이터:
- 연간/분기 손익계산서: 매출액, 매출총이익, 영업이익(EBIT), 당기순이익
- 연간/분기 재무상태표: 총자산, 자본총계, 지배주주지분, 유동자산, 유동부채, 비유동자산, 차입금, 현금, 재고자산
- 연간/분기 현금흐름표: 영업활동현금흐름, 투자활동현금흐름
- 페이지 헤더: PER, 12M PER, 업종 PER, PBR, 배당수익률

사용법:
    from scripts.fnguide_data import get_financial_data
    data = get_financial_data('005930')
"""

import io
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd
import requests

# ============================================================================
# 상수
# ============================================================================

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7',
}

FNGUIDE_FINANCE_URL = 'https://comp.fnguide.com/SVO2/ASP/SVD_Finance.asp'
FNGUIDE_MAIN_URL = 'https://comp.fnguide.com/SVO2/ASP/SVD_Main.asp'

# 재무제표 테이블 인덱스 (SVD_Finance.asp)
TABLE_IDX = {
    'annual_income': 0,     # 연간 손익계산서
    'quarter_income': 1,    # 분기 손익계산서
    'annual_balance': 2,    # 연간 재무상태표
    'quarter_balance': 3,   # 분기 재무상태표
    'annual_cashflow': 4,   # 연간 현금흐름표
    'quarter_cashflow': 5,  # 분기 현금흐름표
}

# 손익계산서 항목 매핑 (FnGuide 한글 → 코드 키)
INCOME_ITEMS = {
    '매출액': 'revenue',
    '매출총이익': 'gross_profit',
    '영업이익': 'operating_income',
    '당기순이익': 'net_income',
    '지배주주순이익': 'net_income_controlling',
    '법인세비용': 'tax_expense',
    '금융원가': 'finance_cost',
}

# 재무상태표 항목 매핑 (key: 영문키, value: 검색할 한글명 리스트)
# FnGuide 요약 테이블에서는 '자산', '자본' 등 짧은 이름 사용
# '계산에참여한계정펼치기' 접미사가 붙는 경우가 있음
BALANCE_SEARCH = {
    'total_assets': ['자산총계', '자산'],
    'total_equity': ['자본총계', '자본'],
    'controlling_equity': ['지배기업주주지분', '지배주주지분'],
    'current_assets': ['유동자산'],
    'current_liabilities': ['유동부채'],
    'non_current_assets': ['비유동자산'],
    'non_current_liabilities': ['비유동부채'],
    'cash': ['현금및현금성자산', '현금'],
    'short_term_debt': ['단기차입금'],
    'long_term_debt': ['장기차입금'],
    'bonds': ['사채'],
    'inventory': ['재고자산'],
}

# SVD_Main.asp table[11] 행 이름 → annual dict key 매핑
MAIN_ANNUAL_ITEMS = {
    '매출액':         'revenue',
    '영업이익':       'operating_income',
    '당기순이익':     'net_income',
    '지배주주순이익': 'net_income_controlling',
    'EPS':            'eps',  # 주당순이익 (5년 EPS 성장률 계산용)
}

# 현금흐름표 항목 매핑
CASHFLOW_ITEMS = {
    '영업활동으로인한현금흐름': 'cfo',
    '영업활동현금흐름': 'cfo',
}

CASHFLOW_INVEST_ITEMS = ['투자활동으로인한현금흐름', '투자활동현금흐름']

# 페이지 헤더 지표 ID 매핑 (corp_group2 영역)
HEADER_METRIC_IDS = {
    'h_per': 'per',
    'h_12m': 'per_12m',
    'h_u_per': 'sector_per',
    'h_pbr': 'pbr',
    'h_rate': 'dividend_yield',
}


# ============================================================================
# 데이터 수집 함수
# ============================================================================

def _fetch_tables(code: str, report_type: str = 'D') -> Optional[tuple]:
    """
    FnGuide 재무제표 페이지에서 테이블들과 HTML을 추출합니다.

    Args:
        code: 종목코드 (6자리)
        report_type: 'D' = 연결, 'I' = 개별

    Returns:
        (6개 테이블 리스트, HTML 텍스트) 튜플 또는 None
    """
    params = {
        'pGB': '1',
        'gicode': f'A{code}',
        'cID': '',
        'MenuYn': 'Y',
        'ReportGB': report_type,
        'NewMenuID': '103',
        'stkGb': '701',
    }

    try:
        resp = requests.get(FNGUIDE_FINANCE_URL, params=params, headers=HEADERS, timeout=20)
        resp.raise_for_status()

        html_text = resp.text
        tables = pd.read_html(
            io.StringIO(html_text), encoding='utf-8', displayed_only=False
        )

        if len(tables) < 6:
            # 개별 재무제표로 재시도
            if report_type == 'D':
                return _fetch_tables(code, 'I')
            return None

        return tables, html_text

    except Exception as e:
        if report_type == 'D':
            # 개별 재무제표로 재시도
            return _fetch_tables(code, 'I')
        return None


def _fetch_tables_main(code: str) -> Optional[pd.DataFrame]:
    """SVD_Main.asp에서 연간 Financial Highlight 테이블만 추출. 실패 시 None.

    div#div15 (Financial Highlight 섹션) 내 6개 테이블 중 인덱스 1번 사용:
      [0] IFRS(연결) Annual+Quarter 전체  [1] IFRS(연결) Annual ← 사용
      [2] IFRS(연결) Net Quarter         [3] IFRS(별도) Annual+Quarter
      [4] IFRS(별도) Annual              [5] IFRS(별도) Net Quarter
    """
    params = {
        'pGB': '1', 'gicode': f'A{code}', 'cID': '',
        'MenuYn': 'Y', 'ReportGB': '', 'NewMenuID': '101', 'stkGb': '701',
    }
    try:
        resp = requests.get(FNGUIDE_MAIN_URL, params=params, headers=HEADERS, timeout=20)
        resp.raise_for_status()
        html = resp.text

        # 전체 파싱 후 인덱스 11 사용 (17개 테이블 중 div#div15의 Annual 연결 테이블)
        # 로컬 파싱 시간: ~0.03s, 네트워크가 실질적 병목
        tables = pd.read_html(io.StringIO(html), encoding='utf-8', displayed_only=False)
        if len(tables) >= 12:
            return tables[11]
    except Exception:
        pass
    return None


def _parse_main_annual(table: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    SVD_Main.asp table[11] (Financial Highlight 연간) 파싱.
    반환: {'revenue': {'2021/12': 값, ...}, 'operating_income': {...}, ...}
    """
    result = {}
    # MultiIndex 컬럼 → 두 번째 레벨(날짜)만 사용
    cols = []
    for c in table.columns:
        if isinstance(c, tuple):
            cols.append(c[1])
        else:
            cols.append(str(c))
    table = table.copy()
    table.columns = cols

    # 첫 번째 열을 인덱스로
    item_col = table.columns[0]
    table = table.set_index(item_col)
    table.index = table.index.astype(str).str.strip()

    # 날짜 컬럼만 필터 (YYYY/MM 형식)
    date_cols = [c for c in table.columns if re.match(r'\d{4}/\d{2}', str(c))]

    for kr_name, en_key in MAIN_ANNUAL_ITEMS.items():
        matched = None
        for idx in table.index:
            if kr_name == idx or kr_name in idx:
                matched = idx
                break
        if matched is None:
            continue
        row = table.loc[matched, date_cols]
        row = pd.to_numeric(row, errors='coerce').dropna()
        if not row.empty:
            result[en_key] = row.to_dict()

    return result


def _parse_header_metrics(html: str) -> Dict[str, float]:
    """
    FnGuide 페이지 헤더에서 PER, 12M PER, 업종 PER, PBR, 배당수익률을 추출합니다.
    corp_group2 영역의 anchor ID로 매핑합니다.
    """
    metrics = {}
    pattern = r'class="tip_in"\s*id="(\w+)"[^>]*>.*?</a>\s*</dt>\s*<dd>([^<]*)</dd>'
    for match in re.finditer(pattern, html, re.DOTALL):
        anchor_id = match.group(1)
        value_str = match.group(2).strip().replace('%', '').replace(',', '')
        metric_key = HEADER_METRIC_IDS.get(anchor_id)
        if metric_key:
            try:
                metrics[metric_key] = float(value_str)
            except (ValueError, TypeError):
                pass
    return metrics


def _clean_table(df: pd.DataFrame) -> pd.DataFrame:
    """테이블 정제: 불필요한 열/행 제거, 인덱스 정리"""
    if df is None or df.empty:
        return pd.DataFrame()

    # 첫 번째 열을 항목명으로 설정
    df = df.copy()

    # MultiIndex 처리
    if isinstance(df.columns, pd.MultiIndex):
        # 마지막 레벨만 사용
        df.columns = [col[-1] if isinstance(col, tuple) else col for col in df.columns]

    # 첫 번째 열을 인덱스로
    first_col = df.columns[0]
    df = df.set_index(first_col)

    # 인덱스 정리 (공백, 특수문자 제거)
    df.index = df.index.astype(str).str.strip()
    df.index = df.index.str.replace(r'\s+', '', regex=True)

    # '전년동기' 등 불필요한 열 제거
    cols_to_drop = [c for c in df.columns if '전년' in str(c) or '동기' in str(c)]
    df = df.drop(columns=cols_to_drop, errors='ignore')

    # 컬럼명에서 날짜 추출 및 정리
    new_cols = []
    for col in df.columns:
        col_str = str(col).strip()
        # YYYY/MM 패턴 추출
        match = re.search(r'(\d{4})[/.](\d{2})', col_str)
        if match:
            new_cols.append(f"{match.group(1)}/{match.group(2)}")
        else:
            new_cols.append(col_str)
    df.columns = new_cols

    # 숫자 변환
    for col in df.columns:
        df[col] = pd.to_numeric(
            df[col].astype(str).str.replace(',', '').str.strip(),
            errors='coerce'
        )

    return df


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
# 메인 수집 함수
# ============================================================================

def get_financial_data(code: str, retry: int = 2) -> Optional[Dict[str, Any]]:
    """
    FnGuide에서 종목의 재무데이터를 수집합니다.

    Args:
        code: 종목코드 (6자리)
        retry: 재시도 횟수

    Returns:
        {
            'code': '005930',
            'annual': {
                'revenue': {'2024/12': 178707, '2023/12': ...},
                'ebit': {...},
                'net_income': {...},
                'gross_profit': {...},
                ...
            },
            'quarter': {
                'revenue': {'2025/09': ..., '2025/06': ..., ...},
                ...
            },
            'balance': {
                'total_assets': value,
                'total_equity': value,
                'current_assets': value,
                'current_liabilities': value,
                'non_current_assets': value,
                'cash': value,
                'total_debt': value,  # 단기+장기+사채
                ...
            },
            'cashflow': {
                'cfo_annual': {'2024/12': value, ...},
                'cfo_ttm': value,  # TTM or latest annual
            }
        }
    """
    for attempt in range(retry):
        try:
            # 두 페이지 동시 요청 (벽시계 시간 동일 유지)
            with ThreadPoolExecutor(max_workers=2) as executor:
                future_finance = executor.submit(_fetch_tables, code)
                future_main    = executor.submit(_fetch_tables_main, code)
                fetch_result = future_finance.result(timeout=30)
                main_tables  = future_main.result(timeout=30)

            if not fetch_result:
                if attempt < retry - 1:
                    time.sleep(0.5)
                continue

            tables, html_text = fetch_result
            if len(tables) < 6:
                if attempt < retry - 1:
                    time.sleep(0.5)
                continue

            result = {
                'code': code,
                'annual': {},
                'quarter': {},
                'balance': {},
                'cashflow': {},
            }

            # 0. 페이지 헤더 지표 (PER, PBR, 배당수익률 등)
            result['header'] = _parse_header_metrics(html_text)

            # 1. 연간/분기 손익계산서 (먼저 둘 다 파싱)
            annual_income = _clean_table(tables[TABLE_IDX['annual_income']])
            quarter_income = _clean_table(tables[TABLE_IDX['quarter_income']])

            # 미완성 연간 컬럼 제거: 최신 연간 컬럼 결산월이 /12가 아니면 분기 누적값
            # (한국 상장사 95%+는 12월 결산 → /12 외 패턴은 미완성 연간으로 처리)
            if not annual_income.empty:
                date_annual_cols = sorted(
                    [c for c in annual_income.columns if re.match(r'\d{4}/\d{2}', str(c))],
                    reverse=True,
                )
                if date_annual_cols and not date_annual_cols[0].endswith('/12'):
                    annual_income = annual_income.drop(columns=[date_annual_cols[0]], errors='ignore')

            if not annual_income.empty:
                for kr_name, en_key in INCOME_ITEMS.items():
                    series = _extract_item(annual_income, [kr_name])
                    if not series.empty:
                        result['annual'][en_key] = series.dropna().to_dict()

            if not quarter_income.empty:
                for kr_name, en_key in INCOME_ITEMS.items():
                    series = _extract_item(quarter_income, [kr_name])
                    if not series.empty:
                        result['quarter'][en_key] = series.dropna().to_dict()

            # 3. 재무상태표 (최신 분기 우선, 없으면 연간)
            quarter_balance = _clean_table(tables[TABLE_IDX['quarter_balance']])
            annual_balance = _clean_table(tables[TABLE_IDX['annual_balance']])

            bs_source = quarter_balance if not quarter_balance.empty else annual_balance
            if not bs_source.empty:
                for en_key, kr_names in BALANCE_SEARCH.items():
                    series = _extract_item(bs_source, kr_names)
                    if not series.empty:
                        valid = series.dropna()
                        if len(valid) > 0:
                            # 최신값 (EV, PBR 등 시점 기준 계산용)
                            result['balance'][en_key] = float(valid.iloc[0])

                # 총차입금 계산
                short_debt = result['balance'].get('short_term_debt', 0) or 0
                long_debt = result['balance'].get('long_term_debt', 0) or 0
                bonds = result['balance'].get('bonds', 0) or 0
                result['balance']['total_debt'] = short_debt + long_debt + bonds

            # 재무상태표 평균값 (ROE, GPA, CFO, IC 등 Flow/Stock 혼합 비율용)
            # Stock 항목은 시점 잔액이므로, Flow(TTM) 분자와 매칭하려면
            # 최근 4분기 평균을 사용해야 정확
            result['balance_avg'] = {}
            bs_q = quarter_balance if not quarter_balance.empty else annual_balance
            if not bs_q.empty:
                for en_key, kr_names in BALANCE_SEARCH.items():
                    series = _extract_item(bs_q, kr_names)
                    if not series.empty:
                        valid = series.dropna()
                        if len(valid) > 0:
                            # 최근 4분기 평균 (4개 미만이면 있는 만큼 평균)
                            avg_val = float(valid.head(4).mean())
                            result['balance_avg'][en_key] = avg_val

            # 4. 현금흐름표
            annual_cf = _clean_table(tables[TABLE_IDX['annual_cashflow']])
            quarter_cf = _clean_table(tables[TABLE_IDX['quarter_cashflow']])

            if not annual_cf.empty:
                cfo_series = _extract_item(annual_cf, list(CASHFLOW_ITEMS.keys()))
                if not cfo_series.empty:
                    result['cashflow']['cfo_annual'] = cfo_series.dropna().to_dict()

            if not quarter_cf.empty:
                cfo_q_series = _extract_item(quarter_cf, list(CASHFLOW_ITEMS.keys()))
                if not cfo_q_series.empty:
                    result['cashflow']['cfo_quarter'] = cfo_q_series.dropna().to_dict()

            # 4-2. 투자활동현금흐름
            if not annual_cf.empty:
                invest_series = _extract_item(annual_cf, CASHFLOW_INVEST_ITEMS)
                if not invest_series.empty:
                    result['cashflow']['invest_cf_annual'] = invest_series.dropna().to_dict()

            if not quarter_cf.empty:
                invest_q_series = _extract_item(quarter_cf, CASHFLOW_INVEST_ITEMS)
                if not invest_q_series.empty:
                    result['cashflow']['invest_cf_quarter'] = invest_q_series.dropna().to_dict()

            # SVD_Main.asp 5년 데이터로 연간 보충
            # - 기존 키(revenue 등): 이전 연도만 추가 (덮어쓰기 방지, 미발표 누적값 방지)
            # - 새 키(eps 등): SVD_Finance에 없으므로 전체 기간 추가
            if main_tables is not None:
                try:
                    main_annual = _parse_main_annual(main_tables)
                    existing_years = set()
                    for v in result.get('annual', {}).values():
                        existing_years.update(v.keys())
                    min_existing = min(existing_years) if existing_years else '9999/99'
                    for en_key, year_dict in main_annual.items():
                        if en_key not in result['annual']:
                            # 새 지표: 전체 기간 추가 (단, /12 결산월만)
                            result['annual'][en_key] = {
                                p: v for p, v in year_dict.items()
                                if p.endswith('/12')
                            }
                        else:
                            # 기존 지표: min_existing보다 이전 연도만 보충
                            for period, value in year_dict.items():
                                if period < min_existing:
                                    result['annual'][en_key][period] = value
                except Exception:
                    pass  # Main 파싱 실패해도 기존 4년 데이터로 계속

            # TTM 계산
            _calculate_ttm(result)

            return result

        except Exception as e:
            if attempt < retry - 1:
                time.sleep(0.5)
            continue

    return None


def _calculate_ttm(data: Dict[str, Any]) -> None:
    """
    TTM (Trailing Twelve Months) 계산.
    분기 데이터가 4개 이상 있으면 최근 4분기 합산.
    없으면 최신 연간 데이터 사용.
    """
    for key in ['revenue', 'operating_income', 'net_income', 'gross_profit',
                 'net_income_controlling', 'tax_expense', 'finance_cost']:
        annual = data['annual'].get(key, {})
        quarter = data['quarter'].get(key, {})

        ttm_value = None

        # 분기 데이터로 TTM 시도
        if len(quarter) >= 4:
            # 날짜순 정렬 (최신 먼저)
            sorted_quarters = sorted(quarter.items(), key=lambda x: x[0], reverse=True)
            recent_4 = [v for _, v in sorted_quarters[:4] if v is not None and not np.isnan(v)]
            if len(recent_4) == 4:
                ttm_value = sum(recent_4)

        # TTM 불가 → 최신 연간 데이터
        if ttm_value is None and annual:
            sorted_annual = sorted(annual.items(), key=lambda x: x[0], reverse=True)
            for _, v in sorted_annual:
                if v is not None and not np.isnan(v):
                    ttm_value = v
                    break

        if ttm_value is not None:
            data[f'{key}_ttm'] = ttm_value

    # CFO TTM
    cfo_annual = data['cashflow'].get('cfo_annual', {})
    cfo_quarter = data['cashflow'].get('cfo_quarter', {})

    cfo_ttm = None
    if len(cfo_quarter) >= 4:
        sorted_q = sorted(cfo_quarter.items(), key=lambda x: x[0], reverse=True)
        recent_4 = [v for _, v in sorted_q[:4] if v is not None and not np.isnan(v)]
        if len(recent_4) == 4:
            cfo_ttm = sum(recent_4)

    if cfo_ttm is None and cfo_annual:
        sorted_a = sorted(cfo_annual.items(), key=lambda x: x[0], reverse=True)
        for _, v in sorted_a:
            if v is not None and not np.isnan(v):
                cfo_ttm = v
                break

    if cfo_ttm is not None:
        data['cfo_ttm'] = cfo_ttm

    # 투자활동CF TTM
    inv_annual = data['cashflow'].get('invest_cf_annual', {})
    inv_quarter = data['cashflow'].get('invest_cf_quarter', {})

    inv_ttm = None
    if len(inv_quarter) >= 4:
        sorted_q = sorted(inv_quarter.items(), key=lambda x: x[0], reverse=True)
        recent_4 = [v for _, v in sorted_q[:4] if v is not None and not np.isnan(v)]
        if len(recent_4) == 4:
            inv_ttm = sum(recent_4)

    if inv_ttm is None and inv_annual:
        sorted_a = sorted(inv_annual.items(), key=lambda x: x[0], reverse=True)
        for _, v in sorted_a:
            if v is not None and not np.isnan(v):
                inv_ttm = v
                break

    if inv_ttm is not None:
        data['invest_cf_ttm'] = inv_ttm


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
    """투하자본 = 유동자산 - 유동부채 + 비유동자산 (4분기 평균)"""
    bs = data.get('balance_avg') or data.get('balance', {})
    ca = bs.get('current_assets')
    cl = bs.get('current_liabilities')
    nca = bs.get('non_current_assets')

    if ca is not None and cl is not None and nca is not None:
        return ca - cl + nca
    return None


def get_ev(market_cap: float, data: Dict) -> Optional[float]:
    """
    기업가치 (EV) = 시가총액 + 총부채 - 여유자금
    총부채 = 유동부채 + 비유동부채
    여유자금 = 현금 - MAX(0, 유동부채 - 유동자산 + 현금)
    현금 데이터 없으면 여유자금=0으로 처리
    """
    bs = data.get('balance', {})
    cl = bs.get('current_liabilities', 0) or 0
    ncl = bs.get('non_current_liabilities', 0) or 0
    total_liabilities = cl + ncl

    # 여유자금 계산
    cash = bs.get('cash', 0) or 0
    ca = bs.get('current_assets', 0) or 0
    if cash > 0:
        working_capital_need = max(0, cl - ca + cash)
        excess_cash = cash - working_capital_need
    else:
        excess_cash = 0

    ev = market_cap + total_liabilities - excess_cash
    return ev if ev > 0 else market_cap


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
    if eq is not None and eq > 0 and (long_debt + bonds) > 0:
        return ((long_debt + bonds) / eq) * 100
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
    """FCF = CFO(TTM) + 투자활동CF(TTM)  (투자CF는 음수이므로 + 연산)"""
    cfo = data.get('cfo_ttm')
    inv_cf = data.get('invest_cf_ttm')
    if cfo is not None and inv_cf is not None:
        return cfo + inv_cf
    return None


def get_inventory_ratio(data: Dict) -> Optional[float]:
    """재고/매출 비율 = 재고자산(4Q평균) / 매출액(TTM) × 100"""
    bs = data.get('balance_avg') or data.get('balance', {})
    inv = bs.get('inventory')
    rev = data.get('revenue_ttm')
    if inv is not None and rev is not None and rev > 0:
        return (inv / rev) * 100
    return None


def get_peg(market_cap: float, data: Dict) -> Optional[float]:
    """PEG = PER / EPS성장률(3년평균). 성장률이 음수면 None."""
    per = get_per_from_data(market_cap, data)
    if per is None or per <= 0:
        return None

    growth_rates = get_annual_growth_rates(data, 'eps', years=3)
    if not growth_rates:  # eps 데이터 없으면 net_income으로 fallback (구버전 캐시 호환)
        growth_rates = get_growth_rates_with_ttm(data, 'net_income', years=3)
    valid_rates = [r for r in growth_rates if r is not None and r > 0]
    if not valid_rates:
        return None

    avg_growth = sum(valid_rates) / len(valid_rates)
    if avg_growth <= 0:
        return None

    return per / avg_growth


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
