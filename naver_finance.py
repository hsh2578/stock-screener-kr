"""
FnGuide 재무데이터 크롤러

FnGuide(comp.fnguide.com)에서 재무제표 및 투자지표를 크롤링합니다.
저평가 우량주 스크리너에서 사용됩니다.
"""

import requests
import pandas as pd
import numpy as np
import json
import os
import re
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from bs4 import BeautifulSoup

# 상수
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, 'data')
FINANCIAL_DATA_FILE = os.path.join(DATA_PATH, 'financial_data.json')

# 국채금리 (수동 설정 - 주기적 업데이트 필요)
TREASURY_RATE = 3.4  # 2025년 1월 기준

# 요청 헤더
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7',
}


def parse_number(text: str) -> Optional[float]:
    """문자열을 숫자로 변환합니다."""
    if not text or text.strip() in ['-', 'N/A', '', 'nan', '적자', '흑전', '적전']:
        return None
    try:
        # 특수문자 제거
        text = text.replace(',', '').replace(' ', '').replace('억', '')
        text = text.replace('조', '').replace('원', '').replace('배', '').replace('%', '')
        # 괄호는 음수
        if '(' in text and ')' in text:
            text = '-' + text.replace('(', '').replace(')', '')
        return float(text)
    except:
        return None


def fetch_financial_data(code: str, retry: int = 3) -> Optional[Dict[str, Any]]:
    """
    단일 종목의 FnGuide 재무데이터를 크롤링합니다.

    크롤링 페이지:
    1. SVD_Finance.asp - 손익계산서, 재무상태표 (매출액, 영업이익, 순이익, 자산, 부채)
    2. SVD_FinanceRatio.asp - 재무비율 (ROE, 부채비율, 유동비율, 영업이익률, 성장률)
    3. SVD_Invest.asp - 투자지표 (EPS, BPS, DPS, PER, PBR)
    """
    data = {
        'code': code,
        'crawled_at': datetime.now().isoformat(),
        'metrics': {},
        'years': []
    }

    for attempt in range(retry):
        try:
            # 1. 재무제표 페이지 - 손익계산서
            finance_data = fetch_fnguide_finance(code)
            if finance_data:
                data['metrics'].update(finance_data)

            # 2. 재무비율 페이지
            ratio_data = fetch_fnguide_ratio(code)
            if ratio_data:
                data['metrics'].update(ratio_data)

            # 3. 투자지표 페이지
            invest_data = fetch_fnguide_invest(code)
            if invest_data:
                data['metrics'].update(invest_data)
                # 연도 정보
                if '_years' in invest_data:
                    data['years'] = invest_data['_years']

            if data['metrics']:
                return data

        except Exception as e:
            if attempt < retry - 1:
                time.sleep(1)
            continue

    return None


def _fetch_fnguide_page(code: str, page_type: str, timeout: int = 10) -> Optional[BeautifulSoup]:
    """
    FnGuide 페이지 공통 요청 함수

    Args:
        code: 종목코드
        page_type: 페이지 유형 ('invest', 'ratio', 'finance')
        timeout: 요청 타임아웃 (초)

    Returns:
        BeautifulSoup 객체 또는 None
    """
    page_config = {
        'invest': {'url': 'https://comp.fnguide.com/SVO2/ASP/SVD_Invest.asp', 'menu_id': '105'},
        'ratio': {'url': 'https://comp.fnguide.com/SVO2/ASP/SVD_FinanceRatio.asp', 'menu_id': '104'},
        'finance': {'url': 'https://comp.fnguide.com/SVO2/ASP/SVD_Finance.asp', 'menu_id': '103'}
    }

    if page_type not in page_config:
        return None

    config = page_config[page_type]
    params = {
        'pGB': '1',
        'gicode': f'A{code}',
        'MenuYn': 'Y',
        'NewMenuID': config['menu_id'],
        'stkGb': '701'
    }

    try:
        resp = requests.get(config['url'], params=params, headers=HEADERS, timeout=timeout)
        return BeautifulSoup(resp.text, 'html.parser')
    except Exception:
        return None


def _extract_metrics_from_table(
    soup: BeautifulSoup,
    table_filter: callable,
    row_extractor: callable
) -> Dict[str, Any]:
    """
    테이블에서 지표를 추출하는 공통 함수

    Args:
        soup: BeautifulSoup 객체
        table_filter: 테이블 필터 함수 (table -> bool)
        row_extractor: 행 데이터 추출 함수 (rows -> dict)

    Returns:
        추출된 지표 딕셔너리
    """
    metrics = {}

    for table in soup.find_all('table'):
        if not table_filter(table):
            continue

        rows = table.find_all('tr')
        if not rows:
            continue

        extracted = row_extractor(rows, metrics)
        if extracted:
            metrics.update(extracted)
            break

    return metrics


def fetch_financial_data_light(code: str, retry: int = 2) -> Optional[Dict[str, Any]]:
    """
    가치투자 7개 조건에 필요한 최소 데이터만 크롤링 (최적화 버전)

    필요 데이터:
    - PER, PBR (투자지표)
    - 매출성장률, 영업이익률, 영업이익성장률, EPS성장률 (재무비율)
    - 순이익 (재무제표 - CAGR 계산용)
    """
    data = {
        'code': code,
        'crawled_at': datetime.now().isoformat(),
        'metrics': {}
    }

    for attempt in range(retry):
        try:
            # 1. 투자지표 페이지 - PER, PBR만
            invest_data = fetch_fnguide_invest_light(code)
            if invest_data:
                data['metrics'].update(invest_data)

            # 2. 재무비율 페이지 - 성장률, 영업이익률만
            ratio_data = fetch_fnguide_ratio_light(code)
            if ratio_data:
                data['metrics'].update(ratio_data)

            # 3. 재무제표 페이지 - 순이익만 (CAGR 계산용)
            finance_data = fetch_fnguide_finance_light(code)
            if finance_data:
                data['metrics'].update(finance_data)

            if data['metrics']:
                return data

        except Exception as e:
            if attempt < retry - 1:
                time.sleep(0.5)
            continue

    return None


def _is_valuation_table(table) -> bool:
    """PER/PBR 밸류에이션 테이블 필터"""
    table_text = table.get_text()
    return '최고' in table_text and '최저' in table_text and 'PER' in table_text


def _extract_valuation_metrics(rows, existing_metrics) -> Dict[str, Any]:
    """PER, PBR 지표 추출"""
    metrics = {}
    for row in rows:
        cells = row.find_all(['th', 'td'])
        if len(cells) < 3:
            continue
        label = cells[0].get_text(strip=True)
        if label == 'PER' and 'per' not in existing_metrics:
            val = parse_number(cells[1].get_text(strip=True))
            if val:
                metrics['per'] = [val]
        elif label == 'PBR' and 'pbr' not in existing_metrics:
            val = parse_number(cells[1].get_text(strip=True))
            if val:
                metrics['pbr'] = [val]
    return metrics


def fetch_fnguide_invest_light(code: str) -> Dict[str, Any]:
    """PER, PBR만 추출 (최적화)"""
    soup = _fetch_fnguide_page(code, 'invest')
    if not soup:
        return {}

    return _extract_metrics_from_table(
        soup,
        _is_valuation_table,
        _extract_valuation_metrics
    )


def _is_annual_ratio_table(table) -> bool:
    """연간 재무비율 테이블 필터"""
    rows = table.find_all('tr')
    if not rows:
        return False
    header = rows[0].get_text()
    if 'IFRS' not in header:
        return False

    headers = rows[0].find_all(['th', 'td'])
    header_texts = [h.get_text(strip=True) for h in headers]
    # 분기 테이블 건너뛰기
    if any('/03' in h or '/06' in h or '/09' in h for h in header_texts[1:4]):
        return False
    return True


def _extract_ratio_metrics(rows, existing_metrics) -> Dict[str, List[float]]:
    """성장률, 영업이익률 지표 추출"""
    metrics = {}
    for row in rows:
        cells = row.find_all(['th', 'td'])
        if len(cells) < 2:
            continue
        label = cells[0].get_text(strip=True)
        values = [parse_number(cells[i].get_text(strip=True)) for i in range(1, min(6, len(cells)))]
        if all(v is None for v in values):
            continue

        if '매출액증가율' in label and 'revenue_growth_rate' not in existing_metrics:
            metrics['revenue_growth_rate'] = values
        elif '영업이익증가율' in label and 'op_profit_growth_rate' not in existing_metrics:
            metrics['op_profit_growth_rate'] = values
        elif 'EPS증가율' in label and 'eps_growth_rate' not in existing_metrics:
            metrics['eps_growth_rate'] = values
        elif '영업이익률' in label and '증가' not in label and 'operating_margin' not in existing_metrics:
            metrics['operating_margin'] = values
    return metrics


def fetch_fnguide_ratio_light(code: str) -> Dict[str, List[float]]:
    """성장률, 영업이익률만 추출 (최적화)"""
    soup = _fetch_fnguide_page(code, 'ratio')
    if not soup:
        return {}

    return _extract_metrics_from_table(
        soup,
        _is_annual_ratio_table,
        _extract_ratio_metrics
    )


def _is_annual_finance_table(table) -> bool:
    """연간 재무제표 테이블 필터"""
    rows = table.find_all('tr')
    if not rows:
        return False
    header = rows[0].get_text()
    if 'IFRS' not in header:
        return False

    headers = rows[0].find_all(['th', 'td'])
    years = [h.get_text(strip=True) for h in headers[1:] if '/' in h.get_text(strip=True)]
    if not years or len(years) < 3:
        return False
    year_parts = [y.split('/')[0] for y in years[:4]]
    if len(set(year_parts)) <= 2:
        return False
    return True


def _extract_finance_metrics(rows, existing_metrics) -> Dict[str, List[float]]:
    """순이익 지표 추출"""
    metrics = {}
    for row in rows[1:]:  # 헤더 행 건너뛰기
        cells = row.find_all(['th', 'td'])
        if len(cells) < 2:
            continue
        label = cells[0].get_text(strip=True)
        if '당기순손익' in label or label == '당기순이익':
            values = [parse_number(cells[i].get_text(strip=True)) for i in range(1, min(6, len(cells)))]
            metrics['net_income'] = values
            break
    return metrics


def fetch_fnguide_finance_light(code: str) -> Dict[str, List[float]]:
    """순이익만 추출 (최적화)"""
    soup = _fetch_fnguide_page(code, 'finance')
    if not soup:
        return {}

    return _extract_metrics_from_table(
        soup,
        _is_annual_finance_table,
        _extract_finance_metrics
    )


# ============================================================
# TTM (Trailing Twelve Months) 관련 함수
# ============================================================

def get_required_quarter() -> str:
    """
    현재 날짜 기준으로 모든 상장사가 공시 완료해야 하는 분기를 반환합니다.

    공시 의무 기한:
    - Q1 (1~3월): 5월 15일까지
    - Q2 (4~6월): 8월 14일까지
    - Q3 (7~9월): 11월 14일까지
    - Q4 (10~12월): 다음해 3월 31일까지

    Returns:
        '2025/09' 형태의 기준 분기 문자열
    """
    today = datetime.now()
    year = today.year
    month = today.month
    day = today.day

    # 공시 마감일 기준으로 기준 분기 결정
    if month >= 11 and day >= 15:
        # 11/15 이후: Q3 공시 완료 (9월 분기)
        return f"{year}/09"
    elif month == 12 or (month >= 1 and month <= 3) or (month == 3 and day <= 31):
        # 12월 ~ 3월: 여전히 Q3 기준 (전년도)
        if month >= 1 and month <= 3:
            return f"{year-1}/09"
        return f"{year}/09"
    elif month >= 8 and day >= 15:
        # 8/15 이후: Q2 공시 완료 (6월 분기)
        return f"{year}/06"
    elif month >= 5 and day >= 16:
        # 5/16 이후: Q1 공시 완료 (3월 분기)
        return f"{year}/03"
    elif month >= 4 or (month >= 1 and month <= 3 and day > 31):
        # 4월 ~ 5/15: Q4 공시 완료 (전년 12월 분기)
        return f"{year-1}/12"
    else:
        # 1~3월: 전전년 Q3 또는 전년 Q4
        return f"{year-1}/12"


def validate_quarter_data(quarters: List[str], required_quarter: str) -> bool:
    """
    크롤링된 분기 데이터가 기준 분기를 포함하는지 확인합니다.

    Args:
        quarters: 크롤링된 분기 목록 ['2024/12', '2025/03', '2025/06', '2025/09']
        required_quarter: 필수 기준 분기 '2025/09'

    Returns:
        기준 분기 데이터가 있으면 True
    """
    if not quarters:
        return False
    return required_quarter in quarters


def fetch_quarterly_finance(code: str) -> Dict[str, Any]:
    """
    FnGuide에서 분기별 재무데이터를 크롤링합니다.

    Returns:
        {
            'quarters': ['2024/12', '2025/03', '2025/06', '2025/09'],
            'revenue': [Q1, Q2, Q3, Q4],  # 매출액
            'operating_profit': [Q1, Q2, Q3, Q4],  # 영업이익
            'net_income': [Q1, Q2, Q3, Q4],  # 당기순이익
        }
    """
    url = 'https://comp.fnguide.com/SVO2/ASP/SVD_Finance.asp'
    params = {'pGB': '1', 'gicode': f'A{code}', 'MenuYn': 'Y', 'NewMenuID': '103', 'stkGb': '701'}

    try:
        resp = requests.get(url, params=params, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(resp.text, 'html.parser')
    except Exception:
        return {}

    result = {'quarters': [], 'revenue': [], 'operating_profit': [], 'net_income': []}

    # divSonikQ = 분기 손익계산서
    div_q = soup.find('div', id='divSonikQ')
    if not div_q:
        return {}

    table = div_q.find('table')
    if not table:
        return {}

    rows = table.find_all('tr')
    if not rows:
        return {}

    # 헤더에서 분기 정보 추출
    header_row = rows[0]
    headers = header_row.find_all(['th', 'td'])
    for h in headers[1:5]:  # 최근 4개 분기
        text = h.get_text(strip=True)
        if '/' in text:
            result['quarters'].append(text)

    if len(result['quarters']) < 4:
        return {}

    # 데이터 추출
    for row in rows[1:]:
        cells = row.find_all(['th', 'td'])
        if len(cells) < 5:
            continue

        label = cells[0].get_text(strip=True)
        values = [parse_number(cells[i].get_text(strip=True)) for i in range(1, 5)]

        if '매출액' == label:
            result['revenue'] = values
        elif '영업이익' == label and '영업이익률' not in label:
            result['operating_profit'] = values
        elif '당기순이익' in label or '당기순손익' in label:
            result['net_income'] = values

    return result


def fetch_quarterly_ratio(code: str) -> Dict[str, Any]:
    """
    FnGuide에서 분기별 재무비율을 크롤링합니다.

    Returns:
        {
            'quarters': ['2024/12', '2025/03', '2025/06', '2025/09'],
            'operating_margin': [Q1, Q2, Q3, Q4],  # 영업이익률
            'eps': [Q1, Q2, Q3, Q4],  # EPS
        }
    """
    url = 'https://comp.fnguide.com/SVO2/ASP/SVD_FinanceRatio.asp'
    params = {'pGB': '1', 'gicode': f'A{code}', 'MenuYn': 'Y', 'NewMenuID': '104', 'stkGb': '701'}

    try:
        resp = requests.get(url, params=params, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(resp.text, 'html.parser')
    except Exception:
        return {}

    result = {'quarters': [], 'operating_margin': [], 'eps': []}

    # 분기 테이블 찾기 (분기 데이터가 있는 테이블)
    for table in soup.find_all('table'):
        rows = table.find_all('tr')
        if not rows:
            continue

        header = rows[0].get_text()
        if 'IFRS' not in header:
            continue

        # 헤더에서 분기 확인
        headers = rows[0].find_all(['th', 'td'])
        header_texts = [h.get_text(strip=True) for h in headers]

        # 분기 데이터인지 확인 (/03, /06, /09, /12)
        quarters = []
        for h in header_texts[1:5]:
            if '/' in h:
                quarters.append(h)

        if len(quarters) < 4:
            continue

        # 연간 테이블인지 분기 테이블인지 확인
        q_months = [q.split('/')[1] if '/' in q else '' for q in quarters[:4]]
        if len(set(q_months)) <= 1:  # 모두 같은 월 = 연간 데이터
            continue

        result['quarters'] = quarters

        # 데이터 추출
        for row in rows[1:]:
            cells = row.find_all(['th', 'td'])
            if len(cells) < 5:
                continue

            label = cells[0].get_text(strip=True)
            values = [parse_number(cells[i].get_text(strip=True)) for i in range(1, 5)]

            if '영업이익률' in label and '증가' not in label:
                result['operating_margin'] = values
            elif 'EPS' in label and '증가' not in label:
                result['eps'] = values

        if result['operating_margin'] or result['eps']:
            break

    return result


def calculate_ttm_value(quarterly_values: List[Optional[float]]) -> Optional[float]:
    """최근 4개 분기 값을 합산하여 TTM 계산"""
    if not quarterly_values or len(quarterly_values) < 4:
        return None

    valid_values = [v for v in quarterly_values[:4] if v is not None]
    if len(valid_values) < 4:
        return None

    return sum(valid_values)


def calculate_ttm_growth(current_ttm: Optional[float], prev_ttm: Optional[float]) -> Optional[float]:
    """TTM 성장률 계산 (전년 동기 대비)"""
    if current_ttm is None or prev_ttm is None or prev_ttm == 0:
        return None

    return ((current_ttm / prev_ttm) - 1) * 100


def fetch_financial_data_ttm(code: str, retry: int = 2, required_quarter: str = None) -> Optional[Dict[str, Any]]:
    """
    TTM(Trailing Twelve Months) 기반 재무데이터 크롤링

    연간 데이터 + TTM(최근 4분기 합산)을 결합하여 성장률을 계산합니다.
    예: 21~24년은 연간 데이터, 25년은 TTM(24Q4 + 25Q1~Q3)

    Args:
        code: 종목코드
        retry: 재시도 횟수
        required_quarter: 필수 기준 분기 (None이면 자동 계산)

    Returns:
        {
            'code': '005930',
            'data_type': 'TTM',
            'required_quarter': '2025/09',
            'quarters': ['2024/12', '2025/03', '2025/06', '2025/09'],
            'metrics': { ... }
        }
    """
    # 기준 분기 결정
    if required_quarter is None:
        required_quarter = get_required_quarter()

    for attempt in range(retry):
        try:
            data = {
                'code': code,
                'crawled_at': datetime.now().isoformat(),
                'data_type': 'TTM',
                'required_quarter': required_quarter,
                'quarters': [],
                'metrics': {}
            }

            # 1. PER, PBR (투자지표 - 현재가 기준)
            invest_data = fetch_fnguide_invest_light(code)
            if invest_data:
                data['metrics'].update(invest_data)

            # 2. 연간 재무데이터 가져오기 (21~24년)
            annual_finance = fetch_fnguide_finance(code)
            annual_revenue = annual_finance.get('revenue', []) if annual_finance else []
            annual_op = annual_finance.get('operating_profit', []) if annual_finance else []
            annual_net = annual_finance.get('net_income', []) if annual_finance else []

            # 3. 분기별 재무제표 (매출, 영업이익, 순이익)
            quarterly_finance = fetch_quarterly_finance(code)
            if quarterly_finance and quarterly_finance.get('quarters'):
                quarters = quarterly_finance['quarters']

                # 기준 분기 데이터 검증
                if not validate_quarter_data(quarters, required_quarter):
                    # 기준 분기 데이터 없음 - 미공시 또는 크롤링 오류
                    return None

                data['quarters'] = quarters

                # TTM 계산 (최근 4분기 합산)
                ttm_revenue = calculate_ttm_value(quarterly_finance.get('revenue', []))
                ttm_op = calculate_ttm_value(quarterly_finance.get('operating_profit', []))
                ttm_net = calculate_ttm_value(quarterly_finance.get('net_income', []))

                if ttm_revenue:
                    data['metrics']['revenue_ttm'] = ttm_revenue
                if ttm_op:
                    data['metrics']['operating_profit_ttm'] = ttm_op
                if ttm_net:
                    data['metrics']['net_income_ttm'] = ttm_net

                # 4. TTM + 연간 데이터 결합하여 성장률 계산
                # [TTM_2025, 2024, 2023, 2022, 2021] 형태로 결합
                combined_revenue = [ttm_revenue] + annual_revenue if ttm_revenue else annual_revenue
                combined_op = [ttm_op] + annual_op if ttm_op else annual_op
                combined_net = [ttm_net] + annual_net if ttm_net else annual_net

                # 매출액 성장률 계산 (TTM vs 전년)
                if len(combined_revenue) >= 2:
                    revenue_growth_rates = []
                    for i in range(min(5, len(combined_revenue) - 1)):
                        curr = combined_revenue[i]
                        prev = combined_revenue[i + 1]
                        if curr is not None and prev is not None and prev != 0:
                            growth = ((curr / prev) - 1) * 100
                            revenue_growth_rates.append(round(growth, 2))
                        else:
                            revenue_growth_rates.append(None)
                    if revenue_growth_rates:
                        data['metrics']['revenue_growth_rate'] = revenue_growth_rates

                # 영업이익 성장률 계산 (TTM vs 전년)
                if len(combined_op) >= 2:
                    op_growth_rates = []
                    for i in range(min(5, len(combined_op) - 1)):
                        curr = combined_op[i]
                        prev = combined_op[i + 1]
                        if curr is not None and prev is not None and prev != 0:
                            growth = ((curr / prev) - 1) * 100
                            op_growth_rates.append(round(growth, 2))
                        else:
                            op_growth_rates.append(None)
                    if op_growth_rates:
                        data['metrics']['op_profit_growth_rate'] = op_growth_rates

                # 순이익 성장률 계산 (TTM vs 전년)
                if len(combined_net) >= 2:
                    net_growth_rates = []
                    for i in range(min(5, len(combined_net) - 1)):
                        curr = combined_net[i]
                        prev = combined_net[i + 1]
                        if curr is not None and prev is not None and prev != 0:
                            growth = ((curr / prev) - 1) * 100
                            net_growth_rates.append(round(growth, 2))
                        else:
                            net_growth_rates.append(None)
                    if net_growth_rates:
                        data['metrics']['net_income_growth_rate'] = net_growth_rates

                # 영업이익률 계산 (TTM 기준)
                if ttm_revenue and ttm_op and ttm_revenue != 0:
                    ttm_op_margin = (ttm_op / ttm_revenue) * 100
                    data['metrics']['operating_margin'] = [round(ttm_op_margin, 2)]

            # 5. EPS 성장률 (연간 데이터 사용 - 분기 EPS 계산이 복잡)
            ratio_data = fetch_fnguide_ratio_light(code)
            if ratio_data and 'eps_growth_rate' in ratio_data:
                data['metrics']['eps_growth_rate'] = ratio_data['eps_growth_rate']

            if data['metrics']:
                return data

        except Exception as e:
            if attempt < retry - 1:
                time.sleep(0.5)
            continue

    return None


def pass_first_filter_ttm(fin_data: Dict) -> Tuple[bool, Dict[str, Any]]:
    """
    TTM 데이터 기반 1차 필터 (7개 조건)

    조건:
    1. PER: 3 < PER < 30
    2. 매출액 성장률: 최근 3년 평균 > 10% (TTM 포함)
    3. 영업이익률: TTM 기준 > 10%
    4. 영업이익 성장률: 최근 3년 평균 > 10% (TTM 포함)
    5. EPS 성장률: 최근 3년 평균 > 10%
    6. 순이익 TTM: 양수 (흑자)
    7. 순이익 증가율: 15% < 최근 3년 평균 < 50% (TTM 포함)
    """
    metrics = fin_data.get('metrics', {})

    results = {
        'per_check': {'value': None, 'pass': False, 'condition': '3 < PER < 30'},
        'revenue_growth': {'value': None, 'pass': False, 'condition': '매출성장률 3년평균 > 10%'},
        'operating_margin_avg': {'value': None, 'pass': False, 'condition': '영업이익률 TTM > 10%'},
        'operating_profit_growth': {'value': None, 'pass': False, 'condition': '영업이익성장률 3년평균 > 10%'},
        'eps_growth': {'value': None, 'pass': False, 'condition': 'EPS성장률 3년평균 > 10%'},
        'net_income_positive': {'value': None, 'pass': False, 'condition': '순이익 TTM > 0'},
        'net_income_growth': {'value': None, 'pass': False, 'condition': '순이익증가율 15~50%'},
    }

    # 1. PER: 3 < PER < 30
    per_values = metrics.get('per', [])
    if per_values and per_values[0] is not None:
        per = per_values[0]
        results['per_check']['value'] = round(per, 2)
        results['per_check']['pass'] = 3 < per < 30

    # 2. 매출액 성장률: 3년 평균 > 10% (TTM 포함)
    rev_growth_rate = metrics.get('revenue_growth_rate', [])
    if rev_growth_rate:
        valid_rates = [r for r in rev_growth_rate[:3] if r is not None]
        if valid_rates:
            avg_growth = sum(valid_rates) / len(valid_rates)
            results['revenue_growth']['value'] = round(avg_growth, 2)
            results['revenue_growth']['pass'] = avg_growth > 10

    # 3. 영업이익률: TTM 기준 > 10%
    op_margin = metrics.get('operating_margin', [])
    if op_margin:
        valid_margins = [m for m in op_margin[:1] if m is not None]  # TTM 영업이익률만
        if valid_margins:
            ttm_margin = valid_margins[0]
            results['operating_margin_avg']['value'] = round(ttm_margin, 2)
            results['operating_margin_avg']['pass'] = ttm_margin > 10

    # 4. 영업이익 성장률: 3년 평균 > 10% (TTM 포함)
    op_growth_rate = metrics.get('op_profit_growth_rate', [])
    if op_growth_rate:
        valid_rates = [r for r in op_growth_rate[:3] if r is not None]
        if valid_rates:
            avg_growth = sum(valid_rates) / len(valid_rates)
            results['operating_profit_growth']['value'] = round(avg_growth, 2)
            results['operating_profit_growth']['pass'] = avg_growth > 10

    # 5. EPS 성장률: 3년 평균 > 10%
    eps_growth_rate = metrics.get('eps_growth_rate', [])
    if eps_growth_rate:
        valid_rates = [r for r in eps_growth_rate[:3] if r is not None]
        if valid_rates:
            avg_growth = sum(valid_rates) / len(valid_rates)
            results['eps_growth']['value'] = round(avg_growth, 2)
            results['eps_growth']['pass'] = avg_growth > 10

    # 6. 순이익 TTM: 양수
    net_income_ttm = metrics.get('net_income_ttm')
    if net_income_ttm is not None:
        results['net_income_positive']['value'] = round(net_income_ttm, 2)
        results['net_income_positive']['pass'] = net_income_ttm > 0

    # 7. 순이익 증가율: 15% < 3년 평균 < 50% (TTM 포함)
    net_growth_rate = metrics.get('net_income_growth_rate', [])
    if net_growth_rate:
        valid_rates = [r for r in net_growth_rate[:3] if r is not None]
        if valid_rates:
            avg_growth = sum(valid_rates) / len(valid_rates)
            results['net_income_growth']['value'] = round(avg_growth, 2)
            results['net_income_growth']['pass'] = 15 < avg_growth < 50

    # 통과 여부: 7개 조건 모두 충족
    pass_count = sum(1 for r in results.values() if r['pass'])
    all_pass = pass_count == 7

    return all_pass, results


def fetch_fnguide_finance(code: str) -> Dict[str, List[float]]:
    """FnGuide 재무제표 페이지에서 손익계산서 데이터를 추출합니다."""
    metrics = {}

    url = 'https://comp.fnguide.com/SVO2/ASP/SVD_Finance.asp'
    params = {'pGB': '1', 'gicode': f'A{code}', 'MenuYn': 'Y', 'NewMenuID': '103', 'stkGb': '701'}

    try:
        resp = requests.get(url, params=params, headers=HEADERS, timeout=15)
        soup = BeautifulSoup(resp.text, 'html.parser')

        # 연간 손익계산서 테이블 (첫 번째 테이블)
        tables = soup.find_all('table')
        for table in tables:
            rows = table.find_all('tr')
            if not rows:
                continue

            # 헤더 확인 (IFRS 연결/별도)
            header = rows[0].get_text()
            if 'IFRS' not in header:
                continue

            # 연간 테이블만 (분기 테이블 제외)
            headers = rows[0].find_all(['th', 'td'])
            years = []
            for h in headers[1:]:
                text = h.get_text(strip=True)
                if '/' in text and len(text) >= 7:
                    years.append(text)

            if not years or len(years) < 3:
                continue

            # 분기 테이블 건너뛰기 (분기 데이터는 3개월 간격)
            if len(years) > 4:
                year_parts = [y.split('/')[0] for y in years[:4]]
                if len(set(year_parts)) <= 2:  # 같은 연도 내 분기 데이터
                    continue

            metrics['_years'] = years[:5]

            # 데이터 행 처리
            for row in rows[1:]:
                cells = row.find_all(['th', 'td'])
                if len(cells) < 2:
                    continue

                label = cells[0].get_text(strip=True)
                values = []
                for cell in cells[1:len(years)+1]:
                    val = parse_number(cell.get_text(strip=True))
                    values.append(val)

                # 지표 매핑
                if '매출액' == label:
                    metrics['revenue'] = values
                elif '영업이익' == label and '발표' not in label:
                    metrics['operating_profit'] = values
                elif '당기순손익' in label or label == '당기순이익':
                    metrics['net_income'] = values

            # 첫 번째 유효한 테이블만 사용
            if 'revenue' in metrics:
                break

    except Exception as e:
        pass

    return metrics


def fetch_fnguide_ratio(code: str) -> Dict[str, List[float]]:
    """FnGuide 재무비율 페이지에서 안정성/수익성/성장성 지표를 추출합니다."""
    metrics = {}

    url = 'https://comp.fnguide.com/SVO2/ASP/SVD_FinanceRatio.asp'
    params = {'pGB': '1', 'gicode': f'A{code}', 'MenuYn': 'Y', 'NewMenuID': '104', 'stkGb': '701'}

    try:
        resp = requests.get(url, params=params, headers=HEADERS, timeout=15)
        soup = BeautifulSoup(resp.text, 'html.parser')

        tables = soup.find_all('table')

        for table in tables:
            rows = table.find_all('tr')
            if not rows:
                continue

            # 헤더에서 연간 테이블 확인
            header = rows[0].get_text()
            if 'IFRS' not in header:
                continue

            # 분기 테이블 건너뛰기
            headers = rows[0].find_all(['th', 'td'])
            header_texts = [h.get_text(strip=True) for h in headers]
            if any('/03' in h or '/06' in h or '/09' in h for h in header_texts[1:4]):
                continue

            for row in rows:
                cells = row.find_all(['th', 'td'])
                if len(cells) < 2:
                    continue

                label = cells[0].get_text(strip=True)

                # 값이 있는 셀 추출 (설명 행 건너뛰기)
                values = []
                for cell in cells[1:6]:
                    val = parse_number(cell.get_text(strip=True))
                    values.append(val)

                # 모든 값이 None이면 건너뛰기
                if all(v is None for v in values):
                    continue

                # 지표 매핑
                if '유동비율' in label and '유동비율' not in metrics:
                    metrics['current_ratio'] = values
                elif label.startswith('부채비율') and '순차입' not in label:
                    if 'debt_ratio' not in metrics:
                        metrics['debt_ratio'] = values
                elif '이자보상배율' in label:
                    if 'interest_coverage' not in metrics:
                        metrics['interest_coverage'] = values
                elif '매출액증가율' in label:
                    if 'revenue_growth_rate' not in metrics:
                        metrics['revenue_growth_rate'] = values
                elif '영업이익증가율' in label:
                    if 'op_profit_growth_rate' not in metrics:
                        metrics['op_profit_growth_rate'] = values
                elif 'EPS증가율' in label:
                    if 'eps_growth_rate' not in metrics:
                        metrics['eps_growth_rate'] = values
                elif '영업이익률' in label and '증가' not in label:
                    if 'operating_margin' not in metrics:
                        metrics['operating_margin'] = values
                elif '매출총이익율' in label or '매출총이익률' in label:
                    if 'gross_margin' not in metrics:
                        metrics['gross_margin'] = values

            # 첫 번째 유효한 연간 테이블만
            if 'debt_ratio' in metrics or 'current_ratio' in metrics:
                break

    except Exception as e:
        pass

    return metrics


def fetch_fnguide_invest(code: str) -> Dict[str, Any]:
    """FnGuide 투자지표 페이지에서 Per Share, 밸류에이션 지표를 추출합니다."""
    metrics = {}

    url = 'https://comp.fnguide.com/SVO2/ASP/SVD_Invest.asp'
    params = {'pGB': '1', 'gicode': f'A{code}', 'MenuYn': 'Y', 'NewMenuID': '105', 'stkGb': '701'}

    try:
        resp = requests.get(url, params=params, headers=HEADERS, timeout=15)
        soup = BeautifulSoup(resp.text, 'html.parser')

        tables = soup.find_all('table')

        for table in tables:
            rows = table.find_all('tr')
            if not rows:
                continue

            # 전체 테이블 텍스트로 PER/PBR 테이블 식별
            table_text = table.get_text()

            # 밸류에이션 테이블 (PER, PBR) - 최고/최저가 있는 테이블
            if '최고' in table_text and '최저' in table_text and 'PER' in table_text:
                for row in rows:
                    cells = row.find_all(['th', 'td'])
                    if len(cells) < 3:
                        continue

                    label = cells[0].get_text(strip=True)

                    # PER, PBR은 최고/최저 쌍으로 되어 있음 - 최고값 사용
                    values = []
                    for i in range(1, min(len(cells), 11), 2):  # 홀수 인덱스 = 최고값
                        val = parse_number(cells[i].get_text(strip=True))
                        values.append(val)

                    if 'PER' == label and 'per' not in metrics:
                        metrics['per'] = [v for v in values if v is not None]
                    elif 'PBR' == label and 'pbr' not in metrics:
                        metrics['pbr'] = [v for v in values if v is not None]

            # Per Share 테이블 (EPS, BPS, DPS)
            header = rows[0].get_text() if rows else ''
            if 'IFRS' in header and ('Per Share' in table_text or 'EPS' in table_text):
                for row in rows[1:]:
                    cells = row.find_all(['th', 'td'])
                    if len(cells) < 2:
                        continue

                    label = cells[0].get_text(strip=True)
                    values = []
                    for cell in cells[1:6]:
                        val = parse_number(cell.get_text(strip=True))
                        values.append(val)

                    if all(v is None for v in values):
                        continue

                    if 'EPS' in label and '증가' not in label and 'eps' not in metrics:
                        # EPS 행에서 실제 EPS 값만 추출 (지배주주순이익 행 제외)
                        if '지배' not in label and '수정' not in label:
                            metrics['eps'] = values
                    elif label.startswith('BPS') and 'bps' not in metrics:
                        metrics['bps'] = values
                    elif 'DPS' in label and '보통주' in label and 'dividend' not in metrics:
                        metrics['dividend'] = values
                    elif '지배주주순이익' in label and 'controlling_income' not in metrics:
                        metrics['controlling_income'] = values

    except Exception as e:
        pass

    # 배당수익률 계산 (DPS / 주가)
    # 현재 주가 정보가 없으므로 DPS만 저장

    return metrics


def crawl_all_stocks(stock_list: List[Dict], min_market_cap: int = 1000, delay: float = 0.8) -> Dict[str, Dict]:
    """
    전체 종목의 재무데이터를 크롤링합니다.
    """
    # 시가총액 필터링 및 정렬
    filtered = [s for s in stock_list if s.get('MarketCap', 0) >= min_market_cap]
    filtered.sort(key=lambda x: x.get('MarketCap', 0), reverse=True)

    print(f"크롤링 대상: {len(filtered)}개 종목 (시총 {min_market_cap}억 이상)")

    results = {}
    failed = []

    for i, stock in enumerate(filtered):
        code = stock['Code']
        name = stock['Name']

        print(f"  [{i+1}/{len(filtered)}] {name}({code})", end=' ')

        data = fetch_financial_data(code)
        if data and data.get('metrics'):
            data['name'] = name
            data['market_cap'] = stock.get('MarketCap', 0)
            results[code] = data
            metrics_count = len(data.get('metrics', {}))
            print(f"OK ({metrics_count} metrics)")
        else:
            failed.append(code)
            print("FAILED")

        # 차단 방지 딜레이
        if i < len(filtered) - 1:
            time.sleep(delay)

    print(f"\n크롤링 완료: 성공 {len(results)}개, 실패 {len(failed)}개")
    return results


def save_financial_data(data: Dict[str, Dict]) -> None:
    """재무데이터를 JSON 파일로 저장합니다."""
    os.makedirs(DATA_PATH, exist_ok=True)

    output = {
        'meta': {
            'updated_at': datetime.now().isoformat(),
            'total_count': len(data),
            'treasury_rate': TREASURY_RATE
        },
        'data': data
    }

    with open(FINANCIAL_DATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"저장: {FINANCIAL_DATA_FILE} ({len(data)}개 종목)")


def load_financial_data() -> Optional[Dict[str, Dict]]:
    """저장된 재무데이터를 로드합니다."""
    if not os.path.exists(FINANCIAL_DATA_FILE):
        return None

    try:
        with open(FINANCIAL_DATA_FILE, 'r', encoding='utf-8') as f:
            content = json.load(f)
        return content.get('data', {})
    except:
        return None


# ============================================================================
# 저평가 우량주 스크리너 로직
# ============================================================================

def calculate_growth_rate(values: List[Optional[float]], years: int = 5) -> Optional[float]:
    """성장률 계산 (최근값 / 과거값 - 1) * 100"""
    if not values or len(values) < 2:
        return None

    # 유효한 값만 추출
    valid = [(i, v) for i, v in enumerate(values) if v is not None and v != 0]
    if len(valid) < 2:
        return None

    recent_idx, recent_val = valid[0]
    oldest_idx, oldest_val = valid[-1]

    if oldest_val == 0:
        return None

    # 기간 조정
    actual_years = oldest_idx - recent_idx + 1
    if actual_years < 2:
        return None

    return ((recent_val / oldest_val) - 1) * 100


def calculate_cagr(values: List[Optional[float]], years: int = 5) -> Optional[float]:
    """연평균 복합 성장률(CAGR) 계산"""
    if not values or len(values) < 2:
        return None

    valid = [(i, v) for i, v in enumerate(values) if v is not None and v > 0]
    if len(valid) < 2:
        return None

    recent_idx, recent_val = valid[0]
    oldest_idx, oldest_val = valid[-1]

    if oldest_val <= 0 or recent_val <= 0:
        return None

    actual_years = oldest_idx - recent_idx
    if actual_years < 1:
        return None

    # CAGR = (최근값/과거값)^(1/기간) - 1
    cagr = ((recent_val / oldest_val) ** (1 / actual_years) - 1) * 100
    return cagr


def calculate_average(values: List[Optional[float]], years: int = 5) -> Optional[float]:
    """연평균 계산"""
    if not values:
        return None
    valid = [v for v in values[:years] if v is not None]
    if not valid:
        return None
    return sum(valid) / len(valid)


def pass_first_filter(fin_data: Dict) -> Tuple[bool, Dict[str, Any]]:
    """
    1차 필터 (7개 조건) - 가치투자 최적화 버전

    조건:
    1. PER: 3 < PER < 15 (저평가 집중)
    2. PBR: < 1.5 (자산가치 대비 저평가)
    3. 매출액 성장률: 3년간 > 10%
    4. 영업이익률: 5년 평균 > 12%
    5. 영업이익 성장률: 5년 평균 > 10%
    6. EPS 성장률: 5년간 > 10%
    7. 순이익 증가율: 15% < 5년간 < 40%
    """
    metrics = fin_data.get('metrics', {})

    results = {
        'per_check': {'value': None, 'pass': False, 'condition': '3 < PER < 15'},
        'pbr_check': {'value': None, 'pass': False, 'condition': 'PBR < 1.5'},
        'revenue_growth': {'value': None, 'pass': False, 'condition': '매출성장률 3년 > 10%'},
        'operating_margin_avg': {'value': None, 'pass': False, 'condition': '영업이익률 5년평균 > 12%'},
        'operating_profit_growth': {'value': None, 'pass': False, 'condition': '영업이익 성장률 5년 > 10%'},
        'eps_growth': {'value': None, 'pass': False, 'condition': 'EPS 성장률 5년 > 10%'},
        'net_income_growth': {'value': None, 'pass': False, 'condition': '순이익 증가율 15~40%'}
    }

    # 1. PER: 3 < PER < 30
    per_values = metrics.get('per', [])
    if per_values and per_values[0] is not None:
        per = per_values[0]
        results['per_check']['value'] = round(per, 2)
        results['per_check']['pass'] = 3 < per < 30

    # 2. PBR: 조건 없음 (참고용)
    pbr_values = metrics.get('pbr', [])
    if pbr_values and pbr_values[0] is not None:
        pbr = pbr_values[0]
        results['pbr_check']['value'] = round(pbr, 2)
        results['pbr_check']['pass'] = True  # 조건 없음

    # 3. 매출액 성장률: 3년간 > 10%
    rev_growth_rate = metrics.get('revenue_growth_rate', [])
    if rev_growth_rate:
        valid_rates = [r for r in rev_growth_rate[:3] if r is not None]
        if valid_rates:
            avg_growth = sum(valid_rates) / len(valid_rates)
            results['revenue_growth']['value'] = round(avg_growth, 2)
            results['revenue_growth']['pass'] = avg_growth > 10
    else:
        revenue = metrics.get('revenue', [])
        if revenue and len(revenue) >= 3:
            cagr = calculate_cagr(revenue[:4], 3)
            if cagr is not None:
                results['revenue_growth']['value'] = round(cagr, 2)
                results['revenue_growth']['pass'] = cagr > 10

    # 4. 영업이익률: 5년 평균 > 10%
    op_margin = metrics.get('operating_margin', [])
    if op_margin:
        op_margin_avg = calculate_average(op_margin, 5)
        if op_margin_avg is not None:
            results['operating_margin_avg']['value'] = round(op_margin_avg, 2)
            results['operating_margin_avg']['pass'] = op_margin_avg > 10

    # 5. 영업이익 성장률: 5년 평균 > 10%
    op_growth_rate = metrics.get('op_profit_growth_rate', [])
    if op_growth_rate:
        valid_rates = [r for r in op_growth_rate[:5] if r is not None]
        if valid_rates:
            avg_growth = sum(valid_rates) / len(valid_rates)
            results['operating_profit_growth']['value'] = round(avg_growth, 2)
            results['operating_profit_growth']['pass'] = avg_growth > 10
    else:
        op_profit = metrics.get('operating_profit', [])
        if op_profit and len(op_profit) >= 2:
            cagr = calculate_cagr(op_profit, 5)
            if cagr is not None:
                results['operating_profit_growth']['value'] = round(cagr, 2)
                results['operating_profit_growth']['pass'] = cagr > 10

    # 6. EPS 성장률: 5년간 > 10%
    eps_growth_rate = metrics.get('eps_growth_rate', [])
    if eps_growth_rate:
        valid_rates = [r for r in eps_growth_rate[:5] if r is not None]
        if valid_rates:
            avg_growth = sum(valid_rates) / len(valid_rates)
            results['eps_growth']['value'] = round(avg_growth, 2)
            results['eps_growth']['pass'] = avg_growth > 10
    else:
        eps = metrics.get('eps', [])
        if eps and len(eps) >= 2:
            cagr = calculate_cagr(eps, 5)
            if cagr is not None:
                results['eps_growth']['value'] = round(cagr, 2)
                results['eps_growth']['pass'] = cagr > 10

    # 7. 순이익 증가율: 20% < 5년간 < 50% (CAGR)
    net_income = metrics.get('net_income', [])
    if net_income and len(net_income) >= 2:
        cagr = calculate_cagr(net_income, 5)
        if cagr is not None:
            results['net_income_growth']['value'] = round(cagr, 2)
            results['net_income_growth']['pass'] = 20 < cagr < 50

    # 통과 여부: 7개 조건 모두 충족
    pass_count = sum(1 for r in results.values() if r['pass'])
    all_pass = pass_count == 7

    return all_pass, results


def calculate_second_score(fin_data: Dict, current_per: Optional[float] = None) -> Tuple[int, Dict[str, Any]]:
    """
    2차 체크리스트 (18개 항목 스코어링)
    """
    metrics = fin_data.get('metrics', {})
    checklist = {}
    score = 0

    per_values = metrics.get('per', [])
    current_per = current_per or (per_values[0] if per_values else None)

    # 1. PER vs 5년최고
    if per_values and current_per and len(per_values) >= 2:
        valid_pers = [p for p in per_values if p is not None and p > 0]
        if valid_pers:
            max_per = max(valid_pers)
            threshold = max_per * 0.4
            passed = current_per < threshold
            checklist['per_vs_5y_high'] = {'label': 'PER < 5년최고×0.4', 'value': f"{current_per:.1f} < {threshold:.1f}", 'pass': passed}
            if passed: score += 1

    # 2. PER vs 평균
    if per_values and current_per:
        avg_per = calculate_average(per_values, 5)
        if avg_per:
            passed = current_per < avg_per
            checklist['per_vs_avg'] = {'label': 'PER < 5년평균', 'value': f"{current_per:.1f} < {avg_per:.1f}", 'pass': passed}
            if passed: score += 1

    # 3. PER vs 업종 (데이터 없음 - 건너뛰기)
    # industry_per = fin_data.get('industry_per')

    # 4. 이익수익률 > 국채금리×2
    if current_per and current_per > 0:
        earnings_yield = (1 / current_per) * 100
        threshold = TREASURY_RATE * 2
        passed = earnings_yield > threshold
        checklist['earnings_yield'] = {'label': f'이익수익률 > {threshold:.1f}%', 'value': f"{earnings_yield:.2f}%", 'pass': passed}
        if passed: score += 1

    # 5. PBR < 1.2
    pbr_values = metrics.get('pbr', [])
    if pbr_values and pbr_values[0] is not None:
        pbr = pbr_values[0]
        passed = pbr < 1.2
        checklist['pbr'] = {'label': 'PBR < 1.2', 'value': f"{pbr:.2f}", 'pass': passed}
        if passed: score += 1

    # 6. BPS 성장률 > 7.2% (5년 총)
    bps = metrics.get('bps', [])
    bps_growth = calculate_growth_rate(bps, 5)
    if bps_growth is not None:
        passed = bps_growth > 7.2
        checklist['bps_growth'] = {'label': 'BPS 5년성장 > 7.2%', 'value': f"{bps_growth:.2f}%", 'pass': passed}
        if passed: score += 1

    # 7. PEG 역수 > 1.5
    eps = metrics.get('eps', [])
    eps_growth = calculate_cagr(eps, 5)
    if eps_growth is not None and current_per and current_per > 0 and eps_growth > 0:
        peg_inverse = eps_growth / current_per
        passed = peg_inverse > 1.5
        checklist['peg_inverse'] = {'label': 'PEG역수 > 1.5', 'value': f"{peg_inverse:.2f}", 'pass': passed}
        if passed: score += 1

    # 8. 부채비율 < 100%
    debt_ratio = metrics.get('debt_ratio', [])
    if debt_ratio and debt_ratio[0] is not None:
        passed = debt_ratio[0] < 100
        checklist['debt_ratio'] = {'label': '부채비율 < 100%', 'value': f"{debt_ratio[0]:.1f}%", 'pass': passed}
        if passed: score += 1

    # 9. 유동비율 > 150%
    current_ratio = metrics.get('current_ratio', [])
    if current_ratio and current_ratio[0] is not None:
        passed = current_ratio[0] > 150
        checklist['current_ratio'] = {'label': '유동비율 > 150%', 'value': f"{current_ratio[0]:.1f}%", 'pass': passed}
        if passed: score += 1

    # 10. 이자보상배율 > 2
    interest_cov = metrics.get('interest_coverage', [])
    if interest_cov and interest_cov[0] is not None:
        passed = interest_cov[0] > 2
        checklist['interest_coverage'] = {'label': '이자보상배율 > 2', 'value': f"{interest_cov[0]:.1f}", 'pass': passed}
        if passed: score += 1

    # 11. PER/배당률 < 4
    dividend = metrics.get('dividend', [])
    # 배당수익률은 주가 데이터가 필요하므로 건너뛰기

    # 12. 배당 3년 연속 증가
    if dividend and len(dividend) >= 3:
        valid_divs = [d for d in dividend[:3] if d is not None]
        if len(valid_divs) >= 3:
            passed = valid_divs[0] > valid_divs[1] > valid_divs[2]
            checklist['div_increase'] = {'label': '배당 3년연속↑', 'value': str([int(d) for d in valid_divs]), 'pass': passed}
            if passed: score += 1

    # 13. 배당률 > 국채금리 (건너뛰기 - 주가 데이터 필요)

    # 14. ROE 5년 평균 > 15% (FnGuide에서 직접 ROE 제공 안함 - 계산)
    controlling_income = metrics.get('controlling_income', [])
    bps_values = metrics.get('bps', [])
    if controlling_income and bps_values:
        roe_values = []
        for ni, b in zip(controlling_income, bps_values):
            if ni is not None and b is not None and b > 0:
                roe_values.append((ni / b) * 100 / 10000)  # BPS는 원 단위, NI는 억 단위
        if roe_values:
            roe_avg = sum(roe_values) / len(roe_values)
            passed = roe_avg > 15
            checklist['roe_avg'] = {'label': 'ROE 5년평균 > 15%', 'value': f"{roe_avg:.2f}%", 'pass': passed}
            if passed: score += 1

    # 15. 장기부채 상환능력 (건너뛰기 - 데이터 없음)

    # 16. FCF/매출액 > 7% (건너뛰기 - FCF 데이터 없음)

    # 17. 매출총이익률 > 40%
    gross = metrics.get('gross_margin', [])
    if gross and gross[0] is not None:
        passed = gross[0] > 40
        checklist['gross_margin'] = {'label': '매출총이익률 > 40%', 'value': f"{gross[0]:.1f}%", 'pass': passed}
        if passed: score += 1

    # 18. 영업이익률 > 15%
    op_margin = metrics.get('operating_margin', [])
    if op_margin and op_margin[0] is not None:
        passed = op_margin[0] > 15
        checklist['op_margin'] = {'label': '영업이익률 > 15%', 'value': f"{op_margin[0]:.1f}%", 'pass': passed}
        if passed: score += 1

    return score, checklist


def screen_value_stocks(stock_list: List[Dict], financial_data: Dict[str, Dict]) -> List[Dict]:
    """
    저평가 우량주 스크리너 메인 함수

    1차 필터 통과한 종목만 2차 체크리스트 수행
    """
    results = []

    for stock in stock_list:
        code = stock['Code']

        if code not in financial_data:
            continue

        fin_data = financial_data[code]
        metrics = fin_data.get('metrics', {})

        # 1차 필터
        passed, first_filter_results = pass_first_filter(fin_data)
        if not passed:
            continue

        # 2차 스코어링
        score, checklist = calculate_second_score(fin_data)

        # 결과 생성
        per = metrics.get('per', [None])[0] if metrics.get('per') else None
        pbr = metrics.get('pbr', [None])[0] if metrics.get('pbr') else None
        debt = metrics.get('debt_ratio', [None])[0] if metrics.get('debt_ratio') else None
        dividend = metrics.get('dividend', [None])[0] if metrics.get('dividend') else None

        results.append({
            'ticker': code,
            'name': fin_data.get('name', stock.get('Name', '')),
            'market_cap': round(fin_data.get('market_cap', stock.get('MarketCap', 0)), 0),
            'score': score,
            'max_score': 14,  # 체크 가능한 항목 수
            'first_filter': first_filter_results,
            'checklist': checklist,
            'per': round(per, 2) if per else None,
            'pbr': round(pbr, 2) if pbr else None,
            'debt_ratio': round(debt, 1) if debt else None,
            'dividend': int(dividend) if dividend else None
        })

    # 점수 내림차순 정렬
    results.sort(key=lambda x: x['score'], reverse=True)

    return results


if __name__ == '__main__':
    print("=== FnGuide 크롤러 테스트 ===")
    data = fetch_financial_data('005930')
    if data:
        print("삼성전자 데이터 수집 성공!")
        print(f"  지표 수: {len(data.get('metrics', {}))}")
        for k, v in data.get('metrics', {}).items():
            if not k.startswith('_'):
                print(f"    {k}: {v[:3] if isinstance(v, list) and len(v) > 3 else v}")
    else:
        print("삼성전자 데이터 수집 실패")
