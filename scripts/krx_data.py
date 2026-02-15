"""
KRX 종목 마스터 + WiseIndex 섹터 데이터 수집 모듈

KRX OTP 2단계 통신으로 전 종목 데이터를 1회 수집합니다.
WiseIndex API에서 WICS 섹터별 구성 종목을 수집합니다.

사용법:
    from scripts.krx_data import get_stock_master, get_sector_data
    stocks = get_stock_master()
    sectors = get_sector_data()
"""

import io
import re
import time
from datetime import datetime, timedelta

import pandas as pd
import requests

# ============================================================================
# 상수
# ============================================================================

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
}

KRX_GENERATE_URL = 'http://data.krx.co.kr/comm/fileDn/GenerateOTP/generate.cmd'
KRX_DOWNLOAD_URL = 'http://data.krx.co.kr/comm/fileDn/download_csv/download.cmd'
KRX_REFERER = 'http://data.krx.co.kr/contents/MDC/MDI/mdiLoader/index.cmd'

# WICS 섹터 코드
WICS_SECTORS = {
    'G10': '에너지',
    'G15': '소재',
    'G20': '산업재',
    'G25': '경기소비재',
    'G30': '필수소비재',
    'G35': '건강관리',
    'G40': '금융',
    'G45': 'IT',
    'G50': '통신서비스',
    'G55': '유틸리티',
    'G60': '부동산',
}

# 금융 업종 키워드 (마법공식에서 제외)
FINANCIAL_SECTORS = {'금융'}
FINANCIAL_KEYWORDS = ['은행', '보험', '증권', '캐피탈', '저축은행', '카드']

# 스팩 패턴
SPAC_PATTERN = re.compile(r'스팩|SPAC|기업인수목적', re.IGNORECASE)
REIT_PATTERN = re.compile(r'리츠|REIT|부동산투자', re.IGNORECASE)


# ============================================================================
# KRX OTP 종목 마스터
# ============================================================================

def _get_otp(params: dict) -> str:
    """KRX OTP 발급"""
    resp = requests.get(KRX_GENERATE_URL, params=params, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    return resp.text


def _download_csv(otp: str) -> pd.DataFrame:
    """KRX CSV 다운로드"""
    resp = requests.post(
        KRX_DOWNLOAD_URL,
        data={'code': otp},
        headers={**HEADERS, 'Referer': KRX_REFERER},
        timeout=30
    )
    resp.raise_for_status()
    return pd.read_csv(io.BytesIO(resp.content), encoding='euc-kr')


def get_stock_master(market: str = 'ALL') -> pd.DataFrame:
    """
    전 종목 마스터 데이터를 수집합니다.
    FDR(FinanceDataReader) 기반으로 KOSPI+KOSDAQ 종목을 수집합니다.

    Returns:
        DataFrame with columns:
        - 종목코드, 종목명, 시장구분, 시가총액(억원)
        - 종가
        - is_common (보통주 여부)
        - is_spac, is_reit (스팩/리츠 여부)
    """
    import FinanceDataReader as fdr

    print("종목 마스터 수집 중... (FDR)")

    # KOSPI + KOSDAQ
    kospi = fdr.StockListing('KOSPI')
    kospi['시장구분'] = 'KOSPI'

    kosdaq = fdr.StockListing('KOSDAQ')
    kosdaq['시장구분'] = 'KOSDAQ'

    df = pd.concat([kospi, kosdaq], ignore_index=True)

    # 컬럼 매핑
    df = df.rename(columns={
        'Code': '종목코드',
        'Name': '종목명',
        'Close': '종가',
        'Market': '_market_orig',
    })

    # 시가총액 억원 단위 변환
    if 'Marcap' in df.columns:
        df['시가총액'] = df['Marcap'] / 100000000
    elif 'MarketCap' in df.columns:
        df['시가총액'] = df['MarketCap'] / 100000000
    else:
        df['시가총액'] = 0

    # 종목코드 문자열 처리
    df['종목코드'] = df['종목코드'].astype(str).str.zfill(6)

    # 종목 분류
    df['is_common'] = df['종목코드'].apply(_is_common_stock)
    df['is_spac'] = df['종목명'].apply(lambda x: bool(SPAC_PATTERN.search(str(x))))
    df['is_reit'] = df['종목명'].apply(lambda x: bool(REIT_PATTERN.search(str(x))))

    # 업종 정보는 FDR에 없으므로 빈값
    if 'Dept' in df.columns:
        df['업종'] = df['Dept']
    else:
        df['업종'] = ''

    print(f"  전체 종목: {len(df)}개")
    print(f"  보통주: {df['is_common'].sum()}개")

    return df


def _is_common_stock(code: str) -> bool:
    """보통주 여부 판별 (코드 끝자리 기준)"""
    if not code or len(code) < 6:
        return False
    # 우선주: 끝자리 5 또는 0이 아닌 경우 (일반적으로 끝자리 0이 보통주)
    # 한국 시장에서 보통주는 보통 끝자리가 0, 우선주는 5
    last_digit = code[-1]
    return last_digit == '0'


def get_filtered_stocks(min_market_cap: int = 1000) -> pd.DataFrame:
    """
    필터링된 종목 리스트를 반환합니다.
    run_screeners.py의 get_stock_list() 대체용입니다.

    Args:
        min_market_cap: 최소 시가총액 (억원)

    Returns:
        DataFrame with columns: Code, Name, MarketCap, Market, PER, PBR, 업종, 배당수익률
    """
    df = get_stock_master()

    # 시가총액 필터
    if '시가총액' in df.columns:
        df = df[df['시가총액'] >= min_market_cap].copy()

    # 컬럼 매핑 (기존 코드 호환)
    result = pd.DataFrame({
        'Code': df['종목코드'],
        'Name': df['종목명'],
        'MarketCap': df.get('시가총액', pd.Series(dtype=float)),
        'Market': df.get('시장구분', 'KOSPI'),
        'Close': df.get('종가', pd.Series(dtype=float)),
        'PER': df.get('PER', pd.Series(dtype=float)),
        'PBR': df.get('PBR', pd.Series(dtype=float)),
        'Sector': df.get('업종', ''),
        'DividendYield': df.get('배당수익률', pd.Series(dtype=float)),
        'is_common': df['is_common'],
        'is_spac': df['is_spac'],
        'is_reit': df['is_reit'],
    })

    print(f"  시가총액 {min_market_cap}억 이상: {len(result)}개 종목")

    return result.reset_index(drop=True)


def is_financial_stock(sector: str, name: str = '') -> bool:
    """금융주 여부 판별"""
    if not sector:
        return False
    if sector in FINANCIAL_SECTORS:
        return True
    for keyword in FINANCIAL_KEYWORDS:
        if keyword in str(sector) or keyword in str(name):
            return True
    return False


# ============================================================================
# WiseIndex 섹터 데이터
# ============================================================================

WISEINDEX_URL = 'https://www.wiseindex.com/Index/GetIndexComponets'


def get_sector_data(date: str = None) -> pd.DataFrame:
    """
    WiseIndex에서 WICS 섹터별 구성 종목을 수집합니다.

    Args:
        date: 조회 기준일 (YYYYMMDD). None이면 오늘.

    Returns:
        DataFrame with columns: 종목코드, 종목명, 섹터코드, 섹터명, 비중
    """
    if date is None:
        today = datetime.now()
        # 주말이면 금요일로 조정
        if today.weekday() == 5:
            today -= timedelta(days=1)
        elif today.weekday() == 6:
            today -= timedelta(days=2)
        date = today.strftime('%Y%m%d')

    print(f"WiseIndex 섹터 데이터 수집 중... (기준일: {date})")

    all_data = []
    date_adjusted = False

    for code, name in WICS_SECTORS.items():
        try:
            params = {
                'ceil_yn': '0',
                'dt': date,
                'sec_cd': code,
            }
            resp = requests.get(WISEINDEX_URL, params=params, headers=HEADERS, timeout=15)
            resp.raise_for_status()

            data = resp.json()
            if isinstance(data, list):
                items = data
            elif isinstance(data, dict):
                items = data.get('list', data.get('data', []))
            else:
                items = []

            # 첫 섹터에서 0개면 전 거래일로 재시도 (장 시작 전 등)
            if not items and not date_adjusted and not all_data:
                prev = datetime.strptime(date, '%Y%m%d') - timedelta(days=1)
                while prev.weekday() >= 5:
                    prev -= timedelta(days=1)
                date = prev.strftime('%Y%m%d')
                date_adjusted = True
                print(f"  데이터 없음, 전 거래일로 재시도: {date}")
                params['dt'] = date
                resp = requests.get(WISEINDEX_URL, params=params, headers=HEADERS, timeout=15)
                resp.raise_for_status()
                data = resp.json()
                items = data if isinstance(data, list) else (data.get('list', data.get('data', [])) if isinstance(data, dict) else [])

            for item in items:
                ticker = str(item.get('CMP_CD', item.get('IDX_CD', ''))).replace('A', '')
                all_data.append({
                    '종목코드': ticker.zfill(6),
                    '종목명': item.get('CMP_KOR', item.get('IDX_NM_KOR', '')),
                    '섹터코드': code,
                    '섹터명': name,
                    '비중': item.get('WT', 0),
                })

            time.sleep(2)  # 서버 부하 방지

        except Exception as e:
            print(f"  [경고] 섹터 {name}({code}) 수집 실패: {e}")
            continue

    df = pd.DataFrame(all_data)
    print(f"  섹터 데이터: {len(df)}개 종목, {len(WICS_SECTORS)}개 섹터")

    return df


# ============================================================================
# 유틸리티
# ============================================================================

def merge_sector_info(stocks_df: pd.DataFrame, sector_df: pd.DataFrame) -> pd.DataFrame:
    """
    종목 마스터에 섹터 정보를 병합합니다.

    Args:
        stocks_df: get_stock_master() 또는 get_filtered_stocks() 결과
        sector_df: get_sector_data() 결과

    Returns:
        섹터 정보가 추가된 DataFrame
    """
    code_col = 'Code' if 'Code' in stocks_df.columns else '종목코드'

    sector_map = sector_df.drop_duplicates('종목코드').set_index('종목코드')[['섹터코드', '섹터명']]
    merged = stocks_df.merge(
        sector_map,
        left_on=code_col,
        right_index=True,
        how='left'
    )
    # 섹터 없는 종목은 '기타'
    merged['섹터명'] = merged['섹터명'].fillna('기타')
    merged['섹터코드'] = merged['섹터코드'].fillna('G99')

    return merged


if __name__ == '__main__':
    # 테스트
    print("=" * 60)
    print("KRX 종목 마스터 테스트")
    print("=" * 60)

    stocks = get_stock_master()
    print(f"\n전체 종목 수: {len(stocks)}")
    print(stocks[['종목코드', '종목명', 'PER', 'PBR', '시가총액']].head(10))

    print("\n" + "=" * 60)
    print("WiseIndex 섹터 테스트")
    print("=" * 60)

    sectors = get_sector_data()
    print(f"\n섹터 데이터: {len(sectors)}개")
    for code, name in WICS_SECTORS.items():
        count = len(sectors[sectors['섹터코드'] == code])
        print(f"  {name} ({code}): {count}개")
