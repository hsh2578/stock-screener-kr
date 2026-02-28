"""
KRX 종목 마스터 수집 모듈

KRX OTP 2단계 통신으로 전 종목 데이터를 1회 수집합니다.

사용법:
    from scripts.krx_data import get_stock_master
    stocks = get_stock_master()
"""

import io
import re
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


_KRX_JSON_URL = 'https://www.krx.co.kr/comm/bldAttendant/getJsonData.cmd'
_KRX_JSON_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Referer': 'https://www.krx.co.kr/',
    'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8',
}


def _fetch_krx_listing(mkt_id: str, trd_dd: str) -> list:
    """
    KRX JSON API(MDCSTAT01501)에서 종목 리스트를 수집합니다.
    pykrx가 내부적으로 사용하는 동일 엔드포인트입니다.
    """
    params = {
        'bld': 'dbms/MDC/STAT/standard/MDCSTAT01501',
        'mktId': mkt_id,
        'trdDd': trd_dd,
        'share': '1',
        'money': '1',
        'csvxls_isNo': 'false',
    }
    resp = requests.post(_KRX_JSON_URL, data=params, headers=_KRX_JSON_HEADERS, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return data.get('OutBlock_1', [])


def _parse_krx_number(val: str) -> float:
    """KRX 숫자 문자열(콤마 포함) → float"""
    try:
        return float(str(val).replace(',', ''))
    except (ValueError, TypeError):
        return 0.0


def get_stock_master(market: str = 'ALL') -> pd.DataFrame:
    """
    전 종목 마스터 데이터를 수집합니다.
    KRX JSON API(MDCSTAT01501) 직접 호출 방식 (pykrx 동일 엔드포인트).
    실패 시 FDR StockListing으로 폴백합니다.

    Returns:
        DataFrame with columns:
        - 종목코드, 종목명, 시장구분, 시가총액(억원)
        - 종가
        - is_common (보통주 여부)
        - is_spac, is_reit (스팩/리츠 여부)
    """
    import datetime
    import time

    today = datetime.datetime.now().strftime('%Y%m%d')
    df = None

    # ── 1차: KRX JSON API 직접 호출 ──────────────────────────────────
    print("종목 마스터 수집 중... (KRX JSON API)")
    try:
        rows = []
        for mkt_id, market_name in [('STK', 'KOSPI'), ('KSQ', 'KOSDAQ')]:
            items = _fetch_krx_listing(mkt_id, today)
            if not items:
                raise ValueError(f"{market_name} 빈 응답")
            for item in items:
                code = str(item.get('ISU_SRT_CD', '')).strip().zfill(6)
                if not code or len(code) != 6:
                    continue
                rows.append({
                    '종목코드': code,
                    '종목명': str(item.get('ISU_ABBRV', '')).strip(),
                    '시장구분': market_name,
                    '시가총액': _parse_krx_number(item.get('MKTCAP', '0')),
                    '종가': _parse_krx_number(item.get('TDD_CLSPRC', '0')),
                })
        if rows:
            df = pd.DataFrame(rows)
            print(f"  KRX API 성공: {len(df)}개 종목")
    except Exception as e:
        print(f"  KRX API 실패: {e}")

    # ── 2차 폴백: FDR StockListing ────────────────────────────────────
    if df is None or len(df) == 0:
        import FinanceDataReader as fdr
        print("  FDR 폴백으로 재시도...")

        def _fetch_fdr(mkt, retries=3, delay=10):
            for attempt in range(retries):
                try:
                    result = fdr.StockListing(mkt)
                    if result is not None and len(result) > 0:
                        return result
                    print(f"    [{mkt}] 빈 응답 ({attempt+1}/{retries})")
                except Exception as ex:
                    print(f"    [{mkt}] 오류: {ex} ({attempt+1}/{retries})")
                time.sleep(delay)
            raise RuntimeError(f"fdr.StockListing('{mkt}') 실패")

        kospi = _fetch_fdr('KOSPI')
        kospi['시장구분'] = 'KOSPI'
        kosdaq = _fetch_fdr('KOSDAQ')
        kosdaq['시장구분'] = 'KOSDAQ'
        fdr_df = pd.concat([kospi, kosdaq], ignore_index=True)
        fdr_df = fdr_df.rename(columns={'Code': '종목코드', 'Name': '종목명', 'Close': '종가'})
        if 'Marcap' in fdr_df.columns:
            fdr_df['시가총액'] = fdr_df['Marcap'] / 100000000
        elif 'MarketCap' in fdr_df.columns:
            fdr_df['시가총액'] = fdr_df['MarketCap'] / 100000000
        else:
            fdr_df['시가총액'] = 0
        df = fdr_df

    # ── 공통 후처리 ──────────────────────────────────────────────────
    df['종목코드'] = df['종목코드'].astype(str).str.zfill(6)
    df['is_common'] = df['종목코드'].apply(_is_common_stock)
    df['is_spac'] = df['종목명'].apply(lambda x: bool(SPAC_PATTERN.search(str(x))))
    df['is_reit'] = df['종목명'].apply(lambda x: bool(REIT_PATTERN.search(str(x))))
    df['업종'] = df.get('Dept', pd.Series(['기타'] * len(df))).replace('', '기타').fillna('기타') \
                 if 'Dept' in df.columns else '기타'

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


if __name__ == '__main__':
    # 테스트
    print("=" * 60)
    print("KRX 종목 마스터 테스트")
    print("=" * 60)

    stocks = get_stock_master()
    print(f"\n전체 종목 수: {len(stocks)}")
    print(stocks[['종목코드', '종목명', 'PER', 'PBR', '시가총액']].head(10))
