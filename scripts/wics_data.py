"""
WICS 섹터 데이터 수집 (wiseindex.com API)

와이즈인덱스 WICS 중분류 섹터 매핑 → CSV 캐시.
새 프로젝트(주식 관련 프로젝트)에서 이식.
"""

import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

CACHE_DIR = Path(__file__).parent.parent / '.cache'
WICS_CACHE_FILE = CACHE_DIR / 'wics_sectors.csv'

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
}

# WICS 중분류 섹터 코드 (25개)
WICS_SUB_CODES = [
    'G1010', 'G1510',
    'G2010', 'G2020', 'G2030',
    'G2510', 'G2520', 'G2530', 'G2550', 'G2560',
    'G3010', 'G3020', 'G3030',
    'G3510', 'G3520',
    'G4010', 'G4020', 'G4030', 'G4040', 'G4050',
    'G4510', 'G4520', 'G4530', 'G4540',
    'G5010', 'G5020',
    'G5510',
]


def fetch_wics_sectors() -> pd.DataFrame:
    """
    와이즈인덱스 API에서 WICS 중분류 섹터 매핑 수집 → CSV 캐시 저장.
    """
    print("  WICS 섹터 수집 중 (wiseindex.com)...")

    # 가용 거래일 탐색 (최대 30일 뒤로)
    dt = datetime.now()
    target_dt = None
    for _ in range(30):
        dt_str = dt.strftime('%Y%m%d')
        url = 'https://www.wiseindex.com/Index/GetIndexComponets'
        params = {'ceil_yn': '0', 'dt': dt_str, 'sec_cd': 'G2510'}
        try:
            resp = requests.get(url, params=params, headers=HEADERS, timeout=10)
            items = resp.json().get('list', [])
            if items:
                target_dt = dt_str
                break
        except Exception:
            pass
        dt -= timedelta(days=1)

    if not target_dt:
        raise RuntimeError("WICS 데이터 가용 날짜를 찾을 수 없음")

    print(f"  기준일: {target_dt}")

    rows = []
    for sec_cd in WICS_SUB_CODES:
        url = 'https://www.wiseindex.com/Index/GetIndexComponets'
        params = {'ceil_yn': '0', 'dt': target_dt, 'sec_cd': sec_cd}
        try:
            resp = requests.get(url, params=params, headers=HEADERS, timeout=10)
            data = resp.json()
            items = data.get('list', [])
            for item in items:
                sector_name = item.get('IDX_NM_KOR', '').replace('WICS ', '')
                rows.append({
                    'Code': item.get('CMP_CD', ''),
                    'Name': item.get('CMP_KOR', ''),
                    'WICS_Sector': sector_name,
                    'WICS_Code': sec_cd,
                })
        except Exception:
            pass
        time.sleep(0.05)

    df = pd.DataFrame(rows)
    CACHE_DIR.mkdir(exist_ok=True)
    df.to_csv(WICS_CACHE_FILE, index=False, encoding='utf-8-sig')
    print(f"  WICS 캐시 저장: {WICS_CACHE_FILE}")
    print(f"  {len(df)}개 종목, {df['WICS_Sector'].nunique()}개 중분류")
    return df


def get_sector_map(max_cache_days: int = 7) -> dict:
    """
    WICS 섹터 매핑 딕셔너리 반환. {종목코드: 섹터명}
    캐시가 max_cache_days 이내면 캐시 사용, 아니면 재수집.
    """
    if WICS_CACHE_FILE.exists():
        import os
        age_days = (time.time() - os.path.getmtime(WICS_CACHE_FILE)) / 86400
        if age_days <= max_cache_days:
            df = pd.read_csv(WICS_CACHE_FILE, dtype=str, encoding='utf-8-sig')
            sector_map = dict(zip(df['Code'], df['WICS_Sector']))
            print(f"  WICS 캐시 로드: {len(sector_map)}개 종목, "
                  f"{df['WICS_Sector'].nunique()}개 섹터 (캐시 {age_days:.1f}일)")
            return sector_map

    df = fetch_wics_sectors()
    return dict(zip(df['Code'], df['WICS_Sector']))


if __name__ == '__main__':
    sector_map = get_sector_map(max_cache_days=0)  # 강제 재수집
    print(f"\n총 {len(sector_map)}개 종목 섹터 매핑 완료")
