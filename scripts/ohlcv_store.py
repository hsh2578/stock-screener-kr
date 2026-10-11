"""
날짜별 일봉 저장소 — 종목별 FDR 다운로드 대체.

구조
    .cache/daily/YYYYMMDD.parquet       확정 일봉: 그날 KOSPI+KOSDAQ 전 종목 1파일 (비수정 시세)
        index=Ticker, 컬럼: Open High Low Close Volume Change(등락률 %) MarketCap(원) Name Market
    .cache/daily/YYYYMMDD_live.parquet  오늘 장중 스냅샷 (다음 날 확정 파일로 교체)
    .cache/daily/holidays.txt           (선택) 달력에 없는 임시 휴장일 수동 추가 — 한 줄에 YYYYMMDD

출처
    - 지난 거래일: KRX OpenAPI (KRX_API_KEY 필수, 로그인 불필요, 다음 영업일 아침 게시)
    - OpenAPI 미게시일·오늘(장중): pykrx get_market_ohlcv(date, market=KOSPI/KOSDAQ)
      → KRX 로그인(KRX_ID/KRX_PW). pykrx 는 import 시 1회 로그인 → 필요할 때만, 프로세스당 1회.
    - 거래일 달력: 평일 - holidays.KR(공휴일·대체공휴일·선거일) - 근로자의 날(5/1) - 연말 휴장(12월 마지막 평일)
      (2023-09~2026-10 실측 휴장일과 전부 일치 확인)

실패 정책 (원자적 교체 — 실패가 기존 파일을 깨지 않음)
    - 최신 1~MAX_LAG 거래일만 못 받음(OpenAPI 미게시 + 로그인 실패) → ::warning:: 후 가진 최신일로 진행
      (meta.marketDate 는 run_screeners.get_market_latest_date() 가 실제 마지막 일자로 기록)
    - MAX_LAG 넘게 뒤처지거나, 그 이전 거래일에 구멍 → RuntimeError (non-zero 종료)
    - 오늘 장중만 못 받음 → 경고 후 마지막 확정 거래일로 진행

보존 기간 KEEP_DAYS = 750 거래일 (~3년)
    chart_data.json 이 종목당 ~3년(이전: FDR 1100 캘린더일 ≈ 740행)을 쓴다 → 가장 긴 소비자.
    이평선 수렴(주봉 100봉 신고가 + 30봉 룩백 ≈ 650거래일), 52주(250)+MA150, ML 피처(≤260)는 그 안.

수정주가
    저장소는 비수정 시세다. 거래소 등락률은 '기준가' 대비이므로, 종가/전일종가-1 과 등락률이
    CHANGE_TOL(0.05%p) 넘게 다르면 그날 기준가가 바뀐 것(권리락·분할·감자·병합 등)이다.
    기준가 = 종가/(1+등락률), 조정계수 = 기준가/전일종가 를 그 이전 모든 가격에 곱한다(거래량은 나눈다).
    거래소가 수정주가를 만드는 방식과 같고, 종목별 API 호출이 필요 없다.

시세 기준
    KRX 정규시장 공식 OHLC. FDR(네이버)은 2026-09-14 부터 고가/저가/종가가 NXT 포함 통합 시세라
    (시가·거래량은 KRX 와 동일) 최근 구간 값이 최대 수 % 다르다 — 저장소 쪽이 등락률과 일치.
"""
from __future__ import annotations

import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
DAILY_DIR = ROOT / '.cache' / 'daily'
KEEP_DAYS = 750
CHANGE_TOL = 0.05  # %p — 등락률(소수 2자리) 반올림 오차 0.005 의 10배. 3년치 실측: 0.01~0.05 구간 0건
WORKERS = 8
OPENAPI = 'https://data-dbg.krx.co.kr/svc/apis/sto/{}'
OPENAPI_EP = {'KOSPI': 'stk_bydd_trd', 'KOSDAQ': 'ksq_bydd_trd'}
PRICE = ['Open', 'High', 'Low', 'Close']
MAX_LAG = 2  # 최신 확정 거래일이 이만큼까지 비어도 경고만 하고 진행

# 실행 중 API 호출 수 (로그/검증용)
CALLS = {'openapi': 0, 'pykrx_login': 0, 'pykrx_date': 0}


_LOGIN_ERR: Optional[Exception] = None


def _stock():
    """pykrx 는 import 시점에 KRX 로그인을 한다. 성공은 모듈 캐시, 실패도 기억 → 프로세스당 시도 1회."""
    global _LOGIN_ERR
    if _LOGIN_ERR is not None:
        raise _LOGIN_ERR
    if not (os.getenv('KRX_ID') and os.getenv('KRX_PW')):
        _LOGIN_ERR = RuntimeError('KRX_ID/KRX_PW 환경변수 없음 — 최신 시세(pykrx)는 KRX 로그인이 필요하다')
        raise _LOGIN_ERR
    if 'pykrx.stock' not in sys.modules:
        CALLS['pykrx_login'] += 1
    try:
        from pykrx import stock
    except Exception as e:  # 로그인 응답이 HTML 에러 페이지면 JSONDecodeError
        _LOGIN_ERR = RuntimeError(f'pykrx import(KRX 로그인) 실패: {type(e).__name__}')
        raise _LOGIN_ERR from e
    return stock


def _retry(fn, tries=3):
    for i in range(tries):
        try:
            return fn()
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(1 + i * 2)


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace(',', ''), errors='coerce')


def _fetch_openapi(d: str) -> Optional[pd.DataFrame]:
    """KRX OpenAPI 하루치. 미게시(또는 휴장)면 None, 통신 오류는 예외."""
    key = os.getenv('KRX_API_KEY')
    if not key:
        raise RuntimeError('KRX_API_KEY 환경변수 없음 — 과거 거래일 백필(KRX OpenAPI)에 필요')
    parts = []
    for mkt, ep in OPENAPI_EP.items():
        CALLS['openapi'] += 1
        r = _retry(lambda: requests.get(OPENAPI.format(ep), params={'basDd': d},
                                        headers={'AUTH_KEY': key}, timeout=30))
        r.raise_for_status()
        rows = r.json().get('OutBlock_1', [])
        if not rows:
            return None
        x = pd.DataFrame(rows)
        parts.append(pd.DataFrame({
            'Open': _num(x['TDD_OPNPRC']), 'High': _num(x['TDD_HGPRC']), 'Low': _num(x['TDD_LWPRC']),
            'Close': _num(x['TDD_CLSPRC']), 'Volume': _num(x['ACC_TRDVOL']), 'Change': _num(x['FLUC_RT']),
            'MarketCap': _num(x['MKTCAP']), 'Name': x['ISU_NM'].values, 'Market': mkt,
        }).set_index(x['ISU_CD'].rename('Ticker')))
    return pd.concat(parts)


def _fetch_pykrx(stock, d: str) -> Optional[pd.DataFrame]:
    """pykrx 하루치. 전부 0 = 휴장/장 시작 전 → None. 빈 응답(pykrx 가 오류를 삼킨 경우)은 예외."""
    parts = []
    for mkt in OPENAPI_EP:
        CALLS['pykrx_date'] += 1
        x = _retry(lambda: stock.get_market_ohlcv(d, market=mkt))
        if x is None or x.empty:
            raise RuntimeError(f'pykrx 빈 응답 {d} {mkt}')
        if not (x[['시가', '고가', '저가', '종가']] != 0).any(axis=None):
            return None
        x = x.rename(columns={'시가': 'Open', '고가': 'High', '저가': 'Low', '종가': 'Close',
                              '거래량': 'Volume', '등락률': 'Change', '시가총액': 'MarketCap'})
        x = x[['Open', 'High', 'Low', 'Close', 'Volume', 'Change', 'MarketCap']].assign(Name=None, Market=mkt)
        x.index.name = 'Ticker'
        parts.append(x)
    return pd.concat(parts)


def _write(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix('.tmp')
    df.to_parquet(tmp)
    os.replace(tmp, path)  # 원자적 교체: 중간 실패가 기존 파일을 깨지 않음


def _path(d: str) -> Path:
    p = DAILY_DIR / f'{d}.parquet'
    return p if p.exists() else DAILY_DIR / f'{d}_live.parquet'


def trading_days(start, end, extra_closed=()) -> List[str]:
    """[start, end] KRX 거래일 (YYYYMMDD)."""
    import holidays
    days = pd.bdate_range(start, end)
    years = range(days[0].year, days[-1].year + 1)
    kr = holidays.KR(years=years)
    year_end = set()  # 연말 휴장: 12월 마지막 평일(공휴일 제외)
    for y in years:
        d = pd.Timestamp(y, 12, 31)
        while d.weekday() >= 5 or d in kr:
            d -= pd.Timedelta(days=1)
        year_end.add(d)
    return [d.strftime('%Y%m%d') for d in days
            if d not in kr and (d.month, d.day) != (5, 1) and d not in year_end
            and d.strftime('%Y%m%d') not in extra_closed]


def staleness(expected: List[str], have: set) -> tuple:
    """expected(오늘 이전 거래일, 오름차순) 중 없는 날을 (최신 연속 누락 수, 그 이전 구멍 목록)으로."""
    lag = 0
    for d in reversed(expected):
        if d in have:
            break
        lag += 1
    gaps = [d for d in expected[:len(expected) - lag] if d not in have]
    return lag, gaps


def update_store(keep: int = KEEP_DAYS, now: Optional[datetime] = None) -> List[str]:
    """저장소를 최신화하고 사용할 거래일 목록(오래된→최신, 최대 keep개)을 반환."""
    now = now or datetime.now()
    today = now.strftime('%Y%m%d')
    DAILY_DIR.mkdir(parents=True, exist_ok=True)
    hol_path = DAILY_DIR / 'holidays.txt'
    extra = set(hol_path.read_text().split()) if hol_path.exists() else set()
    cal = trading_days(now - timedelta(days=int(keep * 1.5) + 30), now, extra)
    expected = [d for d in cal if d < today][-keep:]  # 확정됐어야 할 거래일
    final = {p.stem for p in DAILY_DIR.glob('????????.parquet')}
    past = [d for d in expected if d not in final]
    print(f'[저장소] 확정 {len(final & set(expected))}/{len(expected)}일 보유, {len(past)}일 OpenAPI 수집', flush=True)

    def job(d):
        try:
            return d, _fetch_openapi(d), None
        except Exception as e:
            return d, None, e

    with ThreadPoolExecutor(WORKERS) as ex:
        results = list(ex.map(job, past))
    errors, pending = {}, []
    for d, df, err in results:
        if err is not None:
            errors[d] = err
        elif df is None:
            pending.append(d)  # OpenAPI 미게시
        else:
            _write(df, DAILY_DIR / f'{d}.parquet')
            final.add(d)

    # OpenAPI 미게시일 + 오늘 장중 → pykrx (로그인 시도 1회)
    need = pending + ([today] if today in cal else [])
    live_ok = False
    for d in need:
        try:
            df = _fetch_pykrx(_stock(), d)
        except Exception as e:
            errors[d] = e
            continue
        if df is None:  # 전부 0: 달력엔 거래일인데 시세 없음 → 장 시작 전(오늘) 또는 임시 휴장
            print(f'  {d} 시세 없음 — 장 시작 전 또는 임시 휴장 (임시 휴장이면 holidays.txt 에 추가)')
        elif d < today:
            _write(df, DAILY_DIR / f'{d}.parquet')
            final.add(d)
        else:
            _write(df, DAILY_DIR / f'{d}_live.parquet')
            live_ok = True
    if today in errors:
        print(f'::warning::오늘({today}) 장중 시세 수집 실패 → 마지막 확정 거래일로 진행: {errors.pop(today)}', flush=True)

    lag, gaps = staleness(expected, final)
    for d in gaps:
        logging.warning(f'날짜 수집 실패 {d}: {errors.get(d, "OpenAPI 미게시")} (기존 파일 유지)')
    if gaps:
        raise RuntimeError(f'과거 거래일 {len(gaps)}개 누락 (예: {gaps[:5]}) — 다음 실행에서 재시도')
    if lag > MAX_LAG:
        raise RuntimeError(f'저장소가 {lag}거래일 뒤처짐 (누락 {expected[-lag:]}, 허용 {MAX_LAG}) — '
                           f'KRX OpenAPI/로그인 확인')
    if lag:
        print(f'::warning::최신 {lag}거래일({expected[-lag:]}) 시세 미수집 (OpenAPI 미게시 + pykrx 실패) '
              f'→ {expected[-lag - 1]} 기준으로 진행', flush=True)

    dates = (sorted(final & set(expected)) + ([today] if live_ok else []))[-keep:]
    for p in DAILY_DIR.glob('*.parquet'):  # 창 밖 확정 파일, 지난 장중 스냅샷 정리
        d = p.stem[:8]
        if d < dates[0] or (p.stem.endswith('_live') and not (live_ok and d == today)):
            p.unlink(missing_ok=True)
    print(f'[저장소] 사용 {len(dates)}일: {dates[0]} ~ {dates[-1]}{" (장중)" if live_ok else ""}', flush=True)
    return dates


def stock_list(dates: List[str], min_cap_eok: float) -> pd.DataFrame:
    """최신 거래일 시가총액 → 보통주(코드 끝 0) + 시총 필터.
    반환 컬럼: Code, Name, MarketCap(억원), Market (기존 get_stock_list 호환)"""
    snap = pd.read_parquet(_path(dates[-1]))
    names = snap['Name']
    for d in reversed(dates[-10:]):  # pykrx 로 받은 최신일엔 종목명이 없음 → 최근 OpenAPI 파일에서
        if names.notna().all():
            break
        names = names.fillna(pd.read_parquet(_path(d), columns=['Name'])['Name'])
    names = names.reindex(snap.index).fillna(snap.index.to_series())  # 그래도 없으면(당일 신규상장) 코드
    df = pd.DataFrame({'Code': snap.index, 'Name': names.values,
                       'MarketCap': snap['MarketCap'].values / 1e8, 'Market': snap['Market'].values})
    df = df[(df['Code'].str[-1] == '0') & (df['MarketCap'] >= min_cap_eok)]
    return df.sort_values('MarketCap', ascending=False).reset_index(drop=True)


def _load_long(dates: List[str], tickers: set) -> pd.DataFrame:
    parts = []
    for d in dates:
        x = pd.read_parquet(_path(d), columns=PRICE + ['Volume', 'Change'])
        parts.append(x[x.index.isin(tickers)].assign(Date=pd.Timestamp(d)))
    return pd.concat(parts).reset_index().sort_values(['Ticker', 'Date'], ignore_index=True)


def adjust(long: pd.DataFrame) -> pd.DataFrame:
    """long(Ticker, Date 정렬) 비수정 → 수정주가. 반환에 'factor'(이 행에 곱한 계수) 포함."""
    prev = long.groupby('Ticker')['Close'].shift()
    implied = (long['Close'] / prev - 1) * 100
    # 거래정지일(시가 0)과 상장 첫날(전일 없음)은 제외
    hit = (prev > 0) & (long['Open'] > 0) & ((implied - long['Change']).abs() > CHANGE_TOL)
    # ponytail: 등락률 불일치 휴리스틱 — 거래재개 평가가격처럼 권리 변동이 아닌 기준가 변경도
    #           조정하고, 0.05%p 미만 조정은 놓친다. 정확도가 문제되면 KRX 권리락/분할 공시로 확정.
    f = (long['Close'] / (1 + long['Change'] / 100) / prev).where(hit, 1.0)
    # 각 행의 계수 = 그 행 '이후' 이벤트 계수의 곱
    after = f[::-1].groupby(long['Ticker'][::-1]).cumprod()[::-1] / f
    out = long.copy()
    out[PRICE] = (long[PRICE].mul(after, axis=0)).round()
    out['Volume'] = (long['Volume'] / after).round()
    out['factor'] = after
    return out


def build_frames(dates: List[str], tickers: Iterable[str]) -> Dict[str, pd.DataFrame]:
    """저장소 → 종목별 수정 OHLCV (Open/High/Low/Close/Volume int64, DatetimeIndex 'Date').
    기존 FDR DataReader 형식 호환."""
    long = adjust(_load_long(dates, set(tickers)))
    n_ev = long.loc[long['factor'] != 1.0, 'Ticker'].nunique()
    print(f'[저장소] 수정주가 조정 종목 {n_ev}개 (기준가 변경 감지)', flush=True)
    halted = long['Open'] == 0  # 거래정지일: 종가만 존재 → 시/고/저를 종가로 (저가 0 왜곡 방지)
    for c in ('Open', 'High', 'Low'):
        long.loc[halted, c] = long.loc[halted, 'Close']
    long = long.dropna(subset=PRICE + ['Volume'])
    out = {}
    for t, g in long.groupby('Ticker', sort=False):
        g = g.set_index('Date')[PRICE + ['Volume']].astype('int64')
        out[t] = g
    return out


if __name__ == '__main__':
    # 자체 점검: 이벤트 감지 + 수정주가 (네트워크 불필요)
    d = pd.to_datetime(['2026-01-02', '2026-01-05', '2026-01-06', '2026-01-07'])
    long = pd.DataFrame({
        'Ticker': ['A'] * 4 + ['B'] * 4, 'Date': list(d) * 2,
        'Open': [100, 100, 50, 51] + [10, 0, 10, 11],
        'High': [100, 100, 50, 52] + [10, 0, 10, 11],
        'Low': [100, 100, 50, 51] + [10, 0, 10, 11],
        'Close': [100, 100, 50, 52] + [10, 10, 10, 11],
        'Volume': [10, 10, 20, 20] + [5, 0, 5, 5],
        # A: 01-06 2:1 분할 (기준가 50, 등락률 0) / B: 01-05 거래정지
        'Change': [0, 0, 0, 4] + [0, 0, 0, 10],
    })
    a = adjust(long)
    assert a.loc[a.Ticker == 'A', 'Close'].tolist() == [50, 50, 50, 52], a
    assert a.loc[a.Ticker == 'A', 'Volume'].tolist() == [20, 20, 20, 20], a
    assert a.loc[a.Ticker == 'B', 'Close'].tolist() == [10, 10, 10, 11], a

    # 자체 점검: 거래일 달력 (2026 한글날·추석·대체공휴일·근로자의 날, 2023 연말 휴장)
    cal = trading_days('2026-09-21', '2026-10-13')
    for closed in ['20260924', '20260925', '20261005', '20261009']:
        assert closed not in cal, closed
    assert '20261008' in cal and '20261012' in cal
    assert '20260501' not in trading_days('2026-04-27', '2026-05-08')
    assert '20231229' not in trading_days('2023-12-20', '2024-01-05')
    assert '20241231' not in trading_days('2024-12-20', '2025-01-05')

    # 자체 점검: 지연 판정
    exp = ['d1', 'd2', 'd3', 'd4', 'd5']
    assert staleness(exp, set(exp)) == (0, [])
    assert staleness(exp, {'d1', 'd2', 'd3'}) == (2, [])          # 최신 2일 미게시 → 경고 후 진행
    assert staleness(exp, {'d1', 'd2'}) == (3, [])                # 3일 → 실패 (> MAX_LAG)
    assert staleness(exp, {'d1', 'd3', 'd4', 'd5'}) == (0, ['d2'])  # 과거 구멍 → 실패
    assert staleness(exp, {'d1', 'd3'}) == (2, ['d2'])

    # 자체 점검: 갱신 정책 (가짜 OpenAPI/로그인, 임시 폴더)
    import tempfile
    day = pd.DataFrame({'Open': [1], 'High': [1], 'Low': [1], 'Close': [1], 'Volume': [1], 'Change': [0.0],
                        'MarketCap': [1], 'Name': ['x'], 'Market': ['KOSPI']}, index=pd.Index(['000010'], name='Ticker'))
    now = datetime(2026, 10, 13, 11)  # 화 장중. 10-09(금) 한글날, 10-08·10-12 OpenAPI 미게시, 로그인 차단
    published = set(trading_days('2026-09-01', '2026-10-07'))

    def _no_login():
        raise RuntimeError('login blocked')
    _stock = _no_login
    _fetch_openapi = lambda d: day if d in published else None
    with tempfile.TemporaryDirectory() as tmp:
        DAILY_DIR = Path(tmp)
        dates = update_store(keep=10, now=now)          # 2거래일 지연 → 경고 후 10-07 기준
        assert dates[-1] == '20261007', dates
        published -= {'20261007'}
        (DAILY_DIR / '20261007.parquet').unlink()
        try:                                            # 3거래일 지연 → 실패
            update_store(keep=10, now=now)
            raise AssertionError('지연 미감지')
        except RuntimeError as e:
            assert '3거래일' in str(e), e
    print('ok')
