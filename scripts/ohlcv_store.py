"""
날짜별 일봉 저장소 — 종목별 FDR 다운로드 대체.

구조
    .cache/daily/YYYYMMDD.parquet       확정 일봉: 그날 KOSPI+KOSDAQ 전 종목 1파일 (비수정 시세)
        index=Ticker, 컬럼: Open High Low Close Volume Change(등락률 %) MarketCap(원) Name Market
    .cache/daily/YYYYMMDD_live.parquet  오늘 장중 스냅샷 (다음 날 확정 파일로 교체)
    .cache/daily/holidays.txt           평일 휴장일 (재조회 방지)

출처
    - 지난 거래일: KRX OpenAPI (KRX_API_KEY 필수, 로그인 불필요, 다음 영업일 아침 게시)
    - OpenAPI 미게시일·오늘(장중): pykrx get_market_ohlcv(date, market=KOSPI/KOSDAQ)
      → KRX 로그인(KRX_ID/KRX_PW). pykrx 는 import 시 1회 로그인 → 필요할 때만, 프로세스당 1회.
    - 거래일 달력은 따로 받지 않는다: 평일 중 데이터가 없는 날 = 휴장일.

실패 정책 (원자적 교체 — 실패가 기존 파일을 깨지 않음)
    - 과거일 수집 예외 → 이력에 구멍 → RuntimeError (non-zero 종료)
    - 마지막 확정 거래일을 못 받음(OpenAPI 미게시 + 로그인 실패) → RuntimeError
    - 오늘 장중만 못 받음(로그인 실패) → 경고 후 마지막 확정 거래일로 진행

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

# 실행 중 API 호출 수 (로그/검증용)
CALLS = {'openapi': 0, 'pykrx_login': 0, 'pykrx_date': 0}


def _stock():
    """pykrx 는 import 시점에 KRX 로그인을 한다 (모듈 캐시 → 프로세스당 1회). 실패를 명확한 에러로."""
    if not (os.getenv('KRX_ID') and os.getenv('KRX_PW')):
        raise RuntimeError('KRX_ID/KRX_PW 환경변수 없음 — 최신 시세(pykrx)는 KRX 로그인이 필요하다')
    if 'pykrx.stock' not in sys.modules:
        CALLS['pykrx_login'] += 1
    try:
        from pykrx import stock
    except Exception as e:  # 로그인 응답이 HTML 에러 페이지면 JSONDecodeError
        raise RuntimeError(f'pykrx import(KRX 로그인) 실패: {e}') from e
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


def update_store(keep: int = KEEP_DAYS, now: Optional[datetime] = None) -> List[str]:
    """저장소를 최신화하고 사용할 거래일 목록(오래된→최신, 최대 keep개)을 반환."""
    now = now or datetime.now()
    today = now.strftime('%Y%m%d')
    DAILY_DIR.mkdir(parents=True, exist_ok=True)
    hol_path = DAILY_DIR / 'holidays.txt'
    hol = set(hol_path.read_text().split()) if hol_path.exists() else set()
    final = {p.stem for p in DAILY_DIR.glob('????????.parquet')}
    # 평일 후보 (휴장 ~4% 감안해 keep+60 평일 ≈ keep 거래일 이상)
    cands = [d.strftime('%Y%m%d') for d in pd.bdate_range(end=now.date(), periods=keep + 60)]
    floor = min(final) if len(final) >= keep else ''  # 창이 다 찼으면 정리된 오래된 날짜는 다시 받지 않음
    past = [d for d in cands if floor < d < today and d not in final and d not in hol]
    print(f'[저장소] 확정 {len(final)}일 보유, 과거 {len(past)}일 OpenAPI 수집', flush=True)

    def job(d):
        try:
            return d, _fetch_openapi(d), None
        except Exception as e:
            return d, None, e

    with ThreadPoolExecutor(WORKERS) as ex:
        results = list(ex.map(job, past))
    errors, pending = [], []
    for d, df, err in results:
        if err is not None:
            errors.append((d, err))
        elif df is None:
            pending.append(d)
        else:
            _write(df, DAILY_DIR / f'{d}.parquet')
            final.add(d)
    latest = max(final) if final else ''
    # ponytail: 더 최근 날짜가 게시됐는데 비어 있으면 휴장으로 간주 (OpenAPI 가 거래일을 빈 응답으로
    #           주면 구멍이 생긴다 — 그런 일이 보이면 pykrx 로 재확인하도록 바꿀 것)
    hol |= {d for d in pending if d < latest}
    recent = [d for d in pending if d > latest]  # OpenAPI 미게시 → pykrx

    # 최신 구간: OpenAPI 미게시일 + 오늘 장중 (로그인 필요)
    need = recent + ([today] if now.weekday() < 5 and today not in hol else [])
    stale, live_ok = [], False
    stock, login_err = None, None
    if need:
        try:
            stock = _stock()
        except Exception as e:
            login_err = e
    for d in need:
        try:
            if stock is None:
                raise login_err
            df = _fetch_pykrx(stock, d)
        except Exception as e:
            if d < today:
                stale.append((d, e))
            else:
                logging.warning(f'오늘({d}) 장중 시세 수집 실패 → 마지막 확정 거래일로 진행: {e}')
            continue
        if df is None:
            if d < today:
                hol.add(d)
            else:
                print(f'  오늘({d}) 시세 없음 — 장 시작 전 또는 휴장')
        elif d < today:
            _write(df, DAILY_DIR / f'{d}.parquet')
            final.add(d)
        else:
            _write(df, DAILY_DIR / f'{d}_live.parquet')
            live_ok = True
    hol_path.write_text('\n'.join(sorted(hol)))

    for d, err in errors + stale:
        logging.warning(f'날짜 수집 실패 {d}: {err} (기존 파일 유지)')
    if stale:
        raise RuntimeError(f'마지막 확정 거래일 수집 실패 (OpenAPI 미게시 + pykrx 실패): {[d for d, _ in stale]}')
    if errors:
        raise RuntimeError(f'과거 거래일 {len(errors)}개 수집 실패 (예: {[d for d, _ in errors[:5]]}) — 다음 실행에서 재시도')

    dates = (sorted(final) + ([today] if live_ok else []))[-keep:]
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

    # 자체 점검: 갱신 정책 (가짜 OpenAPI/로그인, 임시 폴더)
    import tempfile
    day = pd.DataFrame({'Open': [1], 'High': [1], 'Low': [1], 'Close': [1], 'Volume': [1], 'Change': [0.0],
                        'MarketCap': [1], 'Name': ['x'], 'Market': ['KOSPI']}, index=pd.Index(['000010'], name='Ticker'))
    now = datetime(2026, 10, 13, 11)  # 화요일 장중. 10-09(금) 휴장, 10-12(월) OpenAPI 미게시
    published = {d.strftime('%Y%m%d') for d in pd.bdate_range(end='2026-10-08', periods=20)}

    def _no_login():
        raise RuntimeError('login blocked')
    _stock = _no_login
    _fetch_openapi = lambda d: day if d in published else None
    with tempfile.TemporaryDirectory() as tmp:
        DAILY_DIR = Path(tmp)
        try:  # 10-12 확정분을 못 받음 → 실패해야 함
            update_store(keep=10, now=now)
            raise AssertionError('stale 미감지')
        except RuntimeError as e:
            assert '20261012' in str(e), e
        published.add('20261012')  # OpenAPI 게시 후: 오늘 장중만 로그인 실패 → 경고 후 진행
        dates = update_store(keep=10, now=now)
        assert dates[-1] == '20261012' and '20261009' not in dates, dates
        assert '20261009' in (DAILY_DIR / 'holidays.txt').read_text()
    print('ok')
