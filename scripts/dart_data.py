"""
DART 분기 재무 시계열 (2015~) — dartlab HuggingFace 미러 (CC BY 4.0, API 키 불필요).

FnGuide 는 분기 4개·연간 5년만 주므로 대가 스크리너에 필요한
  - 전년 동기 지배자본 (ROE 평균자본)
  - 전년 TTM 지배순이익 (EPS 성장률)
  - 10년 전 EPS (그레이엄)
를 여기서 가져온다. 미러 파일은 DART fnlttSinglAcntAll 원본 행을 종목별로 모은 parquet 이다.
추출 규칙은 kr-value-lens/scripts/fetch_dart_quarterly.py 와 같다.

자체 점검: python scripts/dart_data.py
"""

import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Iterable, Optional, Tuple

import pandas as pd

DARTLAB_URL = 'https://huggingface.co/datasets/eddmpython/dartlab-data/resolve/main/dart/{kind}/{code}.parquet'
CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.cache', 'dartlab')
CACHE_DAYS = 7
QUARTER = {'11013': 1, '11012': 2, '11014': 3, '11011': 4}  # 보고서 코드 → 회계연도 내 분기

Period = Tuple[int, int]  # (사업연도, 분기 1~4)
Series = Dict[Period, Dict[str, Optional[float]]]

_NI = 'ifrs-full_ProfitLossAttributableToOwnersOfParent'
_NI_TOT = 'ifrs-full_ProfitLoss'
_NI_NCI = 'ifrs-full_ProfitLossAttributableToNoncontrollingInterests'
_EQ = 'ifrs-full_EquityAttributableToOwnersOfParent'
_EQ_TOT = 'ifrs-full_Equity'
_EQ_NCI = 'ifrs-full_NoncontrollingInterests'
_EPS = 'ifrs-full_BasicEarningsLossPerShare'


def _amt(v) -> Optional[float]:
    s = str(v if v is not None else '').replace(',', '').strip()
    if s in ('', '-', 'nan', 'None'):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _pick(rows, account_id, field, sj) -> Optional[float]:
    ids = (account_id, account_id.replace('ifrs-full_', 'ifrs_'))  # 2019 이전 공시는 옛 접두사
    for r in rows:
        if r['account_id'] in ids and r['sj_div'] in sj:
            v = _amt(r.get(field))
            if v is not None:
                return v
    return None


def extract(rows, fs: str, q: int, prior: bool = False) -> Dict[str, Optional[float]]:
    """
    q분기(4 = 사업보고서) 보고서 한 건의 행 → {ni, ni_tot, eq, assets, eps}. 손익은 당해 누적.
    prior=True 면 같은 보고서의 '전기' 비교 칸: 손익 = 전년 동기 누적, 재무상태 = 전년 말.
    누적 칸(add_amount)이 빈 경우 3개월 칸으로 대체하는 것은 1분기뿐 (2·3분기에 그러면 3개월치를 누적으로 잘못 읽는다).
    """
    pre = 'frmtrm' if prior else 'thstrm'
    flow_fields = {4: (f'{pre}_amount',), 1: (f'{pre}_add_amount', 'frmtrm_q_amount' if prior else 'thstrm_amount')}.get(
        q, (f'{pre}_add_amount',))
    stock_field = f'{pre}_amount'

    def flow(account_id):
        for f in flow_fields:
            v = _pick(rows, account_id, f, ('IS', 'CIS'))
            if v is not None:
                return v
        return None

    def stock(account_id):
        return _pick(rows, account_id, stock_field, ('BS',))

    ni_tot, eq_tot = flow(_NI_TOT), stock(_EQ_TOT)
    if fs == 'OFS':
        ni, eq = ni_tot, eq_tot
    else:  # 지배 태그를 안 쓰는 회사는 전체 − 비지배
        ni = flow(_NI)
        if ni is None and ni_tot is not None:
            ni = ni_tot - (flow(_NI_NCI) or 0)
        eq = stock(_EQ)
        if eq is None and eq_tot is not None:
            eq = eq_tot - (stock(_EQ_NCI) or 0)
    return {'ni': ni, 'ni_tot': ni_tot, 'eq': eq, 'assets': stock('ifrs-full_Assets'), 'eps': flow(_EPS)}


_FLOW_KEYS, _STOCK_KEYS = ('ni', 'ni_tot', 'eps'), ('eq', 'assets')


def parse(raw: pd.DataFrame) -> Series:
    """
    미러 parquet → {(연도, 분기): {ni, ni_tot, eq, assets, eps}}. 연결 우선, 없으면 별도. 원화 공시가 아니면 {}.
    미러에 빠진 보고서는 다음 해 보고서의 전기 비교 칸으로 메운다 (손익 → 전년 동기, 재무상태 → 전년 말).
    """
    if not {'fs_div', 'reprt_code', 'bsns_year', 'account_id', 'sj_div'} <= set(raw.columns):
        return {}  # 재무 없는 회사는 스키마가 다르다
    df = raw.dropna(subset=['fs_div', 'reprt_code', 'bsns_year'])
    if 'currency' in df.columns and not df['currency'].dropna().isin(['KRW']).all():
        return {}  # 달러·위안 공시 (원화 시총과 섞으면 비율이 깨진다)
    out: Series = {}
    reports = []
    for (year, reprt, fs), g in sorted(df.groupby(['bsns_year', 'reprt_code', 'fs_div']),
                                       key=lambda kv: kv[0][2] != 'CFS'):  # CFS 먼저
        q = QUARTER.get(reprt)
        if q is None or (int(year), q) in out:
            continue
        rows = g.to_dict('records')
        v = extract(rows, fs, q)
        if v['ni'] is not None or v['eq'] is not None:
            out[(int(year), q)] = v
            reports.append((int(year), q, fs, rows))

    empty = dict.fromkeys(_FLOW_KEYS + _STOCK_KEYS)
    for year, q, fs, rows in reports:
        if not {'frmtrm_amount', 'frmtrm_add_amount'} <= set(rows[0]):
            break
        v = extract(rows, fs, q, prior=True)
        for period, keys in (((year - 1, q), _FLOW_KEYS), ((year - 1, 4), _STOCK_KEYS)):
            for k in keys:
                if v[k] is not None and (out.get(period) or empty)[k] is None:
                    out.setdefault(period, dict(empty))[k] = v[k]
    return out


def _mirror(kind: str, code: str) -> Optional[pd.DataFrame]:
    """미러 parquet (kind: finance=재무제표 원본, report=정기보고서 항목). 내려받기·읽기 실패는 None."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f'{kind}_{code}.parquet')
    if not os.path.exists(path) or time.time() - os.path.getmtime(path) > CACHE_DAYS * 86400:
        try:
            urllib.request.urlretrieve(DARTLAB_URL.format(kind=kind, code=code), path + '.tmp')
            os.replace(path + '.tmp', path)
        except Exception:
            if not os.path.exists(path):
                return None  # 이전 파일이 있으면 그걸 쓴다
    try:
        return pd.read_parquet(path)
    except Exception:
        return None


def dart_series(code: str) -> Series:
    """종목의 분기 시계열. 자료가 없으면 {} (호출부가 '판단 불가'로 처리)."""
    raw = _mirror('finance', code)
    return {} if raw is None else parse(raw)


def parse_dps(raw: pd.DataFrame) -> Optional[Tuple[float, int]]:
    """
    정기보고서 배당 항목 → (보통주 주당 현금배당금, 배당수익률 분모로 쓸 종가의 연도).
    - 최근 보고서가 사업보고서: 당기 배당 / 그 해 연말 종가
    - 최근 보고서가 분기·반기: 당기 누적(분기·중간배당), 비어 있으면 전기(직전 연간) 배당 / 전년 말 종가
    배당이 없으면 None.
    """
    need = {'apiType', 'se', 'thstrm', 'frmtrm', 'year', 'quarter', 'stlm_dt'}
    if not need <= set(raw.columns):
        return None
    df = raw[(raw['apiType'] == 'dividend') & raw['se'].astype(str).str.contains('주당 현금배당금')]
    if 'stock_knd' in df.columns:
        df = df[~df['stock_knd'].astype(str).str.contains('우선')]
    df = df.dropna(subset=['stlm_dt'])
    if df.empty:
        return None
    latest = df[df['stlm_dt'] == df['stlm_dt'].max()]
    year = int(latest['year'].iloc[0])
    annual = str(latest['quarter'].iloc[0]).startswith('4')
    for col in (('thstrm',) if annual else ('thstrm', 'frmtrm')):
        vals = [v for v in map(_amt, latest[col]) if v]
        if vals:
            return vals[0], year if annual else year - 1
    return None


def dart_dps(code: str) -> Optional[Tuple[float, int]]:
    raw = _mirror('report', code)
    try:
        return None if raw is None else parse_dps(raw)
    except Exception:
        return None


def batch(fn, codes: Iterable[str], workers: int = 8) -> dict:
    """fn = dart_series 또는 dart_dps 를 여러 종목에 병렬 적용 → {code: 결과}."""
    codes = list(dict.fromkeys(codes))
    with ThreadPoolExecutor(workers) as ex:
        return dict(zip(codes, ex.map(fn, codes)))


def latest_period(series: Series) -> Optional[Period]:
    """지배순이익과 지배자본이 모두 있는 가장 최근 분기."""
    ok = [p for p, v in series.items() if v['ni'] is not None and v['eq'] is not None]
    return max(ok) if ok else None


def ttm(series: Series, period: Period, key: str) -> Optional[float]:
    """period 까지 최근 4분기 합 = 당해 누적 + 전년 연간 − 전년 동기 누적."""
    year, q = period
    cur = (series.get(period) or {}).get(key)
    if q == 4 or cur is None:
        return cur
    prev_fy = (series.get((year - 1, 4)) or {}).get(key)
    prev_cum = (series.get((year - 1, q)) or {}).get(key)
    if prev_fy is None or prev_cum is None:
        return None
    return cur + prev_fy - prev_cum


def ttm_growth(series: Series, key: str = 'ni') -> Optional[float]:
    """최근 TTM / 1년 전 TTM − 1 (%). 1년 전 TTM 이 0 이하면 None (성장률 정의 불가)."""
    p = latest_period(series)
    if p is None:
        return None
    cur, prev = ttm(series, p, key), ttm(series, (p[0] - 1, p[1]), key)
    if cur is None or prev is None or prev <= 0:
        return None
    return (cur / prev - 1) * 100


def return_on(series: Series, key: str) -> Optional[float]:
    """
    TTM 전체 순이익(비지배 포함) / (최근 분기 key + 1년 전 key) 평균 × 100.
    key='eq' → ROE (지배자본 기준), key='assets' → ROA.
    1년 전 분기가 없으면 그 앞뒤 분기로 대체 (ponytail: 한 분기 어긋난 평균, 미러에 보고서가 빠진 종목만 해당).
    """
    p = latest_period(series)
    if p is None:
        return None
    ni, cur = ttm(series, p, 'ni_tot'), series[p].get(key)
    y, q = p[0] - 1, p[1]
    prev = next((v for v in ((series.get(c) or {}).get(key) for c in ((y, q), (y, q + 1), (y, q - 1))) if v is not None), None)
    if ni is None or cur is None or prev is None or cur + prev <= 0:
        return None
    return ni / ((cur + prev) / 2) * 100


def _prev_quarter(p: Period) -> Period:
    return (p[0], p[1] - 1) if p[1] > 1 else (p[0] - 1, 4)


def all_quarters_profitable(series: Series, n: int = 20, key: str = 'ni_tot', min_known: int = 16) -> bool:
    """
    최근 n개 분기의 분기별(3개월) 값이 전부 0 초과인가. 분기값 = 당해 누적 − 직전 분기 누적.
    미러에 보고서가 빠져 알 수 없는 분기는 건너뛰되, 확인된 분기가 min_known 개 미만이면 False
    (상장한 지 얼마 안 됐거나 자료가 부족해 '5년 흑자'를 확인할 수 없음).
    """
    p = latest_period(series)
    if p is None:
        return False
    known = 0
    for _ in range(n):
        cur = (series.get(p) or {}).get(key)
        prev = 0 if p[1] == 1 else (series.get(_prev_quarter(p)) or {}).get(key)
        if cur is not None and prev is not None:
            if cur - prev <= 0:
                return False
            known += 1
        p = _prev_quarter(p)
    return known >= min_known


def avg3_growth(series: Series, key: str = 'ni', years: int = 10) -> Optional[float]:
    """
    그레이엄 방식 장기 이익 성장률(%): 최근 years 년 동안 매년 같은 분기 시점의 TTM 값을 늘어놓고
    (가장 오래된 3개 평균 → 최근 3개 평균) 증가율. 값이 6개 미만이거나 처음 3개 평균이 0 이하면 None.
    3년 평균을 쓰는 이유: 한 해의 일시적 호황·불황이 시작점·끝점에 걸려 성장률이 왜곡되지 않게 (『현명한 투자자』 14장).
    """
    p = latest_period(series)
    if p is None:
        return None
    vals = [v for v in (ttm(series, (p[0] - k, p[1]), key) for k in range(years, -1, -1)) if v is not None]
    if len(vals) < 6:
        return None
    first, last = sum(vals[:3]) / 3, sum(vals[-3:]) / 3
    return (last / first - 1) * 100 if first > 0 else None


def demo():
    row = lambda sj, acc, cur, add=None: {'sj_div': sj, 'account_id': acc, 'thstrm_amount': cur,
                                          'thstrm_add_amount': add}
    # 분기보고서: 손익은 누적 칸, 자본은 당기 칸. 지배 태그 없으면 전체 − 비지배
    v = extract([row('IS', _NI_TOT, '30', '100'), row('IS', _NI_NCI, '3', '10'),
                 row('BS', _EQ_TOT, '1,000'), row('BS', _EQ_NCI, '200'), row('IS', _EPS, '5', '17')], 'CFS', 3)
    assert v == {'ni': 90.0, 'ni_tot': 100.0, 'eq': 800.0, 'assets': None, 'eps': 17.0}, v
    assert extract([row('IS', 'ifrs_ProfitLoss', '7')], 'OFS', 4)['ni'] == 7.0  # 옛 접두사, 연간은 당기 칸
    assert extract([row('IS', _NI_TOT, '7')], 'OFS', 1)['ni'] == 7.0, '1분기: 누적 칸이 비면 3개월 칸'
    assert extract([row('IS', _NI_TOT, '7')], 'OFS', 3)['ni'] is None, '3분기: 3개월 칸을 누적으로 쓰지 않는다'

    # 미러에 2025 3분기 보고서가 없어도 2026 3분기 보고서의 전기 칸으로 메운다
    rep = lambda sj, acc, **kw: {'bsns_year': '2026', 'reprt_code': '11014', 'fs_div': 'OFS', 'sj_div': sj,
                                 'account_id': acc, 'thstrm_amount': None, 'thstrm_add_amount': None,
                                 'frmtrm_amount': None, 'frmtrm_add_amount': None, **kw}
    filled = parse(pd.DataFrame([rep('IS', _NI_TOT, thstrm_amount='40', thstrm_add_amount='120', frmtrm_add_amount='90'),
                                 rep('BS', _EQ_TOT, thstrm_amount='1200', frmtrm_amount='1050')]))
    assert filled[(2026, 3)]['ni'] == 120 and filled[(2025, 3)]['ni'] == 90 and filled[(2025, 3)]['eq'] is None
    assert filled[(2025, 4)]['eq'] == 1050, '분기보고서 재무상태표의 전기 = 전년 말'

    s = {(2024, 3): {'ni': 60, 'eq': 900}, (2024, 4): {'ni': 100, 'eq': 950},
         (2025, 3): {'ni': 90, 'eq': 1000}, (2025, 4): {'ni': 130, 'eq': 1050},
         (2026, 3): {'ni': 120, 'eq': 1200}}
    assert latest_period(s) == (2026, 3)
    assert ttm(s, (2026, 3), 'ni') == 120 + 130 - 90 == 160
    assert ttm(s, (2025, 3), 'ni') == 90 + 100 - 60 == 130
    assert abs(ttm_growth(s) - (160 / 130 - 1) * 100) < 1e-9
    for v in s.values():
        v['ni_tot'] = v['ni']
    assert abs(return_on(s, 'eq') - 160 / 1100 * 100) < 1e-9
    assert return_on(s, 'assets') is None  # 자산 없음
    gap = {k: dict(v) for k, v in s.items() if k != (2025, 3)}
    gap[(2025, 3)] = {'ni': 90, 'ni_tot': 90, 'eq': None}
    assert abs(return_on(gap, 'eq') - 160 / ((1200 + 1050) / 2) * 100) < 1e-9, '1년 전 분기 자본이 없으면 전년 말로 대체'

    # 분기 흑자: 2025 3분기 누적 90 → 연간 130 (4분기 +40), 2026 3분기까지 누적만 있으면 1·2분기는 알 수 없음
    q = {(2025, k): {'ni': v, 'ni_tot': v, 'eq': 1} for k, v in zip((1, 2, 3, 4), (10, 30, 90, 130))}
    assert all_quarters_profitable(q, n=4, min_known=4)
    assert not all_quarters_profitable({**q, (2025, 4): {'ni': 80, 'ni_tot': 80, 'eq': 1}}, n=4, min_known=4), '4분기 적자(-10)'
    assert not all_quarters_profitable(q, n=20), '확인된 분기 4개 < 16'
    yearly = {(y, 4): {'ni': v, 'eq': 1} for y, v in zip(range(2019, 2026), (100, 100, 100, 50, 120, 150, 150))}
    assert abs(avg3_growth(yearly) - 40) < 1e-9, '(120+150+150)/3 = 140 vs 100'
    assert avg3_growth({k: v for k, v in yearly.items() if k[0] >= 2021}) is None, '값 6개 미만'

    div = pd.DataFrame({'apiType': 'dividend', 'se': '주당 현금배당금(원)', 'year': '2026',
                        'quarter': ['2분기', '2분기', '3분기', '3분기'],
                        'stlm_dt': ['2026-06-30', '2026-06-30', '2026-09-30', '2026-09-30'],
                        'stock_knd': ['보통주', '우선주', None, '우선주'],
                        'thstrm': ['1,200', '1,250', '-', '-'], 'frmtrm': ['2,400', '2,450', '2,400', '2,450']})
    assert parse_dps(div) == (2400.0, 2025)  # 최근 보고서 당기가 비면 전기
    assert parse_dps(div[div['stlm_dt'] == '2026-06-30']) == (1200.0, 2025)  # 분기 누적 배당도 전년 말 종가로 나눈다
    assert parse_dps(div.assign(quarter='4분기')) is None  # 사업보고서인데 당기 배당 없음 = 무배당
    assert ttm_growth({(2026, 3): {'ni': 1, 'eq': 1}}) is None  # 전년 자료 없음
    print('dart_data self-check OK')


if __name__ == '__main__':
    demo()
