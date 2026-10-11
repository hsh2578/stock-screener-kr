"""
ML 학습용 이벤트 데이터셋 생성 (로컬 전용 — 상장폐지 종목 포함 일봉 패널 필요)

현재 run_screeners.py 의 스크리너 규칙을 과거 전 기간에 그대로 적용해 이벤트를 만들고,
피처는 추론과 같은 함수로 계산한다.
  - 박스권 돌파   : run_screeners.is_box_range + calculate_box_features
  - 52주 신고가   : ml/src/collect_training_data.calculate_features (run_screeners.calculate_52w_features 와 동일)
  - 52주 근접     : run_screeners.calculate_near_high_features

피처 계산 입력은 이벤트일까지 최근 FEATURE_HISTORY_ROWS 행 (추론도 같은 길이로 자른다).

라벨 (C = 이벤트일 종가)
  - box       : max(High[t+1..t+20]) / C - 1 >= 15%
  - 52w       : max(Close[t+1..t+20]) / C - 1 >= 15%
  - near_high : any Close[t+1..t+10] > 52주고가 x 1.015  (사이트 돌파 판정과 같은 종가 기준)
상장폐지로 미래 구간이 짧으면 남은 구간으로 라벨을 매긴다 (생존편향 방지). 상장 유지 종목은
미래 구간이 다 차야 라벨을 쓴다.

사용: python ml/build_events.py [--panel <dir>]
출력: ml/data/events_box.csv, events_52w.csv, events_near_high.csv
"""
import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT_DIR = ROOT / "ml" / "data"
DEFAULT_PANEL = Path(r"C:/Users/hsh/Desktop/vibecoding/기술적분석 에이전트/backtest/data/panel")

MIN_MCAP = 1000 * 1e8      # 1000억 (시점 기준)
FEATURE_HISTORY_ROWS = 500  # run_screeners.FEATURE_HISTORY_ROWS 와 같아야 한다
HORIZON = 20
NEAR_HORIZON = 10
GAIN_TH = 15.0

_rs = None
_market = None


def _init(market):
    global _rs, _market
    import run_screeners as rs
    rs._kospi_data = market
    _rs, _market = rs, market


def _future(arr, t, n, delisted):
    seg = arr[t + 1:t + 1 + n]
    if len(seg) < n and not delisted:
        return None
    return seg if len(seg) else None


def _scan_stock(args):
    code, df, delisted = args
    rs = _rs
    from ml.src.collect_training_data import calculate_features
    close, high = df['Close'], df['High']
    c, h, v = close.values, high.values, df['Volume'].values
    mcap_ok = (df['mcap'] >= MIN_MCAP).values
    n = len(df)
    out = {'box': [], '52w': [], 'near_high': []}

    def win(t):
        s = max(0, t - FEATURE_HISTORY_ROWS + 1)
        return df.iloc[s:t + 1], t - s

    # ---------- 52주 신고가 / 근접 공통: H = max High[t-265, t-15) ----------
    H = high.shift(rs.HIGH_52W_GAP_DAYS + 1).rolling(rs.HIGH_52W_PERIOD).max()
    lvl = H * rs.BREAKOUT_THRESHOLD

    # 52주 신고가 돌파 (screen_new_high_52w)
    ma150 = close.rolling(150).mean()
    already = close.shift(1).rolling(rs.HIGH_52W_GAP_DAYS).max() > lvl
    recent_high = high.shift(1).rolling(rs.HIGH_52W_GAP_DAYS).max()
    cond = (close > ma150) & (ma150 > ma150.shift(20)) & (close > lvl) & (close > recent_high) & ~already
    market_kr = _market.rename(columns={'Close': '종가'})
    for t in np.flatnonzero(cond.values & mcap_ok):
        if t < rs.HIGH_52W_PERIOD + rs.HIGH_52W_GAP_DAYS + 1:
            continue
        fut = _future(c, t, HORIZON, delisted)
        if fut is None:
            continue
        w, i = win(t)
        f = calculate_features(w.rename(columns=rs._KR_COLS), i, 0, market_kr, 'KOSPI')
        if f is None:
            continue
        gain = (fut.max() / c[t] - 1) * 100
        out['52w'].append({'ticker': code, 'date': df.index[t], **f,
                           'max_gain_20d': gain, 'label': int(gain >= GAIN_TH), 'delisted': delisted})

    # 52주 신고가 근접 (screen_near_high_52w)
    ever = close.rolling(rs.HIGH_52W_GAP_DAYS + 1).max() > lvl
    high_hit = high.rolling(rs.HIGH_52W_GAP_DAYS + 1).max() > H
    pre_gap = (H - close.shift(rs.NEAR_HIGH_MAX_DAYS + 1)) / H * 100
    gap = (H - close) / H * 100
    cond = (~ever & ~high_hit & (close >= H * 0.9)
            & ~((pre_gap > 0) & (pre_gap <= rs.NEAR_HIGH_52W_THRESHOLD))
            & (gap <= rs.NEAR_HIGH_52W_THRESHOLD) & (close < H))
    last = -10 ** 9
    min_rows = rs.HIGH_52W_PERIOD + rs.NEAR_HIGH_MAX_DAYS + 20
    for t in np.flatnonzero(cond.values & mcap_ok):
        if t < min_rows - 1 or t - last <= rs.NEAR_HIGH_MAX_DAYS:  # 사이트는 8거래일 동안 같은 진입을 보존
            continue
        fut10 = _future(c, t, NEAR_HORIZON, delisted)
        futh = _future(h, t, HORIZON, delisted)
        if fut10 is None or futh is None:
            continue
        w, i = win(t)
        f = rs.calculate_near_high_features(w, i, float(H.iloc[t]))
        if f is None:
            continue
        last = t
        out['near_high'].append({'ticker': code, 'date': df.index[t], 'high_52w': float(H.iloc[t]), **f,
                                 'max_gain_20d': (futh.max() / c[t] - 1) * 100,
                                 'label': int((fut10 > lvl.iloc[t]).any()), 'delisted': delisted})

    # ---------- 박스권 돌파 (screen_box_breakout_simple / screen_box_breakout) ----------
    P, BW = rs.BOX_PERIOD, rs.BREAKOUT_WINDOW
    cmax, cmin = close.rolling(P).max(), close.rolling(P).min()
    quick = (cmax - cmin) / cmin * 100 <= rs.MAX_BOX_RANGE_PERCENT        # quick_range_check
    fwd = close[::-1].rolling(BW, min_periods=1).max()[::-1].shift(-1)     # max Close[e+1..e+11]
    cand = quick & (fwd > cmax * 1.015)                                    # 필요조건 (box_high = int(cmax) <= cmax)
    breaks = {}
    for e in np.flatnonzero(cand.values):
        if e < 2 * P - 1:
            continue
        rs._ATR_CACHE.clear()
        ok, box = rs.is_box_range(df.iloc[e - 2 * P + 1:e + 1], P, use_cache=False)
        if not ok:
            continue
        above = np.flatnonzero(c[e + 1:e + 1 + BW] > box['box_high'] * 1.015)
        if len(above):
            b = e + 1 + above[0]
            breaks.setdefault(b, (box['box_high'], box['box_low']))  # e 오름차순 → 가장 오래된 박스 우선
    vol20 = df['Volume'].shift(1).rolling(20).mean()
    last = -10 ** 9
    for b in sorted(breaks):
        if b - last <= BW - 1 or not mcap_ok[b] or b < 20:  # 사이트는 10거래일 동안 같은 돌파를 보존
            continue
        futh = _future(h, b, HORIZON, delisted)
        if futh is None:
            continue
        bh, bl = breaks[b]
        w, i = win(b)
        f = rs.calculate_box_features(w, i, bh, bl)
        if f is None:
            continue
        last = b
        vr = v[b] / vol20.iloc[b] if vol20.iloc[b] > 0 else 0
        gain = (futh.max() / c[b] - 1) * 100
        out['box'].append({'ticker': code, 'date': df.index[b], 'box_high': bh, 'box_low': bl, **f,
                           'vol_screener': int(vr >= 2 and c[b] > ma150.iloc[b]),
                           'max_gain_20d': gain, 'label': int(gain >= GAIN_TH), 'delisted': delisted})
    return out


def load_panel(panel: Path):
    rd = lambda n: pd.read_parquet(panel / f"{n}.parquet")
    o, h, l, c = rd('adj_open'), rd('adj_high'), rd('adj_low'), rd('adj_close')
    vol = rd('trdval') / c          # 수정주가 기준 거래량 → Close x Volume = 실제 거래대금
    mcap = rd('mktcap')
    meta = rd('meta')
    return o, h, l, c, vol, mcap, meta


def market_proxy(c, mcap):
    """run_screeners.load_kospi_data 와 같은 등가중 대리지수. 편입 = 전일 시총 1000억+ (시점 기준)."""
    ret = c.pct_change(fill_method=None)
    member = (mcap.shift(1) >= MIN_MCAP)
    avg = ret.where(member).mean(axis=1).fillna(0)
    return pd.DataFrame({'Close': (1 + avg).cumprod() * 1000})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--panel', type=Path, default=DEFAULT_PANEL)
    ap.add_argument('--limit', type=int, default=0, help='디버그용 종목 수 제한')
    ap.add_argument('--workers', type=int, default=20)
    a = ap.parse_args()

    o, h, l, c, vol, mcap, meta = load_panel(a.panel)
    market = market_proxy(c, mcap)
    end = c.index[-1]
    codes = list(c.columns[:a.limit] if a.limit else c.columns)
    print(f"패널 {c.index[0].date()} ~ {end.date()}, 종목 {len(codes)}")

    def tasks():
        for code in codes:
            df = pd.DataFrame({'Open': o[code], 'High': h[code], 'Low': l[code], 'Close': c[code],
                               'Volume': vol[code], 'mcap': mcap[code]})
            s, e = df['Close'].first_valid_index(), df['Close'].last_valid_index()
            if s is None:
                continue
            df = df.loc[s:e].ffill().fillna({'Volume': 0})
            if len(df) < 300:
                continue
            yield code, df, bool(e < end)

    res = {'box': [], '52w': [], 'near_high': []}
    with ProcessPoolExecutor(a.workers, initializer=_init, initargs=(market,)) as ex:
        for k, r in enumerate(ex.map(_scan_stock, tasks(), chunksize=8)):
            for m in res:
                res[m] += r[m]
            if k % 200 == 0:
                print(f"  {k} 종목 처리, 이벤트 box={len(res['box'])} 52w={len(res['52w'])} near={len(res['near_high'])}", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for m, rows in res.items():
        df = pd.DataFrame(rows).sort_values(['date', 'ticker']).reset_index(drop=True)
        df.to_csv(OUT_DIR / f"events_{m}.csv", index=False, encoding='utf-8-sig')
        print(f"[{m}] {len(df)}건, {df['ticker'].nunique()}종목 (상폐 {df.loc[df.delisted, 'ticker'].nunique()}), "
              f"{df['date'].min().date()}~{df['date'].max().date()}, 양성 {df['label'].mean():.3f}")


if __name__ == '__main__':
    main()
