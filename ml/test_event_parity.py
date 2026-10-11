"""
build_events.py 이벤트가 실제 run_screeners 스크리너 판정과 같은지 확인 (로컬 패널 필요).
몇 종목·구간에서 매일 스크리너를 '그날까지 데이터'로 돌려 비교하고, 피처 값도 대조한다.
사용: python ml/test_event_parity.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import build_events as be  # noqa: E402  (ml/ 가 sys.path[0])
import run_screeners as rs  # noqa: E402

CODES = ['005930', '035720', '086520', '000080', '000270', '000230', '000180', '000050']
START, END = '2019-06-01', '2021-06-30'


def main():
    o, h, l, c, vol, mcap, _ = be.load_panel(be.DEFAULT_PANEL)
    market = be.market_proxy(c, mcap)
    be._init(market)
    rs._preserve_prev_results = lambda _f, new, *a, **k: new
    rs.load_box_breakout_models(); rs.load_52w_models(); rs.load_near_high_models()
    bad = 0
    for code in CODES:
        df = pd.DataFrame({'Open': o[code], 'High': h[code], 'Low': l[code], 'Close': c[code],
                           'Volume': vol[code], 'mcap': mcap[code]})
        s, e = df['Close'].first_valid_index(), df['Close'].last_valid_index()
        df = df.loc[s:e].ffill().fillna({'Volume': 0})
        ev = be._scan_stock((code, df, bool(e < c.index[-1])))
        evd = {m: {pd.Timestamp(r['date']): r for r in ev[m]} for m in ev}
        got = {'52w': {}, 'near_high': {}, 'box': {}}
        for d in df.loc[START:END].index:
            cut = df.loc[:d]
            rs._DATA_CACHE.clear(); rs._DATA_CACHE[code] = cut.drop(columns='mcap')
            rs._market_latest_date_cache = None
            stocks = pd.DataFrame([{'Code': code, 'Name': code, 'MarketCap': cut['mcap'].iloc[-1] / 1e8}])
            for r in rs.screen_new_high_52w(stocks):
                got['52w'][pd.Timestamp(r['breakout_date'])] = r
            for r in rs.screen_near_high_52w(stocks):
                got['near_high'][pd.Timestamp(r['entry_date'])] = r
            for r in rs.screen_box_breakout_simple(stocks):
                got['box'].setdefault(pd.Timestamp(r['breakout_date']), r)
        win = lambda x: {k: v for k, v in x.items() if pd.Timestamp(START) <= k <= pd.Timestamp(END)}
        for m, dedupe in (('52w', 0), ('near_high', 9), ('box', 11)):
            mine, theirs = win(evd[m]), got[m]
            missing = [d for d in mine if d not in theirs]
            idx = df.index
            extra = [d for d in theirs if d not in mine and not any(
                0 < idx.get_loc(d) - idx.get_loc(x) < dedupe for x in evd[m])]
            print(f"{code} {m}: 이벤트 {len(mine)} / 스크리너 {len(theirs)}  누락 {missing[:3]} 초과 {extra[:3]}")
            bad += len(missing) + len(extra)
        # 피처 대조: 52w 는 추론 경로(run_screeners.calculate_52w_features + 500행)로 다시 계산
        for d, r in win(evd['52w']).items():
            full = df.drop(columns='mcap').loc[:d]
            f = rs.calculate_52w_features(full.tail(rs.FEATURE_HISTORY_ROWS), len(full.tail(rs.FEATURE_HISTORY_ROWS)) - 1)
            diff = max(abs(f[k] - r[k]) for k in rs.FEATURES_52W)
            if diff > 1e-6:
                print(f"  52w 피처 불일치 {code} {d.date()} {diff}"); bad += 1
    print("OK" if bad == 0 else f"불일치 {bad}건")
    assert bad == 0


if __name__ == '__main__':
    main()
