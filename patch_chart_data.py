"""
chart_data.json 패치: 퀀트/대가 스크리너 누락 종목 차트 데이터 추가

value-screener.yml에서 모든 스크리너 실행 후 호출합니다.
chart_data.json에 없는 종목의 OHLCV를 수집하여 추가합니다.
"""

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

import FinanceDataReader as fdr

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
CHART_DATA_FILE = os.path.join(DATA_PATH, 'chart_data.json')

# 패치 대상 JSON 파일
TARGET_FILES = [
    'value_stocks.json', 'ma60w_quality.json',
    'magic_formula.json', 'multi_factor.json',
    'buffett.json', 'ackman.json', 'lynch.json', 'graham.json',
]


def fetch_chart_data(ticker: str) -> tuple:
    """단일 종목 차트 데이터 수집"""
    try:
        start = (datetime.now() - timedelta(days=1100)).strftime('%Y-%m-%d')
        df = fdr.DataReader(ticker, start)
        if df is None or len(df) < 10:
            return (ticker, None)

        df = df.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'])
        if len(df) < 10:
            return (ticker, None)

        # 벡터화 변환
        result = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        result.columns = ['open', 'high', 'low', 'close', 'volume']
        result = result.astype(int)
        result.insert(0, 'date', df.index.strftime('%Y-%m-%d'))
        return (ticker, result.to_dict(orient='records'))
    except Exception as e:
        print(f"  경고: {ticker} 차트 수집 실패: {e}")
        return (ticker, None)


def main():
    print("=" * 60)
    print("chart_data.json 패치: 누락 종목 추가")
    print(f"실행 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # 1. 기존 chart_data.json 로드
    chart_data = {}
    if os.path.exists(CHART_DATA_FILE):
        with open(CHART_DATA_FILE, 'r', encoding='utf-8') as f:
            chart_data = json.load(f)
        print(f"기존 chart_data.json: {len(chart_data)}개 종목")
    else:
        print("chart_data.json 없음 — 새로 생성")

    # 2. 모든 퀀트/대가 JSON에서 종목 수집
    all_tickers = set()
    for filename in TARGET_FILES:
        filepath = os.path.join(DATA_PATH, filename)
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                for item in data.get('data', []):
                    ticker = item.get('ticker')
                    if ticker:
                        all_tickers.add(ticker)
            except Exception:
                pass

    print(f"퀀트/대가 스크리너 종목 수: {len(all_tickers)}개")

    # 3. 누락 종목 확인
    missing = sorted(all_tickers - set(chart_data.keys()))
    if not missing:
        print("누락 종목 없음 — 패치 불필요")
        return

    print(f"누락 종목: {len(missing)}개 → 차트 데이터 수집 시작")

    # 4. 병렬 수집
    added = 0
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(fetch_chart_data, t): t for t in missing}
        for future in as_completed(futures):
            ticker, data = future.result()
            if data:
                chart_data[ticker] = data
                added += 1
                print(f"  + {ticker} ({len(data)}일)")

    # 5. 저장
    if added > 0:
        with open(CHART_DATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(chart_data, f, ensure_ascii=False)
        print(f"\nchart_data.json 패치 완료: {added}개 종목 추가 (총 {len(chart_data)}개)")
    else:
        print("\n추가된 종목 없음")


if __name__ == '__main__':
    main()
