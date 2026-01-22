"""
차트 데이터 생성 스크립트
FinanceDataReader를 사용하여 OHLCV 데이터를 가져옵니다.
"""
import FinanceDataReader as fdr
import json
from datetime import datetime, timedelta
import os

# 데이터 저장 경로
data_path = r'C:\Users\hsh\Desktop\vibecoding\주식웹사이트\stock-screener-kr\web\public\data'

# 차트에 표시할 종목들 (모든 스크리너에 표시된 종목들)
tickers = [
    ('005930', '삼성전자'),
    ('000660', 'SK하이닉스'),
    ('035720', '카카오'),
    ('005380', '현대차'),
    ('373220', 'LG에너지솔루션'),
    ('006400', '삼성SDI'),
    ('035420', 'NAVER'),
    ('000270', '기아'),
    ('055550', '신한지주'),
    ('003550', 'LG'),
    ('034730', 'SK'),
    ('096770', 'SK이노베이션'),
    ('028260', '삼성물산'),
]

def get_ohlcv_data(ticker, name, days=200):
    """종목의 OHLCV 데이터를 가져옵니다."""
    try:
        # 200일치 데이터 조회
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days + 50)  # 여유 있게 조회

        df = fdr.DataReader(ticker, start_date.strftime('%Y-%m-%d'))

        if len(df) == 0:
            print(f"  {name}: 데이터 없음")
            return None

        # 최근 200일만 사용
        df = df.tail(days)

        # 차트 데이터 형식으로 변환
        chart_data = []
        for idx, row in df.iterrows():
            chart_data.append({
                'date': idx.strftime('%Y-%m-%d'),
                'open': int(row['Open']),
                'high': int(row['High']),
                'low': int(row['Low']),
                'close': int(row['Close']),
                'volume': int(row['Volume'])
            })

        print(f"  {name}: {len(chart_data)}일 데이터")
        return chart_data

    except Exception as e:
        print(f"  {name} 오류: {e}")
        return None

def main():
    print("차트 데이터 생성 중...")
    print("=" * 50)

    chart_data = {}

    for ticker, name in tickers:
        data = get_ohlcv_data(ticker, name)
        if data:
            chart_data[ticker] = data

    # JSON 파일로 저장
    output_path = os.path.join(data_path, 'chart_data.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(chart_data, f, ensure_ascii=False)

    print("=" * 50)
    print(f"차트 데이터 저장 완료: {output_path}")
    print(f"총 {len(chart_data)}개 종목")

if __name__ == '__main__':
    main()
