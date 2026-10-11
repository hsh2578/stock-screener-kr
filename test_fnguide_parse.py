"""
FnGuide 신규 API(wcomp.fnguide.com) 파싱 로직 점검 — 네트워크 없이 실행.

    python test_fnguide_parse.py
"""
from scripts.fnguide_data import (
    _dataset_to_df, _extract_item, _latest_value, _parse_header_metrics, _parse_snapshot_eps,
)

# 분기 손익: VAL1=가장 오래된 분기 … VAL4=최근 분기, VAL5=전년동기, VAL6=증감률
Q_INCOME = {
    'header': [
        {'YYMM': '2025/09', 'CD': 'VAL1'}, {'YYMM': '2025/12', 'CD': 'VAL2'},
        {'YYMM': '2026/03', 'CD': 'VAL3'}, {'YYMM': '2026/06 (최근분기)', 'CD': 'VAL4'},
        {'YYMM': '2025/06 (전년동기)', 'CD': 'VAL5'}, {'YYMM': '전년동기대비(%)', 'CD': 'VAL6'},
    ],
    'data': [
        {'NAME': '매출액(수익)', 'VAL1': '10', 'VAL2': '20', 'VAL3': '30', 'VAL4': '40', 'VAL5': '5', 'VAL6': '700'},
        {'NAME': '    기타', 'VAL1': '1', 'VAL2': '1', 'VAL3': '1', 'VAL4': '1', 'VAL5': '1', 'VAL6': '0'},
        {'NAME': '    기타', 'VAL1': '9', 'VAL2': '9', 'VAL3': '9', 'VAL4': '9', 'VAL5': '9', 'VAL6': '0'},
        {'NAME': '(지배주주지분)당기순이익', 'VAL1': '1', 'VAL2': None, 'VAL3': '3', 'VAL4': '4'},
    ],
}

# 연간 손익: 마지막 두 컬럼은 반기 누적·전년동기 → 연간 시계열에서 제외해야 함
Y_INCOME = {
    'header': [
        {'YYMM': '2023/12', 'CD': 'VAL1'}, {'YYMM': '2024/12', 'CD': 'VAL2'},
        {'YYMM': '2025/12', 'CD': 'VAL3'}, {'YYMM': '2026/06 (최근분기)', 'CD': 'VAL4'},
        {'YYMM': '2025/06 (전년동기)', 'CD': 'VAL5'},
    ],
    'data': [{'NAME': '매출액(수익)', 'VAL1': '100', 'VAL2': '200', 'VAL3': '300', 'VAL4': '150', 'VAL5': '90'}],
}

q = _dataset_to_df(Q_INCOME, quarterly=True)
assert list(q.columns) == ['2025/09', '2025/12', '2026/03', '2026/06'], q.columns
assert _extract_item(q, ['매출액']).to_dict() == {'2025/09': 10, '2025/12': 20, '2026/03': 30, '2026/06': 40}
assert _extract_item(q, ['기타']).iloc[0] == 1, '중복 계정명은 첫 행(상위 계정)을 써야 함'
assert _latest_value(_extract_item(q, ['매출액'])) == 40, '최신값은 가장 오른쪽(최근) 분기'
assert _latest_value(_extract_item(q, ['(지배주주지분)당기순이익'])) == 4

y = _dataset_to_df(Y_INCOME, quarterly=False)
assert list(y.columns) == ['2023/12', '2024/12', '2025/12'], y.columns

HEADER_HTML = '''
<button type="button" class="tip_in" id="h_per">PER</button> </li> <li>39.92 </li> </ul>
<button type="button" class="tip_in" id="h_12m">12M PER</button> </li> <li>- </li> </ul>
<button type="button" class="tip_in" id="h_pbr">PBR</button> </li> <li>4.09 </li> </ul>
<button type="button" class="tip_in" id="h_rate"><span class="ko">배당수익률</span></button> </li> <li>0.64 % </li>
'''
assert _parse_header_metrics(HEADER_HTML) == {'per': 39.92, 'pbr': 4.09, 'dividend_yield': 0.64}

SNAP_HTML = 'snpFinancial: {"header":[{"YYMM":"2024/12","EP_CHK":null,"CD":"VAL1"},' \
    '{"YYMM":"2025/12","EP_CHK":null,"CD":"VAL2"},{"YYMM":"2026/12","EP_CHK":"E","CD":"VAL3"},' \
    '{"YYMM":"2026/06","EP_CHK":" ","CD":"VAL4"}],' \
    '"data":[{"NAME":"EPS","VAL1":"4949.65","VAL2":"6563.57","VAL3":"45418.6","VAL4":"10717.8"}]},'
assert _parse_snapshot_eps(SNAP_HTML) == {'2024/12': 4949.65, '2025/12': 6563.57}, '추정치(E)·분기 제외'

from collect_quant_data import apply_history
new = {'annual': {'revenue': {'2023/12': 1, '2024/12': 2, '2025/12': 3}},
       'quarter': {'revenue': {f'2026/0{m}': m for m in (3, 6)}}, 'cashflow': {'cfo_quarter': {'2026/06': 7}}}
hist = {'annual': {'revenue': {'2020/12': 0, '2021/12': 9, '2022/12': 8, '2024/12': 999}, 'eps': {'2022/12': 5}},
        'quarter': {'revenue': {'2024/06': 1, '2024/09': 1, '2024/12': 1, '2025/03': 1, '2025/06': 1,
                                '2025/09': 1, '2025/12': 1}}}
apply_history(new, hist)
assert new['annual']['revenue'] == {'2021/12': 9, '2022/12': 8, '2023/12': 1, '2024/12': 2, '2025/12': 3}, '새 값 우선 + 최대 5년'
assert new['annual']['eps'] == {'2022/12': 5}
assert list(new['quarter']['revenue']) == ['2024/09', '2024/12', '2025/03', '2025/06', '2025/09', '2025/12',
                                           '2026/03', '2026/06'], '분기 최대 8개'
assert hist['annual']['revenue'] == new['annual']['revenue'] and hist['cashflow']['cfo_quarter'] == {'2026/06': 7}

print('OK: FnGuide 파싱 점검 통과')

from scripts.fnguide_data import get_cagr, get_ytd_growth, get_fcf
d = {'annual': {'revenue': {'2020/12': 50, '2021/12': 100, '2023/12': 150, '2025/12': 200},
                'eps': {'2024/12': -5, '2025/12': 10}},
     'ytd': {'revenue': {'cur': 120, 'prev': 100}, 'net_income': {'cur': 5, 'prev': -3}},
     'cfo_ttm': 100, 'capex_ttm': 30}
assert get_cagr(d, 'revenue', years=4) == round(((200 / 100) ** (1 / 4) - 1) * 100, 2), '최근 5개 연도, 실제 연도 간격'
assert get_cagr(d, 'eps') is None, '적자 끝값이면 CAGR 없음'
assert get_ytd_growth(d, 'revenue') == 20.0 and get_ytd_growth(d, 'net_income') is None
assert get_fcf(d) == 70 and get_fcf({'cfo_ttm': 1}) is None
print('OK: 재무 지표 헬퍼 점검 통과')

# --- 스크리너 정의 개편 (2026-10): 유형 투하자본, 이자부채 EV, PEGY, 대가 조건 ---
from scripts.fnguide_data import (
    _parse_snapshot_row, get_invested_capital, get_ev, get_long_term_debt_ratio, get_pegy,
)
SNAP_ROE = SNAP_HTML.replace('"data":[', '"data":[{"NAME":"ROE","VAL1":"9.03","VAL2":"10.85","VAL3":"53.5","VAL4":"31.4"},')
assert _parse_snapshot_row(SNAP_ROE, 'ROE') == {'2024/12': 9.03, '2025/12': 10.85}
assert _parse_snapshot_eps(SNAP_ROE) == {'2024/12': 4949.65, '2025/12': 6563.57}

bs = {'current_assets': 100, 'cash': 30, 'current_liabilities': 80, 'short_term_debt': 20,
      'current_portion_ltd': 10, 'ppe': 50, 'long_term_debt': 40, 'bonds': 0, 'total_equity': 200,
      'total_debt': 70, 'total_assets': 500}
assert get_invested_capital({'balance': bs}) == (70 - 50) + 50, 'NWC=(CA-현금)-(CL-단기차입-유동성장기) + 유형자산'
assert get_invested_capital({'balance': {**bs, 'cash': 90}}) == 50, '순운전자본 음수 → 0'
assert get_invested_capital({'balance': {k: v for k, v in bs.items() if k != 'ppe'}}) is None
assert get_invested_capital({'balance': {**bs, 'total_assets': 701}}) is None, 'IC 70 < 총자산 10% → None (금융성 자산 기업)'
assert get_invested_capital({'balance': {**bs, 'total_assets': 700}}) == 70, '정확히 10% 는 통과'
assert get_invested_capital({'balance': {k: v for k, v in bs.items() if k != 'total_assets'}}) is None
assert get_ev(1000, {'balance': bs}) == 1000 + 70 - 20, '여유자금 = 현금30 - 운전자본 부족분 max(0, 80-100+30)=10'
assert get_ev(1000, {'balance': {**bs, 'current_liabilities': 200}}) == 1000 + 70, '현금이 운전자본에 묶이면 여유자금 0'
assert get_ev(1000, {'balance': {'cash': 5}}) is None, '이자부채 계정 없음(금융업) → None'
assert get_long_term_debt_ratio({'balance': bs}) == 20.0
assert get_long_term_debt_ratio({'balance': {'total_equity': 100}}) == 0.0, '무차입은 0'

g = {'header': {'per': 10, 'dividend_yield': 2},
     'annual': {'eps': {'2021/12': 100, '2025/12': 100 * 1.08 ** 4}}}
assert abs(get_pegy(1, g) - 10 / (8 + 2)) < 1e-6
assert get_pegy(1, {**g, 'annual': {'eps': {'2021/12': 100, '2025/12': 900}}}) is None, '성장률 50% 초과 제외'
assert get_pegy(1, {**g, 'annual': {'eps': {'2021/12': 100, '2025/12': 90}}}) is None, '역성장 제외'

from guru_screeners import _graham_eps_growth, _per_3y, _buffett_quality, _net_debt_to_ebit
eps5 = {'2021/12': 100, '2022/12': 100, '2023/12': 90, '2024/12': 110, '2025/12': 110}
actual, required = _graham_eps_growth({'annual': {'eps': eps5}})
assert round(actual, 6) == 10.0 and round(required, 4) == round((1.03 ** 3 - 1) * 100, 4), '중심 간격 3년'
assert _graham_eps_growth({'annual': {'eps': {'2023/12': 1, '2024/12': 2, '2025/12': 3}}}) is None, 'EPS 4개 미만'
assert _per_3y(300, {'annual': {'net_income': {'2022/12': 5, '2023/12': 10, '2024/12': 20, '2025/12': 30}}}) == 15
assert _per_3y(300, {'annual': {'net_income': {'2024/12': 20, '2025/12': 30}}}) is None
ok = {'annual': {'roe': {'2023/12': 12, '2024/12': 16, '2025/12': 20}, 'eps': {'2023/12': 5, '2024/12': 4, '2025/12': 6}}}
assert _buffett_quality(ok) == 16
assert _buffett_quality({'annual': {**ok['annual'], 'roe': {'2023/12': 9, '2024/12': 20, '2025/12': 30}}}) is None, 'ROE 10% 이하 연도'
assert _buffett_quality({'annual': {**ok['annual'], 'eps': {'2023/12': 5, '2024/12': 4, '2025/12': 3}}}) is None, 'EPS 감소 2회'
assert _buffett_quality({'annual': {**ok['annual'], 'eps': {'2024/12': 4, '2025/12': 6}}}) is None, 'EPS 3년 미만'
assert _net_debt_to_ebit({'balance': bs, 'operating_income_ttm': 20}) == (70 - 30) / 20
assert _net_debt_to_ebit({'balance': {**bs, 'cash': 100}, 'operating_income_ttm': 20}) < 0, '순현금은 음수(통과)'
assert _net_debt_to_ebit({'balance': {}, 'operating_income_ttm': 20}) is None

from test_value_screener import annual_operating_margins, growth_ok
v = {'annual': {'revenue': {'2024/12': 100, '2025/12': 200}, 'operating_income': {'2024/12': 10, '2025/12': 30}},
     'ytd': {'revenue': {'cur': 110, 'prev': 100}}, 'revenue_ttm': 999, 'operating_income_ttm': 999}
assert annual_operating_margins(v) == [10.0, 15.0], 'TTM 미포함'
assert growth_ok(v, 'revenue', 3, 10) == (True, 100.0)
assert growth_ok({**v, 'ytd': {'revenue': {'cur': 90, 'prev': 100}}}, 'revenue', 3, 10)[0] is False, '누적 역성장'
assert growth_ok({**v, 'ytd': {}}, 'revenue', 3, 10)[0] is True, '누적 없음 → 최근 연간 YoY(+100%) 로 대체'
assert growth_ok({'annual': {'revenue': {'2022/12': 50, '2024/12': 200, '2025/12': 150}}}, 'revenue', 3, 10)[0] is False,     '누적 없음 + 최근 연간 역성장 → 탈락'
assert growth_ok(v, 'revenue', 3, 20, 50)[0] is False, '상한 초과'

from magic_formula import is_utility_stock
assert is_utility_stock('한국전력') and is_utility_stock('인천도시가스') and is_utility_stock('삼천리')
assert not is_utility_stock('삼천리자전거') and not is_utility_stock('LG에너지솔루션')

import pandas as pd
from multi_factor import calc_return_12m
close = pd.DataFrame({'Close': [100.0] * 231 + [150.0] + [999.0] * 21})  # 253일, 최근 21일은 제외
assert calc_return_12m(close) == 50.0, '12-1 모멘텀: Close[-22]/Close[-253]-1'
assert calc_return_12m(close.tail(252)) is None
from collect_quant_data import sanitize_annual
ann = {'net_income': {'2021/12': 360, '2022/12': 100, '2023/12': -5},
       'net_income_controlling': {'2021/12': 655, '2022/12': 100.5, '2023/12': -3},
       'eps': {'2021/12': 685, '2022/12': 50, '2023/12': -1}}
assert sanitize_annual(ann) == 1, '지배 > 전체×1.01 (둘 다 양수)만 제거'
assert '2021/12' not in ann['net_income_controlling'] and '2021/12' not in ann['eps']
assert ann['net_income']['2021/12'] == 360 and ann['eps']['2022/12'] == 50, '1% 이내·적자 연도는 유지'
new = {'annual': {'net_income': {'2025/12': 10}}, 'quarter': {}, 'cashflow': {}}
hist = {'annual': {'net_income': {'2021/12': 360}, 'net_income_controlling': {'2021/12': 655}, 'eps': {'2021/12': 685}}}
apply_history(new, hist)
assert new['annual']['eps'] == {} and hist['annual']['net_income_controlling'] == {}, 'apply_history 에서 양쪽 정리'
new = {'annual': {'net_income': {'2025/12': 564}, 'net_income_controlling': {'2025/12': 591}, 'eps': {'2025/12': 617}},
       'quarter': {}, 'cashflow': {}}
apply_history(new, {})
assert new['annual']['net_income_controlling'] == {'2025/12': 591}, '신규 API 연도는 비지배 적자로 성립 가능 → 유지'
# 결산월 직후: 누적 컬럼 없음 → 최근 2개 연간 YoY 로 대체
fy = {'annual': {'revenue': {'2024/12': 100, '2025/12': 130}, 'net_income': {'2024/12': -5, '2025/12': 10}}}
assert get_ytd_growth(fy, 'revenue') == 30.0 and get_ytd_growth(fy, 'net_income') is None
assert get_ytd_growth({'annual': {'revenue': {'2025/12': 1}}}, 'revenue') is None
assert get_ytd_growth({**fy, 'ytd': {'revenue': {'cur': 90, 'prev': 100}}}, 'revenue') == -10.0, '누적값이 있으면 누적 우선'

from collect_quant_data import seed_history, save_history, load_history
import collect_quant_data, json, os, tempfile
hist = {'annual': {'revenue': {'2024/12': 2}}}
seed_history(hist, {'annual': {'revenue': {'2021/12': 7, '2024/12': 999}, 'eps': {'2022/12': 3}}})
assert hist['annual'] == {'revenue': {'2021/12': 7, '2024/12': 2}, 'eps': {'2022/12': 3}}, '이전 캐시로 빈 연도만 채움'
seed_history(hist, None)  # 이전 캐시 없음 → 변화 없음
tmp = os.path.join(tempfile.mkdtemp(), 'h.json')
orig, collect_quant_data.HISTORY_FILE = collect_quant_data.HISTORY_FILE, tmp
save_history({'B': {'annual': {'x': {'2025/12': 1.234}}}, 'A': {}})
lines = open(tmp, encoding='utf-8').read().splitlines()
assert lines == ['{', '"A":{},', '"B":{"annual":{"x":{"2025/12":1.23}}}', '}'], lines
assert load_history() == {'A': {}, 'B': {'annual': {'x': {'2025/12': 1.23}}}}
collect_quant_data.HISTORY_FILE = orig
print('OK: 스크리너 정의 점검 통과')
