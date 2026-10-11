"""_weinstein_stage 판별 규칙 self-check: python test_sector_stage.py"""
from run_screeners import _weinstein_stage

assert _weinstein_stage(110, 100, 2.0, 1.5)[0] == 2    # 상승 이평 위
assert _weinstein_stage(90, 100, -2.0, -1.5)[0] == 4   # 하락 이평 아래
assert _weinstein_stage(110, 100, -2.0, -1.5)[0] == 1  # 하락 이평 위 반등 → 4단계 아님
assert _weinstein_stage(105, 100, 0.3, 2.0)[0] == 3    # 상승하던 이평이 평평해짐
assert _weinstein_stage(95, 100, 2.0, 2.0)[0] == 3     # 상승 이평 아래로 이탈
assert _weinstein_stage(101, 100, 0.2, -0.5)[0] == 1   # 평평한 이평 횡보
assert _weinstein_stage(101, 100, 0.2, None)[0] == 1
print("ok")
