# CLAUDE.md — 주식 스크리너 프로젝트 가이드

> **새 세션에서 가장 먼저 읽어야 할 파일.** 이 문서만으로 프로젝트 전체를 파악할 수 있다.

## 1. 핵심 사실 (가장 중요)

| 항목 | 값 |
|------|-----|
| **배포 파일** | `주식_스크리너_전체.html` (단독 SPA, ~4800줄) |
| **배포 URL** | https://hsh2578.github.io/stock-screener-kr/ |
| **GitHub** | `hsh2578/stock-screener-kr` |
| **로컬 경로** | `C:\Users\hsh\Desktop\vibecoding\주식웹사이트\국내주식웹사이트\stock-screener-kr` |

### !! 주의 !!
- **`web/` 디렉토리는 미배포.** React/Vite 앱이지만 실제 서비스에는 사용되지 않음.
- **UI 수정은 반드시 `주식_스크리너_전체.html`에서** 해야 한다. `web/src/`를 수정해도 배포에 반영 안 됨.
- 배포 흐름: `주식_스크리너_전체.html` → deploy.yml이 `_site/index.html`로 복사 → GitHub Pages

---

## 2. 프로젝트 구조

```
stock-screener-kr/
├── 주식_스크리너_전체.html    ★ 배포되는 메인 파일 (CSS+HTML+JS 올인원 SPA)
├── run_screeners.py           ★ 메인 스크리너 엔진 (12개 기술적 스크리너 + ML 통합)
├── naver_finance.py             FnGuide 크롤러 (PER/PBR/성장률)
├── test_value_screener.py       저평가 우량주 스크리너 (TTM 재무)
├── ma60w_quality.py             60주선 우량주 스크리너
├── scheduler.py                 로컬 스케줄러
├── data/                      ★ JSON 출력 (screener → JSON → HTML이 읽음)
│   ├── box_range.json           박스권 횡보
│   ├── box_breakout.json        박스권 돌파 (거래량)
│   ├── box_breakout_simple.json 박스권 돌파 (단순)
│   ├── pullback.json            풀백
│   ├── volume_explosion.json    거래량 폭발
│   ├── volume_dry_up.json       거래량 급감
│   ├── new_high_52w.json        52주 신고가
│   ├── near_high_52w.json       52주 신고가 근접
│   ├── ma_convergence.json      이평선 수렴
│   ├── bottom_breakout.json     바닥 탈출
│   ├── fallen_rebound.json      낙폭과대 반등
│   ├── sector_stage.json        업종별 4단계
│   ├── value_stocks.json        저평가 우량주
│   ├── ma60w_quality.json       60주선 우량주
│   ├── financial_data.json      재무 데이터 캐시
│   └── chart_data.json          TradingView 차트 데이터 (37MB+)
├── scripts/screeners/           모듈화된 스크리너 구현체
├── ml/                          ML 모델 (3개: 박스권 돌파, 52주 신고가, 52주 근접)
│   ├── models/box_breakout/     박스권 돌파 ML (XGBoost + Ridge)
│   ├── models/52w_high/         52주 신고가 ML (LightGBM)
│   ├── models/near_high_52w/    52주 근접 ML (Lasso + LogisticRegression)
│   ├── src/                     학습 데이터 수집 + 모델 훈련 스크립트
│   └── data/                    학습 데이터 CSV
├── docs/                        프로젝트 문서
│   └── 스크리너_조건식_설명서.md 전체 스크리너 조건식/로직 설명
├── .github/workflows/
│   ├── deploy.yml               평일 4회 자동 배포 (OHLCV 캐시 사용)
│   ├── value-screener.yml       토요일 가치주 스크리너
│   └── ml-retrain.yml           월간 ML 재훈련 (3개 모델)
├── web/                         ⚠️ React 앱 (미배포, 참고용)
└── .cache/                      OHLCV pickle 캐시 (CI에서도 actions/cache로 유지)
```

---

## 3. `주식_스크리너_전체.html` 내부 구조

이 파일은 CSS, HTML, JS가 모두 합쳐진 단일 SPA이다.

### 영역별 라인 범위 (대략적, 수정 시 변동 가능)

| 라인 범위 | 영역 | 설명 |
|-----------|------|------|
| 1~16 | `<head>` | 메타, 폰트 로드 |
| 17~1515 | `<style>` | **전체 CSS** |
| 17~60 | CSS: 변수 | `:root` 색상/간격 변수 |
| 62~160 | CSS: 헤더/카드/배지 | 글로벌 헤더, 카드 시스템 |
| 167~306 | CSS: 홈페이지 | 홈, 스크리너 카드 그리드, 푸터 |
| 308~560 | CSS: 상세페이지 | 페이지 헤더, 데이터 테이블, 업종 분석 |
| 564~1515 | CSS: 유틸리티/반응형 | 정렬, 접근성, 토스트, 차트, 필터, 페이지네이션, 반응형 |
| 1517~2093 | `<body>` HTML | **정적 HTML 페이지들** |
| 1539~1554 | HTML: 홈 | `id="page-home"` |
| 1555~2090 | HTML: 상세페이지 ×14 | `id="page-box-range"` ~ `id="page-ma-convergence"` |
| 2094~2125 | `<script>` #1 | TradingView 라이브러리 + Lazy Load |
| 2127~4823 | `<script>` #2 | **전체 JavaScript** |
| 2130~2180 | JS: 상태 관리 | 글로벌 변수, 데이터 저장소 |
| 2183~2315 | JS: 데이터 로딩 | `loadAllData()`, 로딩/에러 UI |
| 2318~2436 | JS: 아이콘/스크리너 정의 | `ICONS` 객체, `screenerGroups` 배열 |
| 2438~2750 | JS: UI 유틸리티 | 모바일 카드, 차트, 페이지네이션 |
| 2757~3085 | JS: 정렬/차트 모달 | 데이터 정렬, TradingView 차트 |
| 3086~3143 | JS: 홈 렌더링 | `renderHome()` |
| 3145~3627 | JS: 테이블 렌더링 | 공통 테이블 렌더러, 행 생성 |
| 3628~3510 | JS: 개별 스크리너 렌더러 | `renderBoxRange()` ~ `renderMaConvergence()` |
| 4511~4558 | JS: 페이지 전환 | `showPage()` |
| 4560~4608 | JS: 앱 초기화 | `initApp()` |

### 정적 HTML 페이지 ID 목록

```
page-home, page-box-range, page-box-breakout, page-box-breakout-simple,
page-pullback, page-volume-dry-up, page-volume-explosion, page-new-high-52w,
page-near-high-52w, page-bottom-breakout, page-fallen-rebound, page-sector-stage,
page-value-stocks, page-ma60w-quality, page-ma-convergence
```

### JS 핵심 객체

```javascript
// 아이콘 (SVG 문자열)
const ICONS = { package, rocket, trendUp, target, volDown, zap, mountain, ... };

// 스크리너 그룹 정의 (홈페이지 카드 렌더링용)
const screenerGroups = [
    { title: '박스권 패턴', icon: ICONS.package, screeners: [...] },
    { title: '거래량 분석', icon: ICONS.barChart, screeners: [...] },
    // ... 6개 그룹, 14개 스크리너
];
```

---

## 4. 데이터 흐름

```
[FinanceDataReader]     [FnGuide 크롤링]
     │ (OHLCV)              │ (재무)
     ▼                      ▼
 run_screeners.py      naver_finance.py
     │                      │
     ▼                      ▼
 data/*.json ◄──────── financial_data.json
     │
     ▼
 주식_스크리너_전체.html (fetch → JSON 파싱 → 테이블 렌더링)
     │
     ▼
 GitHub Pages (deploy.yml: cp HTML → _site/index.html)
```

### JSON 메타 키 주의
- `run_screeners.py` 출력: **camelCase** (`lastUpdated`, `totalCount`)
- 개별 스크립트 출력: **snake_case** (`updated_at`, `total_count`)
- HTML에서 **둘 다 체크**하도록 구현됨

---

## 5. 자동화 스케줄

| 워크플로우 | 실행 시점 | 실행 내용 |
|-----------|----------|----------|
| `deploy.yml` | 평일 12/14/16/18시 KST | `run_screeners.py` + 배포 |
| `deploy.yml` (18시) | 평일 18시에만 추가 | `ma60w_quality.py` |
| `value-screener.yml` | 토요일 12시 KST | `test_value_screener.py` (전체 재무 크롤링) |
| `ml-retrain.yml` | 월간 | ML 모델 재훈련 |

---

## 6. 명령어

```bash
# 로컬에서 HTML 보기
cd stock-screener-kr
python -m http.server 8080
# → http://localhost:8080/주식_스크리너_전체.html

# 스크리너 전체 실행
python run_screeners.py

# 캐시 무시하고 새로 수집
python run_screeners.py --fresh

# 개별 실행
python test_value_screener.py   # 저평가 우량주
python ma60w_quality.py         # 60주선 우량주

# GitHub CLI
"C:\Program Files\GitHub CLI\gh.exe" ...  # Windows에서는 전체 경로 또는 path 설정 필요
```

---

## 7. 수정 패턴 가이드

### 새 스크리너 추가 시
1. `scripts/screeners/`에 새 스크리너 Python 파일 작성
2. `run_screeners.py`에 통합 또는 별도 실행 설정
3. `주식_스크리너_전체.html`에서:
   - 정적 HTML에 `<div id="page-새스크리너" class="page">` 추가
   - `ICONS`에 아이콘 추가
   - `screenerGroups`에 항목 추가
   - `renderXxx()` 함수 작성
   - `pageRenderMap`에 매핑 추가
   - `showPage()`의 날짜/카운트 업데이트에 매핑 추가

### UI 스타일 변경 시
- CSS는 `<style>` 내부 (라인 17~1515)
- `:root` 변수로 색상/간격 관리 → 변수만 변경하면 전체 테마 변경 가능
- 라이트 테마 사용 중 (밝은 배경, 파란 액센트)

### 아이콘 변경 시
- `ICONS` JS 객체에 SVG 문자열로 정의됨
- 정적 HTML 페이지 헤더에도 동일한 SVG가 인라인으로 들어감
- **두 곳 모두** 변경해야 함 (JS + HTML)
- 현재: Filled SVG 스타일, `fill="currentColor"`, `width="1em" height="1em"`

---

## 8. ML 모델 정보

| 스크리너 | 회귀 모델 | 분류 모델 | 피처 수 | 예측 내용 |
|----------|----------|----------|---------|----------|
| 박스권 돌파 | Ridge | XGBoost | 15개 | 20일 수익률 / 10%+ 상승 확률 |
| 52주 신고가 | LGBMRegressor | LGBMClassifier | 14개 | 20일 수익률 / 15%+ 상승 확률 |
| 52주 근접 | Lasso | LogisticRegression | 13개 | 20일 수익률 / 10일 내 돌파 확률 |

- AI점수 공식: `확률 × 0.7 + min(100, max(0, 예상수익 × 3)) × 0.3`
- 상세 조건식: `docs/스크리너_조건식_설명서.md` 참조

---

## 9. 알려진 제약사항

- FDR 크롤링 속도: 로컬 ~3.5분 (캐시), CI ~5분 (증분 캐시)
- `fdr.DataReader()`에 타임아웃 없음 → 간헐적 멈춤 가능
- chart_data.json이 37MB+로 매우 큼 (첫 로딩 시 지연)
- `pd.DataFrame._append`는 deprecated → `pd.concat()` 사용
- GitHub Actions 스케줄은 최대 2시간 지연 가능
- CI 캐시: `actions/cache`로 `.cache/` 유지 (당일 pkl 삭제 → 증분 업데이트)

---

## 10. 스크리너 목록 (14개)

| ID | 한글명 | JSON 파일 | 핵심 조건 | ML |
|----|--------|-----------|----------|----|
| box-range | 박스권 횡보 | box_range.json | 60일+ 횡보, ATR×6, 지지/저항 터치 | - |
| box-breakout | 박스권 돌파 (거래량) | box_breakout.json | 저항선 돌파 + 거래량 2배 + 150MA 위 | O |
| box-breakout-simple | 박스권 돌파 (단순) | box_breakout_simple.json | 저항선 돌파, 10일 이내 | O |
| pullback | 풀백 | pullback.json | 돌파 후 되돌림 ±5%, 거래량 -50% | - |
| volume-explosion | 거래량 폭발 | volume_explosion.json | 40일 평균 6배 + 6% 양봉 | - |
| volume-dry-up | 거래량 급감 | volume_dry_up.json | 8%+4배 급등 후 거래량 60% 감소 | - |
| new-high-52w | 52주 신고가 | new_high_52w.json | 52주 신고가 + 거래량 1.5배, 8거래일 | O |
| near-high-52w | 52주 신고가 근접 | near_high_52w.json | 52주 고가 5% 이내 근접 (미돌파) | O |
| ma-convergence | 이평선 수렴 | ma_convergence.json | MA 수렴 + 정배열 + 신고가 | - |
| bottom-breakout | 바닥 탈출 | bottom_breakout.json | 150MA 크로스오버 + 점수제 6점+ | - |
| fallen-rebound | 낙폭과대 반등 | fallen_rebound.json | 52주 고가 -40% + 바닥 상승 | - |
| sector-stage | 업종별 4단계 | sector_stage.json | 업종 추세 분석 (와인스테인 4단계) | - |
| value-stocks | 저평가 우량주 | value_stocks.json | PER/PBR/배당 기반 (TTM) | - |
| ma60w-quality | 60주선 우량주 | ma60w_quality.json | 60주 EMA 지지 + 영업이익률/성장률 | - |

---

## 11. 언어 및 스타일

- 사용자 대면 텍스트: **한국어**
- 기술 용어: 영어 허용
- UI: 라이트 테마, Pretendard 폰트, 네이버 금융 스타일
- 이모지: 사용자가 요청하지 않으면 사용하지 않음
