# CLAUDE.md — 주식 스크리너 프로젝트 가이드

> **새 세션에서 가장 먼저 읽어야 할 파일.** 이 문서만으로 프로젝트 전체를 파악할 수 있다.

## 1. 핵심 사실 (가장 중요)

| 항목 | 값 |
|------|-----|
| **배포 파일** | `주식_스크리너_전체.html` (단독 SPA, ~5600줄) |
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
├── collect_quant_data.py        FnGuide 1회 수집 → .cache/fnguide_*.pkl
├── magic_formula.py             마법공식 스크리너 (이익수익률+ROC)
├── multi_factor.py              멀티팩터 스크리너 (Q+V+M)
├── guru_screeners.py            대가 스크리너 4종 (Buffett/Ackman/Lynch/Graham)
├── naver_finance.py             (레거시) FnGuide 크롤러 — 참고용
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
│   ├── magic_formula.json       마법공식
│   ├── multi_factor.json        멀티팩터
│   ├── buffett.json             워렌 버핏 전략
│   ├── ackman.json              빌 애크먼 전략
│   ├── lynch.json               피터 린치 전략
│   ├── graham.json              벤저민 그레이엄 전략
│   ├── financial_data.json      재무 데이터 캐시
│   └── chart_data.json          TradingView 차트 데이터 (37MB+)
├── scripts/
│   ├── krx_data.py              KRX OTP 종목 마스터 + WiseIndex 섹터 + 배당수익률
│   ├── fnguide_data.py          FnGuide read_html 재무제표 수집 + 헬퍼 함수
│   ├── screeners/               모듈화된 스크리너 구현체
│   └── utils/                   유틸리티
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
│   ├── value-screener.yml       평일 20시 퀀트+대가 스크리너
│   └── ml-retrain.yml           월간 ML 재훈련 (3개 모델)
├── web/                         ⚠️ React 앱 (미배포, 참고용)
└── .cache/                      OHLCV pickle 캐시 (CI에서도 actions/cache로 유지)
```

---

## 3. `주식_스크리너_전체.html` 내부 구조

이 파일은 CSS, HTML, JS가 모두 합쳐진 단일 SPA이다. (~5600줄)

### 주요 기능
- **해시 라우팅**: `#screener-id` URL로 직접 접근/뒤로가기/북마크 지원 (`history.pushState` + `popstate`)
- **데이터 경로 캐싱**: `_resolvedDataPath`로 1회 탐색 후 재사용
- **Lazy Rendering**: 페이지 진입 시에만 렌더링 (`pageRenderMap` + `renderedPages`)
- **formatNumber**: 캐시된 `Intl.NumberFormat('ko-KR')` 인스턴스 사용

### 정적 HTML 페이지 ID 목록

```
page-home, page-box-range, page-box-breakout, page-box-breakout-simple,
page-pullback, page-volume-dry-up, page-volume-explosion, page-new-high-52w,
page-near-high-52w, page-bottom-breakout, page-fallen-rebound, page-sector-stage,
page-value-stocks, page-ma60w-quality, page-ma-convergence,
page-magic-formula, page-multi-factor,
page-buffett, page-ackman, page-lynch, page-graham
```

### JS 핵심 객체

```javascript
// 아이콘 (SVG 문자열)
const ICONS = { package, rocket, trendUp, target, volDown, zap, mountain,
                gem, buffett, ackman, lynch, graham, ... };

// 스크리너 그룹 정의 (홈페이지 카드 렌더링용)
const screenerGroups = [
    { title: '박스권 패턴', ... },
    { title: '거래량 분석', ... },
    { title: '추세/모멘텀', ... },
    { title: '저평가 우량주', ... },
    { title: '퀀트 전략', ... },
    { title: '투자 대가', ... },   // Buffett, Ackman, Lynch, Graham
];
```

---

## 4. 데이터 흐름

```
[deploy.yml 12/14/16/18시]              [value-screener.yml 20시]
 run_screeners.py                        collect_quant_data.py
     │ (FDR OHLCV)                           │ (FnGuide 1회 수집)
     ▼                                       ▼
 .cache/stock_data_*.pkl              .cache/fnguide_*.pkl
 data/*.json (12개 기술 스크리너)             │
     │                                       ├→ test_value_screener.py → data/value_stocks.json
     │                                       ├→ magic_formula.py → data/magic_formula.json
     │                                       ├→ multi_factor.py → data/multi_factor.json
     │                                       ├→ ma60w_quality.py → data/ma60w_quality.json
     │                                       └→ guru_screeners.py → data/{buffett,ackman,lynch,graham}.json
     ▼                                                                      │
 주식_스크리너_전체.html (fetch → JSON 파싱 → 테이블 렌더링) ◄─────────────┘
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
| `deploy.yml` | 평일 10/12/14/16/18시 KST | `run_screeners.py` + 배포 |
| `value-screener.yml` | 평일 20시 KST | `collect_quant_data.py` → 가치주 + 마법공식 + 멀티팩터 + 60주선 + 대가 4종 |
| `ml-retrain.yml` | 월간 | ML 모델 재훈련 |

### value-screener.yml 실행 순서
```
collect_quant_data.py → .cache/fnguide_*.pkl (~6min)
test_value_screener.py → 캐시 읽기 (~30sec)
magic_formula.py → 캐시 읽기 (~30sec)
multi_factor.py → 캐시 + OHLCV 캐시 (~30sec)
ma60w_quality.py → 캐시 + OHLCV 캐시 (~30sec)
guru_screeners.py → 캐시 + 배당 수집 (~1min)
```

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
python magic_formula.py         # 마법공식
python multi_factor.py          # 멀티팩터
python guru_screeners.py        # 대가 4종

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
   - `DATA_FILES`에 JSON 매핑 추가
   - `renderXxx()` 함수 작성
   - `pageRenderMap`에 매핑 추가
   - `showPage()`의 날짜/카운트 업데이트에 매핑 추가
4. `run_screeners.py`의 `extra_files` 리스트에 JSON 추가 (chart_data 포함용)

### UI 스타일 변경 시
- CSS는 `<style>` 내부
- `:root` 변수로 색상/간격 관리 → 변수만 변경하면 전체 테마 변경 가능
- 색상 변수: `--color-accent` (파란 액센트), `--color-primary`는 미정의 — 사용 금지
- 라이트 테마 사용 중 (밝은 배경, 파란 액센트)

### 아이콘 변경 시
- `ICONS` JS 객체에 SVG 문자열로 정의됨
- 정적 HTML 페이지 헤더에도 동일한 SVG가 인라인으로 들어감
- **두 곳 모두** 변경해야 함 (JS + HTML)
- 현재: Filled SVG 스타일, `fill="currentColor"`, `width="1em" height="1em"`

### renderCache 사용 시
- `renderCache`는 **Map** 객체 — `renderCache.delete('key')` 사용
- `renderCache['key'] = null` 은 **잘못된 사용** (Map이 아닌 객체 프로퍼티 설정)

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

- **FnGuide 연간 데이터**: SVD_Finance.asp 4년 + SVD_Main.asp table[11] 1년 보충 → 최대 5년치 (2026-03-01)
  - 미발표 종목(연간 최신 컬럼이 `/12`가 아닌 경우) → 분기 누적값 자동 제거
  - `_fetch_tables_main()` + `_parse_main_annual()` 함수 추가, ThreadPoolExecutor(max_workers=2) 동시 요청
  - 기존 4년 값 덮어쓰기 방지 (min_existing 체크), Main 파싱 실패 시 자동 fallback
- FDR 크롤링 속도: 로컬 ~3.5분 (캐시), CI ~5분 (증분 캐시)
- `fdr.DataReader()`에 타임아웃 없음 → 간헐적 멈춤 가능
- chart_data.json이 37MB+로 매우 큼 (3초 후 백그라운드 프리로드로 개선)
- `pd.DataFrame._append`는 deprecated → `pd.concat()` 사용
- GitHub Actions 스케줄은 최대 2시간 지연 가능
- CI 캐시: `actions/cache`로 `.cache/` 유지 (당일 pkl 삭제 → 증분 업데이트)
- B/S 항목: `balance` = 최신값(EV/PBR용), `balance_avg` = 4분기 평균(ROE/GPA/IC용)
- EBIT = 영업이익 (한국 회계 기준, `영업이익 - 법인세`는 이중 차감 오류)
- 멀티팩터: 금융주 제외 안 함, Winsorize `_orig` 컬럼에 원본값 보존
- 배당수익률: `krx_data.py`에서 pykrx `get_market_fundamental()` 수집

---

## 10. 스크리너 목록 (21개)

### 기술적 분석 (13개)

| ID | 한글명 | JSON 파일 | 핵심 조건 | ML |
|----|--------|-----------|----------|----|
| box-range | 박스권 횡보 | box_range.json | 60일+ 횡보, ATR×6, 지지/저항 터치 | - |
| box-breakout | 박스권 돌파 (거래량) | box_breakout.json | 저항선 돌파 + 거래량 2배 + 150MA 위 | O |
| box-breakout-simple | 박스권 돌파 (단순) | box_breakout_simple.json | 저항선 돌파, 10일 이내 | O |
| pullback | 풀백 | pullback.json | 돌파 후 되돌림 ±5%, 거래량 -50% | - |
| volume-explosion | 거래량 폭발 | volume_explosion.json | 40일 평균 2배 + 12% 양봉 | - |
| price-rise | 상승률 | price_rise.json | 당일 등락률 상위 20종목 | - |
| volume-dry-up | 거래량 급감 | volume_dry_up.json | 8%+4배 급등 후 거래량 60% 감소 | - |
| new-high-52w | 52주 신고가 | new_high_52w.json | 52주 신고가 + 거래량 1.5배, 8거래일 | O |
| near-high-52w | 52주 신고가 근접 | near_high_52w.json | 52주 고가 5% 이내 근접 (미돌파) | O |
| ma-convergence | 이평선 수렴 | ma_convergence.json | MA 수렴 + 정배열 + 신고가 | - |
| bottom-breakout | 바닥 탈출 | bottom_breakout.json | 150MA 크로스오버 + 점수제 6점+ | - |
| fallen-rebound | 낙폭과대 반등 | fallen_rebound.json | 52주 고가 -40% + 바닥 상승 | - |
| sector-stage | 업종별 4단계 | sector_stage.json | 업종 추세 분석 (와인스테인 4단계) | - |

### 가치투자 (2개)

**저평가 우량주** (`value-stocks`, `value_stocks.json`) — 6개 조건 전부 충족
1. 3 < PER < 30
2. 매출 3년 CAGR > 10% **그리고** 최근 누적 전년동기 대비 증가 (`get_ytd_growth > 0`)
3. 연간 영업이익률 평균 > 10% (최대 5년, TTM 미포함 — `annual_operating_margins`)
4. 영업이익 5년 CAGR > 10% + 최근 누적 YoY > 0
5. EPS 5년 CAGR > 10% (누적 EPS 없음 → YoY 조건 없음)
6. 20% < 순이익 5년 CAGR < 50% + 최근 누적 YoY > 0
- CAGR = `get_cagr` (가용 연간 최대 N년 양 끝값, 끝값 ≤ 0 이면 None → 탈락)

**60주선 우량주** (`ma60w-quality`, `ma60w_quality.json`) — 코스피 상위 200 대상
1. 60주 EMA(300일) 이격도 0%~10% (지지 구간)
2. 종가 >= 60주 EMA (이중 확인)
3. 영업이익률 TTM(없으면 최근 연간) >= 10%
4. 영업이익 3년 CAGR >= 10% + 최근 누적 YoY > 0

### 퀀트 전략 (2개)

**마법공식** (`magic-formula`, `magic_formula.json`) — 조엘 그린블라트
- EBIT/EV 순위 + EBIT/IC 순위 합산 → 낮을수록 우량저평가
- EBIT = 영업이익(TTM), EV = 시총 + 이자부채(단기차입금+유동성장기부채+장기차입금+사채) - 여유자금
- 여유자금 = max(0, 현금 - max(0, 유동부채 - 유동자산 + 현금)) (운전자본에 묶인 현금은 빼지 않음 — 결제대행·광고대행 쏠림 방지)
- IC = 유형 투하자본 = max(0, (유동자산-현금) - (유동부채-단기차입금-유동성장기부채)) + 유형자산 (최신 분기)
- 시총 1000억+, 보통주, 스팩/리츠 제외, 금융주·유틸리티 제외 (업종이 전부 '기타'라 종목명 키워드로 판별)
- 유형 IC < 총자산 × `MIN_IC_TO_ASSETS`(0.1, `fnguide_data.py` 공통 상수, 애크먼과 공유) 제외 — 지주사·결제대행 등

**멀티팩터** (`multi-factor`, `multi_factor.json`) — Q+V+M 글로벌 Z-Score
- Quality: ROE, GPA, CFO/총자산
- Value: PER, PBR, PSR, PCR, 배당수익률
- Momentum: 12-1 수익률 `Close[-22]/Close[-253]-1` (Fama-French Prior 2-12, 253거래일 미만 None), K-Ratio
- Winsorize 1%-99% (원본은 `_orig` 컬럼 보존), 유효 지표 최소 5/10개, 팩터그룹별 1개 이상 필수
- 금융주 제외 안 함, 섹터 중립화 아님 (글로벌 Z-Score)

### 투자 대가 (4개) — 원전 기준을 5년 데이터로 축소

**워렌 버핏** (`buffett`, `buffett.json`) — Validea Patient Investor / Buffettology (연간 ROE 평균 내림차순)
- 연간 ROE(FnGuide Snapshot) 전부 > 10% & 평균 >= 15% (3년 이상)
- 연간 EPS 전부 > 0, 전년 대비 감소 최대 1회 (3년 이상)
- 장기부채(장기차입금+사채) <= 순이익TTM × 5 (무차입 통과), FCF > 0
- 이익수익률(1/PER)은 표시만 (CI에 국채금리 없음). PBR/유동비율/부채비율은 표시만

**빌 애크먼** (`ackman`, `ackman.json`) — **애크먼 원칙에서 착안한 자체 규칙** (FCF 수익률 내림차순, 상위 30)
- 유형 IC >= 총자산 × `MIN_IC_TO_ASSETS`(10%, 마법공식과 공유) (자산 경량 기업의 ROIC 과대 방지)
- ROIC(유형 IC) > 15%, ROE > 12%, PER < 15, FCF > 0, 순차입금/EBIT < 2 (순현금 통과)
- `data` = FCF/시총 상위 30, `meta.totalCount` = 전체 통과 수
- 이자부채 계정이 없는 금융주는 자동 제외

**피터 린치** (`lynch`, `lynch.json`) — AAII Lynch 스크린 PEGY (PEGY 오름차순)
- 5 <= PER < 25 (PER 5 미만 = 일회성 이익 신호), 영업이익TTM > 0
- PEGY = PER / (EPS 5년 CAGR + 배당수익률) <= 0.5, 0 < CAGR <= 50
- 이자부채 / 자본총계 < 80%
- 재고 추이 조건 없음 (분기 재고 이력 미저장). JSON `peg` 키 = PEGY 값 (HTML 호환)

**벤저민 그레이엄** (`graham`, `graham.json`) — 『현명한 투자자』 14장 방어적 투자자 (PER×PBR 오름차순)
- 매출TTM > 1000억, 유동비율 >= 2, 장기부채 <= 순유동자산
- 가용 연간 전 기간 흑자 (순이익+영업이익), 배당수익률 > 0
- 최근 2년 EPS 평균 >= 최초 2년 EPS 평균 × 1.03^(최근연도-최초연도-1) (연간 EPS 4개 이상, 10년 ⅓ → 연 3%)
- PER(3년 평균 지배순이익) <= 15, PBR <= 1.5 또는 PER×PBR <= 22.5
- JSON `per` = 3년 평균 이익 기준 PER, `eps_growth` = 실제 증가율(%)

---

## 11. fnguide_data.py 헬퍼 함수

| 함수 | 계산식 | 사용처 |
|------|--------|--------|
| `get_per_from_data` | 헤더 PER, 없으면 시총 / 순이익TTM | 전체 |
| `get_pbr_from_data` | 헤더 PBR, 없으면 시총 / 자본총계 | 전체 |
| `get_roe` | 순이익TTM / 자본총계(4Q평균) × 100 | 멀티팩터, 애크먼 |
| `get_ebit` | 영업이익TTM | 마법공식, 애크먼 |
| `get_ev` | 시총 + 이자부채 - 여유자금 (이자부채 계정 없으면 None) | 마법공식 |
| `get_invested_capital` | 순운전자본(현금·단기차입 제외, 0 하한) + 유형자산 (최신 분기) | 마법공식, 애크먼 |
| `get_current_ratio` | 유동자산 / 유동부채 | 그레이엄 |
| `get_debt_ratio` | (유동+비유동부채) / 자본총계 × 100 | 표시용 |
| `get_long_term_debt_ratio` | (장기차입금+사채) / 자본총계 × 100, 무차입 0.0 | - |
| `get_roa` | 순이익TTM / 총자산(4Q평균) × 100 | 린치(표시) |
| `get_roic` | EBIT / 유형 IC × 100 | 애크먼 |
| `get_fcf` | CFO(TTM) - CAPEX(TTM, 유형+무형자산 증가) | 버핏, 애크먼 |
| `get_cagr` | 연간 (최근/N년 전)^(1/N)-1, 끝값 ≤ 0 이면 None | 가치·60주선·린치·버핏 |
| `get_ytd_growth` | 최근 누적 vs 전년동기 누적 증가율 | 가치·60주선 |
| `get_pegy` | PER / (EPS 5년 CAGR + 배당수익률), 0<g<=50 | 린치 |
| `get_inventory_ratio` | 재고자산(4Q평균) / 매출TTM × 100 | - |
| `get_net_current_assets` | 유동자산 - 유동부채 | 그레이엄 |
| `get_annual_growth_rates` | 연간 YoY 성장률만 | - |
| `get_growth_rates_with_ttm` | TTM + 연간 YoY 성장률 시리즈 | financial_data.json |

- `balance['total_debt']` = 단기차입금 + 유동성장기부채 + 장기차입금 + 사채. 일반 기업 양식(유동자산 있음)이면 무차입도 0으로 생성, 금융업은 미생성
- `annual['roe']`, `annual['eps']` = Snapshot Financial Highlight 확정 연간값 (추정치 제외). `SCHEMA_VERSION = 3`
- `collect_quant_data.sanitize_annual`: 지배순이익 > 순이익×1.01(둘 다 양수)인 연도는 연결/별도 혼재로 보고 지배순이익·EPS 삭제 (`apply_history` 에서 매번 적용)

---

## 12. 언어 및 스타일

- 사용자 대면 텍스트: **한국어**
- 기술 용어: 영어 허용
- UI: 라이트 테마, Pretendard 폰트, 네이버 금융 스타일
- 이모지: 사용자가 요청하지 않으면 사용하지 않음

---

## 13. 주요 변경 이력

### 2026-03-01

**FnGuide 5년 연간 데이터 확장** (`scripts/fnguide_data.py`)
- `_fetch_tables_main(code)` 추가: SVD_Main.asp GET 요청, table[11] Financial Highlight 연간 파싱
- `_parse_main_annual(table)` 추가: MultiIndex 컬럼 → YYYY/MM 날짜, MAIN_ANNUAL_ITEMS 매핑
- `MAIN_ANNUAL_ITEMS` 항목: 매출액, 영업이익, 당기순이익, 지배주주순이익, EPS
- `get_financial_data()` 수정: ThreadPoolExecutor(max_workers=2)로 SVD_Finance.asp + SVD_Main.asp 동시 요청
- 미발표 종목 처리: 연간 테이블 최신 컬럼이 `/12`로 끝나지 않으면 제거 (분기 누적값 방지)
- 보충 로직: 기존 키(revenue 등)는 min_existing 이전 연도만, 신규 키(eps)는 전체 `/12` 연도

**EPS 성장률 교체** (`test_value_screener.py`)
- 조건 5: 순이익 성장률 대신 `get_annual_growth_rates(fin_data, 'eps', years=5)` 사용
- SVD_Main.asp의 실제 EPS 데이터로 5년 평균 계산

**FnGuide 동시 요청 부하 감소** (`collect_quant_data.py`)
- `MAX_WORKERS = 5` → `MAX_WORKERS = 3` (최대 concurrent = 3×2 = 6개)

**HTML 버그/성능/UX 5건 수정** (`주식_스크리너_전체.html`)
- BUG-3: `showPage()` 잘못된 hash URL → 홈 fallback (null 체크 추가)
- BUG-5: `renderBoxRange` 필터 적용 시 카운트 표시 일관성 수정
- PERF-2: `updateTableRowsIncremental` 60줄 복잡한 DOM 업데이트 → `tbody.innerHTML` 3줄로 단순화
- UX-2: `loadFinancialData` 실패 시 `financialDataLoadFailed` 플래그로 반복 재시도 방지
- UX-3: 데스크톱 카드 뷰 렌더링 조건 수정 (`isCardViewActive` 체크)
