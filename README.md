# 국내 주식 종목 발굴 스크리너

기술적 분석 기반 국내 주식 스크리닝 도구입니다.
GitHub Actions와 GitHub Pages를 이용해 100% 무료로 운영됩니다.

**[사이트 바로가기](https://hsh2578.github.io/stock-screener-kr/)**

## 스크리너 목록

### 박스권 패턴
| 스크리너 | 설명 |
|----------|------|
| **박스권 횡보** | 60거래일 이상 횡보 중인 종목 (변동폭 25% 이내) |
| **박스권 돌파 (거래량)** | 저항선 돌파 + 거래량 2배 + 150일선 위 |
| **박스권 돌파 (단순)** | 저항선 돌파 후 10일 이내 |
| **풀백** | 돌파 후 저항선으로 되돌림 (거래량 50% 감소) |

### 거래량 분석
| 스크리너 | 설명 |
|----------|------|
| **거래량 폭발** | 40일 평균 대비 6배 거래량 + 6% 이상 상승 |
| **거래량 급감** | 8% 급등 후 거래량 50% 이상 감소 (세력 보유 신호) |

### 추세 및 신고가
| 스크리너 | 설명 |
|----------|------|
| **52주 신고가** | 신고가 돌파 후 8거래일 이내 |
| **이평선 수렴** | 이동평균선 수렴 + 정배열 + 신고가 |

### 반등 및 저점매수
| 스크리너 | 설명 |
|----------|------|
| **바닥 탈출** | 150일선 근접 + 저점 상승 + MACD 신호 (점수제) |
| **낙폭과대 반등** | 52주 고가 대비 -40% + 바닥 상승 전환 |

### 가치 및 우량주
| 스크리너 | 설명 |
|----------|------|
| **저평가 우량주** | PER/PBR + 매출/영업이익/EPS 성장률 기반 (FnGuide TTM) |
| **60주선 우량주** | 60주선 지지 + 영업이익률 10%↑ + 성장률 10%↑ |

### 업종 분석
| 스크리너 | 설명 |
|----------|------|
| **업종별 4단계** | 섹터 로테이션 기반 업종 추세 분석 |

## 자동 업데이트 스케줄

| 스크리너 | 실행 시간 (KST) |
|----------|----------------|
| 기술적 스크리너 (박스권, 거래량 등) | 매일 12:00, 14:00, 16:00, 18:00 |
| 저평가 우량주 (TTM) | 매주 토요일 12:00 |

## 로컬 실행

### 1. 환경 설정

```bash
# 저장소 클론
git clone https://github.com/hsh2578/stock-screener-kr.git
cd stock-screener-kr

# 가상환경 생성 및 활성화
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # macOS/Linux

# 패키지 설치
pip install -r requirements.txt
```

### 2. 스크리너 실행

```bash
# 전체 스크리너 실행 (병렬 처리, ~20초)
python run_screeners.py

# 캐시 무시하고 새로 다운로드
python run_screeners.py --fresh

# 개별 스크리너 실행
python test_value_screener.py   # 저평가 우량주
python ma60w_quality.py         # 60주선 우량주
```

### 3. 결과 확인

```bash
# 로컬 서버 실행
python -m http.server 8000

# 브라우저에서 열기
# http://localhost:8000/주식_스크리너_전체.html
```

## 프로젝트 구조

```
stock-screener-kr/
├── .github/workflows/       # GitHub Actions (자동 업데이트/배포)
│   ├── deploy.yml           # 기술적 스크리너 (매일 4회)
│   └── value-screener.yml   # 저평가 우량주 (주 1회)
├── scripts/screeners/       # 모듈화된 스크리너
│   ├── bottom_breakout.py   # 바닥 탈출
│   └── ma_convergence.py    # 이평선 수렴
├── data/                    # 스크리닝 결과 JSON
├── run_screeners.py         # 메인 스크리너 (병렬 실행)
├── test_value_screener.py   # 저평가 우량주 스크리너
├── ma60w_quality.py         # 60주선 우량주 스크리너
├── naver_finance.py         # FnGuide 재무데이터 크롤러
└── 주식_스크리너_전체.html   # 단일 페이지 뷰어 (SPA)
```

## 기술 스택

| 구분 | 기술 |
|------|------|
| 데이터 수집 | FinanceDataReader (KRX), FnGuide (재무) |
| 백엔드 | Python 3.10+, pandas, numpy |
| 프론트엔드 | 단일 HTML (Vanilla JS, TradingView 차트) |
| 인프라 | GitHub Actions, GitHub Pages |

## 면책 조항

본 서비스는 투자 참고용이며, 투자 판단의 책임은 본인에게 있습니다.

## 라이선스

MIT License
