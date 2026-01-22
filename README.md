# 국내 주식 종목 발굴 스크리너

개인용 가치투자 종목 발굴을 위한 국내 주식 스크리너입니다.
GitHub Actions와 GitHub Pages를 이용해 100% 무료로 운영됩니다.

## 스크리너 기능

| # | 스크리너 | 설명 |
|---|----------|------|
| 1 | **저PER** | 업종 평균 대비 저평가된 종목 발굴 |
| 2 | **저PBR** | 자산가치 대비 저평가된 종목 발굴 |
| 3 | **고배당** | 배당수익률이 높은 종목 발굴 |
| 4 | **성장주** | 매출/이익 성장률이 높은 종목 발굴 |
| 5 | **52주 신저가** | 52주 최저가 근처의 종목 발굴 |
| 6 | **거래량 급증** | 거래량이 급증한 종목 발굴 |
| 7 | **모멘텀** | 가격 모멘텀이 강한 종목 발굴 |

## 프로젝트 구조

```
stock-screener-kr/
├── .github/workflows/      # GitHub Actions 워크플로우
├── scripts/                # Python 스크립트
│   ├── screeners/          # 스크리너 로직
│   └── utils/              # 유틸리티 (지표 계산 등)
├── data/                   # 결과 JSON 파일 저장소
├── logs/                   # 실행 로그 저장
├── web/                    # React 웹앱 (Vite + TypeScript)
└── README.md
```

## 설치 및 실행

### 1. Python 환경 설정

```bash
# 가상환경 생성 (권장)
python -m venv venv

# 가상환경 활성화
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

# 패키지 설치
pip install -r requirements.txt
```

### 2. 웹앱 개발 서버 실행

```bash
cd web

# 패키지 설치
npm install

# 개발 서버 실행
npm run dev
```

### 3. 웹앱 빌드 (배포용)

```bash
cd web
npm run build
```

빌드 결과물은 `web/dist` 폴더에 생성됩니다.

## GitHub Actions 자동화

### 데이터 수집 워크플로우

- **실행 주기**: 매일 장 마감 후 (평일 18:00 KST)
- **수행 작업**:
  1. pykrx를 이용해 KRX에서 주가 데이터 수집
  2. 7가지 스크리너 조건에 맞는 종목 필터링
  3. 결과를 JSON 파일로 `data/` 폴더에 저장
  4. 변경사항 자동 커밋 및 푸시

### 웹 배포 워크플로우

- **실행 조건**: `web/` 폴더 변경 시 또는 수동 트리거
- **수행 작업**:
  1. React 앱 빌드
  2. GitHub Pages에 자동 배포

## 기술 스택

### Backend (데이터 수집)
- Python 3.11+
- pykrx - KRX 주가 데이터 수집
- pandas - 데이터 처리
- numpy - 수치 계산

### Frontend (웹)
- React 18+
- TypeScript
- Vite - 빌드 도구
- Tailwind CSS - 스타일링

### Infrastructure
- GitHub Actions - CI/CD 자동화
- GitHub Pages - 정적 웹 호스팅

## 데이터 소스

- **주가 데이터**: KRX (한국거래소) via pykrx
- **재무 데이터**: KRX 공시 데이터

## 면책 조항

이 서비스는 개인 학습 및 투자 참고용으로 제작되었습니다.
제공되는 정보는 투자 권유가 아니며, 투자 결정에 따른 책임은 본인에게 있습니다.

## 라이선스

MIT License
