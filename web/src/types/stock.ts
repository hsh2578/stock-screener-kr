/**
 * 주식 데이터 타입 정의
 */

// 일별 주가 데이터
export interface StockData {
  date: string;          // YYYY-MM-DD 형식
  open: number;          // 시가
  high: number;          // 고가
  low: number;           // 저가
  close: number;         // 종가
  volume: number;        // 거래량
}

// 기본 스크리너 결과
export interface ScreenerResult {
  ticker: string;        // 종목코드 (6자리)
  name: string;          // 종목명
  price: number;         // 현재가
  change: number;        // 전일대비 변동률 (%)
  marketCap: number;     // 시가총액 (억원)
  volume: number;        // 거래량
  updatedAt: string;     // 데이터 갱신 시각
}

// 1. 저PER 스크리너 결과
export interface LowPERResult extends ScreenerResult {
  per: number;           // PER
  eps: number;           // EPS (주당순이익)
  industry: string;      // 업종
  industryAvgPER: number; // 업종 평균 PER
}

// 2. 저PBR 스크리너 결과
export interface LowPBRResult extends ScreenerResult {
  pbr: number;           // PBR
  bps: number;           // BPS (주당순자산)
  roe: number;           // ROE (자기자본이익률)
}

// 3. 고배당 스크리너 결과
export interface HighDividendResult extends ScreenerResult {
  dividendYield: number;     // 배당수익률 (%)
  dividendPerShare: number;  // 주당배당금
  payoutRatio: number;       // 배당성향 (%)
  consecutiveYears: number;  // 연속 배당 연수
}

// 4. 성장주 스크리너 결과
export interface GrowthStockResult extends ScreenerResult {
  revenueGrowth: number;     // 매출액 성장률 (%)
  profitGrowth: number;      // 영업이익 성장률 (%)
  pegRatio: number;          // PEG 비율
  revenueCAGR3Y: number;     // 3년 매출 CAGR (%)
}

// 5. 52주 신저가 스크리너 결과
export interface Week52LowResult extends ScreenerResult {
  week52High: number;        // 52주 최고가
  week52Low: number;         // 52주 최저가
  fromHigh: number;          // 고점 대비 하락률 (%)
  fromLow: number;           // 저점 대비 상승률 (%)
}

// 6. 거래량 급증 스크리너 결과
export interface VolumeSpikeResult extends ScreenerResult {
  avgVolume20D: number;      // 20일 평균 거래량
  volumeRatio: number;       // 거래량 비율 (당일/20일평균)
  priceChange5D: number;     // 5일 가격 변동률 (%)
}

// 7. 모멘텀 스크리너 결과
export interface MomentumResult extends ScreenerResult {
  return1M: number;          // 1개월 수익률 (%)
  return3M: number;          // 3개월 수익률 (%)
  return6M: number;          // 6개월 수익률 (%)
  return12M: number;         // 12개월 수익률 (%)
  rsi14: number;             // 14일 RSI
}

// 스크리너 종류
export type ScreenerType =
  | 'low-per'
  | 'low-pbr'
  | 'high-dividend'
  | 'growth'
  | 'week52-low'
  | 'volume-spike'
  | 'momentum';

// 스크리너 메타 정보
export interface ScreenerMeta {
  type: ScreenerType;
  name: string;              // 스크리너 한글명
  description: string;       // 설명
  lastUpdated: string;       // 마지막 업데이트 시각
  totalCount: number;        // 전체 결과 수
}

// 스크리너 데이터 파일 구조
export interface ScreenerDataFile<T extends ScreenerResult> {
  meta: ScreenerMeta;
  data: T[];
}

// 정렬 옵션
export interface SortOption {
  field: string;
  direction: 'asc' | 'desc';
}

// 필터 옵션
export interface FilterOption {
  field: string;
  operator: 'eq' | 'ne' | 'gt' | 'gte' | 'lt' | 'lte' | 'between';
  value: number | [number, number];
}
