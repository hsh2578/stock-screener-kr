"""
52주 신고가 돌파 스크리너 (52-Week High Breakout Screener) + ML 예측

52주 신고가를 돌파한 종목을 포착하고,
ML 모델로 예상 수익률과 성공 확률을 예측합니다.

조건:
    - 시가총액: 1,000억 원 이상
    - 52주 신고가: 최근 250거래일 중 최고 종가
    - 돌파: 종가 > 52주 신고가
    - 기간: 돌파 후 8거래일 이내

ML 예측:
    - 예상수익: 돌파 후 20거래일 내 최고 수익률 예측 (회귀)
    - 성공확률: 10% 이상 상승할 확률 예측 (분류)
"""
import sys
import pickle
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from utils.logger import setup_logger

logger = setup_logger()

# ============================================================================
# 상수 정의
# ============================================================================
HIGH_52W_PERIOD = 250  # 52주 = 약 250거래일
MAX_DAYS_SINCE_BREAKOUT = 8  # 돌파 후 최대 거래일
MIN_MARKET_CAP = 1000  # 시가총액 1000억 이상
VOLUME_RATIO_MIN = 1.5  # 돌파일 거래량 >= 20일 평균 * 1.5

# ML 피처 (14개)
FEATURES = [
    'breakout_pct', 'volume_surge', 'close_strength',
    'base_length', 'volatility_contraction', 'volume_dry_up',
    'ma200_slope', 'pct_above_52w_low',
    'rs_vs_market', 'market_return',
    'ma20_deviation', 'liquidity',
    'days_since_ath', 'atr_ratio'
]

# ML 모델 경로
MODEL_DIR = PROJECT_ROOT / "ml" / "models" / "52w_high"

# 전역 변수 (모델 캐싱)
_regressor = None
_classifier = None
_scaler = None
_meta = None
_kospi_data = None
_models_loaded = False


def load_52w_models():
    """52주 신고가 ML 모델 로드"""
    global _regressor, _classifier, _scaler, _meta, _models_loaded

    if _models_loaded:
        return True

    try:
        regressor_path = MODEL_DIR / "regressor.pkl"
        classifier_path = MODEL_DIR / "classifier.pkl"
        scaler_path = MODEL_DIR / "scaler.pkl"
        meta_path = MODEL_DIR / "model_meta.pkl"

        if all(p.exists() for p in [regressor_path, classifier_path, scaler_path]):
            with open(regressor_path, 'rb') as f:
                _regressor = pickle.load(f)
            with open(classifier_path, 'rb') as f:
                _classifier = pickle.load(f)
            with open(scaler_path, 'rb') as f:
                _scaler = pickle.load(f)
            if meta_path.exists():
                with open(meta_path, 'rb') as f:
                    _meta = pickle.load(f)

            _models_loaded = True
            logger.info("[ML] 52주 신고가 모델 로드 완료")
            return True
        else:
            logger.warning("[ML] 52주 신고가 모델 파일 없음")
            return False
    except Exception as e:
        logger.error(f"[ML] 모델 로드 실패: {e}")
        return False


def set_kospi_data(kospi_df: pd.DataFrame):
    """KOSPI 지수 데이터 설정 (외부에서 주입)"""
    global _kospi_data
    _kospi_data = kospi_df


@dataclass
class NewHigh52wResult:
    """52주 신고가 돌파 스크리너 결과"""
    ticker: str
    name: str
    price: int
    change_rate: float
    high_52w: int
    breakout_date: str
    days_since: int
    above_high_percent: float
    market_cap: int
    # ML 예측 필드
    volume_ratio: float  # 거래량비 (돌파일)
    breakout_close: int  # 돌파일 종가
    vs_breakout_close: float  # 돌파종가대비 현재가
    predicted_gain: Optional[float]  # 예상수익 (ML)
    success_probability: Optional[float]  # 성공확률 (ML)
    recommendation: str  # 추천 (BUY/HOLD/PASS)
    updated_at: str

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "name": self.name,
            "price": int(self.price),
            "change_rate": round(float(self.change_rate), 2),
            "high_52w": int(self.high_52w),
            "breakout_date": self.breakout_date,
            "days_since": int(self.days_since),
            "above_high_percent": round(float(self.above_high_percent), 2),
            "market_cap": int(self.market_cap),
            "volume_ratio": round(float(self.volume_ratio), 1),
            "breakout_close": int(self.breakout_close),
            "vs_breakout_close": round(float(self.vs_breakout_close), 2),
            "predicted_gain": round(float(self.predicted_gain), 1) if self.predicted_gain else None,
            "success_probability": round(float(self.success_probability), 1) if self.success_probability else None,
            "recommendation": self.recommendation,
            "updated_at": self.updated_at
        }


def validate_data(data: pd.DataFrame, required_days: int) -> bool:
    """데이터 유효성 검증"""
    if len(data) < required_days:
        return False

    required_cols = ["종가", "거래량"]
    for col in required_cols:
        if col not in data.columns:
            return False
        if data[col].tail(required_days).isna().any():
            return False
        if np.isinf(data[col].tail(required_days)).any():
            return False

    if (data["종가"].tail(required_days) <= 0).any():
        return False

    return True


def find_breakout_info(df: pd.DataFrame) -> Optional[tuple]:
    """
    52주 신고가 돌파 정보를 찾습니다.

    Returns:
        (52주신고가, 52주저가, 돌파일인덱스, 돌파일날짜str, 경과일, 돌파종가, 거래량비) 또는 None
    """
    total_len = len(df)
    required_len = HIGH_52W_PERIOD + MAX_DAYS_SINCE_BREAKOUT + 1

    if total_len < required_len:
        return None

    # 9일 전 기준으로 52주 신고가/저가 계산
    base_idx = total_len - 1 - MAX_DAYS_SINCE_BREAKOUT - 1
    high_52w_prices = df["종가"].iloc[base_idx - HIGH_52W_PERIOD:base_idx]
    low_52w_prices = df["저가"].iloc[base_idx - HIGH_52W_PERIOD:base_idx] if "저가" in df.columns else df["종가"].iloc[base_idx - HIGH_52W_PERIOD:base_idx]

    if high_52w_prices.empty:
        return None

    high_52w = float(high_52w_prices.max())
    low_52w = float(low_52w_prices.min())

    if pd.isna(high_52w) or high_52w <= 0:
        return None

    # 8일 전부터 오늘까지 돌파일 찾기
    for days_ago in range(MAX_DAYS_SINCE_BREAKOUT, -1, -1):
        idx = total_len - 1 - days_ago
        close = df["종가"].iloc[idx]
        volume = df["거래량"].iloc[idx]

        if close > high_52w:
            # 거래량 확인
            if idx >= 20:
                avg_vol_20 = df["거래량"].iloc[idx - 20:idx].mean()
                if avg_vol_20 > 0 and volume >= avg_vol_20 * VOLUME_RATIO_MIN:
                    breakout_date = df.index[idx]
                    volume_ratio = volume / avg_vol_20
                    return high_52w, low_52w, idx, breakout_date.strftime("%Y-%m-%d"), days_ago, close, volume_ratio

    return None


def calculate_features(df: pd.DataFrame, breakout_idx: int, low_52w: float) -> Optional[dict]:
    """
    ML 피처 계산 - 돌파일 기준

    ai_screener.py와 동일한 피처 계산 로직
    """
    global _kospi_data

    try:
        close = df['종가']
        high = df['고가'] if '고가' in df.columns else df['종가']
        low = df['저가'] if '저가' in df.columns else df['종가']
        volume = df['거래량']

        breakout_close = close.iloc[breakout_idx]

        # 52주 고가 (돌파 전)
        lookback_start = max(0, breakout_idx - HIGH_52W_PERIOD)
        high_52w = close.iloc[lookback_start:breakout_idx].max()

        # 1. breakout_pct: 돌파 강도
        breakout_pct = (breakout_close / high_52w - 1) * 100 if high_52w > 0 else 0

        # 2. volume_surge: 거래량 급증
        if breakout_idx >= 20:
            vol_20d_avg = volume.iloc[breakout_idx - 20:breakout_idx].mean()
            vol_breakout = volume.iloc[breakout_idx]
            volume_surge = vol_breakout / vol_20d_avg if vol_20d_avg > 0 else 1
        else:
            volume_surge = 1

        # 3. close_strength: 종가 강도
        breakout_high = high.iloc[breakout_idx]
        breakout_low = low.iloc[breakout_idx]
        if breakout_high != breakout_low:
            close_strength = (breakout_close - breakout_low) / (breakout_high - breakout_low)
        else:
            close_strength = 0.5

        # 4. base_length: 베이스 기간 (60일 중 횡보일수)
        if breakout_idx >= 60:
            pct_change = close.iloc[breakout_idx - 60:breakout_idx].pct_change().abs()
            base_length = (pct_change < 0.03).sum()
        else:
            base_length = 0

        # 5. volatility_contraction: 변동성 수축
        if breakout_idx >= 60:
            recent_close = close.iloc[breakout_idx - 20:breakout_idx]
            past_close = close.iloc[breakout_idx - 60:breakout_idx - 20]
            recent_vol = recent_close.std() / recent_close.mean() if recent_close.mean() > 0 else 0
            past_vol = past_close.std() / past_close.mean() if past_close.mean() > 0 else 0
            volatility_contraction = recent_vol / past_vol if past_vol > 0 else 1
        else:
            volatility_contraction = 1

        # 6. volume_dry_up: 거래량 고갈
        if breakout_idx >= 40:
            vol_recent = volume.iloc[breakout_idx - 10:breakout_idx].mean()
            vol_past = volume.iloc[breakout_idx - 40:breakout_idx - 10].mean()
            volume_dry_up = vol_recent / vol_past if vol_past > 0 else 1
        else:
            volume_dry_up = 1

        # 7. ma200_slope: 200일선 기울기
        if breakout_idx >= 220:
            ma200_now = close.iloc[breakout_idx - 200:breakout_idx].mean()
            ma200_20d_ago = close.iloc[breakout_idx - 220:breakout_idx - 20].mean()
            ma200_slope = (ma200_now / ma200_20d_ago - 1) * 100 if ma200_20d_ago > 0 else 0
        else:
            ma200_slope = 0

        # 8. pct_above_52w_low: 52주 저점 대비 상승률
        pct_above_52w_low = (breakout_close / low_52w - 1) * 100 if low_52w > 0 else 0

        # 9, 10. rs_vs_market, market_return: 시장 대비 상대강도
        rs_vs_market = 0
        market_return = 0
        if _kospi_data is not None and len(_kospi_data) > 0:
            try:
                breakout_date = df.index[breakout_idx]
                kospi_breakout_idx = _kospi_data.index.get_indexer([breakout_date], method='nearest')[0]
                if kospi_breakout_idx >= 20:
                    kospi_close = _kospi_data['Close'] if 'Close' in _kospi_data.columns else _kospi_data['종가']
                    market_return = (kospi_close.iloc[kospi_breakout_idx] / kospi_close.iloc[kospi_breakout_idx - 20] - 1) * 100
                    if breakout_idx >= 20:
                        stock_return = (breakout_close / close.iloc[breakout_idx - 20] - 1) * 100
                        rs_vs_market = stock_return - market_return
            except:
                pass

        # 11. ma20_deviation: 20일선 대비 이격도
        if breakout_idx >= 20:
            ma20 = close.iloc[breakout_idx - 20:breakout_idx].mean()
            ma20_deviation = (breakout_close / ma20 - 1) * 100 if ma20 > 0 else 0
        else:
            ma20_deviation = 0

        # 12. liquidity: 유동성 (20일 평균 거래대금)
        if breakout_idx >= 20:
            liquidity = (close.iloc[breakout_idx - 20:breakout_idx] * volume.iloc[breakout_idx - 20:breakout_idx]).mean()
        else:
            liquidity = 0

        # 13. days_since_ath: ATH 이후 경과일
        ath_start = max(0, breakout_idx - HIGH_52W_PERIOD)
        try:
            ath_idx = high.iloc[ath_start:breakout_idx].idxmax()
            breakout_date = df.index[breakout_idx]
            days_since_ath = (breakout_date - ath_idx).days
        except:
            days_since_ath = 0

        # 14. atr_ratio: ATR 비율
        if breakout_idx >= 15:
            h = high.iloc[breakout_idx - 14:breakout_idx].values
            l = low.iloc[breakout_idx - 14:breakout_idx].values
            c_prev = close.iloc[breakout_idx - 15:breakout_idx - 1].values

            tr1 = h - l
            tr2 = np.abs(h - c_prev)
            tr3 = np.abs(l - c_prev)
            tr = np.maximum(np.maximum(tr1, tr2), tr3)

            atr_14 = tr.mean()
            atr_ratio = atr_14 / breakout_close * 100 if breakout_close > 0 else 0
        else:
            atr_ratio = 0

        return {
            'breakout_pct': breakout_pct,
            'volume_surge': volume_surge,
            'close_strength': close_strength,
            'base_length': base_length,
            'volatility_contraction': volatility_contraction,
            'volume_dry_up': volume_dry_up,
            'ma200_slope': ma200_slope,
            'pct_above_52w_low': pct_above_52w_low,
            'rs_vs_market': rs_vs_market,
            'market_return': market_return,
            'ma20_deviation': ma20_deviation,
            'liquidity': liquidity,
            'days_since_ath': days_since_ath,
            'atr_ratio': atr_ratio
        }

    except Exception as e:
        logger.debug(f"피처 계산 오류: {e}")
        return None


def predict_ml(features: dict) -> tuple[Optional[float], Optional[float], str]:
    """
    ML 모델로 예측

    Returns:
        (예상수익, 성공확률, 추천)
    """
    global _regressor, _classifier, _scaler

    if not _models_loaded or _regressor is None or _classifier is None or _scaler is None:
        return None, None, "N/A"

    try:
        X = np.array([[features.get(f, 0) for f in FEATURES]])
        X_scaled = _scaler.transform(X)

        predicted_gain = float(_regressor.predict(X_scaled)[0])
        success_proba = float(_classifier.predict_proba(X_scaled)[0][1]) * 100

        # 추천 결정
        if success_proba >= 60 and predicted_gain >= 10:
            recommendation = 'BUY'
        elif success_proba >= 50 or predicted_gain >= 5:
            recommendation = 'HOLD'
        else:
            recommendation = 'PASS'

        return predicted_gain, success_proba, recommendation
    except Exception as e:
        logger.debug(f"ML 예측 오류: {e}")
        return None, None, "N/A"


def analyze_new_high_52w(
    data: pd.DataFrame,
    ticker: str,
    name: str,
    market_cap: int
) -> Optional[NewHigh52wResult]:
    """개별 종목 분석 + ML 예측"""
    required_days = HIGH_52W_PERIOD + MAX_DAYS_SINCE_BREAKOUT + 2
    if not validate_data(data, required_days):
        return None

    # 돌파 정보 찾기
    breakout_result = find_breakout_info(data)
    if breakout_result is None:
        return None

    high_52w, low_52w, breakout_idx, breakout_date, days_since, breakout_close, volume_ratio = breakout_result

    # 현재가 및 등락률
    current_close = data["종가"].iloc[-1]
    prev_close = data["종가"].iloc[-2]

    if pd.isna(current_close) or pd.isna(prev_close) or prev_close <= 0:
        return None

    if current_close <= high_52w:
        return None

    change_rate = (current_close - prev_close) / prev_close * 100
    above_high_percent = (current_close - high_52w) / high_52w * 100
    vs_breakout_close = (current_close / breakout_close - 1) * 100 if breakout_close > 0 else 0

    # ML 피처 계산 및 예측
    predicted_gain = None
    success_probability = None
    recommendation = "N/A"

    if _models_loaded:
        features = calculate_features(data, breakout_idx, low_52w)
        if features:
            predicted_gain, success_probability, recommendation = predict_ml(features)

    return NewHigh52wResult(
        ticker=ticker,
        name=name,
        price=int(current_close),
        change_rate=change_rate,
        high_52w=int(high_52w),
        breakout_date=breakout_date,
        days_since=days_since,
        above_high_percent=above_high_percent,
        market_cap=market_cap,
        volume_ratio=volume_ratio,
        breakout_close=int(breakout_close),
        vs_breakout_close=vs_breakout_close,
        predicted_gain=predicted_gain,
        success_probability=success_probability,
        recommendation=recommendation,
        updated_at=datetime.now().isoformat()
    )


def screen_new_high_52w(
    stock_data: dict[str, pd.DataFrame],
    stock_info: pd.DataFrame,
    kospi_data: pd.DataFrame = None
) -> list[dict]:
    """
    52주 신고가 돌파 스크리너 실행

    Args:
        stock_data: OHLCV 데이터
        stock_info: 종목 정보
        kospi_data: KOSPI 지수 데이터 (ML 피처용)
    """
    # ML 모델 로드
    load_52w_models()

    # KOSPI 데이터 설정
    if kospi_data is not None:
        set_kospi_data(kospi_data)

    results = []
    total = len(stock_data)
    passed_cap = 0
    passed_data = 0

    logger.info(f"[52주 신고가] 시작: {total}개 종목 (ML: {'활성화' if _models_loaded else '비활성화'})")

    for ticker, data in stock_data.items():
        stock_row = stock_info[stock_info["ticker"] == ticker]
        if stock_row.empty:
            continue

        name = stock_row.iloc[0].get("name", "")
        market_cap = stock_row.iloc[0].get("market_cap", 0)

        if market_cap < MIN_MARKET_CAP:
            continue
        passed_cap += 1

        required_days = HIGH_52W_PERIOD + MAX_DAYS_SINCE_BREAKOUT + 2
        if not validate_data(data, required_days):
            continue
        passed_data += 1

        result = analyze_new_high_52w(data, ticker, name, market_cap)
        if result:
            results.append(result)

    logger.info(
        f"[52주 신고가] 완료: "
        f"전체 {total}개 → 시총 {passed_cap}개 → "
        f"데이터 {passed_data}개 → 돌파 {len(results)}개"
    )

    # 돌파일 기준 정렬 (최신순), 같은 날이면 성공확률 높은 순
    results.sort(key=lambda x: (x.days_since, -(x.success_probability or 0)))
    return [r.to_dict() for r in results]
