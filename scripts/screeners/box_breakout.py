"""
박스권 돌파 스크리너 - 거래량 동반 (Box Breakout with Volume)

박스권 상단 돌파 + 거래량 폭발 종목을 포착합니다. (1차 매수 타점)
거래량 없는 돌파는 가짜 돌파일 가능성이 높으므로 거래량 조건을 포함합니다.

조건:
    - 시가총액: 1,000억 원 이상
    - 사전 조건: 40거래일 이상 박스권 (종가 변동폭 20% 이내)
    - 돌파 조건: 종가 > 저항선 × 1.02
    - 거래량: 당일 거래량 ≥ 20일 평균 × 2배
    - 이평선: 돌파 시점에 주가가 150일선 위에 위치

ML 모델:
    - 회귀: 10개 피처 + Ridge (R² = 0.0701)
    - 분류: 15개 피처 + XGBoost (F1 = 0.32, Precision = 50.6%)
"""
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import joblib

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from utils.logger import setup_logger

logger = setup_logger()

# ============================================================================
# 상수 정의
# ============================================================================
BOX_PERIOD = 40
MAX_RANGE_PERCENT = 20.0
BREAKOUT_THRESHOLD = 1.02
VOLUME_MULTIPLIER = 2.0
MA_PERIOD = 150
VOLUME_AVG_PERIOD = 20
MIN_MARKET_CAP = 1000
MAX_DAYS_SINCE_BREAKOUT = 5  # 돌파 후 최대 경과일

# 모델 경로
MODEL_DIR = PROJECT_ROOT / "ml" / "models"
REG_MODEL_PATH = MODEL_DIR / "box_breakout_regression_model.joblib"
CLS_MODEL_PATH = MODEL_DIR / "box_breakout_classification_model.joblib"

# 글로벌 모델 변수
_reg_model = None
_cls_model = None
_models_loaded = False

# KOSPI 데이터 (시장 수익률 계산용)
_kospi_data = None


def load_models():
    """ML 모델 로드"""
    global _reg_model, _cls_model, _models_loaded

    if _models_loaded:
        return True

    try:
        if REG_MODEL_PATH.exists() and CLS_MODEL_PATH.exists():
            _reg_model = joblib.load(REG_MODEL_PATH)
            _cls_model = joblib.load(CLS_MODEL_PATH)
            _models_loaded = True
            logger.info("[박스권 돌파] ML 모델 로드 완료")
            return True
        else:
            logger.warning("[박스권 돌파] ML 모델 파일 없음 - 점수 계산 생략")
            return False
    except Exception as e:
        logger.error(f"[박스권 돌파] ML 모델 로드 실패: {e}")
        return False


def set_kospi_data(kospi_df: pd.DataFrame):
    """KOSPI 데이터 설정"""
    global _kospi_data
    _kospi_data = kospi_df


@dataclass
class BoxBreakoutResult:
    """박스권 돌파 (거래량 동반) 결과"""
    ticker: str
    name: str
    price: int
    resistance: int
    breakout_date: str
    volume_ratio: float
    ma150_diff: float  # 현재가 vs 150일선 차이 (%)
    updated_at: str
    # ML 예측 필드
    success_prob: float = 0.0
    predicted_gain: float = 0.0
    ai_score: float = 0.0

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "name": self.name,
            "price": self.price,
            "resistance": self.resistance,
            "breakout_date": self.breakout_date,
            "volume_ratio": round(self.volume_ratio, 2),
            "ma150_diff": round(self.ma150_diff, 2),
            "success_prob": round(self.success_prob, 1),
            "predicted_gain": round(self.predicted_gain, 1),
            "ai_score": round(self.ai_score, 1),
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


def calculate_ma(close: pd.Series, period: int) -> pd.Series:
    """이동평균선 계산"""
    return close.rolling(window=period, min_periods=period).mean()


def check_box_range(data: pd.DataFrame, end_idx: int) -> Optional[tuple[int, float]]:
    """박스권 확인 및 저항선 반환"""
    start_idx = end_idx - BOX_PERIOD
    if start_idx < 0:
        return None

    box_data = data.iloc[start_idx:end_idx]
    if len(box_data) < BOX_PERIOD:
        return None

    close = box_data["종가"]
    if close.isna().any() or (close <= 0).any():
        return None

    box_high = close.max()
    box_low = close.min()

    if box_low <= 0:
        return None

    range_percent = (box_high - box_low) / box_low * 100
    if range_percent > MAX_RANGE_PERCENT:
        return None

    return int(box_high), range_percent


def calculate_features(data: pd.DataFrame, breakout_idx: int, resistance: float, support: float) -> Optional[dict]:
    """
    15개 피처 계산

    회귀 모델용 10개:
    - ma20_deviation, pct_above_52w_low, breakout_strength,
    - close_strength, days_since_ath, atr_ratio,
    - liquidity, breakout_gap, rs_vs_market, market_return

    분류 모델용 추가 5개:
    - volume_surge, box_range_pct, volume_dry_up, ma200_slope, volatility_contraction
    """
    try:
        close = data["종가"]
        high = data["고가"] if "고가" in data.columns else data["종가"]
        low = data["저가"] if "저가" in data.columns else data["종가"]
        volume = data["거래량"]
        open_price = data["시가"] if "시가" in data.columns else data["종가"]

        breakout_close = close.iloc[breakout_idx]
        breakout_high = high.iloc[breakout_idx]
        breakout_low = low.iloc[breakout_idx]
        breakout_open = open_price.iloc[breakout_idx]

        features = {}

        # 1. box_range_pct - 박스권 크기
        features['box_range_pct'] = (resistance - support) / support * 100 if support > 0 else 0

        # 2. breakout_strength - 돌파 강도
        features['breakout_strength'] = (breakout_close / resistance - 1) * 100 if resistance > 0 else 0

        # 3. volume_surge - 거래량 급증
        if breakout_idx >= 20:
            vol_20d_avg = volume.iloc[breakout_idx - 20:breakout_idx].mean()
            features['volume_surge'] = volume.iloc[breakout_idx] / vol_20d_avg if vol_20d_avg > 0 else 1.0
        else:
            features['volume_surge'] = 1.0

        # 4. volume_dry_up - 박스권 내 거래량 감소
        if breakout_idx >= 40:
            vol_recent = volume.iloc[breakout_idx - 10:breakout_idx].mean()
            vol_past = volume.iloc[breakout_idx - 40:breakout_idx - 10].mean()
            features['volume_dry_up'] = vol_recent / vol_past if vol_past > 0 else 1.0
        else:
            features['volume_dry_up'] = 1.0

        # 5. close_strength - 종가 강도
        if breakout_high != breakout_low:
            features['close_strength'] = (breakout_close - breakout_low) / (breakout_high - breakout_low)
        else:
            features['close_strength'] = 0.5

        # 6. volatility_contraction - 변동성 수축
        if breakout_idx >= 60:
            recent_close = close.iloc[breakout_idx - 20:breakout_idx]
            past_close = close.iloc[breakout_idx - 60:breakout_idx - 20]

            recent_vol = recent_close.std() / recent_close.mean() if recent_close.mean() > 0 else 0
            past_vol = past_close.std() / past_close.mean() if past_close.mean() > 0 else 0

            features['volatility_contraction'] = recent_vol / past_vol if past_vol > 0 else 1.0
        else:
            features['volatility_contraction'] = 1.0

        # 7. ma20_deviation - 20일선 이격도
        if breakout_idx >= 20:
            ma20 = close.iloc[breakout_idx - 20:breakout_idx].mean()
            features['ma20_deviation'] = (breakout_close / ma20 - 1) * 100 if ma20 > 0 else 0
        else:
            features['ma20_deviation'] = 0

        # 8. breakout_gap - 돌파일 갭
        if breakout_idx >= 1:
            prev_close = close.iloc[breakout_idx - 1]
            features['breakout_gap'] = (breakout_open / prev_close - 1) * 100 if prev_close > 0 else 0
        else:
            features['breakout_gap'] = 0

        # 9. ma200_slope - 200일선 기울기
        if breakout_idx >= 220:
            ma200_now = close.iloc[breakout_idx - 200:breakout_idx].mean()
            ma200_20d_ago = close.iloc[breakout_idx - 220:breakout_idx - 20].mean()
            features['ma200_slope'] = (ma200_now / ma200_20d_ago - 1) * 100 if ma200_20d_ago > 0 else 0
        else:
            features['ma200_slope'] = 0

        # 10. pct_above_52w_low - 52주 저점 대비
        lookback_start = max(0, breakout_idx - 250)
        low_52w = low.iloc[lookback_start:breakout_idx].min()
        features['pct_above_52w_low'] = (breakout_close / low_52w - 1) * 100 if low_52w > 0 else 0

        # 11. days_since_ath - 최고가 이후 경과일
        try:
            high_52w_idx = high.iloc[lookback_start:breakout_idx].idxmax()
            breakout_date = data.index[breakout_idx]
            features['days_since_ath'] = (breakout_date - high_52w_idx).days
        except:
            features['days_since_ath'] = 0

        # 12. rs_vs_market - 시장 대비 상대강도
        # 13. market_return - 시장 수익률
        if _kospi_data is not None and breakout_idx >= 20:
            try:
                breakout_date = data.index[breakout_idx]
                kospi_idx = _kospi_data.index.get_indexer([breakout_date], method='nearest')[0]

                if kospi_idx >= 20:
                    market_return = (_kospi_data['Close'].iloc[kospi_idx] /
                                   _kospi_data['Close'].iloc[kospi_idx - 20] - 1) * 100
                    stock_return = (breakout_close / close.iloc[breakout_idx - 20] - 1) * 100

                    features['market_return'] = market_return
                    features['rs_vs_market'] = stock_return - market_return
                else:
                    features['market_return'] = 0
                    features['rs_vs_market'] = 0
            except:
                features['market_return'] = 0
                features['rs_vs_market'] = 0
        else:
            features['market_return'] = 0
            features['rs_vs_market'] = 0

        # 14. atr_ratio - ATR 비율
        if breakout_idx >= 14:
            tr_values = []
            for j in range(breakout_idx - 14, breakout_idx):
                h = high.iloc[j]
                l = low.iloc[j]
                c_prev = close.iloc[j - 1] if j > 0 else close.iloc[j]

                tr = max(h - l, abs(h - c_prev), abs(l - c_prev))
                tr_values.append(tr)

            atr_14 = np.mean(tr_values)
            features['atr_ratio'] = atr_14 / breakout_close * 100 if breakout_close > 0 else 0
        else:
            features['atr_ratio'] = 0

        # 15. liquidity - 유동성 (5일 평균 거래대금, 로그 스케일)
        if breakout_idx >= 5:
            liquidity = (close.iloc[breakout_idx - 5:breakout_idx] *
                        volume.iloc[breakout_idx - 5:breakout_idx]).mean()
            features['liquidity'] = np.log1p(liquidity)
        else:
            features['liquidity'] = 0

        return features

    except Exception as e:
        return None


def predict_with_model(features: dict) -> tuple[float, float, float]:
    """
    ML 모델로 예측

    Returns:
        (success_prob, predicted_gain, ai_score)
    """
    if not _models_loaded or _reg_model is None or _cls_model is None:
        return 0.0, 0.0, 0.0

    try:
        # 회귀 모델용 피처 (10개)
        reg_features = _reg_model['features']
        X_reg = np.array([[features.get(f, 0) for f in reg_features]])
        X_reg_scaled = _reg_model['scaler'].transform(X_reg)
        predicted_gain = _reg_model['model'].predict(X_reg_scaled)[0]

        # 분류 모델용 피처 (15개)
        cls_features = _cls_model['features']
        X_cls = np.array([[features.get(f, 0) for f in cls_features]])
        X_cls_scaled = _cls_model['scaler'].transform(X_cls)
        success_prob = _cls_model['model'].predict_proba(X_cls_scaled)[0][1] * 100

        # AI 종합 점수 (성공확률 70% + 예측수익률 30%)
        gain_score = min(100, max(0, predicted_gain * 3))  # 0~33% 수익률을 0~100점으로
        ai_score = success_prob * 0.7 + gain_score * 0.3

        return success_prob, predicted_gain, ai_score

    except Exception as e:
        logger.debug(f"예측 실패: {e}")
        return 0.0, 0.0, 0.0


def find_volume_breakout(
    data: pd.DataFrame,
    ticker: str
) -> Optional[tuple[int, int, str, float, float, dict]]:
    """
    거래량 동반 돌파일을 찾습니다.

    Returns:
        (저항선, 지지선, 돌파일, 거래량비율, 150일선대비%, 피처dict) 또는 None
    """
    required_days = BOX_PERIOD + MA_PERIOD
    if len(data) < required_days:
        return None

    close = data["종가"]
    volume = data["거래량"]

    # 150일 이동평균
    ma150 = calculate_ma(close, MA_PERIOD)

    # 20일 평균 거래량
    vol_avg = volume.rolling(window=VOLUME_AVG_PERIOD, min_periods=VOLUME_AVG_PERIOD).mean()

    # 최근 MAX_DAYS_SINCE_BREAKOUT 내에서 돌파 탐색
    for days_ago in range(0, MAX_DAYS_SINCE_BREAKOUT + 1):
        idx = len(data) - 1 - days_ago
        if idx < BOX_PERIOD + MA_PERIOD:
            continue

        # 박스권 확인 (돌파일 직전까지)
        box_result = check_box_range(data, idx)
        if box_result is None:
            continue

        resistance, box_range_pct = box_result

        # 지지선 계산
        box_data = data.iloc[idx - BOX_PERIOD:idx]
        support = int(box_data["종가"].min())

        # 돌파 조건: 종가 > 저항선 × 1.02
        breakout_close = close.iloc[idx]
        if pd.isna(breakout_close) or breakout_close <= resistance * BREAKOUT_THRESHOLD:
            continue

        # 거래량 조건: 당일 거래량 ≥ 20일 평균 × 2배
        day_volume = volume.iloc[idx]
        avg_vol = vol_avg.iloc[idx - 1] if idx > 0 else vol_avg.iloc[idx]

        if pd.isna(day_volume) or pd.isna(avg_vol) or avg_vol <= 0:
            continue

        volume_ratio = day_volume / avg_vol
        if volume_ratio < VOLUME_MULTIPLIER:
            continue

        # 이평선 조건: 150일선 위에 위치
        ma150_value = ma150.iloc[idx]
        if pd.isna(ma150_value) or ma150_value <= 0:
            continue

        if breakout_close <= ma150_value:
            continue

        ma150_diff = (breakout_close / ma150_value - 1) * 100

        # 돌파일 이전에 이미 돌파한 적이 없는지 확인
        pre_period = close.iloc[idx - BOX_PERIOD:idx]
        if (pre_period > resistance * BREAKOUT_THRESHOLD).any():
            continue

        breakout_date = data.index[idx]
        if isinstance(breakout_date, pd.Timestamp):
            breakout_date_str = breakout_date.strftime("%Y-%m-%d")
        else:
            breakout_date_str = str(breakout_date)

        # 피처 계산
        features = calculate_features(data, idx, resistance, support)

        return resistance, support, breakout_date_str, volume_ratio, ma150_diff, features

    return None


def analyze_box_breakout(
    data: pd.DataFrame,
    ticker: str,
    name: str
) -> Optional[BoxBreakoutResult]:
    """개별 종목 분석"""
    required_days = BOX_PERIOD + MA_PERIOD
    if not validate_data(data, required_days):
        return None

    result = find_volume_breakout(data, ticker)
    if result is None:
        return None

    resistance, support, breakout_date, volume_ratio, ma150_diff, features = result
    current_price = int(data["종가"].iloc[-1])

    # ML 예측
    success_prob, predicted_gain, ai_score = 0.0, 0.0, 0.0
    if features is not None:
        success_prob, predicted_gain, ai_score = predict_with_model(features)

    return BoxBreakoutResult(
        ticker=ticker,
        name=name,
        price=current_price,
        resistance=resistance,
        breakout_date=breakout_date,
        volume_ratio=volume_ratio,
        ma150_diff=ma150_diff,
        success_prob=success_prob,
        predicted_gain=predicted_gain,
        ai_score=ai_score,
        updated_at=datetime.now().isoformat()
    )


def screen_box_breakout(
    stock_data: dict[str, pd.DataFrame],
    stock_info: pd.DataFrame,
    kospi_data: pd.DataFrame = None
) -> list[dict]:
    """박스권 돌파 (거래량 동반) 스크리너 실행"""

    # KOSPI 데이터 설정
    if kospi_data is not None:
        set_kospi_data(kospi_data)

    # 모델 로드
    load_models()

    results = []
    total = len(stock_data)
    passed_cap = 0
    passed_data = 0

    logger.info(f"[박스권 돌파(거래량)] 시작: {total}개 종목")

    for ticker, data in stock_data.items():
        stock_row = stock_info[stock_info["ticker"] == ticker]
        if stock_row.empty:
            continue

        name = stock_row.iloc[0].get("name", "")
        market_cap = stock_row.iloc[0].get("market_cap", 0)

        if market_cap < MIN_MARKET_CAP:
            continue
        passed_cap += 1

        required_days = BOX_PERIOD + MA_PERIOD
        if not validate_data(data, required_days):
            continue
        passed_data += 1

        result = analyze_box_breakout(data, ticker, name)
        if result:
            results.append(result)

    logger.info(
        f"[박스권 돌파(거래량)] 완료: "
        f"전체 {total}개 → 시총 {passed_cap}개 → "
        f"데이터 {passed_data}개 → 돌파 {len(results)}개"
    )

    # AI 점수 내림차순 정렬 (모델 있으면), 없으면 거래량비율
    if _models_loaded:
        results.sort(key=lambda x: x.ai_score, reverse=True)
    else:
        results.sort(key=lambda x: x.volume_ratio, reverse=True)

    return [r.to_dict() for r in results]
