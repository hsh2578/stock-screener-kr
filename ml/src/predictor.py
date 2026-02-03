"""
52주 신고가 돌파 종목 예측기

학습된 모델을 사용하여 새로운 종목의:
1. 예상 최고 수익률 (20일 내)
2. 10% 이상 상승 확률

을 예측합니다.
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# 경로
ML_DIR = Path(__file__).parent.parent
MODEL_PATH = ML_DIR / "models"

# 필요한 피처
FEATURES = [
    'breakout_pct', 'volume_surge', 'close_strength',
    'base_length', 'volatility_contraction', 'volume_dry_up',
    'ma200_slope', 'pct_above_52w_low',
    'rs_vs_market', 'market_return',
    'ma20_deviation', 'liquidity',
    'days_since_ath', 'atr_ratio'
]


class BreakoutPredictor:
    """52주 신고가 돌파 종목 예측기"""

    def __init__(self):
        self.regressor = None
        self.classifier = None
        self.scaler = None
        self.meta = None
        self._load_models()

    def _load_models(self):
        """모델 로드"""
        try:
            with open(MODEL_PATH / "regressor.pkl", 'rb') as f:
                self.regressor = pickle.load(f)

            with open(MODEL_PATH / "classifier.pkl", 'rb') as f:
                self.classifier = pickle.load(f)

            with open(MODEL_PATH / "scaler.pkl", 'rb') as f:
                self.scaler = pickle.load(f)

            with open(MODEL_PATH / "model_meta.pkl", 'rb') as f:
                self.meta = pickle.load(f)

            print("모델 로드 완료")
            print(f"  학습 샘플: {self.meta['samples']}개")
            print(f"  회귀 R2: {self.meta['regressor_performance']['r2']:.4f}")
            print(f"  분류 Precision: {self.meta['classifier_performance']['precision']:.4f}")

        except FileNotFoundError as e:
            raise RuntimeError(f"모델 파일을 찾을 수 없습니다: {e}")

    def predict(self, features: Dict[str, float]) -> Dict[str, float]:
        """
        단일 종목 예측

        Args:
            features: 피처 딕셔너리
                {
                    'breakout_pct': 5.2,
                    'volume_surge': 3.5,
                    ...
                }

        Returns:
            {
                'predicted_gain': 15.2,  # 예상 최고 수익률 (%)
                'success_probability': 0.62,  # 10%+ 확률
                'recommendation': 'BUY'  # 추천 ('BUY', 'HOLD', 'PASS')
            }
        """
        # 피처 배열 생성
        X = np.array([[features.get(f, 0) for f in FEATURES]])

        # 스케일링
        X_scaled = self.scaler.transform(X)

        # 예측
        predicted_gain = self.regressor.predict(X_scaled)[0]
        success_proba = self.classifier.predict_proba(X_scaled)[0][1]

        # 추천 결정
        if success_proba >= 0.6 and predicted_gain >= 10:
            recommendation = 'BUY'
        elif success_proba >= 0.5 or predicted_gain >= 5:
            recommendation = 'HOLD'
        else:
            recommendation = 'PASS'

        return {
            'predicted_gain': round(predicted_gain, 2),
            'success_probability': round(success_proba, 4),
            'success_probability_pct': round(success_proba * 100, 1),
            'recommendation': recommendation
        }

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        배치 예측

        Args:
            df: 피처가 포함된 DataFrame

        Returns:
            예측 결과가 추가된 DataFrame
        """
        # 피처 추출
        missing_features = [f for f in FEATURES if f not in df.columns]
        if missing_features:
            raise ValueError(f"누락된 피처: {missing_features}")

        X = df[FEATURES].values

        # 스케일링
        X_scaled = self.scaler.transform(X)

        # 예측
        df = df.copy()
        df['predicted_gain'] = self.regressor.predict(X_scaled).round(2)
        df['success_probability'] = self.classifier.predict_proba(X_scaled)[:, 1].round(4)
        df['success_probability_pct'] = (df['success_probability'] * 100).round(1)

        # 추천
        def get_recommendation(row):
            if row['success_probability'] >= 0.6 and row['predicted_gain'] >= 10:
                return 'BUY'
            elif row['success_probability'] >= 0.5 or row['predicted_gain'] >= 5:
                return 'HOLD'
            else:
                return 'PASS'

        df['recommendation'] = df.apply(get_recommendation, axis=1)

        return df

    def get_model_info(self) -> Dict:
        """모델 정보 반환"""
        return {
            'features': FEATURES,
            'regressor_performance': self.meta['regressor_performance'],
            'classifier_performance': self.meta['classifier_performance'],
            'trained_at': self.meta['trained_at'],
            'samples': self.meta['samples'],
            'baseline_success_rate': self.meta['positive_rate']
        }


def predict_from_dict(features: Dict[str, float]) -> Dict[str, float]:
    """간편 예측 함수"""
    predictor = BreakoutPredictor()
    return predictor.predict(features)


def predict_from_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """DataFrame 예측 함수"""
    predictor = BreakoutPredictor()
    return predictor.predict_batch(df)


# 테스트
if __name__ == "__main__":
    print("=" * 60)
    print("52주 신고가 돌파 예측기 테스트")
    print("=" * 60)

    # 예측기 초기화
    predictor = BreakoutPredictor()

    # 테스트 피처 (가상 데이터)
    test_features = {
        'breakout_pct': 5.0,       # 돌파 강도 5%
        'volume_surge': 3.0,       # 거래량 3배
        'close_strength': 0.8,     # 종가 강도
        'base_length': 45,         # 베이스 기간
        'volatility_contraction': 0.6,
        'volume_dry_up': 0.5,
        'ma200_slope': 0.1,
        'pct_above_52w_low': 50,   # 52주 저점 대비 50% 위
        'rs_vs_market': 5.0,       # 시장 대비 5% 아웃퍼폼
        'market_return': 2.0,      # 시장 수익률 2%
        'ma20_deviation': 3.0,     # 20일선 대비 3%
        'liquidity': 5000000000,   # 유동성 50억
        'days_since_ath': 100,     # ATH 이후 100일
        'atr_ratio': 1.2           # ATR 비율
    }

    print("\n테스트 피처:")
    for k, v in test_features.items():
        print(f"  {k}: {v}")

    # 예측
    result = predictor.predict(test_features)

    print("\n" + "=" * 60)
    print("예측 결과")
    print("=" * 60)
    print(f"  예상 최고 수익률: {result['predicted_gain']:+.2f}%")
    print(f"  10%+ 확률: {result['success_probability_pct']:.1f}%")
    print(f"  추천: {result['recommendation']}")

    # 모델 정보
    print("\n" + "=" * 60)
    print("모델 정보")
    print("=" * 60)
    info = predictor.get_model_info()
    print(f"  학습 샘플: {info['samples']}개")
    print(f"  기준 성공률: {info['baseline_success_rate']*100:.1f}%")
    print(f"  회귀 MAE: {info['regressor_performance']['mae']:.2f}%")
    print(f"  분류 Precision: {info['classifier_performance']['precision']*100:.1f}%")
