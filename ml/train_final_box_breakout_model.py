"""
박스권 돌파 최종 모델 학습 및 저장

최적 조합:
- 회귀: 10개 피처 + Ridge (alpha=10.0)
- 분류: 15개 피처 + XGBoost (튜닝된 파라미터)
"""

import pandas as pd
import numpy as np
import joblib
import json
import warnings
from pathlib import Path
from datetime import datetime

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_predict, KFold, StratifiedKFold
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

from xgboost import XGBClassifier

warnings.filterwarnings('ignore')

# 경로
BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "data" / "box_breakout_history.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

# 피처 정의
FEATURES_10 = [
    'ma20_deviation', 'pct_above_52w_low', 'breakout_strength',
    'close_strength', 'days_since_ath', 'atr_ratio',
    'liquidity', 'breakout_gap', 'rs_vs_market', 'market_return'
]

FEATURES_15 = FEATURES_10 + [
    'volume_surge', 'box_range_pct', 'volume_dry_up',
    'ma200_slope', 'volatility_contraction'
]

# 최적 파라미터
RIDGE_PARAMS = {'alpha': 10.0}

XGBOOST_PARAMS = {
    'n_estimators': 200,
    'max_depth': 7,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.9,
    'random_state': 42,
    'eval_metric': 'logloss',
    'verbosity': 0
}


def main():
    print("=" * 80)
    print("박스권 돌파 최종 모델 학습")
    print("=" * 80)

    # 데이터 로드
    print("\n[1/5] 데이터 로드...")
    df = pd.read_csv(DATA_PATH, encoding='utf-8-sig')

    y_reg = df['max_gain_20d'].values
    y_cls = df['success'].values

    print(f"  샘플 수: {len(df):,}")
    print(f"  성공률: {y_cls.mean()*100:.2f}%")

    # 회귀 모델 학습
    print("\n[2/5] 회귀 모델 학습 (10개 피처 + Ridge)...")

    X_reg = df[FEATURES_10].values
    scaler_reg = StandardScaler()
    X_reg_scaled = scaler_reg.fit_transform(X_reg)

    reg_model = Ridge(**RIDGE_PARAMS, max_iter=5000)

    # CV 성능 확인
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    y_pred_cv = cross_val_predict(reg_model, X_reg_scaled, y_reg, cv=kfold)

    reg_r2 = r2_score(y_reg, y_pred_cv)
    reg_mae = mean_absolute_error(y_reg, y_pred_cv)
    reg_rmse = np.sqrt(mean_squared_error(y_reg, y_pred_cv))

    print(f"  CV R²: {reg_r2:.4f}")
    print(f"  CV MAE: {reg_mae:.2f}%")
    print(f"  CV RMSE: {reg_rmse:.2f}%")

    # 전체 데이터로 학습
    reg_model.fit(X_reg_scaled, y_reg)
    print("  모델 학습 완료!")

    # 분류 모델 학습
    print("\n[3/5] 분류 모델 학습 (15개 피처 + XGBoost)...")

    X_cls = df[FEATURES_15].values
    scaler_cls = StandardScaler()
    X_cls_scaled = scaler_cls.fit_transform(X_cls)

    cls_model = XGBClassifier(**XGBOOST_PARAMS)

    # CV 성능 확인
    skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred_cv = cross_val_predict(cls_model, X_cls_scaled, y_cls, cv=skfold)
    y_proba_cv = cross_val_predict(cls_model, X_cls_scaled, y_cls, cv=skfold, method='predict_proba')[:, 1]

    cls_acc = accuracy_score(y_cls, y_pred_cv)
    cls_prec = precision_score(y_cls, y_pred_cv)
    cls_rec = recall_score(y_cls, y_pred_cv)
    cls_f1 = f1_score(y_cls, y_pred_cv)
    cls_auc = roc_auc_score(y_cls, y_proba_cv)

    print(f"  CV Accuracy: {cls_acc:.4f}")
    print(f"  CV Precision: {cls_prec:.4f}")
    print(f"  CV Recall: {cls_rec:.4f}")
    print(f"  CV F1: {cls_f1:.4f}")
    print(f"  CV AUC: {cls_auc:.4f}")

    # 전체 데이터로 학습
    cls_model.fit(X_cls_scaled, y_cls)
    print("  모델 학습 완료!")

    # 모델 저장
    print("\n[4/5] 모델 저장...")

    # 회귀 모델
    reg_path = MODEL_DIR / "box_breakout_regression_model.joblib"
    joblib.dump({
        'model': reg_model,
        'scaler': scaler_reg,
        'features': FEATURES_10,
        'params': RIDGE_PARAMS,
        'metrics': {
            'r2': reg_r2,
            'mae': reg_mae,
            'rmse': reg_rmse
        }
    }, reg_path)
    print(f"  회귀 모델: {reg_path}")

    # 분류 모델
    cls_path = MODEL_DIR / "box_breakout_classification_model.joblib"
    joblib.dump({
        'model': cls_model,
        'scaler': scaler_cls,
        'features': FEATURES_15,
        'params': XGBOOST_PARAMS,
        'metrics': {
            'accuracy': cls_acc,
            'precision': cls_prec,
            'recall': cls_rec,
            'f1': cls_f1,
            'auc': cls_auc
        }
    }, cls_path)
    print(f"  분류 모델: {cls_path}")

    # 메타데이터 저장
    print("\n[5/5] 메타데이터 저장...")

    metadata = {
        'model_type': 'box_breakout',
        'created_at': datetime.now().isoformat(),
        'data_samples': len(df),
        'success_rate': float(y_cls.mean()),
        'regression': {
            'model': 'Ridge',
            'features': FEATURES_10,
            'params': RIDGE_PARAMS,
            'cv_r2': float(reg_r2),
            'cv_mae': float(reg_mae),
            'cv_rmse': float(reg_rmse)
        },
        'classification': {
            'model': 'XGBoost',
            'features': FEATURES_15,
            'params': {k: v for k, v in XGBOOST_PARAMS.items() if k != 'verbosity'},
            'cv_accuracy': float(cls_acc),
            'cv_precision': float(cls_prec),
            'cv_recall': float(cls_rec),
            'cv_f1': float(cls_f1),
            'cv_auc': float(cls_auc)
        }
    }

    meta_path = MODEL_DIR / "box_breakout_model_metadata.json"
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"  메타데이터: {meta_path}")

    # 최종 결과 출력
    print("\n" + "=" * 80)
    print("최종 모델 성능 요약")
    print("=" * 80)

    print("\n[회귀 모델] 10개 피처 + Ridge")
    print(f"  R² Score: {reg_r2:.4f}")
    print(f"  MAE: {reg_mae:.2f}% (평균 오차)")
    print(f"  파라미터: alpha={RIDGE_PARAMS['alpha']}")

    print("\n[분류 모델] 15개 피처 + XGBoost")
    print(f"  F1 Score: {cls_f1:.4f}")
    print(f"  AUC: {cls_auc:.4f}")
    print(f"  Precision: {cls_prec:.4f} (예측 정확도)")
    print(f"  Recall: {cls_rec:.4f} (실제 성공 포착률)")

    print("\n" + "=" * 80)
    print("저장된 파일")
    print("=" * 80)
    print(f"  1. {reg_path}")
    print(f"  2. {cls_path}")
    print(f"  3. {meta_path}")

    return reg_model, cls_model


if __name__ == "__main__":
    start_time = datetime.now()
    reg_model, cls_model = main()
    elapsed = datetime.now() - start_time
    print(f"\n완료! (소요시간: {elapsed})")
