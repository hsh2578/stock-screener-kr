"""
최종 모델 학습 및 저장

1. 회귀 모델: 20일 내 최고 수익률 예측 (LightGBM Regressor)
2. 분류 모델: 15% 이상 상승 확률 예측 (LightGBM Classifier)
"""

import pandas as pd
import numpy as np
import pickle
import warnings
from pathlib import Path
from datetime import datetime

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict, StratifiedKFold, KFold
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
from lightgbm import LGBMClassifier, LGBMRegressor

warnings.filterwarnings('ignore')

# 경로
ML_DIR = Path(__file__).parent.parent
DATA_PATH = ML_DIR / "data" / "training_data_v2.csv"
MODEL_PATH = ML_DIR / "models" / "52w_high"
MODEL_PATH.mkdir(exist_ok=True)

# 피처셋 A (14개)
FEATURES = [
    'breakout_pct', 'volume_surge', 'close_strength',
    'base_length', 'volatility_contraction', 'volume_dry_up',
    'ma200_slope', 'pct_above_52w_low',
    'rs_vs_market', 'market_return',
    'ma20_deviation', 'liquidity',
    'days_since_ath', 'atr_ratio'
]

# 회귀 모델 파라미터 (튜닝 결과)
REGRESSOR_PARAMS = {
    'n_estimators': 200,
    'max_depth': 3,
    'learning_rate': 0.01,
    'subsample': 0.8,
    'reg_alpha': 1.0,
    'reg_lambda': 5.0,
    'random_state': 42,
    'verbose': -1,
    'n_jobs': -1
}

# 분류 모델 파라미터
CLASSIFIER_PARAMS = {
    'n_estimators': 200,
    'max_depth': 5,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'verbose': -1,
    'n_jobs': -1
}


def train_regression_model(X_scaled, y_reg):
    """회귀 모델 학습: 최고 수익률 예측"""
    print("\n  === 회귀 모델 (최고 수익률 예측) ===")

    model = LGBMRegressor(**REGRESSOR_PARAMS)

    # Cross-validation
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    y_pred = cross_val_predict(model, X_scaled, y_reg, cv=kfold)

    # 성능
    mae = mean_absolute_error(y_reg, y_pred)
    rmse = np.sqrt(mean_squared_error(y_reg, y_pred))
    r2 = r2_score(y_reg, y_pred)

    print(f"  MAE:  {mae:.2f}%")
    print(f"  RMSE: {rmse:.2f}%")
    print(f"  R2:   {r2:.4f} ({r2*100:.1f}% 설명력)")

    # 전체 데이터로 학습
    model.fit(X_scaled, y_reg)

    return model, {'mae': mae, 'rmse': rmse, 'r2': r2}


def train_classification_model(X_scaled, y_cls):
    """분류 모델 학습: 15%+ 확률 예측"""
    print("\n  === 분류 모델 (15%+ 확률 예측) ===")

    model = LGBMClassifier(**CLASSIFIER_PARAMS)

    # Cross-validation
    skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred = cross_val_predict(model, X_scaled, y_cls, cv=skfold)
    y_proba = cross_val_predict(model, X_scaled, y_cls, cv=skfold, method='predict_proba')[:, 1]

    # 성능
    acc = accuracy_score(y_cls, y_pred)
    prec = precision_score(y_cls, y_pred)
    rec = recall_score(y_cls, y_pred)
    f1 = f1_score(y_cls, y_pred)
    auc = roc_auc_score(y_cls, y_proba)

    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f} (예측 중 실제 성공 비율)")
    print(f"  Recall:    {rec:.4f} (실제 성공 중 포착 비율)")
    print(f"  F1:        {f1:.4f}")
    print(f"  AUC:       {auc:.4f}")

    # Confusion Matrix
    tn, fp, fn, tp = confusion_matrix(y_cls, y_pred).ravel()
    print(f"\n  Confusion Matrix:")
    print(f"    TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}")

    # 전체 데이터로 학습
    model.fit(X_scaled, y_cls)

    return model, {
        'accuracy': acc, 'precision': prec, 'recall': rec,
        'f1': f1, 'auc': auc
    }


def main():
    print("=" * 60)
    print("최종 모델 학습 (회귀 + 분류)")
    print("=" * 60)

    # 데이터 로드
    print("\n[1/5] 데이터 로드...")
    df = pd.read_csv(DATA_PATH)

    X = df[FEATURES].values
    y_reg = df['max_gain_20d'].values  # 회귀용: 실제 수익률
    y_cls = (df['max_gain_20d'] >= 15).astype(int).values  # 분류용: 15%+ 여부

    print(f"  샘플 수: {len(df)}")
    print(f"  성공(15%+): {y_cls.sum()} ({y_cls.mean()*100:.1f}%)")
    print(f"  평균 수익률: {y_reg.mean():.2f}%")
    print(f"  피처 수: {len(FEATURES)}")

    # 스케일러 학습
    print("\n[2/5] 스케일러 학습...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 회귀 모델 학습
    print("\n[3/5] 회귀 모델 학습...")
    reg_model, reg_perf = train_regression_model(X_scaled, y_reg)

    # 분류 모델 학습
    print("\n[4/5] 분류 모델 학습...")
    cls_model, cls_perf = train_classification_model(X_scaled, y_cls)

    # 피처 중요도
    print("\n  === 피처 중요도 (분류 모델 기준) ===")
    importance = pd.DataFrame({
        'feature': FEATURES,
        'importance': cls_model.feature_importances_
    }).sort_values('importance', ascending=False)

    for _, row in importance.head(10).iterrows():
        bar = "#" * int(row['importance'] / importance['importance'].max() * 30)
        print(f"  {row['feature']:25}: {row['importance']:>6.0f} {bar}")

    # 저장
    print("\n[5/5] 모델 저장...")

    # 회귀 모델 저장
    reg_file = MODEL_PATH / "regressor.pkl"
    with open(reg_file, 'wb') as f:
        pickle.dump(reg_model, f)
    print(f"  회귀 모델: {reg_file}")

    # 분류 모델 저장
    cls_file = MODEL_PATH / "classifier.pkl"
    with open(cls_file, 'wb') as f:
        pickle.dump(cls_model, f)
    print(f"  분류 모델: {cls_file}")

    # 스케일러 저장
    scaler_file = MODEL_PATH / "scaler.pkl"
    with open(scaler_file, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"  스케일러: {scaler_file}")

    # 메타데이터 저장
    meta = {
        'features': FEATURES,
        'regressor_params': REGRESSOR_PARAMS,
        'classifier_params': CLASSIFIER_PARAMS,
        'regressor_performance': reg_perf,
        'classifier_performance': cls_perf,
        'trained_at': datetime.now().isoformat(),
        'samples': len(df),
        'positive_rate': float(y_cls.mean()),
        'avg_gain': float(y_reg.mean())
    }

    meta_file = MODEL_PATH / "model_meta.pkl"
    with open(meta_file, 'wb') as f:
        pickle.dump(meta, f)
    print(f"  메타데이터: {meta_file}")

    # 요약
    print("\n" + "=" * 60)
    print("최종 모델 요약")
    print("=" * 60)
    print(f"""
[회귀 모델] 20일 내 최고 수익률 예측
  - MAE: {reg_perf['mae']:.2f}% (평균 오차)
  - R2: {reg_perf['r2']:.4f} ({reg_perf['r2']*100:.1f}% 설명력)
  - 용도: predicted_gain 칼럼

[분류 모델] 15% 이상 상승 확률 예측
  - Precision: {cls_perf['precision']*100:.1f}% (예측 적중률)
  - AUC: {cls_perf['auc']:.4f}
  - 용도: success_probability 칼럼

[웹사이트 표시 예시]
  종목명 | 예상 수익률 | 성공 확률
  삼성전자 | +15.2%     | 62%
""")

    return reg_model, cls_model, scaler, meta


if __name__ == "__main__":
    start_time = datetime.now()
    reg_model, cls_model, scaler, meta = main()
    elapsed = datetime.now() - start_time
    print(f"완료! (소요시간: {elapsed})")
