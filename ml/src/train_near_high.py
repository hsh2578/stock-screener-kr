"""
52주 신고가 근접 ML 모델 학습

1. 회귀 모델: 20거래일 내 최고 수익률 예측 (Lasso Regression)
2. 분류 모델: 10거래일 내 52주 고가 돌파 확률 예측 (Logistic Regression)

모델 선정 근거 (near_high_model_comparison.py 결과):
  - 회귀: Lasso(MAE=12.30%) > Ridge > RF > LightGBM > XGBoost
  - 분류: LogisticRegression(AUC=0.7352) > RF > XGBoost > LightGBM > GB
  - 단순 선형 모델이 트리 모델을 이김 (피처 간 선형 관계 + 노이즈 환경)
"""

import pandas as pd
import numpy as np
import pickle
import warnings
from pathlib import Path
from datetime import datetime

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict, StratifiedKFold, KFold
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)

warnings.filterwarnings('ignore')

# 경로
ML_DIR = Path(__file__).parent.parent
DATA_PATH = ML_DIR / "data" / "near_high_training_data.csv"
MODEL_PATH = ML_DIR / "models" / "near_high_52w"
MODEL_PATH.mkdir(parents=True, exist_ok=True)

# 피처 (13개)
FEATURES = [
    'gap_pct', 'resistance_tests', 'days_since_high',
    'momentum_5d', 'volume_ratio', 'volume_trend',
    'volatility_contraction', 'atr_pct',
    'ma_alignment', 'trend_strength',
    'rs_vs_market', 'market_return',
    'pct_from_52w_low'
]

# 회귀 모델 파라미터 (Lasso, 튜닝 결과)
REGRESSOR_PARAMS = {
    'alpha': 0.05,
    'max_iter': 5000,
    'random_state': 42,
}

# 분류 모델 파라미터 (LogisticRegression, 튜닝 결과)
CLASSIFIER_PARAMS = {
    'C': 0.01,
    'penalty': 'l2',
    'solver': 'saga',
    'max_iter': 1000,
    'random_state': 42,
    'n_jobs': -1,
}


def train_regression_model(X_scaled, y_reg):
    """회귀 모델 학습: 20거래일 내 최고 수익률 예측 (Lasso)"""
    print("\n  === 회귀 모델: Lasso (20일 최고 수익률 예측) ===")

    model = Lasso(**REGRESSOR_PARAMS)

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
    """분류 모델 학습: 10거래일 내 돌파 확률 예측 (LogisticRegression)"""
    print("\n  === 분류 모델: LogisticRegression (10일 내 돌파 확률 예측) ===")

    model = LogisticRegression(**CLASSIFIER_PARAMS)

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
    print(f"  Precision: {prec:.4f} (예측 돌파 중 실제 돌파 비율)")
    print(f"  Recall:    {rec:.4f} (실제 돌파 중 포착 비율)")
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
    print("52주 신고가 근접 ML 모델 학습 (Lasso + LogisticRegression)")
    print("=" * 60)

    # 데이터 로드
    print("\n[1/5] 데이터 로드...")
    df = pd.read_csv(DATA_PATH)
    print(f"  원본 샘플 수: {len(df)}")

    # 데이터 품질 필터링 (atr_pct >= 50 제거 - 저유동성 데이터 오류)
    bad_mask = df['atr_pct'] >= 50
    if bad_mask.sum() > 0:
        print(f"  atr_pct >= 50 제거: {bad_mask.sum()}건")
        df = df[~bad_mask].reset_index(drop=True)

    X = df[FEATURES].values
    y_reg = df['max_gain_20d'].values              # 회귀용: 20일 최고 수익률
    y_cls = df['broke_through_10d'].astype(int).values  # 분류용: 10일 내 돌파 여부

    print(f"  최종 샘플 수: {len(df)}")
    print(f"  돌파 성공: {y_cls.sum()} ({y_cls.mean()*100:.1f}%)")
    print(f"  돌파 실패: {len(df) - y_cls.sum()} ({(1-y_cls.mean())*100:.1f}%)")
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

    # 피처 계수 (선형 모델의 feature importance)
    print("\n  === 피처 계수 (절대값 기준) ===")

    print("\n  [분류 모델 - LogisticRegression]")
    cls_coefs = pd.DataFrame({
        'feature': FEATURES,
        'coefficient': cls_model.coef_[0]
    })
    cls_coefs['abs_coef'] = cls_coefs['coefficient'].abs()
    cls_coefs = cls_coefs.sort_values('abs_coef', ascending=False)

    for _, row in cls_coefs.iterrows():
        bar = "#" * int(row['abs_coef'] / cls_coefs['abs_coef'].max() * 30)
        sign = "+" if row['coefficient'] > 0 else "-"
        print(f"  {row['feature']:25}: {sign}{row['abs_coef']:>6.3f} {bar}")

    print("\n  [회귀 모델 - Lasso]")
    reg_coefs = pd.DataFrame({
        'feature': FEATURES,
        'coefficient': reg_model.coef_
    })
    reg_coefs['abs_coef'] = reg_coefs['coefficient'].abs()
    reg_coefs = reg_coefs.sort_values('abs_coef', ascending=False)

    for _, row in reg_coefs.iterrows():
        bar = "#" * int(row['abs_coef'] / max(reg_coefs['abs_coef'].max(), 0.001) * 30)
        sign = "+" if row['coefficient'] > 0 else "-"
        print(f"  {row['feature']:25}: {sign}{row['abs_coef']:>6.3f} {bar}")

    # 저장
    print("\n[5/5] 모델 저장...")

    # 회귀 모델
    reg_file = MODEL_PATH / "regressor.pkl"
    with open(reg_file, 'wb') as f:
        pickle.dump(reg_model, f)
    print(f"  회귀 모델 (Lasso): {reg_file}")

    # 분류 모델
    cls_file = MODEL_PATH / "classifier.pkl"
    with open(cls_file, 'wb') as f:
        pickle.dump(cls_model, f)
    print(f"  분류 모델 (LogisticRegression): {cls_file}")

    # 스케일러
    scaler_file = MODEL_PATH / "scaler.pkl"
    with open(scaler_file, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"  스케일러: {scaler_file}")

    # 메타데이터
    meta = {
        'features': FEATURES,
        'regressor_type': 'Lasso',
        'classifier_type': 'LogisticRegression',
        'regressor_params': REGRESSOR_PARAMS,
        'classifier_params': CLASSIFIER_PARAMS,
        'regressor_performance': reg_perf,
        'classifier_performance': cls_perf,
        'trained_at': datetime.now().isoformat(),
        'samples': len(df),
        'positive_rate': float(y_cls.mean()),
        'avg_gain': float(y_reg.mean()),
        'classification_target': 'broke_through_10d',
        'regression_target': 'max_gain_20d',
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
[회귀 모델] Lasso (alpha={REGRESSOR_PARAMS['alpha']})
  - MAE: {reg_perf['mae']:.2f}% (평균 오차)
  - R2: {reg_perf['r2']:.4f} ({reg_perf['r2']*100:.1f}% 설명력)
  - 용도: predicted_gain

[분류 모델] LogisticRegression (C={CLASSIFIER_PARAMS['C']})
  - Precision: {cls_perf['precision']*100:.1f}% (예측 돌파 중 실제 돌파율)
  - AUC: {cls_perf['auc']:.4f}
  - 용도: breakout_probability
""")

    return reg_model, cls_model, scaler, meta


if __name__ == "__main__":
    start_time = datetime.now()
    reg_model, cls_model, scaler, meta = main()
    elapsed = datetime.now() - start_time
    print(f"완료! (소요시간: {elapsed})")
