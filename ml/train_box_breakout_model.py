"""
박스권 돌파 최종 모델 학습 및 저장

모델 비교 결과 기반:
1. 회귀 모델: 20일 내 최고 수익률 예측 (Lasso - R²=0.0677)
2. 분류 모델: 15% 이상 상승 확률 예측 (XGBoost - F1=0.2821)
"""

import pandas as pd
import numpy as np
import pickle
import warnings
from pathlib import Path
from datetime import datetime

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict, StratifiedKFold, KFold
from sklearn.linear_model import Lasso
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)

# XGBoost
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("[ERROR] XGBoost not installed. Please install: pip install xgboost")
    exit(1)

warnings.filterwarnings('ignore')

# 경로
BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "data" / "box_breakout_history.csv"
MODEL_PATH = BASE_DIR / "models" / "box_breakout"
MODEL_PATH.mkdir(parents=True, exist_ok=True)

# 피처셋 A (15개 전부 - 최고 성능)
FEATURES = [
    'box_range_pct', 'breakout_strength', 'volume_surge', 'volume_dry_up',
    'close_strength', 'volatility_contraction', 'ma20_deviation',
    'breakout_gap', 'ma200_slope', 'pct_above_52w_low', 'days_since_ath',
    'market_return', 'rs_vs_market', 'atr_ratio', 'liquidity'
]

# 회귀 모델 파라미터 (Lasso)
REGRESSOR_PARAMS = {
    'alpha': 0.1,
    'max_iter': 5000,
    'random_state': 42
}

# 분류 모델 파라미터 (XGBoost)
CLASSIFIER_PARAMS = {
    'n_estimators': 100,
    'max_depth': 5,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'eval_metric': 'logloss',
    'verbosity': 0
}


def train_regression_model(X_scaled, y_reg):
    """회귀 모델 학습: 최고 수익률 예측"""
    print("\n  === 회귀 모델 (최고 수익률 예측) - Lasso ===")

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
    print(f"  R2:   {r2:.4f} ({r2*100:.2f}% 설명력)")

    # 전체 데이터로 학습
    model.fit(X_scaled, y_reg)

    # Lasso 계수 확인 (0이 아닌 피처)
    non_zero_features = np.sum(model.coef_ != 0)
    print(f"  선택된 피처: {non_zero_features}/{len(FEATURES)}개")

    return model, {'mae': mae, 'rmse': rmse, 'r2': r2}


def train_classification_model(X_scaled, y_cls):
    """분류 모델 학습: 15%+ 확률 예측"""
    print("\n  === 분류 모델 (15%+ 확률 예측) - XGBoost ===")

    model = XGBClassifier(**CLASSIFIER_PARAMS)

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
    print(f"    TP: {tp:5,}, FP: {fp:5,}")
    print(f"    FN: {fn:5,}, TN: {tn:5,}")

    # 전체 데이터로 학습
    model.fit(X_scaled, y_cls)

    return model, {
        'accuracy': acc, 'precision': prec, 'recall': rec,
        'f1': f1, 'auc': auc
    }


def main():
    print("=" * 80)
    print("박스권 돌파 최종 모델 학습 (회귀: Lasso + 분류: XGBoost)")
    print("=" * 80)

    # 데이터 로드
    print("\n[1/5] 데이터 로드...")
    df = pd.read_csv(DATA_PATH, encoding='utf-8-sig')

    X = df[FEATURES].values
    y_reg = df['max_gain_20d'].values  # 회귀용: 실제 수익률
    y_cls = df['success'].values  # 분류용: 15%+ 여부

    print(f"  샘플 수: {len(df):,}")
    print(f"  성공(15%+): {y_cls.sum():,} ({y_cls.mean()*100:.2f}%)")
    print(f"  평균 수익률: {y_reg.mean():.2f}%")
    print(f"  중앙값 수익률: {np.median(y_reg):.2f}%")
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

    for _, row in importance.head(15).iterrows():
        bar = "#" * int(row['importance'] / importance['importance'].max() * 40)
        print(f"  {row['feature']:25}: {row['importance']:>8.0f} {bar}")

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
        'regressor_type': 'Lasso',
        'classifier_type': 'XGBoost',
        'regressor_params': REGRESSOR_PARAMS,
        'classifier_params': CLASSIFIER_PARAMS,
        'regressor_performance': reg_perf,
        'classifier_performance': cls_perf,
        'trained_at': datetime.now().isoformat(),
        'samples': len(df),
        'success_rate': float(y_cls.mean()),
        'avg_gain': float(y_reg.mean()),
        'median_gain': float(np.median(y_reg)),
        'threshold': 15.0  # 성공 기준
    }

    meta_file = MODEL_PATH / "model_meta.pkl"
    with open(meta_file, 'wb') as f:
        pickle.dump(meta, f)
    print(f"  메타데이터: {meta_file}")

    # 요약
    print("\n" + "=" * 80)
    print("최종 모델 요약")
    print("=" * 80)
    print(f"""
[회귀 모델] 20일 내 최고 수익률 예측 (Lasso)
  - MAE: {reg_perf['mae']:.2f}% (평균 오차)
  - RMSE: {reg_perf['rmse']:.2f}%
  - R2: {reg_perf['r2']:.4f} ({reg_perf['r2']*100:.2f}% 설명력)
  - 용도: predicted_gain 칼럼

[분류 모델] 15% 이상 상승 확률 예측 (XGBoost)
  - Precision: {cls_perf['precision']*100:.2f}% (예측 적중률)
  - Recall: {cls_perf['recall']*100:.2f}% (성공 포착률)
  - F1: {cls_perf['f1']:.4f}
  - AUC: {cls_perf['auc']:.4f}
  - 용도: success_probability 칼럼

[웹사이트 표시 예시]
  종목명     | 현재가  | 박스권      | 예상 수익률 | 성공 확률
  삼성전자   | 82,400  | 71,000~80,800 | +12.5%     | 45%
  SK하이닉스 | 150,000 | 125,600~142,800 | +18.3%     | 58%

[모델 저장 위치]
  {MODEL_PATH}
  - regressor.pkl (Lasso 회귀 모델)
  - classifier.pkl (XGBoost 분류 모델)
  - scaler.pkl (StandardScaler)
  - model_meta.pkl (메타데이터)
""")

    return reg_model, cls_model, scaler, meta


if __name__ == "__main__":
    start_time = datetime.now()
    reg_model, cls_model, scaler, meta = main()
    elapsed = datetime.now() - start_time
    print(f"완료! (소요시간: {elapsed})")
