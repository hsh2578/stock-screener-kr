"""
모델 비교: 3가지 피처셋 × 7개 모델

피처셋:
A. 14개 전부 (다중공선성 감수)
B. 상관관계 상위 8개
C. VIF 기준 8개 + volume_surge (9개)

모델:
1. Ridge
2. Lasso
3. Random Forest
4. XGBoost
5. LightGBM
6. Stacking (RF + XGB + LGBM + Ridge)
7. Voting (RF + XGB + LGBM)

평가:
- 회귀: MAE, RMSE, R²
- 분류: Accuracy, Precision, Recall, F1, AUC-ROC (10% 기준)
"""

import pandas as pd
import numpy as np
import warnings
from pathlib import Path
from datetime import datetime

from sklearn.model_selection import cross_val_predict, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, VotingRegressor, StackingRegressor
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

# XGBoost, LightGBM
try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("XGBoost not installed")

try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
    print("LightGBM not installed")

warnings.filterwarnings('ignore')

# 경로
ML_DIR = Path(__file__).parent.parent
DATA_PATH = ML_DIR / "data" / "training_data_v2.csv"
RESULT_PATH = ML_DIR / "data" / "model_comparison_results.csv"

# ============================================================================
# 피처셋 정의
# ============================================================================

# A. 14개 전부
FEATURES_A = [
    'breakout_pct', 'volume_surge', 'close_strength',
    'base_length', 'volatility_contraction', 'volume_dry_up',
    'ma200_slope', 'pct_above_52w_low',
    'rs_vs_market', 'market_return',
    'ma20_deviation', 'liquidity',
    'days_since_ath', 'atr_ratio'
]

# B. 상관관계 상위 8개
FEATURES_B = [
    'ma20_deviation', 'breakout_pct', 'rs_vs_market', 'pct_above_52w_low',
    'atr_ratio', 'volume_dry_up', 'volatility_contraction', 'volume_surge'
]

# C. VIF 기준 8개 + volume_surge
FEATURES_C = [
    'breakout_pct', 'base_length', 'ma200_slope', 'pct_above_52w_low',
    'rs_vs_market', 'market_return', 'days_since_ath', 'atr_ratio',
    'volume_surge'  # 추가
]

FEATURE_SETS = {
    'A_14features': FEATURES_A,
    'B_top8_corr': FEATURES_B,
    'C_vif8_vol': FEATURES_C
}


# ============================================================================
# 모델 정의
# ============================================================================

def get_models():
    """모델 딕셔너리 반환"""
    models = {
        'Ridge': Ridge(alpha=1.0),
        'Lasso': Lasso(alpha=0.1),
        'RandomForest': RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=10,
            random_state=42,
            n_jobs=-1
        ),
    }

    if HAS_XGB:
        models['XGBoost'] = XGBRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0
        )

    if HAS_LGBM:
        models['LightGBM'] = LGBMRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1
        )

    # Voting (평균)
    if HAS_XGB and HAS_LGBM:
        models['Voting'] = VotingRegressor(
            estimators=[
                ('rf', RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)),
                ('xgb', XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, verbosity=0)),
                ('lgbm', LGBMRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, verbose=-1))
            ],
            n_jobs=-1
        )

        # Stacking
        models['Stacking'] = StackingRegressor(
            estimators=[
                ('rf', RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)),
                ('xgb', XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, verbosity=0)),
                ('lgbm', LGBMRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, verbose=-1))
            ],
            final_estimator=Ridge(alpha=1.0),
            cv=3,
            n_jobs=-1
        )

    return models


# ============================================================================
# 평가 함수
# ============================================================================

def evaluate_regression(y_true, y_pred):
    """회귀 평가 지표"""
    return {
        'MAE': mean_absolute_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'R2': r2_score(y_true, y_pred)
    }


def evaluate_classification(y_true, y_pred, threshold=10):
    """분류 평가 지표 (예측값 >= threshold를 positive로)"""
    y_true_binary = (y_true >= threshold).astype(int)
    y_pred_binary = (y_pred >= threshold).astype(int)

    # AUC 계산 (예측값 그대로 사용)
    try:
        auc = roc_auc_score(y_true_binary, y_pred)
    except:
        auc = 0.5

    return {
        'Accuracy': accuracy_score(y_true_binary, y_pred_binary),
        'Precision': precision_score(y_true_binary, y_pred_binary, zero_division=0),
        'Recall': recall_score(y_true_binary, y_pred_binary, zero_division=0),
        'F1': f1_score(y_true_binary, y_pred_binary, zero_division=0),
        'AUC': auc
    }


# ============================================================================
# 메인 비교 함수
# ============================================================================

def run_comparison():
    """모델 비교 실행"""
    print("=" * 70)
    print("모델 비교: 3가지 피처셋 x 7개 모델")
    print("=" * 70)

    # 데이터 로드
    print("\n[1/3] 데이터 로드...")
    df = pd.read_csv(DATA_PATH)
    print(f"  샘플 수: {len(df)}")

    y = df['max_gain_20d'].values

    # 결과 저장용
    results = []

    # 모델 준비
    models = get_models()
    print(f"  모델 수: {len(models)}")
    print(f"  모델: {', '.join(models.keys())}")

    # 5-Fold CV
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    # 피처셋별 모델별 평가
    print("\n[2/3] 모델 학습 및 평가...")
    total_combinations = len(FEATURE_SETS) * len(models)
    current = 0

    for feat_name, features in FEATURE_SETS.items():
        print(f"\n  === 피처셋: {feat_name} ({len(features)}개 피처) ===")

        # 피처 추출 및 스케일링
        X = df[features].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        for model_name, model in models.items():
            current += 1
            print(f"    [{current}/{total_combinations}] {model_name}...", end=" ")

            try:
                # Cross-validation 예측
                y_pred = cross_val_predict(model, X_scaled, y, cv=kfold, n_jobs=-1)

                # 회귀 평가
                reg_metrics = evaluate_regression(y, y_pred)

                # 분류 평가 (10% 기준)
                cls_metrics = evaluate_classification(y, y_pred, threshold=10)

                # 결과 저장
                result = {
                    'feature_set': feat_name,
                    'n_features': len(features),
                    'model': model_name,
                    **reg_metrics,
                    **cls_metrics
                }
                results.append(result)

                print(f"R2={reg_metrics['R2']:.4f}, F1={cls_metrics['F1']:.4f}")

            except Exception as e:
                print(f"ERROR: {e}")
                continue

    # 결과 정리
    print("\n[3/3] 결과 정리...")
    results_df = pd.DataFrame(results)

    # R2 기준 정렬
    results_df = results_df.sort_values('R2', ascending=False)

    # 저장
    results_df.to_csv(RESULT_PATH, index=False, encoding='utf-8-sig')
    print(f"\n결과 저장: {RESULT_PATH}")

    return results_df


def print_results(results_df):
    """결과 출력"""
    print("\n" + "=" * 70)
    print("회귀 성능 (R2 기준 정렬)")
    print("=" * 70)
    print(f"{'피처셋':<15} {'모델':<12} {'MAE':>8} {'RMSE':>8} {'R2':>8}")
    print("-" * 70)

    for _, row in results_df.head(10).iterrows():
        print(f"{row['feature_set']:<15} {row['model']:<12} "
              f"{row['MAE']:>8.2f} {row['RMSE']:>8.2f} {row['R2']:>8.4f}")

    print("\n" + "=" * 70)
    print("분류 성능 (F1 기준 정렬)")
    print("=" * 70)
    results_f1 = results_df.sort_values('F1', ascending=False)
    print(f"{'피처셋':<15} {'모델':<12} {'Acc':>7} {'Prec':>7} {'Recall':>7} {'F1':>7} {'AUC':>7}")
    print("-" * 70)

    for _, row in results_f1.head(10).iterrows():
        print(f"{row['feature_set']:<15} {row['model']:<12} "
              f"{row['Accuracy']:>7.4f} {row['Precision']:>7.4f} "
              f"{row['Recall']:>7.4f} {row['F1']:>7.4f} {row['AUC']:>7.4f}")

    # 최고 성능 모델
    print("\n" + "=" * 70)
    print("최고 성능 모델")
    print("=" * 70)

    best_r2 = results_df.iloc[0]
    best_f1 = results_f1.iloc[0]

    print(f"\n[R2 기준 최고]")
    print(f"  피처셋: {best_r2['feature_set']}")
    print(f"  모델: {best_r2['model']}")
    print(f"  R2: {best_r2['R2']:.4f}, MAE: {best_r2['MAE']:.2f}")

    print(f"\n[F1 기준 최고]")
    print(f"  피처셋: {best_f1['feature_set']}")
    print(f"  모델: {best_f1['model']}")
    print(f"  F1: {best_f1['F1']:.4f}, Precision: {best_f1['Precision']:.4f}, Recall: {best_f1['Recall']:.4f}")

    # 피처셋별 평균 성능
    print("\n" + "=" * 70)
    print("피처셋별 평균 성능")
    print("=" * 70)
    feat_avg = results_df.groupby('feature_set')[['R2', 'F1', 'Precision']].mean()
    print(feat_avg.round(4))

    # 모델별 평균 성능
    print("\n" + "=" * 70)
    print("모델별 평균 성능")
    print("=" * 70)
    model_avg = results_df.groupby('model')[['R2', 'F1', 'Precision']].mean()
    model_avg = model_avg.sort_values('R2', ascending=False)
    print(model_avg.round(4))


def main():
    start_time = datetime.now()

    results_df = run_comparison()
    print_results(results_df)

    elapsed = datetime.now() - start_time
    print(f"\n완료! (소요시간: {elapsed})")


if __name__ == "__main__":
    main()
