"""
박스권 돌파 ML 모델 비교

피처셋:
A. 15개 전부
B. 상관관계 상위 10개
C. VIF 기준 선택

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
- 분류: Accuracy, Precision, Recall, F1, AUC-ROC (15% 기준)
"""

import pandas as pd
import numpy as np
import warnings
from pathlib import Path
from datetime import datetime

from sklearn.model_selection import cross_val_predict, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, VotingRegressor, VotingClassifier, StackingRegressor, StackingClassifier
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

# XGBoost, LightGBM
try:
    from xgboost import XGBRegressor, XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("[WARN] XGBoost not installed")

try:
    from lightgbm import LGBMRegressor, LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
    print("[WARN] LightGBM not installed")

warnings.filterwarnings('ignore')

# 경로
BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "data" / "box_breakout_history.csv"
RESULT_PATH = BASE_DIR / "data" / "box_breakout_model_comparison.csv"

# ============================================================================
# 피처셋 정의
# ============================================================================

# A. 15개 전부
FEATURES_A = [
    'box_range_pct', 'breakout_strength', 'volume_surge', 'volume_dry_up',
    'close_strength', 'volatility_contraction', 'ma20_deviation',
    'breakout_gap', 'ma200_slope', 'pct_above_52w_low', 'days_since_ath',
    'market_return', 'rs_vs_market', 'atr_ratio', 'liquidity'
]

# B. 상관관계 상위 10개 (데이터 분석 결과 기반)
FEATURES_B = [
    'ma200_slope', 'breakout_gap', 'volume_surge', 'breakout_strength',
    'pct_above_52w_low', 'ma20_deviation', 'rs_vs_market', 'market_return',
    'atr_ratio', 'volatility_contraction'
]

# C. VIF 기준 선택 (다중공선성 제거)
# ma20_deviation, pct_above_52w_low, rs_vs_market는 상관관계가 높으므로 하나만 선택
FEATURES_C = [
    'breakout_strength', 'volume_surge', 'close_strength',
    'box_range_pct', 'ma200_slope', 'breakout_gap',
    'days_since_ath', 'atr_ratio', 'liquidity',
    'volatility_contraction'  # 10개
]

FEATURE_SETS = {
    'A_15features': FEATURES_A,
    'B_top10_corr': FEATURES_B,
    'C_vif10': FEATURES_C
}


# ============================================================================
# 모델 정의
# ============================================================================

def get_regression_models():
    """회귀 모델 딕셔너리 반환"""
    models = {
        'Ridge': Ridge(alpha=1.0),
        'Lasso': Lasso(alpha=0.1, max_iter=5000),
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


def get_classification_models():
    """분류 모델 딕셔너리 반환"""
    from sklearn.linear_model import LogisticRegression

    models = {
        'LogisticRidge': LogisticRegression(penalty='l2', C=1.0, max_iter=5000, random_state=42),
        'LogisticLasso': LogisticRegression(penalty='l1', C=1.0, solver='saga', max_iter=5000, random_state=42),
        'RandomForest': RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=10,
            random_state=42,
            n_jobs=-1
        ),
    }

    if HAS_XGB:
        models['XGBoost'] = XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0,
            eval_metric='logloss'
        )

    if HAS_LGBM:
        models['LightGBM'] = LGBMClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1
        )

    # Voting
    if HAS_XGB and HAS_LGBM:
        models['Voting'] = VotingClassifier(
            estimators=[
                ('rf', RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)),
                ('xgb', XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, verbosity=0, eval_metric='logloss')),
                ('lgbm', LGBMClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, verbose=-1))
            ],
            voting='soft',
            n_jobs=-1
        )

        # Stacking
        models['Stacking'] = StackingClassifier(
            estimators=[
                ('rf', RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)),
                ('xgb', XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, verbosity=0, eval_metric='logloss')),
                ('lgbm', LGBMClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, verbose=-1))
            ],
            final_estimator=LogisticRegression(max_iter=5000, random_state=42),
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


def evaluate_classification(y_true_binary, y_pred_proba):
    """분류 평가 지표"""
    y_pred_binary = (y_pred_proba >= 0.5).astype(int)

    # AUC 계산
    try:
        auc = roc_auc_score(y_true_binary, y_pred_proba)
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
    print("=" * 80)
    print("박스권 돌파 ML 모델 비교: 3가지 피처셋 x 7개 모델")
    print("=" * 80)

    # 데이터 로드
    print("\n[1/4] 데이터 로드...")
    df = pd.read_csv(DATA_PATH, encoding='utf-8-sig')
    print(f"  샘플 수: {len(df):,}")
    print(f"  성공률: {df['success'].mean()*100:.2f}%")

    y_regression = df['max_gain_20d'].values
    y_classification = df['success'].values

    # 결과 저장용
    results = []

    # 5-Fold CV
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    # ========================================================================
    # 회귀 모델 비교
    # ========================================================================
    print("\n[2/4] 회귀 모델 학습 및 평가...")
    reg_models = get_regression_models()
    print(f"  모델 수: {len(reg_models)}")

    total_combinations = len(FEATURE_SETS) * len(reg_models)
    current = 0

    for feat_name, features in FEATURE_SETS.items():
        print(f"\n  === 피처셋: {feat_name} ({len(features)}개 피처) ===")

        # 피처 추출 및 스케일링
        X = df[features].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        for model_name, model in reg_models.items():
            current += 1
            print(f"    [{current}/{total_combinations}] {model_name}...", end=" ")

            try:
                # Cross-validation 예측
                y_pred = cross_val_predict(model, X_scaled, y_regression, cv=kfold, n_jobs=-1)

                # 회귀 평가
                reg_metrics = evaluate_regression(y_regression, y_pred)

                # 결과 저장
                result = {
                    'task': 'regression',
                    'feature_set': feat_name,
                    'n_features': len(features),
                    'model': model_name,
                    **reg_metrics,
                    'Accuracy': None,
                    'Precision': None,
                    'Recall': None,
                    'F1': None,
                    'AUC': None
                }
                results.append(result)

                print(f"R2={reg_metrics['R2']:.4f}, MAE={reg_metrics['MAE']:.2f}")

            except Exception as e:
                print(f"ERROR: {e}")
                continue

    # ========================================================================
    # 분류 모델 비교
    # ========================================================================
    print("\n[3/4] 분류 모델 학습 및 평가...")
    cls_models = get_classification_models()
    print(f"  모델 수: {len(cls_models)}")

    total_combinations = len(FEATURE_SETS) * len(cls_models)
    current = 0

    for feat_name, features in FEATURE_SETS.items():
        print(f"\n  === 피처셋: {feat_name} ({len(features)}개 피처) ===")

        # 피처 추출 및 스케일링
        X = df[features].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        for model_name, model in cls_models.items():
            current += 1
            print(f"    [{current}/{total_combinations}] {model_name}...", end=" ")

            try:
                # Cross-validation 예측 (확률)
                y_pred_proba = cross_val_predict(model, X_scaled, y_classification, cv=kfold, method='predict_proba', n_jobs=-1)[:, 1]

                # 분류 평가
                cls_metrics = evaluate_classification(y_classification, y_pred_proba)

                # 결과 저장
                result = {
                    'task': 'classification',
                    'feature_set': feat_name,
                    'n_features': len(features),
                    'model': model_name,
                    'MAE': None,
                    'RMSE': None,
                    'R2': None,
                    **cls_metrics
                }
                results.append(result)

                print(f"F1={cls_metrics['F1']:.4f}, AUC={cls_metrics['AUC']:.4f}")

            except Exception as e:
                print(f"ERROR: {e}")
                continue

    # 결과 정리
    print("\n[4/4] 결과 정리...")
    results_df = pd.DataFrame(results)

    # 저장
    results_df.to_csv(RESULT_PATH, index=False, encoding='utf-8-sig')
    print(f"\n결과 저장: {RESULT_PATH}")

    return results_df


def print_results(results_df):
    """결과 출력"""
    # 회귀 결과
    reg_results = results_df[results_df['task'] == 'regression'].copy()
    reg_results = reg_results.sort_values('R2', ascending=False)

    print("\n" + "=" * 80)
    print("회귀 성능 (R2 기준 상위 10개)")
    print("=" * 80)
    print(f"{'피처셋':<18} {'모델':<15} {'MAE':>8} {'RMSE':>8} {'R2':>8}")
    print("-" * 80)

    for _, row in reg_results.head(10).iterrows():
        print(f"{row['feature_set']:<18} {row['model']:<15} "
              f"{row['MAE']:>8.2f} {row['RMSE']:>8.2f} {row['R2']:>8.4f}")

    # 분류 결과
    cls_results = results_df[results_df['task'] == 'classification'].copy()
    cls_results = cls_results.sort_values('F1', ascending=False)

    print("\n" + "=" * 80)
    print("분류 성능 (F1 기준 상위 10개)")
    print("=" * 80)
    print(f"{'피처셋':<18} {'모델':<15} {'Precision':>10} {'Recall':>10} {'F1':>8} {'AUC':>8}")
    print("-" * 80)

    for _, row in cls_results.head(10).iterrows():
        print(f"{row['feature_set']:<18} {row['model']:<15} "
              f"{row['Precision']:>10.4f} {row['Recall']:>10.4f} "
              f"{row['F1']:>8.4f} {row['AUC']:>8.4f}")

    # 최고 성능 모델
    print("\n" + "=" * 80)
    print("최고 성능 모델")
    print("=" * 80)

    best_r2 = reg_results.iloc[0]
    best_f1 = cls_results.iloc[0]

    print(f"\n[회귀 최고]")
    print(f"  피처셋: {best_r2['feature_set']}")
    print(f"  모델: {best_r2['model']}")
    print(f"  R2: {best_r2['R2']:.4f}, MAE: {best_r2['MAE']:.2f}, RMSE: {best_r2['RMSE']:.2f}")

    print(f"\n[분류 최고]")
    print(f"  피처셋: {best_f1['feature_set']}")
    print(f"  모델: {best_f1['model']}")
    print(f"  F1: {best_f1['F1']:.4f}, Precision: {best_f1['Precision']:.4f}, Recall: {best_f1['Recall']:.4f}, AUC: {best_f1['AUC']:.4f}")

    # 피처셋별 평균 성능
    print("\n" + "=" * 80)
    print("피처셋별 평균 성능")
    print("=" * 80)

    print("\n[회귀]")
    reg_feat_avg = reg_results.groupby('feature_set')[['R2', 'MAE', 'RMSE']].mean()
    print(reg_feat_avg.round(4))

    print("\n[분류]")
    cls_feat_avg = cls_results.groupby('feature_set')[['F1', 'Precision', 'Recall', 'AUC']].mean()
    print(cls_feat_avg.round(4))

    # 모델별 평균 성능
    print("\n" + "=" * 80)
    print("모델별 평균 성능")
    print("=" * 80)

    print("\n[회귀]")
    reg_model_avg = reg_results.groupby('model')[['R2', 'MAE', 'RMSE']].mean()
    reg_model_avg = reg_model_avg.sort_values('R2', ascending=False)
    print(reg_model_avg.round(4))

    print("\n[분류]")
    cls_model_avg = cls_results.groupby('model')[['F1', 'Precision', 'Recall', 'AUC']].mean()
    cls_model_avg = cls_model_avg.sort_values('F1', ascending=False)
    print(cls_model_avg.round(4))


def main():
    start_time = datetime.now()

    results_df = run_comparison()
    print_results(results_df)

    elapsed = datetime.now() - start_time
    print(f"\n완료! (소요시간: {elapsed})")


if __name__ == "__main__":
    main()
