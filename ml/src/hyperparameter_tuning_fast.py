"""
빠른 하이퍼파라미터 튜닝 (축소된 그리드)
"""

import pandas as pd
import numpy as np
import warnings
from pathlib import Path
from datetime import datetime

from sklearn.model_selection import GridSearchCV, cross_val_predict, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

warnings.filterwarnings('ignore')

# 경로
ML_DIR = Path(__file__).parent.parent
DATA_PATH = ML_DIR / "data" / "training_data_v2.csv"
RESULT_PATH = ML_DIR / "data" / "tuning_results.csv"

# 피처셋 - 성능 좋았던 A와 B만 사용
FEATURES_A = [
    'breakout_pct', 'volume_surge', 'close_strength',
    'base_length', 'volatility_contraction', 'volume_dry_up',
    'ma200_slope', 'pct_above_52w_low',
    'rs_vs_market', 'market_return',
    'ma20_deviation', 'liquidity',
    'days_since_ath', 'atr_ratio'
]

FEATURES_B = [
    'ma20_deviation', 'breakout_pct', 'rs_vs_market', 'pct_above_52w_low',
    'atr_ratio', 'volume_dry_up', 'volatility_contraction', 'volume_surge'
]


def evaluate_model(y_true, y_pred):
    """전체 평가"""
    y_true_binary = (y_true >= 10).astype(int)
    y_pred_binary = (y_pred >= 10).astype(int)

    try:
        auc = roc_auc_score(y_true_binary, y_pred)
    except:
        auc = 0.5

    return {
        'MAE': mean_absolute_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'R2': r2_score(y_true, y_pred),
        'Accuracy': accuracy_score(y_true_binary, y_pred_binary),
        'Precision': precision_score(y_true_binary, y_pred_binary, zero_division=0),
        'Recall': recall_score(y_true_binary, y_pred_binary, zero_division=0),
        'F1': f1_score(y_true_binary, y_pred_binary, zero_division=0),
        'AUC': auc
    }


def tune_model(X, y, model, param_grid, model_name, cv=5):
    """모델 하이퍼파라미터 튜닝"""
    print(f"\n  {model_name} 튜닝 중...")

    # GridSearchCV
    grid_search = GridSearchCV(
        model,
        param_grid,
        cv=cv,
        scoring='r2',
        n_jobs=-1,
        verbose=0
    )

    grid_search.fit(X, y)

    print(f"    최적 파라미터: {grid_search.best_params_}")
    print(f"    최고 R2 (CV): {grid_search.best_score_:.4f}")

    # 최적 모델로 예측
    best_model = grid_search.best_estimator_
    y_pred = cross_val_predict(best_model, X, y, cv=cv, n_jobs=-1)

    # 평가
    metrics = evaluate_model(y, y_pred)
    metrics['best_params'] = str(grid_search.best_params_)

    print(f"    최종 성능: R2={metrics['R2']:.4f}, F1={metrics['F1']:.4f}, Prec={metrics['Precision']:.4f}")

    return metrics, best_model


def main():
    print("=" * 70)
    print("빠른 하이퍼파라미터 튜닝")
    print("=" * 70)

    # 데이터 로드
    print("\n[1/4] 데이터 로드...")
    df = pd.read_csv(DATA_PATH)
    y = df['max_gain_20d'].values
    print(f"  샘플 수: {len(df)}")

    # 스케일링
    scaler_A = StandardScaler()
    scaler_B = StandardScaler()
    X_A = scaler_A.fit_transform(df[FEATURES_A].values)
    X_B = scaler_B.fit_transform(df[FEATURES_B].values)

    results = []

    # ========================================
    # 2. 선형 모델 튜닝
    # ========================================
    print("\n[2/4] 선형 모델 튜닝...")

    # Lasso (피처셋 A) - 축소된 그리드
    lasso_params = {
        'alpha': [0.01, 0.1, 1.0, 5.0]
    }
    metrics, _ = tune_model(X_A, y, Lasso(max_iter=10000), lasso_params, "Lasso (A)")
    results.append({'feature_set': 'A', 'model': 'Lasso', **metrics})

    # Lasso (피처셋 B)
    metrics, _ = tune_model(X_B, y, Lasso(max_iter=10000), lasso_params, "Lasso (B)")
    results.append({'feature_set': 'B', 'model': 'Lasso', **metrics})

    # Ridge (피처셋 A)
    ridge_params = {
        'alpha': [0.1, 1.0, 10.0, 100.0]
    }
    metrics, _ = tune_model(X_A, y, Ridge(), ridge_params, "Ridge (A)")
    results.append({'feature_set': 'A', 'model': 'Ridge', **metrics})

    # ElasticNet
    elastic_params = {
        'alpha': [0.1, 1.0],
        'l1_ratio': [0.3, 0.5, 0.7]
    }
    metrics, _ = tune_model(X_A, y, ElasticNet(max_iter=10000), elastic_params, "ElasticNet (A)")
    results.append({'feature_set': 'A', 'model': 'ElasticNet', **metrics})

    # ========================================
    # 3. 트리 모델 튜닝
    # ========================================
    print("\n[3/4] 트리 모델 튜닝...")

    # RandomForest (피처셋 B) - 축소된 그리드
    rf_params = {
        'n_estimators': [100, 200],
        'max_depth': [5, 10, None],
        'min_samples_split': [10, 20]
    }
    metrics, _ = tune_model(X_B, y, RandomForestRegressor(random_state=42, n_jobs=-1),
                            rf_params, "RandomForest (B)")
    results.append({'feature_set': 'B', 'model': 'RandomForest', **metrics})

    # XGBoost (피처셋 A) - 축소된 그리드
    xgb_params = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5],
        'learning_rate': [0.05, 0.1],
        'reg_alpha': [0.1, 1.0]
    }
    metrics, _ = tune_model(X_A, y, XGBRegressor(random_state=42, verbosity=0),
                            xgb_params, "XGBoost (A)")
    results.append({'feature_set': 'A', 'model': 'XGBoost', **metrics})

    # LightGBM (피처셋 A)
    lgbm_params = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5],
        'learning_rate': [0.05, 0.1],
        'reg_alpha': [0.1, 1.0]
    }
    metrics, _ = tune_model(X_A, y, LGBMRegressor(random_state=42, verbose=-1),
                            lgbm_params, "LightGBM (A)")
    results.append({'feature_set': 'A', 'model': 'LightGBM', **metrics})

    # GradientBoosting
    gb_params = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5],
        'learning_rate': [0.05, 0.1]
    }
    metrics, _ = tune_model(X_A, y, GradientBoostingRegressor(random_state=42),
                            gb_params, "GradientBoosting (A)")
    results.append({'feature_set': 'A', 'model': 'GradientBoosting', **metrics})

    # ========================================
    # 4. 결과 정리
    # ========================================
    print("\n[4/4] 결과 정리...")

    results_df = pd.DataFrame(results)

    # R2 기준 정렬
    results_df = results_df.sort_values('R2', ascending=False)

    # 저장
    results_df.to_csv(RESULT_PATH, index=False, encoding='utf-8-sig')

    # 출력
    print("\n" + "=" * 70)
    print("튜닝 결과 (R2 기준 정렬)")
    print("=" * 70)
    print(f"{'피처셋':<5} {'모델':<18} {'R2':>8} {'F1':>8} {'Prec':>8} {'Recall':>8}")
    print("-" * 70)

    for _, row in results_df.iterrows():
        print(f"{row['feature_set']:<5} {row['model']:<18} "
              f"{row['R2']:>8.4f} {row['F1']:>8.4f} "
              f"{row['Precision']:>8.4f} {row['Recall']:>8.4f}")

    # 최고 성능
    print("\n" + "=" * 70)
    print("최고 성능 모델")
    print("=" * 70)

    best_r2 = results_df.iloc[0]
    print(f"\n[R2 기준]")
    print(f"  피처셋: {best_r2['feature_set']}, 모델: {best_r2['model']}")
    print(f"  R2: {best_r2['R2']:.4f}, MAE: {best_r2['MAE']:.2f}")
    print(f"  파라미터: {best_r2['best_params']}")

    best_f1 = results_df.sort_values('F1', ascending=False).iloc[0]
    print(f"\n[F1 기준]")
    print(f"  피처셋: {best_f1['feature_set']}, 모델: {best_f1['model']}")
    print(f"  F1: {best_f1['F1']:.4f}, Precision: {best_f1['Precision']:.4f}")
    print(f"  파라미터: {best_f1['best_params']}")

    best_prec = results_df.sort_values('Precision', ascending=False).iloc[0]
    print(f"\n[Precision 기준]")
    print(f"  피처셋: {best_prec['feature_set']}, 모델: {best_prec['model']}")
    print(f"  Precision: {best_prec['Precision']:.4f}, Recall: {best_prec['Recall']:.4f}")
    print(f"  파라미터: {best_prec['best_params']}")

    # 튜닝 전후 비교
    print("\n" + "=" * 70)
    print("튜닝 전후 비교 (기준: model_comparison_results.csv)")
    print("=" * 70)
    print(f"{'항목':<15} {'튜닝 전':>12} {'튜닝 후':>12} {'개선':>10}")
    print("-" * 50)
    print(f"{'R2 (최고)':<15} {'0.0304':>12} {best_r2['R2']:>12.4f} {(best_r2['R2']-0.0304)*100:>+10.2f}%p")
    print(f"{'F1 (최고)':<15} {'0.5859':>12} {best_f1['F1']:>12.4f} {(best_f1['F1']-0.5859)*100:>+10.2f}%p")
    print(f"{'Precision':<15} {'0.4537':>12} {best_prec['Precision']:>12.4f} {(best_prec['Precision']-0.4537)*100:>+10.2f}%p")

    print(f"\n결과 저장: {RESULT_PATH}")

    return results_df


if __name__ == "__main__":
    start_time = datetime.now()
    results_df = main()
    elapsed = datetime.now() - start_time
    print(f"\n완료! (소요시간: {elapsed})")
