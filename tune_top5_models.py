"""
TOP 5 모델 하이퍼파라미터 튜닝

회귀 TOP 5:
1. 10개 + Ridge
2. 10개 + Lasso
3. 11개 + Ridge
4. 11개 + Lasso
5. 12개 + Lasso

분류 TOP 5:
1. 15개 + XGBoost
2. 14개 + XGBoost
3. 15개 + LightGBM
4. 14개 + LightGBM
5. 13개 + XGBoost
"""

import pandas as pd
import numpy as np
import warnings
from pathlib import Path
from datetime import datetime

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold, KFold
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import make_scorer, r2_score, f1_score

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings('ignore')

# 경로
BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "data" / "box_breakout_history.csv"

# 피처셋
FEATURES_10 = [
    'ma20_deviation', 'pct_above_52w_low', 'breakout_strength',
    'close_strength', 'days_since_ath', 'atr_ratio',
    'liquidity', 'breakout_gap', 'rs_vs_market', 'market_return'
]

FEATURES_11 = FEATURES_10 + ['volume_surge']
FEATURES_12 = FEATURES_11 + ['box_range_pct']
FEATURES_13 = FEATURES_12 + ['volume_dry_up']
FEATURES_14 = FEATURES_13 + ['ma200_slope']
FEATURES_15 = FEATURES_14 + ['volatility_contraction']

FEATURE_SETS = {
    '10features': FEATURES_10,
    '11features': FEATURES_11,
    '12features': FEATURES_12,
    '13features': FEATURES_13,
    '14features': FEATURES_14,
    '15features': FEATURES_15,
}

# 파라미터 그리드
RIDGE_PARAMS = {
    'alpha': [0.01, 0.1, 0.5, 1.0, 5.0, 10.0]
}

LASSO_PARAMS = {
    'alpha': [0.001, 0.01, 0.05, 0.1, 0.5, 1.0]
}

XGBOOST_PARAMS = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9]
}

LIGHTGBM_PARAMS = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9]
}


def tune_regression_model(X, y, model_name, feature_name):
    """회귀 모델 튜닝"""
    print(f"\n  [{feature_name} + {model_name}]")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    if model_name == 'Ridge':
        model = Ridge(max_iter=5000)
        param_grid = RIDGE_PARAMS
    else:  # Lasso
        model = Lasso(max_iter=5000)
        param_grid = LASSO_PARAMS

    grid_search = GridSearchCV(
        model,
        param_grid,
        cv=kfold,
        scoring='r2',
        n_jobs=-1,
        verbose=0
    )

    grid_search.fit(X_scaled, y)

    print(f"    최적 파라미터: {grid_search.best_params_}")
    print(f"    최고 R2: {grid_search.best_score_:.4f}")

    return {
        'feature_set': feature_name,
        'model': model_name,
        'best_params': grid_search.best_params_,
        'best_score': grid_search.best_score_
    }


def tune_classification_model(X, y, model_name, feature_name):
    """분류 모델 튜닝"""
    print(f"\n  [{feature_name} + {model_name}]")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    if model_name == 'XGBoost':
        model = XGBClassifier(random_state=42, eval_metric='logloss', verbosity=0)
        param_grid = XGBOOST_PARAMS
    else:  # LightGBM
        model = LGBMClassifier(random_state=42, verbose=-1)
        param_grid = LIGHTGBM_PARAMS

    grid_search = GridSearchCV(
        model,
        param_grid,
        cv=skfold,
        scoring='f1',
        n_jobs=-1,
        verbose=0
    )

    grid_search.fit(X_scaled, y)

    print(f"    최적 파라미터: {grid_search.best_params_}")
    print(f"    최고 F1: {grid_search.best_score_:.4f}")

    return {
        'feature_set': feature_name,
        'model': model_name,
        'best_params': grid_search.best_params_,
        'best_score': grid_search.best_score_
    }


def main():
    print("=" * 80)
    print("TOP 5 모델 하이퍼파라미터 튜닝")
    print("=" * 80)

    # 데이터 로드
    print("\n[1/3] 데이터 로드...")
    df = pd.read_csv(DATA_PATH, encoding='utf-8-sig')
    y_reg = df['max_gain_20d'].values
    y_cls = df['success'].values

    print(f"  샘플 수: {len(df):,}")

    # 회귀 TOP 5 튜닝
    print("\n[2/3] 회귀 TOP 5 튜닝...")
    print("=" * 80)

    reg_top5 = [
        ('10features', 'Ridge'),
        ('10features', 'Lasso'),
        ('11features', 'Ridge'),
        ('11features', 'Lasso'),
        ('12features', 'Lasso'),
    ]

    reg_results = []
    for feat_name, model_name in reg_top5:
        X = df[FEATURE_SETS[feat_name]].values
        result = tune_regression_model(X, y_reg, model_name, feat_name)
        reg_results.append(result)

    # 분류 TOP 5 튜닝
    print("\n\n[3/3] 분류 TOP 5 튜닝...")
    print("=" * 80)

    cls_top5 = [
        ('15features', 'XGBoost'),
        ('14features', 'XGBoost'),
        ('15features', 'LightGBM'),
        ('14features', 'LightGBM'),
        ('13features', 'XGBoost'),
    ]

    cls_results = []
    for feat_name, model_name in cls_top5:
        X = df[FEATURE_SETS[feat_name]].values
        result = tune_classification_model(X, y_cls, model_name, feat_name)
        cls_results.append(result)

    # 결과 정리
    print("\n\n" + "=" * 80)
    print("튜닝 결과 요약")
    print("=" * 80)

    print("\n[회귀 모델]")
    print(f"{'순위':<6} {'피처셋':<12} {'모델':<10} {'튜닝 전 R2':>13} {'튜닝 후 R2':>13} {'개선':>10}")
    print("-" * 75)

    baseline_r2 = [0.0687, 0.0684, 0.0684, 0.0683, 0.0683]
    for i, (result, baseline) in enumerate(zip(reg_results, baseline_r2), 1):
        improvement = result['best_score'] - baseline
        print(f"{i:<6} {result['feature_set']:<12} {result['model']:<10} "
              f"{baseline:>13.4f} {result['best_score']:>13.4f} "
              f"{improvement:>+10.4f}")

    print("\n[분류 모델]")
    print(f"{'순위':<6} {'피처셋':<12} {'모델':<10} {'튜닝 전 F1':>13} {'튜닝 후 F1':>13} {'개선':>10}")
    print("-" * 75)

    baseline_f1 = [0.2700, 0.2662, 0.2650, 0.2621, 0.2598]
    for i, (result, baseline) in enumerate(zip(cls_results, baseline_f1), 1):
        improvement = result['best_score'] - baseline
        print(f"{i:<6} {result['feature_set']:<12} {result['model']:<10} "
              f"{baseline:>13.4f} {result['best_score']:>13.4f} "
              f"{improvement:>+10.4f}")

    # 최고 성능
    print("\n" + "=" * 80)
    print("최고 성능 (튜닝 후)")
    print("=" * 80)

    best_reg = max(reg_results, key=lambda x: x['best_score'])
    best_cls = max(cls_results, key=lambda x: x['best_score'])

    print(f"\n[회귀 1위]")
    print(f"  조합: {best_reg['feature_set']} + {best_reg['model']}")
    print(f"  R2: {best_reg['best_score']:.4f}")
    print(f"  파라미터: {best_reg['best_params']}")

    print(f"\n[분류 1위]")
    print(f"  조합: {best_cls['feature_set']} + {best_cls['model']}")
    print(f"  F1: {best_cls['best_score']:.4f}")
    print(f"  파라미터: {best_cls['best_params']}")

    return reg_results, cls_results


if __name__ == "__main__":
    start_time = datetime.now()
    reg_results, cls_results = main()
    elapsed = datetime.now() - start_time
    print(f"\n완료! (소요시간: {elapsed})")
