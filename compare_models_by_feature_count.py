"""
피처 수별 모델 성능 비교

10개 ~ 15개 피처 × 7개 모델 = 42개 조합 비교
"""

import pandas as pd
import numpy as np
import warnings
from pathlib import Path
from datetime import datetime

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict, StratifiedKFold, KFold
from sklearn.linear_model import Ridge, Lasso, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, VotingRegressor, VotingClassifier, StackingRegressor, StackingClassifier
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

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
RESULT_PATH = BASE_DIR / "data" / "model_feature_comparison.csv"

# Forward Selection으로 선택된 10개 피처
FEATURES_10 = [
    'ma20_deviation', 'pct_above_52w_low', 'breakout_strength',
    'close_strength', 'days_since_ath', 'atr_ratio',
    'liquidity', 'breakout_gap', 'rs_vs_market', 'market_return'
]

# 상관관계 높은 순으로 추가되는 5개
ADDITIONAL_FEATURES = [
    'volume_surge',           # 11개
    'box_range_pct',          # 12개
    'volume_dry_up',          # 13개
    'ma200_slope',            # 14개
    'volatility_contraction'  # 15개
]

# 피처셋 정의
FEATURE_SETS = {
    '10features': FEATURES_10[:10],
    '11features': FEATURES_10[:10] + ADDITIONAL_FEATURES[:1],
    '12features': FEATURES_10[:10] + ADDITIONAL_FEATURES[:2],
    '13features': FEATURES_10[:10] + ADDITIONAL_FEATURES[:3],
    '14features': FEATURES_10[:10] + ADDITIONAL_FEATURES[:4],
    '15features': FEATURES_10[:10] + ADDITIONAL_FEATURES[:5],
}


def get_regression_models():
    """회귀 모델"""
    models = {
        'Ridge': Ridge(alpha=1.0),
        'Lasso': Lasso(alpha=0.1, max_iter=5000),
        'RandomForest': RandomForestRegressor(
            n_estimators=100, max_depth=10, min_samples_split=10,
            random_state=42, n_jobs=-1
        ),
    }

    if HAS_XGB:
        models['XGBoost'] = XGBRegressor(
            n_estimators=100, max_depth=5, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbosity=0
        )

    if HAS_LGBM:
        models['LightGBM'] = LGBMRegressor(
            n_estimators=100, max_depth=5, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbose=-1
        )

    if HAS_XGB and HAS_LGBM:
        models['Voting'] = VotingRegressor(
            estimators=[
                ('rf', RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)),
                ('xgb', XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, verbosity=0)),
                ('lgbm', LGBMRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, verbose=-1))
            ],
            n_jobs=-1
        )

        models['Stacking'] = StackingRegressor(
            estimators=[
                ('rf', RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)),
                ('xgb', XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, verbosity=0)),
                ('lgbm', LGBMRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, verbose=-1))
            ],
            final_estimator=Ridge(alpha=1.0),
            cv=3, n_jobs=-1
        )

    return models


def get_classification_models():
    """분류 모델"""
    models = {
        'LogisticRidge': LogisticRegression(penalty='l2', C=1.0, max_iter=5000, random_state=42),
        'LogisticLasso': LogisticRegression(penalty='l1', C=1.0, solver='saga', max_iter=5000, random_state=42),
        'RandomForest': RandomForestClassifier(
            n_estimators=100, max_depth=10, min_samples_split=10,
            random_state=42, n_jobs=-1
        ),
    }

    if HAS_XGB:
        models['XGBoost'] = XGBClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbosity=0, eval_metric='logloss'
        )

    if HAS_LGBM:
        models['LightGBM'] = LGBMClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbose=-1
        )

    if HAS_XGB and HAS_LGBM:
        models['Voting'] = VotingClassifier(
            estimators=[
                ('rf', RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)),
                ('xgb', XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, verbosity=0, eval_metric='logloss')),
                ('lgbm', LGBMClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, verbose=-1))
            ],
            voting='soft', n_jobs=-1
        )

        models['Stacking'] = StackingClassifier(
            estimators=[
                ('rf', RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)),
                ('xgb', XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, verbosity=0, eval_metric='logloss')),
                ('lgbm', LGBMClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, verbose=-1))
            ],
            final_estimator=LogisticRegression(max_iter=5000, random_state=42),
            cv=3, n_jobs=-1
        )

    return models


def run_comparison():
    """모델 비교 실행"""
    print("=" * 80)
    print("피처 수별 모델 성능 비교: 10~15개 피처 × 7개 모델")
    print("=" * 80)

    # 데이터 로드
    print("\n[1/3] 데이터 로드...")
    df = pd.read_csv(DATA_PATH, encoding='utf-8-sig')
    y_reg = df['max_gain_20d'].values
    y_cls = df['success'].values

    print(f"  샘플 수: {len(df):,}")
    print(f"  성공률: {y_cls.mean()*100:.2f}%")

    results = []
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # 회귀 모델 비교
    print("\n[2/3] 회귀 모델 평가...")
    reg_models = get_regression_models()
    print(f"  모델 수: {len(reg_models)}")

    total = len(FEATURE_SETS) * len(reg_models)
    current = 0

    for feat_name, features in FEATURE_SETS.items():
        print(f"\n  === {feat_name} ({len(features)}개 피처) ===")

        X = df[features].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        for model_name, model in reg_models.items():
            current += 1
            print(f"    [{current}/{total}] {model_name}...", end=" ")

            try:
                y_pred = cross_val_predict(model, X_scaled, y_reg, cv=kfold, n_jobs=-1)

                mae = mean_absolute_error(y_reg, y_pred)
                rmse = np.sqrt(mean_squared_error(y_reg, y_pred))
                r2 = r2_score(y_reg, y_pred)

                results.append({
                    'task': 'regression',
                    'n_features': len(features),
                    'feature_set': feat_name,
                    'model': model_name,
                    'MAE': mae,
                    'RMSE': rmse,
                    'R2': r2,
                    'Accuracy': None,
                    'Precision': None,
                    'Recall': None,
                    'F1': None,
                    'AUC': None
                })

                print(f"R2={r2:.4f}")

            except Exception as e:
                print(f"ERROR: {e}")

    # 분류 모델 비교
    print("\n[3/3] 분류 모델 평가...")
    cls_models = get_classification_models()
    print(f"  모델 수: {len(cls_models)}")

    total = len(FEATURE_SETS) * len(cls_models)
    current = 0

    for feat_name, features in FEATURE_SETS.items():
        print(f"\n  === {feat_name} ({len(features)}개 피처) ===")

        X = df[features].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        for model_name, model in cls_models.items():
            current += 1
            print(f"    [{current}/{total}] {model_name}...", end=" ")

            try:
                y_pred = cross_val_predict(model, X_scaled, y_cls, cv=skfold)
                y_proba = cross_val_predict(model, X_scaled, y_cls, cv=skfold, method='predict_proba', n_jobs=-1)[:, 1]

                acc = accuracy_score(y_cls, y_pred)
                prec = precision_score(y_cls, y_pred)
                rec = recall_score(y_cls, y_pred)
                f1 = f1_score(y_cls, y_pred)
                auc = roc_auc_score(y_cls, y_proba)

                results.append({
                    'task': 'classification',
                    'n_features': len(features),
                    'feature_set': feat_name,
                    'model': model_name,
                    'MAE': None,
                    'RMSE': None,
                    'R2': None,
                    'Accuracy': acc,
                    'Precision': prec,
                    'Recall': rec,
                    'F1': f1,
                    'AUC': auc
                })

                print(f"F1={f1:.4f}, AUC={auc:.4f}")

            except Exception as e:
                print(f"ERROR: {e}")

    # 결과 저장
    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULT_PATH, index=False, encoding='utf-8-sig')
    print(f"\n\n결과 저장: {RESULT_PATH}")

    return results_df


def print_results(results_df):
    """결과 출력"""
    print("\n" + "=" * 80)
    print("회귀 성능 TOP 10 (R2 기준)")
    print("=" * 80)

    reg_df = results_df[results_df['task'] == 'regression'].copy()
    reg_df = reg_df.sort_values('R2', ascending=False)

    print(f"{'순위':<6} {'피처수':<8} {'모델':<15} {'R2':>10} {'MAE':>10}")
    print("-" * 55)

    for i, (_, row) in enumerate(reg_df.head(10).iterrows(), 1):
        print(f"{i:<6} {row['n_features']:<8} {row['model']:<15} {row['R2']:>10.4f} {row['MAE']:>10.2f}")

    print("\n" + "=" * 80)
    print("분류 성능 TOP 10 (F1 기준)")
    print("=" * 80)

    cls_df = results_df[results_df['task'] == 'classification'].copy()
    cls_df = cls_df.sort_values('F1', ascending=False)

    print(f"{'순위':<6} {'피처수':<8} {'모델':<15} {'F1':>10} {'AUC':>10} {'Precision':>11}")
    print("-" * 65)

    for i, (_, row) in enumerate(cls_df.head(10).iterrows(), 1):
        print(f"{i:<6} {row['n_features']:<8} {row['model']:<15} {row['F1']:>10.4f} {row['AUC']:>10.4f} {row['Precision']:>11.4f}")

    # 최고 성능
    print("\n" + "=" * 80)
    print("최고 성능 조합")
    print("=" * 80)

    best_reg = reg_df.iloc[0]
    best_cls = cls_df.iloc[0]

    print(f"\n[회귀 1위]")
    print(f"  피처 수: {best_reg['n_features']}개")
    print(f"  모델: {best_reg['model']}")
    print(f"  R2: {best_reg['R2']:.4f}")
    print(f"  MAE: {best_reg['MAE']:.2f}%")

    print(f"\n[분류 1위]")
    print(f"  피처 수: {best_cls['n_features']}개")
    print(f"  모델: {best_cls['model']}")
    print(f"  F1: {best_cls['F1']:.4f}")
    print(f"  AUC: {best_cls['AUC']:.4f}")
    print(f"  Precision: {best_cls['Precision']:.4f}")


def main():
    start_time = datetime.now()

    results_df = run_comparison()
    print_results(results_df)

    elapsed = datetime.now() - start_time
    print(f"\n완료! (소요시간: {elapsed})")


if __name__ == "__main__":
    main()
