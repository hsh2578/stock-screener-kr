"""
52주 신고가 근접 - 모델 비교 + 하이퍼파라미터 튜닝

[회귀 모델 비교]
  Ridge, Lasso, RandomForest, XGBoost, LightGBM

[분류 모델 비교]
  LogisticRegression, RandomForest, XGBoost, LightGBM, GradientBoosting

[하이퍼파라미터 튜닝]
  회귀/분류 각각 상위 2개 모델에 대해 GridSearchCV 수행

[출력]
  1. 모델별 성능 비교표
  2. 최적 하이퍼파라미터
  3. 최종 추천 모델 + 파라미터
"""

import pandas as pd
import numpy as np
import warnings
from pathlib import Path
from datetime import datetime

from sklearn.model_selection import (
    cross_val_predict, KFold, StratifiedKFold, GridSearchCV
)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, LogisticRegression
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier,
    GradientBoostingClassifier
)
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)

from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier

warnings.filterwarnings('ignore')

# 경로
ML_DIR = Path(__file__).parent.parent
DATA_PATH = ML_DIR / "data" / "near_high_training_data.csv"

# 피처 (13개)
FEATURES = [
    'gap_pct', 'resistance_tests', 'days_since_high',
    'momentum_5d', 'volume_ratio', 'volume_trend',
    'volatility_contraction', 'atr_pct',
    'ma_alignment', 'trend_strength',
    'rs_vs_market', 'market_return',
    'pct_from_52w_low'
]

RANDOM_STATE = 42


def load_data():
    """데이터 로드 + 품질 필터링"""
    df = pd.read_csv(DATA_PATH)
    print(f"원본: {len(df)}건")

    # atr_pct >= 50 제거
    bad = df['atr_pct'] >= 50
    if bad.sum() > 0:
        print(f"atr_pct >= 50 제거: {bad.sum()}건")
        df = df[~bad].reset_index(drop=True)

    X = df[FEATURES].values
    y_reg = df['max_gain_20d'].values
    y_cls = df['broke_through_10d'].astype(int).values

    print(f"최종: {len(df)}건, 돌파율: {y_cls.mean()*100:.1f}%")
    print(f"평균 수익률: {y_reg.mean():.2f}%\n")

    return X, y_reg, y_cls


# ============================================================================
# 회귀 모델 비교
# ============================================================================

def get_regressors():
    return {
        'Ridge': Ridge(alpha=1.0, random_state=RANDOM_STATE),
        'Lasso': Lasso(alpha=0.1, max_iter=5000, random_state=RANDOM_STATE),
        'RandomForest': RandomForestRegressor(
            n_estimators=200, max_depth=10, min_samples_split=10,
            random_state=RANDOM_STATE, n_jobs=-1
        ),
        'XGBoost': XGBRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=RANDOM_STATE, verbosity=0, n_jobs=-1
        ),
        'LightGBM': LGBMRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=RANDOM_STATE, verbose=-1, n_jobs=-1
        ),
    }


def compare_regressors(X_scaled, y_reg):
    print("=" * 70)
    print("회귀 모델 비교 (20거래일 최고 수익률 예측)")
    print("=" * 70)

    kfold = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    results = []

    for name, model in get_regressors().items():
        y_pred = cross_val_predict(model, X_scaled, y_reg, cv=kfold)

        mae = mean_absolute_error(y_reg, y_pred)
        rmse = np.sqrt(mean_squared_error(y_reg, y_pred))
        r2 = r2_score(y_reg, y_pred)

        results.append({
            'model': name, 'MAE': mae, 'RMSE': rmse, 'R2': r2
        })
        print(f"  {name:20s}  MAE={mae:6.2f}%  RMSE={rmse:6.2f}%  R2={r2:7.4f}")

    print()
    results_df = pd.DataFrame(results).sort_values('MAE')
    print("  [MAE 기준 순위]")
    for i, row in results_df.iterrows():
        marker = " ★" if row['model'] == results_df.iloc[0]['model'] else ""
        print(f"    {row['model']:20s}  MAE={row['MAE']:6.2f}%{marker}")

    return results_df


# ============================================================================
# 분류 모델 비교
# ============================================================================

def get_classifiers():
    return {
        'LogisticRegression': LogisticRegression(
            C=1.0, max_iter=1000, random_state=RANDOM_STATE, n_jobs=-1
        ),
        'RandomForest': RandomForestClassifier(
            n_estimators=200, max_depth=10, min_samples_split=10,
            random_state=RANDOM_STATE, n_jobs=-1
        ),
        'XGBoost': XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=RANDOM_STATE, verbosity=0, n_jobs=-1,
            eval_metric='logloss'
        ),
        'LightGBM': LGBMClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=RANDOM_STATE, verbose=-1, n_jobs=-1
        ),
        'GradientBoosting': GradientBoostingClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            subsample=0.8, random_state=RANDOM_STATE
        ),
    }


def compare_classifiers(X_scaled, y_cls):
    print("\n" + "=" * 70)
    print("분류 모델 비교 (10거래일 내 돌파 여부)")
    print("=" * 70)

    skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    results = []

    for name, model in get_classifiers().items():
        y_pred = cross_val_predict(model, X_scaled, y_cls, cv=skfold)
        try:
            y_proba = cross_val_predict(
                model, X_scaled, y_cls, cv=skfold, method='predict_proba'
            )[:, 1]
            auc = roc_auc_score(y_cls, y_proba)
        except:
            auc = 0.5

        acc = accuracy_score(y_cls, y_pred)
        prec = precision_score(y_cls, y_pred)
        rec = recall_score(y_cls, y_pred)
        f1 = f1_score(y_cls, y_pred)
        tn, fp, fn, tp = confusion_matrix(y_cls, y_pred).ravel()

        results.append({
            'model': name,
            'Accuracy': acc, 'Precision': prec, 'Recall': rec,
            'F1': f1, 'AUC': auc,
            'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn
        })
        print(f"  {name:20s}  Acc={acc:.4f}  Prec={prec:.4f}  Rec={rec:.4f}  F1={f1:.4f}  AUC={auc:.4f}")

    print()
    results_df = pd.DataFrame(results).sort_values('AUC', ascending=False)
    print("  [AUC 기준 순위]")
    for i, row in results_df.iterrows():
        marker = " ★" if row['model'] == results_df.iloc[0]['model'] else ""
        print(f"    {row['model']:20s}  AUC={row['AUC']:.4f}  F1={row['F1']:.4f}{marker}")

    return results_df


# ============================================================================
# 하이퍼파라미터 튜닝
# ============================================================================

def tune_regressor(X_scaled, y_reg, best_models):
    print("\n" + "=" * 70)
    print("회귀 하이퍼파라미터 튜닝 (상위 2개 모델)")
    print("=" * 70)

    param_grids = {
        'LightGBM': {
            'model': LGBMRegressor(random_state=RANDOM_STATE, verbose=-1, n_jobs=-1),
            'params': {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.05, 0.1],
                'subsample': [0.7, 0.8, 0.9],
                'reg_alpha': [0.0, 0.5, 1.0],
                'reg_lambda': [1.0, 3.0, 5.0],
            }
        },
        'XGBoost': {
            'model': XGBRegressor(random_state=RANDOM_STATE, verbosity=0, n_jobs=-1),
            'params': {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.05, 0.1],
                'subsample': [0.7, 0.8, 0.9],
                'colsample_bytree': [0.7, 0.8, 0.9],
                'reg_alpha': [0.0, 0.5, 1.0],
                'reg_lambda': [1.0, 3.0, 5.0],
            }
        },
        'RandomForest': {
            'model': RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
            'params': {
                'n_estimators': [100, 200, 300],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [5, 10, 20],
                'min_samples_leaf': [2, 5, 10],
            }
        },
        'Ridge': {
            'model': Ridge(random_state=RANDOM_STATE),
            'params': {
                'alpha': [0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0],
            }
        },
        'Lasso': {
            'model': Lasso(random_state=RANDOM_STATE, max_iter=5000),
            'params': {
                'alpha': [0.001, 0.01, 0.05, 0.1, 0.5, 1.0],
            }
        },
    }

    top_2 = best_models.head(2)['model'].tolist()
    best_result = None

    for model_name in top_2:
        if model_name not in param_grids:
            print(f"\n  {model_name}: 튜닝 파라미터 없음, 건너뜀")
            continue

        config = param_grids[model_name]
        print(f"\n  [{model_name}] GridSearchCV 시작...")

        # RandomizedSearchCV 대신 축소된 그리드 사용
        reduced_params = {}
        for k, v in config['params'].items():
            # 파라미터가 너무 많으면 핵심 3개만 선택
            reduced_params[k] = v

        grid = GridSearchCV(
            config['model'], reduced_params,
            cv=KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
            scoring='neg_mean_absolute_error',
            n_jobs=-1, verbose=0,
            # 조합이 너무 많으면 Randomized로 전환
        )

        # 조합 수 확인
        n_combos = 1
        for v in reduced_params.values():
            n_combos *= len(v)

        if n_combos > 500:
            # RandomizedSearchCV로 전환
            from sklearn.model_selection import RandomizedSearchCV
            grid = RandomizedSearchCV(
                config['model'], reduced_params,
                n_iter=100, cv=KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
                scoring='neg_mean_absolute_error',
                n_jobs=-1, verbose=0, random_state=RANDOM_STATE
            )
            print(f"    조합 {n_combos}개 → RandomizedSearchCV (100회)")
        else:
            print(f"    조합 {n_combos}개 → GridSearchCV")

        grid.fit(X_scaled, y_reg)

        best_score = -grid.best_score_
        print(f"    최적 MAE: {best_score:.2f}%")
        print(f"    최적 파라미터:")
        for k, v in grid.best_params_.items():
            print(f"      {k}: {v}")

        if best_result is None or best_score < best_result['mae']:
            best_result = {
                'model': model_name,
                'mae': best_score,
                'params': grid.best_params_,
                'estimator': grid.best_estimator_
            }

    return best_result


def tune_classifier(X_scaled, y_cls, best_models):
    print("\n" + "=" * 70)
    print("분류 하이퍼파라미터 튜닝 (상위 2개 모델)")
    print("=" * 70)

    param_grids = {
        'LightGBM': {
            'model': LGBMClassifier(random_state=RANDOM_STATE, verbose=-1, n_jobs=-1),
            'params': {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.05, 0.1],
                'subsample': [0.7, 0.8, 0.9],
                'colsample_bytree': [0.7, 0.8, 0.9],
            }
        },
        'XGBoost': {
            'model': XGBClassifier(
                random_state=RANDOM_STATE, verbosity=0, n_jobs=-1,
                eval_metric='logloss'
            ),
            'params': {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.05, 0.1],
                'subsample': [0.7, 0.8, 0.9],
                'colsample_bytree': [0.7, 0.8, 0.9],
            }
        },
        'RandomForest': {
            'model': RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
            'params': {
                'n_estimators': [100, 200, 300],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [5, 10, 20],
                'min_samples_leaf': [2, 5, 10],
            }
        },
        'GradientBoosting': {
            'model': GradientBoostingClassifier(random_state=RANDOM_STATE),
            'params': {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.05, 0.1],
                'subsample': [0.7, 0.8, 0.9],
            }
        },
        'LogisticRegression': {
            'model': LogisticRegression(random_state=RANDOM_STATE, max_iter=1000, n_jobs=-1),
            'params': {
                'C': [0.01, 0.1, 0.5, 1.0, 5.0, 10.0],
                'penalty': ['l1', 'l2'],
                'solver': ['saga'],
            }
        },
    }

    top_2 = best_models.head(2)['model'].tolist()
    best_result = None

    for model_name in top_2:
        if model_name not in param_grids:
            print(f"\n  {model_name}: 튜닝 파라미터 없음, 건너뜀")
            continue

        config = param_grids[model_name]
        print(f"\n  [{model_name}] GridSearchCV 시작...")

        n_combos = 1
        for v in config['params'].values():
            n_combos *= len(v)

        if n_combos > 500:
            from sklearn.model_selection import RandomizedSearchCV
            grid = RandomizedSearchCV(
                config['model'], config['params'],
                n_iter=100,
                cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
                scoring='roc_auc',
                n_jobs=-1, verbose=0, random_state=RANDOM_STATE
            )
            print(f"    조합 {n_combos}개 → RandomizedSearchCV (100회)")
        else:
            grid = GridSearchCV(
                config['model'], config['params'],
                cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
                scoring='roc_auc',
                n_jobs=-1, verbose=0,
            )
            print(f"    조합 {n_combos}개 → GridSearchCV")

        grid.fit(X_scaled, y_cls)

        best_score = grid.best_score_
        print(f"    최적 AUC: {best_score:.4f}")
        print(f"    최적 파라미터:")
        for k, v in grid.best_params_.items():
            print(f"      {k}: {v}")

        if best_result is None or best_score > best_result['auc']:
            best_result = {
                'model': model_name,
                'auc': best_score,
                'params': grid.best_params_,
                'estimator': grid.best_estimator_
            }

    return best_result


# ============================================================================
# 최종 검증
# ============================================================================

def final_validation(X_scaled, y_reg, y_cls, best_reg, best_cls):
    print("\n" + "=" * 70)
    print("최종 검증 (최적 모델 + 최적 파라미터)")
    print("=" * 70)

    # 회귀
    print(f"\n  [회귀] {best_reg['model']}")
    reg_model = best_reg['estimator']
    kfold = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    y_pred_reg = cross_val_predict(reg_model, X_scaled, y_reg, cv=kfold)

    mae = mean_absolute_error(y_reg, y_pred_reg)
    rmse = np.sqrt(mean_squared_error(y_reg, y_pred_reg))
    r2 = r2_score(y_reg, y_pred_reg)
    print(f"    MAE:  {mae:.2f}%")
    print(f"    RMSE: {rmse:.2f}%")
    print(f"    R2:   {r2:.4f}")

    # 분류
    print(f"\n  [분류] {best_cls['model']}")
    cls_model = best_cls['estimator']
    skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    y_pred_cls = cross_val_predict(cls_model, X_scaled, y_cls, cv=skfold)
    y_proba_cls = cross_val_predict(
        cls_model, X_scaled, y_cls, cv=skfold, method='predict_proba'
    )[:, 1]

    acc = accuracy_score(y_cls, y_pred_cls)
    prec = precision_score(y_cls, y_pred_cls)
    rec = recall_score(y_cls, y_pred_cls)
    f1 = f1_score(y_cls, y_pred_cls)
    auc = roc_auc_score(y_cls, y_proba_cls)
    tn, fp, fn, tp = confusion_matrix(y_cls, y_pred_cls).ravel()

    print(f"    Accuracy:  {acc:.4f}")
    print(f"    Precision: {prec:.4f}")
    print(f"    Recall:    {rec:.4f}")
    print(f"    F1:        {f1:.4f}")
    print(f"    AUC:       {auc:.4f}")
    print(f"    TP={tp}, FP={fp}, TN={tn}, FN={fn}")


def main():
    print("=" * 70)
    print("52주 신고가 근접 - 모델 비교 + 하이퍼파라미터 튜닝")
    print("=" * 70)

    # 데이터 로드
    X, y_reg, y_cls = load_data()

    # 스케일링
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 1. 모델 비교
    reg_results = compare_regressors(X_scaled, y_reg)
    cls_results = compare_classifiers(X_scaled, y_cls)

    # 2. 하이퍼파라미터 튜닝
    best_reg = tune_regressor(X_scaled, y_reg, reg_results)
    best_cls = tune_classifier(X_scaled, y_cls, cls_results)

    # 3. 최종 검증
    if best_reg and best_cls:
        final_validation(X_scaled, y_reg, y_cls, best_reg, best_cls)

    # 4. 최종 추천
    print("\n" + "=" * 70)
    print("최종 추천")
    print("=" * 70)

    if best_reg:
        print(f"\n  [회귀 모델] {best_reg['model']}")
        print(f"  파라미터:")
        for k, v in best_reg['params'].items():
            print(f"    '{k}': {v},")

    if best_cls:
        print(f"\n  [분류 모델] {best_cls['model']}")
        print(f"  파라미터:")
        for k, v in best_cls['params'].items():
            print(f"    '{k}': {v},")

    print("\n  -> train_near_high.py의 REGRESSOR_PARAMS, CLASSIFIER_PARAMS를 위 값으로 업데이트하세요.")


if __name__ == "__main__":
    start = datetime.now()
    main()
    elapsed = datetime.now() - start
    print(f"\n완료! (소요시간: {elapsed})")
