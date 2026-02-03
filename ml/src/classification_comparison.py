"""
분류 모델 비교: 10% 이상 상승 여부 직접 예측

회귀 모델과 달리 이진 분류로 직접 예측
"""

import pandas as pd
import numpy as np
import warnings
from pathlib import Path
from datetime import datetime

from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings('ignore')

# 경로
ML_DIR = Path(__file__).parent.parent
DATA_PATH = ML_DIR / "data" / "training_data_v2.csv"
RESULT_PATH = ML_DIR / "data" / "classification_results.csv"

# 피처셋
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


def get_classifiers():
    """분류 모델 딕셔너리"""
    return {
        'LogisticRegression': LogisticRegression(
            C=1.0, max_iter=1000, random_state=42, n_jobs=-1
        ),
        'LogisticRegression_L1': LogisticRegression(
            C=1.0, penalty='l1', solver='saga', max_iter=1000, random_state=42, n_jobs=-1
        ),
        'RandomForest': RandomForestClassifier(
            n_estimators=200, max_depth=10, min_samples_split=10,
            random_state=42, n_jobs=-1
        ),
        'XGBoost': XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbosity=0, n_jobs=-1
        ),
        'LightGBM': LGBMClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbose=-1, n_jobs=-1
        ),
        'GradientBoosting': GradientBoostingClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            subsample=0.8, random_state=42
        ),
    }


def evaluate_classifier(y_true, y_pred, y_proba=None):
    """분류 평가 지표"""
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'F1': f1_score(y_true, y_pred, zero_division=0),
    }

    if y_proba is not None:
        try:
            metrics['AUC'] = roc_auc_score(y_true, y_proba)
        except:
            metrics['AUC'] = 0.5
    else:
        metrics['AUC'] = 0.5

    # Confusion Matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics['TP'] = tp
    metrics['FP'] = fp
    metrics['TN'] = tn
    metrics['FN'] = fn

    return metrics


def main():
    print("=" * 70)
    print("분류 모델 비교: 10% 이상 상승 여부 직접 예측")
    print("=" * 70)

    # 데이터 로드
    print("\n[1/3] 데이터 로드...")
    df = pd.read_csv(DATA_PATH)

    # 이진 레이블 생성
    y = (df['max_gain_20d'] >= 10).astype(int)

    print(f"  샘플 수: {len(df)}")
    print(f"  성공(10%+): {y.sum()} ({y.mean()*100:.1f}%)")
    print(f"  실패(<10%): {(1-y).sum()} ({(1-y.mean())*100:.1f}%)")

    # 피처 준비
    scaler_A = StandardScaler()
    scaler_B = StandardScaler()
    X_A = scaler_A.fit_transform(df[FEATURES_A].values)
    X_B = scaler_B.fit_transform(df[FEATURES_B].values)

    # Stratified K-Fold (클래스 비율 유지)
    skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # 모델 준비
    classifiers = get_classifiers()

    results = []

    # 모델 학습 및 평가
    print("\n[2/3] 분류 모델 학습 및 평가...")

    feature_sets = {'A': (X_A, FEATURES_A), 'B': (X_B, FEATURES_B)}
    total = len(classifiers) * len(feature_sets)
    current = 0

    for feat_name, (X, features) in feature_sets.items():
        print(f"\n  === 피처셋 {feat_name} ({len(features)}개) ===")

        for model_name, model in classifiers.items():
            current += 1
            print(f"    [{current}/{total}] {model_name}...", end=" ")

            try:
                # Cross-validation 예측
                y_pred = cross_val_predict(model, X, y, cv=skfold, n_jobs=-1)

                # 확률 예측 (AUC용)
                try:
                    y_proba = cross_val_predict(model, X, y, cv=skfold,
                                                method='predict_proba', n_jobs=-1)[:, 1]
                except:
                    y_proba = None

                # 평가
                metrics = evaluate_classifier(y, y_pred, y_proba)

                result = {
                    'feature_set': feat_name,
                    'n_features': len(features),
                    'model': model_name,
                    **metrics
                }
                results.append(result)

                print(f"Prec={metrics['Precision']:.4f}, Recall={metrics['Recall']:.4f}, "
                      f"F1={metrics['F1']:.4f}, AUC={metrics['AUC']:.4f}")

            except Exception as e:
                print(f"ERROR: {e}")
                continue

    # 결과 정리
    print("\n[3/3] 결과 정리...")
    results_df = pd.DataFrame(results)

    # F1 기준 정렬
    results_df = results_df.sort_values('F1', ascending=False)

    # 저장
    results_df.to_csv(RESULT_PATH, index=False, encoding='utf-8-sig')

    # 결과 출력
    print("\n" + "=" * 70)
    print("분류 모델 결과 (F1 기준 정렬)")
    print("=" * 70)
    print(f"{'피처':<3} {'모델':<22} {'Acc':>7} {'Prec':>7} {'Recall':>7} {'F1':>7} {'AUC':>7}")
    print("-" * 70)

    for _, row in results_df.iterrows():
        print(f"{row['feature_set']:<3} {row['model']:<22} "
              f"{row['Accuracy']:>7.4f} {row['Precision']:>7.4f} "
              f"{row['Recall']:>7.4f} {row['F1']:>7.4f} {row['AUC']:>7.4f}")

    # 최고 성능
    print("\n" + "=" * 70)
    print("최고 성능 모델")
    print("=" * 70)

    best_f1 = results_df.iloc[0]
    print(f"\n[F1 기준]")
    print(f"  피처셋: {best_f1['feature_set']}, 모델: {best_f1['model']}")
    print(f"  F1: {best_f1['F1']:.4f}, Precision: {best_f1['Precision']:.4f}, Recall: {best_f1['Recall']:.4f}")
    print(f"  TP: {int(best_f1['TP'])}, FP: {int(best_f1['FP'])}, TN: {int(best_f1['TN'])}, FN: {int(best_f1['FN'])}")

    best_prec = results_df.sort_values('Precision', ascending=False).iloc[0]
    print(f"\n[Precision 기준]")
    print(f"  피처셋: {best_prec['feature_set']}, 모델: {best_prec['model']}")
    print(f"  Precision: {best_prec['Precision']:.4f}, Recall: {best_prec['Recall']:.4f}")

    best_auc = results_df.sort_values('AUC', ascending=False).iloc[0]
    print(f"\n[AUC 기준]")
    print(f"  피처셋: {best_auc['feature_set']}, 모델: {best_auc['model']}")
    print(f"  AUC: {best_auc['AUC']:.4f}")

    # 회귀 모델과 비교
    print("\n" + "=" * 70)
    print("회귀 모델 vs 분류 모델 비교")
    print("=" * 70)
    print(f"{'항목':<15} {'회귀(튜닝후)':>12} {'분류':>12} {'차이':>10}")
    print("-" * 50)
    print(f"{'F1 (최고)':<15} {'0.5819':>12} {best_f1['F1']:>12.4f} {(best_f1['F1']-0.5819)*100:>+10.2f}%p")
    print(f"{'Precision':<15} {'0.4418':>12} {best_prec['Precision']:>12.4f} {(best_prec['Precision']-0.4418)*100:>+10.2f}%p")
    print(f"{'AUC':<15} {'0.6118':>12} {best_auc['AUC']:>12.4f} {(best_auc['AUC']-0.6118)*100:>+10.2f}%p")

    print(f"\n결과 저장: {RESULT_PATH}")

    return results_df


if __name__ == "__main__":
    start_time = datetime.now()
    results_df = main()
    elapsed = datetime.now() - start_time
    print(f"\n완료! (소요시간: {elapsed})")
