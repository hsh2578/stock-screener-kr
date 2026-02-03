"""
박스권 돌파 피처 분석 및 선택

1. 상관분석 - 타겟과의 상관관계
2. VIF 분석 - 다중공선성
3. Forward Selection - 1개씩 추가하면서 성능 비교
"""

import pandas as pd
import numpy as np
import warnings
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score

warnings.filterwarnings('ignore')

# 경로
BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "data" / "box_breakout_history.csv"

# 전체 피처
FEATURES = [
    'box_range_pct', 'breakout_strength', 'volume_surge', 'volume_dry_up',
    'close_strength', 'volatility_contraction', 'ma20_deviation',
    'breakout_gap', 'ma200_slope', 'pct_above_52w_low', 'days_since_ath',
    'market_return', 'rs_vs_market', 'atr_ratio', 'liquidity'
]


def calculate_vif(X):
    """VIF (Variance Inflation Factor) 계산"""
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    vif_data = pd.DataFrame()
    vif_data["Feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
    return vif_data.sort_values('VIF', ascending=False)


def analyze_correlation(df, features, target):
    """상관분석"""
    print("=" * 80)
    print("[1] 상관분석 - 타겟과의 상관관계")
    print("=" * 80)

    # 타겟과의 상관관계
    correlations = []
    for feat in features:
        corr = df[feat].corr(df[target])
        correlations.append({'feature': feat, 'correlation': abs(corr), 'raw_corr': corr})

    corr_df = pd.DataFrame(correlations).sort_values('correlation', ascending=False)

    print(f"\n타겟({target})과의 상관관계 (절대값 기준 정렬):")
    print(f"{'피처':<25} {'상관계수':>12} {'절대값':>10}")
    print("-" * 50)
    for _, row in corr_df.iterrows():
        print(f"{row['feature']:<25} {row['raw_corr']:>12.4f} {row['correlation']:>10.4f}")

    return corr_df


def analyze_vif(df, features):
    """VIF 분석 - 다중공선성"""
    print("\n\n" + "=" * 80)
    print("[2] VIF 분석 - 다중공선성 확인")
    print("=" * 80)

    X = df[features].copy()

    # 표준화 (VIF 계산 시 스케일 영향 제거)
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        columns=X.columns
    )

    vif_df = calculate_vif(X_scaled)

    print(f"\n다중공선성 분석 (VIF > 10은 문제 있음):")
    print(f"{'피처':<25} {'VIF':>12} {'상태':>10}")
    print("-" * 50)
    for _, row in vif_df.iterrows():
        status = "[WARN]" if row['VIF'] > 10 else "[OK]"
        print(f"{row['Feature']:<25} {row['VIF']:>12.2f}  {status:>10}")

    high_vif = vif_df[vif_df['VIF'] > 10]
    if len(high_vif) > 0:
        print(f"\n[WARN] VIF > 10인 피처: {len(high_vif)}개")
        print("       이들 피처는 다른 피처와 높은 상관관계가 있습니다.")

    return vif_df


def forward_selection(df, features, target, max_features=15):
    """
    Forward Selection - 피처를 1개씩 추가하면서 성능 비교

    Returns:
        results: 각 단계별 성능 결과
        selected_features: 선택된 피처 리스트
    """
    print("\n\n" + "=" * 80)
    print("[3] Forward Selection - 피처를 1개씩 추가하면서 성능 비교")
    print("=" * 80)

    y = df[target].values
    remaining_features = features.copy()
    selected_features = []
    results = []

    scaler = StandardScaler()
    model = Lasso(alpha=0.1, max_iter=5000, random_state=42)
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    print(f"\n모델: Lasso (alpha=0.1)")
    print(f"평가: 5-Fold Cross-Validation R²")
    print(f"\n{'단계':<6} {'추가 피처':<25} {'누적 피처 수':>12} {'CV R²':>10} {'개선':>10}")
    print("-" * 75)

    best_score = -np.inf

    for step in range(max_features):
        if len(remaining_features) == 0:
            break

        best_feature = None
        best_step_score = -np.inf

        # 남은 피처 중 가장 성능 향상이 큰 피처 찾기
        for feature in remaining_features:
            test_features = selected_features + [feature]
            X = df[test_features].values
            X_scaled = scaler.fit_transform(X)

            # Cross-validation
            scores = cross_val_score(model, X_scaled, y, cv=kfold, scoring='r2')
            score = scores.mean()

            if score > best_step_score:
                best_step_score = score
                best_feature = feature

        # 선택된 피처 추가
        selected_features.append(best_feature)
        remaining_features.remove(best_feature)

        improvement = best_step_score - best_score if step > 0 else best_step_score
        best_score = best_step_score

        results.append({
            'step': step + 1,
            'feature': best_feature,
            'n_features': len(selected_features),
            'cv_r2': best_step_score,
            'improvement': improvement
        })

        status = "[+]" if improvement > 0 or step == 0 else "[-]"
        print(f"{step+1:<6} {best_feature:<25} {len(selected_features):>12} {best_step_score:>10.4f} {improvement:>10.4f} {status}")

    # 최고 성능 찾기
    results_df = pd.DataFrame(results)
    best_idx = results_df['cv_r2'].idxmax()
    best_result = results_df.iloc[best_idx]

    print("\n" + "=" * 80)
    print(f"최고 성능: {best_result['n_features']}개 피처 사용 시 R² = {best_result['cv_r2']:.4f}")
    print("=" * 80)

    optimal_features = selected_features[:int(best_result['n_features'])]
    print(f"\n선택된 피처 ({len(optimal_features)}개):")
    for i, feat in enumerate(optimal_features, 1):
        print(f"  {i}. {feat}")

    return results_df, optimal_features


def analyze_feature_correlation_matrix(df, features):
    """피처 간 상관관계 매트릭스"""
    print("\n\n" + "=" * 80)
    print("[4] 피처 간 상관관계 분석")
    print("=" * 80)

    corr_matrix = df[features].corr()

    # 높은 상관관계 쌍 찾기 (|corr| > 0.7, 자기 자신 제외)
    high_corr_pairs = []
    for i in range(len(features)):
        for j in range(i+1, len(features)):
            corr = corr_matrix.iloc[i, j]
            if abs(corr) > 0.7:
                high_corr_pairs.append({
                    'feature1': features[i],
                    'feature2': features[j],
                    'correlation': corr
                })

    if high_corr_pairs:
        print(f"\n높은 상관관계 쌍 (|상관계수| > 0.7):")
        print(f"{'피처 1':<25} {'피처 2':<25} {'상관계수':>12}")
        print("-" * 65)

        high_corr_df = pd.DataFrame(high_corr_pairs).sort_values('correlation', key=abs, ascending=False)
        for _, row in high_corr_df.iterrows():
            print(f"{row['feature1']:<25} {row['feature2']:<25} {row['correlation']:>12.4f}")

        print(f"\n[WARN] 총 {len(high_corr_pairs)}쌍의 피처가 높은 상관관계를 가집니다.")
    else:
        print("\n[OK] 높은 상관관계(|r| > 0.7)를 가진 피처 쌍이 없습니다.")

    return corr_matrix


def main():
    print("=" * 80)
    print("박스권 돌파 피처 분석 및 선택")
    print("=" * 80)

    # 데이터 로드
    df = pd.read_csv(DATA_PATH, encoding='utf-8-sig')
    print(f"\n데이터: {len(df):,}개 샘플, {len(FEATURES)}개 피처")

    # 1. 상관분석
    corr_df = analyze_correlation(df, FEATURES, 'max_gain_20d')

    # 2. VIF 분석
    vif_df = analyze_vif(df, FEATURES)

    # 3. 피처 간 상관관계
    corr_matrix = analyze_feature_correlation_matrix(df, FEATURES)

    # 4. Forward Selection
    results_df, optimal_features = forward_selection(df, FEATURES, 'max_gain_20d', max_features=15)

    # 결과 저장
    output_path = BASE_DIR / "data" / "feature_analysis_results.csv"
    results_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n\nForward Selection 결과 저장: {output_path}")

    # 요약
    print("\n\n" + "=" * 80)
    print("분석 요약")
    print("=" * 80)

    print("\n[상관분석 상위 5개]")
    for i, row in corr_df.head(5).iterrows():
        print(f"  {i+1}. {row['feature']:<25} (r = {row['raw_corr']:>7.4f})")

    print("\n[VIF 상위 5개 (다중공선성 높음)]")
    for i, row in vif_df.head(5).iterrows():
        print(f"  {i+1}. {row['Feature']:<25} (VIF = {row['VIF']:>7.2f})")

    print(f"\n[Forward Selection 최적 결과]")
    print(f"  최적 피처 수: {len(optimal_features)}개")
    print(f"  최고 R²: {results_df['cv_r2'].max():.4f}")
    print(f"  15개 전부 사용 R²: {results_df['cv_r2'].iloc[-1]:.4f}")
    print(f"  성능 차이: {(results_df['cv_r2'].iloc[-1] - results_df['cv_r2'].max()):.4f}")

    return corr_df, vif_df, results_df, optimal_features


if __name__ == "__main__":
    from datetime import datetime
    start_time = datetime.now()

    try:
        corr_df, vif_df, results_df, optimal_features = main()
        elapsed = datetime.now() - start_time
        print(f"\n완료! (소요시간: {elapsed})")
    except ImportError as e:
        print(f"\n[ERROR] 필요한 패키지가 설치되지 않았습니다.")
        print(f"  pip install statsmodels")
        print(f"\n상세: {e}")
