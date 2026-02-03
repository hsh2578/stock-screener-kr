"""
피처 분석: 상관관계 및 다중공선성(VIF) 분석

1. 상관관계 히트맵
2. VIF (Variance Inflation Factor) 분석
3. 피처 선택 권장안
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from statsmodels.stats.outliers_influence import variance_inflation_factor

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 경로 설정
ML_DIR = Path(__file__).parent.parent
DATA_PATH = ML_DIR / "data" / "training_data.csv"
OUTPUT_DIR = ML_DIR / "data"

# 피처 목록
FEATURES = [
    'breakout_pct', 'volume_surge', 'close_strength',  # A. 돌파 품질
    'base_length', 'volatility_contraction', 'volume_dry_up',  # B. 베이스 품질
    'ma200_slope', 'pct_above_52w_low',  # C. 추세 상태
    'rs_vs_market', 'market_return',  # D. 상대강도
    'ma20_deviation', 'liquidity',  # E. 리스크
    'days_since_ath', 'atr_ratio'  # F. 저항선/변동성
]


def load_data():
    """데이터 로드"""
    df = pd.read_csv(DATA_PATH)
    print(f"데이터 로드: {len(df)}개 샘플")
    return df


def analyze_correlation(df):
    """상관관계 분석"""
    print("\n" + "=" * 60)
    print("1. 상관관계 분석")
    print("=" * 60)

    # 피처 데이터
    X = df[FEATURES].copy()

    # 상관관계 행렬
    corr_matrix = X.corr()

    # 히트맵 저장
    plt.figure(figsize=(14, 12))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f',
                cmap='RdBu_r', center=0, vmin=-1, vmax=1,
                square=True, linewidths=0.5)
    plt.title('피처 상관관계 히트맵', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'correlation_heatmap.png', dpi=150)
    plt.close()
    print(f"\n히트맵 저장: {OUTPUT_DIR / 'correlation_heatmap.png'}")

    # 높은 상관관계 쌍 찾기 (|r| > 0.7)
    print("\n높은 상관관계 (|r| > 0.7):")
    print("-" * 50)
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > 0.7:
                pair = (corr_matrix.columns[i], corr_matrix.columns[j], corr_val)
                high_corr_pairs.append(pair)
                print(f"  {pair[0]:25} ↔ {pair[1]:25} : {pair[2]:+.3f}")

    if not high_corr_pairs:
        print("  없음")

    return corr_matrix, high_corr_pairs


def analyze_vif(df):
    """다중공선성 분석 (VIF)"""
    print("\n" + "=" * 60)
    print("2. 다중공선성 분석 (VIF)")
    print("=" * 60)
    print("\nVIF 기준:")
    print("  - VIF < 5  : 문제 없음")
    print("  - VIF 5~10 : 주의 필요")
    print("  - VIF > 10 : 심각한 다중공선성 (제거 권장)")

    # 피처 데이터 (결측치 제거, 무한값 처리)
    X = df[FEATURES].copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.dropna()

    # liquidity 로그 변환 (스케일 문제)
    X['liquidity'] = np.log1p(X['liquidity'])

    # VIF 계산
    vif_data = []
    for i, feature in enumerate(X.columns):
        try:
            vif = variance_inflation_factor(X.values, i)
            vif_data.append({'feature': feature, 'VIF': vif})
        except:
            vif_data.append({'feature': feature, 'VIF': np.nan})

    vif_df = pd.DataFrame(vif_data).sort_values('VIF', ascending=False)

    print("\nVIF 결과:")
    print("-" * 50)
    for _, row in vif_df.iterrows():
        vif_val = row['VIF']
        feature = row['feature']
        if vif_val > 10:
            status = "[X] 제거 권장"
        elif vif_val > 5:
            status = "[!] 주의"
        else:
            status = "[O] 양호"
        print(f"  {feature:25} : {vif_val:8.2f}  {status}")

    return vif_df


def analyze_target_correlation(df):
    """타겟 변수와의 상관관계"""
    print("\n" + "=" * 60)
    print("3. 타겟(max_gain_20d)과의 상관관계")
    print("=" * 60)

    X = df[FEATURES].copy()
    y = df['max_gain_20d']

    # 상관관계 계산
    target_corr = X.corrwith(y).sort_values(key=abs, ascending=False)

    print("\n피처별 상관관계 (절대값 순):")
    print("-" * 50)
    for feature, corr in target_corr.items():
        bar = "#" * int(abs(corr) * 30)
        sign = "+" if corr > 0 else "-"
        print(f"  {feature:25} : {sign}{abs(corr):.3f} {bar}")

    # 막대 그래프 저장
    plt.figure(figsize=(10, 8))
    colors = ['green' if x > 0 else 'red' for x in target_corr.values]
    target_corr.plot(kind='barh', color=colors)
    plt.xlabel('상관계수')
    plt.title('피처와 max_gain_20d 상관관계', fontsize=14, fontweight='bold')
    plt.axvline(x=0, color='black', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'target_correlation.png', dpi=150)
    plt.close()
    print(f"\n차트 저장: {OUTPUT_DIR / 'target_correlation.png'}")

    return target_corr


def recommend_features(corr_matrix, vif_df, target_corr):
    """피처 선택 권장안"""
    print("\n" + "=" * 60)
    print("4. 피처 선택 권장안")
    print("=" * 60)

    # 제거 후보
    remove_candidates = set()

    # VIF > 10인 피처
    high_vif = vif_df[vif_df['VIF'] > 10]['feature'].tolist()
    remove_candidates.update(high_vif)

    # 상관관계 |r| > 0.8 중 타겟 상관관계 낮은 쪽
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > 0.8:
                feat1, feat2 = corr_matrix.columns[i], corr_matrix.columns[j]
                # 타겟과 상관관계 낮은 쪽 제거
                if abs(target_corr[feat1]) < abs(target_corr[feat2]):
                    remove_candidates.add(feat1)
                else:
                    remove_candidates.add(feat2)

    # 권장 피처
    recommended = [f for f in FEATURES if f not in remove_candidates]

    print("\n제거 권장 피처:")
    for f in remove_candidates:
        reason = []
        if f in high_vif:
            reason.append(f"VIF > 10")
        reason.append(f"타겟 상관: {target_corr.get(f, 0):.3f}")
        print(f"  [X] {f:25} ({', '.join(reason)})")

    print(f"\n권장 피처 ({len(recommended)}개):")
    for f in recommended:
        print(f"  [O] {f:25} (타겟 상관: {target_corr.get(f, 0):+.3f})")

    return recommended, list(remove_candidates)


def main():
    """메인 실행"""
    print("=" * 60)
    print("피처 분석: 상관관계 및 다중공선성")
    print("=" * 60)

    # 데이터 로드
    df = load_data()

    # 1. 상관관계 분석
    corr_matrix, high_corr_pairs = analyze_correlation(df)

    # 2. VIF 분석
    vif_df = analyze_vif(df)

    # 3. 타겟 상관관계
    target_corr = analyze_target_correlation(df)

    # 4. 피처 권장안
    recommended, removed = recommend_features(corr_matrix, vif_df, target_corr)

    # 결과 저장
    result = {
        'recommended_features': recommended,
        'removed_features': removed,
        'total_features': len(FEATURES),
        'final_features': len(recommended)
    }

    print("\n" + "=" * 60)
    print("분석 완료!")
    print("=" * 60)
    print(f"원본 피처: {len(FEATURES)}개")
    print(f"권장 피처: {len(recommended)}개")
    print(f"제거 피처: {len(removed)}개")

    return result


if __name__ == "__main__":
    main()
