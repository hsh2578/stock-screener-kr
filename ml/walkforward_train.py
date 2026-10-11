"""
워크포워드(시간순) 평가 + 최종 모델 저장. 입력: ml/build_events.py 가 만든 events_*.csv

평가: 테스트 연도 Y 마다 학습 = (Y 1월 1일 - 30영업일) 이전 이벤트 전부 (확장 윈도우, 라벨 20거래일 엠바고),
      테스트 = Y 연도 이벤트. 스케일러는 Pipeline 안에서 학습 폴드에만 fit. 하이퍼파라미터는 기존 값 그대로 (튜닝 없음).
지표: n, 양성비율, AUC, 상위 10% 적중률·lift, Brier (+ 학습 양성비율 상수 예측 Brier), 단순 휴리스틱 비교.
판정: 평균 OOS AUC < 0.55 또는 평균 상위10% lift < 1.2 → 'hide' 권고, 모델 파일은 덮어쓰지 않음 (--force-save 로 저장).

사용: python ml/walkforward_train.py [--force-save]
"""
import argparse
import json
import pickle
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.linear_model import Lasso, LogisticRegression, Ridge
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ML = Path(__file__).resolve().parent
sys.path.insert(0, str(ML))
sys.path.insert(0, str(ML / "src"))
from train_final_box_breakout_model import FEATURES_10, FEATURES_15, RIDGE_PARAMS, XGBOOST_PARAMS  # noqa: E402
import final_model as m52  # noqa: E402
import train_near_high as mnh  # noqa: E402
from lightgbm import LGBMClassifier, LGBMRegressor  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402

DATA, MODELS = ML / "data", ML / "models"
EMBARGO = pd.tseries.offsets.BDay(30)  # >= 20 거래일 (라벨 구간)
FIRST_TEST_YEAR = 2015

SPECS = {
    'box': dict(cls_feats=FEATURES_15, reg_feats=FEATURES_10,
                cls=XGBClassifier(**XGBOOST_PARAMS), reg=Ridge(**RIDGE_PARAMS, max_iter=5000),
                heuristics=['volume_surge', 'breakout_strength', 'atr_ratio'],
                label_desc='max(High[t+1..t+20]) / Close[t] - 1 >= 15%'),
    '52w': dict(cls_feats=m52.FEATURES, reg_feats=m52.FEATURES,
                cls=LGBMClassifier(**m52.CLASSIFIER_PARAMS), reg=LGBMRegressor(**m52.REGRESSOR_PARAMS),
                heuristics=['volume_surge', 'breakout_pct', 'atr_ratio'],
                label_desc='max(Close[t+1..t+20]) / Close[t] - 1 >= 15%'),
    'near_high': dict(cls_feats=mnh.FEATURES, reg_feats=mnh.FEATURES,
                      cls=LogisticRegression(**mnh.CLASSIFIER_PARAMS), reg=Lasso(**mnh.REGRESSOR_PARAMS),
                      heuristics=['gap_pct', 'momentum_5d', 'atr_pct'],
                      label_desc='any Close[t+1..t+10] > 52w_high x 1.015 (52w_high = max High[t-265, t-15))'),
}


def ai_score(prob, gain):
    """run_screeners 의 AI 점수 공식 (확률 0~1)."""
    return prob * 100 * 0.7 + np.clip(gain * 3, 0, 100) * 0.3


def top_decile(y, s):
    k = max(1, int(np.ceil(len(y) * 0.1)))
    return y[np.argsort(-s, kind='stable')[:k]].mean()


def score_block(y, s):
    p10 = top_decile(y, s)
    return {'auc': roc_auc_score(y, s), 'p10': p10, 'lift10': p10 / y.mean()}


def walk_forward(df, spec):
    years = sorted(df['date'].dt.year.unique())
    rows = []
    for Y in [y for y in years if y >= FIRST_TEST_YEAR]:
        start = pd.Timestamp(f"{Y}-01-01")
        tr = df[df['date'] < start - EMBARGO]
        te = df[df['date'].dt.year == Y]
        if len(te) < 50 or tr['label'].nunique() < 2 or te['label'].nunique() < 2:
            continue
        cls = make_pipeline(StandardScaler(), clone(spec['cls'])).fit(tr[spec['cls_feats']], tr['label'])
        reg = make_pipeline(StandardScaler(), clone(spec['reg'])).fit(tr[spec['reg_feats']], tr['max_gain_20d'])
        y = te['label'].values
        p = cls.predict_proba(te[spec['cls_feats']])[:, 1]
        g = reg.predict(te[spec['reg_feats']])
        r = {'year': int(Y), 'n_train': len(tr), 'train_end': str((start - EMBARGO).date()), 'n': len(te),
             'base_rate': y.mean(), **score_block(y, p),
             'brier': brier_score_loss(y, p), 'brier_const': brier_score_loss(y, np.full(len(y), tr['label'].mean())),
             'reg_spearman': spearmanr(g, te['max_gain_20d']).statistic}
        r.update({f'ai_{k}': v for k, v in score_block(y, ai_score(p, g)).items()})
        for h in spec['heuristics']:
            r.update({f'{h}_{k}': v for k, v in score_block(y, te[h].values).items()})
        if 'vol_screener' in te:  # 박스권 돌파(거래량) 스크리너 부분집합
            m = te['vol_screener'].values == 1
            if m.sum() >= 30 and 0 < y[m].mean() < 1:
                r.update({f'volsub_{k}': v for k, v in score_block(y[m], p[m]).items()}, volsub_n=int(m.sum()),
                         volsub_base=y[m].mean())
        rows.append(r)
    return pd.DataFrame(rows)


def save_final(name, df, spec, meta):
    X_c, X_r = df[spec['cls_feats']].values, df[spec['reg_feats']].values  # 추론은 numpy 배열로 넣는다
    if name == 'box':
        sc_c, sc_r = StandardScaler().fit(X_c), StandardScaler().fit(X_r)
        cls = clone(spec['cls']).fit(sc_c.transform(X_c), df['label'])
        reg = clone(spec['reg']).fit(sc_r.transform(X_r), df['max_gain_20d'])
        joblib.dump({'model': reg, 'scaler': sc_r, 'features': spec['reg_feats'], 'params': RIDGE_PARAMS},
                    MODELS / 'box_breakout_regression_model.joblib')
        joblib.dump({'model': cls, 'scaler': sc_c, 'features': spec['cls_feats'], 'params': XGBOOST_PARAMS},
                    MODELS / 'box_breakout_classification_model.joblib')
        (MODELS / 'box_breakout_model_metadata.json').write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding='utf-8')
        return
    d = MODELS / ('52w_high' if name == '52w' else 'near_high_52w')
    sc = StandardScaler().fit(X_c)  # 회귀·분류 같은 피처 → 스케일러 1개 (run_screeners 로드 형식)
    cls = clone(spec['cls']).fit(sc.transform(X_c), df['label'])
    reg = clone(spec['reg']).fit(sc.transform(X_r), df['max_gain_20d'])
    for fn, obj in (('classifier.pkl', cls), ('regressor.pkl', reg), ('scaler.pkl', sc),
                    ('model_meta.pkl', {'features': spec['cls_feats'], **meta})):
        with open(d / fn, 'wb') as f:
            pickle.dump(obj, f)
    (d / 'metadata.json').write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding='utf-8')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--force-save', action='store_true', help='판정과 무관하게 최종 모델 저장')
    a = ap.parse_args()
    pd.set_option('display.width', 250, 'display.max_columns', 40)
    summary = {}
    for name, spec in SPECS.items():
        df = pd.read_csv(DATA / f'events_{name}.csv', parse_dates=['date'], dtype={'ticker': str})
        df = df.dropna(subset=spec['cls_feats'] + ['label']).sort_values('date').reset_index(drop=True)
        wf = walk_forward(df, spec)
        mean = wf.mean(numeric_only=True)
        hide = bool(mean['auc'] < 0.55 or mean['lift10'] < 1.2)
        print(f"\n===== {name}: {len(df)} 이벤트, {df['ticker'].nunique()} 종목 "
              f"(상폐 {df.loc[df['delisted'], 'ticker'].nunique()}), {df['date'].min().date()}~{df['date'].max().date()}")
        cols = ['year', 'n', 'base_rate', 'auc', 'p10', 'lift10', 'brier', 'brier_const', 'ai_auc', 'ai_lift10', 'reg_spearman'] \
            + [f'{h}_{k}' for h in spec['heuristics'] for k in ('auc', 'lift10')] \
            + [c for c in ('volsub_n', 'volsub_auc', 'volsub_lift10') if c in wf]
        print(wf[cols].round(3).to_string(index=False))
        print("평균:", mean[cols[1:]].round(3).to_dict())
        print(f"lift<1.2 연도 {int((wf['lift10'] < 1.2).sum())}/{len(wf)}, AUC<0.55 연도 {int((wf['auc'] < 0.55).sum())}/{len(wf)} → {'HIDE 권고' if hide else '유지 가능'}")
        meta = {
            'model': name, 'created_at': datetime.now().isoformat(timespec='seconds'),
            'events_csv': f'ml/data/events_{name}.csv', 'n_events': len(df), 'n_tickers': int(df['ticker'].nunique()),
            'n_delisted_tickers': int(df.loc[df['delisted'], 'ticker'].nunique()),
            'event_range': [str(df['date'].min().date()), str(df['date'].max().date())],
            'label': spec['label_desc'], 'base_rate_all': float(df['label'].mean()),
            'cls_features': spec['cls_feats'], 'reg_features': spec['reg_feats'],
            'evaluation': 'walk-forward by calendar year, expanding train, 30 business-day embargo, scaler fit on train fold only, no tuning (out-of-sample)',
            'walk_forward': json.loads(wf.to_json(orient='records')),
            'walk_forward_mean': json.loads(mean.to_json()),
            'decision_rule': 'hide if mean OOS AUC < 0.55 or mean top-decile lift < 1.2',
            'recommend_hide': hide,
        }
        summary[name] = meta
        if not hide or a.force_save:
            save_final(name, df, spec, meta)
            print(f"  → 최종 모델 저장 ({len(df)} 이벤트, 전 기간)")
        else:
            print("  → 모델 파일 미저장 (판정 HIDE)")
    (DATA / 'walkforward_summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')


if __name__ == '__main__':
    main()
