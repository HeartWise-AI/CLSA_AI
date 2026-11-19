"""Shared data and statistics for the post hoc A1-A4 outcome-validity analyses (plan locked 2026-10-26,
goal_A1_A4_outcome_validity.md). Frozen predictions only: test = stored test predictions; pooled = development
out-of-fold predictions regenerated exactly as in simple_models_primary.py (same folds, seed 42) + test."""
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

M = '/media/data1/ravram/CLSA_validation_files/manuscript'
RAW = '/media/data1/mbahani/CLSA/CLSA_ALL/CLSA_Data/CLSA/2401002_MontrealHeartInstitute_RAvram_'
APLUS = 'DeepECG-SSL + age + sex + hypertension'
MODELS = {'A+': 'aplus', 'FRS': 'frs', 'PREVENT': 'prevent_cvd_winsor'}
SEED, B = 2026, 1000
lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))

def samples():
    """Return dict test/pooled -> DataFrame indexed by entity_id with y, components, aplus, frs, prevent."""
    sc = pd.read_parquet(f'{M}/data_v4/scores_v4.parquet')
    dev = pd.concat([sc[sc.split == s].sort_values('row_in_split').reset_index() for s in ['train', 'val']], ignore_index=True)
    dev['ecg'] = lg(pd.read_parquet(f'{M}/results_v5/dev_oof_wcr_calibrated.parquet').reindex(dev.entity_id.values).cal_ecgo.values)
    dev['htn'] = dev.hypertension_dx
    yd = dev.incident_F2_primary.values.astype(int)
    fid = np.zeros(len(dev), int)
    for k, (_, i) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(dev, yd)): fid[i] = k
    cols = ['ecg', 'age', 'female', 'htn']
    dev['aplus'] = np.nan
    for j in range(5):
        dev.loc[fid == j, 'aplus'] = LogisticRegression(max_iter=3000).fit(dev.loc[fid != j, cols], yd[fid != j]).predict_proba(dev.loc[fid == j, cols])[:, 1]
    te = pd.read_parquet(f'{M}/results_v5/test_predictions_wcr.parquet')[['entity_id', 'incident_F2_primary', 'frs', 'prevent_cvd_winsor']]
    te = te.merge(pd.read_parquet(f'{M}/results_v5/test_predictions_simple.parquet')[['entity_id', APLUS]].rename(columns={APLUS: 'aplus'}), on='entity_id')
    comp = pd.read_csv(f'{M}/incident_cvd_labels_v3_guideline_components.csv').set_index('entity_id')
    keep = ['entity_id', 'incident_F2_primary', 'aplus', 'frs', 'prevent_cvd_winsor']
    out = {}
    for nm, d in [('test', te[keep]), ('pooled', pd.concat([dev[keep], te[keep]], ignore_index=True))]:
        d = d.set_index('entity_id').join(comp[['specific_F2', 'F1_HEART', 'F12_HEART', 'cvd_death_before_F2']])
        d['y'] = d.incident_F2_primary.astype(int)
        d['hd_only'] = (d.y == 1) & (d.specific_F2 != 1)
        d['specific'] = (d.y == 1) & (d.specific_F2 == 1)
        out[nm] = d
    return out

def aucs(y, d):
    return {k: roc_auc_score(y, d[c].to_numpy()) for k, c in MODELS.items()}

def evaluate(d, y, label, boot=None):
    """events/n, AUROC A+/FRS/PREVENT, delta A+ - FRS and A+ - PREVENT, paired bootstrap 95% CIs."""
    y = np.asarray(y).astype(int); P = {k: d[c].to_numpy() for k, c in MODELS.items()}
    rng = np.random.default_rng(SEED); n = len(y)
    pt = {k: roc_auc_score(y, p) for k, p in P.items()}
    bs = []
    for _ in range(B):
        i = rng.integers(0, n, n)
        if y[i].min() == y[i].max(): continue
        bs.append({k: roc_auc_score(y[i], p[i]) for k, p in P.items()})
    bs = pd.DataFrame(bs)
    q = lambda s: np.percentile(s, [2.5, 97.5])
    r = {'analysis': label, 'events': int(y.sum()), 'n': n}
    for k in P: r[f'AUROC {k}'] = pt[k]; r[f'AUROC {k} lo'], r[f'AUROC {k} hi'] = q(bs[k])
    for o in ['FRS', 'PREVENT']:
        dd = bs['A+'] - bs[o]; r[f'Delta vs {o}'] = pt['A+'] - pt[o]; r[f'Delta vs {o} lo'], r[f'Delta vs {o} hi'] = q(dd)
    r['boot_draws'] = len(bs)
    if boot is not None: boot[label] = bs
    return r
