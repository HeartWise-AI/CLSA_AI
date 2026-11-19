"""npj Step 1 (REANALYSIS_PLAN §8.1): ECG combined with guideline scores, lipids added to Model A+.
Frozen v2.9 inputs only; logistic combinations fitted on development data, applied once to the CLSA test set."""
from pathlib import Path
import json
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
import statsmodels.api as sm
from statsmodels.nonparametric.smoothers_lowess import lowess
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'revision'))  # remaining_analyses.delong_ci
from remaining_analyses import delong_ci

NAS = Path('/media/data1/ravram/CLSA_validation_files/manuscript')
OUT = Path(__import__('os').environ.get('CLSA_WORKSPACE', '/volume/CLSA')) / 'outputs/revision/npj'; OUT.mkdir(parents=True, exist_ok=True)
lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))

sc = pd.read_parquet(NAS / 'data_v4/scores_v4.parquet')
dev = pd.concat([sc[sc.split == s].sort_values('row_in_split').reset_index() for s in ['train', 'val']], ignore_index=True)
dev['ecg'] = lg(pd.read_parquet(NAS / 'results_v5/dev_oof_wcr_calibrated.parquet').reindex(dev.entity_id.values).cal_ecgo.values)
te = pd.read_parquet(NAS / 'results_v5/test_predictions_wcr.parquet'); te['ecg'] = lg(te['DeepECG-SSL ECG-only (calibrated)'].values)
L = pd.read_csv(NAS / 'incident_cvd_labels_v3_guideline_components.csv', index_col=0)
for d in (dev, te):
    d['htn'] = d.hypertension_dx
    for c in ['frs', 'pce', 'score2', 'prevent_cvd_winsor', 'prevent_ascvd_winsor']: d[f's_{c}'] = lg(d[c].values)
    for e in ['specific_F2', 'hard_F2']: d[e] = L.reindex(d.entity_id.values)[e].values
assert len(te) == 3906 and te.incident_F2_primary.sum() == 394

AP = ['ecg', 'age', 'female', 'htn']
SCORES = {'FRS': 's_frs', 'PREVENT total CVD': 's_prevent_cvd_winsor', 'PREVENT-ASCVD': 's_prevent_ascvd_winsor',
          'SCORE2/OP': 's_score2', 'PCE': 's_pce'}
MODELS = {'Model A+': AP, 'A+ + lipids': AP + ['tc_mmol', 'hdl_mmol']}
for k, c in SCORES.items(): MODELS[f'{k} recal'] = [c]; MODELS[f'{k} + ECG'] = [c, 'ecg']
COMP = [('A+ + lipids', 'Model A+')] + [(f'{k} + ECG', f'{k} recal') for k in SCORES] + [(f'{k} + ECG', 'Model A+') for k in SCORES]
REC = {'incident_F2_primary': 'FRS', 'specific_F2': 'FRS', 'hard_F2': 'SCORE2/OP'}  # interpretation-rule reference
TH = np.round(np.arange(0.05, 0.2001, 0.01), 2)

def calib(y, p):
    x = lg(p); o = sm.GLM(y, np.ones_like(x), family=sm.families.Binomial(), offset=x).fit()
    s = sm.GLM(y, sm.add_constant(x), family=sm.families.Binomial()).fit()
    sm_ = lowess(y, p, frac=2 / 3, it=0, return_sorted=False)
    return dict(cal_intercept=o.params[0], cal_slope=s.params[1], ICI=float(np.mean(abs(sm_ - p))), brier=brier_score_loss(y, p))

def nb(y, p, t): n = len(y); tp = np.sum((p >= t) & (y == 1)); fp = np.sum((p >= t) & (y == 0)); return tp / n - fp / n * t / (1 - t)

rng = np.random.default_rng(2026); res = {}; rows, crows, drows = [], [], []
for ep in ['incident_F2_primary', 'specific_F2', 'hard_F2']:
    kd = dev[ep].notna().values & dev[[c for v in MODELS.values() for c in v]].notna().all(1).values
    kt = te[ep].notna().values & te[[c for v in MODELS.values() for c in v]].notna().all(1).values
    Xd, Xt = dev[kd], te[kt]; yd, yt = Xd[ep].values.astype(int), Xt[ep].values.astype(int); n = len(yt)
    P = {m: LogisticRegression(max_iter=5000).fit(Xd[c], yd).predict_proba(Xt[c])[:, 1] for m, c in MODELS.items()}
    B = [b for b in (rng.integers(0, n, n) for _ in range(1000)) if 0 < yt[b].sum() < n]
    A = {m: np.array([roc_auc_score(yt[b], p[b]) for b in B]) for m, p in P.items()}
    NB10 = {m: np.array([nb(yt[b], p[b], 0.10) for b in B]) for m, p in P.items()}
    for m, p in P.items():
        a, lo, hi = delong_ci(yt, p)
        rows.append(dict(endpoint=ep, model=m, n=n, events=int(yt.sum()), AUROC=a, lo=np.percentile(A[m], 2.5), hi=np.percentile(A[m], 97.5),
                         delong_lo=lo, delong_hi=hi, NB10=nb(yt, p, .10), **calib(yt, p)))
        for t in TH: drows.append(dict(endpoint=ep, model=m, threshold=t, net_benefit=nb(yt, p, t)))
    for t in TH: drows += [dict(endpoint=ep, model='Treat all', threshold=t, net_benefit=yt.mean() - (1 - yt.mean()) * t / (1 - t)),
                           dict(endpoint=ep, model='Treat none', threshold=t, net_benefit=0.0)]
    for a_, b_ in COMP:
        d = A[a_] - A[b_]; dn = NB10[a_] - NB10[b_]
        crows.append(dict(endpoint=ep, comparison=f'{a_} vs {b_}', dAUROC=roc_auc_score(yt, P[a_]) - roc_auc_score(yt, P[b_]),
                          d_lo=np.percentile(d, 2.5), d_hi=np.percentile(d, 97.5), p_boot=min(1, 2 * min((d <= 0).mean(), (d >= 0).mean())),
                          dNB10=nb(yt, P[a_], .1) - nb(yt, P[b_], .1), dNB10_lo=np.percentile(dn, 2.5), dNB10_hi=np.percentile(dn, 97.5)))
    r = REC[ep]; c = [x for x in crows if x['endpoint'] == ep and x['comparison'] == f'{r} + ECG vs {r} recal'][0]
    res[ep] = dict(reference_score=r, overlap_rule_met=bool(c['d_hi'] < 0.02), d=c['dAUROC'], d_hi=c['d_hi'])

pd.DataFrame(rows).to_csv(OUT / 'step1_models.csv', index=False)
pd.DataFrame(crows).to_csv(OUT / 'step1_comparisons.csv', index=False)
pd.DataFrame(drows).to_csv(OUT / 'step1_dca.csv', index=False)
(OUT / 'step1_overlap_rule.json').write_text(json.dumps(res, indent=2) + '\n')
pd.set_option('display.width', 250)
print(pd.DataFrame(rows)[['endpoint', 'model', 'events', 'AUROC', 'lo', 'hi', 'cal_intercept', 'cal_slope', 'ICI', 'NB10']].round(3).to_string(index=False))
print(pd.DataFrame(crows).round(3).to_string(index=False)); print(json.dumps(res, indent=1))
