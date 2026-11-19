"""npj Step 5 (REANALYSIS_PLAN §8.5): CLSA loss before F2 among baseline CVD-free participants, by reason."""
from pathlib import Path
import json, numpy as np, pandas as pd
NAS = Path('/media/data1/ravram/CLSA_validation_files/manuscript'); OUT = Path(__import__('os').environ.get('CLSA_WORKSPACE', '/volume/CLSA')) / 'outputs/revision/npj'
R = '/media/data1/mbahani/CLSA/CLSA_ALL/CLSA_Data/CLSA/2401002_MontrealHeartInstitute_RAvram_Mortality/'
L = pd.read_csv(NAS / 'incident_cvd_labels_v3_guideline_components.csv', index_col=0)
ps = pd.read_csv(R + '2401002_MontrealHeartInstitute_RAvram_ParticipantStatus_CoP_v3_Sep2022.csv').set_index('entity_id').reindex(L.index)
dq = pd.read_csv(R + '2401002_MontrealHeartInstitute_RAvram_DecedentQuestionnaire_Sep2022.csv').set_index('entity_id').reindex(L.index)
sc = pd.read_parquet(NAS / 'data_v4/scores_v4.parquet')
d2 = pd.to_datetime(ps.clsa_fup2_date.astype(str).replace({'-99999': None}), errors='coerce')
dd = pd.to_datetime(ps.date_death.astype(str).replace({'-99999': None, '-88888': None}), errors='coerce')
miss = L.incident_F2_primary.isna()
died_before_f2 = (ps.death == 1) & (ps.clsa_fup2 != 1)
cvd_cause = dq.DSD_MCAUS_DCQ.isin([2, 3, 4])
reason = np.select([~miss, died_before_f2 & cvd_cause, died_before_f2, ps.withdrawn == 1, ps.clsa_fup2 != 1],
                   ['outcome known', 'died before F2, CVD cause (labels already count these as events)', 'died before F2, non-CVD or unknown cause',
                    'withdrew', 'alive, no F2 visit'], 'F2 visit but indeterminate answers')
res = {}
for nm, ids in [('all baseline CVD-free with ECG labels', L.index), ('modelling cohort (scores_v4)', pd.Index(sc.index if 'entity_id' not in sc else sc.entity_id).intersection(L.index)),
                ('test set', pd.Index(sc[sc.split == 'test'].index if 'entity_id' not in sc else sc[sc.split == 'test'].entity_id).intersection(L.index))]:
    r = pd.Series(reason, index=L.index).reindex(ids)
    res[nm] = dict(n=len(ids), **{k: int(v) for k, v in r.value_counts().items()}, pct_missing=float((r != 'outcome known').mean()))
(OUT / 'step5_clsa_loss.json').write_text(json.dumps(res, indent=2) + '\n'); print(json.dumps(res, indent=1))

# 5b (loss 16.9% > 10% rule): IPCW-weighted test AUROC. P(outcome observed | baseline covariates) is fitted on all
# baseline CVD-free participants; observed test participants are weighted by 1 / P. ECG predictions do not exist for
# lost participants, so the discrete-time model is not run (needs inference on their ECGs; left to the PI).
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
rf = pd.read_parquet(NAS / 'baseline_risk_factors.parquet').reindex(L.index)
X = rf[['age', 'female', 'sbp', 'bp_treated', 'diabetes', 'smoker_current', 'bmi', 'hypertension_dx', 'egfr']].astype(float)
X = X.fillna(X.median()).assign(**{f'prov_{p}': (rf.province == p).astype(float) for p in rf.province.dropna().unique()[1:]})
obs = (~miss).astype(int).values
pobs = pd.Series(LogisticRegression(max_iter=5000).fit((X - X.mean()) / X.std(), obs).predict_proba((X - X.mean()) / X.std())[:, 1], index=L.index)
w_ = pd.read_parquet(NAS / 'results_v5/test_predictions_wcr.parquet'); s_ = pd.read_parquet(NAS / 'results_v5/test_predictions_simple.parquet')
t = w_[['entity_id', 'incident_F2_primary', 'FRS (recal)', 'PREVENT (recal)']].merge(s_[['entity_id', 'DeepECG-SSL + age + sex + hypertension']], on='entity_id')
t['w'] = 1 / t.entity_id.map(pobs).values; y = t.incident_F2_primary.values.astype(int)
M = {'Model A+': 'DeepECG-SSL + age + sex + hypertension', 'FRS': 'FRS (recal)', 'PREVENT': 'PREVENT (recal)'}
rng = np.random.default_rng(2026); n = len(t); B = [rng.integers(0, n, n) for _ in range(1000)]
ipcw = {'weights': dict(min=float(t.w.min()), median=float(t.w.median()), max=float(t.w.max())), 'auroc': {}}
for m, c in M.items():
    ipcw['auroc'][m] = dict(unweighted=roc_auc_score(y, t[c]), ipcw=roc_auc_score(y, t[c], sample_weight=t.w))
for m in ['FRS', 'PREVENT']:
    d = [roc_auc_score(y[i], t[M['Model A+']].values[i], sample_weight=t.w.values[i]) - roc_auc_score(y[i], t[M[m]].values[i], sample_weight=t.w.values[i]) for i in B]
    ipcw['auroc'][f'Model A+ minus {m} (ipcw)'] = dict(d=ipcw['auroc']['Model A+']['ipcw'] - ipcw['auroc'][m]['ipcw'], lo=float(np.percentile(d, 2.5)), hi=float(np.percentile(d, 97.5)))
res['ipcw_test'] = ipcw; (OUT / 'step5_clsa_loss.json').write_text(json.dumps(res, indent=2) + '\n'); print(json.dumps(ipcw, indent=1))
