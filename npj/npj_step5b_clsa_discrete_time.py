"""npj Step 5b (REANALYSIS_PLAN §8.6): CLSA discrete-time (BL-F1, F1-F2) analysis on the test set plus participants with
indeterminate F2 status (frozen ensemble scores from v2/indeterminate). Cause-specific cloglog hazards; Harrell's C."""
from pathlib import Path
import json, numpy as np, pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from lifelines.utils import concordance_index

NAS = Path('/media/data1/ravram/CLSA_validation_files/manuscript'); OUT = Path(__import__('os').environ.get('CLSA_WORKSPACE', '/volume/CLSA')) / 'outputs/revision/npj'
R = '/media/data1/mbahani/CLSA/CLSA_ALL/CLSA_Data/CLSA/2401002_MontrealHeartInstitute_RAvram_Mortality/'
lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
L = pd.read_csv(NAS / 'incident_cvd_labels_v3_guideline_components.csv', index_col=0)
ps = pd.read_csv(R + '2401002_MontrealHeartInstitute_RAvram_ParticipantStatus_CoP_v3_Sep2022.csv').set_index('entity_id')
dt = lambda s: pd.to_datetime(s.astype(str).where(~s.astype(str).isin(['-99999', '-88888'])), errors='coerce', format='mixed')
d1, ddeath = dt(ps.clsa_fup1_date), dt(ps.date_death)

w = pd.read_parquet(NAS / 'results_v5/test_predictions_wcr.parquet'); s = pd.read_parquet(NAS / 'results_v5/test_predictions_simple.parquet')
te = w[['entity_id', 'age', 'female', 'hypertension_dx', 'frs', 'prevent_cvd_winsor']].merge(s[['entity_id', 'DeepECG-SSL + age + sex + hypertension']], on='entity_id')
te = te.rename(columns={'DeepECG-SSL + age + sex + hypertension': 'Aplus'}).assign(group='test')
ind = pd.read_parquet(NAS / 'v2/indeterminate/scores.parquet').reset_index().rename(columns={'index': 'entity_id'})
ind = ind[['entity_id', 'age', 'female', 'hypertension_dx', 'frs', 'prevent_cvd_winsor', 'Aplus']].assign(group='indeterminate')
d = pd.concat([te, ind], ignore_index=True); assert d.entity_id.is_unique

# bedside comparator: logistic on age, sex, hypertension fitted on development data
sc = pd.read_parquet(NAS / 'data_v4/scores_v4.parquet'); dev = sc[sc.split.isin(['train', 'val'])]
bm = LogisticRegression(max_iter=5000).fit(dev[['age', 'female', 'hypertension_dx']], dev.incident_F2_primary.astype(int))
d['bedside'] = bm.predict_proba(d[['age', 'female', 'hypertension_dx']])[:, 1]

lab = L.reindex(d.entity_id); f1, f2 = lab.incident_F1.values, lab.incident_F2_primary.values
cvdd = lab.cvd_death_before_F2.fillna(0).values == 1
death_before_f1 = (ddeath.reindex(d.entity_id) < d1.reindex(d.entity_id)).fillna(False).values | (cvdd & np.isnan(f1))
died = (ps.death.reindex(d.entity_id) == 1).values
rows = []
for i in range(len(d)):
    if f1[i] == 1 or (f2[i] == 1 and cvdd[i] and death_before_f1[i]): rows += [(i, 1, 1)]; continue          # event in P1
    p1_known = f1[i] == 0 or (d.group[i] == 'test' and not np.isnan(f2[i]))
    if not p1_known: continue                                                                                 # censored at baseline
    rows.append((i, 1, 0))
    if f2[i] == 1: rows.append((i, 2, 1))
    elif f2[i] == 0: rows.append((i, 2, 0))
    # f2 unknown -> censored at F1 (non-CVD deaths and losses alike)
pp = pd.DataFrame(rows, columns=['i', 'period', 'event']); pp['p2'] = (pp.period == 2).astype(int)
last = pp.groupby('i').agg(T=('period', 'max'), E=('event', 'max'))
res = {'n_total': len(d), 'n_contributing': int(last.shape[0]), 'n_events': int(last.E.sum()),
       'n_by_group': d.loc[last.index].group.value_counts().to_dict(), 'events_by_group': d.loc[last.index][last.E.values == 1].group.value_counts().to_dict(),
       'indeterminate_contributing_P1_only': int(((d.loc[last.index].group == 'indeterminate') & (last['T'].values == 1)).sum()), 'models': {}}
Z = {'Model A+': lg(d.Aplus.values), 'FRS': lg(d.frs.values), 'PREVENT total CVD': lg(d.prevent_cvd_winsor.values), 'Age + sex + hypertension': lg(d.bedside.values)}
for m, x in Z.items():
    z = (x - x.mean()) / x.std(); X = sm.add_constant(np.column_stack([pp.p2.values, z[pp.i.values]]))
    fit = sm.GLM(pp.event.values, X, family=sm.families.Binomial(link=sm.families.links.CLogLog())).fit()
    hr, (lo, hi) = np.exp(fit.params[2]), np.exp(fit.conf_int()[2])
    res['models'][m] = dict(HR_perSD=hr, lo=lo, hi=hi, harrell_C=concordance_index(last['T'].values, -x[last.index.values], last.E.values))
rng = np.random.default_rng(2026); ids = last.index.values; T, E = last['T'].values, last.E.values
for ref in ['FRS', 'PREVENT total CVD', 'Age + sex + hypertension']:
    bs = []
    for _ in range(1000):
        b = rng.integers(0, len(ids), len(ids))
        bs.append(concordance_index(T[b], -Z['Model A+'][ids[b]], E[b]) - concordance_index(T[b], -Z[ref][ids[b]], E[b]))
    res[f'dC_Aplus_minus_{ref}'] = dict(d=res['models']['Model A+']['harrell_C'] - res['models'][ref]['harrell_C'], lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)))
(OUT / 'step5b_clsa_discrete_time.json').write_text(json.dumps(res, indent=2, default=float) + '\n'); print(json.dumps(res, indent=1, default=float))
