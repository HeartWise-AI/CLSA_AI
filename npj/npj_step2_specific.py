"""npj Step 2: specific CVD (specific_F2) as a co-primary Table 2 row, using the frozen primary-model predictions
(Model A+ as fitted for the primary endpoint; recalibrated scores as stored in v2.9). Same columns as the primary row:
AUROC, AUPRC, calibration, and the 10% operating point (flagged and events found per 1,000), paired bootstrap 1,000."""
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
import statsmodels.api as sm

NAS = Path('/media/data1/ravram/CLSA_validation_files/manuscript')
OUT = Path(__import__('os').environ.get('CLSA_WORKSPACE', '/volume/CLSA')) / 'outputs/revision/npj'; OUT.mkdir(parents=True, exist_ok=True)
lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
w = pd.read_parquet(NAS / 'results_v5/test_predictions_wcr.parquet'); s = pd.read_parquet(NAS / 'results_v5/test_predictions_simple.parquet')
d = w[['entity_id', 'incident_F2_primary', 'FRS (recal)', 'PREVENT (recal)', 'SCORE2/OP (recal)', 'PCE (recal)']].merge(
    s[['entity_id', 'DeepECG-SSL + age + sex + hypertension']], on='entity_id', validate='one_to_one')
d['specific_F2'] = pd.read_csv(NAS / 'incident_cvd_labels_v3_guideline_components.csv', index_col=0).reindex(d.entity_id).specific_F2.values
M = {'Model A+': 'DeepECG-SSL + age + sex + hypertension', 'FRS': 'FRS (recal)', 'PREVENT': 'PREVENT (recal)', 'SCORE2/OP': 'SCORE2/OP (recal)', 'PCE': 'PCE (recal)'}

def stats(y, p):
    x = lg(p); f = p >= .10
    return dict(AUROC=roc_auc_score(y, p), AUPRC=average_precision_score(y, p),
                cal_intercept=sm.GLM(y, np.ones_like(x), family=sm.families.Binomial(), offset=x).fit().params[0],
                cal_slope=sm.GLM(y, sm.add_constant(x), family=sm.families.Binomial()).fit().params[1], brier=brier_score_loss(y, p),
                flagged_per_1000=1000 * f.mean(), events_found_per_1000=1000 * (f & (y == 1)).mean(), sens10=(f & (y == 1)).sum() / y.sum())

rng = np.random.default_rng(2026); rows, comp = [], []
for ep in ['incident_F2_primary', 'specific_F2']:
    g = d[d[ep].notna()]; y = g[ep].values.astype(int); n = len(y)
    B = [b for b in (rng.integers(0, n, n) for _ in range(1000)) if 0 < y[b].sum() < n]
    BS = {m: pd.DataFrame([stats(y[b], g[c].values[b]) for b in B]) for m, c in M.items()}
    for m, c in M.items():
        st = stats(y, g[c].values)
        rows.append(dict(endpoint=ep, model=m, n=n, events=int(y.sum()), **{k: v for k, v in st.items()},
                         **{f'{k}_lo': BS[m][k].quantile(.025) for k in ['AUROC', 'AUPRC', 'flagged_per_1000', 'events_found_per_1000']},
                         **{f'{k}_hi': BS[m][k].quantile(.975) for k in ['AUROC', 'AUPRC', 'flagged_per_1000', 'events_found_per_1000']}))
        if m != 'Model A+':
            st0 = stats(y, g[M['Model A+']].values)
            for k in ['AUROC', 'AUPRC', 'flagged_per_1000', 'events_found_per_1000']:
                dd = BS['Model A+'][k] - BS[m][k]
                comp.append(dict(endpoint=ep, comparison=f'Model A+ minus {m}', metric=k, diff=st0[k] - st[k], lo=dd.quantile(.025), hi=dd.quantile(.975)))
pd.DataFrame(rows).to_csv(OUT / 'step2_table2_rows.csv', index=False); pd.DataFrame(comp).to_csv(OUT / 'step2_comparisons.csv', index=False)
pd.set_option('display.width', 250)
print(pd.DataFrame(rows)[['endpoint', 'model', 'n', 'events', 'AUROC', 'AUPRC', 'cal_intercept', 'cal_slope', 'flagged_per_1000', 'events_found_per_1000', 'sens10']].round(3).to_string(index=False))
print(pd.DataFrame(comp).round(3).to_string(index=False))
