"""npj Step 3 (REANALYSIS_PLAN §8.3): MHI calibration and decision curve after out-of-sample recalibration.
Primary: temporal recalibration (fit 2006-2010, evaluate 2011-2016). Sensitivity: 5-fold cross-fitted (as mhi_threshold.py)."""
from pathlib import Path
import json
import numpy as np, pandas as pd
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.model_selection import StratifiedKFold
import statsmodels.api as sm
from statsmodels.nonparametric.smoothers_lowess import lowess
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

ROOT = Path(__import__('os').environ.get('CLSA_WORKSPACE', '/volume/CLSA')); NAS = Path('/media/data1/ravram/CLSA_validation_files/manuscript')
OUT = ROOT / 'outputs/revision/npj'; OUT.mkdir(parents=True, exist_ok=True)
c = pd.read_parquet(NAS / 'mhi_anyecg/cohort_labels.parquet').merge(pd.read_parquet(NAS / 'mhi_anyecg/predictions.parquet'), on='MRN', validate='one_to_one')
c = c[c.ecg_ok]; b = c[(c.prev == 'no') & ~c.unk_death & ((c.y == 1) | c.fu_ecg_6y)].reset_index(drop=True); assert len(b) == 21647 and b.y.sum() == 2190
P = json.loads((ROOT / 'DeepECG_Docker/clsa_cvd/model_params.json').read_text()); A = P['model_ecg_age_sex_external']; S = P['age_sex_reference']
b['pA'] = expit(A['intercept'] + A['b_ecg'] * b.wcr + A['b_age'] * b.age + A['b_female'] * b.female)
b['pS'] = expit(S['intercept'] + S['b_age'] * b.age + S['b_female'] * b.female)
lg = lambda p: np.log(p / (1 - p)); y = b.y.values.astype(int); yr = b.t0.dt.year.values
assert abs(roc_auc_score(y, b.pA) - .761) < .0005

def calib(y, p):
    x = lg(p); return dict(n=len(y), events=int(y.sum()), observed=y.mean(), predicted=p.mean(), AUROC=roc_auc_score(y, p),
        cal_intercept=sm.GLM(y, np.ones_like(x), family=sm.families.Binomial(), offset=x).fit().params[0],
        cal_slope=sm.GLM(y, sm.add_constant(x), family=sm.families.Binomial()).fit().params[1],
        ICI=float(np.mean(abs(lowess(y, p, frac=2 / 3, it=0, return_sorted=False) - p))), brier=brier_score_loss(y, p))
def nb(y, p, t): n = len(y); return (np.sum((p >= t) & (y == 1)) - np.sum((p >= t) & (y == 0)) * t / (1 - t)) / n
fit = lambda p, yy: LogisticRegression(C=1e6).fit(lg(p).reshape(-1, 1), yy)

# temporal: fit on 2006-2010, evaluate on 2011-2016
tr, ev = yr <= 2010, yr >= 2011
for k in ['pA', 'pS']:
    b[k + '_temp'] = np.nan; b.loc[ev, k + '_temp'] = fit(b[k].values[tr], y[tr]).predict_proba(lg(b[k].values[ev]).reshape(-1, 1))[:, 1]
    rc = np.zeros(len(b))
    for i, j in StratifiedKFold(5, shuffle=True, random_state=0).split(b, y): rc[j] = fit(b[k].values[i], y[i]).predict_proba(lg(b[k].values[j]).reshape(-1, 1))[:, 1]
    b[k + '_cf'] = rc
SETS = {'temporal (eval 2011-2016)': (ev, '_temp'), 'cross-fitted (all)': (np.ones(len(b), bool), '_cf'), 'as transported (eval 2011-2016)': (ev, '')}
TH = np.round(np.arange(.05, .2001, .01), 2); rng = np.random.default_rng(2026); cal, dca, nbd = [], [], []
for nm, (m, suf) in SETS.items():
    yy = y[m]; pa, ps = b['pA' + suf].values[m], b['pS' + suf].values[m]
    cal += [dict(setting=nm, model='DeepECG-SSL + age + sex', **calib(yy, pa)), dict(setting=nm, model='Age + sex', **calib(yy, ps))]
    for t in TH:
        dca += [dict(setting=nm, threshold=t, model='DeepECG-SSL + age + sex', net_benefit=nb(yy, pa, t)), dict(setting=nm, threshold=t, model='Age + sex', net_benefit=nb(yy, ps, t)),
                dict(setting=nm, threshold=t, model='Treat all', net_benefit=yy.mean() - (1 - yy.mean()) * t / (1 - t)), dict(setting=nm, threshold=t, model='Treat none', net_benefit=0.)]
    n = len(yy); B = [rng.integers(0, n, n) for _ in range(1000)]
    for t in [.05, .075, .10, .15, .20]:
        d = np.array([nb(yy[i], pa[i], t) - nb(yy[i], ps[i], t) for i in B])
        fa, fs = pa >= t, ps >= t
        nbd.append(dict(setting=nm, threshold=t, NB_model=nb(yy, pa, t), NB_agesex=nb(yy, ps, t), dNB=nb(yy, pa, t) - nb(yy, ps, t), lo=np.percentile(d, 2.5), hi=np.percentile(d, 97.5),
                        flagged_per_1000_model=1000 * fa.mean(), flagged_per_1000_agesex=1000 * fs.mean(),
                        events_found_per_1000_model=1000 * (fa & (yy == 1)).mean(), events_found_per_1000_agesex=1000 * (fs & (yy == 1)).mean(),
                        net_reduction_interventions_per_100=100 * (nb(yy, pa, t) - nb(yy, ps, t)) / (t / (1 - t))))
cal, dca, nbd = pd.DataFrame(cal), pd.DataFrame(dca), pd.DataFrame(nbd)
cal.to_csv(OUT / 'step3_mhi_calibration.csv', index=False); dca.to_csv(OUT / 'step3_mhi_dca.csv', index=False); nbd.to_csv(OUT / 'step3_mhi_nb_diff.csv', index=False)
fig, ax = plt.subplots(figsize=(5, 4)); g = dca[dca.setting == 'temporal (eval 2011-2016)']
for m, st in [('DeepECG-SSL + age + sex', '-'), ('Age + sex', '--'), ('Treat all', ':'), ('Treat none', '-')]:
    h = g[g.model == m]; ax.plot(h.threshold * 100, h.net_benefit, st, color='grey' if m == 'Treat none' else None, label=m)
ax.set(xlabel='Threshold probability (%)', ylabel='Net benefit', title='MHI, recalibrated (temporal), 2011-2016', ylim=(-.01, None)); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT / 'step3_mhi_dca_temporal.png', dpi=300)
pd.set_option('display.width', 250); print(cal.round(4).to_string(index=False)); print(nbd.round(4).to_string(index=False))
