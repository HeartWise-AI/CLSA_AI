"""npj Step 4 (REANALYSIS_PLAN §8.4): time-to-event analysis at MHI.
Cohort without the 6-y-ECG rule for non-events; deaths outside MHI (cause unknown) kept. In mhi_anyecg_build.py every death
during an MHI stay is a CV-death event, so the only competing deaths are deaths outside MHI (unk_death).
Python: Harrell's C, Uno's C (tau 6 y), IPCW AUC(6 y), Cox HR, bootstrap delta-C. R (survival::finegray): Fine-Gray, Aalen-Johansen."""
from pathlib import Path
import json, subprocess
import numpy as np, pandas as pd
from scipy.special import expit
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sksurv.metrics import concordance_index_ipcw, cumulative_dynamic_auc
from sksurv.util import Surv

ROOT = Path(__import__('os').environ.get('CLSA_WORKSPACE', '/volume/CLSA')); NAS = Path('/media/data1/ravram/CLSA_validation_files/manuscript')
OUT = ROOT / 'outputs/revision/npj'; OUT.mkdir(parents=True, exist_ok=True)
H, END = 2190, pd.Timestamp('2022-03-30')
c = pd.read_parquet(NAS / 'mhi_anyecg/cohort_labels.parquet').merge(pd.read_parquet(NAS / 'mhi_anyecg/predictions.parquet'), on='MRN', validate='one_to_one')
c = c[c.ecg_ok & (c.prev == 'no')].copy()
P = json.loads((ROOT / 'DeepECG_Docker/clsa_cvd/model_params.json').read_text()); A = P['model_ecg_age_sex_external']; S = P['age_sex_reference']
c['lA'] = A['intercept'] + A['b_ecg'] * c.wcr + A['b_age'] * c.age + A['b_female'] * c.female
c['lS'] = S['intercept'] + S['b_age'] * c.age + S['b_female'] * c.female
pt = pd.read_parquet('/media/data1/gjabbour/df/df_mhi_ptindex.parquet').drop_duplicates('id').set_index('id')
c['dday'] = (c.MRN.map(pt.death_date) - c.t0).dt.days
admin = np.minimum(H + 1, (END - c.t0).dt.days.values)  # +1: window end stays censored after the last possible event day (IPCW needs G > 0 at 2,190 d)
dd = np.where(c.dday.notna() & (c.dday >= 8), c.dday, np.inf)
ev = c.y.values == 1; evd = c.event_day.fillna(np.inf).values

def build(censor_last_ecg, unk_compete):
    cens = np.minimum(admin, dd)
    if censor_last_ecg: cens = np.minimum(cens, np.maximum(c.last_ecg_day.fillna(0).values, 0))
    is_ev = ev & (evd <= np.minimum(admin, dd))           # events are observed in MedEcho regardless of later ECGs
    T = np.where(is_ev, evd, cens)
    unk = c.unk_death.values & ~is_ev & (dd <= T)
    status = np.where(is_ev, 1, np.where(unk & unk_compete, 2, 0))  # 0 censored, 1 MACE, 2 competing death
    d = pd.DataFrame({'MRN': c.MRN.values, 'year': c.t0.dt.year.values, 'T': T, 'status': status, 'lA': c.lA.values, 'lS': c.lS.values})
    return d[d['T'] >= 8].reset_index(drop=True)

rng = np.random.default_rng(2026); rows = []
SCEN = {'primary: censor at last MHI ECG, outside deaths compete': (True, True), 'admin censoring, outside deaths compete': (False, True),
        'censor at last MHI ECG, outside deaths censored': (True, False), 'admin censoring, outside deaths censored': (False, False)}
for nm, (cl, uc) in SCEN.items():
    d = build(cl, uc); d.to_csv(OUT / f"step4_tte_{'last' if cl else 'admin'}_{'comp' if uc else 'cens'}.csv", index=False)
    e1 = (d.status == 1).values; T = d['T'].values.astype(float)
    sv = Surv.from_arrays(e1, T); tau = H
    r = dict(scenario=nm, n=len(d), mace=int(e1.sum()), competing_deaths=int((d.status == 2).sum()), median_fu_y=np.median(T) / 365.25)
    for k, m in [('model', 'lA'), ('agesex', 'lS')]:
        x = d[m].values
        r[f'harrell_{k}'] = concordance_index(T, -x, e1)
        r[f'uno_{k}'] = concordance_index_ipcw(sv, sv, x, tau=tau)[0]
        r[f'auc6y_{k}'] = cumulative_dynamic_auc(sv, sv, x, [tau])[0][0]
        z = (x - x.mean()) / x.std(); cph = CoxPHFitter().fit(pd.DataFrame({'T': T, 'E': e1, 'z': z}), 'T', 'E')
        r[f'HR_perSD_{k}'], r[f'HR_lo_{k}'], r[f'HR_hi_{k}'] = np.exp([cph.params_.z, *cph.confidence_intervals_.loc['z'].values])
    n = len(d); dh, du = [], []
    for _ in range(500):
        i = rng.integers(0, n, n); s_ = sv[i]
        dh.append(concordance_index(T[i], -d.lA.values[i], e1[i]) - concordance_index(T[i], -d.lS.values[i], e1[i]))
        du.append(concordance_index_ipcw(sv, s_, d.lA.values[i], tau=tau)[0] - concordance_index_ipcw(sv, s_, d.lS.values[i], tau=tau)[0])
    r.update(dHarrell=r['harrell_model'] - r['harrell_agesex'], dHarrell_lo=np.percentile(dh, 2.5), dHarrell_hi=np.percentile(dh, 97.5),
             dUno=r['uno_model'] - r['uno_agesex'], dUno_lo=np.percentile(du, 2.5), dUno_hi=np.percentile(du, 97.5))
    rows.append(r); print(nm, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
pd.DataFrame(rows).to_csv(OUT / 'step4_mhi_cindex.csv', index=False)
subprocess.run(['Rscript', str(Path(__file__).with_name('npj_step4_finegray.R')), str(OUT)], check=True)
