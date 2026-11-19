"""Bootstrap CIs for the guideline-endpoint test table.
Primary: paired percentile bootstrap, 1,000 resamples of the full test set (with replacement), same resamples for all models.
Also: 100 random 80% subsamples (without replacement), as requested (narrower by construction; reported for comparison)."""
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import guideline_endpoints as ge  # rebuilds endpoints/data (prints the CV table again)

rng = np.random.default_rng(2026); rows = []
for ep, lab in ge.EP.items():
    yd_all = lab.reindex(ge.dev.entity_id.values).values; kd = ~np.isnan(yd_all); yd = yd_all[kd].astype(int); Xd = ge.dev[kd].reset_index(drop=True)
    yt_all = lab.reindex(ge.te.entity_id.values).values; kt = ~np.isnan(yt_all); yt = yt_all[kt].astype(int); Xt = ge.te[kt].reset_index(drop=True)
    models = dict(ge.MODELS)
    for nm, c in ge.SCORE[ep]:
        Xd[f"s_{c}"] = ge.lg(Xd[c].values); Xt[f"s_{c}"] = ge.lg(Xt[c].values); models[nm] = [f"s_{c}"]; models[f"{nm.split(' (')[0]} + AI-ECG"] = [f"s_{c}", "ecg"]
    P = {nm: LogisticRegression(max_iter=3000).fit(Xd[cols], yd).predict_proba(Xt[cols])[:, 1] for nm, cols in models.items()}
    ref = ge.SCORE[ep][0][0]; n = len(yt)
    B = [rng.integers(0, n, n) for _ in range(1000)]; B = [b for b in B if 0 < yt[b].sum() < n]
    S = [rng.choice(n, int(0.8 * n), replace=False) for _ in range(100)]
    for nm, p in P.items():
        boot = np.array([roc_auc_score(yt[b], p[b]) for b in B]); bref = np.array([roc_auc_score(yt[b], P[ref][b]) for b in B])
        sub = np.array([roc_auc_score(yt[s], p[s]) for s in S]); sref = np.array([roc_auc_score(yt[s], P[ref][s]) for s in S])
        d = boot - bref
        rows.append(dict(endpoint=ep, model=nm, events=int(yt.sum()), n=n, AUROC=roc_auc_score(yt, p),
                         lo=np.percentile(boot, 2.5), hi=np.percentile(boot, 97.5),
                         d=roc_auc_score(yt, p) - roc_auc_score(yt, P[ref]), d_lo=np.percentile(d, 2.5), d_hi=np.percentile(d, 97.5),
                         p_boot=min(1, 2 * min((d <= 0).mean(), (d >= 0).mean())) if nm != ref else np.nan,
                         sub80_lo=np.percentile(sub, 2.5), sub80_hi=np.percentile(sub, 97.5),
                         sub80_d_lo=np.percentile(sub - sref, 2.5), sub80_d_hi=np.percentile(sub - sref, 97.5)))
r = pd.DataFrame(rows); r.to_csv(f"{ge.RES}/guideline_endpoints_bootstrap.csv", index=False)
order = ["recommended", "AI-ECG alone", "Retina alone", "AI-ECG + retina", "AI-ECG + age + sex", "AI-ECG + age + sex + hypertension",
         "AI-ECG + retina + age + sex", "Age + sex only", "+ AI-ECG"]
for ep, g in r.groupby("endpoint", sort=False):
    print(f"\n== {ep} (test n={g.n.iloc[0]}, events={g.events.iloc[0]})")
    for _, x in g.iterrows():
        dtxt = "" if np.isnan(x.p_boot) else f"{x.d:+.3f} ({x.d_lo:+.3f} to {x.d_hi:+.3f}) p={x.p_boot:.3f}"
        print(f"  {x.model:36s} {x.AUROC:.3f} ({x.lo:.3f}-{x.hi:.3f}) | Δ vs rec {dtxt} | 80%x100: {x.sub80_lo:.3f}-{x.sub80_hi:.3f}")
