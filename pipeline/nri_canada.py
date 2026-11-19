"""NRI for the Canadian (FRS general-CVD) endpoint, CLSA test set: Model A+ (refitted on the Canadian endpoint in development
data, as in Table 3) vs FRS and vs PREVENT total CVD, each recalibrated (intercept + slope) on the same development endpoint.
Categories <5%, 5-10%, >=10% predicted risk by F2; also continuous (category-free) NRI. 1,000 bootstrap resamples."""
import os, json, numpy as np
from sklearn.linear_model import LogisticRegression
os.environ["AIECG"] = "wcr"
import guideline_endpoints as ge
lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
lab = ge.EP["CAN_FRS_CVD"]
yd_all = lab.reindex(ge.dev.entity_id.values).values; kd = ~np.isnan(yd_all); Xd = ge.dev[kd].reset_index(drop=True); yd = yd_all[kd].astype(int)
yt_all = lab.reindex(ge.te.entity_id.values).values; kt = ~np.isnan(yt_all); Xt = ge.te[kt].reset_index(drop=True); yt = yt_all[kt].astype(int)
cols = ["ecg", "age", "female", "htn"]
pA = LogisticRegression(max_iter=3000).fit(Xd[cols], yd).predict_proba(Xt[cols])[:, 1]
def recal(c): m = LogisticRegression(C=1e6).fit(lg(Xd[c].values).reshape(-1, 1), yd); return m.predict_proba(lg(Xt[c].values).reshape(-1, 1))[:, 1]
cats = lambda q: np.digitize(q, [0.05, 0.10])
def nri(y, new, old):
    cn, co = cats(new), cats(old); up, dn = cn > co, cn < co
    e = up[y == 1].mean() - dn[y == 1].mean(); ne = dn[y == 0].mean() - up[y == 0].mean()
    cu, cd = new > old, new < old; ce = cu[y == 1].mean() - cd[y == 1].mean(); cne = cd[y == 0].mean() - cu[y == 0].mean()
    return np.array([e, ne, e + ne, ce + cne])
out = {"endpoint": f"Canadian FRS general-CVD endpoint, test n={len(yt)}, events={yt.sum()}"}
rng = np.random.default_rng(2026); B = [rng.integers(0, len(yt), len(yt)) for _ in range(1000)]
for nm, c in [("FRS", "frs"), ("PREVENT total CVD", "prevent_cvd_winsor")]:
    pO = recal(c); pt = nri(yt, pA, pO); bt = np.array([nri(yt[b], pA[b], pO[b]) for b in B])
    lo, hi = np.percentile(bt, 2.5, 0), np.percentile(bt, 97.5, 0)
    out[f"Model A+ vs {nm}"] = {k: f"{pt[i]:+.3f} ({lo[i]:+.3f} to {hi[i]:+.3f})" for i, k in enumerate(["NRI_events", "NRI_nonevents", "NRI_categorical_total", "NRI_continuous"])}
    out[f"Model A+ vs {nm}"]["reclassified_%"] = f"{100 * (cats(pA) != cats(pO)).mean():.1f}"
json.dump(out, open("v2/tables/_nri_canada.json", "w"), indent=1, ensure_ascii=False); print(json.dumps(out, indent=1, ensure_ascii=False))
