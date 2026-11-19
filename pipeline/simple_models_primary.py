"""Manuscript v2: lab-free DeepECG-SSL models on the PRIMARY CLSA outcome (incident CVD by F2), locked test.
DeepECG-SSL ECG-only (fold-calibrated logit) + age + sex [+ hypertension]; logistic fitted on development data
(5-fold OOF for CV), applied once to test. Compared with each guideline score (DeLong) and with age + sex."""
import os, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from final_test_evaluation import auc_ci, delong_test
lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
D, R = "data_v4", "results_v5"
sc = pd.read_parquet(f"{D}/scores_v4.parquet")
dev = pd.concat([sc[sc.split == s].sort_values("row_in_split").reset_index() for s in ["train", "val"]], ignore_index=True)
dev["ecg"] = lg(pd.read_parquet(f"{R}/dev_oof_wcr_calibrated.parquet").reindex(dev.entity_id.values).cal_ecgo.values)
te = pd.read_parquet(f"{R}/test_predictions_wcr.parquet"); te["ecg"] = lg(te["DeepECG-SSL ECG-only (calibrated)"].values)
for d in (dev, te): d["htn"] = d.hypertension_dx
from scipy.stats import norm, rankdata
CLIN = ["age", "female", "sbp", "bp_treated", "tc_mmol", "hdl_mmol", "diabetes", "smoker_current", "egfr", "bmi", "statin"]  # = stage1b_wcr_cv.CLIN
o = pd.read_parquet(f"{D}/stage1b_oof_all.parquet").reindex(dev.entity_id.values)
dev["ret"] = norm.ppf((o.groupby("fold")["Retina retina"].rank().values - 0.5) / o.groupby("fold")["Retina retina"].transform("size").values)  # within-fold rank->normal
u = te[["entity_id"]].merge(pd.read_parquet("results_v4/test_predictions_unimodal.parquet"), on="entity_id", how="left")
te["ret"] = norm.ppf((rankdata(u.retina_only.values) - 0.5) / len(u))
yd, yt = dev.incident_F2_primary.values.astype(int), te.incident_F2_primary.values.astype(int)
fid = np.zeros(len(dev), int)
for k, (_, i) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(dev, yd)): fid[i] = k
M = {"DeepECG-SSL ECG-only": ["ecg"], "Age + sex": ["age", "female"], "DeepECG-SSL + age + sex": ["ecg", "age", "female"],
     "DeepECG-SSL + age + sex + hypertension": ["ecg", "age", "female", "htn"],
     "Retina + age + sex + hypertension": ["ret", "age", "female", "htn"],
     "DeepECG-SSL + retina + age + sex + hypertension": ["ecg", "ret", "age", "female", "htn"],
     "Clinical risk factors only (logistic, 11 PREVENT/FRS inputs)": CLIN, "Age + sex + hypertension": ["age", "female", "htn"]}
rows = []
for nm, cols in M.items():
    oof = np.zeros(len(yd))
    for j in range(5): oof[fid == j] = LogisticRegression(max_iter=3000).fit(dev.loc[fid != j, cols], yd[fid != j]).predict_proba(dev.loc[fid == j, cols])[:, 1]
    p = LogisticRegression(max_iter=3000).fit(dev[cols], yd).predict_proba(te[cols])[:, 1]; te[nm] = p
    a, lo, hi = auc_ci(yt, p); r = dict(model=nm, CV_AUROC=round(np.mean([roc_auc_score(yd[fid == j], oof[fid == j]) for j in range(5)]), 3), test_AUROC=f"{a:.3f} ({lo:.3f}-{hi:.3f})")
    for k, c in [("FRS", "frs"), ("PCE", "pce"), ("SCORE2/OP", "score2"), ("PREVENT", "prevent_cvd_winsor")]:
        d_, dl, dh, pp = delong_test(yt, p, te[c].values); r[f"vs {k}"] = f"{d_:+.3f} ({dl:+.3f} to {dh:+.3f}), p={pp:.2g}"
    rows.append(r)
r = pd.DataFrame(rows); r.to_csv(f"{R}/simple_models_primary.csv", index=False)
te[["entity_id"] + list(M)].to_parquet(f"{R}/test_predictions_simple.parquet")
pd.set_option("display.width", 300); print(r.to_string(index=False))
