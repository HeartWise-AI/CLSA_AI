"""Guideline-specific endpoints (new by F2, baseline CVD/TIA-free) and simple AI-ECG / retina models vs the score each
guideline recommends.
  CAN  (CCS 2021, FRS 2008 general CVD): heart disease/HF, MI, angina, revasc, stroke, TIA, PVD(F1), any CVD death;
       baseline PVD excluded.
  US   (ACC/AHA 2026 PREVENT-ASCVD; PCE): MI, stroke, death from heart disease or stroke. Other events = non-events.
  EU   (ESC SCORE2/SCORE2-OP): MI, stroke, any CVD death (incl. HF death). Other events = non-events.
Models (logistic regression on the deep-model outputs; deep models = ECG-only ViT / retina-only ConvNeXt trained in the
5-fold CV, never on test):
  AI-ECG alone | Retina alone | AI-ECG + retina | AI-ECG + age + sex | AI-ECG + age + sex + hypertension (paper A+ style) |
  AI-ECG + retina + age + sex | Guideline score | Guideline score + AI-ECG
Dev: same 5 folds (out-of-fold). Test: logistic refit on all dev, applied once to test (post hoc; test used before)."""
import os, numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import final_test_evaluation as fe

HERE = os.path.dirname(os.path.abspath(__file__)); D = os.path.join(HERE, "data_v4")
lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
L = pd.read_csv(os.path.join(HERE, "incident_cvd_labels_v3_guideline_components.csv"), index_col=0)
det = L.incident_F2_primary.notna()
mi_st = (L.F12_AMI == 1) | (L.F12_CVA == 1)
anyev = (L.incident_F2_primary == 1) | (L.F12_TIA == 1) | (L.F1_PVD == 1)
EP = {
    "CAN_FRS_CVD": pd.Series(np.where(~det | (L.BL_PVD == 1), np.nan, anyev.astype(float)), index=L.index),
    "US_ASCVD": pd.Series(np.where(~det, np.nan, (mi_st | (L.death_chd_before_F2 == 1) | (L.death_stroke_before_F2 == 1)).astype(float)), index=L.index),
    "EU_SCORE2": pd.Series(np.where(~det, np.nan, (mi_st | (L.cvd_death_before_F2 == 1)).astype(float)), index=L.index),
}
# CAN controls must be free of every component incl. TIA and PVD
EP["CAN_FRS_CVD"][(EP["CAN_FRS_CVD"] == 0) & ((L.F12_TIA == 1) | (L.F1_PVD == 1))] = 1
SCORE = {"CAN_FRS_CVD": [("FRS (recommended)", "frs"), ("PREVENT total CVD", "prevent_cvd_winsor")],
         "US_ASCVD": [("PREVENT-ASCVD (recommended)", "prevent_ascvd_winsor"), ("PCE", "pce")],
         "EU_SCORE2": [("SCORE2/OP (recommended)", "score2")]}

sc = pd.read_parquet(os.path.join(D, "scores_v4.parquet"))
dev = pd.concat([sc[sc.split == s].sort_values("row_in_split").reset_index() for s in ["train", "val"]], ignore_index=True)
fid = np.zeros(len(dev), int)
for k, (_, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(dev, dev.incident_F2_primary.astype(int))): fid[te] = k
o = pd.read_parquet(os.path.join(D, "stage1b_oof_all.parquet")).reindex(dev.entity_id.values)
dev["ecg"], dev["ret"] = lg(o["ViT ecg"].values), lg(o["Retina retina"].values)
te = sc[sc.split == "test"].reset_index().merge(pd.read_parquet(os.path.join(HERE, "results_v4", "test_predictions_unimodal.parquet")), on="entity_id")
te["ecg"], te["ret"] = lg(te.ecg_only.values), lg(te.retina_only.values)
RES = "results_v4"
if os.environ.get("AIECG") == "wcr":  # manuscript v2: DeepECG-SSL (WCRv2) ECG-only, fold-calibrated (final_test_evaluation_wcr.py)
    RES = "results_v5"
    dev["ecg"] = lg(pd.read_parquet(os.path.join(HERE, RES, "dev_oof_wcr_calibrated.parquet")).reindex(dev.entity_id.values).cal_ecgo.values)
    te["ecg"] = lg(te[["entity_id"]].merge(pd.read_parquet(os.path.join(HERE, RES, "test_predictions_wcr.parquet"))[["entity_id", "DeepECG-SSL ECG-only (calibrated)"]], on="entity_id", how="left").iloc[:, 1].values)
dev["htn"], te["htn"] = dev.hypertension_dx, te.hypertension_dx
MODELS = {"AI-ECG alone": ["ecg"], "Retina alone": ["ret"], "AI-ECG + retina": ["ecg", "ret"],
          "AI-ECG + age + sex": ["ecg", "age", "female"], "AI-ECG + age + sex + hypertension": ["ecg", "age", "female", "htn"],
          "AI-ECG + retina + age + sex": ["ecg", "ret", "age", "female"], "Age + sex only": ["age", "female"]}
rows = []
for ep, lab in EP.items():
    yd_all = lab.reindex(dev.entity_id.values).values; kd = ~np.isnan(yd_all); yd = yd_all[kd].astype(int); fd = fid[kd]; Xd = dev[kd].reset_index(drop=True)
    yt_all = lab.reindex(te.entity_id.values).values; kt = ~np.isnan(yt_all); yt = yt_all[kt].astype(int); Xt = te[kt].reset_index(drop=True)
    models = dict(MODELS)
    for nm, c in SCORE[ep]:
        Xd[f"s_{c}"] = lg(Xd[c].values); Xt[f"s_{c}"] = lg(Xt[c].values); models[nm] = [f"s_{c}"]; models[f"{nm.split(' (')[0]} + AI-ECG"] = [f"s_{c}", "ecg"]
    ref_name, ref_col = SCORE[ep][0]
    for nm, cols in models.items():
        oof = np.zeros(len(yd))
        for j in range(5):
            m = LogisticRegression(max_iter=3000).fit(Xd.loc[fd != j, cols], yd[fd != j]); oof[fd == j] = m.predict_proba(Xd.loc[fd == j, cols])[:, 1]
        cv = np.mean([roc_auc_score(yd[fd == j], oof[fd == j]) for j in range(5)])
        cv_ref = np.mean([roc_auc_score(yd[fd == j], Xd[ref_col].values[fd == j]) for j in range(5)])
        pt = LogisticRegression(max_iter=3000).fit(Xd[cols], yd).predict_proba(Xt[cols])[:, 1]
        a, lo, hi = fe.auc_ci(yt, pt); d, dl, dh, p = fe.delong_test(yt, pt, Xt[ref_col].values)
        rows.append(dict(endpoint=ep, model=nm, dev_cases=int(yd.sum()), dev_n=len(yd), test_cases=int(yt.sum()), test_n=len(yt),
                         CV_AUROC=round(cv, 3), CV_delta_vs_rec=round(cv - cv_ref, 3), test_AUROC=f"{a:.3f} ({lo:.3f}-{hi:.3f})",
                         test_delta_vs_rec=("" if cols == [f"s_{ref_col}"] else f"{d:+.3f} ({dl:+.3f} to {dh:+.3f}), p={p:.2g}")))
r = pd.DataFrame(rows); r.to_csv(os.path.join(HERE, RES, "guideline_endpoints_results.csv"), index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 40)
for ep, g in r.groupby("endpoint", sort=False):
    print(f"\n== {ep}: dev n={g.dev_n.iloc[0]} cases={g.dev_cases.iloc[0]} | test n={g.test_n.iloc[0]} cases={g.test_cases.iloc[0]}")
    print(g[["model", "CV_AUROC", "CV_delta_vs_rec", "test_AUROC", "test_delta_vs_rec"]].to_string(index=False))
