"""Guideline scores and Model A+ for CLSA participants with indeterminate F2 status (same imputer, same code as
compute_scores_v4.py; imputer refitted on the identical training split with the same seed). Output: v2/indeterminate/scores.parquet"""
import os, sys, numpy as np, pandas as pd, json
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
H = os.path.dirname(os.path.abspath(__file__)); M = os.path.dirname(H); sys.path.insert(0, M); os.chdir(M)
import compute_scores_v4 as cs
from risk_scores import frs2008, pce2013, frs2008_bmi, MMOL_TO_MGDL
lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
co = pd.read_parquet("data_v4/cohort_v4.parquet"); rf = pd.read_parquet("baseline_risk_factors.parquet")
e = pd.read_parquet("v2/indeterminate/ecg_scores.parquet"); e = e[~e.check].set_index("entity_id")
d = rf.reindex(e.index).copy(); d["p_ecg"] = e.p_ecg
L = pd.read_csv("incident_cvd_labels_v2_components.csv", index_col=0); d["cvd_death_before_F2"] = L.cvd_death_before_F2.reindex(d.index)
F = cs.CONT + cs.BIN + ["age", "female", "diabetes", "bp_treated", "statin"]
imp = IterativeImputer(random_state=0, max_iter=20).fit(co[F].astype(float)[co.split == "train"])
Xi = pd.DataFrame(imp.transform(d[F].astype(float)), index=d.index, columns=F)
d["bmi_raw"] = d.bmi
for c in cs.CONT: d[c] = Xi[c]
d["smoker_current"] = (Xi.smoker_current >= 0.5).astype(int)
for c in ["diabetes", "bp_treated", "statin", "hypertension_dx"]: d[c] = d[c].fillna(0)
a, f = d.age.values, d.female.values; tc, hdl = d.tc_mmol.values * MMOL_TO_MGDL, d.hdl_mmol.values * MMOL_TO_MGDL
d["frs"] = frs2008(a, f, tc, hdl, d.sbp, d.bp_treated, d.smoker_current, d.diabetes)
d["pce"] = pce2013(a, f, d.race_pce == "AA", tc, hdl, d.sbp, d.bp_treated, d.smoker_current, d.diabetes)
d["score2"] = cs.score2(a, f, d.smoker_current.values, d.sbp.values, d.diabetes.values, d.tc_mmol.values, d.hdl_mmol.values, region="Low")
w = d.copy()
for c, (lo, hi) in {"age": (30, 79), "sbp": (90, 180), "tc_mmol": (3.37, 8.27), "hdl_mmol": (0.52, 2.58), "bmi": (18.5, 39.9), "egfr": (15, 140)}.items(): w[c] = w[c].clip(lo, hi)
d["prevent_cvd_winsor"] = cs.prevent_r(w).total_cvd.reindex(d.index).values
p = json.load(open("/media/data1/ravram/CLSA_validation_files/hf_release/model_params.json"))["model_A_plus"]
d["Aplus"] = 1 / (1 + np.exp(-(p["intercept"] + p["b_ecg"] * lg(d.p_ecg) + p["b_age"] * d.age + p["b_female"] * d.female + p["b_htn"] * d.hypertension_dx)))
d.to_parquet("v2/indeterminate/scores.parquet")
print(d[["age", "female", "frs", "pce", "score2", "prevent_cvd_winsor", "Aplus", "p_ecg"]].describe().round(3).to_string()); print("PREVENT NA", d.prevent_cvd_winsor.isna().sum())
# check the equation reproduces stored test A+
te = pd.read_parquet("results_v5/test_predictions_wcr.parquet").merge(pd.read_parquet("results_v5/test_predictions_simple.parquet")[["entity_id", "DeepECG-SSL + age + sex + hypertension"]], on="entity_id")
z = p["intercept"] + p["b_ecg"] * lg(te["DeepECG-SSL ECG-only (calibrated)"]) + p["b_age"] * te.age + p["b_female"] * te.female + p["b_htn"] * te.hypertension_dx
print("A+ eq check max diff", np.abs(1 / (1 + np.exp(-z)) - te["DeepECG-SSL + age + sex + hypertension"]).max())
