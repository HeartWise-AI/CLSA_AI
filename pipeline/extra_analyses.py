"""Iteration-1 reviewer requests (existing data only): CLSA secondary/specific/hard outcomes for Model A+, ECG increment,
office-based FRS in CLSA, validated-age-range analyses, Holm for Model A+, Canadian A+ vs PREVENT-total paired delta,
outcome-composition shares; MHI calibration, landmark sensitivity (events after day 90/180), Biobank A+ vs age+sex,
attrition explanation. Output: v2/tables/_extra2.json and v2/tables/*.json pieces consumed by build_tables_figures.py."""
import os, sys, json, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
HERE = os.path.dirname(os.path.abspath(__file__)); M = os.path.dirname(HERE); sys.path.insert(0, M); os.chdir(M)
from final_test_evaluation import auc_ci, delong_test, holm, cal
from risk_scores import frs2008_bmi
lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
f3 = lambda a, lo, hi: f"{a:.3f} ({lo:.3f}–{hi:.3f})"; fd = lambda d, lo, hi: f"{d:+.3f} ({lo:+.3f} to {hi:+.3f})"
fp = lambda p: "<0.001" if p < 0.001 else f"{p:.3f}"
X = {}
te = pd.read_parquet("results_v5/test_predictions_wcr.parquet").drop(columns=["DeepECG-SSL ECG-only"]).merge(pd.read_parquet("results_v5/test_predictions_simple.parquet"), on="entity_id")
AP, AS, A = "DeepECG-SSL + age + sex + hypertension", "Age + sex + hypertension", "DeepECG-SSL ECG-only"
y = te.incident_F2_primary.values.astype(int)
# Holm for the four A+ vs score comparisons; ECG increment
ps = {k: delong_test(y, te[AP].values, te[c].values) for k, c in [("FRS", "frs"), ("PCE", "pce"), ("SCORE2/OP", "score2"), ("PREVENT", "prevent_cvd_winsor")]}
X["holm_Aplus"] = dict(zip(ps, [round(float(v), 4) for v in holm([v[3] for v in ps.values()])]))
d = delong_test(y, te[AP].values, te[AS].values); X["ecg_increment"] = f"{fd(*d[:3])}; p={fp(d[3])}"
# office-based FRS (BMI) in CLSA test
rf = pd.read_parquet("baseline_risk_factors.parquet"); t2 = te.join(rf[["bmi", "sbp", "bp_treated", "smoker_current", "diabetes"]].add_suffix("_rf"), on="entity_id")
t2["ofrs"] = frs2008_bmi(t2.age.clip(30, 79), t2.female, t2.bmi_rf.fillna(t2.bmi_rf.median()).clip(13, 70), t2.sbp_rf.fillna(t2.sbp_rf.median()), t2.bp_treated_rf.fillna(0), t2.smoker_current_rf.fillna(0), t2.diabetes_rf.fillna(0))
a = auc_ci(y, t2.ofrs.values); d = delong_test(y, te[AP].values, t2.ofrs.values); d2 = delong_test(y, t2.ofrs.values, te.frs.values)
X["office_frs_test"] = dict(auroc=f3(*a), Aplus_minus_office=f"{fd(*d[:3])}; p={fp(d[3])}", office_minus_labFRS=f"{fd(*d2[:3])}; p={fp(d2[3])}")
# validated age ranges
rows = []
for lab_, mask, comp in [("Age ≤74 (FRS validated range)", te.age <= 74, [("FRS", "frs"), ("PREVENT", "prevent_cvd_winsor")]), ("Age ≤79 (PCE/PREVENT validated range)", te.age <= 79, [("PCE", "pce"), ("PREVENT", "prevent_cvd_winsor")]),
                       ("Age 40–69 (SCORE2 range)", te.age <= 69, [("SCORE2", "score2")]), ("PREVENT inputs not winsorised", te.prevent_inputs_winsorised == False, [("PREVENT", "prevent_cvd_winsor")])]:
    g = te[mask.values]; yy = g.incident_F2_primary.values.astype(int)
    for k, c in comp:
        dd = delong_test(yy, g[AP].values, g[c].values); rows.append([lab_, f"{len(g):,} / {yy.sum()}", f3(*auc_ci(yy, g[AP].values)), k, f3(*auc_ci(yy, g[c].values)), f"{fd(*dd[:3])}; p={fp(dd[3])}"])
X["in_range_rows"] = rows
# secondary + F2 specific/hard outcomes for Model A+
L2 = pd.read_csv("incident_cvd_labels_v2_components.csv", index_col=0)
for c in ["specific_F2", "hard_F2"]: te[c] = L2.reindex(te.entity_id)[c].values
rows = []
for oc, lab_ in [("incident_F1", "Incident CVD by F1 (median 2.9 y)"), ("strict_F1", "Incident CVD by F1, strict (no pre-baseline age at diagnosis)"),
                 ("noHEARTonly_F1", "Incident CVD by F1, excluding 'heart disease'-only cases"), ("hard_ascvd_F1", "Hard ASCVD (MI or stroke) by F1"),
                 ("specific_F2", "Specific CVD by F2 (excluding the non-specific 'heart disease' item)"), ("hard_F2", "Hard CVD by F2 (MI, stroke, CVD death)")]:
    g = te.dropna(subset=[oc]); yy = g[oc].values.astype(int)
    rows.append([f"{lab_} — {yy.sum()} events / {len(g):,}", "", "", ""])
    for nm, c in [("Model A+", AP), ("Model A (ECG only)", A), ("Age + sex + hypertension", AS), ("FRS", "frs"), ("PCE", "pce"), ("SCORE2/OP", "score2"), ("PREVENT", "prevent_cvd_winsor")]:
        aa = auc_ci(yy, g[c].values)
        if c == AP: rows.append(["   " + nm, f3(*aa), "", ""]); continue
        dd = delong_test(yy, g[AP].values, g[c].values); rows.append(["   " + nm, f3(*aa), fd(*dd[:3]), fp(dd[3])])
X["secondary_rows"] = rows
# outcome composition (analytic cohort)
L1 = pd.read_csv("incident_cvd_labels_v2_components.csv", index_col=0)
sc = pd.read_parquet("data_v4/scores_v4.parquet"); ac = L1.reindex(sc.entity_id if "entity_id" in sc else sc.index)
cases = ac[ac.incident_F2_primary == 1]
X["composition"] = dict(cases=int(len(cases)), specific=int((cases.specific_F2 == 1).sum()), hard=int((cases.hard_F2 == 1).sum()),
                        heart_item_only=int(((cases.specific_F2 != 1)).sum()), share_heart_only=round(float((cases.specific_F2 != 1).mean()), 3))
# Canadian endpoint: Model A+ vs PREVENT total CVD (paired bootstrap), from guideline script machinery
os.environ["AIECG"] = "wcr"
import guideline_endpoints as ge
lab = ge.EP["CAN_FRS_CVD"]; yd_all = lab.reindex(ge.dev.entity_id.values).values; kd = ~np.isnan(yd_all); Xd = ge.dev[kd].reset_index(drop=True); yd = yd_all[kd].astype(int)
yt_all = lab.reindex(ge.te.entity_id.values).values; kt = ~np.isnan(yt_all); Xt = ge.te[kt].reset_index(drop=True); yt = yt_all[kt].astype(int)
cols = ["ecg", "age", "female", "htn"]; pA = LogisticRegression(max_iter=3000).fit(Xd[cols], yd).predict_proba(Xt[cols])[:, 1]; pP = Xt.prevent_cvd_winsor.values
rng = np.random.default_rng(2026); B = [rng.integers(0, len(yt), len(yt)) for _ in range(1000)]
bd = np.array([roc_auc_score(yt[b], pA[b]) - roc_auc_score(yt[b], pP[b]) for b in B]); dd = roc_auc_score(yt, pA) - roc_auc_score(yt, pP)
X["can_Aplus_vs_prevent_total"] = dict(delta=fd(dd, np.percentile(bd, 2.5), np.percentile(bd, 97.5)), p=fp(2 * min((bd <= 0).mean(), (bd >= 0).mean())), heart_item_share_can=None)
# ---------------- MHI
c = pd.read_parquet("mhi_anyecg/cohort_labels.parquet").merge(pd.read_parquet("mhi_anyecg/predictions.parquet"), on="MRN"); c = c[c.ecg_ok]
base = c[(c.prev == "no") & ~c.unk_death & ((c.y == 1) | c.fu_ecg_6y)].reset_index(drop=True)
dv = pd.concat([sc[sc.split == s].sort_values("row_in_split").reset_index() for s in ["train", "val"]], ignore_index=True)
o = pd.read_parquet("data_v4/stage1b_oof_all.parquet").reindex(dv.entity_id.values); dv["wcr"] = lg(o["WCR wcr"].values)
fA = LogisticRegression(max_iter=3000).fit(dv[["wcr", "age", "female"]], dv.incident_F2_primary.astype(int)); fS = LogisticRegression(max_iter=3000).fit(dv[["age", "female"]], dv.incident_F2_primary.astype(int))
base["pA"] = fA.predict_proba(base[["wcr", "age", "female"]])[:, 1]; base["pS"] = fS.predict_proba(base[["age", "female"]])[:, 1]
s_, itl, br, mp = cal(base.y.values, base.pA.values); X["mhi_calibration"] = dict(slope=round(s_, 2), citl=round(itl, 2), mean_pred=round(mp, 3), observed=round(base.y.mean(), 3), brier=round(br, 4))
rows = []
for k in [91, 181, 366]:
    g = base[(base.y == 0) | (base.event_day >= k)]; yy = g.y.values.astype(int); rng = np.random.default_rng(7); B = [rng.integers(0, len(yy), len(yy)) for _ in range(500)]
    bt = np.array([roc_auc_score(yy[b], g.pA.values[b]) - roc_auc_score(yy[b], g.pS.values[b]) for b in B])
    rows.append([f"Events from day {k} only (earlier events removed)", f"{len(g):,} / {yy.sum():,}", f"{roc_auc_score(yy, g.pS):.3f}", f"{roc_auc_score(yy, g.pA):.3f}",
                 fd(roc_auc_score(yy, g.pA) - roc_auc_score(yy, g.pS), np.percentile(bt, 2.5), np.percentile(bt, 97.5))])
X["mhi_landmark_rows"] = rows
X["mhi_early_events"] = dict(by_day90=int((base.event_day < 91).sum()), by_day365=int((base.event_day < 366).sum()), total=int(base.y.sum()))
# Biobank: DeepECG-SSL + age + sex vs age + sex
b = pd.read_csv("results_v4/mhi_biobank_frs_results.csv"); X["biobank_rows"] = b[b.analysis.str.startswith("1. PRIMARY")][["model", "AUROC"]].values.tolist()
# attrition: MUSE patients lacking an MHI patient-index record
pt = pd.read_parquet("/media/data1/gjabbour/df/df_mhi_ptindex.parquet")
X["note_attrition"] = "patients without an MHI patient-index record (no birth year/sex available) could not be aged and were excluded"
json.dump(X, open(os.path.join(HERE, "tables", "_extra2.json"), "w"), indent=1, ensure_ascii=False, default=str)
print(json.dumps({k: v for k, v in X.items() if not k.endswith("rows")}, indent=1, ensure_ascii=False, default=str))
for k in ["in_range_rows", "secondary_rows", "mhi_landmark_rows"]:
    print("==", k); [print(" | ".join(map(str, r))) for r in X[k]]
