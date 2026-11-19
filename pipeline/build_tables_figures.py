"""Manuscript v2 (iteration 2): every table (v2/tables/*.json) and figure (v2/figures/*.png) generated from existing results.
Sources: results_v5/ (CLSA, DeepECG-SSL), v2/tables/_extra2.json (v2/extra_analyses.py), results_v4/mhi_* and mhi_anyecg/.
Supplementary tables are numbered in order of first citation in the text. Run from manuscript/ with python3."""
import os, sys, re, json, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score
HERE = os.path.dirname(os.path.abspath(__file__)); M = os.path.dirname(HERE); sys.path.insert(0, M); os.chdir(M)
from final_test_evaluation import auc_ci, delong_test, net_benefit, cal, holm
T, F = os.path.join(HERE, "tables"), os.path.join(HERE, "figures"); os.makedirs(T, exist_ok=True); os.makedirs(F, exist_ok=True)
for f in os.listdir(T):
    if f.endswith(".json") and not f.startswith("_"): os.remove(os.path.join(T, f))
lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
f3 = lambda a, lo, hi: f"{a:.3f} ({lo:.3f}–{hi:.3f})"
fd = lambda d, lo, hi: f"{d:+.3f} ({lo:+.3f} to {hi:+.3f})"
fp = lambda p: "<0.001" if p < 0.001 else f"{p:.3f}"
dash = lambda s: re.sub(r"(^|[\s(;])-(?=\d)", lambda m: m.group(1) + "−", str(s)).replace("+-", "−").replace("p=<", "p<").replace("−0.000", "0.000")
pv = lambda p: "p<0.001" if p < 0.001 else f"p={p:.3f}"
ABBRS = {"ACC/AHA": "American College of Cardiology/American Heart Association", "ASCVD": "atherosclerotic cardiovascular disease", "AUPRC": "area under the precision–recall curve",
         "AUROC": "area under the receiver operating characteristic curve", "BMI": "body mass index", "BP": "blood pressure", "CCS": "Canadian Cardiovascular Society", "CI": "confidence interval",
         "CLSA": "Canadian Longitudinal Study on Aging", "CVD": "cardiovascular disease", "ECG": "electrocardiogram", "eGFR": "estimated glomerular filtration rate", "ESC": "European Society of Cardiology",
         "FRS": "Framingham Risk Score", "HDL": "high-density lipoprotein", "MACE": "major adverse cardiovascular events", "MHI": "Montreal Heart Institute", "MI": "myocardial infarction",
         "MUSE": "GE MUSE ECG management system", "NPV": "negative predictive value", "NRI": "net reclassification improvement", "PCE": "Pooled Cohort Equations", "PPV": "positive predictive value",
         "PREVENT": "AHA Predicting Risk of cardiovascular disease EVENTs equations", "SCORE2/OP": "SCORE2/SCORE2-Older Persons", "SCORE2": "Systematic COronary Risk Evaluation 2",
         "TIA": "transient ischaemic attack", "ICD-10-CA": "International Classification of Diseases, 10th revision, Canada", "SD": "standard deviation", "F1": "Follow-up 1", "F2": "Follow-up 2"}
def abbr_for(text):
    found = [k for k in ABBRS if re.search(r"(?<![A-Za-z0-9])" + re.escape(k) + r"(?![A-Za-z0-9])", text)]
    if "SCORE2/OP" in found and "SCORE2" in found: found.remove("SCORE2")
    return ("Abbreviations: " + "; ".join(f"{k}, {ABBRS[k]}" for k in sorted(found, key=str.lower)) + ".") if found else ""
X2 = json.load(open(os.path.join(T, "_extra2.json")))
def save(name, title, header, rows, note=""):
    note = note.replace(" + ABBR", "").strip(); rows = [[dash(c) for c in r] for r in rows]
    a = abbr_for(" ".join([title] + header + [" ".join(r) for r in rows] + [note]))
    note = (dash(note) + " " + a).strip()
    json.dump(dict(title=title, header=header, rows=rows, note=note), open(os.path.join(T, f"{name}.json"), "w"), ensure_ascii=False, indent=1)
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
RED, GREYS = "#c0392b", {"PREVENT": "#1f4e79", "SCORE2/OP": "#5b9bd5", "PCE": "#7f7f7f", "FRS": "#d68910"}
ABBR = ("Abbreviations: ASCVD, atherosclerotic cardiovascular disease; AUPRC, area under the precision–recall curve; AUROC, area under the receiver "
        "operating characteristic curve; CI, confidence interval; CVD, cardiovascular disease; ECG, electrocardiogram; FRS, Framingham Risk Score; "
        "PCE, Pooled Cohort Equations; PREVENT, AHA Predicting Risk of cardiovascular disease EVENTs equations; SCORE2/OP, Systematic COronary Risk "
        "Evaluation 2 / SCORE2-Older Persons.")

# ------------------------------------------------------------------ CLSA data
te = pd.read_parquet("results_v5/test_predictions_wcr.parquet").drop(columns=["DeepECG-SSL ECG-only"]).merge(pd.read_parquet("results_v5/test_predictions_simple.parquet"), on="entity_id")
y = te.incident_F2_primary.values.astype(int)
AP, AS, A = "DeepECG-SSL + age + sex + hypertension", "Age + sex + hypertension", "DeepECG-SSL ECG-only"
sc = pd.read_parquet("data_v4/scores_v4.parquet")
c_ = pd.read_parquet("mhi_anyecg/cohort_labels.parquet").merge(pd.read_parquet("mhi_anyecg/predictions.parquet"), on="MRN"); c_ = c_[c_.ecg_ok]
mbase = c_[(c_.prev == "no") & ~c_.unk_death & ((c_.y == 1) | c_.fu_ecg_6y)].reset_index(drop=True)

# ---- Table 1
def col(d, mhi=False):
    ms = lambda v, k=1: f"{d[v].mean():.{k}f} ({d[v].std():.{k}f})"; pc = lambda v: f"{100 * d[v].mean():.1f}"
    if mhi: return [f"{len(d):,}", ms("age"), pc("female")] + ["—"] * 11 + [f"{100 * (d.age >= 75).mean():.1f}", f"{100 * d.y.mean():.1f}†"]
    return [f"{len(d):,}", ms("age"), pc("female"), pc("hypertension_dx"), ms("sbp", 0), pc("bp_treated"), ms("tc_mmol", 2), ms("hdl_mmol", 2), pc("diabetes"), pc("smoker_current"),
            pc("statin"), ms("bmi"), ms("egfr", 0), f"{100 * (d.race_pce == 'AA').mean():.1f}", f"{100 * (d.age >= 75).mean():.1f}", f"{100 * d.incident_F2_primary.mean():.1f}"]
labs = ["Participants, n", "Age, years", "Female, %", "Hypertension (self-reported diagnosis), %", "Systolic BP, mmHg", "Antihypertensive treatment, %", "Total cholesterol, mmol/L",
        "HDL cholesterol, mmol/L", "Diabetes, %", "Current smoker, %", "Statin use, %", "BMI, kg/m²", "eGFR, mL/min/1.73 m²", "Black race, %", "Age ≥75 years, %", "Incident CVD (outcome), %"]
cols = [col(sc), col(sc[sc.split == "train"]), col(sc[sc.split == "val"]), col(sc[sc.split == "test"]), col(sc[sc.incident_F2_primary == 1]), col(sc[sc.incident_F2_primary == 0]), col(mbase, True)]
for k in (4, 5): cols[k][-1] = "—"
save("T1", "Table 1. Baseline characteristics of the CLSA analytic cohort (by data split and incident CVD status) and of the MHI external-validation cohort",
     ["Characteristic", "CLSA overall", "Training", "Validation", "Test", "Incident CVD", "No incident CVD", "MHI external"], [[l] + [c[i] for c in cols] for i, l in enumerate(labs)],
     "Values are mean (SD) or %. CLSA values at the baseline visit (2010–2015) after imputation of missing clinical values (9.6% of participants; missing before imputation: total and HDL cholesterol and eGFR 8.7% each, smoking 0.6%, systolic BP 0.4%, BMI 0.1%, others <0.1%). MHI: first ECG in 2006–2016; "
     "risk factors other than age and sex are not available in the MHI ECG cohort (—). †6-year hard MACE (MHI outcome), not the CLSA outcome. "
     "eGFR by CKD-EPI 2021.")

# ---- Table 2
MOD = [("Model A+ (primary)", AP, "ECG, age, sex, hypertension", "No"), ("Model A", A, "12-lead ECG only", "No"), ("Model A++", "DeepECG-SSL + retina + age + sex + hypertension", "Model A+ inputs + fundus photographs", "No"),
       ("Model B", "Retina + age + sex + hypertension", "Fundus photographs, age, sex, hypertension", "No"), ("Age, sex and hypertension only", AS, "Age, sex, hypertension", "No"),
       ("Model C", "Clinical risk factors only (logistic, 11 PREVENT/FRS inputs)", "11 risk factors incl. lipids, eGFR", "Yes")]
rf = pd.read_parquet("baseline_risk_factors.parquet")
from risk_scores import frs2008_bmi
t2 = te.join(rf[["bmi", "sbp", "bp_treated", "smoker_current", "diabetes"]].add_suffix("_rf"), on="entity_id")
te["ofrs"] = frs2008_bmi(t2.age.clip(30, 79), t2.female, t2.bmi_rf.fillna(t2.bmi_rf.median()).clip(13, 70), t2.sbp_rf.fillna(t2.sbp_rf.median()), t2.bp_treated_rf.fillna(0), t2.smoker_current_rf.fillna(0), t2.diabetes_rf.fillna(0))
SC = [("FRS 2008, lipid version (CCS-recommended)", "frs", "Yes"), ("FRS 2008, office-based (BMI) version", "ofrs", "No"), ("PCE 2013", "pce", "Yes"),
      ("PREVENT, total CVD (ACC/AHA)", "prevent_cvd_winsor", "Yes"), ("SCORE2/SCORE2-OP (ESC)", "score2", "Yes")]
holmA = X2["holm_Aplus"]; rows = [["DeepECG-SSL and benchmark models", "", "", "", "", "", ""]]
for short, colm, inp, labs_ in MOD:
    a = auc_ci(y, te[colm].values); d1 = delong_test(y, te[colm].values, te.frs.values); d2 = delong_test(y, te[colm].values, te.prevent_cvd_winsor.values)
    p1 = pv(holmA['FRS']) + "*" if colm == AP else pv(d1[3]); p2 = pv(holmA['PREVENT']) + "*" if colm == AP else pv(d2[3])
    rows.append(["   " + short, inp, labs_, f3(*a), f"{average_precision_score(y, te[colm]):.3f}", f"{fd(*d1[:3])}; {p1}", f"{fd(*d2[:3])}; {p2}"])
rows.append(["Guideline risk scores", "", "", "", "", "", ""])
for nm, colm, labs_ in SC:
    a = auc_ci(y, te[colm].values)
    d1 = "reference" if colm == "frs" else (lambda d: f"{fd(*d[:3])}; {pv(d[3])}")(delong_test(y, te[colm].values, te.frs.values))
    d2 = "reference" if colm == "prevent_cvd_winsor" else (lambda d: f"{fd(*d[:3])}; {pv(d[3])}")(delong_test(y, te[colm].values, te.prevent_cvd_winsor.values))
    rows.append(["   " + nm, "Published equation" if colm != "ofrs" else "Published equation, no lipids", labs_, f3(*a), f"{average_precision_score(y, te[colm]):.3f}", d1, d2])
save("T2", "Table 2. Discrimination for incident CVD by Follow-up 2 in the held-out CLSA test set (n = 3,906; 394 events)",
     ["Model", "Inputs", "Blood test needed", "AUROC (95% CI)", "AUPRC", "ΔAUROC vs FRS (95% CI); p", "ΔAUROC vs PREVENT (95% CI); p"], rows,
     "ΔAUROC = model minus comparator. 95% CIs and two-sided p values by DeLong's method. *Holm-adjusted across the four comparisons of Model A+ with FRS, PCE, SCORE2/OP and PREVENT "
     "(model structure predefined; primary designation post hoc, see Methods). Model A+ vs PCE: ΔAUROC +0.024 (+0.008 to +0.039); vs SCORE2/OP: +0.025 (+0.009 to +0.042). Holm p values: "
     f"FRS {fp(holmA['FRS'])}, PCE {fp(holmA['PCE'])}, SCORE2/OP {fp(holmA['SCORE2/OP'])}, PREVENT {fp(holmA['PREVENT'])}; other p values unadjusted. "
     f"Model A+ vs age, sex and hypertension only: ΔAUROC {dash(X2['ecg_increment'])}. Logistic models (A+, A++, B, C) were fitted on development data and applied to the test set without refitting; "
     "hypertension = self-reported physician diagnosis. Event rate 10.1%. ")

dv = pd.concat([sc[sc.split == s] for s in ["train", "val"]]); REC = {}
for nm, colm in [("FRS", "frs"), ("PCE", "pce"), ("PREVENT", "prevent_cvd_winsor"), ("SCORE2/OP", "score2")]:
    m = LogisticRegression(C=1e6).fit(lg(dv[colm].values).reshape(-1, 1), dv.incident_F2_primary.astype(int)); te[nm + " recal"] = m.predict_proba(lg(te[colm].values).reshape(-1, 1))[:, 1]; REC[nm] = colm

# ---- Figure 2
fig, ax = plt.subplots(1, 3, figsize=(17, 5.3))
for nm, cc, k in [("Model A+ (ECG + age, sex, hypertension)", AP, None), ("PREVENT", "prevent_cvd_winsor", "PREVENT"), ("SCORE2/OP", "score2", "SCORE2/OP"), ("PCE", "pce", "PCE"), ("FRS", "frs", "FRS")]:
    f_, t_, _ = roc_curve(y, te[cc]); ax[0].plot(f_, t_, color=RED if k is None else GREYS[k], lw=2.4 if k is None else 1.2, ls="-" if k in (None, "PREVENT", "FRS") else "--", label=f"{nm}: {roc_auc_score(y, te[cc]):.3f}")
ax[0].plot([0, 1], [0, 1], "k:", lw=.8); ax[0].set(xlabel="1 − specificity", ylabel="Sensitivity"); ax[0].legend(fontsize=8.5, loc="lower right")
for nm, cc, colr in [("Model A+", AP, RED), ("PREVENT (recalibrated)", "PREVENT recal", GREYS["PREVENT"]), ("FRS (recalibrated)", "FRS recal", GREYS["FRS"])]:
    q = pd.qcut(te[cc], 10, duplicates="drop"); g = te.groupby(q, observed=True); ax[1].plot(g[cc].mean(), g.incident_F2_primary.mean(), "o-", color=colr, label=nm)
ax[1].plot([0, .4], [0, .4], "k:", lw=.8); ax[1].set(xlabel="Predicted risk by Follow-up 2", ylabel="Observed risk by Follow-up 2"); ax[1].legend(fontsize=8.5)
ths = np.arange(0.02, 0.205, 0.01); ax[2].plot(ths, [y.mean() - (1 - y.mean()) * t / (1 - t) for t in ths], color="0.7", lw=1, label="Treat all"); ax[2].axhline(0, color="k", lw=.6, label="Treat none")
for nm, cc, k in [("Model A+", AP, None), ("PREVENT", "PREVENT recal", "PREVENT"), ("SCORE2/OP", "SCORE2/OP recal", "SCORE2/OP"), ("PCE", "PCE recal", "PCE"), ("FRS", "FRS recal", "FRS")]:
    ax[2].plot(ths, [net_benefit(y, te[cc].values, t) for t in ths], color=RED if k is None else GREYS[k], lw=2.4 if k is None else 1.2, ls="-" if k in (None, "PREVENT", "FRS") else "--", label=nm)
ax[2].set(ylim=(-0.01, 0.07), xlabel="Threshold probability", ylabel="Net benefit"); ax[2].legend(fontsize=8.5)
for i, a_ in enumerate(ax): a_.text(-0.12, 1.02, "ABC"[i], transform=a_.transAxes, fontsize=14, fontweight="bold")
plt.tight_layout(); plt.savefig(os.path.join(F, "Figure2_CLSA_ROC_calibration_DCA.png"), dpi=220); plt.close()

# ---- S4a–c
cA = cal(y, te[AP].values); calrows = [["Model A+", f"{cA[0]:.2f}", f"{cA[1]:.2f}", f"{cA[2]:.3f}", f"{cA[3]:.3f}"]]
for nm, colm in REC.items():
    for lab_, cc in [(f"{nm}, recalibrated", nm + " recal"), (f"{nm}, as published (10-year risk)", colm)]:
        s_, itl, b, mp = cal(y, te[cc].values); calrows.append([lab_, f"{s_:.2f}", f"{itl:.2f}", f"{b:.3f}", f"{mp:.3f}"])
save("S4a", "Supplementary Table S4a. Calibration in the CLSA test set", ["Model", "Calibration slope", "Calibration-in-the-large", "Brier score", "Mean predicted risk"], calrows,
     f"Observed event rate {y.mean():.3f}. Recalibrated = logistic intercept and slope fitted on development data for the observed horizon (median 5.8 years). ")
cats = lambda q: np.digitize(q, [0.05, 0.10]); rows = []
for nm in REC:
    cn, co_ = cats(te[AP].values), cats(te[nm + " recal"].values); up, dn = cn > co_, cn < co_
    ne, nn = up[y == 1].mean() - dn[y == 1].mean(), dn[y == 0].mean() - up[y == 0].mean(); rows.append([f"Model A+ vs {nm} (recalibrated)", f"{ne:+.3f}", f"{nn:+.3f}", f"{ne + nn:+.3f}"])
save("S4b", "Supplementary Table S4b. Categorical net reclassification of Model A+ (CLSA test set)", ["Comparison", "NRI, events", "NRI, non-events", "NRI, total"], rows,
     "Risk categories <5%, 5–10% and ≥10% predicted risk by Follow-up 2. Point estimates only (no confidence intervals); interpret with caution. ")
rows = []
for nm, cc in [("Model A+", AP), ("PREVENT (recalibrated)", "PREVENT recal"), ("FRS (recalibrated)", "FRS recal")]:
    pp = te[cc].values >= 0.10; tp = (pp & (y == 1)).sum(); fpn = (pp & (y == 0)).sum(); fn = (~pp & (y == 1)).sum(); tn = (~pp & (y == 0)).sum()
    rows.append([nm, f"{pp.mean():.1%}", f"{tp / (tp + fn):.3f}", f"{tn / (tn + fpn):.3f}", f"{tp / (tp + fpn):.3f}", f"{tn / (tn + fn):.3f}"] + [f"{net_benefit(y, te[cc].values, t):.4f}" for t in [0.05, 0.10, 0.15, 0.20]])
save("S4c", "Supplementary Table S4c. Operating characteristics at a 10% risk threshold and net benefit at selected thresholds (CLSA test set)",
     ["Model", "Flagged", "Sensitivity", "Specificity", "PPV", "NPV", "Net benefit 5%", "Net benefit 10%", "Net benefit 15%", "Net benefit 20%"], rows,
     "Net benefit point estimates without confidence intervals. ")

# ---- Table 3
g = pd.read_csv("results_v5/guideline_endpoints_bootstrap.csv")
REN = {"AI-ECG alone": "Model A (DeepECG-SSL ECG-only)", "AI-ECG + age + sex": "DeepECG-SSL + age + sex", "AI-ECG + age + sex + hypertension": "Model A+ (DeepECG-SSL + age, sex, hypertension)",
       "Age + sex only": "Age + sex", "FRS (recommended)": "FRS 2008 — recommended by CCS", "FRS + AI-ECG": "FRS + DeepECG-SSL", "PREVENT total CVD": "PREVENT total CVD",
       "PREVENT total CVD + AI-ECG": "PREVENT total CVD + DeepECG-SSL", "PREVENT-ASCVD (recommended)": "PREVENT-ASCVD — recommended by ACC/AHA",
       "PREVENT-ASCVD + AI-ECG": "PREVENT-ASCVD + DeepECG-SSL", "PCE": "PCE", "PCE + AI-ECG": "PCE + DeepECG-SSL",
       "SCORE2/OP (recommended)": "SCORE2/SCORE2-OP — recommended by ESC", "SCORE2/OP + AI-ECG": "SCORE2/OP + DeepECG-SSL"}
EPN = {"CAN_FRS_CVD": "Canada: FRS general-CVD endpoint", "US_ASCVD": "United States: hard ASCVD endpoint", "EU_SCORE2": "Europe: SCORE2 endpoint"}
keep = ["FRS (recommended)", "PREVENT-ASCVD (recommended)", "SCORE2/OP (recommended)", "PREVENT total CVD", "PCE", "Age + sex only", "AI-ECG alone", "AI-ECG + age + sex",
        "AI-ECG + age + sex + hypertension", "FRS + AI-ECG", "PREVENT total CVD + AI-ECG", "PREVENT-ASCVD + AI-ECG", "PCE + AI-ECG", "SCORE2/OP + AI-ECG"]
rows = []
for ep, gg in g.groupby("endpoint", sort=False):
    rows.append([f"{EPN[ep]} ({int(gg.events.iloc[0])} events / {int(gg.n.iloc[0]):,})", "", "", ""])
    for k in keep:
        r = gg[gg.model == k]
        if not len(r): continue
        r = r.iloc[0]; isref = "recommended" in k
        rows.append(["   " + REN.get(k, k), f3(r.AUROC, r.lo, r.hi), "reference" if isref else fd(r.d, r.d_lo, r.d_hi), "" if isref else fp(r.p_boot)])
ca = X2["can_Aplus_vs_prevent_total"]
save("T3", "Table 3. Lab-free DeepECG-SSL models versus the risk score each guideline recommends, on that guideline's own endpoint (CLSA test set)",
     ["Endpoint / model", "AUROC (95% CI)", "ΔAUROC vs recommended score (95% CI)", "p"], rows,
     "Endpoints are new events by Follow-up 2 in participants free of CVD at baseline. Canada: FRS general-CVD definition (self-reported heart disease including heart failure, "
     "myocardial infarction, angina, coronary revascularisation, stroke, TIA, peripheral vascular disease reported at Follow-up 1, or CVD death); the 144 test participants with "
     "baseline peripheral vascular disease were excluded. United States: myocardial infarction, stroke, coronary or stroke death. Europe: myocardial infarction, stroke or CVD death. "
     "For the US and European endpoints other CVD events were treated as non-events. 95% CIs and p values from 1,000 paired bootstrap resamples, unadjusted. On the Canadian endpoint, "
     f"Model A+ vs PREVENT total CVD: ΔAUROC {dash(ca['delta'])}, p = {ca['p']}. Canadian endpoint NRI (categories <5%, 5–10%, ≥10%; scores recalibrated on development data; 1,000 bootstrap resamples): Model A+ vs FRS categorical +0.152 (+0.109 to +0.198; events −0.002, non-events +0.154), continuous +0.340 (+0.244 to +0.439); vs PREVENT total CVD categorical +0.018 (−0.019 to +0.058), continuous +0.128 (+0.030 to +0.224). For each endpoint the logistic models (DeepECG-SSL score with age, sex and hypertension; score + DeepECG-SSL) were refitted on that endpoint in development data and applied to the test set without refitting; these endpoint analyses were not prespecified. "
     "")

fig, ax = plt.subplots(figsize=(10, 6.6)); yy_, labs_, k_ = [], [], 0
show = [("Model A (DeepECG-SSL ECG-only)", "AI-ECG alone"), ("DeepECG-SSL + age + sex", "AI-ECG + age + sex"), ("Model A+ (DeepECG-SSL + age, sex, hypertension)", "AI-ECG + age + sex + hypertension")]
EC = {"CAN_FRS_CVD": "#c0392b", "US_ASCVD": "#1f4e79", "EU_SCORE2": "#17a589"}
for ep, gg in g.groupby("endpoint", sort=False):
    sc_ = [m for m in gg.model if m.endswith("+ AI-ECG") and m.startswith(("FRS", "PREVENT-ASCVD", "SCORE2"))]
    for lab_, m in show + [(REN[sc_[0]], sc_[0])]:
        r = gg[gg.model == m].iloc[0]; ax.errorbar(r.d, -k_, xerr=[[r.d - r.d_lo], [r.d_hi - r.d]], fmt="o", color=EC[ep], capsize=3); yy_.append(-k_); labs_.append(lab_); k_ += 1
    ax.text(0.088, -(k_ - 2.5), {"CAN_FRS_CVD": "Canada vs FRS\n(486 events)", "US_ASCVD": "United States vs\nPREVENT-ASCVD\n(118 events)", "EU_SCORE2": "Europe vs\nSCORE2/OP\n(126 events)"}[ep], va="center", fontsize=9, fontweight="bold", color=EC[ep]); k_ += 0.8
ax.axvline(0, color="k", lw=.8); ax.set_yticks(yy_); ax.set_yticklabels(labs_, fontsize=8.5); ax.set_xlim(-0.10, 0.16)
ax.set_xlabel("ΔAUROC vs the guideline-recommended score (95% CI)"); plt.tight_layout(); plt.savefig(os.path.join(F, "Figure3_guideline_forest.png"), dpi=220); plt.close()

# ---- S3 full-clinical
pm = pd.read_csv("results_v5/metrics_primary.csv"); rows = []
for _, r in pm.iterrows():
    if r.model == "DeepECG-SSL ECG-only": continue
    nm = {"DeepECG-SSL + clinical": "DeepECG-SSL fine-tuned with the 11 clinical inputs (incl. lipids)", "Fusion: DeepECG-SSL + retina": "DeepECG-SSL + clinical, fused with retina + clinical", "Retina + clinical": "Retina + 11 clinical inputs"}.get(r.model, r.model)
    rows.append([nm, r.AUROC.replace("-", "–"), f"{float(r.AUPRC):.3f}", "" if pd.isna(r.get("dAUROC_main_minus")) else r.dAUROC_main_minus, "" if pd.isna(r.get("p_holm")) else fp(r.p_holm)])
cv = pd.read_csv("results_v5/metrics_cv_dev.csv"); sp = pd.read_csv("results_v5/simple_models_primary.csv").set_index("model")
save("S3", "Supplementary Table S3. DeepECG-SSL fine-tuned jointly with the 11 PREVENT/FRS clinical inputs (CLSA test set)",
     ["Model", "AUROC (95% CI)", "AUPRC", "ΔAUROC, joint model minus comparator (95% CI)", "Holm-adjusted p"], rows,
     "Holm adjustment across the four guideline scores. This model requires a lipid panel and creatinine.")
cvrows = [["Model A+ (DeepECG-SSL + age, sex, hypertension)", f"{sp.loc[AP, 'CV_AUROC']:.3f}"], ["Model A (DeepECG-SSL ECG-only)", f"{sp.loc[A, 'CV_AUROC']:.3f}"],
          ["Age, sex and hypertension only", f"{sp.loc[AS, 'CV_AUROC']:.3f}"]]
NM = {"DeepECG-SSL + clinical": "DeepECG-SSL fine-tuned with 11 clinical inputs", "DeepECG-SSL ECG-only": None, "Retina + clinical": "Retina + 11 clinical inputs",
      "Fusion: DeepECG-SSL + retina": "DeepECG-SSL + clinical fused with retina + clinical"}
for _, r_ in cv.iterrows():
    nm_ = NM.get(r_.model, r_.model)
    if nm_: cvrows.append([nm_, r_.AUROC.split(" ")[0]])
save("S3b", "Supplementary Table S3b. Development 5-fold cross-validation AUROC (n = 15,839; 1,520 events)", ["Model", "Mean within-fold AUROC"], cvrows,
     "Neural-network out-of-fold predictions were compared within folds (fold-specific probability scales).")
# ---- S5 in-range, S6 subgroups, S7 secondary
save("S5", "Supplementary Table S5. Model A+ versus guideline scores within each score's validated age range (CLSA test set)",
     ["Restriction", "n / events", "Model A+ AUROC", "Comparator", "Comparator AUROC", "ΔAUROC (95% CI); p"], X2["in_range_rows"],
     "DeLong 95% CIs, unadjusted. PREVENT inputs outside accepted ranges were winsorised in the main analysis (13% of participants). ")
te["age_band"] = pd.cut(te.age, [0, 55, 65, 75, 200], right=False, labels=["45–54", "55–64", "65–74", "≥75"]); rows, fr = [], []
for colm, vals, nm in [("female", [0, 1], ["Men", "Women"]), ("age_band", ["45–54", "55–64", "65–74", "≥75"], None), ("ECG category", ["normal", "borderline", "abnormal"], ["ECG normal", "ECG borderline", "ECG abnormal"]),
                       ("diabetes", [0, 1], ["No diabetes", "Diabetes"]), ("statin", [0, 1], ["No statin", "Statin"])]:
    for i, v in enumerate(vals):
        gg = te[te[colm] == v]; yy = gg.incident_F2_primary.values.astype(int); lab_ = nm[i] if nm else f"Age {v}"
        a = auc_ci(yy, gg[AP].values); b = auc_ci(yy, gg.prevent_cvd_winsor.values); c2 = auc_ci(yy, gg.frs.values)
        d = delong_test(yy, gg[AP].values, gg.prevent_cvd_winsor.values); e = delong_test(yy, gg[AP].values, gg.frs.values)
        rows.append([lab_, f"{len(gg):,} / {yy.sum()}", f3(*a), f3(*c2), f3(*b), fd(*e[:3]), fd(*d[:3])]); fr.append((lab_, e[:3], d[:3]))
save("S6", "Supplementary Table S6. Model A+ by subgroup (CLSA test set; exploratory)", ["Subgroup", "n / events", "Model A+ AUROC", "FRS AUROC", "PREVENT AUROC", "Δ vs FRS (95% CI)", "Δ vs PREVENT (95% CI)"], rows,
     "Exploratory, not adjusted for multiplicity; no formal interaction tests. ECG category from the DeepECG-SSL 77-label diagnostic classifier (normal, borderline, abnormal). "
     "FRS is validated to age 74 and SCORE2 to 69 (SCORE2-OP above), so the ≥75 subgroup lies outside the FRS validated range (see Supplementary Table S5). DeLong 95% CIs. ")
fig, ax = plt.subplots(figsize=(8, 6))
for i, (lab_, e, d) in enumerate(fr):
    ax.errorbar(e[0], -i + 0.12, xerr=[[e[0] - e[1]], [e[2] - e[0]]], fmt="o", color=GREYS["FRS"], capsize=2, label="vs FRS" if i == 0 else None)
    ax.errorbar(d[0], -i - 0.12, xerr=[[d[0] - d[1]], [d[2] - d[0]]], fmt="s", color=GREYS["PREVENT"], capsize=2, label="vs PREVENT" if i == 0 else None)
ax.axvline(0, color="k", lw=.8); ax.set_yticks([-i for i in range(len(fr))]); ax.set_yticklabels([f[0] for f in fr]); ax.legend(); ax.set_xlabel("ΔAUROC, Model A+ minus score (95% CI)")
plt.tight_layout(); plt.savefig(os.path.join(F, "FigureS1_subgroups.png"), dpi=220); plt.close()
co = X2["composition"]
save("S7", "Supplementary Table S7. Model A+ and comparators for secondary and more specific outcomes (CLSA test set)", ["Outcome / model", "AUROC (95% CI)", "ΔAUROC, Model A+ minus model (95% CI)", "p"], X2["secondary_rows"],
     f"Of the {co['cases']:,} incident CVD events in the analytic cohort, {co['heart_item_only']:,} ({100 * co['share_heart_only']:.1f}%) were the non-specific self-reported 'heart disease (including heart failure)' item alone; "
     f"{co['specific']:,} included a specific component (myocardial infarction, angina, stroke, revascularisation or CVD death) and {co['hard']:,} a hard component (myocardial infarction, stroke or CVD death). "
     "F1, Follow-up 1 (median 2.9 years); F2, Follow-up 2 (median 5.8 years). DeLong 95% CIs, unadjusted. ")

# ------------------------------------------------------------------ MHI
r = pd.read_csv("results_v4/mhi_anyecg_results.csv"); s = json.load(open("results_v4/mhi_anyecg_summary.json"))
MREN = {"Age + sex": "Age + sex", "AI-ECG (WCRv2) alone": "DeepECG-SSL ECG-only", "AI-ECG (WCRv2) + age + sex": "DeepECG-SSL + age + sex"}
def blocks(keys):
    out = []
    for key, lab_ in keys:
        gg = r[r.analysis.str.startswith(key)]; out.append([f"{lab_} ({int(gg.events.iloc[0]):,} events / {int(gg.n.iloc[0]):,})", "", ""])
        for m, nm in MREN.items():
            q = gg[gg.model == m].iloc[0]; out.append(["   " + nm, q.AUROC.replace("-", "–"), "reference" if m == "Age + sex" else q["vs age+sex"]])
    return out
rows = blocks([("1. PRIMARY", "Primary: 6-year hard MACE"), ("C. component: HF", "Heart-failure hospitalisation"), ("C. component: stroke", "Stroke"), ("C. component: MI", "Myocardial infarction"),
               ("2. Broad MACE", "Broad MACE (adds coronary revascularisation after day 90 and unstable angina)")])
mc = X2["mhi_calibration"]
save("T4", "Table 4. External validation at the Montreal Heart Institute: DeepECG-SSL with age and sex versus age and sex alone (frozen CLSA coefficients)",
     ["Outcome / model", "AUROC (95% CI)", "ΔAUROC vs age + sex (95% CI)"], rows,
     f"Cohort: each patient's first MHI ECG in 2006–2016 without hospital-coded prior CVD (n = {s['primary_n']:,}; mean age {s['age'].replace('±', ' ± ')} years; {100 * s['female']:.1f}% women). "
     "Outcomes from MedEcho ICD-10-CA discharge codes (MHI admissions only) over 6 years. Component analyses compare patients with that component as first event against patients "
     "without any event (patients whose first event was another component are excluded), hence the different denominators. Logistic coefficients were fitted in CLSA and applied "
     f"unchanged. Calibration of DeepECG-SSL + age + sex at MHI (no recalibration): slope {mc['slope']}, calibration-in-the-large {mc['citl']}, mean predicted {mc['mean_pred']:.3f} vs "
     f"observed {mc['observed']:.3f}; local recalibration would be required. 95% CIs from 1,000 paired bootstrap resamples.")
rows = blocks([("S0.", "No follow-up-ECG requirement"), ("S1. incl", "Including patients with uncertain prior CVD"), ("S1b.", "Excluding all patients with undated prior-CVD codes"),
               ("S2.", "Deaths of unknown cause counted as events"), ("S3.", "Age 45–85 years (CLSA range)")])
for lr in X2["mhi_landmark_rows"]:
    rows.append([f"Landmark: {lr[0].lower()} ({lr[1]} events)".replace(" / ", " patients / "), "", ""])
    rows.append(["   Age + sex", lr[2], "reference"]); rows.append(["   DeepECG-SSL + age + sex", lr[3], lr[4]])
b = pd.read_csv("results_v4/mhi_biobank_frs_results.csv"); bt = pd.read_csv("results_v4/mhi_biobank_frs_tte_results.csv")
for key, lab_, src, cn in [("1. PRIMARY", "MHI clinical subset (measured risk factors and follow-up): 6-year hard MACE", b, "AUROC"), ("2. BROAD", "MHI clinical subset: broad MACE", b, "AUROC"),
                           ("1. PRIMARY", "MHI clinical subset, index to 2021, time-to-event (Harrell's C)", bt, "C")]:
    gg = src[src.analysis.str.startswith(key)]; rows.append([f"{lab_} ({int(gg.events.iloc[0])} events / {int(gg.n.iloc[0]):,})", "", ""])
    for m, nm in [("Framingham 2008", "Framingham 2008, office-based (BMI)"), ("Age + sex", "Age + sex"), ("AI-ECG (WCRv2) + age + sex", "DeepECG-SSL + age + sex"), ("Framingham + AI-ECG (WCRv2)", "Framingham + DeepECG-SSL")]:
        q = gg[gg.model.str.startswith(m)]
        if not len(q): continue
        q = q.iloc[0]; rows.append(["   " + nm, q[cn].replace("-", "–"), "reference (vs Framingham)" if m == "Framingham 2008" else f"vs Framingham {q['vs Framingham']}"])
save("S8", "Supplementary Table S8. MHI sensitivity analyses, landmark analyses and the Framingham comparison in the MHI clinical subset", ["Analysis / model", "AUROC (95% CI)", "ΔAUROC (95% CI)"], rows,
     "Δ vs age + sex unless stated. Clinical subset: office-based (BMI) 2008 Framingham general-CVD equation computed from systolic blood pressure, BMI, smoking, antihypertensive "
     "medication and diabetes measured at a research visit; no lipid values are available at MHI. 95% CIs from paired bootstrap resamples (1,000; 500 for landmark analyses). ")

# ---- Figure 4
o = pd.read_parquet("data_v4/stage1b_oof_all.parquet"); dvp = pd.concat([sc[sc.split == s_].sort_values("row_in_split").reset_index() for s_ in ["train", "val"]], ignore_index=True)
dvp["wcr"] = lg(o.reindex(dvp.entity_id.values)["WCR wcr"].values)
fA = LogisticRegression(max_iter=3000).fit(dvp[["wcr", "age", "female"]], dvp.incident_F2_primary.astype(int)); fS = LogisticRegression(max_iter=3000).fit(dvp[["age", "female"]], dvp.incident_F2_primary.astype(int))
mbase["pA"] = fA.predict_proba(mbase[["wcr", "age", "female"]])[:, 1]; mbase["pS"] = fS.predict_proba(mbase[["age", "female"]])[:, 1]; yb = mbase.y.values
fig, ax = plt.subplots(1, 3, figsize=(17, 5.2))
for nm, cc, colr, lw in [("DeepECG-SSL + age + sex", "pA", RED, 2.4), ("DeepECG-SSL ECG-only", "wcr", "#e6a0a0", 1.3), ("Age + sex", "pS", "0.4", 1.3)]:
    f_, t_, _ = roc_curve(yb, mbase[cc]); ax[0].plot(f_, t_, color=colr, lw=lw, label=f"{nm}: {roc_auc_score(yb, mbase[cc]):.3f}")
ax[0].plot([0, 1], [0, 1], "k:", lw=.8); ax[0].legend(fontsize=9, loc="lower right"); ax[0].set(xlabel="1 − specificity", ylabel="Sensitivity")
comp = [("All hard MACE (primary)", "1. PRIMARY"), ("Heart failure", "C. component: HF"), ("Stroke", "C. component: stroke"), ("Myocardial infarction", "C. component: MI"), (None, None), ("Broad MACE (incl. revascularisation)", "2. Broad MACE")]
yt_, yl_ = [], []
for i, (nm, key) in enumerate(comp):
    if nm is None: continue
    q = r[r.analysis.str.startswith(key) & (r.model == "AI-ECG (WCRv2) + age + sex")].iloc[0]["vs age+sex"]
    d_ = float(q.split(" ")[0]); lo, hi = [float(x) for x in q[q.index("(") + 1:q.index(")")].split(" to ")]
    ax[1].errorbar(d_, -i, xerr=[[d_ - lo], [hi - d_]], fmt="o", color=RED, capsize=3); yt_.append(-i); yl_.append(nm)
ax[1].axvline(0, color="k", lw=.8); ax[1].set_yticks(yt_); ax[1].set_yticklabels(yl_); ax[1].set(xlabel="ΔAUROC vs age + sex (95% CI)")
from lifelines import KaplanMeierFitter
mbase["Q"] = pd.qcut(mbase.pA, 4, labels=["Q1 (lowest)", "Q2", "Q3", "Q4 (highest)"]); Tt = np.where(mbase.y == 1, mbase.event_day, 2190) / 365.25
for qn, cl in zip(["Q1 (lowest)", "Q2", "Q3", "Q4 (highest)"], ["#1a9850", "#fdae61", "#f46d43", "#a50026"]):
    m_ = mbase.Q == qn; k = KaplanMeierFitter().fit(Tt[m_], mbase.y[m_]); ax[2].plot(k.survival_function_.index, 1 - k.survival_function_.iloc[:, 0], color=cl, lw=2, label=f"{qn}: {mbase.y[m_].mean():.1%}")
ax[2].set(xlabel="Years after index ECG", ylabel="Observed cumulative incidence of hard MACE"); ax[2].legend(fontsize=9)
for i, a_ in enumerate(ax): a_.text(-0.12, 1.02, "ABC"[i], transform=a_.transAxes, fontsize=14, fontweight="bold")
plt.tight_layout(); plt.savefig(os.path.join(F, "Figure4_MHI_external.png"), dpi=220); plt.close()
qtab = mbase.groupby("Q", observed=True).y.agg(["size", "sum", "mean"])

# ---- S2 MHI assembly and adjudication
k = pd.read_parquet("mhi_anyecg/sample_key.parquet"); a = pd.read_json("mhi_anyecg/adjudication/adjudicated_all.jsonl", lines=True).astype(str)
mm_ = k.merge(a, left_on="sid", right_on="study_id"); cur = c_.set_index("MRN"); mm_["y_now"] = mm_.MRN.map(cur.y); mm_["prev_now"] = mm_.MRN.map(cur.prev)
nowp = mm_[(mm_.prev_now == "no") & (mm_.y_now == 1)]; pos0 = mm_[mm_.y == 1]; neg0 = mm_[mm_.y == 0]
adj = dict(pos0=len(pos0), ppv0=(pos0.primary_event == "1").mean(), neg0=len(neg0), npv0=(neg0.primary_event == "0").mean(), pos1=len(nowp), ppv1=(nowp.primary_event == "1").mean(),
           unc_prev=(pos0.prevalent_cvd == "uncertain").mean(), incident_ppv0=((pos0.primary_event == "1") & (pos0.prevalent_cvd == "no")).mean())
save("S2", "Supplementary Table S2. MHI external-validation cohort assembly and validation of the outcome definition", ["Step", "n"], [
    ["First MUSE ECG per patient, April 2006 – March 2016", "166,044"],
    ["Excluded: age or sex could not be determined (no MHI patient-index record and no age in the ECG header), or age outside 18–100 years", "81,507"],
    ["Age 18–100 years, sex known", "84,537"],
    ["Excluded: hospital-coded prevalent CVD (coded admission starting ≤ day 7; chronic CVD codes or revascularisation ≤ day 30; in-hospital death ≤ day 7)", "46,085"],
    ["Excluded: uncertain prior CVD (undated history codes such as old myocardial infarction or stent status) — included in a sensitivity analysis", "3,037"],
    ["Without hospital-coded prior CVD at the index ECG", "35,415"],
    ["Excluded: death outside MHI before any event (cause unknown) — counted as events in a sensitivity analysis", "2,069"],
    ["Excluded: no event and no ECG ≥ 6 years later (follow-up not documented) — included in a sensitivity analysis", "11,699"],
    [f"Primary analysis cohort: {s['events']:,} hard MACE (myocardial infarction {s['event_types']['MI']:,}; heart-failure hospitalisation {s['event_types']['HF']}; stroke {s['event_types']['stroke']}; in-hospital death {s['event_types']['CV death']})", f"{s['primary_n']:,}"],
    ["Adjudication sample under the initial rule: rule-positive / rule-negative", f"{adj['pos0']} / {adj['neg0']}"],
    ["   Hard MACE confirmed among rule-positive", f"{adj['ppv0']:.1%}"],
    ["   Absence of hard MACE confirmed among rule-negative", f"{adj['npv0']:.1%}"],
    ["   Incident hard MACE confirmed with no prior CVD among rule-positive", f"{adj['incident_ppv0']:.1%}"],
    ["   Rule-positive with possible undated prior CVD flagged by reviewers", f"{adj['unc_prev']:.1%}"],
    ["After rule revision (in-hospital deaths ≤ day 7 reclassified as presenting events): rule-positive patients in the sample", f"{adj['pos1']}"],
    ["   Hard MACE confirmed among these", f"{adj['ppv1']:.1%}"]],
    "Outcome window day 8 to day 2,190 after the index ECG: myocardial infarction I21/I22 (any position), stroke I60–I64, heart-failure hospitalisation (I50 as principal diagnosis, "
    "which excludes post-operative heart failure), or death during an MHI admission (cause not adjudicated). Adjudication used de-identified dossiers (relative days; no names, chart "
    "numbers or calendar dates) reviewed by eight large-language-model agents (Claude Opus 5.5, Anthropic) following a written charter, without clinician review; the rule revision was "
    "informed by the same sample, so the post-revision value is optimistic.")

# ---- Figure 1
fig, ax = plt.subplots(figsize=(14, 9.5)); ax.axis("off"); ax.set_xlim(-0.01, 1.01); ax.set_ylim(0.07, 1)
def box(x, y_, w, h, t, fc="#eaf2f8", fs=8.2):
    ax.add_patch(FancyBboxPatch((x, y_), w, h, boxstyle="round,pad=0.006", fc=fc, ec="0.3", lw=1)); ax.text(x + w / 2, y_ + h / 2, t, ha="center", va="center", fontsize=fs)
def arr(x1, y1, x2, y2): ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", color="0.3", lw=1))
ax.text(0.24, 0.985, "CLSA (development and internal test)", ha="center", fontsize=11.5, fontweight="bold"); ax.text(0.76, 0.985, "Montreal Heart Institute (external validation)", ha="center", fontsize=11.5, fontweight="bold")
Lx, Lw, Ex, Ew, H_ = 0.0, 0.25, 0.27, 0.215, 0.075; Ly = [0.87, 0.70, 0.53, 0.36]
L = ["CLSA Comprehensive baseline\n(2010–2015)\nn = 30,097", "CVD/TIA-free at baseline\nn = 24,729", "Determinate status at Follow-up 2\n(median 5.8 y)\nn = 20,539 (2,017 incident CVD)", "Baseline ECG and fundus\nphotograph available\nn = 19,745 (1,914 incident CVD)"]
E = ["Baseline CVD or TIA (5,057)\nUnknown baseline status (311)", "Lost to follow-up, non-CVD death\nor indeterminate answers (4,190)", "No ECG or fundus photograph,\nor no split assignment (794)"]
for y_, t in zip(Ly, L): box(Lx, y_, Lw, H_, t)
for i in range(3):
    arr(Lx + Lw / 2, Ly[i], Lx + Lw / 2, Ly[i + 1] + H_); ym = (Ly[i] + Ly[i + 1] + H_) / 2; box(Ex, ym - 0.03, Ew, 0.06, "Excluded:\n" + E[i], fc="#fdedec", fs=7.8); arr(Lx + Lw / 2, ym, Ex, ym)
for i, (t, x) in enumerate([("Training\n13,867\n(1,316 events)", 0.0), ("Validation\n1,972\n(204 events)", 0.165), ("Held-out test\n3,906\n(394 events)", 0.33)]):
    box(x, 0.14, 0.15, 0.09, t, fc="#e9f7ef" if i < 2 else "#fef9e7"); arr(Lx + Lw / 2, Ly[3], x + 0.075, 0.23)
ax.text(0.155, 0.095, "Participant-level 5-fold cross-validation\non training + validation (n = 15,839)", ha="center", fontsize=8.2, style="italic")
Rx, Rw, REx, REw = 0.52, 0.23, 0.77, 0.225
R_ = ["First MHI ECG per patient\n(April 2006 – March 2016)\nn = 166,044", "Age 18–100 y, sex known\nn = 84,537", "No hospital-coded prior CVD\nat the index ECG\nn = 35,415", "6-year follow-up documented\n(event, or ECG ≥ 6 y later)\nn = 21,647 (2,190 hard MACE)"]
RE = ["Age or sex not determinable,\nor age outside 18–100 y (81,507)", "Prevalent CVD (46,085)\nUncertain prior CVD (3,037)", "Death outside MHI before\nany event (2,069); no event and\nno ECG ≥ 6 y later (11,699)"]
for y_, t in zip(Ly, R_): box(Rx, y_, Rw, H_, t, fc="#f4ecf7")
for i in range(3):
    arr(Rx + Rw / 2, Ly[i], Rx + Rw / 2, Ly[i + 1] + H_); ym = (Ly[i] + Ly[i + 1] + H_) / 2; box(REx, ym - 0.035, REw, 0.07, "Excluded:\n" + RE[i], fc="#fdedec", fs=7.8); arr(Rx + Rw / 2, ym, REx, ym)
box(Rx, 0.14, 0.475, 0.09, "MHI clinical subset with measured risk factors and follow-up\n(blood pressure, BMI, smoking, diabetes, antihypertensive treatment)\nn = 1,864 (45 hard MACE): office-based Framingham comparison", fc="#fef9e7", fs=7.8)
arr(Rx + Rw / 2, Ly[3], Rx + Rw / 2, 0.23)
plt.savefig(os.path.join(F, "Figure1_flow.png"), dpi=220, bbox_inches="tight"); plt.close()


# ---- S9: retinal disease signs (graded fundus photographs, Labelbox CLSA_RETINA project)
rs = json.load(open("retina_annotations/sensitivity_retinal_signs.json"))
lab9 = {"all annotated": "All participants with graded photographs", "no signs": "No retinal disease signs", "signs": "Signs of retinal disease (hypertensive retinopathy, diabetic retinopathy or macular oedema)", "indeterminate": "Indeterminate grading only"}
save("S3c", "Supplementary Table S3c. Model A+ by retinal disease signs on graded fundus photographs (CLSA test-set participants; exploratory)",
     ["Group", "n / events", "Model A+ AUROC (95% CI)", "FRS AUROC (95% CI)", "PREVENT AUROC (95% CI)"],
     [[lab9[k], f"{rs[k]['n']} / {rs[k]['events']}", rs[k]["Model A+"], rs[k]["FRS"], rs[k]["PREVENT"]] for k in ["all annotated", "no signs", "signs", "indeterminate"]],
     "Both eyes of 161 test participants were graded by clinical annotators for hypertensive retinopathy, diabetic retinopathy and diabetic macular oedema. Signs = any eye "
     "graded mild or worse (all were hypertensive retinopathy: mild 21 and moderate 4 eyes) or suspected macular oedema; no signs = no positive grade and at least one eye graded "
     "none on all three. AUROC is not estimable with no events. DeLong 95% CIs; very few events, interpret with caution.")

# ---- S1 model details
save("S1", "Supplementary Table S1. DeepECG-SSL fine-tuning and model specification", ["Item", "Specification"], [
    ["Encoder", "DeepECG-SSL (12-layer, 768-dimensional ECG transformer; wav2vec 2.0, contrastive multi-segment coding and random-lead-masking pretraining on ~3.7 million unlabelled 5-s ECG segments from MHI, CODE-15 and MIMIC-IV; amplitude-preserving millivolt scaling); public weights"],
    ["Input", "Resting 12-lead ECG, 10 s; GE CardioSoft raw 5 µV per least significant bit converted to mV; resampled 500→250 Hz (12 × 2,500 samples); no per-lead normalisation. MHI: GE MUSE, 250 Hz, 4.88 µV per bit converted to mV"],
    ["Head", "Mean-pooled 768-d representation, dropout 0.3, linear layer"],
    ["Training", "Full fine-tuning; AdamW (encoder learning rate 2×10⁻⁵, head 1×10⁻³, weight decay 0.01), one-cycle schedule, batch 32, up to 12 epochs, early stopping (patience 4) on the AUROC of a 10% inner validation split; binary cross-entropy weighted for class imbalance; gradient clipping 1.0; seed 0"],
    ["Augmentation", "Amplitude scaling ×0.9–1.1, Gaussian noise (σ 0.01 mV), random drop of 1–2 leads, circular time shift up to ±1 s"],
    ["Cross-validation", "Participant-level stratified 5-fold cross-validation on training + validation (n = 15,839); one model per fold, never trained on test data"],
    ["Calibration and test inference", "Each fold model calibrated by Platt scaling on its own held-out development fold; test prediction = mean of the five calibrated fold predictions"],
    ["Models A+, A++, B", "Logistic regression of the calibrated DeepECG-SSL logit (A+, A++) and/or the retinal score (A++, B) with age, sex and self-reported hypertension, fitted on development out-of-fold predictions and applied to the test set without refitting; the retinal score was rank-normal transformed within folds (development) and within the test set"],
    ["Model A+ equation", "logit(risk) = −3.8386 + 0.6503 × (calibrated DeepECG-SSL logit) + 0.0487 × age (years) − 0.3648 × female + 0.3416 × hypertension (L2-penalised logistic regression, C = 1, fitted on development out-of-fold predictions)"],
    ["Model C", "Logistic regression on the 11 PREVENT/FRS clinical inputs (no ECG)"],
    ["Retina model", "ConvNeXt-Tiny (ImageNet initialisation) shared across both eyes; dark-border crop; contrast-limited adaptive histogram equalisation; same folds"],
    ["MHI model", "DeepECG-SSL + age + sex: logit of the mean of the five fold predictions with age and sex; coefficients fitted on CLSA development out-of-fold predictions and applied unchanged"]], "")

json.dump(dict(adj=adj, mhi_quartiles=qtab.reset_index().astype(str).values.tolist(), ofrs=X2["office_frs_test"], calA=list(cA)), open(os.path.join(T, "_extra_numbers.json"), "w"), indent=1, default=float)
print("tables:", sorted(os.listdir(T))); print("adj", adj); print(qtab)
