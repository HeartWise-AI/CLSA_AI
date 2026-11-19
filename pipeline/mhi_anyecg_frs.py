"""Framingham at MHI, any-ECG cohort (index ECG 2012-01..2016-03, 6-y MedEcho hard MACE; mhi_anyecg_build.py).
No lipids exist at MHI -> D'Agostino 2008 office-based (BMI) FRS with administrative inputs, all known by day 30:
  SBP     : cart BP of the index ECG if 'entered' (new value on that cart, not carried over from the previous patient;
            plausible 70-260/30-150), else median of entered BPs on outpatient-clinic ECGs day -365..+30.
            (Pre-2012 cart BPs are placeholders -> index restricted to 2012+.)
  BP treated (proxy) : coded hypertension (MedEcho I10-I15 / 401-405, 2_Diagnosis_Dates) <= day 30
            OR >= 2 outpatient entered BPs >= 140/90 in day -730..+30 (user's suggestion: HTN from outpatient ECG BP).
  Diabetes: coded diabetes <= day 30.  Smoking: MedEcho Z72.0 / ICD-9 305.1 / F17 <= day 30.
  BMI     : MUSE height/weight if plausible (rare), else 32 if coded obesity <= day 30, else sex-specific Biobank median.
Primary: patients with an SBP; sensitivity: SBP imputed (sex/age-band median) for all.
Comparators as in mhi_biobank_frs.py (CLSA-fitted combination coefficients, frozen)."""
import os, glob, json, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from risk_scores import frs2008_bmi
D, O = "data_v4", "mhi_anyecg"; A = "/media/data1/gjabbour/Admin_data/"
lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
mrn = lambda s: pd.to_numeric(s.astype(str).str.replace(r"\D", "", regex=True), errors="coerce")
c = pd.read_parquet(f"{O}/cohort_labels.parquet").merge(pd.read_parquet(f"{O}/predictions.parquet"), on="MRN")
c = c[c.ecg_ok & (c.t0 >= "2012-01-01")].copy(); ids = set(c.MRN); t0 = c.set_index("MRN").t0
# --- BP from cart runs
bp = pd.read_parquet(f"{O}/ecg_bp_runs.parquet"); bp = bp[bp.MRN.isin(ids) & bp.bp_ok].copy(); bp["MRN"] = bp.MRN.astype(np.int64)
bp["dd"] = (bp.d.dt.normalize() - bp.MRN.map(t0)).dt.days
OUTPT = ["CARDIOLOGIE GENERALE", "BUREAU PRIVE", "CLINIQUE", "CL PREV", "INSUFFISANCE", "PRE-EFFORT", "RESIDENTS", "RECHERCHE", "CL TRANSPLANTATION", "pré admission"]
bp["outpt"] = bp.RestingECG_TestDemographics_LocationName.fillna("").str.contains("|".join(OUTPT), case=False)
idx_bp = bp[bp.npy_path.isin(c.npy_path)].groupby("MRN")[["sbp", "dbp"]].first()
near = bp[bp.outpt & bp.dd.between(-365, 30)].groupby("MRN")[["sbp", "dbp"]].median()
c["sbp"] = c.MRN.map(idx_bp.sbp).fillna(c.MRN.map(near.sbp)); c["sbp_src"] = np.where(c.MRN.isin(idx_bp.index), "index", np.where(c.sbp.notna(), "outpatient", "none"))
hi = bp[bp.outpt & bp.dd.between(-730, 30) & ((bp.sbp >= 140) | (bp.dbp >= 90))].groupby("MRN").size()
c["htn_bp"] = c.MRN.map(hi).fillna(0) >= 2
# --- coded conditions (dates)
dd = pd.read_parquet(A + "CSV/2_Diagnosis_Dates.parquet"); dd["MRN"] = mrn(dd.NDOS); dd = dd[dd.MRN.isin(ids)].drop_duplicates("MRN").set_index("MRN")
lim = c.MRN.map(t0) + pd.Timedelta(days=30)
for k, col in [("htn_dx", "MinDate_Hypertension"), ("dm", "MinDate_Diabetes"), ("obese", "MinDate_Obesity")]:
    c[k] = (c.MRN.map(dd[col]) <= lim).fillna(False).values
sm = []
for f in sorted(glob.glob(A + "CSV/1_Diagnostics_MedEcho_*.csv")):
    x = pd.read_csv(f, sep=";", dtype=str, usecols=["NDOS", "StartDateTime", "CDDIAGNOSTIC"]); x["MRN"] = mrn(x.NDOS)
    k = x.CDDIAGNOSTIC.fillna("").str.upper().str.replace(".", "", regex=False)
    sm.append(x[x.MRN.isin(ids) & k.str.startswith(("Z720", "3051", "F17"))][["MRN", "StartDateTime"]])
sm = pd.concat(sm); sm["MRN"] = sm.MRN.astype(np.int64); sm["d"] = pd.to_datetime(sm.StartDateTime, format="mixed")
c["smoker"] = c.MRN.isin(set(sm[sm.d <= sm.MRN.map(t0) + pd.Timedelta(days=30)].MRN))
c["bptx"] = c.htn_dx | c.htn_bp
c["bmi"] = np.where(c.obese, 32.0, np.where(c.female == 1, 26.6, 27.8))  # Biobank sex medians (MHI) ; obesity code -> 32
c["frs"] = frs2008_bmi(c.age.clip(30, 79), c.female, c.bmi, c.sbp.fillna(c.groupby(["female", pd.cut(c.age, [0, 50, 60, 70, 120])]).sbp.transform("median")).clip(70, 260),
                       c.bptx, c.smoker, c.dm)
c["l_frs"] = lg(c.frs)
# --- CLSA fit
rf = pd.read_parquet("baseline_risk_factors.parquet"); sc = pd.read_parquet(f"{D}/scores_v4.parquet")
dv = pd.concat([sc[sc.split == s].sort_values("row_in_split").reset_index() for s in ["train", "val"]], ignore_index=True)
o = pd.read_parquet(f"{D}/stage1b_oof_all.parquet").reindex(dv.entity_id.values)
dv = dv.join(rf[["bmi", "sbp", "bp_treated", "smoker_current", "diabetes"]].add_suffix("_rf"), on="entity_id")
dv["vit"], dv["wcr"] = lg(o["ViT ecg"].values), lg(o["WCR wcr"].values)
dv["l_frs"] = lg(frs2008_bmi(dv.age.clip(30, 79), dv.female, dv.bmi_rf.fillna(dv.bmi_rf.median()).clip(13, 70), dv.sbp_rf.fillna(dv.sbp_rf.median()),
                             dv.bp_treated_rf.fillna(0), dv.smoker_current_rf.fillna(0), dv.diabetes_rf.fillna(0)))
yd = dv.incident_F2_primary.values.astype(int)
MODELS = {"Framingham 2008 (office/BMI, admin inputs)": "l_frs", "Age + sex": ["age", "female"], "AI-ECG (ViT) alone": "vit", "AI-ECG (WCRv2) alone": "wcr",
          "AI-ECG (ViT) + age + sex": ["vit", "age", "female"], "AI-ECG (WCRv2) + age + sex": ["wcr", "age", "female"],
          "Framingham + AI-ECG (ViT)": ["l_frs", "vit"], "Framingham + AI-ECG (WCRv2)": ["l_frs", "wcr"]}
FIT = {k: LogisticRegression(max_iter=3000).fit(dv[v], yd) for k, v in MODELS.items() if isinstance(v, list)}
REF = "Framingham 2008 (office/BMI, admin inputs)"
def evaluate(name, g, y):
    g = g.reset_index(drop=True); y = np.asarray(y).astype(int)
    P = {k: (g[v].values if isinstance(v, str) else FIT[k].predict_proba(g[v])[:, 1]) for k, v in MODELS.items()}
    rng = np.random.default_rng(2026); Bs = [rng.integers(0, len(y), len(y)) for _ in range(1000)]
    bt = {k: np.array([roc_auc_score(y[b], p[b]) for b in Bs]) for k, p in P.items()}; rows = []
    for k, p in P.items():
        r = dict(analysis=name, model=k, n=len(y), events=int(y.sum()), AUROC=f"{roc_auc_score(y, p):.3f} ({np.percentile(bt[k], 2.5):.3f}-{np.percentile(bt[k], 97.5):.3f})")
        if k != REF:
            d_ = bt[k] - bt[REF]; r["vs Framingham"] = f"{roc_auc_score(y, p) - roc_auc_score(y, P[REF]):+.3f} ({np.percentile(d_, 2.5):+.3f} to {np.percentile(d_, 97.5):+.3f})"
        rows.append(r)
    return rows
inc = c[c.prev == "no"]; pa = inc[~inc.unk_death & ((inc.y == 1) | inc.fu_ecg_6y)]
base = pa[pa.sbp.notna()]
R = evaluate("1. PRIMARY: index 2012-16, measured SBP (index or outpatient ECG)", base, base.y)
R += evaluate("S1. all (SBP imputed when missing)", pa, pa.y)
s2 = base[base.sbp_src == "index"]; R += evaluate("S2. SBP from the index ECG only", s2, s2.y)
s3 = base[base.age.between(45, 85)]; R += evaluate("S3. age 45-85", s3, s3.y)
s4 = inc[~inc.unk_death & inc.sbp.notna()]; R += evaluate("S4. no ECG follow-up requirement", s4, s4.y)
res = pd.DataFrame(R); res.to_csv("results_v4/mhi_anyecg_frs_results.csv", index=False)
summ = dict(primary_n=len(base), events=int(base.y.sum()), sbp_src=base.sbp_src.value_counts().to_dict(), sbp_median=base.sbp.median(),
            htn_bp_treated=round(base.bptx.mean(), 3), htn_dx=round(base.htn_dx.mean(), 3), htn_by_outpatient_bp=round(base.htn_bp.mean(), 3),
            diabetes=round(base.dm.mean(), 3), smoker=round(base.smoker.mean(), 3), obese=round(base.obese.mean(), 3), frs_median=round(base.frs.median(), 3),
            age=f"{base.age.mean():.1f}±{base.age.std():.1f}", female=round(base.female.mean(), 3))
json.dump(summ, open("results_v4/mhi_anyecg_frs_summary.json", "w"), indent=1, default=str); print(json.dumps(summ, indent=1, default=str))
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 70)
for an, g in res.groupby("analysis", sort=False): print("\n==", an, f"(n={g.n.iloc[0]}, events={g.events.iloc[0]})"); print(g[["model", "AUROC", "vs Framingham"]].fillna("").to_string(index=False))
