"""Framingham at MHI (MHI Biobank). No MHI source has lipids (Softlab extracts = CBC/creatinine/electrolytes/troponin/
BNP only; searched Notion + NAS), so we use the D'Agostino 2008 OFFICE-BASED (BMI) general-CVD Framingham score, which
the same paper provides for settings without labs (validated here in CLSA: Spearman 0.95 vs lab FRS, AUROC 0.700 vs 0.694).
Inputs measured at Biobank enrolment (BB_CHARGEAF.csv: SBP, height/weight, smoking, antihypertensive medication,
diabetes), linked to chart numbers via BB_POAF_PGS_ECG-AI.csv (11,622 participants).
Index = MUSE ECG closest to enrolment (|gap| <= 365 d; enrolment date: true date, else birth-year + baseline age, else
ECG-age match), dated 2006-04..2016-03; outcome/exclusion rules identical to mhi_anyecg_build.py (6-y MedEcho hard
MACE, non-events need ECG >= 6 y). Also excluded: Biobank-reported baseline MI or HF.
Models: FRS(BMI) alone; AI-ECG alone; AI-ECG + age + sex; FRS + AI-ECG (logistic coefficients fitted in CLSA
development data with the CLSA BMI-FRS, applied frozen). Paired bootstrap 1,000."""
import os, json, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from risk_scores import frs2008_bmi
D, O = "data_v4", "mhi_biobank"
lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
B = "/media/data1/gjabbour/biobank/"
bb = pd.read_csv(B + "BB_CHARGEAF.csv", sep=";").rename(columns={"ID Biobanque": "BIO_UNIQUE"})
lk = pd.read_parquet("mhi_anyecg/biobank_linked.parquet")[["BIO_UNIQUE", "MRN"]]
bb = bb.merge(lk, on="BIO_UNIQUE"); bb["MRN"] = bb.MRN.astype(np.int64)
c = pd.read_parquet(f"{O}/cohort_labels.parquet").merge(pd.read_parquet(f"{O}/predictions.parquet"), on="MRN").merge(bb, on="MRN")
c = c[c.ecg_ok]
c["bmi"] = c["Weight (Kg)"] / (c["Height (cm)"] / 100) ** 2
c["sbp"] = c["Systolic P"]; c["smoker"] = c["Smoking state"]; c["bptx"] = c["Hypertension medication"]; c["dm"] = c["Diabetes"]
ok = c.bmi.between(13, 70) & c.sbp.between(70, 260) & c[["smoker", "bptx", "dm"]].notna().all(1)
c["frs"] = np.where(ok, frs2008_bmi(c.age.clip(30, 79), c.female, c.bmi.clip(13, 70), c.sbp.clip(70, 260), c.bptx.fillna(0), c.smoker.fillna(0), c.dm.fillna(0)), np.nan)
c["l_frs"] = lg(c.frs)
# CLSA development fit for combinations (CLSA BMI-FRS + OOF AI-ECG)
from risk_scores import frs2008_bmi as F
rf = pd.read_parquet("baseline_risk_factors.parquet"); sc = pd.read_parquet(f"{D}/scores_v4.parquet")
dv = pd.concat([sc[sc.split == s].sort_values("row_in_split").reset_index() for s in ["train", "val"]], ignore_index=True)
o = pd.read_parquet(f"{D}/stage1b_oof_all.parquet").reindex(dv.entity_id.values)
dv = dv.join(rf[["bmi", "sbp", "bp_treated", "smoker_current", "diabetes"]], on="entity_id", rsuffix="_rf")
dv["vit"], dv["wcr"] = lg(o["ViT ecg"].values), lg(o["WCR wcr"].values)
dv["l_frs"] = lg(F(dv.age.clip(30, 79), dv.female, dv.bmi.fillna(dv.bmi.median()).clip(13, 70), dv.sbp_rf.fillna(dv.sbp_rf.median()) if "sbp_rf" in dv else dv.sbp.fillna(dv.sbp.median()),
                   dv.bp_treated.fillna(0), dv.smoker_current.fillna(0), dv.diabetes.fillna(0)))
yd = dv.incident_F2_primary.values.astype(int)
MODELS = {"Framingham 2008 (office/BMI)": "l_frs", "Age + sex": ["age", "female"], "AI-ECG (ViT) alone": "vit", "AI-ECG (WCRv2) alone": "wcr",
          "AI-ECG (ViT) + age + sex": ["vit", "age", "female"], "AI-ECG (WCRv2) + age + sex": ["wcr", "age", "female"],
          "Framingham + AI-ECG (ViT)": ["l_frs", "vit"], "Framingham + AI-ECG (WCRv2)": ["l_frs", "wcr"]}
FIT = {k: LogisticRegression(max_iter=3000).fit(dv[v], yd) for k, v in MODELS.items() if isinstance(v, list)}
print("CLSA dev AUROC check: FRS(BMI)", round(roc_auc_score(yd, dv.l_frs), 3))
def evaluate(name, g, y, ref="Framingham 2008 (office/BMI)"):
    g = g.reset_index(drop=True); y = np.asarray(y).astype(int)
    P = {k: (g[v].values if isinstance(v, str) else FIT[k].predict_proba(g[v])[:, 1]) for k, v in MODELS.items()}
    rng = np.random.default_rng(2026); Bs = [rng.integers(0, len(y), len(y)) for _ in range(1000)]
    bt = {k: np.array([roc_auc_score(y[b], p[b]) for b in Bs]) for k, p in P.items()}; rows = []
    for k, p in P.items():
        r = dict(analysis=name, model=k, n=len(y), events=int(y.sum()), AUROC=f"{roc_auc_score(y, p):.3f} ({np.percentile(bt[k], 2.5):.3f}-{np.percentile(bt[k], 97.5):.3f})")
        if k != ref:
            dd = bt[k] - bt[ref]; r["vs Framingham"] = f"{roc_auc_score(y, p) - roc_auc_score(y, P[ref]):+.3f} ({np.percentile(dd, 2.5):+.3f} to {np.percentile(dd, 97.5):+.3f})"
        rows.append(r)
    return rows
excl_bb = (c["Myocardial infarction"] == 1) | (c["Heart failure"] == 1)
inc = c[(c.prev == "no") & ~excl_bb & c.frs.notna()]
base = inc[~inc.unk_death & ((inc.y == 1) | inc.fu_ecg_6y)]
R = evaluate("1. PRIMARY: 6-y hard MACE, Biobank, non-events with ECG >= 6 y", base, base.y)
bb2 = inc[~inc.unk_death & ((inc.y_broad == 1) | inc.fu_ecg_6y)]; R += evaluate("2. BROAD MACE (+revasc >90 d, unstable angina; closest to CLSA primary)", bb2, bb2.y_broad)
s0 = inc[~inc.unk_death]; R += evaluate("S0. no ECG follow-up requirement", s0, s0.y)
s1a = c[c.prev.isin(["no", "unc"]) & ~excl_bb & c.frs.notna() & ~c.unk_death]; s1 = s1a[(s1a.y == 1) | s1a.fu_ecg_6y]; R += evaluate("S1. incl. prevalent-uncertain", s1, s1.y)
s2 = base[base.age.between(45, 85)]; R += evaluate("S2. age 45-85 (CLSA range)", s2, s2.y)
pt = pd.read_parquet("/media/data1/gjabbour/df/df_mhi_ptindex.parquet"); s3 = base[base.MRN.isin(pt.id)]; R += evaluate("S3. only patients in MHI patient index (death data available)", s3, s3.y)
res = pd.DataFrame(R); res.to_csv("results_v4/mhi_biobank_frs_results.csv", index=False)
summ = dict(linked=int(len(bb)), with_index_ecg=int(len(c)), frs_computable=int(c.frs.notna().sum()), primary_n=len(base), events=int(base.y.sum()),
            event_types=base[base.y == 1].event_type.value_counts().to_dict(), age=f"{base.age.mean():.1f}±{base.age.std():.1f}", female=round(base.female.mean(), 3),
            frs_median=round(base.frs.median(), 3), bmi_median=round(base.bmi.median(), 1), sbp_median=base.sbp.median(), smoker=round(base.smoker.mean(), 3),
            bp_treated=round(base.bptx.mean(), 3), diabetes=round(base.dm.mean(), 3))
json.dump(summ, open("results_v4/mhi_biobank_frs_summary.json", "w"), indent=1, default=str); print(json.dumps(summ, indent=1, default=str))
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 70)
for an, g in res.groupby("analysis", sort=False): print("\n==", an, f"(n={g.n.iloc[0]}, events={g.events.iloc[0]})"); print(g[["model", "AUROC", "vs Framingham"]].fillna("").to_string(index=False))
