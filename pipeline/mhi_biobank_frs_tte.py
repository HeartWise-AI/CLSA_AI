"""Framingham at MHI, MHI Biobank, TIME-TO-EVENT (adds 2016-2021 enrolees to gain events).
Index = ECG nearest Biobank enrolment (|gap| <= 365 d), 2006-04..2021-03. Follow-up from day 8 to the first of: hard
MACE (MedEcho rules of mhi_anyecg_build.py), 6 y, end of MedEcho (2022-03-30), or death outside MHI (censored).
Prevalent CVD (MedEcho rules) and Biobank-reported MI/HF excluded. No ECG follow-up requirement (censoring handles it).
Metric: Harrell's C (lifelines), paired bootstrap 1,000 for delta-C vs Framingham(BMI). Scores and combination
coefficients as in mhi_biobank_frs.py (CLSA-fitted, frozen)."""
import os, json, numpy as np, pandas as pd
from lifelines.utils import concordance_index
from sklearn.linear_model import LogisticRegression
from risk_scores import frs2008_bmi
D, O = "data_v4", "mhi_biobank_tte"; END = pd.Timestamp("2022-03-30")
lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
B = "/media/data1/gjabbour/biobank/"
bb = pd.read_csv(B + "BB_CHARGEAF.csv", sep=";").rename(columns={"ID Biobanque": "BIO_UNIQUE"})
bb = bb.merge(pd.read_parquet("mhi_anyecg/biobank_linked.parquet")[["BIO_UNIQUE", "MRN"]], on="BIO_UNIQUE"); bb["MRN"] = bb.MRN.astype(np.int64)
c = pd.read_parquet(f"{O}/cohort_labels.parquet").merge(pd.read_parquet(f"{O}/predictions.parquet"), on="MRN").merge(bb, on="MRN")
c = c[c.ecg_ok]
c["bmi"] = c["Weight (Kg)"] / (c["Height (cm)"] / 100) ** 2; c["sbp"] = c["Systolic P"]
ok = c.bmi.between(13, 70) & c.sbp.between(70, 260) & c[["Smoking state", "Hypertension medication", "Diabetes"]].notna().all(1)
c["l_frs"] = np.where(ok, lg(frs2008_bmi(c.age.clip(30, 79), c.female, c.bmi.clip(13, 70), c.sbp, c["Hypertension medication"].fillna(0), c["Smoking state"].fillna(0), c["Diabetes"].fillna(0))), np.nan)
inc = c[(c.prev == "no") & (c["Myocardial infarction"] != 1) & (c["Heart failure"] != 1) & c.l_frs.notna()].copy()
pt = pd.read_parquet("/media/data1/gjabbour/df/df_mhi_ptindex.parquet").drop_duplicates("id").set_index("id")
dday = (inc.MRN.map(pt.death_date) - inc.t0).dt.days
cens = np.minimum(2190, (END - inc.t0).dt.days); cens = np.where(inc.unk_death & dday.notna(), np.minimum(cens, dday), cens)
inc["T"] = np.where(inc.y == 1, inc.event_day, cens); inc["E"] = ((inc.y == 1) & (inc.event_day <= cens + 0)).astype(int)
inc = inc[inc["T"] > 7]
# CLSA-fitted combinations
rf = pd.read_parquet("baseline_risk_factors.parquet"); sc = pd.read_parquet(f"{D}/scores_v4.parquet")
dv = pd.concat([sc[sc.split == s].sort_values("row_in_split").reset_index() for s in ["train", "val"]], ignore_index=True)
o = pd.read_parquet(f"{D}/stage1b_oof_all.parquet").reindex(dv.entity_id.values)
dv = dv.join(rf[["bmi", "sbp", "bp_treated", "smoker_current", "diabetes"]].add_suffix("_rf"), on="entity_id")
dv["vit"], dv["wcr"] = lg(o["ViT ecg"].values), lg(o["WCR wcr"].values)
dv["l_frs"] = lg(frs2008_bmi(dv.age.clip(30, 79), dv.female, dv.bmi_rf.fillna(dv.bmi_rf.median()).clip(13, 70), dv.sbp_rf.fillna(dv.sbp_rf.median()),
                             dv.bp_treated_rf.fillna(0), dv.smoker_current_rf.fillna(0), dv.diabetes_rf.fillna(0)))
yd = dv.incident_F2_primary.values.astype(int)
MODELS = {"Framingham 2008 (office/BMI, measured)": "l_frs", "Age + sex": ["age", "female"], "AI-ECG (ViT) alone": "vit", "AI-ECG (WCRv2) alone": "wcr",
          "AI-ECG (ViT) + age + sex": ["vit", "age", "female"], "AI-ECG (WCRv2) + age + sex": ["wcr", "age", "female"],
          "Framingham + AI-ECG (ViT)": ["l_frs", "vit"], "Framingham + AI-ECG (WCRv2)": ["l_frs", "wcr"]}
FIT = {k: LogisticRegression(max_iter=3000).fit(dv[v], yd) for k, v in MODELS.items() if isinstance(v, list)}
REF = "Framingham 2008 (office/BMI, measured)"
def run(name, g):
    g = g.reset_index(drop=True); T, E = g["T"].values, g["E"].values
    P = {k: (g[v].values if isinstance(v, str) else FIT[k].decision_function(g[v])) for k, v in MODELS.items()}
    C = {k: concordance_index(T, -p, E) for k, p in P.items()}
    rng = np.random.default_rng(2026); Bs = [rng.integers(0, len(T), len(T)) for _ in range(1000)]
    bt = {k: np.array([concordance_index(T[b], -p[b], E[b]) for b in Bs]) for k, p in P.items()}; rows = []
    for k in P:
        r = dict(analysis=name, model=k, n=len(T), events=int(E.sum()), median_fu_y=round(np.median(T) / 365.25, 1),
                 C=f"{C[k]:.3f} ({np.percentile(bt[k], 2.5):.3f}-{np.percentile(bt[k], 97.5):.3f})")
        if k != REF:
            d_ = bt[k] - bt[REF]; r["vs Framingham"] = f"{C[k] - C[REF]:+.3f} ({np.percentile(d_, 2.5):+.3f} to {np.percentile(d_, 97.5):+.3f})"
        rows.append(r)
    return rows
R = run("1. PRIMARY: Biobank TTE (enrolment 2006-2021, censor 6 y / 2022-03 / non-MHI death)", inc)
R += run("S1. age 45-85", inc[inc.age.between(45, 85)])
R += run("S2. in MHI patient index (death data)", inc[inc.MRN.isin(pt.index)])
R += run("S3. enrolment <= 2016-03 (full 6-y window possible)", inc[inc.t0 <= "2016-03-30"])
res = pd.DataFrame(R); res.to_csv("results_v4/mhi_biobank_frs_tte_results.csv", index=False)
summ = dict(n=len(inc), events=int(inc.E.sum()), event_types=inc[inc.E == 1].event_type.value_counts().to_dict(), median_fu_y=round(inc["T"].median() / 365.25, 2),
            enrol_years=inc.t0.dt.year.value_counts().sort_index().to_dict())
json.dump(summ, open("results_v4/mhi_biobank_frs_tte_summary.json", "w"), indent=1, default=str); print(json.dumps(summ, indent=1, default=str))
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 70)
for an, g in res.groupby("analysis", sort=False): print("\n==", an, f"(n={g.n.iloc[0]}, events={g.events.iloc[0]}, median FU {g.median_fu_y.iloc[0]} y)"); print(g[["model", "C", "vs Framingham"]].fillna("").to_string(index=False))
