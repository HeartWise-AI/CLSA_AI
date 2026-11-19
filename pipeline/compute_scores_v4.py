"""Compute FRS 2008, PCE 2013, SCORE2/SCORE2-OP (low-risk region; moderate as sensitivity) and PREVENT (base,
total CVD and ASCVD, via R preventr 0.12.0) for the v4 cohort. 10-year risks; age-range flags kept.
Missing inputs: IterativeImputer fitted on TRAIN only (single imputation); a complete-case flag is kept.
Output: data_v4/scores_v4.parquet
"""
import os, subprocess, tempfile, numpy as np, pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from risk_scores import frs2008, pce2013, MMOL_TO_MGDL

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "data_v4")
CONT = ["sbp", "tc_mmol", "hdl_mmol", "egfr", "bmi"]
BIN = ["smoker_current"]

# SCORE2 / SCORE2-OP: coefficients identical to RiskScorescvd::SCORE2 (Hageman 2021; SCORE2-OP 2021), vectorised, unrounded
REGION = {("Low", "m", 0): (-0.5699, 0.7476), ("Low", "f", 0): (-0.7380, 0.7019),
          ("Moderate", "m", 0): (-0.1565, 0.8009), ("Moderate", "f", 0): (-0.3143, 0.7701),
          ("Low", "m", 1): (-0.34, 1.19), ("Low", "f", 1): (-0.52, 1.01),
          ("Moderate", "m", 1): (0.01, 1.25), ("Moderate", "f", 1): (-0.1, 1.1)}


def score2(age, female, smoker, sbp, dm, tc, hdl, region="Low"):
    a, s, d = (age - 60) / 5, smoker, dm
    b, c, h = (sbp - 120) / 20, tc - 6, (hdl - 1.3) / 0.5
    lm = (0.3742 * a + 0.6012 * s + 0.2777 * b + 0.6457 * d + 0.1458 * c - 0.2698 * h
          - 0.0755 * a * s - 0.0255 * a * b - 0.0281 * a * c + 0.0426 * a * h - 0.0983 * a * d)
    lf = (0.4648 * a + 0.7744 * s + 0.3131 * b + 0.8096 * d + 0.1002 * c - 0.2606 * h
          - 0.1088 * a * s - 0.0277 * a * b - 0.0226 * a * c + 0.0613 * a * h - 0.1272 * a * d)
    ao = age - 73; bo = sbp - 150; co = tc - 6; ho = hdl - 1.4
    om = (0.0634 * ao + 0.4245 * d + 0.3524 * s + 0.0094 * bo + 0.085 * co - 0.3564 * ho
          - 0.0174 * ao * d - 0.0247 * ao * s - 0.0005 * ao * bo + 0.0073 * ao * co + 0.0091 * ao * ho)
    of = (0.0789 * ao + 0.601 * d + 0.4921 * s + 0.0102 * bo + 0.0605 * co - 0.304 * ho
          - 0.0107 * ao * d - 0.0255 * ao * s - 0.0004 * ao * bo - 0.0009 * ao * co + 0.0154 * ao * ho)
    raw = np.where(age < 70,
                   np.where(female == 1, 1 - 0.9776 ** np.exp(lf), 1 - 0.9605 ** np.exp(lm)),
                   np.where(female == 1, 1 - 0.8082 ** np.exp(of - 0.229), 1 - 0.7576 ** np.exp(om - 0.0929)))
    op = (age >= 70).astype(int); sx = np.where(female == 1, "f", "m")
    s1 = np.array([REGION[(region, x, o)][0] for x, o in zip(sx, op)])
    s2 = np.array([REGION[(region, x, o)][1] for x, o in zip(sx, op)])
    return 1 - np.exp(-np.exp(s1 + s2 * np.log(-np.log(1 - raw))))


def prevent_r(df):
    """Batch PREVENT via R preventr (base model, 10-year). Ages outside 30-79 -> NA (validated range)."""
    with tempfile.TemporaryDirectory() as td:
        i, o = os.path.join(td, "in.csv"), os.path.join(td, "out.csv")
        dd = pd.DataFrame({"id": df.index, "age": df.age, "sex": np.where(df.female == 1, "female", "male"),
                           "sbp": df.sbp, "bp_tx": df.bp_treated == 1, "total_c": df.tc_mmol, "hdl_c": df.hdl_mmol,
                           "statin": df.statin == 1, "dm": df.diabetes == 1, "smoking": df.smoker_current == 1,
                           "egfr": df.egfr, "bmi": df.bmi})
        for b in ["bp_tx", "statin", "dm", "smoking"]:
            dd[b] = np.where(dd[b], "TRUE", "FALSE")  # R read.csv only parses TRUE/FALSE as logical
        dd.to_csv(i, index=False)
        r = f'''suppressMessages(library(preventr)); d <- read.csv("{i}")
res <- estimate_risk(use_dat=d, time="10yr", model="base", chol_unit="mmol/L", progress=FALSE, quiet=TRUE)
write.csv(res, "{o}", row.names=FALSE)'''
        subprocess.run(["Rscript", "-e", r], check=True, capture_output=True)
        res = pd.read_csv(o)
    res = res.set_index("id")
    return res


def r_score2_check(df, n=200):
    """Cross-check the Python SCORE2/OP port against RiskScorescvd::SCORE2 (which rounds to 0.1%)."""
    s = df.sample(n, random_state=1)
    with tempfile.TemporaryDirectory() as td:
        i, o = os.path.join(td, "i.csv"), os.path.join(td, "o.csv")
        s[["age", "female", "smoker_current", "sbp", "diabetes", "tc_mmol", "hdl_mmol"]].to_csv(i, index=False)
        r = f'''suppressMessages(library(RiskScorescvd)); d <- read.csv("{i}")
x <- mapply(function(a,f,s,b,dm,tc,h) SCORE2("Low", a, ifelse(f==1,"female","male"), s, b, dm, tc, h, classify=FALSE),
            d$age, d$female, d$smoker_current, d$sbp, d$diabetes, d$tc_mmol, d$hdl_mmol)
write.csv(data.frame(r=x), "{o}", row.names=FALSE)'''
        subprocess.run(["Rscript", "-e", r], check=True, capture_output=True)
        rr = pd.read_csv(o).r.values / 100
    py = score2(s.age.values, s.female.values, s.smoker_current.values, s.sbp.values, s.diabetes.values,
                s.tc_mmol.values, s.hdl_mmol.values)
    err = np.abs(np.round(py, 3) - rr).max()
    print(f"SCORE2/OP python vs RiskScorescvd: max |diff| = {err:.4f} (R rounds to 0.001)")
    assert err <= 0.0011


def main():
    co = pd.read_parquet(os.path.join(D, "cohort_v4.parquet"))
    co["complete_case"] = co[CONT + BIN].notna().all(1)
    imp = IterativeImputer(random_state=0, max_iter=20)
    X = co[CONT + BIN + ["age", "female", "diabetes", "bp_treated", "statin"]].astype(float)
    imp.fit(X[co.split == "train"])
    Xi = pd.DataFrame(imp.transform(X), index=co.index, columns=X.columns)
    for c in CONT: co[c] = Xi[c]
    co["smoker_current"] = (Xi.smoker_current >= 0.5).astype(int)
    a, f = co.age.values, co.female.values
    tc, hdl = co.tc_mmol.values * MMOL_TO_MGDL, co.hdl_mmol.values * MMOL_TO_MGDL
    co["frs"] = frs2008(a, f, tc, hdl, co.sbp, co.bp_treated, co.smoker_current, co.diabetes)
    co["pce"] = pce2013(a, f, co.race_pce == "AA", tc, hdl, co.sbp, co.bp_treated, co.smoker_current, co.diabetes)
    args = (a, f, co.smoker_current.values, co.sbp.values, co.diabetes.values, co.tc_mmol.values, co.hdl_mmol.values)
    co["score2"] = score2(*args, region="Low"); co["score2_moderate"] = score2(*args, region="Moderate")
    r_score2_check(co)
    pr = prevent_r(co)
    co["prevent_cvd"] = pr.total_cvd.reindex(co.index).values
    co["prevent_ascvd"] = pr.ascvd.reindex(co.index).values
    co["prevent_input_problems"] = pr.input_problems.reindex(co.index).values
    print("PREVENT input problems:", co.prevent_input_problems.notna().sum(), co.prevent_input_problems.dropna().str[:60].value_counts().head(6).to_dict())
    # Full-coverage PREVENT: winsorise every input to the preventr-accepted range (age 30-79, SBP 90-180,
    # TC 3.36-8.28 / HDL 0.52-2.59 mmol/L, BMI 18.5-39.9, eGFR 15-140); flag who was changed.
    w = co.copy()
    lim = {"age": (30, 79), "sbp": (90, 180), "tc_mmol": (3.37, 8.27), "hdl_mmol": (0.52, 2.58),
           "bmi": (18.5, 39.9), "egfr": (15, 140)}
    changed = np.zeros(len(w), bool)
    for c, (lo, hi) in lim.items():
        changed |= ~w[c].between(lo, hi).values; w[c] = w[c].clip(lo, hi)
    pw = prevent_r(w)
    co["prevent_cvd_winsor"] = pw.total_cvd.reindex(co.index).values
    co["prevent_ascvd_winsor"] = pw.ascvd.reindex(co.index).values
    co["prevent_inputs_winsorised"] = changed
    print("PREVENT winsorised: n changed", int(changed.sum()), "| still NA", int(co.prevent_cvd_winsor.isna().sum()))
    co["in_range_frs"] = co.age.between(30, 74); co["in_range_pce"] = co.age.between(40, 79)
    co["in_range_prevent"] = co.age.between(30, 79)
    co.to_parquet(os.path.join(D, "scores_v4.parquet"))
    print(co[["frs", "pce", "score2", "prevent_cvd", "prevent_cvd_winsor", "prevent_ascvd_winsor"]].describe().round(3).to_string())
    print("in validated age range: FRS", co.in_range_frs.mean().round(3), "PCE", co.in_range_pce.mean().round(3),
          "PREVENT", co.in_range_prevent.mean().round(3), "| complete-case", co.complete_case.mean().round(3))


if __name__ == "__main__":
    main()
