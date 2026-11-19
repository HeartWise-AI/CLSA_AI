"""Reviewer requests, round 3 (existing data and released models only). Output: v2/tables/_extra3.json, v2/figures/FigureS2_MHI_calibration.png
A. CLSA missing F2 status (4,190 indeterminate): inverse-probability-of-follow-up weighting of the test set; outcome bounds
   using DeepECG-SSL/score predictions for the indeterminate participants (v2/infer_indeterminate.py, v2/scores_indeterminate.py).
B. CLSA test set: bootstrap CIs (1,000) for AUPRC, calibration, net benefit and operating characteristics at 5%, 7.5%, 10% and 20%.
C. MHI: model equations, performance by sex and age group, calibration of both models (as transported and after logistic recalibration).
Run from manuscript/ with python3."""
import os, sys, json, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
H = os.path.dirname(os.path.abspath(__file__)); M = os.path.dirname(H); sys.path.insert(0, M); os.chdir(M)
from risk_scores import frs2008_bmi
lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
f3 = lambda a, lo, hi: f"{a:.3f} ({lo:.3f}–{hi:.3f})"; fd = lambda d, lo, hi: f"{d:+.3f} ({lo:+.3f} to {hi:+.3f})"
ci = lambda v: (np.nanpercentile(v, 2.5), np.nanpercentile(v, 97.5))
NB = 1000; X = {}


def cal_fast(y, p):
    """Calibration slope (logistic of y on logit p), calibration-in-the-large (intercept with logit p as offset), Brier."""
    x = lg(p); slope = LogisticRegression(C=1e6).fit(x.reshape(-1, 1), y).coef_[0, 0]; a = 0.0
    for _ in range(25):
        q = expit(x + a); a += (y - q).sum() / max((q * (1 - q)).sum(), 1e-9)
    return slope, a, np.mean((p - y) ** 2)


def nb(y, p, t):
    n = len(y); f = p >= t; return ((f & (y == 1)).sum() - (f & (y == 0)).sum() * t / (1 - t)) / n


def opc(y, p, t):
    f = p >= t; tp = (f & (y == 1)).sum(); fp_ = (f & (y == 0)).sum(); fn = (~f & (y == 1)).sum(); tn = (~f & (y == 0)).sum()
    return np.array([f.mean(), tp / max(tp + fn, 1), tn / max(tn + fp_, 1), tp / max(tp + fp_, 1), tn / max(tn + fn, 1)])


# ============================================================== CLSA test set
te = pd.read_parquet("results_v5/test_predictions_wcr.parquet").drop(columns=["DeepECG-SSL ECG-only"]).merge(pd.read_parquet("results_v5/test_predictions_simple.parquet"), on="entity_id")
y = te.incident_F2_primary.values.astype(int)
AP = "DeepECG-SSL + age + sex + hypertension"
sc = pd.read_parquet("data_v4/scores_v4.parquet"); dv = pd.concat([sc[sc.split == s] for s in ["train", "val"]])
REC = {}
for nm, colm in [("FRS", "frs"), ("PCE", "pce"), ("PREVENT", "prevent_cvd_winsor"), ("SCORE2/OP", "score2")]:
    m = LogisticRegression(C=1e6).fit(lg(dv[colm].values).reshape(-1, 1), dv.incident_F2_primary.astype(int))
    te[nm + " recal"] = m.predict_proba(lg(te[colm].values).reshape(-1, 1))[:, 1]; REC[nm] = (colm, m)
rf = pd.read_parquet("baseline_risk_factors.parquet")
t2 = te.join(rf[["bmi", "sbp", "bp_treated", "smoker_current", "diabetes"]].add_suffix("_rf"), on="entity_id")
te["ofrs"] = frs2008_bmi(t2.age.clip(30, 79), t2.female, t2.bmi_rf.fillna(t2.bmi_rf.median()).clip(13, 70), t2.sbp_rf.fillna(t2.sbp_rf.median()), t2.bp_treated_rf.fillna(0), t2.smoker_current_rf.fillna(0), t2.diabetes_rf.fillna(0))
rng = np.random.default_rng(2026); BS = [rng.integers(0, len(y), len(y)) for _ in range(NB)]
BS = [b for b in BS if 0 < y[b].sum() < len(b)]

# ---- B1. AUPRC with bootstrap CIs
AUPRC_MODELS = [("Model A+ (primary)", AP), ("Model A", "DeepECG-SSL ECG-only (calibrated)"), ("Model A++", "DeepECG-SSL + retina + age + sex + hypertension"),
                ("Model B", "Retina + age + sex + hypertension"), ("Age, sex and hypertension only", "Age + sex + hypertension"),
                ("Model C", "Clinical risk factors only (logistic, 11 PREVENT/FRS inputs)"), ("FRS, lipid version", "frs"), ("FRS, office-based", "ofrs"),
                ("PCE", "pce"), ("PREVENT, total CVD", "prevent_cvd_winsor"), ("SCORE2/SCORE2-OP", "score2")]
assert np.allclose(te["DeepECG-SSL ECG-only (calibrated)"].rank(), te["DeepECG-SSL ECG-only (calibrated)"].rank())
rows = []
for nm, c in AUPRC_MODELS:
    p = te[c].values; a0 = average_precision_score(y, p); bt = np.array([average_precision_score(y[b], p[b]) for b in BS])
    if c != AP:
        pa = te[AP].values; d0 = average_precision_score(y, pa) - a0; bd = np.array([average_precision_score(y[b], pa[b]) - average_precision_score(y[b], p[b]) for b in BS])
        rows.append([nm, f3(a0, *ci(bt)), fd(d0, *ci(bd))])
    else: rows.append([nm, f3(a0, *ci(bt)), "reference"])
X["auprc_rows"] = rows; X["prevalence"] = round(float(y.mean()), 3)
print("AUPRC done")

# ---- B2. calibration with bootstrap CIs
CALM = [("Model A+", AP)] + [(f"{k}, recalibrated", k + " recal") for k in REC] + [(f"{k}, as published (10-year risk)", REC[k][0]) for k in REC]
rows = []
for nm, c in CALM:
    p = te[c].values; s0, i0, b0 = cal_fast(y, p); bt = np.array([cal_fast(y[b], p[b]) for b in BS])
    rows.append([nm, f"{s0:.2f} ({ci(bt[:, 0])[0]:.2f}–{ci(bt[:, 0])[1]:.2f})", f"{i0:.2f} ({ci(bt[:, 1])[0]:.2f} to {ci(bt[:, 1])[1]:.2f})",
                 f"{b0:.4f} ({ci(bt[:, 2])[0]:.4f}–{ci(bt[:, 2])[1]:.4f})", f"{p.mean():.3f}"])
X["cal_rows"] = rows; print("calibration done")

# ---- B3. net benefit and operating characteristics at 5%, 7.5%, 10%, 20% (bootstrap)
THS = [0.05, 0.075, 0.10, 0.20]
OPM = [("Model A+", AP), ("PREVENT (recalibrated)", "PREVENT recal"), ("SCORE2/OP (recalibrated)", "SCORE2/OP recal"), ("PCE (recalibrated)", "PCE recal"), ("FRS (recalibrated)", "FRS recal")]
rows, nbrows = [], []
for t in THS:
    pa = te[AP].values
    for nm, c in OPM:
        p = te[c].values; o0 = opc(y, p, t); ob = np.array([opc(y[b], p[b], t) for b in BS])
        n0 = nb(y, p, t); nbb = np.array([nb(y[b], p[b], t) for b in BS])
        cell = lambda k, pct=False: (f"{100 * o0[k]:.1f}% ({100 * ci(ob[:, k])[0]:.1f}–{100 * ci(ob[:, k])[1]:.1f})" if pct else f"{o0[k]:.3f} ({ci(ob[:, k])[0]:.3f}–{ci(ob[:, k])[1]:.3f})")
        rows.append([f"{100 * t:g}%", nm, cell(0, True), cell(1), cell(2), cell(3), cell(4)])
        if c == AP: nbrows.append([f"{100 * t:g}%", f"{y.mean() - (1 - y.mean()) * t / (1 - t):.4f}", f"{n0:.4f} ({ci(nbb)[0]:.4f} to {ci(nbb)[1]:.4f})", "reference"])
        else:
            dnb = np.array([nb(y[b], pa[b], t) - nb(y[b], p[b], t) for b in BS]); d0 = nb(y, pa, t) - n0
            nbrows.append([f"{100 * t:g}%", "", f"{nm}: {n0:.4f} ({ci(nbb)[0]:.4f} to {ci(nbb)[1]:.4f})", f"{d0:+.4f} ({ci(dnb)[0]:+.4f} to {ci(dnb)[1]:+.4f})"])
X["op_rows"] = rows; X["nb_rows"] = nbrows; print("operating points done")

# ============================================================== A. missing F2 status
ind = pd.read_parquet("v2/indeterminate/scores.parquet")
L = pd.read_csv("incident_cvd_labels_v2_components.csv", index_col=0)
COV = ["age", "female", "sbp", "bmi", "smoker_current", "diabetes", "bp_treated", "hypertension_dx", "statin", "tc_mmol", "hdl_mmol", "egfr"]
det = sc[COV + ["province", "split"]].copy(); det["R"] = 1
nd = ind[COV + ["province"]].copy(); nd["R"] = 0; nd["split"] = "indeterminate"
P = pd.concat([det, nd]); P["province"] = P.province.fillna("NA")
Z = pd.get_dummies(P[COV + ["province"]].assign(age2=(P.age - 65) ** 2, age_f=P.age * P.female), columns=["province"], drop_first=True).astype(float)
mu, sd = Z.mean(), Z.std().replace(0, 1); Zs = (Z - mu) / sd
pm = LogisticRegression(C=1e6, max_iter=5000).fit(Zs, P.R); pr = pm.predict_proba(Zs)[:, 1]
X["followup_model_auc"] = round(float(roc_auc_score(P.R, pr)), 3)
P["pR"] = pr; w = (P.R.mean() / P.pR).values; lo_, hi_ = np.percentile(w[P.R == 1], [1, 99]); P["w"] = np.clip(w, lo_, hi_)
X["followup_or"] = {k: round(float(np.exp(v)), 2) for k, v in zip(Z.columns, pm.coef_[0]) if k in ["age", "female", "smoker_current", "diabetes", "hypertension_dx", "egfr"]}
ww = P.loc[te.entity_id.values, "w"].values
X["weights"] = dict(min=round(float(ww.min()), 2), max=round(float(ww.max()), 2), mean=round(float(ww.mean()), 2))
X["ind_vs_det"] = dict(n_ind=int(len(L) - L.incident_F2_primary.notna().sum()), n_scored=int(len(ind)), age_ind=round(float(ind.age.mean()), 1), age_det=round(float(sc.age.mean()), 1),
                       female_ind=round(float(ind.female.mean()), 3), female_det=round(float(sc.female.mean()), 3),
                       Aplus_ind=round(float(ind.Aplus.mean()), 3), Aplus_test=round(float(te[AP].mean()), 3), frs_ind=round(float(ind.frs.mean()), 3), frs_det=round(float(sc.frs.mean()), 3))
COMP = [("FRS", "frs"), ("PCE", "pce"), ("SCORE2/OP", "score2"), ("PREVENT", "prevent_cvd_winsor")]
def wauc(yy, p, ww_): return roc_auc_score(yy, p, sample_weight=ww_)
rows = []
for lab_, wv in [("Unweighted (complete case, as reported)", np.ones(len(y))), ("Inverse-probability-of-follow-up weighted", ww)]:
    a0 = wauc(y, te[AP].values, wv); ba = np.array([wauc(y[b], te[AP].values[b], wv[b]) for b in BS])
    r_ = [lab_, f3(a0, *ci(ba))]
    for nm, c in COMP:
        d0 = a0 - wauc(y, te[c].values, wv); bd = np.array([wauc(y[b], te[AP].values[b], wv[b]) - wauc(y[b], te[c].values[b], wv[b]) for b in BS]); r_.append(fd(d0, *ci(bd)))
    rows.append(r_)
X["ipw_rows"] = rows; print("IPW done")
# bounds: indeterminate participants added to the test set with weight 3,906/19,745 (their expected share of a 20% test split)
wi = len(te) / len(sc); q = float(y.mean())
yA, sA = y, {c: te[c].values for _, c in COMP}; sA[AP] = te[AP].values
iS = {c: ind[c].values for _, c in COMP}; iS[AP] = ind.Aplus.values; ni = len(ind)
def combo(yi):
    yy = np.r_[y, yi]; wv = np.r_[np.ones(len(y)), np.full(ni, wi)]
    a = wauc(yy, np.r_[sA[AP], iS[AP]], wv); return [a] + [a - wauc(yy, np.r_[sA[c], iS[c]], wv) for _, c in COMP]
k = int(round(q * ni)); rA = pd.Series(iS[AP]).rank().values; rF = pd.Series(iS["frs"]).rank().values
scen = []
scen.append(("All indeterminate participants without CVD", combo(np.zeros(ni, int))))
scen.append(("All indeterminate participants with CVD", combo(np.ones(ni, int))))
yi = np.zeros(ni, int); yi[np.argsort(rA)[:k]] = 1; scen.append((f"Worst case for Model A+: events ({100 * q:.1f}%) in those with the lowest Model A+ risk", combo(yi)))
yi = np.zeros(ni, int); yi[np.argsort(-rA)[:k]] = 1; scen.append((f"Best case for Model A+: events ({100 * q:.1f}%) in those with the highest Model A+ risk", combo(yi)))
yi = np.zeros(ni, int); yi[np.argsort(rA - rF)[:k]] = 1; scen.append((f"Worst case for Model A+ vs FRS: events ({100 * q:.1f}%) where Model A+ ranks lowest relative to FRS", combo(yi)))
yi = np.zeros(ni, int); yi[np.argsort(rF - rA)[:k]] = 1; scen.append((f"Best case for Model A+ vs FRS: events ({100 * q:.1f}%) where Model A+ ranks highest relative to FRS", combo(yi)))
rs_ = np.random.default_rng(1); sims = np.array([combo(rs_.binomial(1, np.clip(ind.Aplus.values, 0, 1))) for _ in range(200)])
scen.append(("Events drawn from Model A+ predicted risk (200 draws; median and 2.5th–97.5th percentiles)", [f"{np.median(sims[:, j]):.3f} ({np.percentile(sims[:, j], 2.5):.3f} to {np.percentile(sims[:, j], 97.5):.3f})" for j in range(5)]))
rs_ = np.random.default_rng(2); sims = np.array([combo(rs_.binomial(1, np.clip(ind.frs.values * 0.6, 0, 1))) for _ in range(200)])
scen.append(("Events drawn from FRS predicted risk ×0.6 (≈6-year horizon; 200 draws)", [f"{np.median(sims[:, j]):.3f} ({np.percentile(sims[:, j], 2.5):.3f} to {np.percentile(sims[:, j], 97.5):.3f})" for j in range(5)]))
X["bound_rows"] = [[lab_] + [v if isinstance(v, str) else (f"{v:.3f}" if j == 0 else f"{v:+.3f}") for j, v in enumerate(vals)] for lab_, vals in scen]
X["bound_meta"] = dict(n_ind_scored=ni, weight=round(wi, 3), k_events=k, n_no_ecg=int(X["ind_vs_det"]["n_ind"] - ni))
print("bounds done")

# ============================================================== C. MHI
c = pd.read_parquet("mhi_anyecg/cohort_labels.parquet").merge(pd.read_parquet("mhi_anyecg/predictions.parquet"), on="MRN"); c = c[c.ecg_ok]
base = c[(c.prev == "no") & ~c.unk_death & ((c.y == 1) | c.fu_ecg_6y)].reset_index(drop=True)
dvp = pd.concat([sc[sc.split == s].sort_values("row_in_split").reset_index() for s in ["train", "val"]], ignore_index=True)
o = pd.read_parquet("data_v4/stage1b_oof_all.parquet").reindex(dvp.entity_id.values); dvp["wcr"] = lg(o["WCR wcr"].values)
fA = LogisticRegression(max_iter=3000).fit(dvp[["wcr", "age", "female"]], dvp.incident_F2_primary.astype(int))
fS = LogisticRegression(max_iter=3000).fit(dvp[["age", "female"]], dvp.incident_F2_primary.astype(int))
X["mhi_eq"] = dict(A=dict(intercept=round(float(fA.intercept_[0]), 4), wcr=round(float(fA.coef_[0, 0]), 4), age=round(float(fA.coef_[0, 1]), 4), female=round(float(fA.coef_[0, 2]), 4)),
                   S=dict(intercept=round(float(fS.intercept_[0]), 4), age=round(float(fS.coef_[0, 0]), 4), female=round(float(fS.coef_[0, 1]), 4)))
base["pA"] = fA.predict_proba(base[["wcr", "age", "female"]])[:, 1]; base["pS"] = fS.predict_proba(base[["age", "female"]])[:, 1]
yb = base.y.values.astype(int)
# subgroups
grp = [("Men", base.female == 0), ("Women", base.female == 1), ("Age 18–44 years", base.age < 45), ("Age 45–64 years", base.age.between(45, 64.999)),
       ("Age 65–74 years", base.age.between(65, 74.999)), ("Age ≥75 years", base.age >= 75)]
rows = []
for nm, mk in grp:
    g = base[mk.values]; yy = g.y.values.astype(int); r2 = np.random.default_rng(7); B2 = [r2.integers(0, len(yy), len(yy)) for _ in range(NB)]
    aA, aS = roc_auc_score(yy, g.pA), roc_auc_score(yy, g.pS)
    bA = np.array([roc_auc_score(yy[b], g.pA.values[b]) for b in B2]); bS = np.array([roc_auc_score(yy[b], g.pS.values[b]) for b in B2])
    rows.append([nm, f"{len(g):,} / {yy.sum():,}", f3(aS, *ci(bS)), f3(aA, *ci(bA)), fd(aA - aS, *ci(bA - bS)), f"{g.pA.mean():.3f} / {yy.mean():.3f}"])
X["mhi_sub_rows"] = rows; print("MHI subgroups done")
# calibration as transported and after logistic recalibration (5-fold cross-fitted at MHI)
from sklearn.model_selection import StratifiedKFold
r3 = np.random.default_rng(11); B3 = [r3.integers(0, len(yb), len(yb)) for _ in range(NB)]
rows = []
for nm, cc in [("DeepECG-SSL + age + sex", "pA"), ("Age + sex", "pS")]:
    p = base[cc].values; rc = np.zeros(len(p))
    for tr, ts in StratifiedKFold(5, shuffle=True, random_state=0).split(p, yb):
        rc[ts] = LogisticRegression(C=1e6).fit(lg(p[tr]).reshape(-1, 1), yb[tr]).predict_proba(lg(p[ts]).reshape(-1, 1))[:, 1]
    base[cc + "_rc"] = rc
    for lab_, pp in [("as transported from CLSA", p), ("after logistic recalibration at MHI (5-fold cross-fitted)", rc)]:
        s0, i0, b0 = cal_fast(yb, pp); bt = np.array([cal_fast(yb[b], pp[b]) for b in B3])
        rows.append([f"{nm}, {lab_}", f"{s0:.2f} ({ci(bt[:, 0])[0]:.2f}–{ci(bt[:, 0])[1]:.2f})", f"{i0:.2f} ({ci(bt[:, 1])[0]:.2f} to {ci(bt[:, 1])[1]:.2f})",
                     f"{b0:.4f} ({ci(bt[:, 2])[0]:.4f}–{ci(bt[:, 2])[1]:.4f})", f"{pp.mean():.3f}", f"{yb.mean():.3f}"])
X["mhi_cal_rows"] = rows; print("MHI calibration done")
fig, ax = plt.subplots(1, 2, figsize=(11.5, 5))
for j, (ttl, suf) in enumerate([("As transported from CLSA", ""), ("After logistic recalibration at MHI", "_rc")]):
    for nm, cc, colr in [("DeepECG-SSL + age + sex", "pA", "#c0392b"), ("Age + sex", "pS", "0.4")]:
        v = base[cc + suf]; qq = pd.qcut(v, 10, duplicates="drop"); g = base.groupby(qq, observed=True)
        obs = g.y.mean(); n_ = g.y.size(); se = np.sqrt(obs * (1 - obs) / n_)
        ax[j].errorbar(g[cc + suf].mean(), obs, yerr=1.96 * se, fmt="o-", color=colr, capsize=2, label=nm)
    ax[j].plot([0, .6], [0, .6], "k:", lw=.8); ax[j].set(xlim=(0, .6), ylim=(0, .6), xlabel="Predicted 6-year risk", ylabel="Observed 6-year hard MACE", title=ttl)
    ax[j].legend(fontsize=9, loc="upper left"); ax[j].text(-0.12, 1.02, "AB"[j], transform=ax[j].transAxes, fontsize=14, fontweight="bold")
plt.tight_layout(); plt.savefig(os.path.join(H, "figures", "FigureS2_MHI_calibration.png"), dpi=220); plt.close()

json.dump(X, open(os.path.join(H, "tables", "_extra3.json"), "w"), indent=1, ensure_ascii=False, default=str)
for k_, v_ in X.items():
    if k_.endswith("rows"): print("==", k_); [print(" | ".join(map(str, r))) for r in v_]
    else: print(k_, v_)
