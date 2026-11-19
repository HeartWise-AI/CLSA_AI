"""Manuscript v2 (2026-10-06): CLSA locked-test evaluation with DeepECG-SSL (WCRv2) as THE AI-ECG model.
Transparency: the pre-specified primary model was a ViT-based ensemble (final_test_evaluation.py, test AUROC 0.733); the
two were tied in development CV (0.735 vs 0.731) and DeepECG-SSL transported better to MHI, so the manuscript uses
DeepECG-SSL; its test evaluation is therefore post hoc (stated in Methods; ViT result kept in the supplement).

Models (5 CLSA fold models each, mean of fold probabilities; frozen):
  DeepECG-SSL ECG-only           stage1b_wcr/model_wcr_fold{k}_seed0.pt
  DeepECG-SSL + clinical (main)  stage1b_wcr/model_wcr_clin_fold{k}_seed0.pt (11 clinical inputs = PREVENT/FRS inputs)
  Retina + clinical (existing test predictions), and rank fusion DeepECG-SSL+clinical with retina+clinical (CV and test).
Outputs: results_v5/ (metrics_*.csv/json, test_predictions_wcr.parquet, figures)."""
import os, json, numpy as np, pandas as pd, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from stage1b_wcr_cv import Net, CLIN
from final_test_evaluation import delong_test, auc_ci, holm, cal, net_benefit

HERE = os.path.dirname(os.path.abspath(__file__)); D = os.path.join(HERE, "data_v4"); OUT = os.path.join(HERE, "results_v5"); os.makedirs(OUT, exist_ok=True)
lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
rk = lambda s: rankdata(s) / len(s)
MAIN, ECGO = "DeepECG-SSL + clinical", "DeepECG-SSL ECG-only"


def infer():
    dev = torch.device("cuda:0"); sc = pd.read_parquet(os.path.join(D, "scores_v4.parquet"))
    devdf = pd.concat([sc[sc.split == s].sort_values("row_in_split").reset_index() for s in ["train", "val"]], ignore_index=True)
    te = pd.read_parquet(os.path.join(HERE, "results_v4", "test_predictions.parquet"))
    X = np.load(os.path.join(D, "ecg_wcr_test.npy")); y_dev = devdf.incident_F2_primary.values.astype(int)
    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(devdf, y_dev)); pc, pe = [], []
    for f in range(5):
        tr, _ = train_test_split(folds[f][0], test_size=0.1, stratify=y_dev[folds[f][0]], random_state=0)
        C = StandardScaler().fit(devdf.loc[tr, CLIN]).transform(te[CLIN]).astype(np.float32)
        for k, nclin, P in [(f"model_wcr_clin_fold{f}_seed0.pt", len(CLIN), pc), (f"model_wcr_fold{f}_seed0.pt", 0, pe)]:
            m = Net(nclin); m.load_state_dict(torch.load(os.path.join(D, "stage1b_wcr", k), map_location="cpu")); m.to(dev).eval(); out = []
            with torch.no_grad():
                for i in range(0, len(te), 128):
                    c = torch.from_numpy(C[i:i + 128]).to(dev) if nclin else torch.zeros(min(128, len(te) - i), 1, device=dev)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        out.append(torch.sigmoid(m(torch.from_numpy(X[i:i + 128]).to(dev), c)).float().cpu().numpy().ravel())
            P.append(np.concatenate(out)); m.cpu()
    te[MAIN], te[ECGO] = np.mean(pc, 0), np.mean(pe, 0)
    for f in range(5): te[f"main_fold{f}"], te[f"ecgo_fold{f}"] = pc[f], pe[f]
    te.to_parquet(os.path.join(OUT, "test_predictions_wcr.parquet")); return te, devdf


def main():
    p = os.path.join(OUT, "test_predictions_wcr.parquet")
    if os.path.exists(p):
        te = pd.read_parquet(p); sc = pd.read_parquet(os.path.join(D, "scores_v4.parquet"))
        devdf = pd.concat([sc[sc.split == s].sort_values("row_in_split").reset_index() for s in ["train", "val"]], ignore_index=True)
    else:
        te, devdf = infer()
    oof = pd.read_parquet(os.path.join(D, "stage1b_oof_all.parquet")).reindex(devdf.entity_id.values)
    ydv = oof.y.values.astype(int)
    # calibrated probability: Platt on development OOF
    # per-fold Platt (each fold model calibrated on its own held-out development fold), then averaged; fold scales differ
    plattk = {f: LogisticRegression(C=1e6).fit(lg(g["WCR wcr_clin"].values).reshape(-1, 1), g.y.astype(int)) for f, g in oof.groupby("fold")}
    te[f"{MAIN} (calibrated)"] = np.mean([plattk[f].predict_proba(lg(te[f"main_fold{f}"].values).reshape(-1, 1))[:, 1] for f in range(5)], 0)
    oof["cal"] = 0.0
    for f, g in oof.groupby("fold"): oof.loc[g.index, "cal"] = plattk[f].predict_proba(lg(g["WCR wcr_clin"].values).reshape(-1, 1))[:, 1]
    plattE = {f: LogisticRegression(C=1e6).fit(lg(g["WCR wcr"].values).reshape(-1, 1), g.y.astype(int)) for f, g in oof.groupby("fold")}
    te[f"{ECGO} (calibrated)"] = np.mean([plattE[f].predict_proba(lg(te[f"ecgo_fold{f}"].values).reshape(-1, 1))[:, 1] for f in range(5)], 0)
    oof["cal_ecgo"] = 0.0
    for f, g in oof.groupby("fold"): oof.loc[g.index, "cal_ecgo"] = plattE[f].predict_proba(lg(g["WCR wcr"].values).reshape(-1, 1))[:, 1]
    oof[["y", "fold", "cal", "cal_ecgo"]].to_parquet(os.path.join(OUT, "dev_oof_wcr_calibrated.parquet"))
    for c_ in ["WCR wcr_clin", "WCR wcr", "Retina retina_clin"]: oof[c_ + " rk"] = oof.groupby("fold")[c_].rank(pct=True)  # within-fold ranks for CV AUROC
    te["Fusion: DeepECG-SSL + retina"] = (rk(te[MAIN]) + rk(te["Retina+clin"])) / 2
    scores = {"PREVENT": "prevent_cvd_winsor", "SCORE2/OP": "score2", "PCE": "pce", "FRS": "frs"}
    for k, c in scores.items():
        m = LogisticRegression(C=1e6).fit(lg(devdf[c].values).reshape(-1, 1), devdf.incident_F2_primary.astype(int))
        te[f"{k} (recal)"] = m.predict_proba(lg(te[c].values).reshape(-1, 1))[:, 1]
    y = te.incident_F2_primary.values.astype(int)

    # ---- development CV (OOF) for the same models
    cvrows = []
    for nm, s in [(MAIN, oof["WCR wcr_clin rk"]), (ECGO, oof["WCR wcr rk"]), ("Retina + clinical", oof["Retina retina_clin rk"]),
                  ("Fusion: DeepECG-SSL + retina", (oof["WCR wcr_clin rk"] + oof["Retina retina_clin rk"]) / 2),
                  ("PREVENT", oof.prevent_cvd_winsor), ("SCORE2/OP", oof.score2), ("PCE", oof.pce), ("FRS", oof.frs)]:
        a, lo, hi = auc_ci(ydv, s.values); cvrows.append(dict(model=nm, n=len(ydv), cases=int(ydv.sum()), AUROC=f"{a:.3f} ({lo:.3f}-{hi:.3f})"))
        if nm != "PREVENT":
            d, dl, dh, pp = delong_test(ydv, s.values, oof.prevent_cvd_winsor.values); cvrows[-1]["vs PREVENT"] = f"{d:+.3f} ({dl:+.3f} to {dh:+.3f}), p={pp:.3g}"
    pd.DataFrame(cvrows).to_csv(os.path.join(OUT, "metrics_cv_dev.csv"), index=False)

    def block(g, oc, label):
        yy = g[oc].values.astype(int); main = g[MAIN].values; rows = []
        a, lo, hi = auc_ci(yy, main)
        rows.append(dict(outcome=label, model=MAIN, n=len(yy), cases=int(yy.sum()), AUROC=f"{a:.3f} ({lo:.3f}-{hi:.3f})", AUPRC=f"{average_precision_score(yy, main):.3f}"))
        comps = {**{k: g[c].values for k, c in scores.items()}, ECGO: g[ECGO].values, "Retina + clinical": g["Retina+clin"].values,
                 "Fusion: DeepECG-SSL + retina": g["Fusion: DeepECG-SSL + retina"].values}
        for k, s in comps.items():
            a2, lo2, hi2 = auc_ci(yy, s); d, dl, dh, pp = delong_test(yy, main, s)
            rows.append(dict(outcome=label, model=k, n=len(yy), cases=int(yy.sum()), AUROC=f"{a2:.3f} ({lo2:.3f}-{hi2:.3f})", AUPRC=f"{average_precision_score(yy, s):.3f}",
                             dAUROC_main_minus=f"{d:+.3f} ({dl:+.3f} to {dh:+.3f})", p_delong=pp))
        for r, pa in zip(rows[1:5], holm([r["p_delong"] for r in rows[1:5]])): r["p_holm"] = pa
        return rows
    prim = pd.DataFrame(block(te, "incident_F2_primary", "Incident CVD by F2 (primary)")); prim.to_csv(os.path.join(OUT, "metrics_primary.csv"), index=False)

    calrows = []
    for k in [f"{MAIN} (calibrated)", "PREVENT (recal)", "SCORE2/OP (recal)", "PCE (recal)", "FRS (recal)", "prevent_cvd_winsor", "score2", "pce", "frs"]:
        s, itl, b, mp = cal(y, te[k].values)
        calrows.append(dict(model=k, slope=round(s, 2), intercept_in_large=round(itl, 2), brier=round(b, 4), mean_pred=round(mp, 3), observed=round(y.mean(), 3)))
    pd.DataFrame(calrows).to_csv(os.path.join(OUT, "metrics_calibration.csv"), index=False)
    ths = np.arange(0.02, 0.205, 0.01)
    dca = pd.DataFrame({"threshold": ths, "treat_all": [y.mean() - (1 - y.mean()) * t / (1 - t) for t in ths]})
    for k in [f"{MAIN} (calibrated)", "PREVENT (recal)", "SCORE2/OP (recal)", "PCE (recal)", "FRS (recal)"]:
        dca[k] = [net_benefit(y, te[k].values, t) for t in ths]
    dca.to_csv(os.path.join(OUT, "metrics_dca.csv"), index=False)
    pdv = oof["cal"].values; thr90 = np.quantile(pdv[ydv == 1], 0.10)
    op = []
    for name, t in [("10% predicted risk", 0.10), (f"90% development sensitivity (t={thr90:.3f})", thr90)]:
        for k in [f"{MAIN} (calibrated)", "PREVENT (recal)", "FRS (recal)"]:
            pp = te[k].values >= t; tp = (pp & (y == 1)).sum(); fp = (pp & (y == 0)).sum(); fn = (~pp & (y == 1)).sum(); tn = (~pp & (y == 0)).sum()
            op.append(dict(operating_point=name, model=k, flagged=f"{pp.mean():.1%}", sens=round(tp / (tp + fn), 3), spec=round(tn / (tn + fp), 3),
                           ppv=round(tp / max(tp + fp, 1), 3), npv=round(tn / max(tn + fn, 1), 3)))
    pd.DataFrame(op).to_csv(os.path.join(OUT, "metrics_operating_points.csv"), index=False)
    cats = lambda q: np.digitize(q, [0.05, 0.10]); nri = {}
    for ref in ["PREVENT (recal)", "FRS (recal)", "SCORE2/OP (recal)", "PCE (recal)"]:
        cn, co_ = cats(te[f"{MAIN} (calibrated)"].values), cats(te[ref].values); up, dn = cn > co_, cn < co_
        ne, nn = up[y == 1].mean() - dn[y == 1].mean(), dn[y == 0].mean() - up[y == 0].mean()
        nri[ref] = dict(NRI_events=round(ne, 3), NRI_nonevents=round(nn, 3), NRI_total=round(ne + nn, 3))
    nri["categories"] = "<5%, 5-10%, >=10% (~6-y risk)"; json.dump(nri, open(os.path.join(OUT, "metrics_nri.json"), "w"), indent=1)

    sec = []
    for oc, lab in [("incident_F1", "Incident CVD by F1 (~3 y)"), ("hard_ascvd_F1", "Hard ASCVD by F1"), ("strict_F1", "Incident by F1, strict"),
                    ("noHEARTonly_F1", "Incident by F1 excl. heart-disease-only")]:
        sec += block(te.dropna(subset=[oc]), oc, lab)
    pd.DataFrame(sec).to_csv(os.path.join(OUT, "metrics_secondary.csv"), index=False)
    sub = []; te["age_band"] = pd.cut(te.age, [0, 55, 65, 75, 200], right=False, labels=["45-54", "55-64", "65-74", ">=75"])
    for col, vals in [("female", [0, 1]), ("age_band", ["45-54", "55-64", "65-74", ">=75"]), ("ECG category", ["normal", "borderline", "abnormal"]),
                      ("diabetes", [0, 1]), ("statin", [0, 1]), ("in_range_prevent", [True])]:
        for v in vals:
            g = te[te[col] == v]; yy = g.incident_F2_primary.values.astype(int)
            if yy.sum() < 10: continue
            a, lo, hi = auc_ci(yy, g[MAIN].values); b_, blo, bhi = auc_ci(yy, g.prevent_cvd_winsor.values); f_, flo, fhi = auc_ci(yy, g.frs.values)
            d, dl, dh, pp = delong_test(yy, g[MAIN].values, g.prevent_cvd_winsor.values)
            sub.append(dict(subgroup=f"{col}={v}", n=len(g), cases=int(yy.sum()), DeepECG_SSL=f"{a:.3f} ({lo:.3f}-{hi:.3f})", PREVENT=f"{b_:.3f} ({blo:.3f}-{bhi:.3f})",
                            FRS=f"{f_:.3f} ({flo:.3f}-{fhi:.3f})", delta_vs_PREVENT=f"{d:+.3f} ({dl:+.3f} to {dh:+.3f})"))
    pd.DataFrame(sub).to_csv(os.path.join(OUT, "metrics_subgroups.csv"), index=False)
    te.to_parquet(os.path.join(OUT, "test_predictions_wcr.parquet"))

    # Figure 2: ROC / calibration / DCA
    fig, ax = plt.subplots(1, 3, figsize=(18, 5.5))
    for k, c in [(MAIN, MAIN), ("PREVENT", "prevent_cvd_winsor"), ("SCORE2/OP", "score2"), ("PCE", "pce"), ("FRS", "frs")]:
        f_, t_, _ = roc_curve(y, te[c]); ax[0].plot(f_, t_, label=f"{k} ({roc_auc_score(y, te[c]):.3f})")
    ax[0].plot([0, 1], [0, 1], "k:"); ax[0].set(title="A. ROC, incident CVD (CLSA test)", xlabel="1 - specificity", ylabel="Sensitivity"); ax[0].legend(fontsize=9)
    for k, lab in [(f"{MAIN} (calibrated)", MAIN), ("PREVENT (recal)", "PREVENT (recalibrated)"), ("FRS (recal)", "FRS (recalibrated)")]:
        q = pd.qcut(te[k], 10, duplicates="drop"); g = te.groupby(q, observed=True); ax[1].plot(g[k].mean(), g.incident_F2_primary.mean(), "o-", label=lab)
    ax[1].plot([0, .4], [0, .4], "k:"); ax[1].set(title="B. Calibration (deciles)", xlabel="Predicted risk", ylabel="Observed risk"); ax[1].legend(fontsize=9)
    for k, lab in [(f"{MAIN} (calibrated)", MAIN), ("PREVENT (recal)", "PREVENT"), ("SCORE2/OP (recal)", "SCORE2/OP"), ("PCE (recal)", "PCE"), ("FRS (recal)", "FRS"), ("treat_all", "Treat all")]:
        ax[2].plot(dca.threshold, dca[k], label=lab)
    ax[2].axhline(0, color="k", lw=.5); ax[2].set(ylim=(-0.01, 0.11), title="C. Decision curve", xlabel="Threshold probability", ylabel="Net benefit"); ax[2].legend(fontsize=9)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig2_roc_calibration_dca.png"), dpi=200); plt.close()
    pd.set_option("display.width", 250)
    print("== CV (development)"); print(pd.DataFrame(cvrows).fillna("").to_string(index=False))
    print("\n== PRIMARY (test)"); print(prim.drop(columns=["p_delong"]).fillna("").to_string(index=False))
    print("\n== CALIBRATION"); print(pd.DataFrame(calrows).to_string(index=False))
    print("\n== NRI", json.dumps(nri)); print("\n== OPERATING POINTS"); print(pd.DataFrame(op).to_string(index=False))
    print("\n== DCA"); print(dca[dca.threshold.round(2).isin([0.05, 0.075, 0.1, 0.15, 0.2])].round(4).to_string(index=False))
    print("\n== SECONDARY"); print(pd.DataFrame(sec)[["outcome", "model", "n", "cases", "AUROC", "dAUROC_main_minus"]].fillna("").to_string(index=False))
    print("\n== SUBGROUPS"); print(pd.DataFrame(sub).to_string(index=False))


if __name__ == "__main__":
    main()
