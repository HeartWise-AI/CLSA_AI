"""FINAL locked-test evaluation (plan 7.3). Run ONCE. Model spec frozen in REANALYSIS_PLAN.md 7.10.
Outputs -> results_v4/: test_predictions.parquet, metrics_*.csv, figures (ROC, calibration, DCA).
"""
import os, sys, glob, json, numpy as np, pandas as pd, torch
from scipy.stats import rankdata, norm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, roc_curve
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); D = os.path.join(HERE, "data_v4"); OUT = os.path.join(HERE, "results_v4")
os.makedirs(OUT, exist_ok=True); sys.path.insert(0, HERE)
from stage1_vit_cv import Net as VitNet, CLIN
from stage2_retina_cv import Net as RetNet, to_tensor
from stratify_by_deepecg import classify
dev = torch.device("cuda:0")
lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
rk = lambda s: rankdata(s) / len(s)


# ---------------- DeLong (Sun & Xu 2014) ----------------
def _midrank(x):
    J = np.argsort(x); Z = x[J]; N = len(x); T = np.zeros(N); i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]: j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1; i = j
    T2 = np.empty(N); T2[J] = T; return T2


def delong(y, preds):
    """preds: (k, n). Returns aucs (k,), covariance (k,k)."""
    pos, neg = preds[:, y == 1], preds[:, y == 0]; m, n = pos.shape[1], neg.shape[1]; k = preds.shape[0]
    tx = np.array([_midrank(p) for p in pos]); ty = np.array([_midrank(q) for q in neg])
    tz = np.array([_midrank(np.concatenate([p, q])) for p, q in zip(pos, neg)])
    aucs = tz[:, :m].sum(1) / m / n - (m + 1) / 2 / n
    v01 = (tz[:, :m] - tx) / n; v10 = 1 - (tz[:, m:] - ty) / m
    S = np.cov(v01) / m + np.cov(v10) / n
    return aucs, np.atleast_2d(S)


def delong_test(y, a, b):
    aucs, S = delong(y, np.vstack([a, b])); d = aucs[0] - aucs[1]; se = np.sqrt(S[0, 0] + S[1, 1] - 2 * S[0, 1])
    return d, d - 1.96 * se, d + 1.96 * se, 2 * norm.sf(abs(d / se))


def auc_ci(y, s):
    a, S = delong(y, s[None]); se = np.sqrt(S[0, 0]); return a[0], a[0] - 1.96 * se, a[0] + 1.96 * se


def holm(p):
    p = np.asarray(p); o = np.argsort(p); adj = np.empty_like(p); run = 0
    for i, j in enumerate(o): run = max(run, (len(p) - i) * p[j]); adj[j] = min(1, run)
    return adj


def cal(y, p):
    x = lg(p); m = LogisticRegression(C=1e6).fit(x.reshape(-1, 1), y); slope = m.coef_[0, 0]
    # calibration-in-the-large: intercept with logit(p) as offset
    import statsmodels.api as sm
    itl = sm.GLM(y, np.ones((len(y), 1)), family=sm.families.Binomial(), offset=x).fit().params[0]
    return slope, itl, brier_score_loss(y, p), p.mean()


def net_benefit(y, p, t):
    n = len(y); tp = ((p >= t) & (y == 1)).sum(); fp = ((p >= t) & (y == 0)).sum(); return tp / n - fp / n * t / (1 - t)


# ---------------- data ----------------
def main():
    sc = pd.read_parquet(os.path.join(D, "scores_v4.parquet"))
    te = sc[sc.split == "test"].sort_values("row_in_split").reset_index()
    devdf = pd.concat([sc[sc.split == s].sort_values("row_in_split").reset_index() for s in ["train", "val"]], ignore_index=True)
    oof = pd.read_parquet(os.path.join(D, "stage2_oof_all.parquet"))

    # ViT: 5 folds x 2 seeds, two-half TTA
    Xv = np.load(os.path.join(D, "ecg_vit_test.npy"))
    vit = {}
    for seed in [0, 1]:
        ps = []
        for f in range(5):
            sd = torch.load(os.path.join(D, "stage1_vit", f"model_ecg_clin_fold{f}_seed{seed}.pt"), map_location="cpu")
            from sklearn.model_selection import StratifiedKFold, train_test_split
            y_dev = devdf.incident_F2_primary.values.astype(int)
            tr_all, _ = list(StratifiedKFold(5, shuffle=True, random_state=42).split(devdf, y_dev))[f]
            tr, _ = train_test_split(tr_all, test_size=0.1, stratify=y_dev[tr_all], random_state=seed)
            scl = StandardScaler().fit(devdf.loc[tr, CLIN]); C = torch.from_numpy(scl.transform(te[CLIN]).astype(np.float32))
            m = VitNet(len(CLIN)); m.load_state_dict(sd); m.to(dev).eval(); out = []
            with torch.no_grad():
                for i in range(0, len(te), 256):
                    x = torch.from_numpy(Xv[i:i + 256].astype(np.float32)); c = C[i:i + 256].to(dev)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        p = (torch.sigmoid(m(x[:, :, :2500].to(dev), c)) + torch.sigmoid(m(x[:, :, 2500:].to(dev), c))) / 2
                    out.append(p.float().cpu().numpy())
            ps.append(np.concatenate(out))
        vit[seed] = np.mean(ps, 0)
    # DeepECG-77 + clinical logistic refit on all dev
    E_dev = np.concatenate([np.load(os.path.join(D, f"deepecg77_{s}.npy")) for s in ["train", "val"]])
    E_te = np.load(os.path.join(D, "deepecg77_test.npy"))
    Fd = np.hstack([lg(E_dev), devdf[CLIN].values]); Ft = np.hstack([lg(E_te), te[CLIN].values])
    e77 = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=3000)).fit(Fd, devdf.incident_F2_primary.astype(int))
    te["ViT s0"], te["ViT s1"], te["ECG77+clin"] = vit[0], vit[1], e77.predict_proba(Ft)[:, 1]
    # Final ensemble (rank, as pre-specified) + calibrated probability (Platt on dev OOF prob-average)
    te["AI-ECG (final)"] = (rk((rk(te["ViT s0"]) + rk(te["ViT s1"])) / 2) + rk(te["ECG77+clin"])) / 2
    pav_dev = ((oof["ViT ecg_clin"] + oof["ViT ecg_clin s1"]) / 2 + oof["ECG77+clinical (logistic)"]) / 2
    platt = LogisticRegression(C=1e6).fit(lg(pav_dev.values).reshape(-1, 1), oof.y.astype(int))
    pav_te = ((te["ViT s0"] + te["ViT s1"]) / 2 + te["ECG77+clin"]) / 2
    te["AI-ECG (final, calibrated prob)"] = platt.predict_proba(lg(pav_te.values).reshape(-1, 1))[:, 1]
    # Retina + clinical (secondary): 5 fold models
    R = np.load(os.path.join(D, "retina_test.npy"), mmap_mode="r"); RM = np.load(os.path.join(D, "retina_mask_test.npy"))
    rp = []
    for f in range(5):
        y_dev = devdf.incident_F2_primary.values.astype(int)
        from sklearn.model_selection import StratifiedKFold, train_test_split
        tr_all, _ = list(StratifiedKFold(5, shuffle=True, random_state=42).split(devdf, y_dev))[f]
        tr, _ = train_test_split(tr_all, test_size=0.1, stratify=y_dev[tr_all], random_state=0)
        scl = StandardScaler().fit(devdf.loc[tr, CLIN]); C = scl.transform(te[CLIN]).astype(np.float32)
        m = RetNet(len(CLIN)); m.load_state_dict(torch.load(os.path.join(D, "stage2_retina", f"model_retina_clin_fold{f}_seed0.pt"), map_location="cpu"))
        m.to(dev).eval(); out = []
        with torch.no_grad():
            for i in range(0, len(te), 64):
                im = torch.stack([torch.stack([to_tensor(np.asarray(R[j][e]), False) for e in range(2)]) for j in range(i, min(i + 64, len(te)))])
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out.append(torch.sigmoid(m(im.to(dev), torch.from_numpy(RM[i:i + 64]).to(dev), torch.from_numpy(C[i:i + 64]).to(dev))).float().cpu().numpy())
        rp.append(np.concatenate(out))
    te["Retina+clin"] = np.mean(rp, 0)
    te["ECG category"] = classify(E_te)[0]
    te.to_parquet(os.path.join(OUT, "test_predictions.parquet"))

    # Recalibrated scores (intercept+slope on dev) for calibration/DCA/NRI
    scores = {"PREVENT": "prevent_cvd_winsor", "SCORE2/OP": "score2", "PCE": "pce", "FRS": "frs"}
    for k, c in scores.items():
        m = LogisticRegression(C=1e6).fit(lg(devdf[c].values).reshape(-1, 1), devdf.incident_F2_primary.astype(int))
        te[f"{k} (recal)"] = m.predict_proba(lg(te[c].values).reshape(-1, 1))[:, 1]

    # ---------------- primary metrics ----------------
    def metrics_block(g, oc, label):
        y = g[oc].values.astype(int); rows = []
        main = g["AI-ECG (final)"].values
        comps = {**{k: g[c].values for k, c in scores.items()}, "Retina+clin": g["Retina+clin"].values,
                 "ViT ECG+clin (2 seeds)": ((g["ViT s0"] + g["ViT s1"]) / 2).values, "DeepECG77+clin": g["ECG77+clin"].values}
        a, lo, hi = auc_ci(y, main)
        rows.append(dict(outcome=label, model="AI-ECG (final)", n=len(y), cases=int(y.sum()), AUROC=f"{a:.3f} ({lo:.3f}-{hi:.3f})",
                         AUPRC=f"{average_precision_score(y, main):.3f}"))
        ps = []
        for k, s in comps.items():
            a2, lo2, hi2 = auc_ci(y, s); d, dl, dh, p = delong_test(y, main, s); ps.append(p)
            rows.append(dict(outcome=label, model=k, n=len(y), cases=int(y.sum()), AUROC=f"{a2:.3f} ({lo2:.3f}-{hi2:.3f})",
                             AUPRC=f"{average_precision_score(y, s):.3f}", dAUROC_AIECG_minus=f"{d:+.3f} ({dl:+.3f} to {dh:+.3f})", p_delong=p))
        adj = holm([r["p_delong"] for r in rows[1:5]])
        for r, pa in zip(rows[1:5], adj): r["p_holm"] = pa
        return rows

    y = te.incident_F2_primary.values.astype(int)
    prim = pd.DataFrame(metrics_block(te, "incident_F2_primary", "Incident CVD by F2 (primary)"))
    prim.to_csv(os.path.join(OUT, "metrics_primary.csv"), index=False)

    # calibration
    calrows = []
    for k in ["AI-ECG (final, calibrated prob)", "PREVENT (recal)", "SCORE2/OP (recal)", "PCE (recal)", "FRS (recal)",
              "prevent_cvd_winsor", "score2", "pce", "frs"]:
        s, itl, b, mp = cal(y, te[k].values)
        calrows.append(dict(model=k, slope=round(s, 2), intercept_in_large=round(itl, 2), brier=round(b, 4), mean_pred=round(mp, 3), observed=round(y.mean(), 3)))
    pd.DataFrame(calrows).to_csv(os.path.join(OUT, "metrics_calibration.csv"), index=False)

    # DCA + operating points + NRI vs PREVENT (recal)
    ths = np.arange(0.02, 0.205, 0.01)
    dca = pd.DataFrame({"threshold": ths, "treat_all": [y.mean() - (1 - y.mean()) * t / (1 - t) for t in ths]})
    for k in ["AI-ECG (final, calibrated prob)", "PREVENT (recal)", "SCORE2/OP (recal)", "PCE (recal)", "FRS (recal)"]:
        dca[k] = [net_benefit(y, te[k].values, t) for t in ths]
    dca.to_csv(os.path.join(OUT, "metrics_dca.csv"), index=False)
    op = []
    pdv = platt.predict_proba(lg(pav_dev.values).reshape(-1, 1))[:, 1]; ydv = oof.y.values.astype(int)
    thr90 = np.quantile(pdv[ydv == 1], 0.10)  # dev threshold giving 90% sensitivity
    for name, t in [("10% predicted risk", 0.10), (f"90% dev sensitivity (t={thr90:.3f})", thr90)]:
        for k in ["AI-ECG (final, calibrated prob)", "PREVENT (recal)"]:
            p = te[k].values >= t; tp = (p & (y == 1)).sum(); fp = (p & (y == 0)).sum(); fn = (~p & (y == 1)).sum(); tn = (~p & (y == 0)).sum()
            op.append(dict(operating_point=name, model=k, flagged=f"{p.mean():.1%}", sens=round(tp / (tp + fn), 3), spec=round(tn / (tn + fp), 3),
                           ppv=round(tp / max(tp + fp, 1), 3), npv=round(tn / max(tn + fn, 1), 3)))
    pd.DataFrame(op).to_csv(os.path.join(OUT, "metrics_operating_points.csv"), index=False)
    cats = lambda p: np.digitize(p, [0.05, 0.10])
    cn, co_ = cats(te["AI-ECG (final, calibrated prob)"].values), cats(te["PREVENT (recal)"].values)
    up, dn = cn > co_, cn < co_
    nri_e = up[y == 1].mean() - dn[y == 1].mean(); nri_ne = dn[y == 0].mean() - up[y == 0].mean()
    nri = dict(NRI_events=round(nri_e, 3), NRI_nonevents=round(nri_ne, 3), NRI_total=round(nri_e + nri_ne, 3), categories="<5%, 5-10%, >=10%")
    json.dump(nri, open(os.path.join(OUT, "metrics_nri.json"), "w"), indent=1)

    # secondary outcomes + subgroups
    sec = []
    for oc, lab in [("incident_F1", "Incident CVD by F1 (~3 y)"), ("hard_ascvd_F1", "Hard ASCVD by F1"),
                    ("strict_F1", "Incident by F1, strict"), ("noHEARTonly_F1", "Incident by F1 excl. heart-disease-only")]:
        g = te.dropna(subset=[oc]); sec += metrics_block(g, oc, lab)
    sub = []
    te["age_band"] = pd.cut(te.age, [0, 55, 65, 75, 200], right=False, labels=["45-54", "55-64", "65-74", ">=75"])
    for col, vals in [("female", [0, 1]), ("age_band", ["45-54", "55-64", "65-74", ">=75"]), ("ECG category", ["normal", "borderline", "abnormal"]),
                      ("diabetes", [0, 1]), ("statin", [0, 1]), ("in_range_prevent", [True])]:
        for v in vals:
            g = te[te[col] == v]
            if g.incident_F2_primary.sum() < 10: continue
            yy = g.incident_F2_primary.values.astype(int)
            a, lo, hi = auc_ci(yy, g["AI-ECG (final)"].values); b_, blo, bhi = auc_ci(yy, g.prevent_cvd_winsor.values)
            d, dl, dh, p = delong_test(yy, g["AI-ECG (final)"].values, g.prevent_cvd_winsor.values)
            sub.append(dict(subgroup=f"{col}={v}", n=len(g), cases=int(yy.sum()), AIECG=f"{a:.3f} ({lo:.3f}-{hi:.3f})",
                            PREVENT=f"{b_:.3f} ({blo:.3f}-{bhi:.3f})", delta=f"{d:+.3f} ({dl:+.3f} to {dh:+.3f})"))
    pd.DataFrame(sec).to_csv(os.path.join(OUT, "metrics_secondary.csv"), index=False)
    pd.DataFrame(sub).to_csv(os.path.join(OUT, "metrics_subgroups.csv"), index=False)

    # figures
    fig, ax = plt.subplots(1, 3, figsize=(18, 5.5))
    for k, c in [("AI-ECG (final)", "AI-ECG (final)"), ("PREVENT", "prevent_cvd_winsor"), ("SCORE2/OP", "score2"), ("PCE", "pce"), ("FRS", "frs")]:
        f_, t_, _ = roc_curve(y, te[c]); ax[0].plot(f_, t_, label=f"{k} {roc_auc_score(y, te[c]):.3f}")
    ax[0].plot([0, 1], [0, 1], "k:"); ax[0].set(title="ROC, incident CVD by F2 (test)", xlabel="1 - specificity", ylabel="Sensitivity"); ax[0].legend()
    for k in ["AI-ECG (final, calibrated prob)", "PREVENT (recal)", "prevent_cvd_winsor"]:
        q = pd.qcut(te[k], 10, duplicates="drop"); g = te.groupby(q, observed=True)
        ax[1].plot(g[k].mean(), g.incident_F2_primary.mean(), "o-", label=k)
    ax[1].plot([0, .4], [0, .4], "k:"); ax[1].set(title="Calibration (deciles)", xlabel="Predicted", ylabel="Observed"); ax[1].legend(fontsize=8)
    for k in ["AI-ECG (final, calibrated prob)", "PREVENT (recal)", "SCORE2/OP (recal)", "PCE (recal)", "FRS (recal)", "treat_all"]:
        ax[2].plot(dca.threshold, dca[k], label=k)
    ax[2].axhline(0, color="k", lw=.5); ax[2].set(ylim=(-0.01, 0.11), title="Decision curve", xlabel="Threshold probability", ylabel="Net benefit"); ax[2].legend(fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig_roc_calibration_dca.png"), dpi=150)

    pd.set_option("display.width", 250)
    print("== PRIMARY"); print(prim.drop(columns=["p_delong"]).to_string(index=False))
    print("\n== CALIBRATION"); print(pd.DataFrame(calrows).to_string(index=False))
    print("\n== OPERATING POINTS"); print(pd.DataFrame(op).to_string(index=False))
    print("\n== NRI vs PREVENT (recal):", nri)
    print("\n== DCA (selected thresholds)"); print(dca[dca.threshold.round(2).isin([0.05, 0.075, 0.1, 0.15, 0.2])].round(4).to_string(index=False))
    print("\n== SECONDARY"); print(pd.DataFrame(sec)[["outcome", "model", "n", "cases", "AUROC", "dAUROC_AIECG_minus"]].fillna("").to_string(index=False))
    print("\n== SUBGROUPS"); print(pd.DataFrame(sub).to_string(index=False))


if __name__ == "__main__":
    main()
