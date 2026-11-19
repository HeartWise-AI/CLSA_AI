"""Evaluate frozen CLSA AI-ECG models on the MHI any-ECG cohort (6-y MedEcho hard MACE; see mhi_anyecg_build.py).
Also: label validity from the blinded adjudication sample (PPV of rule positives, NPV of rule negatives)."""
import os, json, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
D, O = "data_v4", "mhi_anyecg"
lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
c = pd.read_parquet(f"{O}/cohort_labels.parquet").merge(pd.read_parquet(f"{O}/predictions.parquet"), on="MRN")
c = c[c.ecg_ok].reset_index(drop=True)
sc = pd.read_parquet(f"{D}/scores_v4.parquet")
dv = pd.concat([sc[sc.split == s].sort_values("row_in_split").reset_index() for s in ["train", "val"]], ignore_index=True)
o = pd.read_parquet(f"{D}/stage1b_oof_all.parquet").reindex(dv.entity_id.values)
dv["vit"], dv["wcr"] = lg(o["ViT ecg"].values), lg(o["WCR wcr"].values); yd = dv.incident_F2_primary.values.astype(int)
MODELS = {"Age + sex": ["age", "female"], "AI-ECG (ViT) alone": "vit", "AI-ECG (WCRv2) alone": "wcr",
          "AI-ECG (ViT) + age + sex": ["vit", "age", "female"], "AI-ECG (WCRv2) + age + sex": ["wcr", "age", "female"],
          "AI-ECG (ViT+WCRv2) + age + sex": ["vit", "wcr", "age", "female"]}
FIT = {k: LogisticRegression(max_iter=3000).fit(dv[v], yd) for k, v in MODELS.items() if isinstance(v, list)}
def evaluate(name, g, y, nb=1000):
    g = g.reset_index(drop=True); y = np.asarray(y).astype(int)
    P = {k: (g[v].values if isinstance(v, str) else FIT[k].predict_proba(g[v])[:, 1]) for k, v in MODELS.items()}
    rng = np.random.default_rng(2026); B = [rng.integers(0, len(y), len(y)) for _ in range(nb)]
    bt = {k: np.array([roc_auc_score(y[b], p[b]) for b in B]) for k, p in P.items()}; rows = []
    for k, p in P.items():
        r = dict(analysis=name, model=k, n=len(y), events=int(y.sum()), AUROC=f"{roc_auc_score(y, p):.3f} ({np.percentile(bt[k], 2.5):.3f}-{np.percentile(bt[k], 97.5):.3f})")
        if k != "Age + sex":
            dd = bt[k] - bt["Age + sex"]; r["vs age+sex"] = f"{roc_auc_score(y, p) - roc_auc_score(y, P['Age + sex']):+.3f} ({np.percentile(dd, 2.5):+.3f} to {np.percentile(dd, 97.5):+.3f})"
        rows.append(r)
    return rows
inc = c[c.prev == "no"]; keep = lambda g, y: g[(y == 1) | g.fu_ecg_6y]
base = keep(inc[~inc.unk_death], inc[~inc.unk_death].y)
R = evaluate("1. PRIMARY: 6-y hard MACE (MI/stroke/HF-principal/CV death), non-events with ECG >= 6 y", base, base.y)
b2 = keep(inc[~inc.unk_death], inc[~inc.unk_death].y_broad); R += evaluate("2. Broad MACE (+revasc >90 d, UA)", b2, b2.y_broad)
s0 = inc[~inc.unk_death]; R += evaluate("S0. no ECG follow-up requirement", s0, s0.y)
s1a = c[c.prev.isin(["no", "unc"]) & ~c.unk_death]; s1 = keep(s1a, s1a.y); R += evaluate("S1. incl. prevalent-uncertain", s1, s1.y)
s1b = base[~base.unc_any]; R += evaluate("S1b. excl. any undated prior-disease code (adjudicator definition)", s1b, s1b.y)
s2a = inc; s2 = keep(s2a, s2a.y | s2a.unk_death); R += evaluate("S2. unknown-cause deaths = events", s2, s2.y | s2.unk_death)
s3 = base[base.age.between(45, 85)]; R += evaluate("S3. age 45-85 (CLSA range)", s3, s3.y)
for t in ["MI", "HF", "stroke"]:
    g = base[(base.y == 0) | (base.event_type == t)]; R += evaluate(f"C. component: {t}", g, g.event_type == t, nb=300)
res = pd.DataFrame(R); res.to_csv("results_v4/mhi_anyecg_results.csv", index=False)
summ = dict(primary_n=len(base), events=int(base.y.sum()), event_types=base[base.y == 1].event_type.value_counts().to_dict(),
            age=f"{base.age.mean():.1f}±{base.age.std():.1f}", female=round(base.female.mean(), 3))
if os.path.exists(f"{O}/adjudication/adjudicated_all.jsonl"):
    k = pd.read_parquet(f"{O}/sample_key.parquet"); a = pd.read_json(f"{O}/adjudication/adjudicated_all.jsonl", lines=True).astype(str)
    m = k.merge(a, left_on="sid", right_on="study_id")
    pos, neg = m[m.y == 1], m[m.y == 0]
    summ["adjudication"] = dict(n=len(m), rule_pos=len(pos), PPV_event=round((pos.primary_event == "1").mean(), 3), pos_adj_uncertain=int((pos.primary_event == "uncertain").sum()),
                                pos_prevalent_by_adjudicator=int((pos.prevalent_cvd == "yes").sum()), rule_neg=len(neg), NPV=round((neg.primary_event == "0").mean(), 3),
                                neg_prevalent_by_adjudicator=int((neg.prevalent_cvd == "yes").sum()),
                                PPV_incident_event=round(((pos.primary_event == "1") & (pos.prevalent_cvd == "no")).mean(), 3))
json.dump(summ, open("results_v4/mhi_anyecg_summary.json", "w"), indent=1, default=str); print(json.dumps(summ, indent=1, default=str))
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 70)
for an, g in res.groupby("analysis", sort=False): print("\n==", an, f"(n={g.n.iloc[0]}, events={g.events.iloc[0]})"); print(g[["model", "AUROC", "vs age+sex"]].fillna("").to_string(index=False))
