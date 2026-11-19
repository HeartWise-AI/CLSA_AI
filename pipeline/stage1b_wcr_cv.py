"""Stage 1b: fine-tune the DeepECG-SSL v2 WCR encoder (amp-preserved, SSL-only) for incident CVD by F2.
Input: data_v4/ecg_wcr_{split}.npy = raw CLSA x0.005 (mV), 250 Hz, 12x2500 (full 10 s), NO z-score.
Encoder loaded with the public fairseq-signals code (tools/fairseq-signals; venv tools/venv_fss).
Same 5-fold CV / inner split as Stage 1 (StratifiedKFold(5, rs=42)).
Variants: wcr (ECG only) | wcr_clin (ECG + PREVENT inputs).
Usage: tools/venv_fss/bin/python stage1b_wcr_cv.py --fold 0 --variant wcr_clin --gpu 0
"""
import os, sys, math, argparse, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__)); D = os.path.join(HERE, "data_v4")
sys.path.insert(0, os.path.join(HERE, "tools", "fairseq-signals"))
from fairseq_signals.utils import checkpoint_utils
WCR = "/media/data1/models/DeepECG-SSL/wcr-v2/deepecg-ssl-v2-wcr-amp-preserved-encoder.pt"
FT77 = "/media/data1/models/DeepECG-SSL/wcr-v2/deepecg-ssl-v2-amp-preserved-ft77-best.pt"  # Notion: DeepECG-SSL v2 (Amp-Preserved) 77-label


def load_ft77_encoder(enc):
    """Overwrite the SSL encoder with the 77-label fine-tuned encoder weights (all 207 tensors).
    Old weight-norm names (weight_g/weight_v) -> torch parametrizations (original0/original1)."""
    ft = torch.load(FT77, map_location="cpu", weights_only=False)["model"]
    sd = {k[len("encoder."):]: v for k, v in ft.items() if k.startswith("encoder.")}
    ren = {"conv_pos.pos_conv.0.weight_g": "conv_pos.pos_conv.0.parametrizations.weight.original0",
           "conv_pos.pos_conv.0.weight_v": "conv_pos.pos_conv.0.parametrizations.weight.original1"}
    sd = {ren.get(k, k): v for k, v in sd.items()}
    own = enc.state_dict(); ok = {k: v for k, v in sd.items() if k in own and own[k].shape == v.shape}
    assert len(ok) == len(sd) == 207, (len(ok), len(sd))
    enc.load_state_dict(ok, strict=False); return len(ok)
CLIN = ["age", "female", "sbp", "bp_treated", "tc_mmol", "hdl_mmol", "diabetes", "smoker_current", "egfr", "bmi", "statin"]


class Net(nn.Module):
    def __init__(self, n_clin, init="ssl"):
        super().__init__()
        models, _, _ = checkpoint_utils.load_model_and_task(WCR)
        self.enc = models[0] if isinstance(models, list) else models
        self.n_init = load_ft77_encoder(self.enc) if init == "ft77" else 0
        for n in ["quantizer", "project_q", "final_proj", "target_glu"]:  # SSL-only heads, unused
            if hasattr(self.enc, n): setattr(self.enc, n, None)
        self.n_clin = n_clin
        self.clin = nn.Sequential(nn.Linear(n_clin, 64), nn.GELU(), nn.Dropout(0.1)) if n_clin else None
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(768 + (64 if n_clin else 0), 1))

    def forward(self, x, c=None):
        z = self.enc(source=x, mask=False, features_only=True)["x"].mean(1)
        if self.n_clin: z = torch.cat([z, self.clin(c)], 1)
        return self.head(z).squeeze(1)


def load_dev():
    sc = pd.read_parquet(os.path.join(D, "scores_v4.parquet")); parts, X = [], []
    for sp in ["train", "val"]:
        g = sc[sc.split == sp].sort_values("row_in_split"); parts.append(g.reset_index())
        X.append(np.load(os.path.join(D, f"ecg_wcr_{sp}.npy")))
    d = pd.concat(parts, ignore_index=True); X = np.concatenate(X); assert len(X) == len(d); return d, X


def augment(x):  # amplitude-preserving augmentations only (no z-score)
    if np.random.rand() < 0.5: x = x * np.random.uniform(0.9, 1.1)
    if np.random.rand() < 0.5: x = x + np.random.normal(0, 0.01, x.shape).astype(np.float32)
    if np.random.rand() < 0.2: x[np.random.choice(12, np.random.randint(1, 3), replace=False)] = 0
    if np.random.rand() < 0.5: x = np.roll(x, np.random.randint(-250, 250), axis=1)
    return x


@torch.no_grad()
def predict(model, idx, X, C, dev):
    model.eval(); out = []
    for i in range(0, len(idx), 128):
        b = idx[i:i + 128]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(torch.sigmoid(model(torch.from_numpy(X[b]).to(dev), torch.from_numpy(C[b]).to(dev))).float().cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--fold", type=int); ap.add_argument("--variant", default="wcr_clin")
    ap.add_argument("--gpu", default="0"); ap.add_argument("--epochs", type=int, default=12); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=2e-5); ap.add_argument("--init", default="ssl", choices=["ssl", "ft77"])
    a = ap.parse_args(); torch.manual_seed(a.seed); np.random.seed(a.seed); dev = torch.device(f"cuda:{a.gpu}")
    d, X = load_dev(); y = d.incident_F2_primary.values.astype(np.float32)
    tr_all, te = list(StratifiedKFold(5, shuffle=True, random_state=42).split(d, y.astype(int)))[a.fold]
    tr, va = train_test_split(tr_all, test_size=0.1, stratify=y[tr_all].astype(int), random_state=a.seed)
    cols = CLIN if a.variant == "wcr_clin" else []
    C = StandardScaler().fit(d.loc[tr, cols]).transform(d[cols]).astype(np.float32) if cols else np.zeros((len(d), 1), np.float32)
    model = Net(len(cols), a.init).to(dev)
    enc_p = [p for p in model.enc.parameters() if p.requires_grad]; other = [p for n, p in model.named_parameters() if not n.startswith("enc.")]
    opt = torch.optim.AdamW([{"params": enc_p, "lr": a.lr}, {"params": other, "lr": 1e-3}], weight_decay=0.01)
    bs = 32; steps = a.epochs * math.ceil(len(tr) / bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[a.lr, 1e-3], total_steps=steps, pct_start=0.1)
    pw = torch.tensor(math.sqrt((1 - y[tr].mean()) / y[tr].mean()), device=dev)
    best, best_state, bad = -1, None, 0
    for ep in range(a.epochs):
        model.train(); perm = np.random.permutation(tr)
        for i in range(0, len(perm), bs):
            b = perm[i:i + bs]
            xb = torch.from_numpy(np.stack([augment(X[j].copy()) for j in b])).to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = F.binary_cross_entropy_with_logits(model(xb, torch.from_numpy(C[b]).to(dev)).float(), torch.from_numpy(y[b]).to(dev), pos_weight=pw)
            opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
        auc = roc_auc_score(y[va], predict(model, va, X, C, dev))
        print(f"fold {a.fold} {a.variant} ep {ep} inner-val AUROC {auc:.4f}", flush=True)
        if auc > best: best, bad, best_state = auc, 0, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= 4: break
    model.load_state_dict(best_state); p = predict(model, te, X, C, dev)
    od = os.path.join(D, "stage1b_wcr" if a.init == "ssl" else "stage1b_wcr_ft77"); os.makedirs(od, exist_ok=True)
    pd.DataFrame({"entity_id": d.entity_id.values[te], "y": y[te], "pred": p}).to_parquet(os.path.join(od, f"oof_{a.variant}_fold{a.fold}_seed{a.seed}.parquet"))
    torch.save(best_state, os.path.join(od, f"model_{a.variant}_fold{a.fold}_seed{a.seed}.pt"))
    print(f"DONE fold {a.fold} {a.variant} init={a.init} (ft77 tensors {model.n_init}): inner best {best:.4f}, outer-fold AUROC {roc_auc_score(y[te], p):.4f}", flush=True)


if __name__ == "__main__":
    main()
