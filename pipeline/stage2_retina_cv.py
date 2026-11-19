"""Stage 2a: retrain the retinal model (new Model B) on v4 BASELINE fundus photos for incident CVD by F2.
Backbone: ImageNet ConvNeXt-Tiny shared across eyes; per-participant feature = mean over available eyes.
Variants: retina = images only; retina_clin = images + PREVENT inputs.
Same 5-fold CV on train+val as Stage 1 (StratifiedKFold(5, rs=42)); inner 10% early stopping.
Preprocessing (Methods): dark-border crop + resize 256 (cache) -> CLAHE on L channel -> RandomResizedCrop(224, 0.85-1)
+ hflip + small rotation (train) / CenterCrop 224 (eval); ImageNet normalisation.
Usage: python stage2_retina_cv.py --fold 0 --variant retina_clin --gpu 0
"""
import os, math, argparse, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F, cv2, torchvision
from torchvision.transforms import v2 as T
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__)); D = os.path.join(HERE, "data_v4")
CLIN = ["age", "female", "sbp", "bp_treated", "tc_mmol", "hdl_mmol", "diabetes", "smoker_current", "egfr", "bmi", "statin"]
MEAN, STD = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1), torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB); lab[..., 0] = _clahe.apply(lab[..., 0]); return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


TR = T.Compose([T.RandomResizedCrop(224, scale=(0.85, 1.0), antialias=True), T.RandomHorizontalFlip(), T.RandomRotation(10)])
EV = T.CenterCrop(224)


def to_tensor(img, train):
    x = torch.from_numpy(clahe(img)).permute(2, 0, 1).float() / 255
    x = TR(x) if train else EV(x)
    return (x - MEAN) / STD


class Net(nn.Module):
    def __init__(self, n_clin):
        super().__init__()
        m = torchvision.models.convnext_tiny(weights="IMAGENET1K_V1"); m.classifier[2] = nn.Identity(); self.bb = m
        self.n_clin = n_clin
        self.clin = nn.Sequential(nn.Linear(n_clin, 64), nn.GELU(), nn.Dropout(0.1)) if n_clin else None
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(768 + (64 if n_clin else 0), 1))

    def forward(self, imgs, mask, c):  # imgs (B,2,3,224,224), mask (B,2)
        B = imgs.shape[0]; f = self.bb(imgs.flatten(0, 1)).view(B, 2, -1)
        w = mask.float().unsqueeze(-1); z = (f * w).sum(1) / w.sum(1).clamp(min=1)
        if self.n_clin: z = torch.cat([z, self.clin(c)], 1)
        return self.head(z).squeeze(1)


def load_dev():
    sc = pd.read_parquet(os.path.join(D, "scores_v4.parquet")); parts, I, M = [], [], []
    for sp in ["train", "val"]:
        g = sc[sc.split == sp].sort_values("row_in_split"); parts.append(g.reset_index())
        I.append(np.load(os.path.join(D, f"retina_{sp}.npy"), mmap_mode="r")); M.append(np.load(os.path.join(D, f"retina_mask_{sp}.npy")))
    d = pd.concat(parts, ignore_index=True); offs = np.cumsum([0] + [len(x) for x in I])
    return d, I, offs, np.concatenate(M)


def get_img(I, offs, j):
    k = np.searchsorted(offs, j, side="right") - 1; return I[k][j - offs[k]]


def batch(idx, I, offs, M, C, y, train):
    imgs = torch.stack([torch.stack([to_tensor(np.asarray(get_img(I, offs, j)[e]), train) for e in range(2)]) for j in idx])
    return imgs, torch.from_numpy(M[idx]), torch.from_numpy(C[idx]), torch.from_numpy(y[idx])


@torch.no_grad()
def predict(model, idx, I, offs, M, C, y, dev):
    model.eval(); out = []
    for i in range(0, len(idx), 64):
        im, m, c, _ = batch(idx[i:i + 64], I, offs, M, C, y, False)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(torch.sigmoid(model(im.to(dev), m.to(dev), c.to(dev))).float().cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--fold", type=int); ap.add_argument("--variant", default="retina_clin")
    ap.add_argument("--gpu", default="0"); ap.add_argument("--epochs", type=int, default=12); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); torch.manual_seed(a.seed); np.random.seed(a.seed); dev = torch.device(f"cuda:{a.gpu}")
    torch.set_num_threads(8)
    d, I, offs, M = load_dev(); y = d.incident_F2_primary.values.astype(np.float32)
    keep = M.any(1)  # participants with >=1 eye; others get no retina prediction
    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(d, y.astype(int)))
    tr_all, te = folds[a.fold]; tr_all = tr_all[keep[tr_all]]
    tr, va = train_test_split(tr_all, test_size=0.1, stratify=y[tr_all].astype(int), random_state=a.seed)
    cols = CLIN if a.variant == "retina_clin" else []
    C = StandardScaler().fit(d.loc[tr, cols]).transform(d[cols]).astype(np.float32) if cols else np.zeros((len(d), 1), np.float32)
    model = Net(len(cols)).to(dev).to(memory_format=torch.channels_last)
    bb = list(model.bb.parameters()); other = [p for n, p in model.named_parameters() if not n.startswith("bb.")]
    opt = torch.optim.AdamW([{"params": bb, "lr": 5e-5}, {"params": other, "lr": 1e-3}], weight_decay=0.05)
    bs = 32; steps = a.epochs * math.ceil(len(tr) / bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[5e-5, 1e-3], total_steps=steps, pct_start=0.1)
    pw = torch.tensor(math.sqrt((1 - y[tr].mean()) / y[tr].mean()), device=dev)
    best, best_state, bad = -1, None, 0
    for ep in range(a.epochs):
        model.train(); perm = np.random.permutation(tr)
        for i in range(0, len(perm), bs):
            im, m, c, yb = batch(perm[i:i + bs], I, offs, M, C, y, True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = F.binary_cross_entropy_with_logits(model(im.to(dev), m.to(dev), c.to(dev)).float(), yb.to(dev), pos_weight=pw)
            opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
        auc = roc_auc_score(y[va], predict(model, va, I, offs, M, C, y, dev))
        print(f"fold {a.fold} {a.variant} ep {ep} inner-val AUROC {auc:.4f}", flush=True)
        if auc > best: best, bad, best_state = auc, 0, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= 4: break
    model.load_state_dict(best_state)
    te_k = te[keep[te]]; p = predict(model, te_k, I, offs, M, C, y, dev)
    od = os.path.join(D, "stage2_retina"); os.makedirs(od, exist_ok=True)
    pd.DataFrame({"entity_id": d.entity_id.values[te_k], "y": y[te_k], "pred": p}).to_parquet(os.path.join(od, f"oof_{a.variant}_fold{a.fold}_seed{a.seed}.parquet"))
    torch.save(best_state, os.path.join(od, f"model_{a.variant}_fold{a.fold}_seed{a.seed}.pt"))
    print(f"DONE fold {a.fold} {a.variant}: inner best {best:.4f}, outer-fold AUROC {roc_auc_score(y[te_k], p):.4f} (n={len(te_k)})", flush=True)


if __name__ == "__main__":
    main()
