"""AI-ECG inference for the MHI any-ECG cohort (patients analysable: prevalence no/uncertain, npy present).
Frozen CLSA fold models (mean of 5): ViT ECG-only (250->500 Hz, CLSA filter + z-score, two 5-s halves) and WCRv2
ECG-only (ADC x0.00488 mV, 250 Hz). Parallel npy loading. Output mhi_anyecg/predictions.parquet. Run with tools/venv_fss/bin/python."""
import os, numpy as np, pandas as pd, torch
from concurrent.futures import ThreadPoolExecutor
from scipy.signal import resample_poly
from build_v4_dataset import vit_preprocess
from stage1_vit_cv import Net as VitNet
from stage1b_wcr_cv import Net as WcrNet
D = "data_v4"; dev = torch.device("cuda:0")
lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
O = os.environ.get("MHI_OUT", "mhi_anyecg"); c = pd.read_parquet(f"{O}/cohort_labels.parquet"); c = c[c.npy_ok & c.prev.isin(["no", "unc"])].reset_index(drop=True)
print("n to score:", len(c), flush=True)
vits = [VitNet(0) for _ in range(5)]; wcrs = [WcrNet(0) for _ in range(5)]
for k in range(5):
    vits[k].load_state_dict(torch.load(f"{D}/stage1_vit/model_ecg_fold{k}_seed0.pt", map_location="cpu")); vits[k].to(dev).eval()
    wcrs[k].load_state_dict(torch.load(f"{D}/stage1b_wcr/model_wcr_fold{k}_seed0.pt", map_location="cpu")); wcrs[k].to(dev).eval()
def load(p):
    try:
        x = np.squeeze(np.load(p)).astype(np.float64); x = x.T if x.shape[0] != 12 else x
        if x.shape != (12, 2500) or not np.isfinite(x).all(): return None
        return vit_preprocess(resample_poly(x, 2, 1, axis=-1)[:, :5000]).astype(np.float32), (x * 0.00488).astype(np.float32)
    except Exception: return None
V, Wc, ok = np.full(len(c), np.nan), np.full(len(c), np.nan), np.zeros(len(c), bool)
B = 256
with ThreadPoolExecutor(48) as ex:
    for i in range(0, len(c), B):
        res = list(ex.map(load, c.npy_path.values[i:i + B])); good = [j for j, r in enumerate(res) if r is not None]
        if not good: continue
        xv = torch.from_numpy(np.stack([res[j][0] for j in good])).to(dev); xw = torch.from_numpy(np.stack([res[j][1] for j in good])).to(dev); z = torch.zeros(len(good), 1, device=dev)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            pv = torch.stack([(torch.sigmoid(m(xv[:, :, :2500], z)) + torch.sigmoid(m(xv[:, :, 2500:], z))) / 2 for m in vits]).mean(0).float().cpu().numpy().ravel()
            pw = torch.stack([torch.sigmoid(m(xw, z)) for m in wcrs]).mean(0).float().cpu().numpy().ravel()
        idx = np.arange(i, min(i + B, len(c)))[good]; V[idx], Wc[idx], ok[idx] = lg(pv), lg(pw), True
        if (i // B) % 40 == 0: print(i, "/", len(c), flush=True)
pd.DataFrame({"MRN": c.MRN, "vit": V, "wcr": Wc, "ecg_ok": ok}).to_parquet(f"{O}/predictions.parquet"); print("done; ok:", ok.sum())
