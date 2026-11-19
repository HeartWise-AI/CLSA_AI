"""Score CLSA participants with indeterminate F2 status (plus 300 test participants as a reproducibility check)
with the released 5-fold DeepECG-SSL-CLSA-CVD ensemble. Run with tools/venv_fss/bin/python."""
import os, sys, numpy as np, pandas as pd
from scipy.signal import resample_poly
H = os.path.dirname(os.path.abspath(__file__)); M = os.path.dirname(H)
sys.path.insert(0, os.path.join(M, "tools", "fairseq-signals")); sys.path.insert(0, "/media/data1/ravram/CLSA_validation_files/hf_release")
from inference import DeepECGSSLCVD
RAW = "/media/data1/mbahani/CLSA/CLSA_ALL/CLSA_Data/CLSA/2401002_MontrealHeartInstitute_ECG_RAW_BL_DATA/WaveformData/{}_ecg_WaveformData.npy"
L = pd.read_csv(os.path.join(M, "incident_cvd_labels_v2_components.csv"), index_col=0)
ind = L.index[L.incident_F2_primary.isna()].tolist()
te = pd.read_parquet(os.path.join(M, "results_v5/test_predictions_wcr.parquet"))
chk = te.entity_id.sample(300, random_state=0).tolist()
ids, X = [], []
for e in ind + chk:
    p = RAW.format(e)
    if not os.path.exists(p): continue
    x = np.squeeze(np.load(p)).astype(np.float64)
    if x.shape != (12, 5000) or not np.isfinite(x).all(): continue
    ids.append(e); X.append(resample_poly(x * 0.005, 1, 2, axis=-1).astype(np.float32))
X = np.stack(X); print("scored", len(ids), "of", len(ind) + len(chk))
m = DeepECGSSLCVD("/media/data1/ravram/CLSA_validation_files/hf_release/weights", "/media/data1/models/DeepECG-SSL/wcr-v2/deepecg-ssl-v2-wcr-amp-preserved-encoder.pt", device="cuda")
p = m.ecg_probability(X)
out = pd.DataFrame({"entity_id": ids, "p_ecg": p, "check": [e in set(chk) for e in ids]})
out.to_parquet(os.path.join(H, "indeterminate", "ecg_scores.parquet"))
c = out[out.check].merge(te[["entity_id", "DeepECG-SSL ECG-only"]], on="entity_id")
print("check corr", np.corrcoef(c.p_ecg, c["DeepECG-SSL ECG-only"])[0, 1], "max abs diff", (c.p_ecg - c["DeepECG-SSL ECG-only"]).abs().max())
