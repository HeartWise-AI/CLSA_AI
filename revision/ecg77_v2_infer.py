"""Run DeepECG-SSL v2 (amp-preserved) 77-label ONNX on CLSA baseline ECGs with BERT report labels.
Input handling matches the CVD loader: raw CardioSoft x0.005 mV, 500->250 Hz by resample_poly.
Output (internal, participant-level; never package): outputs/ecg77/clsa_bl_v2_logits.npz"""
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd, onnxruntime as ort
from scipy.signal import resample_poly

GT = '/media/data1/datasets/CLSA/clsa_bert_gt_deepecg_tok.parquet'
ONNX = '/media/data1/models/DeepECG-SSL/wcr-v2/deepecg-ssl-v2-amp-preserved-ft77-best.onnx'
OUT = '/volume/CLSA/outputs/ecg77/clsa_bl_v2_logits.npz'

def load(p):
    try:
        x = np.squeeze(np.load(p.replace('/mnt/data1/', '/media/data1/'))).astype(np.float64)
    except Exception:
        return None
    if x.ndim != 2: return None
    if x.shape[0] != 12: x = x.T
    if x.shape != (12, 5000): return None
    x = resample_poly(x * 0.005, 1, 2, axis=-1)
    return x.astype(np.float32) if np.isfinite(x).all() else None

def main():
    gt = pd.read_parquet(GT)
    s = ort.InferenceSession(ONNX, providers=['CPUExecutionProvider'])
    name = s.get_inputs()[0].name
    logits = np.full((len(gt), 77), np.nan, np.float32)
    paths = gt.ecg_path.tolist()
    with ThreadPoolExecutor(16) as ex:
        for st in range(0, len(paths), 256):
            X = list(ex.map(load, paths[st:st + 256]))
            ok = [i for i, x in enumerate(X) if x is not None]
            if ok:
                E = np.stack([X[i] for i in ok])
                out = np.concatenate([s.run(None, {name: E[j:j + 64]})[0] for j in range(0, len(E), 64)])
                logits[[st + i for i in ok]] = out
            print(st + 256, flush=True)
    np.savez_compressed(OUT, logits=logits, ecg_path=np.array(paths))

if __name__ == '__main__':
    main()
