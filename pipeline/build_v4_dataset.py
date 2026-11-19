"""Build the v4 BASELINE-ONLY analytic dataset for the incident-CVD reanalysis.

One row per participant who is (a) CVD/TIA-free at baseline with a determinate incident_F2_primary label,
(b) has a baseline ECG and >=1 baseline retinal image in the existing v3 split frames (split kept as-is).
Writes data_v4/:
  cohort_v4.parquet            participant table: split, labels, risk factors, ECG + retina paths
  ecg_wcr_{split}.npy          (N,12,2500) float32 mV @250 Hz (x0.005, resample_poly 1:2), NO z-score  -> WCRv2
  ecg_vit_{split}.npy          (N,12,5000) float16 @500 Hz: notch 50/60 + bandpass 0.5-100 Hz, clip 0.5/99.5 pct,
                               per-lead z-score (JEPA-ViT pipeline)                               -> ViT-JEPA
Row order of every array == row order of cohort_v4 within that split (column `row_in_split`); asserted on load.
"""
import os, numpy as np, pandas as pd
from scipy.signal import resample_poly, iirnotch, butter, filtfilt
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data_v4"); os.makedirs(OUT, exist_ok=True)
DF = "/media/data1/mbahani/dataframes"
RAW = "/media/data1/mbahani/CLSA/CLSA_ALL/CLSA_Data/CLSA/2401002_MontrealHeartInstitute_ECG_RAW_BL_DATA/WaveformData/{}_ecg_WaveformData.npy"
FS = 500


def vit_preprocess(x):
    for f0 in (50, 60):
        b, a = iirnotch(f0, 30, FS); x = filtfilt(b, a, x, axis=1)
    b, a = butter(2, [0.5 / (FS / 2), 100 / (FS / 2)], btype="band"); x = filtfilt(b, a, x, axis=1)
    lo, hi = np.percentile(x, 0.5, axis=1, keepdims=True), np.percentile(x, 99.5, axis=1, keepdims=True)
    x = np.clip(x, lo, hi)
    return ((x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + 1e-6)).astype(np.float16)


def load_one(eid):
    x = np.squeeze(np.load(RAW.format(eid))).astype(np.float64)
    if x.shape != (12, 5000) or not np.isfinite(x).all():
        return None
    return resample_poly(x * 0.005, 1, 2, axis=-1).astype(np.float32), vit_preprocess(x)


def main():
    lab = pd.read_csv(os.path.join(HERE, "incident_cvd_labels_v1_F2primary.csv"), index_col=0)
    rf = pd.read_parquet(os.path.join(HERE, "baseline_risk_factors.parquet"))
    rows = []
    for sp in ["train", "val", "test"]:
        t = pd.read_csv(f"{DF}/{sp}_df_v3.csv")
        t = t[t.retinal_image.str.contains("_RET_IMG_BL", na=False)]
        g = t.groupby("entity_id")
        r = pd.DataFrame({"split": sp,
                          "retina_left": g.retinal_image.apply(lambda s: next((p for p in s if "left" in p), None)),
                          "retina_right": g.retinal_image.apply(lambda s: next((p for p in s if "right" in p), None))})
        rows.append(r)
    co = pd.concat(rows)
    assert not co.index.duplicated().any(), "participant in >1 split"
    co = co.join(lab, how="inner").dropna(subset=["incident_F2_primary"]).join(rf, how="left")
    co["ecg_raw_path"] = [RAW.format(e) for e in co.index]
    co = co[[os.path.exists(p) for p in co.ecg_raw_path]]
    co.index.name = "entity_id"

    keep, arrays = [], {}
    for sp in ["train", "val", "test"]:
        ids = co.index[co.split == sp].values
        with ProcessPoolExecutor(32) as ex:
            res = list(ex.map(load_one, ids, chunksize=64))
        ok = [i for i, r in enumerate(res) if r is not None]
        np.save(os.path.join(OUT, f"ecg_wcr_{sp}.npy"), np.stack([res[i][0] for i in ok]))
        np.save(os.path.join(OUT, f"ecg_vit_{sp}.npy"), np.stack([res[i][1] for i in ok]))
        k = pd.DataFrame({"entity_id": ids[ok].astype(np.int64), "row_in_split": np.arange(len(ok))})
        keep.append(k); print(sp, "participants", len(ids), "ECG ok", len(ok))
    k = pd.concat(keep).set_index("entity_id")
    co = co.join(k, how="inner")
    co.to_parquet(os.path.join(OUT, "cohort_v4.parquet"))
    s = co.groupby("split").agg(n=("incident_F2_primary", "size"), cases=("incident_F2_primary", "sum"),
                                f1_cases=("incident_F1", lambda v: int((v == 1).sum())))
    s["prev"] = (s.cases / s.n).round(4)
    print(s.to_string())


if __name__ == "__main__":
    main()
