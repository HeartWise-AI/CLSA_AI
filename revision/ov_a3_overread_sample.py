"""A3 quality check: blinded cardiologist over-read sample (~200 baseline tracings, test set, enriched for positives).
100 machine-abnormal (stratified by A3 category) + 100 machine-normal, random (seed 2026), shuffled under blinded IDs.
INTERNAL ONLY (participant waveforms): written to the NAS, never packaged. Key file is kept separate from the reading form."""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.signal import resample_poly
from outcome_validity_common import samples
from ov_a3_detection import baseline_flags, ABN

OUT = '/media/data1/ravram/CLSA_validation_files/manuscript/overread_A3/'
LEADS = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

def plot(path, fn):
    x = np.squeeze(np.load(path.replace('/mnt/data1/', '/media/data1/'))).astype(float)
    if x.shape[0] != 12: x = x.T
    x = x * 0.005  # mV, 500 Hz
    t = np.arange(x.shape[1]) / 500
    fig, ax = plt.subplots(12, 1, figsize=(11, 14), sharex=True)
    for i in range(12):
        ax[i].plot(t, x[i], lw=.6, color='k'); ax[i].set_ylabel(LEADS[i], rotation=0, labelpad=14)
        ax[i].set_ylim(-2, 2); ax[i].set_xticks(np.arange(0, 10.01, .2), minor=True); ax[i].set_yticks(np.arange(-2, 2.01, .5), minor=True)
        ax[i].grid(which='both', color='#f2b8b8', lw=.3); ax[i].tick_params(labelsize=6)
    ax[-1].set_xlabel('s (25 mm/s equivalent grid: 0.2 s, 0.5 mV)'); plt.tight_layout(); plt.savefig(fn, dpi=110); plt.close()

def main():
    f, _ = baseline_flags()
    d = samples()['test'].join(f)
    d = d[d.abnormal_ecg.notna()]
    g = pd.read_parquet('/media/data1/datasets/CLSA/clsa_bert_gt_deepecg_tok.parquet')
    g.index = g.ecg_path.str.extract(r'/(\d+)_ecg')[0].astype(int).values
    rng = np.random.default_rng(2026)
    ab = d[d.abnormal_ecg.astype(bool)]; picked = []
    cats = list(ABN)
    per = {c: min(int(ab[c].sum()), 100 // len(cats) + 1) for c in cats}  # stratified, rare categories taken in full
    for c in cats:
        pool = ab[ab[c].astype(bool) & ~ab.index.isin(picked)].index.to_numpy()
        picked += list(rng.choice(pool, min(per[c], len(pool)), replace=False))
    rest = ab.index[~ab.index.isin(picked)].to_numpy()
    picked += list(rng.choice(rest, max(0, 100 - len(picked)), replace=False))
    picked = picked[:100] if len(picked) > 100 else picked
    norm = rng.choice(d[~d.abnormal_ecg.astype(bool)].index.to_numpy(), 100, replace=False)
    ids = np.array(list(picked) + list(norm)); rng.shuffle(ids)
    key = pd.DataFrame({'blind_id': [f'R{i + 1:03d}' for i in range(len(ids))], 'entity_id': ids})
    key['machine_abnormal'] = d.loc[ids, 'abnormal_ecg'].astype(bool).to_numpy()
    for c in cats: key[f'machine_{c}'] = d.loc[ids, c].astype(bool).to_numpy()
    key['report_text'] = g.loc[ids, 'reports'].to_numpy()
    for b, e in zip(key.blind_id, key.entity_id): plot(g.loc[e, 'ecg_path'], OUT + f'{b}.png')
    key.to_csv(OUT + 'KEY_do_not_share_with_reader.csv', index=False)
    form = key[['blind_id']].copy()
    for c in cats + ['Overall: abnormal per A3 definition (Y/N)', 'Uninterpretable (Y/N)', 'Comment']: form[c] = ''
    form.to_csv(OUT + 'reading_form.csv', index=False)
    print(len(key), key.machine_abnormal.sum(), {c: int(key[f'machine_{c}'].sum()) for c in cats})

if __name__ == '__main__':
    main()
