"""A3 quality check (owner decision 27 Oct 2026, replacing the cardiologist over-read): DeepECG-SSL v2 77-label model as the
reference for the abnormal-ECG categories, at the prespecified MHI Youden thresholds. Reports agreement of the CardioSoft
machine statements with the model per category, and reruns A3 with the model-defined abnormal ECG."""
import ast, json, re
import numpy as np, pandas as pd
from sklearn.metrics import cohen_kappa_score
from scipy.stats import norm
from outcome_validity_common import samples, evaluate
from ov_a3_detection import baseline_flags, ABN

OUT = '/volume/CLSA/outputs/revision/outcome_validity/'
src = open('/volume/CLSA/DeepECG_Docker/utils/constants.py').read()
CONV = ast.literal_eval(re.search(r'WCR_COLUMN_CONVERSION = (\[.*?\])', src, re.S).group(1))

def model_flags():
    g = pd.read_parquet('/media/data1/datasets/CLSA/clsa_bert_gt_deepecg_tok.parquet', columns=['ecg_path'] + [])
    L = pd.read_parquet('/media/data1/datasets/CLSA/clsa_bert_gt_deepecg_tok.parquet').columns[2:].tolist()
    z = np.load('/volume/CLSA/outputs/ecg77/clsa_bl_v2_logits.npz')
    assert (z['ecg_path'] == g.ecg_path.to_numpy()).all()
    P = pd.DataFrame(1 / (1 + np.exp(-z['logits'][:, CONV])), columns=L, index=g.ecg_path.str.extract(r'/(\d+)_ecg')[0].astype(int).values)
    thr = pd.read_csv('/volume/CLSA/outputs/ecg77/mhi_v2_youden_thresholds.csv').set_index('label').thr
    f = pd.DataFrame({k: (P[v] >= thr[v]).any(axis=1) for k, v in ABN.items()})
    f['abnormal_ecg'] = f.any(axis=1)
    return f

def main():
    mach, _ = baseline_flags(); mod = model_flags(); _, med = baseline_flags()
    agree, rows, boot, inter = {}, [], {}, {}
    for s, d in samples().items():
        ix = d.index[d.index.isin(mach.index) & d.index.isin(mod.index)]
        a = {}
        for k in list(ABN) + ['abnormal_ecg']:
            m, r = mach.loc[ix, k].astype(bool), mod.loc[ix, k].astype(bool)
            n_ref = int(r.sum()); n_m = int(m.sum()); both = int((m & r).sum())
            a[k] = {'machine_pos': n_m, 'model_pos': n_ref, 'both': both, 'kappa': cohen_kappa_score(m, r),
                    'percent_agreement': 100 * (m == r).mean(),
                    'machine_sens_vs_model': both / n_ref if n_ref else None, 'machine_ppv_vs_model': both / n_m if n_m else None}
        agree[s] = a
        dd = d.loc[ix].join(mod[['abnormal_ecg']].rename(columns={'abnormal_ecg': 'abn_model'})).join(med)
        dd['cardiac_med'] = dd.cardiac_med.fillna(False).astype(bool); ab = dd.abn_model.astype(bool)
        x1 = dd[~ab]; x2 = dd[~ab & ~dd.cardiac_med]; x3 = dd[ab]
        rows.append({'sample': s, **evaluate(x1, x1.y, f'{s}: A3(1) normal ECG only, DeepECG-SSL reference', boot)})
        rows.append({'sample': s, **evaluate(x2, x2.y, f'{s}: A3(2) normal ECG, no cardiac medication, DeepECG-SSL reference')})
        rows.append({'sample': s, **evaluate(x3, x3.y, f'{s}: A3 abnormal ECG only, DeepECG-SSL reference', boot)})
        bn, ba = boot[f'{s}: A3(1) normal ECG only, DeepECG-SSL reference'], boot[f'{s}: A3 abnormal ECG only, DeepECG-SSL reference']
        mlen = min(len(bn), len(ba)); diff = (bn['A+'] - bn['FRS']).to_numpy()[:mlen] - (ba['A+'] - ba['FRS']).to_numpy()[:mlen]
        pt = rows[-3]['Delta vs FRS'] - rows[-1]['Delta vs FRS']
        inter[s] = {'estimate': pt, 'ci': list(np.percentile(diff, [2.5, 97.5])), 'p': float(2 * norm.sf(abs(pt) / diff.std(ddof=1))), 'abnormal_n': int(ab.sum())}
    r = pd.DataFrame(rows); r.to_csv(OUT + 'A3_deepecg_reference_results.csv', index=False)
    json.dump({'agreement_machine_vs_model': agree, 'interaction': inter}, open(OUT + 'A3_deepecg_reference_agreement.json', 'w'), indent=1, default=float)
    pd.set_option('display.width', 250)
    print(r[['analysis', 'events', 'n', 'AUROC A+', 'AUROC FRS', 'AUROC PREVENT', 'Delta vs FRS', 'Delta vs FRS lo', 'Delta vs FRS hi', 'Delta vs PREVENT', 'Delta vs PREVENT lo', 'Delta vs PREVENT hi']].round(3).to_string(index=False))
    for s in agree: print(s, {k: (v['machine_pos'], v['model_pos'], round(v['kappa'], 3), round(v['percent_agreement'], 1)) for k, v in agree[s].items()}, inter[s])

if __name__ == '__main__':
    main()
