"""A3 (post hoc, plan locked 2026-10-26): detection versus prediction.
Abnormal baseline ECG from CardioSoft statements (BERT 77-label ground truth, prespecified mapping); baseline cardiac
medication from baseline ATC codes. Participants without a baseline statement are excluded from A3 subsets."""
import json, re, sys
import numpy as np, pandas as pd
from outcome_validity_common import samples, evaluate, RAW, SEED

ABN = {'AF/flutter': ['Afib', 'Atrial flutter'], 'Paced': ['Ventricular paced', 'Atrial paced'],
       'LBBB': ['Left bundle branch block'], 'RBBB': ['Right bundle branch block'],
       '2nd/3rd-degree AV block': ['2nd degree AV block - mobitz 1', '2nd degree AV block - mobitz 2', 'Third Degree AV Block'],
       'LVH (voltage)': ['Left ventricular hypertrophy', 'SV1 + RV5 or RV6 > 35 mm', 'RaVL > 11 mm'],
       'Pathological Q waves': ['Q wave (inferior - II, III, aVF)', 'Q wave (lateral- I, aVL, V5-V6)', 'Q wave (anterior - V3-V4)',
                                'Q wave (septal- V1-V2)', 'Q wave (posterior - V7-V9)']}
CARDIAC_ATC = r'B01AA|B01AE|B01AF|C01AA|C01B|C07AA07|C03C|C03EB|C01DA'
OUT = '/volume/CLSA/outputs/revision/outcome_validity/'

def baseline_flags():
    g = pd.read_parquet('/media/data1/datasets/CLSA/clsa_bert_gt_deepecg_tok.parquet')
    g.index = g.ecg_path.str.extract(r'/(\d+)_ecg')[0].astype(int).values
    f = pd.DataFrame({k: (g[v].sum(1) > 0) for k, v in ABN.items()})
    f['abnormal_ecg'] = f.any(axis=1)
    bl = RAW + 'BL/2401002_MontrealHeartInstitute_RAvram_Baseline_CoPv7.csv'
    hdr = pd.read_csv(bl, nrows=0).columns
    atc = [c for c in hdr if re.fullmatch(r'MEDI_ID_ATC_\d+_COM', c)]
    d = pd.read_csv(bl, usecols=['entity_id'] + atc, dtype=str).set_index('entity_id'); d.index = d.index.astype(int)
    med = d.apply(lambda c: c.str.upper().str.strip().str.match(CARDIAC_ATC, na=False)).any(axis=1).rename('cardiac_med')
    return f, med

def main():
    import os; os.makedirs(OUT, exist_ok=True)
    f, med = baseline_flags()
    rows, desc, boot = [], {}, {}
    for s, d in samples().items():
        d = d.join(f).join(med)
        d['cardiac_med'] = d.cardiac_med.fillna(False).astype(bool)
        has = d.abnormal_ecg.notna()
        d['abnormal_ecg'] = d.abnormal_ecg.fillna(False).astype(bool)
        desc[s] = {'n': len(d), 'no_baseline_statement': int((~has).sum()),
                   'abnormal_ecg_n': int(d.abnormal_ecg.sum()), 'abnormal_ecg_events': int(d.loc[d.abnormal_ecg, 'y'].sum()),
                   'cardiac_med_n': int(d.cardiac_med.sum()), 'cardiac_med_not_abnormal_ecg_n': int((d.cardiac_med & ~d.abnormal_ecg).sum()),
                   'components_n': {k: int(d[k].fillna(False).astype(bool).sum()) for k in ABN}}
        rows.append({'sample': s, **evaluate(d, d.y, f'{s}: primary outcome')})
        a1 = d[has & ~d.abnormal_ecg]
        rows.append({'sample': s, **evaluate(a1, a1.y, f'{s}: A3(1) normal ECG only', boot)})
        a2 = d[has & ~d.abnormal_ecg & ~d.cardiac_med]
        rows.append({'sample': s, **evaluate(a2, a2.y, f'{s}: A3(2) normal ECG, no cardiac medication')})
        ab = d[has & d.abnormal_ecg]
        rows.append({'sample': s, **evaluate(ab, ab.y, f'{s}: A3 abnormal ECG only', boot)})
        # Interaction: difference in delta (A+ - FRS) between normal and abnormal ECG, independent bootstraps per stratum
        dn = boot[f'{s}: A3(1) normal ECG only']; da = boot[f'{s}: A3 abnormal ECG only']
        m = min(len(dn), len(da))
        diff = (dn['A+'] - dn['FRS']).to_numpy()[:m] - (da['A+'] - da['FRS']).to_numpy()[:m]
        rn, ra = rows[-3], rows[-1]
        pt = rn['Delta vs FRS'] - ra['Delta vs FRS']
        se = diff.std(ddof=1)
        from scipy.stats import norm
        desc[s]['interaction_delta_normal_minus_abnormal'] = {'estimate': pt, 'ci': list(np.percentile(diff, [2.5, 97.5])),
                                                              'p_wald_bootstrap_se': float(2 * norm.sf(abs(pt) / se))}
    r = pd.DataFrame(rows); r.to_csv(OUT + 'A3_results.csv', index=False)
    json.dump(desc, open(OUT + 'A3_descriptives.json', 'w'), indent=1, default=float)
    pd.set_option('display.width', 250)
    print(r[['analysis', 'events', 'n', 'AUROC A+', 'AUROC FRS', 'AUROC PREVENT', 'Delta vs FRS', 'Delta vs FRS lo', 'Delta vs FRS hi', 'Delta vs PREVENT', 'Delta vs PREVENT lo', 'Delta vs PREVENT hi']].round(3).to_string(index=False))
    print(json.dumps(desc, indent=1, default=float))

if __name__ == '__main__':
    main()
