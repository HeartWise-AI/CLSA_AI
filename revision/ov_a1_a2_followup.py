"""A1 (corroborated heart-disease-only events) and A2 (consistent reporting across waves). Post hoc; plan locked
2026-10-26 with feasibility amendments recorded in goal_A1_A4_outcome_validity.md.
Corroboration at or before the reporting wave (F1-onset: F1 sources; F2-onset: F1 or F2 sources):
 med  new cardiac ATC at F1 absent at baseline (no ATC inventory exists at F2)
 ecg  new major abnormality in the CardioSoft statement (AF/flutter, paced, complete LBBB, high-degree AV block) absent at baseline
 proc new angiogram or pacemaker vs baseline, or heart surgery in the last 3 months (no cardiac-specific hospitalisation item)"""
import glob, json, re
import numpy as np, pandas as pd
from outcome_validity_common import samples, evaluate, RAW

OUT = '/volume/CLSA/outputs/revision/outcome_validity/'
NEW_MED = r'B01AA|B01AE|B01AF|C03C|C03EB|C01B|C07AA07|C01AA|C01DA|C09DX04'
ECG = {'AF/flutter': r'atrial fibrillation|atrial flutter', 'paced': r'pacemaker|paced rhythm',
       'LBBB': r'(?<!incomplete )left bundle branch block', 'high-degree AV block': r'mobitz ii\b|2:1 av|complete heart block|3rd degree av|third degree av|complete av block'}

def read(w, cols):
    f = RAW + 'BL/2401002_MontrealHeartInstitute_RAvram_Baseline_CoPv7.csv' if w == 'BL' else glob.glob(RAW + w + '/*.csv')[0]
    hdr = pd.read_csv(f, nrows=0).columns
    cols = [c for c in hdr if any(re.fullmatch(p, c) for p in cols)]
    d = pd.read_csv(f, usecols=['entity_id'] + cols, dtype=str).set_index('entity_id'); d.index = d.index.astype(int)
    return d

def ecg_flags(s):
    s = s.fillna('').str.lower().str.replace(r'junctional\s+pacemaker', '', regex=True)
    return pd.DataFrame({k: s.str.contains(p, regex=True) for k, p in ECG.items()})

def atc_any(d):
    return d.apply(lambda c: c.str.upper().str.strip().str.match(NEW_MED, na=False)).any(axis=1)

def build():
    bl = read('BL', [r'MEDI_ID_ATC_\d+_COM', 'ECG_DIAGNOSIS_DETAILS_COM', 'IHD_ANGIO_COM', 'ICQ_PACEMKR_COM'])
    f1 = read('FUP1', [r'MEDI_ID_ATC_\d+_COF1', 'ECG_DIAGNOSIS_DETAILS_COF1', 'IHD_ANGIO_COF1', 'ICQ_PACEMKR_COF1', 'ICQ_SRGYHRT_COF1'])
    f2 = read('FUP2', ['ECG_DIAGNOSIS_DETAILS_COF2', 'IHD_ANGIO_COF2', 'ICQ_PACEMKR_COF2', 'ICQ_SRGYHRT_COF2', 'CCC_HEART_COF2', 'CCC_HEARTCHANGE_COF2'])
    ix = bl.index.union(f1.index).union(f2.index)
    bl, f1, f2 = bl.reindex(ix), f1.reindex(ix), f2.reindex(ix)
    eb, e1, e2 = ecg_flags(bl.ECG_DIAGNOSIS_DETAILS_COM), ecg_flags(f1.ECG_DIAGNOSIS_DETAILS_COF1), ecg_flags(f2.ECG_DIAGNOSIS_DETAILS_COF2)
    medb = atc_any(bl.filter(regex='MEDI_ID_ATC'))
    c = pd.DataFrame(index=ix)
    c['med_F1'] = atc_any(f1.filter(regex='MEDI_ID_ATC')) & ~medb
    c['ecg_F1'] = (e1 & ~eb).any(axis=1); c['ecg_F2'] = (e2 & ~eb).any(axis=1)
    yes = lambda s: s == '1'
    c['proc_F1'] = (yes(f1.IHD_ANGIO_COF1) & ~yes(bl.IHD_ANGIO_COM)) | (yes(f1.ICQ_PACEMKR_COF1) & ~yes(bl.ICQ_PACEMKR_COM)) | yes(f1.ICQ_SRGYHRT_COF1)
    c['proc_F2'] = (yes(f2.IHD_ANGIO_COF2) & ~yes(bl.IHD_ANGIO_COM)) | (yes(f2.ICQ_PACEMKR_COF2) & ~yes(bl.ICQ_PACEMKR_COM)) | yes(f2.ICQ_SRGYHRT_COF2)
    c['heart_F2'] = f2.CCC_HEART_COF2; c['heartchange_F2'] = f2.CCC_HEARTCHANGE_COF2
    return c

def main():
    import os; os.makedirs(OUT, exist_ok=True)
    c = build()
    rows, desc = [], {}
    for s, d in samples().items():
        d = d.join(c)
        f1on = d.hd_only & (d.F1_HEART == 1)
        src = pd.DataFrame({'medication': d.med_F1.fillna(False),
                            'ecg': (d.ecg_F1.fillna(False) | (~f1on & d.ecg_F2.fillna(False))),
                            'procedure': (d.proc_F1.fillna(False) | (~f1on & d.proc_F2.fillna(False)))}).astype(bool)
        corr = src.any(axis=1) & d.hd_only
        h = d.hd_only
        desc[s] = {'hd_only': int(h.sum()), 'hd_only_F1_onset': int(f1on.sum()), 'hd_only_F2_onset': int((h & ~f1on).sum()),
                   'corroborated': int(corr.sum()), 'corroborated_pct': 100 * corr.sum() / h.sum(),
                   'by_source_n': {k: int((src[k] & h).sum()) for k in src}, 'by_source_pct': {k: 100 * (src[k] & h).sum() / h.sum() for k in src},
                   'non_events_with_any_source_pct': 100 * (src.any(axis=1) & (d.y == 0)).mean()}
        rows.append({'sample': s, **evaluate(d, d.y, f'{s}: primary outcome')})
        a = d[~(h & ~corr)]
        rows.append({'sample': s, **evaluate(a, a.y, f'{s}: A1(i) uncorroborated HD-only dropped')})
        rows.append({'sample': s, **evaluate(d, (d.y == 1) & ~(h & ~corr), f'{s}: A1(ii) uncorroborated HD-only as non-events')})
        # A2
        st = pd.Series('not assessable (F2 onset)', index=d.index)
        st[f1on & d.heart_F2.isin(['1', '11'])] = 'consistent'
        st[f1on & (d.heart_F2 == '2')] = 'reversed'
        st[f1on & ~d.heart_F2.isin(['1', '11', '2'])] = 'missing at F2'
        st[~h] = ''
        desc[s]['A2_status'] = st[h].value_counts().to_dict()
        desc[s]['A2_F1onset_heartchange_F2'] = d.loc[f1on, 'heartchange_F2'].value_counts(dropna=False).to_dict()
        b = d[st != 'reversed']
        rows.append({'sample': s, **evaluate(b, b.y, f'{s}: A2 reversed HD-only excluded')})
        b2 = d[~st.isin(['reversed', 'missing at F2', 'not assessable (F2 onset)'])]
        rows.append({'sample': s, **evaluate(b2, b2.y, f'{s}: A2 secondary: only consistent HD-only retained')})
    r = pd.DataFrame(rows); r.to_csv(OUT + 'A1_A2_results.csv', index=False)
    json.dump(desc, open(OUT + 'A1_A2_descriptives.json', 'w'), indent=1, default=float)
    pd.set_option('display.width', 250)
    print(r[['analysis', 'events', 'n', 'AUROC A+', 'AUROC FRS', 'AUROC PREVENT', 'Delta vs FRS', 'Delta vs FRS lo', 'Delta vs FRS hi', 'Delta vs PREVENT', 'Delta vs PREVENT lo', 'Delta vs PREVENT hi']].round(3).to_string(index=False))
    print(json.dumps(desc, indent=1, default=float))

if __name__ == '__main__':
    main()
