"""Per-label performance of DeepECG-SSL v2 77-label model vs BERT report labels on CLSA baseline ECGs.
Aggregate outputs only; labels with <6 positives are suppressed (CLSA small-cell rule)."""
import ast, json, re
import numpy as np, pandas as pd
from scipy.stats import norm
from sklearn.metrics import average_precision_score, roc_auc_score

R = '/volume/CLSA/outputs/ecg77/'
GT = '/media/data1/datasets/CLSA/clsa_bert_gt_deepecg_tok.parquet'
SCORES = '/media/data1/ravram/CLSA_validation_files/manuscript/data_v4/scores_v4.parquet'
src = open('/volume/CLSA/DeepECG_Docker/utils/constants.py').read()
CONV = ast.literal_eval(re.search(r'WCR_COLUMN_CONVERSION = (\[.*?\])', src, re.S).group(1))
CATS = ast.literal_eval(re.search(r'ECG_CATEGORIES = (\{.*?\n\})', src, re.S).group(1))
MIN_N = 6

def delong(y, s):
    pos, neg = s[y == 1], s[y == 0]
    m, n = len(pos), len(neg)
    allr = pd.Series(np.r_[pos, neg]).rank().to_numpy()
    rp, rn = pd.Series(pos).rank().to_numpy(), pd.Series(neg).rank().to_numpy()
    v10 = (allr[:m] - rp) / n
    v01 = 1 - (allr[m:] - rn) / m
    auc = v10.mean()
    se = np.sqrt(v10.var(ddof=1) / m + v01.var(ddof=1) / n)
    return auc, max(0, auc - 1.96 * se), min(1, auc + 1.96 * se)

def evaluate(Y, P, thr):
    rows = []
    for c in Y.columns:
        y, p = Y[c].to_numpy(), P[c].to_numpy()
        npos = int(y.sum())
        r = {'label': c, 'n_pos': npos if npos >= MIN_N else f'<{MIN_N}', 'prevalence_pct': round(100 * y.mean(), 2) if npos >= MIN_N else None}
        if npos >= MIN_N and npos < len(y):
            a, lo, hi = delong(y, p)
            pred = p >= thr[c]
            tp = (pred & (y == 1)).sum(); fp = (pred & (y == 0)).sum()
            r.update(auroc=a, auroc_lo=lo, auroc_hi=hi, auprc=average_precision_score(y, p),
                     sens=tp / npos, spec=1 - fp / (len(y) - npos), ppv=tp / max(tp + fp, 1))
        rows.append(r)
    return pd.DataFrame(rows)

def main():
    gt = pd.read_parquet(GT)
    L = gt.columns[2:].tolist()
    z = np.load(R + 'clsa_bl_v2_logits.npz')
    assert (z['ecg_path'] == gt.ecg_path.to_numpy()).all()
    ok = np.isfinite(z['logits']).all(1)
    prob = 1 / (1 + np.exp(-z['logits'][ok]))
    Y = gt.loc[ok, L].astype(int).reset_index(drop=True)
    # Choose output order empirically: identity vs the Docker WCR column conversion.
    order = {}
    for k, idx in {'identity': list(range(77)), 'wcr_conversion': CONV}.items():
        P = pd.DataFrame(prob[:, idx], columns=L)
        order[k] = float(np.nanmean([roc_auc_score(Y[c], P[c]) for c in L if Y[c].sum() >= 50]))
    best = max(order, key=order.get)
    P = pd.DataFrame(prob[:, list(range(77)) if best == 'identity' else CONV], columns=L)
    t = pd.read_csv(R + 'mhi_v2_youden_thresholds.csv').set_index('label')
    thr = t.thr.to_dict()
    cat = {l: k for k, v in CATS.items() for l in v}
    res = evaluate(Y, P, thr)
    res['category'] = res.label.map(cat).fillna('Other diagnoses')
    res = res.join(t[['mhi_auroc', 'mhi_auprc']], on='label')
    # Analytic cohort subset
    eid = gt.loc[ok, 'ecg_path'].str.extract(r'/(\d+)_ecg')[0].astype(int).to_numpy()
    inc = np.isin(eid, pd.read_parquet(SCORES, columns=['split']).index)
    sub = evaluate(Y[inc].reset_index(drop=True), P[inc].reset_index(drop=True), thr)
    res = res.merge(sub[['label', 'auroc']].rename(columns={'auroc': 'auroc_analytic_cohort'}), on='label')
    res.to_csv(R + 'clsa_v2_77label_performance.csv', index=False)
    ev = res.dropna(subset=['auroc'])
    summ = {'n_ecg_with_labels': int(len(gt)), 'n_ecg_scored': int(ok.sum()), 'n_analytic_cohort_scored': int(inc.sum()),
            'output_order_check_mean_auroc': order, 'order_used': best,
            'macro_auroc_all_evaluable': float(ev.auroc.mean()), 'macro_auprc_all_evaluable': float(ev.auprc.mean()),
            'n_labels_evaluable': int(len(ev)),
            'by_category': ev.groupby('category').agg(n_labels=('label', 'size'), macro_auroc=('auroc', 'mean'), macro_auprc=('auprc', 'mean')).round(4).to_dict('index')}
    json.dump(summ, open(R + 'clsa_v2_77label_summary.json', 'w'), indent=2)
    print(json.dumps(summ, indent=2))

if __name__ == '__main__':
    main()
