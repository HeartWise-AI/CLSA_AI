"""A4 quantitative bias analysis (post hoc; plan locked 2026-10-26).
Deterministic: reclassify heart-disease-only events to non-events in descending Model A+ order; find the smallest k at
which the bootstrap lower 95% bound (1,000 paired resamples, seed 2026) and the point estimate of delta AUROC (A+ - FRS) reach 0.
Probabilistic: 5,000 iterations, each = misclassification draw + participant bootstrap. logit(1-PPV_i)=a0+a1*z_i over
HD-only events, logit(1-NPV_i)=b0+b1*z_i over non-events; a0/b0 solved so the mean equals the drawn 1-PPV / 1-NPV.
z = standardized logit of Model A+. PPV ~ trapezoid(0.50, 0.60, 0.80, 0.90); NPV ~ trapezoid(0.970, 0.980, 0.990, 0.995).
Specific-component events never reclassified."""
import json, sys
from multiprocessing import Pool
import numpy as np, pandas as pd
from scipy.optimize import brentq
from scipy.stats import rankdata, trapezoid
from outcome_validity_common import samples, lg, SEED, B

OUT = '/volume/CLSA/outputs/revision/outcome_validity/'
A1S, B1S = [1, 1.5, 2, 3], [1, 0.67, 0.5]
PPV_D = trapezoid(0.25, 0.75, loc=0.50, scale=0.40)
NPV_D = trapezoid(0.4, 0.8, loc=0.970, scale=0.025)
NSIM = 5000

def auc(y, s):
    r = rankdata(s); n1 = y.sum(); n0 = len(y) - n1
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

def delta(y, a, f): return auc(y, a) - auc(y, f)

def solve(z, target, slope):
    b = np.log(slope)
    a0 = brentq(lambda a: (1 / (1 + np.exp(-(a + b * z)))).mean() - target, -30, 30)
    return 1 / (1 + np.exp(-(a0 + b * z)))

G = {}
def init(d):
    G['y'] = d.y.to_numpy().astype(bool); G['hd'] = d.hd_only.to_numpy(); G['a'] = d.aplus.to_numpy(); G['f'] = d.frs.to_numpy()
    z = lg(G['a']); G['z'] = (z - z.mean()) / z.std()

def sim(args):
    a1, b1, ppv, seed = args
    rng = np.random.default_rng(seed)
    y, hd, z = G['y'].copy(), G['hd'], G['z']
    ppv = PPV_D.rvs(random_state=rng) if ppv is None else ppv
    npv = NPV_D.rvs(random_state=rng)
    p_fp = solve(z[hd], 1 - ppv, a1); flip = np.zeros(len(y), bool); flip[np.where(hd)[0]] = rng.random(hd.sum()) < p_fp
    ne = ~G['y']; p_fn = solve(z[ne], 1 - npv, b1); add = np.zeros(len(y), bool); add[np.where(ne)[0]] = rng.random(ne.sum()) < p_fn
    y = (y & ~flip) | add
    i = rng.integers(0, len(y), len(y))
    return delta(y[i], G['a'][i], G['f'][i])

def tipping(d):
    y0 = d.y.to_numpy().astype(bool); a, f = d.aplus.to_numpy(), d.frs.to_numpy()
    order = np.where(d.hd_only.to_numpy())[0][np.argsort(-a[d.hd_only.to_numpy()])]
    rng = np.random.default_rng(SEED); idx = [rng.integers(0, len(y0), len(y0)) for _ in range(B)]
    def at(k):
        y = y0.copy(); y[order[:k]] = False
        return y
    pt = lambda k: delta(at(k), a, f)
    lo = lambda k: np.percentile([delta(at(k)[i], a[i], f[i]) for i in idx], 2.5)
    def first(fun, hi):
        l, h = 0, hi
        if fun(h) > 0: return None
        while h - l > 1:
            m = (l + h) // 2
            (l, h) = (m, h) if fun(m) > 0 else (l, m)
        return h
    N = len(order)
    kl, kp = first(lo, N), first(pt, N)
    return {'hd_only': N, 'k_lower_bound_zero': kl, 'pct_lower_bound_zero': None if kl is None else 100 * kl / N,
            'k_point_zero': kp, 'pct_point_zero': None if kp is None else 100 * kp / N}

def main():
    res, rows, heat = {}, [], []
    for s, d in samples().items():
        res[s] = {'tipping': tipping(d)}
        print(s, res[s], flush=True)
        with Pool(48, initializer=init, initargs=(d,)) as pool:
            for a1 in A1S:
                for b1 in B1S:
                    v = np.array(pool.map(sim, [(a1, b1, None, SEED * 100000 + i) for i in range(NSIM)], chunksize=50))
                    rows.append({'sample': s, 'exp_a1': a1, 'exp_b1': b1, 'median_delta': np.median(v), 'lo': np.percentile(v, 2.5),
                                 'hi': np.percentile(v, 97.5), 'p_delta_le_0': (v <= 0).mean()})
                    print(rows[-1], flush=True)
            for b1 in [1, 0.5]:
                for ppv in np.round(np.arange(0.50, 0.901, 0.05), 2):
                    for a1 in A1S:
                        v = np.array(pool.map(sim, [(a1, b1, ppv, SEED * 100000 + i) for i in range(1000)], chunksize=50))
                        heat.append({'sample': s, 'exp_b1': b1, 'ppv': ppv, 'exp_a1': a1, 'median_delta': np.median(v),
                                     'lo': np.percentile(v, 2.5), 'hi': np.percentile(v, 97.5)})
    pd.DataFrame(rows).to_csv(OUT + 'A4_probabilistic.csv', index=False)
    pd.DataFrame(heat).to_csv(OUT + 'A4_heatmap_grid.csv', index=False)
    json.dump(res, open(OUT + 'A4_tipping.json', 'w'), indent=1, default=float)

if __name__ == '__main__':
    main()
