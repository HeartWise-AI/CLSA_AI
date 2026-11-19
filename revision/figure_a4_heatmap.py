"""Supplementary figure: A4 probabilistic bias analysis heat map (CLSA test set). Median delta AUROC (Model A+ minus FRS)
over fixed PPV of the heart-disease item x exp(alpha1); panels for exp(beta1) = 1 and 0.5. Dotted cells: 95% simulation interval includes 0."""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
ROOT = Path(__import__('os').environ.get('CLSA_WORKSPACE', '/volume/CLSA')); OUT = ROOT / 'outputs/revision'
g = pd.read_csv(OUT / 'outcome_validity/A4_heatmap_grid.csv'); g = g[g['sample'] == 'test']
INK, INK2 = '#0b0b0b', '#52514e'
cmap = LinearSegmentedColormap.from_list('div', ['#c4441c', '#f2c9b4', '#ecebe7', '#a9c8ef', '#1f5fae'])
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.edgecolor': INK2, 'xtick.color': INK2, 'ytick.color': INK})
lim = max(0.05, g.median_delta.abs().max()); norm = TwoSlopeNorm(0, -lim, lim)
fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), sharey=True)
A1 = [1, 1.5, 2, 3]; P = sorted(g.ppv.unique(), reverse=True)
for ax, b1, title in zip(axes, [1, 0.5], ['A  Non-differential missed cases (exp(β₁) = 1)', 'B  Missed cases concentrated at low scores (exp(β₁) = 0.5)']):
    h = g[g.exp_b1 == b1]
    M = np.array([[h[(h.ppv == p) & (h.exp_a1 == a)].median_delta.iloc[0] for a in A1] for p in P])
    L = np.array([[h[(h.ppv == p) & (h.exp_a1 == a)].lo.iloc[0] for a in A1] for p in P])
    im = ax.imshow(M, cmap=cmap, norm=norm, aspect='auto')
    for i in range(len(P)):
        for j in range(len(A1)):
            ax.text(j, i, f'{M[i, j]:+.3f}'.replace('-', '−') + ('' if L[i, j] > 0 else '*'), ha='center', va='center', fontsize=8, color=INK)
            ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False, ec='white', lw=2))
    ax.set_xticks(range(len(A1))); ax.set_xticklabels([f'{a:g}' for a in A1])
    ax.set_xlabel('exp(α₁): false-positive reports concentrated\nat high Model A+ scores (1 = non-differential)')
    ax.set_title(title, loc='left', fontsize=10, color=INK, fontweight='bold')
    for sp in ax.spines.values(): sp.set_visible(False)
axes[1].tick_params(axis='y', length=0); axes[0].set_yticks(range(len(P))); axes[0].set_yticklabels([f'{p:.2f}' for p in P]); axes[0].set_ylabel('PPV of the self-reported heart-disease item')
cb = fig.colorbar(im, ax=axes, fraction=.035, pad=.02); cb.set_label('Median ΔAUROC, Model A+ minus FRS'); cb.outline.set_visible(False)
fig.text(.01, -.04, '* 95% simulation interval includes 0. 1,000 iterations per cell; each iteration = misclassification draw + participant bootstrap.', fontsize=7.5, color=INK2)
out = OUT / 'Figure_A4_bias_heatmap.png'; plt.savefig(out, dpi=300, facecolor='white', bbox_inches='tight'); plt.savefig(out.with_suffix('.pdf'), facecolor='white', bbox_inches='tight'); print(out)
