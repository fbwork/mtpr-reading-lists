"""Regenerates Reports/fig5_bias_vs_transition.png (Fig. 2) from the per-query CSC file."""
import csv, collections
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

rows = list(csv.DictReader(open('components/resources/gs/csc_per_query.csv')))
by = collections.defaultdict(list)
for r in rows:
    by[r['mode']].append(float(r['ocsc']))
mean = {m: float(np.mean(v)) for m, v in by.items()}
fams = [('Author', 'modified', 'modified_bias', 'modified_transition'),
        ('Degree', 'd_modified', 'd_modified_bias', 'd_modified_transition'),
        ('Closeness', 'cl_modified', 'cl_modified_bias', 'cl_modified_transition'),
        ('Betweenness', 'bw_modified', 'bw_modified_bias', 'bw_modified_transition'),
        ('Katz', 'kz_modified', 'kz_modified_bias', 'kz_modified_transition')]
fig, ax = plt.subplots(figsize=(9.6, 6))
x = np.arange(len(fams)); w = 0.26
cols = ['#a9c4e0', '#2f5f8f', '#c9a9a6']
labels = ['Weighted bias + baseline $T^{*}$', 'Weighted bias + uniform transition', 'Weighted transition']
for j in range(3):
    vals = [mean[f[1 + j]] for f in fams]
    bars = ax.bar(x + (j - 1) * w, vals, w, color=cols[j], edgecolor='black', linewidth=1.0, label=labels[j])
    for b_, v in zip(bars, vals):
        ax.text(b_.get_x() + b_.get_width() / 2, v + 0.008, f'{v:.3f}', ha='center', va='bottom', fontsize=10)
ax.set_xticks(x); ax.set_xticklabels([f[0] for f in fams], fontsize=14)
ax.set_ylabel('Mean OCSC', fontsize=14); ax.set_ylim(0, 0.75)
ax.yaxis.grid(True, linestyle=':', color='#bbbbbb'); ax.set_axisbelow(True)
for s in ('top', 'right'):
    ax.spines[s].set_visible(False)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=3, frameon=False, fontsize=12)
fig.tight_layout()
fig.savefig('Reports/fig5_bias_vs_transition.png', dpi=200)
print({f[0]: [round(mean[f[i]], 3) for i in (1, 2, 3)] for f in fams})
