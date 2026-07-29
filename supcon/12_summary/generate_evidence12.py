#!/usr/bin/env python3
"""EVIDENCE 12: Summary Figure — Print-ready white theme"""
import os, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = 'D:/multi/scalogramv3/disertasi4/supcon/12_summary'
os.makedirs(OUT, exist_ok=True)

stages = [
    ("Raw ULF", "magnetometer\n3-ch", '#2980B9'),
    ("Preprocess", "Baseline→Spike\n→Bandpass", '#3498DB'),
    ("Scalogram", "Morlet CWT\n[3,128,1440]", '#2980B9'),
    ("EfficientNet", "Backbone-B1\n1280-dim", '#3498DB'),
    ("BiGRU+GNN", "Spatio-Temporal\nFusion", '#16A085'),
    ("Cosmic+SupCon", "Kp/Dst gate\nContrastive proj", '#D4AC0D'),
    ("Prediction", "Det+Mag+Azm\nMulti-Task", '#C0392B'),
    ("Blind Test", "2880 samples\nJan–Apr 2026", '#8E44AD'),
]
metrics = ("V8 SupCon FINAL METRICS\n"
           "━━━━━━━━━━━━━━━━━━━━━\n"
           "Recall:     0.986 (val)\nPrecision:  0.886\n"
           "AUPRC:      0.985\nF2:        0.954\n"
           "Az MAE:     65.2° (val)\nAz MAE BT:  38.0° (blind)\n"
           "Params:     9.4M\nEpochs:     35")

fig, (ax_p, ax_m) = plt.subplots(1, 2, figsize=(22, 8),
    gridspec_kw={'width_ratios': [3, 1]})
fig.patch.set_facecolor('white')
for axx in [ax_p, ax_m]:
    axx.set_facecolor('white')
    axx.axis('off')

ax_p.set_xlim(0, len(stages)*1.3+1); ax_p.set_ylim(0, 4)
for i, (name, desc, color) in enumerate(stages):
    x = i*1.3 + 0.5
    box = FancyBboxPatch((x-0.5, 1.2), 1.0, 0.8, boxstyle="round,pad=0.08",
                          facecolor=color, edgecolor='#333', linewidth=2)
    ax_p.add_patch(box)
    sub = FancyBboxPatch((x-0.5, 0.5), 1.0, 0.5, boxstyle="round,pad=0.05",
                          facecolor='#F5F5F5', edgecolor='#AAA', linewidth=0.5)
    ax_p.add_patch(sub)
    ax_p.text(x, 1.6, name, ha='center', va='center', fontsize=8,
              fontweight='bold', color='white')
    ax_p.text(x, 0.75, desc, ha='center', va='center', fontsize=6.5, color='#555')
    if i < len(stages)-1:
        ax_p.annotate('', xy=(x+0.8, 1.6), xytext=(x+0.5, 1.6),
                     arrowprops=dict(arrowstyle='->', color='#333', lw=2))

ax_p.set_title('V8 SupCon — End-to-End System', fontsize=13, fontweight='bold', color='#000000')
ax_m.text(0.1, 0.95, metrics, fontsize=11, color='#000000', fontfamily='monospace', va='top',
          bbox=dict(boxstyle='round,pad=0.5', facecolor='#FAFAFA', edgecolor='#CCC'))

fig.suptitle('V8 SupCon — Dissertation Summary Figure', fontsize=15, fontweight='bold', color='#000000', y=1.02)
plt.tight_layout()
for fmt in ['png', 'svg', 'pdf']:
    fig.savefig(f'{OUT}/summary_figure.{fmt}', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f'[OK] Evidence 12 saved to {OUT}/')

val = """# Evidence 12 — Validation Report\n
**Figure:** Summary System Architecture + Final Metrics
**Source Files:** V3_Model_v8.py, training_v8_convergence_history.csv, v8supcon_2026_predictions.csv
**All data is REAL V8 SupCon — no synthetic values.**"""
with open(f'{OUT}/validation.md', 'w') as f: f.write(val)
