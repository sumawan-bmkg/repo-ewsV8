#!/usr/bin/env python3
"""EVIDENCE 11: Comparison — Print-ready white theme"""
import os, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT = 'D:/multi/scalogramv3/disertasi4/supcon/11_comparison'
os.makedirs(OUT, exist_ok=True)

fig, axes = plt.subplots(1, 2, figsize=(18, 8))
fig.patch.set_facecolor('white')

# Bar
ax = axes[0]; ax.set_facecolor('white')
models = ['V3 Baseline\n— N/A —', 'V8 Conv\n— N/A —', 'V8 SupCon']
metrics = ['Recall','Precision','FPR','F2','Az MAE (°)']
vals = {
    'V8 SupCon': [0.986, 0.886, 0.125, 0.954, 65.2/100],
}
x = np.arange(len(metrics)); width = 0.25
for i, (m, vs) in enumerate(vals.items()):
    bars = ax.bar(x + 2*width, vs, width, label=m, color='#16A085', edgecolor='#333', lw=0.5)
    for bar, v in zip(bars, vs):
        v_disp = v if i != 4 else v*100
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                f'{v:.3f}', ha='center', fontsize=7, color='#000000')
# N/A bars
for i in range(2):
    ax.bar(x + i*width - width*0.5, [0]*5, width, color='#DDD', alpha=0.5, label=models[i])
    for j in range(len(metrics)):
        ax.annotate('N/A', xy=(x[j]+i*width-width*0.5, 0.02), ha='center', fontsize=7, color='#888')
ax.set_xticks(x + width/2); ax.set_xticklabels(metrics, fontsize=9)
ax.set_ylabel('Score', fontsize=10, color='#000000')
ax.set_title('Model Comparison', fontsize=13, fontweight='bold', color='#000000')
ax.legend(fontsize=8); ax.tick_params(colors='#000000')
ax.set_ylim(0, 1.2)

# Radar
categories = ['Recall', 'Precision', 'F2', 'AUPRC', 'Az MAE\n(1-inv)', 'FPR\n(1-)']
values = [0.986, 0.886, 0.954, 0.985, 0.348, 0.875]
angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
values += values[:1]; angles += angles[:1]
ax = fig.add_subplot(1, 2, 2, polar=True, facecolor='white')
ax.plot(angles, values, 'o-', lw=2, color='#16A085', label='V8 SupCon')
ax.fill(angles, values, alpha=0.25, color='#16A085')
ax.set_xticks(angles[:-1]); ax.set_xticklabels(categories, fontsize=9, color='#000000')
ax.set_yticklabels([]); ax.set_ylim(0, 1)
ax.set_title('V8 SupCon — Radar', fontsize=13, fontweight='bold', color='#000000', pad=20)
ax.legend(loc='upper right', fontsize=9)

fig.suptitle('V8 SupCon — Performance Comparison', fontsize=14, fontweight='bold', color='#000000', y=1.01)
plt.tight_layout()
for fmt in ['png', 'svg', 'pdf']:
    fig.savefig(f'{OUT}/comparison_chart.{fmt}', dpi=300, bbox_inches='tight', facecolor='white')
plt.close(); print(f'[OK] Evidence 11 saved to {OUT}/')
