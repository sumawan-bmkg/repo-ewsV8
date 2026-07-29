#!/usr/bin/env python3
"""EVIDENCE 4: Training History — Print-ready white theme, 300 DPI"""
import os, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

OUT = 'D:/multi/scalogramv3/disertasi4/supcon/04_training'
os.makedirs(OUT, exist_ok=True)
df = pd.read_csv('D:/multi/scalogramv3/pull_real/logs/training_v8_convergence_history.csv')

panels = [
    ['train_loss', 'val_loss'],
    ['precision', 'recall', 'f2'],
    ['fpr', 'ews_score'],
    ['brier', 'azm_mae'],
    ['lr'],
]

fig, axes = plt.subplots(len(panels), 1, figsize=(16, 3*len(panels)))
fig.patch.set_facecolor('white')

for idx, cols in enumerate(panels):
    ax = axes[idx]
    ax.set_facecolor('white')
    for col in cols:
        data = df[col].values
        if col == 'lr':
            label = 'LR × 1e5'
            data = data * 1e5
        else:
            label = col.replace('_', ' ').title()
        ax.plot(df['epoch'], data, label=label, linewidth=1.5, marker='o', markersize=3, alpha=0.85)
    ax.set_title(f'Training History — {" vs ".join(cols)}', fontsize=11, fontweight='bold', color='#000000')
    ax.set_xlabel('Epoch', fontsize=9, color='#000000')
    ax.legend(fontsize=8, loc='best')
    ax.tick_params(colors='#000000', labelsize=8)
    ax.spines['bottom'].set_color('#CCCCCC')
    ax.spines['left'].set_color('#CCCCCC')
    ax.grid(True, alpha=0.15, color='#000000')
    ax.set_ylabel('Value', fontsize=9, color='#000000')

fig.suptitle('V8 SupCon — Training Convergence (35 epochs)',
             fontsize=14, fontweight='bold', color='#000000', y=1.01)
plt.tight_layout()
for fmt in ['png', 'svg', 'pdf']:
    fig.savefig(f'{OUT}/training_history.{fmt}', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f'[OK] Evidence 4 saved to {OUT}/')
