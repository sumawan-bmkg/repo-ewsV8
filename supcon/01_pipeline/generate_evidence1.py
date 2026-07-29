#!/usr/bin/env python3
"""EVIDENCE 1: Pipeline Preprocessing — Print-ready white theme, 300 DPI"""
import os, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

OUT = 'D:/multi/scalogramv3/disertasi4/supcon/01_pipeline'
os.makedirs(OUT, exist_ok=True)

stages = [
    ("Raw ULF\n(1 Hz, 3-ch)",       "Hx, Hy, Hz\nmagnetometer"),
    ("Baseline\nRemoval",           "Subtract daily\nmedian/savgol"),
    ("Spike\nRemoval",              "Hampel filter\n3σ threshold"),
    ("Bandpass\nPc3–Pc4",           "20–100 s\nButterworth 4-pole"),
    ("Normalization",               "Z-score\nper channel"),
    ("CWT\nScalogram",              "Morlet wavelet\n128 freq × 1440 time"),
    ("COI Mask",                    "Cone of Influence\nedge removal"),
    ("Tensor\nBuild",               "[3, 128, 1440]\nhalf-precision"),
]

fig, ax = plt.subplots(1, 1, figsize=(16, 5))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')
ax.set_xlim(0, 10)
ax.set_ylim(0, 4)
ax.axis('off')

for i, (name, desc) in enumerate(stages):
    x = i * 1.2 + 0.3
    y = 2.0
    box = mpatches.FancyBboxPatch(
        (x - 0.45, y - 0.55), 0.9, 0.6,
        boxstyle="round,pad=0.08",
        facecolor='#4A7FB5' if i % 2 == 0 else '#5A8FC5',
        edgecolor='#2C3E50', linewidth=2, zorder=3)
    ax.add_patch(box)
    sub = mpatches.FancyBboxPatch(
        (x - 0.45, y - 1.05), 0.9, 0.4,
        boxstyle="round,pad=0.05",
        facecolor='#E8F0F5', edgecolor='#999999', linewidth=1, zorder=2)
    ax.add_patch(sub)
    ax.text(x, y + 0.02, name, ha='center', va='center',
            fontsize=8, fontweight='bold', color='white', zorder=4)
    ax.text(x, y - 0.85, desc, ha='center', va='center',
            fontsize=6.5, color='#555555', zorder=4)

for i in range(len(stages) - 1):
    x1 = i * 1.2 + 0.75
    x2 = (i + 1) * 1.2 + 0.3
    ax.annotate('', xy=(x2, 2.0), xytext=(x1, 2.0),
                arrowprops=dict(arrowstyle='->', color='#333333', lw=2.5))

ax.set_title('V8 SupCon — Preprocessing Pipeline',
             fontsize=14, fontweight='bold', color='#000000', pad=20)

for fmt in ['png', 'svg', 'pdf']:
    fig.savefig(f'{OUT}/pipeline_diagram.{fmt}', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f'[OK] Evidence 1 saved to {OUT}/')
