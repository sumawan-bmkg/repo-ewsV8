#!/usr/bin/env python3
"""EVIDENCE 5: Ablation — NOT AVAILABLE notice, white theme, 300 DPI"""
import os, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = 'D:/multi/scalogramv3/disertasi4/supcon/05_ablation'
os.makedirs(OUT, exist_ok=True)

fig, ax = plt.subplots(1, 1, figsize=(12, 6))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')
ax.axis('off')
msg = ("NOT AVAILABLE\n\n"
       "Controlled ablation study requires systematic re-training\n"
       "with per-component removal. Current V8 SupCon training\n"
       "was a single consolidated run with all components active.\n\n"
       "Available component-level evidence:\n"
       "   S2 SinCos Azimuth: 67° MAE improvement\n"
       "   S3 Balanced Sampling: 27% → 49% pos ratio\n"
       "   S11 Dynamic Threshold: 0.0 → 1.0 recall\n"
       "   S12 Label Smoothing: gradient 1000× reduced")
ax.text(0.5, 0.5, msg, ha='center', va='center', fontsize=12, color='#C0392B',
        fontfamily='monospace', transform=ax.transAxes,
        bbox=dict(boxstyle='round,pad=1', facecolor='#FAFAFA', edgecolor='#C0392B', linewidth=2))
ax.set_title('Evidence 5: Ablation Study — NOT AVAILABLE', fontsize=14, fontweight='bold', color='#000000')
for fmt in ['png', 'svg', 'pdf']:
    fig.savefig(f'{OUT}/ablation_waterfall.{fmt}', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f'[OK] Evidence 5 saved to {OUT}/')
val = """# Evidence 5 — Validation Report\n\n**Figure:** Waterfall Chart (NOT AVAILABLE)\n\n**Reason:** Controlled ablation not conducted for V8 SupCon.\n"""
with open(f'{OUT}/validation.md', 'w') as f: f.write(val)
