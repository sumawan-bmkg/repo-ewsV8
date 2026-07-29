#!/usr/bin/env python3
"""EVIDENCE 10: XAI GradCAM — NOT AVAILABLE notice, white theme"""
import os, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = 'D:/multi/scalogramv3/disertasi4/supcon/10_xai'
os.makedirs(OUT, exist_ok=True)
fig, ax = plt.subplots(1, 1, figsize=(12, 6))
fig.patch.set_facecolor('white'); ax.set_facecolor('white')
ax.axis('off')
msg = ("NOT AVAILABLE\n\n"
       "GradCAM activation maps require model re-execution with\n"
       "pytorch_grad_cam hooked to EfficientNet-B1 backbone.\n\n"
       "No pre-computed activation maps on disk.\n\n"
       "Available for future generation:\n"
       "  - Checkpoint: checkpoints/v3_v8_conv_fpr_best_weights.pth (38MB)\n"
       "  - GradCAM lib in venv_cuda_310, venv_cuda_311\n"
       "  - HDF5 samples in 2026/scalogram/\n\n"
       "To generate: Run model with GradCAM on backbone.features layer.")
ax.text(0.5, 0.5, msg, ha='center', va='center', fontsize=12, color='#C0392B',
        fontfamily='monospace', transform=ax.transAxes,
        bbox=dict(boxstyle='round,pad=1', facecolor='#FAFAFA', edgecolor='#C0392B', linewidth=2))
ax.set_title('Evidence 10: XAI / GradCAM — NOT AVAILABLE', fontsize=14, fontweight='bold', color='#000000')
for fmt in ['png', 'svg', 'pdf']:
    fig.savefig(f'{OUT}/xai_gradcam.{fmt}', dpi=300, bbox_inches='tight', facecolor='white')
plt.close(); print(f'[OK] Evidence 10 saved to {OUT}/')
