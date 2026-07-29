#!/usr/bin/env python3
"""EVIDENCE 8: Spatial GNN — NOT AVAILABLE notice, white theme"""
import os, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = 'D:/multi/scalogramv3/disertasi4/supcon/08_spatial'
os.makedirs(OUT, exist_ok=True)
fig, ax = plt.subplots(1, 1, figsize=(12, 6))
fig.patch.set_facecolor('white'); ax.set_facecolor('white')
ax.axis('off')
msg = ("NOT AVAILABLE\n\n"
       "GNN attention weights (att_weights) are ephemeral forward() outputs.\n"
       "No saved attention/edge tensors found on disk.\n\n"
       "Available without GNN re-extraction:\n"
       "  - Station-level predictions: 2880 rows\n"
       "  - Azimuth quadrant CM from v8_confusion_matrices.txt\n"
       "  - Station distribution in E2\n\nTo generate: Modify inference to capture att_weights per sample.")
ax.text(0.5, 0.5, msg, ha='center', va='center', fontsize=12, color='#C0392B',
        fontfamily='monospace', transform=ax.transAxes,
        bbox=dict(boxstyle='round,pad=1', facecolor='#FAFAFA', edgecolor='#C0392B', linewidth=2))
ax.set_title('Evidence 8: Spatial GNN — NOT AVAILABLE', fontsize=14, fontweight='bold', color='#000000')
for fmt in ['png', 'svg', 'pdf']:
    fig.savefig(f'{OUT}/spatial_gnn_attention.{fmt}', dpi=300, bbox_inches='tight', facecolor='white')
plt.close(); print(f'[OK] Evidence 8 saved to {OUT}/')
