#!/usr/bin/env python3
"""EVIDENCE 7: Latent Space — Print-ready white theme"""
import os, matplotlib, warnings
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
warnings.filterwarnings('ignore')

OUT = 'D:/multi/scalogramv3/disertasi4/supcon/07_latent'
os.makedirs(OUT, exist_ok=True)
emb = pd.read_csv('D:/multi/scalogramv3/eswa/tsne_embedding_v8.csv')

fig, axes = plt.subplots(1, 3, figsize=(22, 7))
fig.patch.set_facecolor('white')

ax = axes[0]; ax.set_facecolor('white')
colors = ['#2980B9', '#C0392B']
for label in [0, 1]:
    mask = emb['label'] == label
    ax.scatter(emb.loc[mask,'tsne_x'], emb.loc[mask,'tsne_y'],
               c=colors[label], label=f'Class {label}', alpha=0.6, s=15)
ax.set_title('t-SNE Embedding (by Class)', fontsize=12, fontweight='bold', color='#000000')
ax.legend(fontsize=10); ax.tick_params(colors='#000000')
ax.set_xlabel('t-SNE Dim 1', fontsize=10, color='#000000')
ax.set_ylabel('t-SNE Dim 2', fontsize=10, color='#000000')

ax = axes[1]; ax.set_facecolor('white')
hb = ax.hexbin(emb['tsne_x'], emb['tsne_y'], gridsize=30, cmap='viridis', alpha=0.8)
ax.set_title('t-SNE Density Map', fontsize=12, fontweight='bold', color='#000000')
ax.tick_params(colors='#000000')
ax.set_xlabel('t-SNE Dim 1', fontsize=10, color='#000000')
ax.set_ylabel('t-SNE Dim 2', fontsize=10, color='#000000')
cb = plt.colorbar(hb, ax=ax, shrink=0.7)
cb.set_label('Density', fontsize=9); cb.ax.tick_params(colors='#000000')

ax = axes[2]; ax.axis('off')
n_pos, n_neg = int((emb['label']==1).sum()), int((emb['label']==0).sum())
txt = (f"Embedding Analysis\n\nSource: eswa/tsne_embedding_v8.csv\n"
       f"Total points: {len(emb)}\nClass 0: {n_neg} ({n_neg/len(emb)*100:.1f}%)\n"
       f"Class 1: {n_pos} ({n_pos/len(emb)*100:.1f}%)\n\n"
       "SupCon projection head (128-dim)\nPre-trained val set (500 samples)\nBalanced: 250 each")
ax.text(0.1, 0.9, txt, fontsize=10, color='#000000', fontfamily='monospace', va='top')

fig.suptitle('V8 SupCon — Latent Space (t-SNE from SupCon projection)',
             fontsize=14, fontweight='bold', color='#000000', y=1.02)
plt.tight_layout()
for fmt in ['png', 'svg', 'pdf']:
    fig.savefig(f'{OUT}/latent_space_embedding.{fmt}', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f'[OK] Evidence 7 saved to {OUT}/')
