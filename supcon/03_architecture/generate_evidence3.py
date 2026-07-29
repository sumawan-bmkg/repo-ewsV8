#!/usr/bin/env python3
"""EVIDENCE 3: Architecture — Print-ready white theme, 300 DPI"""
import os, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = 'D:/multi/scalogramv3/disertasi4/supcon/03_architecture'
os.makedirs(OUT, exist_ok=True)

blocks = [
    ("Input Tensor\n[3, 128, 1440]", "3-ch scalogram\nhalf-precision", '#2980B9'),
    ("EfficientNet-B1\nBackbone", "1280-dim feature\nmap", '#2980B9'),
    ("Adaptive Pool\n+ GRU Proj", "AvgPool2D(1,None)\n1280→256", '#3498DB'),
    ("BiGRU\n(2 layers)", "256 hidden, bidir\n512-dim output", '#3498DB'),
    ("Spatial GNN\n(8 virtual stn)", "Multi-head attn\n512→512", '#16A085'),
    ("Cosmic Gate\n(MLP)", "Kp/Dst→gating\n512→512 (sigmoid)", '#D4AC0D'),
    ("v_fusion\n[512]", "Gated fusion vec\nshared rep", '#CA6F1E'),
]
heads = [
    ("Detection\nHead", "Linear→128→2\n(softmax logit)", '#C0392B'),
    ("Magnitude\nHead", "Linear→128→5\n(Mw class)", '#2980B9'),
    ("Azimuth\nHead", "Linear→128→2\n[sin, cos] unit", '#8E44AD'),
    ("SupCon\nProjection", "128→ReLU→128\nL2-norm unit", '#16A085'),
]

fig, (ax_main, ax_h) = plt.subplots(1, 2, figsize=(18, 10),
    gridspec_kw={'width_ratios': [2, 1]})
fig.patch.set_facecolor('white')
for axx in [ax_main, ax_h]:
    axx.set_facecolor('white')
    axx.axis('off')

ax_main.set_xlim(0, 5)
ax_main.set_ylim(-1, len(blocks) + 1)
ax_h.set_xlim(0, 3)
ax_h.set_ylim(-1, len(heads) + 1)

for i, (name, desc, color) in enumerate(blocks):
    y = i + 0.5
    box = FancyBboxPatch((0.5, y-0.4), 4, 0.8, boxstyle="round,pad=0.1",
                          facecolor=color, edgecolor='#333333', linewidth=2)
    ax_main.add_patch(box)
    ax_main.text(2.5, y+0.1, name, ha='center', va='center', fontsize=9,
                 fontweight='bold', color='white')
    ax_main.text(2.5, y-0.25, desc, ha='center', va='center', fontsize=7, color='#555555')
    if i < len(blocks)-1:
        ax_main.annotate('', xy=(2.5, i+0.9), xytext=(2.5, i+0.1),
                         arrowprops=dict(arrowstyle='->', color='#333333', lw=2))

for i, (name, desc, color) in enumerate(heads):
    y = i + 0.5
    box = FancyBboxPatch((0.3, y-0.4), 2.4, 0.8, boxstyle="round,pad=0.1",
                          facecolor=color, edgecolor='#333333', linewidth=2)
    ax_h.add_patch(box)
    ax_h.text(1.5, y+0.1, name, ha='center', va='center', fontsize=9,
              fontweight='bold', color='white')
    ax_h.text(1.5, y-0.25, desc, ha='center', va='center', fontsize=7, color='#555555')

ax_main.text(2.5, -0.5, 'Total Parameters: ~9.4M', ha='center', fontsize=11,
             fontweight='bold', color='#2C3E50', style='italic')
ax_main.text(2.5, -0.8, 'Loss: L_det(L1+Focal) + L_azm(SineCosine) + λ·L_supcon(SupCon)',
             ha='center', fontsize=8, color='#666666', style='italic')

fig.suptitle('V8 SupCon — Model Architecture (MultiTaskScalogramV3_v8)',
             fontsize=14, fontweight='bold', color='#000000', y=0.95)
for fmt in ['png', 'svg', 'pdf']:
    fig.savefig(f'{OUT}/architecture_diagram.{fmt}', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f'[OK] Evidence 3 saved to {OUT}/')
