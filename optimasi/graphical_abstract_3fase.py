#!/usr/bin/env python3
"""
graphical_abstract_3fase.py — Graphical Abstract perbandingan Blind Test
sebelum dan sesudah optimasi 3 Fase.

Output: fig_0_graphical_abstract.png (600 DPI, format jurnal)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import json
from pathlib import Path

# ─── Muat metrik ───────────────────────────────────────────────
METRICS_PATH = Path("D:/multi/scalogramv3/disertasi4/DISERTASI_BLINDTEST_OPTIMIZED_EVIDENCE/logs/blindtest_optimized_metrics.json")
OUTPUT_DIR   = Path("D:/multi/scalogramv3/disertasi4/DISERTASI_BLINDTEST_OPTIMIZED_EVIDENCE/plots/")

with open(METRICS_PATH) as f:
    raw = json.load(f)

def get(key):
    return raw[key]

A = get("A_Baseline_(Mentah)")
B = get("B_Fase_1_(Temp._Scaling)")
C = get("C_Fase_2/3_(Focal_Loss)")

# ─── Palet warna ilmiah (Nature Communications style) ─────────
BASELINE_CLR = '#2166AC'    # biru tua — sebelum
PHASE1_CLR   = '#F4A582'    # salmon — transisi
PHASE2_CLR   = '#B2182B'    # merah tua — sesudah
BG_CLR       = '#FAFAFA'
TEXT_CLR     = '#1A1A1A'
GOLD         = '#D4AF37'

# ─── Setup canvas ──────────────────────────────────────────────
fig = plt.figure(figsize=(16, 7), facecolor='white')
fig.patch.set_facecolor('white')

# Grid: 3 kolom utama (A, B, C) + 1 kolom panah
gs = fig.add_gridspec(2, 7, hspace=0.35, wspace=0.15,
                       left=0.04, right=0.97, top=0.88, bottom=0.08)

# ─── JUDUL ─────────────────────────────────────────────────────
fig.suptitle('Graphical Abstract — Optimasi 3 Fase Blind Test 2026',
             fontsize=16, fontweight='bold', color=TEXT_CLR, y=0.96,
             fontfamily='sans-serif')
fig.text(0.5, 0.925, 'EfficientNet-B1 SupCon | Temperature Scaling + Focal Loss + Decoupled Training',
         ha='center', fontsize=10, color='#555555', fontstyle='italic')

# ─── Panel helper ──────────────────────────────────────────────
def probability_strip(ax, probs, color, label, mean_val, n_pos, n_neg, tp, fn, fp, tn, recall, f2):
    """Probability distribution strip + confusion callout."""
    # Density strip
    y_jitter = np.random.default_rng(42).uniform(-0.15, 0.15, len(probs))
    ax.scatter(probs, y_jitter, s=2, alpha=0.15, color=color, edgecolors='none', zorder=2)
    
    # Mean line
    ax.axvline(mean_val, ymin=0.15, ymax=0.85, color=color, linewidth=3,
               linestyle='-', zorder=5)
    ax.text(mean_val, 0.92, r'$\mu$=' + f'{mean_val:.3f}',
            ha='center', fontsize=8, color=color, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.15', facecolor='white', alpha=0.7))
    
    # Threshold diamond
    thresh = raw[label.replace(' ', '_').replace(':', '')]['threshold']
    ax.scatter([thresh], [0.55], marker='D', s=80, color=GOLD, edgecolors='black',
               linewidth=1.2, zorder=10)
    ax.text(thresh, 0.65, r'$\phi$=' + f'{thresh:.3f}', ha='center', fontsize=7,
            color=GOLD, fontweight='bold')
    
    # Confusion callout — small icons
    ax.text(0.02, 0.45, f'TP={tp}  FN={fn}\nFP={fp}  TN={tn}',
            transform=ax.transAxes, fontsize=7.5, color=TEXT_CLR,
            fontfamily='monospace', va='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    # Recall + F2 big numbers
    ax.text(0.98, 0.08, f'Recall\n{recall:.1f}%',
            transform=ax.transAxes, fontsize=11, fontweight='bold',
            ha='right', va='bottom', color=color)
    ax.text(0.98, 0.40, f'F2\n{f2:.2f}',
            transform=ax.transAxes, fontsize=10, fontweight='bold',
            ha='right', va='top', color=color)
    
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.25, 1.0)
    ax.set_xlabel('Probabilitas', fontsize=8)
    ax.set_yticks([])
    ax.spines[['left', 'top', 'right']].set_visible(False)
    ax.tick_params(labelsize=7)
    ax.set_facecolor(BG_CLR)


def violation_plot(ax, y_true, probs_before, probs_after, color_before, color_after):
    """Per-sample before vs after line plot."""
    idx = np.argsort(y_true * 1000 + probs_before)[::3]
    idx = idx[:200]
    
    for i in idx:
        ax.plot([0, 1], [probs_before[i], probs_after[i]],
                color='#CCCCCC', linewidth=0.4, alpha=0.5)
    
    ax.scatter(np.zeros(len(idx)), probs_before[idx], s=3, color=color_before, alpha=0.4, zorder=5)
    ax.scatter(np.ones(len(idx)), probs_after[idx], s=3, color=color_after, alpha=0.4, zorder=5)
    
    # Mean arrows
    mean_before = probs_before.mean()
    mean_after  = probs_after.mean()
    ax.plot([0, 1], [mean_before, mean_after], color='#1A1A1A',
            linewidth=2, linestyle='--', zorder=6)
    ax.annotate(f'{mean_before:.3f}', (0, mean_before), fontsize=7,
                ha='right', va='bottom', color=color_before, fontweight='bold')
    ax.annotate(f'{mean_after:.3f}', (1, mean_after), fontsize=7,
                ha='left', va='bottom', color=color_after, fontweight='bold')
    
    ax.set_xlim(-0.3, 1.3)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Before', 'After'])
    ax.set_ylabel('Probabilitas', fontsize=8)
    ax.set_ylim(-0.05, 1.05)
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(labelsize=7)
    ax.set_facecolor(BG_CLR)


def metric_card(ax, label, recall, precision, f2, fpr, fnr, color, x_shift=0):
    """Single metric card."""
    ax.text(0.5, 0.85, label, ha='center', fontsize=11, fontweight='bold',
            color=color, transform=ax.transAxes)
    
    metrics = [
        ('Recall ↑', recall/100, recall),
        ('Precision', precision/100, precision),
        ('F2-Score ↑', f2, f2*100),
        ('FPR ↓', 1-fpr/100, fpr),
    ]
    
    for i, (name, ratio, val) in enumerate(metrics):
        y = 0.70 - i*0.16
        ax.barh(y, ratio, height=0.08, color=color, alpha=0.7, edgecolor=color, linewidth=0.5)
        val_str = f'{val:.1f}' if 'Recall' in name or 'Precision' in name else f'{val:.2f}'
        if 'FPR' in name: val_str = f'{val:.1f}%'
        ax.text(ratio + 0.02, y, f'{name}: {val_str}', fontsize=8, va='center',
                transform=ax.transAxes)
    
    ax.set_xlim(0, 1.4)
    ax.set_ylim(-0.05, 0.9)
    ax.axis('off')


# ─── Muat data ─────────────────────────────────────────────────
import pandas as pd
CSV_PATH = Path("D:/multi/scalogramv3/disertasi4/DISERTASI_BLINDTEST_OPTIMIZED_EVIDENCE/data/predictions_2026_optimized_all_phases.csv")
df = pd.read_csv(CSV_PATH)
y_true = df['True_Label'].values
p_baseline = df['Prob_Baseline'].values
p_phase1   = df['Prob_Phase1_TempScaling'].values
p_phase2   = df['Prob_Phase2_FocalLoss'].values

# ─── Panel 1: Baseline ─────────────────────────────────────────
ax1 = fig.add_subplot(gs[:, 0])
ax1.set_title('A: Baseline (Mentah)', fontsize=10, fontweight='bold', pad=5, color=BASELINE_CLR)
probability_strip(ax1, p_baseline, BASELINE_CLR, 'A_Baseline_(Mentah)',
                  A['mean_prob'], A['n_pos'], A['n_neg'],
                  A['tp'], A['fn'], A['fp'], A['tn'],
                  A['recall']*100, A['f2_score'])

# ─── Arrow 1 -> 2 ──────────────────────────────────────────────
ax_arrow1 = fig.add_subplot(gs[:, 1])
ax_arrow1.axis('off')
ax_arrow1.annotate('', xy=(0.9, 0.5), xytext=(0.1, 0.5),
                   arrowprops=dict(arrowstyle='->', lw=2.5, color='#888888'),
                   transform=ax_arrow1.transAxes)
ax_arrow1.text(0.5, 0.88, 'Fase 1:\nTemp. Scaling', ha='center', fontsize=8,
               color='#888888', fontweight='bold', linespacing=1.3)
ax_arrow1.text(0.5, 0.12, 'T = 6.3', ha='center', fontsize=9,
               color=PHASE1_CLR, fontweight='bold')

# ─── Panel 2: Fase 1 ───────────────────────────────────────────
ax2 = fig.add_subplot(gs[:, 2])
ax2.set_title('B: Fase 1 (T = 6.3)', fontsize=10, fontweight='bold', pad=5, color=PHASE1_CLR)
probability_strip(ax2, p_phase1, PHASE1_CLR, 'B_Fase_1_(Temp._Scaling)',
                  B['mean_prob'], B['n_pos'], B['n_neg'],
                  B['tp'], B['fn'], B['fp'], B['tn'],
                  B['recall']*100, B['f2_score'])

# ─── Arrow 2 -> 3 ──────────────────────────────────────────────
ax_arrow2 = fig.add_subplot(gs[:, 3])
ax_arrow2.axis('off')
ax_arrow2.annotate('', xy=(0.9, 0.5), xytext=(0.1, 0.5),
                   arrowprops=dict(arrowstyle='->', lw=2.5, color='#888888'),
                   transform=ax_arrow2.transAxes)
ax_arrow2.text(0.5, 0.88, 'Fase 2/3:\nFocal + Decoupled', ha='center', fontsize=8,
               color='#888888', fontweight='bold', linespacing=1.3)
ax_arrow2.text(0.5, 0.12, r'$\gamma=2.0$  $\alpha=3.41$', ha='center', fontsize=8,
               color=PHASE2_CLR, fontweight='bold')

# ─── Panel 3: Fase 2/3 ─────────────────────────────────────────
ax3 = fig.add_subplot(gs[:, 4])
ax3.set_title('C: Fase 2/3 (Focal Loss)', fontsize=10, fontweight='bold', pad=5, color=PHASE2_CLR)
probability_strip(ax3, p_phase2, PHASE2_CLR, 'C_Fase_2/3_(Focal_Loss)',
                  C['mean_prob'], C['n_pos'], C['n_neg'],
                  C['tp'], C['fn'], C['fp'], C['tn'],
                  C['recall']*100, C['f2_score'])

# ─── Panel 4: Before vs After trajectory ───────────────────────
ax4 = fig.add_subplot(gs[:, 5])
ax4.set_title('Trajectory Per Sampel', fontsize=10, fontweight='bold', pad=5, color=TEXT_CLR)
violation_plot(ax4, y_true, p_baseline, p_phase2, BASELINE_CLR, PHASE2_CLR)

# ─── Panel 5: Ringkasan metrik ─────────────────────────────────
ax5 = fig.add_subplot(gs[:, 6])
ax5.set_title('Ringkasan Metrik', fontsize=10, fontweight='bold', pad=5, color=TEXT_CLR)

# Mini bar chart komparasi
scenarios = ['Baseline', 'Fase 1', 'Fase 2/3']
rec_vals  = [A['recall']*100, B['recall']*100, C['recall']*100]
prec_vals = [A['precision']*100, B['precision']*100, C['precision']*100]
f2_vals   = [A['f2_score'], B['f2_score'], C['f2_score']]
fpr_vals  = [A['fpr']*100, B['fpr']*100, C['fpr']*100]

x = np.arange(len(scenarios))
w = 0.2

ax5.bar(x - 1.5*w, rec_vals, w, color=BASELINE_CLR, label='Recall', edgecolor='white')
ax5.bar(x - 0.5*w, prec_vals, w, color=PHASE1_CLR, label='Precision', edgecolor='white')
ax5.bar(x + 0.5*w, [v*100 for v in f2_vals], w, color=PHASE2_CLR, label='F2×100', edgecolor='white')
ax5.bar(x + 1.5*w, fpr_vals, w, color='#C0C0C0', label='FPR', edgecolor='white')

ax5.set_xticks(x)
ax5.set_xticklabels(scenarios, fontsize=7)
ax5.set_ylabel('%', fontsize=8)
ax5.legend(fontsize=6, loc='upper left', framealpha=0.8)
ax5.spines[['top', 'right']].set_visible(False)
ax5.tick_params(labelsize=7)
ax5.set_ylim(0, 115)
ax5.grid(axis='y', alpha=0.3)

# Anotasi kenaikan recall
ax5.annotate(f'+{rec_vals[2]-rec_vals[0]:.0f}pp',
             xy=(2, rec_vals[2]), xytext=(1.5, rec_vals[2]+15),
             fontsize=8, color=PHASE2_CLR, fontweight='bold',
             ha='center',
             arrowprops=dict(arrowstyle='->', color=PHASE2_CLR, lw=1.5))

# ─── Simpan ────────────────────────────────────────────────────
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
save_path = OUTPUT_DIR / "fig_0_graphical_abstract.png"
plt.savefig(save_path, dpi=600, bbox_inches='tight', facecolor='white')
plt.close()

print(f"[OK] Graphical Abstract saved: {save_path}")
print(f"     Size: {save_path.stat().st_size / 1024:.0f} KB")
