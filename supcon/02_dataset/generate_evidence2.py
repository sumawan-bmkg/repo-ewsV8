#!/usr/bin/env python3
"""EVIDENCE 2: Dataset Distribution — Print-ready white theme, 300 DPI"""
import os, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

OUT = 'D:/multi/scalogramv3/disertasi4/supcon/02_dataset'
os.makedirs(OUT, exist_ok=True)
preds = pd.read_csv('D:/multi/scalogramv3/blind_test_2026_v8_results/v8supcon_2026_predictions.csv')
n_pos = (preds['True_Label'] == 1).sum()
n_neg = (preds['True_Label'] == 0).sum()

fig, axes = plt.subplots(2, 2, figsize=(20, 14))
fig.patch.set_facecolor('white')

# A: Class distribution
ax = axes[0, 0]
ax.set_facecolor('white')
bars = ax.bar(['Positive\n(event=1)', 'Negative\n(event=0)'],
              [n_pos, n_neg], color=['#C0392B', '#2980B9'],
              edgecolor='#333333', linewidth=0.5, width=0.5)
for bar, val in zip(bars, [n_pos, n_neg]):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+30,
            f'{val:,}\n({val/(n_pos+n_neg)*100:.1f}%)', ha='center', fontsize=11,
            fontweight='bold', color='#000000')
ax.set_title('Class Distribution (Blind Test 2026)', fontsize=13, fontweight='bold', color='#000000')
ax.set_ylabel('Samples', fontsize=11, color='#000000')
ax.tick_params(colors='#000000')
ax.spines['bottom'].set_color('#CCCCCC')
ax.spines['left'].set_color('#CCCCCC')

# B: Station distribution
ax = axes[0, 1]
ax.set_facecolor('white')
preds['Station'].value_counts().plot(kind='bar', ax=ax, color='#16A085',
    edgecolor='#333333', linewidth=0.3, width=0.7)
ax.set_title('Station Distribution', fontsize=13, fontweight='bold', color='#000000')
ax.set_xlabel('Station', fontsize=10, color='#000000')
ax.set_ylabel('Samples', fontsize=10, color='#000000')
ax.tick_params(colors='#000000', labelsize=8)
ax.spines['bottom'].set_color('#CCCCCC')
ax.spines['left'].set_color('#CCCCCC')

# C: Daily event count
ax = axes[1, 0]
ax.set_facecolor('white')
preds['Date'] = pd.to_datetime(preds['Date'])
daily = preds.groupby(preds['Date'].dt.date)['True_Label'].sum()
daily.plot(ax=ax, color='#C0392B', linewidth=1.5, alpha=0.8)
ax.fill_between(daily.index, daily.values, alpha=0.15, color='#C0392B')
ax.set_title('Daily Event Count (2026 Blind Test)', fontsize=13, fontweight='bold', color='#000000')
ax.set_xlabel('Date', fontsize=10, color='#000000')
ax.set_ylabel('Events per Day', fontsize=10, color='#000000')
ax.tick_params(colors='#000000', labelsize=8)
ax.spines['bottom'].set_color('#CCCCCC')
ax.spines['left'].set_color('#CCCCCC')

# D: Kp distribution
ax = axes[1, 1]
ax.set_facecolor('white')
ax.hist(preds['Kp_Raw'], bins=20, color='#D4AC0D', edgecolor='#333333', linewidth=0.3, alpha=0.8)
ax.axvline(4.0, color='#C0392B', linestyle='--', linewidth=1.5, label='Kp=4 (storm threshold)')
ax.set_title('Kp-Index Distribution', fontsize=13, fontweight='bold', color='#000000')
ax.set_xlabel('Kp Index', fontsize=10, color='#000000')
ax.set_ylabel('Frequency', fontsize=10, color='#000000')
ax.legend(fontsize=9)
ax.tick_params(colors='#000000')
ax.spines['bottom'].set_color('#CCCCCC')
ax.spines['left'].set_color('#CCCCCC')

fig.suptitle('V8 SupCon — Dataset Analysis', fontsize=16, fontweight='bold', color='#000000', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.95])
for fmt in ['png', 'svg', 'pdf']:
    fig.savefig(f'{OUT}/dataset_distribution.{fmt}', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f'[OK] Evidence 2 saved to {OUT}/')
