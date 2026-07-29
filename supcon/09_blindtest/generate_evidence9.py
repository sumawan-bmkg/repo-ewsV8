#!/usr/bin/env python3
"""EVIDENCE 9: Blind Test Timeline — Print-ready white theme"""
import os, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

OUT = 'D:/multi/scalogramv3/disertasi4/supcon/09_blindtest'
os.makedirs(OUT, exist_ok=True)
df = pd.read_csv('D:/multi/scalogramv3/blind_test_2026_v8_results/v8supcon_2026_predictions.csv')
df['Date'] = pd.to_datetime(df['Date'])
daily = df.groupby(df['Date'].dt.date).agg(
    n_samples=('True_Label','count'), n_events=('True_Label','sum'),
    n_detections=('Pred_Binary','sum'), mean_prob=('Pred_Prob','mean'),
    max_prob=('Pred_Prob','max'), mean_kp=('Kp_Raw','mean'),
    n_fp=('Pred_Binary',lambda x: ((x==1)&(df.loc[x.index,'True_Label']==0)).sum())
).reset_index()
daily['Date'] = pd.to_datetime(daily['Date'])

fig, axes = plt.subplots(4, 1, figsize=(22, 16), sharex=True)
fig.patch.set_facecolor('white')

ax = axes[0]; ax.set_facecolor('white')
ax.plot(daily['Date'], daily['max_prob'], color='#C0392B', lw=1.5, label='Max Prob')
ax.plot(daily['Date'], daily['mean_prob'], color='#D4AC0D', lw=1, label='Mean Prob')
ax.axhline(0.5, color='#000000', ls='--', lw=0.8, alpha=0.5, label='Threshold=0.5')
ax.set_title('Daily Prediction Probability', fontsize=12, fontweight='bold', color='#000000')
ax.set_ylabel('Probability', fontsize=10, color='#000000')
ax.legend(fontsize=9); ax.tick_params(colors='#000000')
ax.spines['bottom'].set_color('#CCCCCC'); ax.spines['left'].set_color('#CCCCCC')
ax.set_ylim(0, 1.05)

ax = axes[1]; ax.set_facecolor('white')
ax.bar(daily['Date'], daily['n_events'], color='#2980B9', alpha=0.7, label='True Events', width=0.4)
ax.bar(daily['Date'], daily['n_detections'], color='#C0392B', alpha=0.5, label='Detected (th=0.5)', width=0.4, align='edge')
ax.set_title('Daily Detection vs True Events', fontsize=12, fontweight='bold', color='#000000')
ax.set_ylabel('Count', fontsize=10, color='#000000')
ax.legend(fontsize=9); ax.tick_params(colors='#000000')
ax.spines['bottom'].set_color('#CCCCCC'); ax.spines['left'].set_color('#CCCCCC')

ax = axes[2]; ax.set_facecolor('white')
ax.bar(daily['Date'], daily['n_fp'], color='#C0392B', alpha=0.8, label='False Positives', width=0.6)
ax.set_title('Daily False Positives', fontsize=12, fontweight='bold', color='#000000')
ax.set_ylabel('False Positives', fontsize=10, color='#000000')
ax.legend(fontsize=9); ax.tick_params(colors='#000000')
ax.spines['bottom'].set_color('#CCCCCC'); ax.spines['left'].set_color('#CCCCCC')

ax = axes[3]; ax.set_facecolor('white')
ax.plot(daily['Date'], daily['mean_kp'], color='#D4AC0D', lw=1.5, label='Mean Kp')
ax.fill_between(daily['Date'], daily['mean_kp'], 4.0, where=(daily['mean_kp']>=4.0),
                color='red', alpha=0.3, label='Storm (Kp>=4)')
ax.axhline(4.0, color='#C0392B', ls='--', lw=1, alpha=0.7, label='Storm Threshold')
ax.set_title('Geomagnetic Activity (Kp-index)', fontsize=12, fontweight='bold', color='#000000')
ax.set_xlabel('Date (2026)', fontsize=10, color='#000000')
ax.set_ylabel('Kp Index', fontsize=10, color='#000000')
ax.legend(fontsize=9); ax.tick_params(colors='#000000')
ax.spines['bottom'].set_color('#CCCCCC'); ax.spines['left'].set_color('#CCCCCC')

fig.suptitle('V8 SupCon — Blind Test Timeline (Jan–Apr 2026)',
             fontsize=14, fontweight='bold', color='#000000', y=1.01)
plt.tight_layout()
for fmt in ['png', 'svg', 'pdf']:
    fig.savefig(f'{OUT}/blind_test_timeline.{fmt}', dpi=300, bbox_inches='tight', facecolor='white')
plt.close(); print(f'[OK] Evidence 9 saved to {OUT}/')
