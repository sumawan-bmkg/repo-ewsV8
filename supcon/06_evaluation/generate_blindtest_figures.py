#!/usr/bin/env python3
"""Generate Blind Test 2026 figures for dashboard: timeline + station heatmap"""
import os, warnings, matplotlib
os.environ['MPLBACKEND'] = 'Agg'
matplotlib.use('Agg')
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd, numpy as np

OUT = 'D:/multi/scalogramv3/disertasi4/supcon/06_evaluation/kinerja'
p = pd.read_csv('D:/multi/scalogramv3/blind_test_2026_v8_results/v8supcon_2026_predictions.csv')
p['Date'] = pd.to_datetime(p['Date'])

# ── Daily aggregation ──
daily = p.groupby(p['Date'].dt.date).agg(
    n=('True_Label','count'), events=('True_Label','sum'),
    detections=('Pred_Binary','sum'), prob_mean=('Pred_Prob','mean'),
    prob_max=('Pred_Prob','max'), kp_mean=('Kp_Raw','mean')).reset_index()
daily.columns = ['Date'] + list(daily.columns[1:])
daily['Date'] = pd.to_datetime(daily['Date'])

# ── 1. Timeline 4-panel ──
fig, axes = plt.subplots(4, 1, figsize=(20, 14), sharex=True)
fig.patch.set_facecolor('white')

ax = axes[0]; ax.set_facecolor('white')
ax.plot(daily['Date'], daily['events'], 'o-', color='#2980B9', lw=1.5, ms=3, label='True Events')
ax.plot(daily['Date'], daily['detections'], 's-', color='#C0392B', lw=1.5, ms=3, label='Detected')
ax.fill_between(daily['Date'], daily['events'], alpha=0.08, color='#2980B9')
ax.legend(fontsize=9); ax.set_ylabel('Count', fontsize=11)
ax.set_title('A. Daily Events & Detections', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.15); ax.tick_params(colors='#000000')
ax.spines['bottom'].set_color('#CCC'); ax.spines['left'].set_color('#CCC')

ax = axes[1]; ax.set_facecolor('white')
ax.plot(daily['Date'], daily['prob_max'], 'o-', color='#C0392B', lw=1.5, ms=3, label='Max Probability')
ax.plot(daily['Date'], daily['prob_mean'], 's-', color='#D4AC0D', lw=1, ms=2, label='Mean Probability')
ax.axhline(0.225, color='#8E44AD', ls='--', lw=1.5, alpha=0.7, label='Optimal th=0.225')
ax.axhline(0.5, color='#666', ls=':', lw=1, alpha=0.5, label='Default th=0.5')
ax.legend(fontsize=8); ax.set_ylabel('Probability', fontsize=11)
ax.set_ylim(0, 1.05); ax.set_title('B. Prediction Probability Timeline', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.15); ax.tick_params(colors='#000000')
ax.spines['bottom'].set_color('#CCC'); ax.spines['left'].set_color('#CCC')

ax = axes[2]; ax.set_facecolor('white')
ax.plot(daily['Date'], daily['kp_mean'], color='#E67E22', lw=2, label='Mean Kp')
ax.fill_between(daily['Date'], daily['kp_mean'], 4, where=(daily['kp_mean']>=4),
                color='#C0392B', alpha=0.25, label='Storm (Kp>=4)')
ax.axhline(4, color='#C0392B', ls='--', lw=1.5, alpha=0.7, label='Storm threshold')
ax.legend(fontsize=9); ax.set_ylabel('Kp Index', fontsize=11)
ax.set_title('C. Geomagnetic Activity', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.15); ax.tick_params(colors='#000000')
ax.spines['bottom'].set_color('#CCC'); ax.spines['left'].set_color('#CCC')

ax = axes[3]; ax.set_facecolor('white')
# Cumulative detection rate
daily['cum_events'] = daily['events'].cumsum()
daily['cum_det'] = daily['detections'].cumsum()
daily['cum_rate'] = daily['cum_det'] / daily['cum_events'] * 100
ax.plot(daily['Date'], daily['cum_rate'], 'o-', color='#16A085', lw=2, ms=3)
ax.axhline(7.5, color='#666', ls='--', lw=1, alpha=0.6, label='Overall: 7.5%')
ax.legend(fontsize=9); ax.set_ylabel('Cumulative Detection Rate (%)', fontsize=11)
ax.set_xlabel('Date (2026)', fontsize=11)
ax.set_title('D. Cumulative Detection Rate', fontsize=12, fontweight='bold')
ax.set_ylim(0, 20); ax.grid(True, alpha=0.15); ax.tick_params(colors='#000000')
ax.spines['bottom'].set_color('#CCC'); ax.spines['left'].set_color('#CCC')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))

fig.suptitle('V8 SupCon — Blind Test 2026 Operational Timeline', fontsize=15, fontweight='bold', y=1.01)
plt.tight_layout()
fig.savefig(f'{OUT}/blindtest_timeline.png', dpi=300, bbox_inches='tight', facecolor='white')
fig.savefig(f'{OUT}/blindtest_timeline.pdf', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print('[OK] Blind test timeline saved')

# ── 2. Station Detection Heatmap ──
stn_stats = p.groupby('Station').agg(
    total=('True_Label','count'),
    events=('True_Label', 'sum'),
    detected=('Pred_Binary', lambda x: (x==1).sum()),
    prob_mean=('Pred_Prob','mean'),
    lat=('Station_Lat','first'), lon=('Station_Lon','first')).reset_index()
stn_stats['det_rate'] = stn_stats['detected']/stn_stats['events']*100
stn_stats['det_rate'] = stn_stats['det_rate'].fillna(0)

# Station detection rate bar chart
stn_stats_sorted = stn_stats.sort_values('det_rate', ascending=False)
fig, ax = plt.subplots(figsize=(14, 6))
fig.patch.set_facecolor('white'); ax.set_facecolor('white')
colors_bar = ['#27AE60' if v>8 else '#E67E22' if v>6 else '#C0392B' for v in stn_stats_sorted['det_rate']]
bars = ax.bar(range(len(stn_stats_sorted)), stn_stats_sorted['det_rate'], color=colors_bar, edgecolor='#333', lw=0.5, width=0.7)
ax.set_xticks(range(len(stn_stats_sorted)))
ax.set_xticklabels(stn_stats_sorted['Station'], fontsize=8, rotation=45, ha='right')
ax.set_ylabel('Detection Rate (%)', fontsize=11)
ax.set_title('Detection Rate by Station (th=0.5)', fontsize=13, fontweight='bold')
ax.axhline(stn_stats_sorted['det_rate'].mean(), color='#2980B9', ls='--', lw=1.5, label=f'Mean: {stn_stats_sorted["det_rate"].mean():.1f}%')
ax.legend(fontsize=9); ax.grid(True, axis='y', alpha=0.15)
ax.tick_params(colors='#000000'); ax.spines['bottom'].set_color('#CCC'); ax.spines['left'].set_color('#CCC')
for bar, v in zip(bars, stn_stats_sorted['det_rate']):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3, f'{v:.1f}%',
            ha='center', fontsize=6.5, fontweight='bold')
plt.tight_layout()
fig.savefig(f'{OUT}/station_detection_rate.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print('[OK] Station detection rate saved')

# ── 3. Daily prediction count heatmap-style (station x date) ──
pivot = p.pivot_table(index='Station', columns=p['Date'].dt.date,
                      values='Pred_Prob', aggfunc='mean')
fig, ax = plt.subplots(figsize=(22, 8))
fig.patch.set_facecolor('white'); ax.set_facecolor('white')
im = ax.imshow(pivot.values, cmap='YlOrRd', aspect='auto', vmin=0, vmax=0.5)
ax.set_yticks(range(len(pivot.index)))
ax.set_yticklabels(pivot.index, fontsize=7)
ax.set_xticks(range(0, len(pivot.columns), 7))
ax.set_xticklabels([str(d) for d in pivot.columns[::7]], fontsize=6, rotation=45, ha='right')
ax.set_xlabel('Date (2026)', fontsize=10)
ax.set_ylabel('Station', fontsize=10)
ax.set_title('Spatio-Temporal Prediction Activity (mean Pred_Prob)', fontsize=13, fontweight='bold')
cb = plt.colorbar(im, ax=ax, shrink=0.6)
cb.set_label('Mean Predicted Probability', fontsize=9)
plt.tight_layout()
fig.savefig(f'{OUT}/spatiotemporal_heatmap.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print('[OK] Spatiotemporal heatmap saved')
