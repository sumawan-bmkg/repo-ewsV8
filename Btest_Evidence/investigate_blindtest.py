#!/usr/bin/env python3
"""Investigasi validitas blind test 2026 — V8 SupCon"""

import os, sys, json, hashlib, pandas as pd, numpy as np

OUT = 'D:/multi/scalogramv3/disertasi4/supcon/06_evaluation/kinerja'
print('='*70)
print('BLIND TEST 2026 — VALIDITY INVESTIGATION')
print('='*70)

# 1. Basic stats
p = pd.read_csv('D:/multi/scalogramv3/blind_test_2026_v8_results/v8supcon_2026_predictions.csv')
print(f'\n[1] FILE INTEGRITY')
print(f'  Path: D:/multi/scalogramv3/blind_test_2026_v8_results/v8supcon_2026_predictions.csv')
print(f'  Rows: {len(p)}')
print(f'  Columns: {len(p.columns)}')
print(f'  Checksum: {hashlib.md5(p.to_csv(index=False).encode()).hexdigest()[:16]}...')

# 2. Source pipeline
print(f'\n[2] PIPELINE SOURCE')
pipe_path = 'D:/multi/scalogramv3/v8supcon_2026_blind_test_full.py'
if os.path.exists(pipe_path):
    print(f'  Pipeline script: {pipe_path}')
    print(f'  Size: {os.path.getsize(pipe_path)} bytes')
    with open(pipe_path, encoding='utf-8') as f:
        content = f.read()
    # Check if using real HDF5 data
    if 'h5' in content.lower() or 'hdf5' in content.lower():
        print(f'  HDF5 references: FOUND')
    else:
        print(f'  HDF5 references: NOT FOUND — check for torch.zeros')
    if 'zeros' in content:
        print(f'  WARNING: torch.zeros reference found!')
    # Check how many samples
    import re
    files_matched = re.findall(r'2026/scalogram/([^"]+)', content)
    print(f'  HDF5 file references in pipeline: {len(files_matched)}')
    # Check model loading
    if 'v3_v8_conv_fpr_best_weights.pth' in content:
        print(f'  Model checkpoint: v3_v8_conv_fpr_best_weights.pth [OK]')
else:
    print(f'  Pipeline script NOT FOUND at {pipe_path}')

# 3. HDF5 data existence
print(f'\n[3] HDF5 DATA CHECK')
h5_dir = 'D:/multi/scalogramv3/2026/scalogram'
if os.path.exists(h5_dir):
    h5_files = [f for f in os.listdir(h5_dir) if f.endswith('.h5')]
    print(f'  HDF5 files found: {len(h5_files)}')
    # Expected: 24 stations x ~120 days = ~2880 files
    print(f'  Expected: ~2880 (24 stations × 120 days)')
    if len(h5_files) > 0:
        print(f'  Sample files: {h5_files[:3]}...')
    else:
        print(f'  WARNING: No HDF5 files found!')
else:
    print(f'  WARNING: HDF5 directory not found!')
    # Check alternative patterns
    for alt in ['D:/multi/scalogramv3/2026', 'D:/multi/scalogramv3/data/2026']:
        if os.path.exists(alt):
            print(f'  Found alternative: {alt}')
            print(f'    Contents: {os.listdir(alt)[:10]}')

# 4. EQ Catalogue validity
print(f'\n[4] EQ CATALOGUE CHECK')
cat_path = 'D:/multi/scalogramv3/2026/merge2026.csv'
if os.path.exists(cat_path):
    cat = pd.read_csv(cat_path, comment='#', on_bad_lines='skip')
    print(f'  Catalogue: {cat_path}')
    print(f'  Events: {len(cat)}')
    cols = list(cat.columns)
    print(f'  Columns: {cols}')
    if 'Magnitude' in cols:
        print(f'  Magnitude range: {cat["Magnitude"].min()} - {cat["Magnitude"].max()}')
    if 'Date time' in cols or 'Date' in cols or 'origin_time' in cols:
        date_col = [c for c in cols if 'date' in c.lower() or 'time' in c.lower()][0]
        dates = pd.to_datetime(cat[date_col], errors='coerce')
        print(f'  Date range: {dates.min()} to {dates.max()}')
else:
    print(f'  WARNING: {cat_path} not found!')
    # Look for alternative
    alt_cats = [f for f in os.listdir('D:/multi/scalogramv3/2026') if f.endswith('.csv')]
    print(f'  Alternative CSVs in 2026/: {alt_cats}')

# 5. Prediction value integrity
print(f'\n[5] PREDICTION VALUE ANALYSIS')
print(f'  Pred_Prob unique values: {p["Pred_Prob"].nunique()}')
print(f'  Pred_Prob stats:')
print(f'    min={p["Pred_Prob"].min():.6f}, max={p["Pred_Prob"].max():.6f}')
print(f'    mean={p["Pred_Prob"].mean():.6f}, std={p["Pred_Prob"].std():.6f}')
print(f'    median={p["Pred_Prob"].median():.6f}')
# Check for suspicious uniformity
probs = p['Pred_Prob'].values
diff = np.diff(np.sort(probs))
uniformity_metric = diff.std() / diff.mean() if diff.mean() > 0 else 0
print(f'  Uniformity score (std/mean of sorted diffs): {uniformity_metric:.6f}')
print(f'  Low score ~0 = suspiciously uniform predictions')
# Check batch effects - are predictions the same within days?
p['Date'] = pd.to_datetime(p['Date'])
daily_probs = p.groupby(p['Date'].dt.date)['Pred_Prob'].agg(['mean','std','count'])
print(f'  Days with zero std (constant preds all samples): {(daily_probs["std"]==0).sum()}/{len(daily_probs)}')
# Check for repeated exact values (model collapse symptom)
from collections import Counter
value_counts = Counter(probs).most_common(10)
print(f'  Most common Pred_Prob values:')
for v, c in value_counts[:5]:
    print(f'    P={v:.6f}: {c} occurrences ({c/len(p)*100:.1f}%)')
# Check correlation with Kp
print(f'\n  Correlation Pred_Prob ~ Kp: {p["Pred_Prob"].corr(p["Kp_Raw"]):.4f}')

# 6. Azimuth validity
print(f'\n[6] AZIMUTH ANALYSIS')
az = p['Az_Error']
print(f'  Az_Error (only event=1): n={len(az)}')
print(f'  MAE: {az.mean():.2f}°, Med: {az.median():.2f}°')
print(f'  Min: {az.min():.2f}°, Max: {az.max():.2f}°')
print(f'  Unique values: {az.nunique()}')
# Check True_Az vs Pred_Az
print(f'  True_Az unique: {p["True_Az"].nunique()}')
print(f'  Pred_Az unique: {p["Pred_Az"].nunique()}')
# Check if azimuth values are reasonable direction vectors
az_sincos = np.sqrt(p['Pred_Az'].values**2)  # should be ~1 if sin/cos
print(f'  Pred_Az range: [{p["Pred_Az"].min():.2f}, {p["Pred_Az"].max():.2f}]')

# 7. True label validity
print(f'\n[7] LABEL VALIDITY')
print(f'  True_Label distribution:')
print(f'    Event=1 (positive): {(p["True_Label"]==1).sum()} ({((p["True_Label"]==1).sum()/len(p))*100:.2f}%)')
print(f'    Event=0 (negative): {(p["True_Label"]==0).sum()} ({((p["True_Label"]==0).sum()/len(p))*100:.2f}%)')
# Check if event=0 samples come from specific stations
neg_stations = p[p['True_Label']==0]['Station'].unique()
print(f'  Stations with negative samples: {sorted(neg_stations)}')
neg_dates = p[p['True_Label']==0]['Date'].unique()
print(f'  Dates with negative samples: {len(neg_dates)} days')
# Check Kp distribution for negatives
print(f'  Mean Kp for negatives: {p[p["True_Label"]==0]["Kp_Raw"].mean():.2f}')
print(f'  Mean Kp for positives: {p[p["True_Label"]==1]["Kp_Raw"].mean():.2f}')

# 8. Check for duplicates
print(f'\n[8] DUPLICATE CHECK')
dup = p.duplicated(subset=['Date','Station'], keep=False)
print(f'  Duplicate (Date, Station) rows: {dup.sum()}')
dup_all = p.duplicated(keep=False)
print(f'  Fully duplicate rows: {dup_all.sum()}')

# 9. Conclusion
print(f'\n[9] PRELIMINARY VERDICT')
warnings = []
if len(p) == 0: warnings.append('EMPTY DATASET')
if ((p['True_Label']==1).sum()/len(p)) > 0.99: warnings.append('EXTREME CLASS IMBALANCE (>99%)')
if len(np.unique(probs)) < 10: warnings.append('TOO FEW UNIQUE PROBABILITIES (<10)')
if uniformity_metric < 0.01: warnings.append('SUSPICIOUS UNIFORMITY IN PROBABILITIES')
if (p['Pred_Prob'].max() - p['Pred_Prob'].min()) < 0.1: warnings.append('PREDICTION RANGE TOO NARROW')
if (daily_probs['std']==0).sum() / len(daily_probs) > 0.5: warnings.append('MOST DAYS HAVE CONSTANT PREDICTIONS')
if len([f for f in os.listdir('D:/multi/scalogramv3/2026/scalogram') if f.endswith('.h5')]) < 100: 
    warnings.append('FEW HDF5 FILES (<100)')
if p['Az_Error'].nunique() < 10: warnings.append('AZIMUTH ERROR VALUES SUSPICIOUSLY FEW')

if len(warnings) == 0:
    print(f'  [OK] No warnings - data appears valid')
else:
    print(f'  ⚠ WARNINGS ({len(warnings)}):')
    for w in warnings:
        print(f'    - {w}')

print(f'\n{"="*70}')
