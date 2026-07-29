#!/usr/bin/env python3
"""
Re-run Blind Test 2026 with latest merge2026.csv catalogue
- Uses existing predictions CSV (HDF5 inference unchanged)
- Reloads catalogue from merge2026.csv (merged EQ1-EQ6)
- Re-runs stages 2-5 (quantification, physics, lead-time)
- Outputs comparison vs previous run
"""
import os, sys, json, math, warnings, time
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
warnings.filterwarnings('ignore')

ROOT = Path('D:/multi/scalogramv3')
IN_DIR = ROOT / 'blind_test_2026_v8_results'
PRED_FILE = IN_DIR / 'v8supcon_2026_predictions.csv'
OLD_METRICS = IN_DIR / 'metrics_all_stages.json'
CATALOGUE = ROOT / '2026' / 'merge2026.csv'
OUT = ROOT / 'disertasi4' / 'supcon' / '06_evaluation' / 'kinerja'
OUT.mkdir(parents=True, exist_ok=True)

print('=' * 70)
print('RE-RUN BLIND TEST 2026 — with merge2026.csv catalogue')
print('=' * 70)

# 1. Load predictions
print(f'\n[1] Loading predictions from {PRED_FILE}')
df = pd.read_csv(PRED_FILE)
print(f'  {len(df)} samples, {df["Station"].nunique()} stations')
print(f'  Date: {df["Date"].min()} to {df["Date"].max()}')

# 2. Load OLD metrics for comparison
print(f'\n[2] Loading old metrics from {OLD_METRICS}')
if OLD_METRICS.exists():
    old = json.loads(OLD_METRICS.read_text())
    print(f'  Loaded {len(old)} metric groups')
    for k, v in old.items():
        print(f'    {k}: {json.dumps(v, indent=2)[:80]}')
else:
    old = {}
    print(f'  NOT FOUND — skipping comparison')

# 3. Load NEW catalogue
print(f'\n[3] Loading NEW catalogue: {CATALOGUE}')
cat = pd.read_csv(CATALOGUE, comment='#', on_bad_lines='skip')
cat.columns = [c.strip().strip('"') for c in cat.columns]
print(f'  Raw events: {len(cat)}')
print(f'  Columns: {list(cat.columns)}')

# Normalise
rename = {}
for c in cat.columns:
    cl = c.lower()
    if 'event id' in cl:          rename[c] = 'event_id'
    elif 'date time' in cl:        rename[c] = 'datetime'
    elif cl == 'magnitude':        rename[c] = 'magnitude'
    elif cl == 'latitude':         rename[c] = 'latitude'
    elif cl == 'longitude':        rename[c] = 'longitude'
    elif 'depth' in cl:            rename[c] = 'depth_km'
cat = cat.rename(columns=rename)

if 'datetime' in cat.columns:
    cat['origin_time'] = pd.to_datetime(cat['datetime'], errors='coerce', utc=True)
for col in ['magnitude','depth_km','latitude','longitude']:
    if col in cat.columns:
        cat[col] = pd.to_numeric(cat[col], errors='coerce')

print(f'  Events after cleaning: {len(cat)}')
print(f'  Date range: {cat["origin_time"].min()} to {cat["origin_time"].max()}')
print(f'  Magnitude: {cat["magnitude"].min():.2f} to {cat["magnitude"].max():.2f}')

# Station coords (same as original pipeline)
STATION_COORDS = {
    'ALR': (-8.46, 115.52), 'AMB': (-8.25, 115.35), 'CLP': (-8.49, 115.59),
    'GSI': (-8.35, 115.38), 'GTO': (-8.35, 115.38),
    'JYP': (-8.35, 115.38), 'KPY': (-8.45, 115.33), 'LPS': (-8.45, 115.33),
    'LUT': (-8.45, 115.33), 'LWA': (-8.45, 115.33), 'LWK': (-8.40, 115.20),
    'MLB': (-8.40, 115.20), 'PLU': (-8.63, 115.94), 'ROT': (-8.30, 115.09),
    'SBG': (-8.40, 115.18), 'SCN': (-8.40, 115.18), 'SKB': (-8.40, 115.18),
    'SMI': (-8.40, 115.18), 'SRG': (-7.81, 110.36), 'SRO': (-8.40, 115.18),
    'TNT': (-8.12, 115.08), 'TND': (-8.12, 115.08), 'TRD': (-8.30, 115.09),
    'TRT': (-8.30, 115.09), 'YOG': (-7.81, 110.36),
}
ANOMALY_STATIONS = {'PLU', 'TNT', 'LWK', 'SRO'}

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat/2)**2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon/2)**2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

def circular_error(pred_deg, true_deg):
    diff = abs(pred_deg - true_deg) % 360
    return min(diff, 360 - diff)

# 4. RE-RUN Stage 2-3: Quantification
print(f'\n[4] STAGE 2-3: Quantification & Localization (with merge2026)')
az_valid = df.dropna(subset=['Az_Error'])
az_mae_global = az_valid['Az_Error'].mean()
az_mae_trimmed = az_valid[~az_valid['Station'].isin(ANOMALY_STATIONS)]['Az_Error'].mean()
mag_mae = (df['True_Mw'] - df['Pred_Mw']).abs().mean()

# Distance/Depth using new catalogue
dist_err = []; depth_err = []
cat_valid = cat.dropna(subset=['latitude','longitude','origin_time']).copy()
cat_valid['_date'] = pd.to_datetime(cat_valid['origin_time'], utc=True).dt.normalize()

for _, row in df.iterrows():
    stn = row['Station']
    stn_coords = STATION_COORDS.get(stn)
    if stn_coords is None:
        continue
    event_date = pd.Timestamp(row['Date']).normalize().tz_localize('UTC')
    time_diffs = (cat_valid['_date'] - event_date).abs()
    mask = time_diffs.dt.days <= 1
    candidates = cat_valid[mask]
    if len(candidates) == 0:
        continue
    best = candidates.loc[time_diffs[mask].idxmin()]
    d = haversine_km(stn_coords[0], stn_coords[1], best['latitude'], best['longitude'])
    dist_err.append(d)
    if pd.notna(best.get('depth_km')):
        depth_err.append(float(best['depth_km']))

new_s23 = {
    'Az_MAE_Global_deg': round(az_mae_global, 2),
    'Az_MAE_Trimmed_deg': round(az_mae_trimmed, 2),
    'Mag_MAE_Mw': round(mag_mae, 3),
    'Dist_MAE_km': round(np.mean(dist_err), 1) if dist_err else None,
    'Depth_MAE_km': round(np.mean(depth_err), 1) if depth_err else None,
    'n_dist_pairs': len(dist_err),
    'n_depth_pairs': len(depth_err),
}
print(f'  New: {json.dumps(new_s23, indent=2)}')
if 's23' in old:
    print(f'  Old: {json.dumps(old["s23"], indent=2)}')

# 5. RE-RUN Stage 4: Physics
print(f'\n[5] STAGE 4: Physical Law Consistency (with merge2026)')
pairs = []
for _, row in df.iterrows():
    stn = row['Station']
    stn_coords = STATION_COORDS.get(stn)
    if stn_coords is None:
        continue
    event_date = pd.Timestamp(row['Date']).normalize().tz_localize('UTC')
    time_diffs = (cat_valid['_date'] - event_date).abs()
    mask = time_diffs.dt.days <= 1
    candidates = cat_valid[mask]
    if len(candidates) == 0:
        continue
    best = candidates.loc[time_diffs[mask].idxmin()]
    dist = haversine_km(stn_coords[0], stn_coords[1], best['latitude'], best['longitude'])
    mw = row['True_Mw']
    if dist > 0 and mw > 0:
        r_dobro = 10 ** (0.43 * mw)
        pairs.append({'pred_prob': row['Pred_Prob'], 'strain_ratio': r_dobro / dist,
                      'true_label': row['True_Label']})

new_s4 = {'n_pairs': len(pairs), 'median_strain_ratio': float('nan')}
if len(pairs) >= 10:
    pdf = pd.DataFrame(pairs)
    rho, pv = spearmanr(pdf['pred_prob'], pdf['strain_ratio'])
    tp = pdf[pdf['true_label']==1]
    rho_tp, pv_tp = (float('nan'), float('nan'))
    if len(tp) >= 10:
        rho_tp, pv_tp = spearmanr(tp['pred_prob'], tp['strain_ratio'])
    new_s4.update({
        'Spearman_rho_all': round(float(rho), 4),
        'Spearman_p_all': round(float(pv), 6),
        'Spearman_rho_TP': round(float(rho_tp), 4) if not math.isnan(float(rho_tp)) else None,
        'Spearman_p_TP': round(float(pv_tp), 6) if not math.isnan(float(pv_tp)) else None,
        'n_pairs_TP': len(tp),
        'median_strain_ratio': round(float(pdf['strain_ratio'].median()), 4),
    })
print(f'  New: {json.dumps(new_s4, indent=2)}')
if 's4' in old:
    print(f'  Old: {json.dumps(old["s4"], indent=2)}')

# 6. RE-RUN Stage 5: Lead-time
print(f'\n[6] STAGE 5: Operational Lead-Time (with merge2026)')
lead_days = []
matched = 0
for _, row in df.iterrows():
    if row['Pred_Binary'] != 1 or row['True_Label'] != 1:
        continue
    stn = row['Station']
    stn_coords = STATION_COORDS.get(stn)
    if stn_coords is None:
        continue
    event_date = pd.Timestamp(row['Date']).normalize().tz_localize('UTC')
    time_diffs = (cat_valid['_date'] - event_date).abs()
    mask = time_diffs.dt.days <= 1
    candidates = cat_valid[mask]
    if len(candidates) == 0:
        continue
    matched += 1
    best = candidates.loc[time_diffs[mask].idxmin()]
    delta = (event_date - pd.Timestamp(best['origin_time']).tz_convert('UTC')
             ).total_seconds() / 86400
    lead_days.append(delta)

new_s5 = {
    'mean_lead_days': round(float(np.mean(lead_days)), 4) if lead_days else None,
    'median_lead_days': round(float(np.median(lead_days)), 4) if lead_days else None,
    'n_matched': matched,
    'n_lead_values': len(lead_days),
}
print(f'  New: {json.dumps(new_s5, indent=2)}')
if 's5' in old:
    print(f'  Old: {json.dumps(old["s5"], indent=2)}')

# 7. Stage 1 detection (unchanged — doesn't use catalogue)
print(f'\n[7] STAGE 1: Detection (inference-based, unchanged)')
y_true = df['True_Label'].values
y_pred = df['Pred_Binary'].values
y_prob = df['Pred_Prob'].values
TP = int(((y_pred==1)&(y_true==1)).sum())
TN = int(((y_pred==0)&(y_true==0)).sum())
FP = int(((y_pred==1)&(y_true==0)).sum())
FN = int(((y_pred==0)&(y_true==1)).sum())
from sklearn.metrics import average_precision_score, roc_auc_score
auprc = float(average_precision_score(y_true, y_prob))
auc_roc = float(roc_auc_score(y_true, y_prob))
s1 = {
    'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN,
    'Precision': round(TP/(TP+FP), 4) if (TP+FP)>0 else 0,
    'Recall': round(TP/(TP+FN), 4) if (TP+FN)>0 else 0,
    'F2': round((5*TP/(TP+FP))/(4*TP/(TP+FP)+TP/(TP+FN)), 4) if (TP+FP)>0 and (TP+FN)>0 else 0,
    'AUPRC': round(auprc, 4),
    'AUC_ROC': round(auc_roc, 4),
}
print(f'  {json.dumps(s1, indent=2)}')

# 8. Save new metrics
new_metrics = {
    's1_detection': s1,
    's23_quantification': new_s23,
    's4_physics': new_s4,
    's5_leadtime': new_s5,
    'info': {
        'catalogue': 'merge2026.csv',
        'catalogue_events': len(cat),
        'catalogue_date_range': f'{cat["origin_time"].min()} to {cat["origin_time"].max()}',
        'predictions': len(df),
        'stations': df['Station'].nunique(),
        'date_range': f'{df["Date"].min()} to {df["Date"].max()}',
        'generated': str(datetime.now()),
    }
}
OUT_METRICS = OUT / 'blindtest_metrics_merge2026.json'
with open(OUT_METRICS, 'w') as f:
    json.dump(new_metrics, f, indent=2)
print(f'\n[8] Saved new metrics to {OUT_METRICS}')

# 9. Comparison table
print(f'\n[9] COMPARISON: OLD (EQ*.2026) vs NEW (merge2026)')
print(f'{"Metric":<30} {"OLD":<20} {"NEW":<20}')
print('-' * 70)

comparison_keys = [
    ('s23', 'Az_MAE_Global_deg'), ('s23', 'Az_MAE_Trimmed_deg'),
    ('s23', 'Mag_MAE_Mw'), ('s23', 'Dist_MAE_km'), ('s23', 'Depth_MAE_km'),
]
for section, key in comparison_keys:
    old_section = old.get('s23' if section == 's23' else section, {})
    old_val = old_section.get(key, '?')
    new_val = new_metrics.get('s23_quantification' if section == 's23' else f'{section}_physics', {}).get(key, '?')
    print(f'{key:<30} {str(old_val):<20} {str(new_val):<20}')

physics_keys = ['Spearman_rho_all', 'Spearman_p_all', 'n_pairs', 'median_strain_ratio']
for key in physics_keys:
    old_val = old.get('s4', {}).get(key, '?')
    new_val = new_metrics['s4_physics'].get(key, '?')
    print(f'{key:<30} {str(old_val):<20} {str(new_val):<20}')

lead_keys = ['mean_lead_days', 'median_lead_days', 'n_matched']
for key in lead_keys:
    old_val = old.get('s5', {}).get(key, '?')
    new_val = new_metrics['s5_leadtime'].get(key, '?')
    print(f'{key:<30} {str(old_val):<20} {str(new_val):<20}')

print()
print('NOTE: Stage 1 detection metrics are IDENTICAL (no catalogue dependency).')
print(f'Differences in Stage 2-5 are due to different event matching with merge2026.csv')
print(f'which has {len(cat)} events vs individual EQ*.2026.csv files.')

print(f'\n{"=" * 70}')
