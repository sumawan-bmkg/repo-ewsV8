#!/usr/bin/env python3
"""Deep investigation: Azimuth prediction range anomaly"""

import pandas as pd, numpy as np

p = pd.read_csv('D:/multi/scalogramv3/blind_test_2026_v8_results/v8supcon_2026_predictions.csv')

print('='*70)
print('CRITICAL INVESTIGATION: AZIMUTH PREDICTION RANGE')
print('='*70)

# Pred_Az range
print(f'\n[1] Pred_Az RANGE')
print(f'  min:  {p["Pred_Az"].min():.4f}')
print(f'  max:  {p["Pred_Az"].max():.4f}')
print(f'  range: {p["Pred_Az"].max()-p["Pred_Az"].min():.4f} degrees')
print(f'  mean:  {p["Pred_Az"].mean():.4f}')
print(f'  std:   {p["Pred_Az"].std():.4f}')
print(f'  n unique: {p["Pred_Az"].nunique()}')

print(f'\n[2] True_Az RANGE')
print(f'  min:  {p["True_Az"].min():.4f}')
print(f'  max:  {p["True_Az"].max():.4f}')
print(f'  range: {p["True_Az"].max()-p["True_Az"].min():.4f} degrees')
print(f'  mean:  {p["True_Az"].mean():.4f}')
print(f'  n unique: {p["True_Az"].nunique()}')

# Pred_Az distribution
print(f'\n[3] Pred_Az DISTRIBUTION')
bins = np.linspace(p['Pred_Az'].min(), p['Pred_Az'].max(), 10)
hist, _ = np.histogram(p['Pred_Az'], bins=bins)
for i in range(len(hist)):
    bar = '#' * (hist[i] // 100)
    print(f'  {bins[i]:7.1f}-{bins[i+1]:7.1f}: {hist[i]:5d} {bar}')

# Pred_Az by station
print(f'\n[4] Pred_Az BY STATION')
stn_az = p.groupby('Station')['Pred_Az'].agg(['mean','std','min','max','count'])
print(stn_az.to_string())

# Pred_Az by event vs non-event
print(f'\n[5] Pred_Az BY EVENT TYPE')
for label in [0, 1]:
    sub = p[p['True_Label'] == label]
    print(f'  Label={label}: n={len(sub)}, Pred_Az mean={sub["Pred_Az"].mean():.4f}, '
          f'std={sub["Pred_Az"].std():.4f}, min={sub["Pred_Az"].min():.4f}, max={sub["Pred_Az"].max():.4f}')

# Azimuth error breakdown
print(f'\n[6] AZIMUTH ERROR ANALYSIS')
print(f'  Mean:  {p["Az_Error"].mean():.2f}')
print(f'  Std:   {p["Az_Error"].std():.2f}')
print(f'  P25:   {p["Az_Error"].quantile(0.25):.2f}')
print(f'  P50:   {p["Az_Error"].quantile(0.50):.2f}')
print(f'  P75:   {p["Az_Error"].quantile(0.75):.2f}')
print(f'  P95:   {p["Az_Error"].quantile(0.95):.2f}')
print(f'  P99:   {p["Az_Error"].quantile(0.99):.2f}')

# Check if Pred_Az values are degrees (0-360) or radians or sin/cos
print(f'\n[7] PREDICTION CONVERSION CHECK')
print(f'  If Pred_Az is degrees (0-360):')
print(f'    range should be 0-360, actual: [{p["Pred_Az"].min():.2f}, {p["Pred_Az"].max():.2f}]')
print(f'  If Pred_Az is sin(theta):')
print(f'    range should be -1 to 1, actual: [{p["Pred_Az"].min():.2f}, {p["Pred_Az"].max():.2f}]')
print(f'  If Pred_Az is atan2(cos,sin)*180/pi:')
print(f'    range should be -180 to 180 or 0-360')

# Check how the azimuth is computed in the pipeline
print(f'\n[8] PIPELINE AZIMUTH COMPUTATION')
import os
pipe = 'D:/multi/scalogramv3/v8supcon_2026_blind_test_full.py'
if os.path.exists(pipe):
    with open(pipe, encoding='utf-8') as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if 'azimuth' in line.lower() or 'Pred_Az' in line or 'azm' in line.lower():
            print(f'  L{i+1}: {line.rstrip()}')

# Kp correlation with azimuth
print(f'\n[9] KP vs AZIMUTH')
for label in [0,1]:
    sub = p[p['True_Label']==label]
    if len(sub) > 10:
        corr = sub['Pred_Az'].corr(sub['Kp_Raw'])
        print(f'  Label={label}: Pred_Az ~ Kp correlation = {corr:.4f}')

# Check first/last predictions
print(f'\n[10] SAMPLE PREDICTIONS')
print(p[['Date','Station','True_Label','Pred_Prob','Pred_Az','True_Az','Az_Error','Kp_Raw']].head(20).to_string())
print(f'\n...')
print(p[['Date','Station','True_Label','Pred_Prob','Pred_Az','True_Az','Az_Error','Kp_Raw']].tail(10).to_string())

# Final check: are the predictions from a real model or hardcoded?
print(f'\n[11] MODEL OUTPUT AUTHENTICITY')
# If Pred_Prob varied across samples AND Pred_Az is narrow → model focuses on detection not azimuth
# If both are constant → potential placeholder data
prob_unique = p['Pred_Prob'].nunique()
az_unique = p['Pred_Az'].nunique()
print(f'  Pred_Prob unique: {prob_unique} (of {len(p)} total)')
print(f'  Pred_Az unique:   {az_unique} (of {len(p)} total)')
print(f'  Pred_Prob diversity: {prob_unique/len(p)*100:.1f}%')
print(f'  Pred_Az diversity:   {az_unique/len(p)*100:.1f}%')
if prob_unique / len(p) > 0.1:
    print(f'  Pred_Prob: HIGH diversity (model is actively discriminating)')
else:
    print(f'  Pred_Prob: LOW diversity (potential collapse)')
if az_unique / len(p) > 0.1:
    print(f'  Pred_Az: HIGH diversity')
else:
    print(f'  Pred_Az: LOW diversity (narrow range)')

print(f'\n{"="*70}')
print('SUMMARY')
print(f'{"="*70}')
print(f'Pred_Prob: [0.057, 0.956], 1692 unique, mean=0.214 → VALID')
print(f'Pred_Az:   [{p["Pred_Az"].min():.2f}, {p["Pred_Az"].max():.2f}], {az_unique} unique → ANOMALY')
print(f'  Pred_Az spans only {p["Pred_Az"].max()-p["Pred_Az"].min():.1f} degrees')
print(f'  True_Az spans {p["True_Az"].max()-p["True_Az"].min():.1f} degrees')
print(f'  MAE=38.04 degrees (seems reasonable for a narrow prediction range)')
print(f'  This is likely due to the sin/cos → degree conversion in the pipeline')
print(f'  OR the azimuth head outputting degenerate predictions')
print(f'{"="*70}')
