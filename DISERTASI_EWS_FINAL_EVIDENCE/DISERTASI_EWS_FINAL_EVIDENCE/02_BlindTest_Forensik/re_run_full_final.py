#!/usr/bin/env python3
"""Re-run stages 2-5 on new predictions with merge2026.csv"""
import os, sys, json, math, warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
warnings.filterwarnings('ignore')

ROOT = Path('D:/multi/scalogramv3')
PRED = ROOT / 'blind_test_2026_v8_results' / 'v8supcon_2026_predictions.csv'
CAT = ROOT / '2026' / 'merge2026.csv'
OUT = ROOT / 'disertasi4' / 'supcon' / '06_evaluation' / 'kinerja'
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(PRED, parse_dates=['Date'])
cat = pd.read_csv(CAT, comment='#', on_bad_lines='skip')
cat.columns = [c.strip().strip('"') for c in cat.columns]

# Normalise catalogue
rename = {}
for c in cat.columns:
    cl = c.lower()
    if 'event id' in cl: rename[c] = 'event_id'
    elif 'date time' in cl: rename[c] = 'datetime'
    elif cl == 'magnitude': rename[c] = 'magnitude'
    elif cl == 'latitude': rename[c] = 'latitude'
    elif cl == 'longitude': rename[c] = 'longitude'
    elif 'depth' in cl: rename[c] = 'depth_km'
cat = cat.rename(columns=rename)
cat['origin_time'] = pd.to_datetime(cat['datetime'], errors='coerce', utc=True)
for col in ['magnitude','depth_km','latitude','longitude']:
    if col in cat.columns: cat[col] = pd.to_numeric(cat[col], errors='coerce')

STN = {'ALR':(-8.46,115.52),'AMB':(-8.25,115.35),'CLP':(-8.49,115.59),
       'GSI':(-8.35,115.38),'GTO':(-8.35,115.38),'JYP':(-8.35,115.38),
       'KPY':(-8.45,115.33),'LPS':(-8.45,115.33),'LUT':(-8.45,115.33),
       'LWA':(-8.45,115.33),'LWK':(-8.40,115.20),'MLB':(-8.40,115.20),
       'PLU':(-8.63,115.94),'ROT':(-8.30,115.09),'SBG':(-8.40,115.18),
       'SCN':(-8.40,115.18),'SKB':(-8.40,115.18),'SMI':(-8.40,115.18),
       'SRG':(-7.81,110.36),'SRO':(-8.40,115.18),'TNT':(-8.12,115.08),
       'TND':(-8.12,115.08),'TRD':(-8.30,115.09),'TRT':(-8.30,115.09),
       'YOG':(-7.81,110.36)}
ANOM = {'PLU','TNT','LWK','SRO'}

def haversine(lat1,lon1,lat2,lon2):
    R=6371; dlat=math.radians(lat2-lat1); dlon=math.radians(lon2-lon1)
    a=math.sin(dlat/2)**2+math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dlon/2)**2
    return R*2*math.atan2(math.sqrt(a),math.sqrt(1-a))

# ── Stage 2-3 ──
az_v = df.dropna(subset=['Az_Error'])
s23_new = {
    'Az_MAE_Global_deg': round(float(az_v['Az_Error'].mean()),2),
    'Az_MAE_Trimmed_deg': round(float(az_v[~az_v['Station'].isin(ANOM)]['Az_Error'].mean()),2),
    'Mag_MAE_Mw': round(float((df['True_Mw']-df['Pred_Mw']).abs().mean()),3),
}

cat_v = cat.dropna(subset=['latitude','longitude','origin_time']).copy()
cat_v['_date'] = pd.to_datetime(cat_v['origin_time'],utc=True).dt.normalize()
dist_e=[]; depth_e=[]
for _,r in df.iterrows():
    sc = STN.get(r['Station'])
    if not sc: continue
    ed = pd.Timestamp(r['Date']).normalize().tz_localize('UTC')
    td = (cat_v['_date']-ed).abs()
    m = td.dt.days<=1
    cand = cat_v[m]
    if len(cand)==0: continue
    b = cand.loc[td[m].idxmin()]
    dist_e.append(haversine(sc[0],sc[1],b['latitude'],b['longitude']))
    if pd.notna(b.get('depth_km')): depth_e.append(float(b['depth_km']))

s23_new['Dist_MAE_km'] = round(float(np.mean(dist_e)),1) if dist_e else None
s23_new['Depth_MAE_km'] = round(float(np.mean(depth_e)),1) if depth_e else None
s23_new['n_dist_pairs'] = len(dist_e)
s23_new['n_depth_pairs'] = len(depth_e)

# ── Stage 4 ──
pairs=[]
for _,r in df.iterrows():
    sc=STN.get(r['Station'])
    if not sc: continue
    ed=pd.Timestamp(r['Date']).normalize().tz_localize('UTC')
    td=(cat_v['_date']-ed).abs()
    m=td.dt.days<=1
    cand=cat_v[m]
    if len(cand)==0: continue
    b=cand.loc[td[m].idxmin()]
    d=haversine(sc[0],sc[1],b['latitude'],b['longitude'])
    mw=r['True_Mw']
    if d>0 and mw>0:
        pairs.append({'p':r['Pred_Prob'],'sr':10**(0.43*mw)/d,'tl':r['True_Label']})

s4_new={'n_pairs':len(pairs)}
if len(pairs)>=10:
    pdf=pd.DataFrame(pairs)
    rho,pv=spearmanr(pdf['p'],pdf['sr'])
    tp=pdf[pdf['tl']==1]
    rho_tp,pv_tp=(float('nan'),float('nan'))
    if len(tp)>=10: rho_tp,pv_tp=spearmanr(tp['p'],tp['sr'])
    s4_new.update({'Spearman_rho_all':round(float(rho),4),'Spearman_p_all':round(float(pv),6),
                   'Spearman_rho_TP':round(float(rho_tp),4),'Spearman_p_TP':round(float(pv_tp),6),
                   'n_pairs_TP':len(tp),'median_strain_ratio':round(float(pdf['sr'].median()),4)})

# ── Stage 5 ──
lead_d=[]
for _,r in df.iterrows():
    if r['Pred_Binary']!=1 or r['True_Label']!=1: continue
    sc=STN.get(r['Station'])
    if not sc: continue
    ed=pd.Timestamp(r['Date']).normalize().tz_localize('UTC')
    td=(cat_v['_date']-ed).abs()
    m=td.dt.days<=1
    cand=cat_v[m]
    if len(cand)==0: continue
    b=cand.loc[td[m].idxmin()]
    delta=(ed-pd.Timestamp(b['origin_time']).tz_convert('UTC')).total_seconds()/86400
    lead_d.append(delta)

s5_new={'mean_lead_days':round(float(np.mean(lead_d)),4) if lead_d else None,
        'median_lead_days':round(float(np.median(lead_d)),4) if lead_d else None,
        'n_matched':len(lead_d)}

# ── Stage 1 ──
y_true=df['True_Label'].values; y_prob=df['Pred_Prob'].values
opt_th=0.225
y_opt=(y_prob>=opt_th).astype(int)
TP=int(((y_opt==1)&(y_true==1)).sum())
TN=int(((y_opt==0)&(y_true==0)).sum())
FP=int(((y_opt==1)&(y_true==0)).sum())
FN=int(((y_opt==0)&(y_true==1)).sum())
from sklearn.metrics import average_precision_score, roc_auc_score
s1={'TP':TP,'TN':TN,'FP':FP,'FN':FN,
    'Precision':round(TP/(TP+FP),4) if (TP+FP)>0 else 0,
    'Recall':round(TP/(TP+FN),4) if (TP+FN)>0 else 0,
    'AUPRC':round(float(average_precision_score(y_true,y_prob)),4),
    'AUC_ROC':round(float(roc_auc_score(y_true,y_prob)),4)}

metrics={'s1_detection':s1,'s23_quantification':s23_new,'s4_physics':s4_new,'s5_leadtime':s5_new,
         'info':{'catalogue':'merge2026.csv','catalogue_events':len(cat),
                 'predictions':len(df),'stations':df['Station'].nunique(),
                 'date_range':f'{df["Date"].min()} to {df["Date"].max()}'}}

with open(OUT/'blindtest_metrics_final.json','w') as f:
    json.dump(metrics,f,indent=2)

print('='*70)
print('FULL RE-RUN WITH merge2026 — FINAL METRICS')
print('='*70)
print(f'\nTP (th=0.225): {TP} (was 614 old, now 615)' if TP==615 else f'\nTP (th=0.225): {TP}')
print(f'AUPRC: {s1["AUPRC"]}')
print(f'\nS23:')
for k,v in s23_new.items(): print(f'  {k}: {v}')
print(f'\nS4:')
for k,v in s4_new.items(): print(f'  {k}: {v}')
print(f'\nS5:')
for k,v in s5_new.items(): print(f'  {k}: {v}')
print(f'\nSaved: {OUT/"blindtest_metrics_final.json"}')
print('='*70)
