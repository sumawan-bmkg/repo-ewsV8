"""Periode data blind test 2026 — investigasi lengkap"""
import pandas as pd, numpy as np
from datetime import timedelta

p = pd.read_csv('D:/multi/scalogramv3/blind_test_2026_v8_results/v8supcon_2026_predictions.csv')
p['Date'] = pd.to_datetime(p['Date'])

print('='*70)
print('PERIODE DATA BLIND TEST 2026')
print('='*70)

# 1. Date range
print(f'\n[1] DATE RANGE')
print(f'  First date:  {p["Date"].min()}')
print(f'  Last date:   {p["Date"].max()}')
print(f'  Total days:  {(p["Date"].max() - p["Date"].min()).days + 1}')
print(f'  Unique dates: {p["Date"].nunique()}')

# 2. Expected vs actual coverage
all_dates = pd.date_range(start=p['Date'].min(), end=p['Date'].max(), freq='D')
present = p['Date'].dt.date.unique()
missing_dates = [d for d in all_dates if d.date() not in {pd.Timestamp(x).date() for x in present}]
print(f'\n[2] COVERAGE')
print(f'  Expected dates: {len(all_dates)}')
print(f'  Actual dates:   {len(present)}')
print(f'  Missing dates:  {len(missing_dates)}')
if len(missing_dates) > 0:
    print(f'  Missing: {[str(d.date()) for d in missing_dates[:20]]}')
    if len(missing_dates) > 20:
        print(f'  ... and {len(missing_dates)-20} more')

# 3. Samples per day
print(f'\n[3] SAMPLES PER DAY')
daily_counts = p.groupby(p['Date'].dt.date).size()
print(f'  Mean:   {daily_counts.mean():.1f}')
print(f'  Std:    {daily_counts.std():.1f}')
print(f'  Min:    {daily_counts.min()} (on {daily_counts.idxmin()})')
print(f'  Max:    {daily_counts.max()} (on {daily_counts.idxmax()})')
print(f'  Expected: ~24 (one per station)')
# Days with abnormal counts
abnormal = daily_counts[(daily_counts < 20) | (daily_counts > 28)]
if len(abnormal) > 0:
    print(f'  Days with abnormal counts (<20 or >28): {len(abnormal)}')
    for d, c in abnormal.items():
        print(f'    {d}: {c} samples')

# 4. Station coverage per day
print(f'\n[4] STATION COVERAGE PER DAY')
daily_stn = p.groupby(p['Date'].dt.date)['Station'].nunique()
missing_stn = daily_stn[daily_stn < 24]
if len(missing_stn) > 0:
    print(f'  Days with missing stations: {len(missing_stn)}')
    for d, n in missing_stn.items():
        present_stn = p[p['Date'].dt.date == d]['Station'].unique()
        all_stn = set(p['Station'].unique())
        missing = all_stn - set(present_stn)
        print(f'    {d}: {n}/24 stations, missing: {sorted(missing)}')
else:
    print(f'  All {len(daily_stn)} days have full 24 stations')

# 5. Negative samples timeline
print(f'\n[5] NEGATIVE SAMPLES (event=0)')
neg = p[p['True_Label'] == 0]
neg_by_date = neg.groupby(neg['Date'].dt.date).size()
print(f'  Total negatives: {len(neg)}')
print(f'  Dates with negatives:')
for d, c in neg_by_date.items():
    neg_stn = neg[neg['Date'].dt.date == d]['Station'].unique()
    print(f'    {d}: {c} negatives, stations: {sorted(neg_stn)}')

# 6. Month-by-month
print(f'\n[6] MONTHLY BREAKDOWN')
p['Month'] = p['Date'].dt.month_name()
for month, grp in p.groupby('Month'):
    n_days = grp['Date'].dt.date.nunique()
    n_stn = grp['Station'].nunique()
    n_events = (grp['True_Label'] == 1).sum()
    n_neg = (grp['True_Label'] == 0).sum()
    n_det = (grp['Pred_Binary'] == 1).sum()
    print(f'  {month:10s}: {len(grp):5d} samples, {n_days:2d} days, {n_stn:2d} stn, '
          f'events={n_events:4d}, neg={n_neg:2d}, detections={n_det:3d}')

# 7. Station completeness
print(f'\n[7] STATION COMPLETENESS')
stn_counts = p.groupby('Station').size()
print(f'  Expected per station: ~120 days')
stn_days = p.groupby('Station')['Date'].apply(lambda x: x.dt.date.nunique())
for stn in sorted(stn_days.index):
    cnt = stn_days[stn]
    flag = 'OK' if cnt >= 119 else f'MISSING {120-cnt} days'
    print(f'  {stn:4s}: {cnt:3d} days with data [{flag}]')

# 8. Period anomalies
print(f'\n[8] PERIOD ANOMALIES')
# Check for gaps > 1 day
dates_sorted = sorted(pd.to_datetime(list(present)))
gaps = []
for i in range(1, len(dates_sorted)):
    gap = (dates_sorted[i] - dates_sorted[i-1]).days
    if gap > 1:
        gaps.append((dates_sorted[i-1], dates_sorted[i], gap))
if gaps:
    print(f'  Gaps > 1 day: {len(gaps)}')
    for g in gaps:
        print(f'    {g[0].date()} -> {g[1].date()}: {g[2]} days gap')
else:
    print(f'  No gaps: consecutive daily coverage')

# 9. Quarter breakdown
print(f'\n[9] PERIOD SUMMARY')
for q, grp in p.groupby(p['Date'].dt.quarter):
    n_days = grp['Date'].dt.date.nunique()
    print(f'  Q{q} 2026: {len(grp):5d} samples, {n_days:2d} days, '
          f'events={(grp["True_Label"]==1).sum():4d}, '
          f'detections={(grp["Pred_Binary"]==1).sum():3d}')

print(f'\n{"="*70}')
print(f'PERIOD CONCLUSION')
print(f'{"="*70}')
total_days = len(present)
expected = len(all_dates)
pct = total_days / expected * 100
print(f'  Period: {p["Date"].min().date()} to {p["Date"].max().date()}')
print(f'  Coverage: {total_days}/{expected} days ({pct:.1f}%)')
print(f'  Consecutive: {"YES" if not gaps else f"NO ({len(gaps)} gaps)"}')
print(f'  Full stations: {"ALL DAYS" if stn_days.min() == 24 else stn_days.min()}')
if daily_counts.std() < 5 and daily_counts.min() >= 20:
    print(f'  Samples/day: Consistent (mean={daily_counts.mean():.1f}, std={daily_counts.std():.1f})')
else:
    print(f'  Samples/day: Inconsistent (mean={daily_counts.mean():.1f}, std={daily_counts.std():.1f})')
print(f'{"="*70}')
