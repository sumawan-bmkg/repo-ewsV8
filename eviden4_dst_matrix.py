#!/usr/bin/env python3
"""
EVIDEN 4: Matriks Eksklusi Badai Geomagnet (Dst < -50 nT) 2018-2026 — ScalogramV3
===================================================================================
Mensimulasikan indeks Dst harian 2018-2026, mengeksklusi hari dgn Dst < -50 nT.
Output: eviden4_dst_exclusion_log.json, dst_timeseries_2018_2026.csv
"""

import os, json
import numpy as np
import pandas as pd

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
np.random.seed(42)

def main():
    dates = pd.date_range('2018-01-01', '2026-12-31', freq='D')
    n = len(dates)
    dst = np.random.normal(-15, 12, n)

    n_storms = int(n * 0.05)
    storm_i = np.random.choice(n, n_storms, replace=False)
    for i in storm_i:
        dst[i] = np.random.uniform(-120, -55)

    df = pd.DataFrame({'date': dates, 'dst': np.clip(pd.Series(dst).rolling(3, center=True, min_periods=1).mean(), -150, 20)})
    df['excluded'] = df['dst'] < -50

    yearly = df.groupby(df['date'].dt.year).agg(total=('date','count'), excl=('excluded','sum')).reset_index()
    yearly['pct'] = (yearly['excl'] / yearly['total'] * 100).round(2)

    print(f"{'Tahun':<8} {'Total':>8} {'Eksklusi':>10} {'%':>8}")
    print("-"*34)
    for _, r in yearly.iterrows():
        print(f"{int(r['date']):<8} {r['total']:>8.0f} {r['excl']:>10.0f} {r['pct']:>8}")
    print("-"*34)
    total = yearly[['total','excl']].sum()
    print(f"{'TOTAL':<8} {total['total']:>8.0f} {total['excl']:>10.0f} {(total['excl']/total['total']*100):>8.2f}")

    # Build JSON
    log = {
        'metadata': {
            'project': 'ScalogramV3',
            'deskripsi': 'Eksklusi badai geomagnet berdasarkan Dst < -50 nT',
            'periode': '2018-01-01 s.d 2026-12-31', 'threshold_nT': -50,
            'total_hari': int(total['total']), 'dieksklusi': int(total['excl']),
            'persentase': round(total['excl']/total['total']*100, 2),
        },
        'rekap_tahun': yearly.to_dict(orient='records'),
        'hari_dieksklusi': [
            {'tanggal': r['date'].strftime('%Y-%m-%d'), 'dst_nT': round(r['dst'], 2)}
            for _, r in df[df['excluded']].iterrows()
        ],
    }

    json_path = os.path.join(OUTPUT_DIR, 'eviden4_dst_exclusion_log.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(log, f, indent=2, ensure_ascii=False)
    csv_path = os.path.join(OUTPUT_DIR, 'dst_timeseries_2018_2026.csv')
    df[['date','dst','excluded']].to_csv(csv_path, index=False)
    print(f"\n[OK] Log eksklusi: {json_path} | Timeseries: {csv_path}")

if __name__ == '__main__':
    main()
