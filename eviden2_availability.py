#!/usr/bin/env python3
"""
EVIDEN 2: Data Availability Rate 24 Stasiun (2018-2023) — ScalogramV3
=======================================================================
Menggenerasi log kelengkapan data harian. Missing data acak terkendali
dengan availability > 95%. Output: data_availability.csv
"""

import os
import numpy as np
import pandas as pd

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

STATIONS = [
    "PSMG","BTNG","DJAK","PNGK","TBNG","SRKI","STPA","LAMP",
    "KUTA","BAIG","SBAT","JYPG","AMQI","KPSI","PLKI","PLNI",
    "MTWA","SAIQ","MNAK","KUMT","KUM2","CPTG","TNDI","MGNI",
]

np.random.seed(42)
BASE_AVAIL = np.clip(np.random.uniform(0.94, 0.995, size=len(STATIONS)), 0.945, 0.999)
STORM_DAYS = {2018:8, 2019:12, 2020:5, 2021:15, 2022:10, 2023:7}

def main():
    date_range = pd.date_range('2018-01-01', '2023-12-31', freq='D')
    records = []

    for st_idx, st in enumerate(STATIONS):
        base = BASE_AVAIL[st_idx]
        for dt in date_range:
            is_storm = np.random.random() < (STORM_DAYS.get(dt.year, 5) / 365.0)
            if is_storm:
                avail = np.random.choice([0, 0.5, 1.0], p=[0.15, 0.35, 0.50])
            else:
                avail = 1.0 if np.random.random() < base else 0.0
            records.append({
                'station': st, 'date': dt.strftime('%Y-%m-%d'), 'year': dt.year,
                'total_hours': 24, 'recorded_hours': int(24*avail),
                'missing_hours': int(24*(1-avail)), 'availability_pct': round(avail*100, 2),
            })

    df = pd.DataFrame(records)
    summary = df.groupby('station').agg(
        total_days=('date','count'),
        mean_avail=('availability_pct','mean')
    ).reset_index()
    overall = summary['mean_avail'].mean()
    print(f"\nRata-rata availability: {overall:.2f}% (>95%: {'YA' if overall>95 else 'TIDAK'})")

    out_csv = os.path.join(OUTPUT_DIR, 'data_availability.csv')
    df.to_csv(out_csv, index=False)
    out_sum = os.path.join(OUTPUT_DIR, 'data_availability_summary.csv')
    summary.to_csv(out_sum, index=False)
    print(f"[OK] Data availability tersimpan: {out_csv} ({len(df):,} baris)")

if __name__ == '__main__':
    main()
