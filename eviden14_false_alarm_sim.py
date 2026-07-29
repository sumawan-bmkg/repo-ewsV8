#!/usr/bin/env python3
"""
EVIDEN 14: Simulasi False Alarm Rate Real-time 12 Bulan — ScalogramV3
======================================================================
Memodelkan performa model ScalogramV3 deploy 12 bulan.
Output: eviden14_false_alarm_trend.png, eviden14_false_alarm_report.csv
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
np.random.seed(42)

N_MONTHS = 12
DAYS = 30

def main():
    results = []
    for m in range(1, N_MONTHS+1):
        n_eq = np.random.randint(1, 4)
        eq_days = sorted(np.random.choice(DAYS, n_eq, replace=False))
        pred = np.zeros(DAYS)
        actual = np.zeros(DAYS)
        actual[eq_days] = 1

        for d in range(DAYS):
            fp_p = np.clip(0.08 + np.random.normal(0, 0.03), 0.02, 0.20)
            tp_p = np.clip(0.55 + np.random.normal(0, 0.10), 0.25, 0.85)
            pred[d] = 1 if (actual[d]==1 and np.random.random()<tp_p) or (actual[d]==0 and np.random.random()<fp_p) else 0

        tp = int(np.sum((pred==1)&(actual==1)))
        fp = int(np.sum((pred==1)&(actual==0)))
        fn = int(np.sum((pred==0)&(actual==1)))
        far = fp/(fp+tp+fn)*100 if (fp+tp+fn)>0 else 0
        dr = tp/(tp+fn)*100 if (tp+fn)>0 else 0
        results.append({'bulan':m, 'gempa':n_eq, 'deteksi':tp, 'miss':fn,
                        'fp':fp, 'far_pct':round(far,2), 'detection_pct':round(dr,2)})

    df = pd.DataFrame(results)
    avg_far = df['far_pct'].mean()
    total_fp = df['fp'].sum()

    print(f"{'Bulan':<6} {'Gempa':>6} {'Deteksi':>8} {'FP':>6} {'FAR(%)':>8} {'Detec(%)':>8}")
    print("-"*42)
    for _, r in df.iterrows():
        print(f"{r['bulan']:<6} {r['gempa']:>6} {r['deteksi']:>8} {r['fp']:>6} {r['far_pct']:>8.2f} {r['detection_pct']:>8.2f}")
    print("-"*42)
    print(f"{'RATA':<6} {'':>6} {'':>8} {total_fp:>6} {avg_far:>8.2f} {df['detection_pct'].mean():>8.2f}")

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,5))
    m = df['bulan']
    ax1.bar(m, df['far_pct'], color='#e74c3c', alpha=0.7, edgecolor='white')
    ax1.axhline(avg_far, color='#2c3e50', ls='--', lw=1, label=f'Rata-rata FAR = {avg_far:.1f}%')
    ax1.set_xlabel('Bulan ke-', fontsize=11, fontweight='bold')
    ax1.set_ylabel('False Alarm Rate (%)', fontsize=11, fontweight='bold')
    ax1.set_title('False Alarm Rate ScalogramV3 (12 Bulan)', fontsize=12, fontweight='bold')
    ax1.set_xticks(m); ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3, axis='y')

    w=0.35
    ax2.bar(m-w/2, df['fp'], w, color='#e74c3c', alpha=0.7, label='Alarm Palsu (FP)')
    ax2.bar(m+w/2, df['deteksi'], w, color='#27ae60', alpha=0.7, label='Deteksi Berhasil (TP)')
    ax2.set_xlabel('Bulan ke-', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Jumlah', fontsize=11, fontweight='bold')
    ax2.set_title(f'ScalogramV3: FP vs TP\nTotal FP={total_fp} | Rata FP/bl={total_fp/N_MONTHS:.1f}', fontsize=12, fontweight='bold')
    ax2.set_xticks(m); ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Simulasi False Alarm Rate — ScalogramV3 Deployment Real-time 12 Bulan\n'
                 '(EfficientNet-B1 + Mini-ResNet | Threshold 0.5)',
                 fontsize=13, fontweight='bold', y=1.08)
    plt.tight_layout()
    png_path = os.path.join(OUTPUT_DIR, 'eviden14_false_alarm_trend.png')
    fig.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    csv_path = os.path.join(OUTPUT_DIR, 'eviden14_false_alarm_report.csv')
    df.to_csv(csv_path, index=False)
    print(f"\n[OK] False alarm trend ScalogramV3: {png_path}")
    print(f"[OK] CSV report: {csv_path}")

if __name__ == '__main__':
    main()
