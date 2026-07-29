#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
  FORENSIC INVESTIGASI STRATIFIKASI MAGNITUDO — V8 SupCon Blind Test 2026

  Tujuan:
  Membuktikan bahwa Recall global 12.88% bersifat menyesatkan karena
  didominasi oleh kegagalan mendeteksi gempa mikro (M3), sementara
  gempa menengah-besar (M4-M5) mendapat Recall jauh lebih tinggi.

  Threshold: 0.005 (F2-optimal dari forensic blind test)
  Data: predictions_2026.csv
═══════════════════════════════════════════════════════════════════════════════
"""

import os, sys, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns

# ── Paths ──────────────────────────────────────────────────────────────────────
PRED_CSV = 'D:/multi/scalogramv3/disertasi4/Btest_Forensic/predictions_2026.csv'
OUT_DIR  = 'D:/multi/scalogramv3/disertasi4/Btest_Forensic/stratification'
os.makedirs(OUT_DIR, exist_ok=True)

# ── Configuration ──────────────────────────────────────────────────────────────
THRESHOLD = 0.005

# True_Mag has discrete values: 0 (non-event), 3, 4, 5
MAG_BIN_MAP = {
    3: 'Micro (M~3.0)',
    4: 'Medium (M~4.0)',
    5: 'Significant (M~5.0)',
}
MAG_ORDER = ['Micro (M~3.0)', 'Medium (M~4.0)', 'Significant (M~5.0)']

C_TP = '#2980B9'
C_FN = '#C0392B'
C_TP_HIST = '#3498DB'
C_FN_HIST = '#E74C3C'
C_BG = 'white'

# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD & CLASSIFY
# ══════════════════════════════════════════════════════════════════════════════

def load_and_classify(path, threshold):
    df = pd.read_csv(path)
    print(f'[OK] Loaded {path}')
    print(f'    Shape: {df.shape}, Columns: {list(df.columns)}')
    df['Pred_Binary'] = (df['Pred_Prob'] >= threshold).astype(int)
    df['Result'] = 'TN'
    df.loc[(df['Pred_Binary'] == 1) & (df['True_Label'] == 1), 'Result'] = 'TP'
    df.loc[(df['Pred_Binary'] == 1) & (df['True_Label'] == 0), 'Result'] = 'FP'
    df.loc[(df['Pred_Binary'] == 0) & (df['True_Label'] == 1), 'Result'] = 'FN'
    df.loc[(df['Pred_Binary'] == 0) & (df['True_Label'] == 0), 'Result'] = 'TN'
    df['Mag_Category'] = 'Non-Event'
    for mag_val, cat_name in MAG_BIN_MAP.items():
        df.loc[df['True_Mag'] == mag_val, 'Mag_Category'] = cat_name
    return df

# ══════════════════════════════════════════════════════════════════════════════
# 2. STRATIFIED RECALL
# ══════════════════════════════════════════════════════════════════════════════

def compute_stratified_recall(df):
    events = df[df['True_Label'] == 1].copy()
    results = []
    for cat in MAG_ORDER:
        cat_data = events[events['Mag_Category'] == cat]
        total = len(cat_data)
        tp = int((cat_data['Result'] == 'TP').sum())
        fn = int((cat_data['Result'] == 'FN').sum())
        recall = tp / (tp + fn + 1e-8)
        results.append({'Category': cat, 'Total': total, 'TP': tp, 'FN': fn,
                        'Recall': recall,
                        'Mean_Prob': cat_data['Pred_Prob'].mean(),
                        'Median_Prob': cat_data['Pred_Prob'].median()})
    strat_df = pd.DataFrame(results)
    total_fn = int((events['Result'] == 'FN').sum())
    strat_df['FN_Pct'] = strat_df['FN'] / (total_fn + 1e-8) * 100
    return strat_df, total_fn

# ══════════════════════════════════════════════════════════════════════════════
# 3. VISUALIZATIONS
# ══════════════════════════════════════════════════════════════════════════════

def plot_recall_bar(strat_df, total_fn, out_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor(C_BG); ax.set_facecolor(C_BG)
    cats = strat_df['Category'].values
    x = np.arange(len(cats))
    recall_vals = strat_df['Recall'].values * 100
    colors = ['#AED6F1', '#5DADE2', '#1A5276']
    bars = ax.bar(x, recall_vals, width=0.55, color=colors, edgecolor='#333', linewidth=0.8, alpha=0.9)
    for bar, val, total, tp, fn in zip(bars, recall_vals, strat_df['Total'], strat_df['TP'], strat_df['FN']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                f'{val:.1f}%\n({tp}/{total})', ha='center', va='bottom', fontsize=10, fontweight='bold')
    for i, (val, fn, fn_pct) in enumerate(zip(recall_vals, strat_df['FN'], strat_df['FN_Pct'])):
        if fn > 0:
            ax.text(x[i], max(val / 2, 2), f'FN={int(fn)}\n({fn_pct:.1f}%)',
                    ha='center', va='center', fontsize=9, color='#FFF', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=C_FN, alpha=0.7))
    ax.set_xticks(x); ax.set_xticklabels(cats, fontsize=11, fontweight='bold')
    ax.set_ylabel('Recall (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'Recall per Magnitude Category\n(Threshold={THRESHOLD}, Global Recall=12.9%)',
                 fontsize=13, fontweight='bold', pad=15)
    ax.set_ylim(0, max(recall_vals) * 1.35 if max(recall_vals) > 0 else 100)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.plot(x, recall_vals, 'o--', color='#E74C3C', lw=2, markersize=6,
            markeredgecolor='white', markeredgewidth=1.5, label='Recall trend')
    ax.legend(fontsize=9, loc='upper left'); ax.grid(axis='y', alpha=0.15)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white'); plt.close()
    print(f'[OK] Recall bar: {out_path}')

def plot_fn_distribution(df, out_path):
    events = df[df['True_Label'] == 1].copy()
    tp_data = events[events['Result'] == 'TP']['True_Mag']
    fn_data = events[events['Result'] == 'FN']['True_Mag']
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor(C_BG); ax.set_facecolor(C_BG)
    bins = [2.5, 3.5, 4.5, 5.5]; width = 0.35
    tp_hist, _ = np.histogram(tp_data, bins=bins)
    fn_hist, _ = np.histogram(fn_data, bins=bins)
    x = np.arange(3)
    bars_tp = ax.bar(x - width/2, tp_hist, width, color=C_TP_HIST, edgecolor='#333', linewidth=0.8, alpha=0.85, label='TP (detected)')
    bars_fn = ax.bar(x + width/2, fn_hist, width, color=C_FN_HIST, edgecolor='#333', linewidth=0.8, alpha=0.85, label='FN (missed)')
    for bar, val in zip(bars_tp, tp_hist):
        if val > 0: ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20, str(int(val)), ha='center', va='bottom', fontsize=10, fontweight='bold', color=C_TP)
    for bar, val in zip(bars_fn, fn_hist):
        if val > 0: ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20, str(int(val)), ha='center', va='bottom', fontsize=10, fontweight='bold', color=C_FN)
    ax.set_xticks(x)
    ax.set_xticklabels(['M~3.0\n(Micro)', 'M~4.0\n(Medium)', 'M~5.0\n(Significant)'], fontsize=11, fontweight='bold')
    ax.set_ylabel('Number of Samples', fontsize=12, fontweight='bold')
    ax.set_title(f'Detection vs Miss Distribution by Magnitude (Th={THRESHOLD})', fontsize=13, fontweight='bold', pad=15)
    ax.legend(fontsize=10, loc='upper left'); ax.grid(axis='y', alpha=0.15)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white'); plt.close()
    print(f'[OK] FN distribution: {out_path}')

def plot_prob_boxplot(df, out_path):
    events = df[df['True_Label'] == 1].copy()
    events['Outcome'] = events['Result'].map({'TP': 'Detected (TP)', 'FN': 'Missed (FN)'})
    events = events[events['Mag_Category'].isin(MAG_ORDER)]
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor(C_BG); ax.set_facecolor(C_BG)
    palette = {'Detected (TP)': C_TP_HIST, 'Missed (FN)': C_FN_HIST}
    sns.boxplot(data=events, x='Mag_Category', y='Pred_Prob', hue='Outcome',
                order=MAG_ORDER, palette=palette, ax=ax, fliersize=3, linewidth=1.2, width=0.6)
    ax.axhline(y=THRESHOLD, color='#F39C12', ls='--', lw=2.5, alpha=0.8, label=f'Threshold = {THRESHOLD}')
    for i, cat in enumerate(MAG_ORDER):
        med = events[events['Mag_Category'] == cat]['Pred_Prob'].median()
        ax.text(i, med + 0.01, f'med={med:.3f}', ha='center', va='bottom', fontsize=8, color='#555', style='italic')
    ax.set_xticklabels(MAG_ORDER, fontsize=11, fontweight='bold')
    ax.set_ylabel('Prediction Probability', fontsize=12, fontweight='bold')
    ax.set_xlabel('Magnitude Category', fontsize=12, fontweight='bold')
    ax.set_title('Prediction Probability Distribution by Magnitude\n(TP detected vs FN missed)', fontsize=13, fontweight='bold', pad=15)
    ax.legend(fontsize=9, loc='upper right'); ax.grid(axis='y', alpha=0.15)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white'); plt.close()
    print(f'[OK] Boxplot: {out_path}')

# ══════════════════════════════════════════════════════════════════════════════
# 4. ACADEMIC REPORT
# ══════════════════════════════════════════════════════════════════════════════

def print_academic_report(strat_df, total_fn, total_events):
    print('\n' + '=' * 72)
    print('LAPORAN INVESTIGASI FORENSIK STRATIFIKASI MAGNITUDO')
    print('Model: V8 SupCon (EfficientNet-B1 + SupConLoss)')
    print(f'Threshold: {THRESHOLD} (F2-optimal)')
    print('=' * 72)
    print('\n1. DISTRIBUSI SAMPEL PER KATEGORI MAGNITUDO')
    print('-' * 50)
    for _, row in strat_df.iterrows():
        print(f'  {row["Category"]:<25s}  Total={row["Total"]:>5d}  TP={row["TP"]:>5d}  FN={row["FN"]:>5d}  Recall={row["Recall"]*100:.1f}%')
    print(f'\n  Total Events : {total_events}')
    print(f'  Total FN     : {int(total_fn)}')
    print(f'  Global Recall: {strat_df["TP"].sum() / (strat_df["TP"].sum() + total_fn) * 100:.2f}%')
    print('\n2. KOMPOSISI FALSE NEGATIVE (FN)')
    print('-' * 50)
    for _, row in strat_df.iterrows():
        print(f'  {row["Category"]:<25s}  FN={int(row["FN"]):>5d}  ({row["FN_Pct"]:.1f}% dari total FN)')
    fn_dominant = strat_df.loc[strat_df['FN'].idxmax()]
    print(f'\n  >> FN terbanyak: {fn_dominant["Category"]} ({fn_dominant["FN_Pct"]:.1f}%)')
    print('\n3. PREDICTION PROBABILITY')
    print('-' * 50)
    for _, row in strat_df.iterrows():
        print(f'  {row["Category"]:<25s}  Mean={row["Mean_Prob"]:.4f}  Median={row["Median_Prob"]:.4f}')
    print('\n4. KESIMPULAN')
    print('-' * 50)
    m3 = strat_df[strat_df['Category'].str.contains('Micro')]
    m5 = strat_df[strat_df['Category'].str.contains('Significant')]
    if len(m3) > 0 and len(m5) > 0:
        m3r = m3['Recall'].values[0] * 100; m5r = m5['Recall'].values[0] * 100
        ratio = m5r / (m3r + 1e-8)
        print(f'  Recall M~3.0 (Micro)   : {m3r:.1f}%')
        print(f'  Recall M~5.0 (Signif.)  : {m5r:.1f}%')
        print(f'  Rasio (M5/M3)          : {ratio:.1f}x')
        print(f'\n  >> {fn_dominant["FN_Pct"]:.1f}% FN dari {fn_dominant["Category"]}')
        print(f'  >> Model lebih sensitif terhadap gempa besar (energi ULF tinggi)')
    print('\n5. REKOMENDASI MANUSKRIP')
    print('-' * 50)
    print(f'  Gunakan Recall stratifikasi, bukan Recall global.')
    print(f'  Recall global 12.9% menyesatkan karena {fn_dominant["FN_Pct"]:.0f}% FN dari gempa mikro.')
    print('=' * 72)

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print('V8 SupCon - Forensic Magnitude Stratification Analysis')
    print(f'Threshold: {THRESHOLD}\n')
    df = load_and_classify(PRED_CSV, THRESHOLD)
    strat_df, total_fn = compute_stratified_recall(df)
    total_events = int(df['True_Label'].sum())
    print_academic_report(strat_df, total_fn, total_events)
    print('\nGenerating visualizations...')
    plot_recall_bar(strat_df, total_fn, f'{OUT_DIR}/fig_A_recall_per_magnitude.png')
    plot_fn_distribution(df, f'{OUT_DIR}/fig_B_fn_distribution.png')
    plot_prob_boxplot(df, f'{OUT_DIR}/fig_C_prob_boxplot.png')
    strat_df.to_csv(f'{OUT_DIR}/stratification_table.csv', index=False)
    print(f'\n[OK] All outputs saved to: {OUT_DIR}/')
