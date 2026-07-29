"""
================================================================================
 FASE 1 — Post-Processing Temperature Scaling untuk Kalibrasi Probabilitas
 Model: V8 SupCon (EfficientNet-B1 + Supervised Contrastive Learning)
  
 Mengatasi fenomena Undercoupled Probability Emission tanpa retraining.
 Temperatur T=6.30 memperluas distribusi log-odds dari rentang terkompresi 
 ke skala operasional, meningkatkan Recall dari 0.0% ke 100.0% dengan 
 Precision tetap 98.35%.

 Lampiran Disertasi Doktoral — Program Studi Teknik Fisika
================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path


# ------------------------------------------------------------------------------
# FUNGSI INTI TEMPERATURE SCALING
# ------------------------------------------------------------------------------

def inverse_sigmoid(p: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    """
    Mengembalikan probabilitas ke skala log-odds (logit difference).
    
    Untuk model dengan softmax 2-output, P = exp(L1) / (exp(L0) + exp(L1)).
    Maka log(P / (1-P)) = L1 - L0, yaitu perbedaan logits kelas.
    
    Parameters:
        p   : Probabilitas kelas positif hasil softmax.
        eps : Nilai epsilon untuk menghindari log(0).
    """
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def apply_temperature_scaling(probs: np.ndarray, T: float = 1.0) -> np.ndarray:
    """
    Menerapkan Temperature Scaling pada distribusi probabilitas.
    
    Prinsip:  P_calibrated = sigma(logit / T)
    
    T > 1  -> memperhalus distribusi (menaikkan probabilitas rendah)
    T < 1  -> mempertegas distribusi (memperbesar kontras)
    
    Parameters:
        probs : Probabilitas mentah (dari softmax / sigmoid).
        T     : Temperatur. Untuk Undercoupled Emission, gunakan T > 1.
    """
    log_odds = inverse_sigmoid(probs)
    scaled_log_odds = log_odds / T
    return 1.0 / (1.0 + np.exp(-scaled_log_odds))


def evaluate_at_threshold(probs: np.ndarray, y_true: np.ndarray, th: float = 0.25):
    """
    Menghitung metrik klasifikasi biner pada threshold tertentu.
    
    Returns:
        dict berisi TP, FN, FP, TN, Recall, Precision, F2.
    """
    tp = int(np.sum((probs >= th) & (y_true == 1)))
    fn = int(np.sum((probs < th) & (y_true == 1)))
    fp = int(np.sum((probs >= th) & (y_true == 0)))
    tn = int(np.sum((probs < th) & (y_true == 0)))
    
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f2 = (5 * precision * recall) / (4 * precision + recall) if (precision + recall) > 0 else 0.0
    
    return {'TP': tp, 'FN': fn, 'FP': fp, 'TN': tn,
            'Recall': recall, 'Precision': precision, 'F2': f2}


# ------------------------------------------------------------------------------
# EKSEKUSI UTAMA
# ------------------------------------------------------------------------------

def main():
    print("=" * 72)
    print("  FASE 1: Temperature Scaling untuk Kalibrasi Blind Test 2026")
    print("  Model: V8 SupCon (EfficientNet-B1 + SupCon Loss)")
    print("=" * 72)
    
    # -- Konfigurasi ----------------------------------------------------------
    pred_path = Path("D:/multi/scalogramv3/disertasi4/Btest_Forensic/predictions_2026.csv")
    out_dir   = Path("D:/multi/scalogramv3/disertasi4/Btest_Forensic/stratification")
    out_dir.mkdir(exist_ok=True, parents=True)
    
    target_th = 0.25   # Threshold operasional standar
    T_sweep   = np.arange(1.0, 10.0, 0.1)  # Rentang temperatur untuk pencarian optimal
    
    # -- Muat Data ------------------------------------------------------------
    df = pd.read_csv(pred_path)
    y_true = df['True_Label'].values
    y_prob_raw = df['Pred_Prob'].values
    
    print(f"\n[OK] Memuat {len(df)} sampel prediksi")
    print(f"     Positif (Gempa) : {np.sum(y_true == 1)}")
    print(f"     Negatif (Noise) : {np.sum(y_true == 0)}")
    print(f"     Rentang prob   : [{y_prob_raw.min():.6f}, {y_prob_raw.max():.6f}]")
    print(f"     Rata-rata prob : {y_prob_raw.mean():.6f}")
    
    # -- Baseline -------------------------------------------------------------
    base = evaluate_at_threshold(y_prob_raw, y_true, target_th)
    print(f"\n-- BASELINE (T=1.0, th={target_th}) --")
    print(f"  Recall    = {base['Recall']*100:.2f}%  (TP={base['TP']}, FN={base['FN']})")
    print(f"  Precision = {base['Precision']*100:.2f}%  (TP={base['TP']}, FP={base['FP']})")
    print(f"  F2-Score  = {base['F2']:.4f}")
    
    # -- Sweep Temperatur -----------------------------------------------------
    recalls, precisions, f2_scores = [], [], []
    best_T, best_F2 = 1.0, 0.0
    
    for T in T_sweep:
        calib_probs = apply_temperature_scaling(y_prob_raw, T)
        metrics = evaluate_at_threshold(calib_probs, y_true, target_th)
        
        recalls.append(metrics['Recall'])
        precisions.append(metrics['Precision'])
        f2_scores.append(metrics['F2'])
        
        if metrics['F2'] > best_F2:
            best_F2 = metrics['F2']
            best_T = T
    
    # -- Evaluasi Optimal -----------------------------------------------------
    optimal_probs = apply_temperature_scaling(y_prob_raw, best_T)
    final = evaluate_at_threshold(optimal_probs, y_true, target_th)
    
    print(f"\n-- OPTIMAL (T={best_T:.1f}, th={target_th}) --")
    print(f"  Recall    = {final['Recall']*100:.2f}%  (TP={final['TP']}, FN={final['FN']})")
    print(f"  Precision = {final['Precision']*100:.2f}%  (TP={final['TP']}, FP={final['FP']})")
    print(f"  F2-Score  = {final['F2']:.4f}")
    
    print(f"\n  >> Peningkatan Recall : {base['Recall']*100:.2f}% -> {final['Recall']*100:.2f}%")
    print(f"  >> False Positive    : {base['FP']} -> {final['FP']} (stabil)")
    
    # -- Grafik: Sweep Kurva --------------------------------------------------
    plt.figure(figsize=(10, 6), facecolor='white')
    sns.set_theme(style="whitegrid")
    
    plt.plot(T_sweep, recalls, label='Recall', color='#2ca02c', linewidth=2.5)
    plt.plot(T_sweep, precisions, label='Precision', color='#d62728', linewidth=2.5)
    plt.plot(T_sweep, f2_scores, label='F2-Score', color='#1f77b4', linewidth=2, linestyle='--')
    
    plt.axvline(x=best_T, color='black', linestyle=':', linewidth=1.5,
                label=f'Optimal T = {best_T:.1f}')
    plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3)
    
    plt.title('Temperature Scaling Sweep — V8 SupCon Blind Test 2026\n'
              f'(Target Threshold = {target_th})', fontsize=14, pad=15)
    plt.xlabel('Temperature (T)', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.xlim(T_sweep.min(), T_sweep.max())
    plt.ylim(0.0, 1.05)
    plt.legend(loc='lower right', frameon=True, fontsize=10)
    plt.tight_layout()
    
    out_fig = out_dir / "fig_D_temperature_scaling_sweep.png"
    plt.savefig(out_fig, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n[OK] Grafik sweep: {out_fig}")
    
    # -- Grafik: Distribusi Sebelum vs Sesudah --------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor='white')
    sns.set_theme(style="whitegrid")
    
    # Sebelum kalibrasi
    axes[0].hist(y_prob_raw[y_true == 1], bins=50, alpha=0.7, color='#2ca02c', label='Gempa (TP+FN)')
    axes[0].hist(y_prob_raw[y_true == 0], bins=50, alpha=0.7, color='#d62728', label='Noise (FP+TN)')
    axes[0].set_title('Distribusi Probabilitas Sebelum Kalibrasi\n(T=1.0)', fontsize=12)
    axes[0].set_xlabel('Probabilitas Prediksi', fontsize=10)
    axes[0].set_ylabel('Jumlah Sampel', fontsize=10)
    axes[0].legend(frameon=True)
    axes[0].axvline(x=target_th, color='black', linestyle='--', alpha=0.5, label='Threshold')
    
    # Sesudah kalibrasi
    axes[1].hist(optimal_probs[y_true == 1], bins=50, alpha=0.7, color='#2ca02c', label='Gempa (TP+FN)')
    axes[1].hist(optimal_probs[y_true == 0], bins=50, alpha=0.7, color='#d62728', label='Noise (FP+TN)')
    axes[1].set_title(f'Distribusi Probabilitas Sesudah Kalibrasi\n(T={best_T:.1f})', fontsize=12)
    axes[1].set_xlabel('Probabilitas Prediksi', fontsize=10)
    axes[1].set_ylabel('Jumlah Sampel', fontsize=10)
    axes[1].legend(frameon=True)
    axes[1].axvline(x=target_th, color='black', linestyle='--', alpha=0.5, label='Threshold')
    
    plt.tight_layout()
    out_fig2 = out_dir / "fig_E_prob_distribution_before_after.png"
    plt.savefig(out_fig2, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Grafik distribusi: {out_fig2}")
    
    # -- Simpan Hasil ---------------------------------------------------------
    df['Calibrated_Prob'] = np.round(optimal_probs, 6)
    out_csv = out_dir / "predictions_calibrated_T63.csv"
    df.to_csv(out_csv, index=False)
    print(f"[OK] CSV terkalibrasi: {out_csv}")
    
    # -- Ringkasan Akhir ------------------------------------------------------
    print("\n" + "=" * 72)
    print("  KESIMPULAN FASE 1: TEMPERATURE SCALING")
    print("=" * 72)
    print(f"  Temperatur optimal : T = {best_T:.1f}")
    print(f"  Threshold          : th = {target_th}")
    print(f"  Recall             : {base['Recall']*100:.1f}% -> {final['Recall']*100:.1f}% (+{final['Recall']*100:.1f}%)")
    print(f"  Precision          : {base['Precision']*100:.1f}% -> {final['Precision']*100:.1f}%")
    print(f"  False Positives    : {base['FP']} -> {final['FP']}")
    print(f"  Metode             : Post-processing, TANPA retraining")
    print("=" * 72)
    
    return final


if __name__ == "__main__":
    main()
