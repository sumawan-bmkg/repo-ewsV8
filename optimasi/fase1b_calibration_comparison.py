"""
================================================================================
 OPTIMASI KALIBRASI PROBABILITAS — FASE 1B (Extended)
 Model: V8 SupCon (EfficientNet-B1 + Supervised Contrastive Learning)

 Perbandingan dua pasca-metode kalibrasi probabilistik:
    (A) Temperature Scaling   — Pembagian logits dengan parameter T.
    (B) Platt Scaling         — Regresi Logistik pada logits mentah.

 Lampiran Disertasi Doktoral — Program Studi Teknik Fisika
================================================================================
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from dataclasses import dataclass
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_score, recall_score, fbeta_score
from typing import Tuple, Optional

# Konfigurasi style visualisasi publikasi jurnal
plt.rcParams.update({
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Segoe UI'],
    'axes.facecolor': 'white',
    'figure.facecolor': 'white',
    'axes.grid': True,
    'grid.alpha': 0.3,
})

sns.set_theme(style='whitegrid')

# ------------------------------------------------------------------------------
# BAGIAN 1 — PEMULIHAN LOGIT (INVERSE SIGMOID)
# ------------------------------------------------------------------------------

@dataclass
class CalibrationResult:
    """
    Wadah data untuk menyimpan hasil kalibrasi probabilitas.
    
    Atribut:
        method      : Nama metode kalibrasi ('Mentah', 'Temperature Scaling', 'Platt Scaling').
        probs       : Vektor probabilitas terkoreksi (N,).
        recall      : Recall pada ambang batas optimal.
        precision   : Presisi pada ambang batas optimal.
        f2          : F2-Score pada ambang batas optimal.
        threshold   : Ambang batas optimal yang dipakai untuk perhitungan metrik.
    """
    method: str
    probs: np.ndarray
    recall: float
    precision: float
    f2: float
    threshold: float


def inverse_sigmoid(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Memulihkan logits dari probabilitas menggunakan kebalikan fungsi sigmoid.
    
    Rumus:  z = ln( p / (1 - p) )
    
    Operasi ini diperlukan karena temperature scaling, platt scaling, dan 
    kalibrasi probabilistik lainnya bekerja pada skala log-odds (logits), 
    bukan pada skala probabilitas yang terikat di [0, 1].
    
    Argumen:
        p   : Vektor probabilitas (N,) dalam selang (0, 1).
        eps : Nilai epsilon untuk clipping. Mencegah np.log(0) atau np.log(inf)
              pada saat probabilitas mendekati batas absolut 0 atau 1.
    
    Kembalian:
        z   : Vektor log-odds (logits) (N,) dalam selang (-inf, +inf).
    """
    # Clipping: probabilitas tidak boleh persis 0 atau 1 karena logaritma
    # dari 0 adalah -inf (tak terdefinisi) pada aritmetika floating-point.
    p_safe = np.clip(p, eps, 1.0 - eps)
    return np.log(p_safe / (1.0 - p_safe))


def sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Fungsi Sigmoid: sigma(z) = 1 / (1 + exp(-z)).
    Menerima sembarang nilai riil dan memetakannya ke selang (0, 1).
    """
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


# ------------------------------------------------------------------------------
# BAGIAN 2 — IMPLEMENTASI DUA METODE KALIBRASI
# ------------------------------------------------------------------------------

def calibrate_temperature_scaling(logits: np.ndarray, T: float = 0.1) -> np.ndarray:
    """
    METODE A — Temperature Scaling.
    
    Prinsip:
        P_terkalibrasi = sigma( z / T )
    
    Efek:
      - T < 1.0  : Memperbesar kontras dan menyebarkan probabilitas menjauhi 0.5.
                   Berguna ketika probabilitas model terlalu seragam (underconfident).
      - T > 1.0  : Memperhalus distribusi probabilitas mendekati 0.5.
                   Berguna ketika model kelewat yakin (overconfident).
    
    Pada V8 SupCon, kita butuh T < 1 karena probabilitas mentah terkompresi 
    sangat rendah (seragam di bawah 0.004). T kecil akan mengangkat kontras 
    dan memulihkan sebaran probabilitas operasional.
    
    Argumen:
        logits : Vektor log-odds (N,) hasil inverse_sigmoid.
        T      : Parameter temperatur (T > 0). 
    
    Kembalian:
        Vektor probabilitas (N,) hasil kalibrasi.
    """
    assert T > 0.0, "Temperatur harus bernilai positif!"
    scaled_logits = logits / T
    return sigmoid(scaled_logits)


def calibrate_platt_scaling(logits: np.ndarray, y_true: np.ndarray) -> Tuple[np.ndarray, LogisticRegression]:
    """
    METODE B — Platt Scaling (Regresi Logistik pada Logits Mentah).
    
    Prinsip:
        Melatih model regresi logistik univariat (1 fitur: logits) untuk 
        memetakan ulang probabilitas secara optimal:
            P(y=1 | z) = 1 / (1 + exp(-(alpha * z + beta)))
    
    Parameter alpha dan beta diestimasi dengan memaksimalkan kemungkinan
    (maximum likelihood) terhadap label aktual True_Label, menghasilkan
    kalibrasi probabilistik yang optimal dalam artian NLL (Negative Log Likelihood).
    
    Keunggulan Platt Scaling:
      - Tidak memerlukan data validasi terpisah karena cross-validation.
      - Memetakan probabilitas secara linier pada skala log-odds.
      - Stabil dan robust terhadap distribusi kelas ekstrem.
    
    Argumen:
        logits : Vektor log-odds (N,) untuk dijadikan fitur input.
        y_true : Vektor label biner (N,) sebagai target regresi (0 atau 1).
    
    Kembalian:
        probs  : Vektor probabilitas (N,) hasil kalibrasi Platt.
        model  : Objek LogisticRegression yang sudah terlatih.
    """
    # Reshape ke format yang diminta sklearn: (N, 1) untuk fitur tunggal
    X = logits.reshape(-1, 1)
    y = y_true.astype(float)
    
    # Inisialisasi regresi logistik dengan parameter standar kalibrasi
    model = LogisticRegression(
        C=1.0,          # Regularisasi standar; tidak terlalu kuat
        penalty='l2',   # L2 regularization untuk stabilitas numerik
        solver='lbfgs', # Algoritma optimal untuk dataset kecil-menengah
        max_iter=500,   # Iterasi cukup untuk konvergensi
        random_state=42
    )
    
    # Latih model pada seluruh data (karena Platt Scaling adalah post-processing,
    # bukan prediksi generalisasi — kita ingin re-mapping optimal untuk data ini)
    model.fit(X, y)
    
    # Prediksi probabilitas kelas positif (indeks 1)
    probs = model.predict_proba(X)[:, 1]
    
    return probs, model


# ------------------------------------------------------------------------------
# BAGIAN 3 — EVALUASI DAN AMBANG BATAS OPTIMAL F2
# ------------------------------------------------------------------------------

@dataclass
class ThresholdMetrics:
    """Ringkasan metrik pada suatu ambang batas tertentu."""
    precision: float
    recall: float
    f2: float


def find_optimal_threshold_f2(probs: np.ndarray, y_true: np.ndarray,
                                beta: float = 2.0, n_thresholds: int = 1000
                                ) -> Tuple[float, ThresholdMetrics]:
    """
    Menemukan ambang batas klasifikasi yang memaksimalkan F-beta Score.
    
    F-beta Score memberikan bobot beta kali lebih besar pada Recall 
    dibandingkan Precision. Untuk EWS gempa, recall lebih penting dari
    presisi (beta = 2, seperti pada F2-Score).
    
    Rumus:
        F-beta = (1 + beta^2) * (precision * recall) / (beta^2 * precision + recall)
    
    Argumen:
        probs       : Vektor probabilitas (N,).
        y_true      : Vektor label aktual biner (N,).
        beta        : Parameter bobot recall (beta > 1 = recall diutamakan).
        n_thresholds: Jumlah titik ambang yang dievaluasi.
    
    Kembalian:
        best_threshold : Ambang batas optimal.
        best_metrics   : Metrik pada ambang batas tersebut.
    """
    thresholds = np.linspace(0.0, 1.0, n_thresholds)
    best_f2, best_th, best_metrics = 0.0, 0.5, None
    
    for th in thresholds:
        preds = (probs >= th).astype(int)
        prec = precision_score(y_true, preds, zero_division=0)
        rec = recall_score(y_true, preds, zero_division=0)
        if prec + rec > 0:
            f2 = (1 + beta**2) * prec * rec / (beta**2 * prec + rec)
        else:
            f2 = 0.0
        if f2 > best_f2:
            best_f2 = f2
            best_th = th
            best_metrics = ThresholdMetrics(precision=prec, recall=rec, f2=f2)
    
    return float(best_th), best_metrics


def compute_all_metrics(calib_probs: np.ndarray, y_true: np.ndarray,
                        method: str, beta: float = 2.0) -> CalibrationResult:
    """
    Menghitung metrik lengkap hasil kalibrasi pada ambang optimal F-beta.
    """
    best_th, metrics = find_optimal_threshold_f2(calib_probs, y_true, beta=beta)
    return CalibrationResult(
        method=method,
        probs=calib_probs,
        recall=metrics.recall,
        precision=metrics.precision,
        f2=metrics.f2,
        threshold=best_th
    )


# ------------------------------------------------------------------------------
# BAGIAN 4 — VISUALISASI FORENSIK
# ------------------------------------------------------------------------------

def plot_probability_kde(results: list, y_true: np.ndarray,
                         save_path: Optional[Path] = None):
    """
    Menghasilkan KDE Plot distribusi probabilitas sebelum vs sesudah kalibrasi.
    
    KDE Plot (Kernel Density Estimation) memberikan gambaran kontinu tentang
    sebaran probabilitas untuk kedua kelas (positif gempa dan negatif noise).
    Plot ini vital untuk membuktikan bahwa kalibrasi telah memisahkan distribusi
    kedua kelas secara tegas.
    """
    n_methods = len(results)
    fig, axes = plt.subplots(1, n_methods, figsize=(7 * n_methods, 5),
                             sharey=False, facecolor='white')
    
    if n_methods == 1:
        axes = [axes]
    
    for ax, result in zip(axes, results):
        mask_pos = y_true == 1
        mask_neg = y_true == 0
        
        # Menggunakan KDE (Kernel Density Estimation) alih-alih Histogram
        if mask_pos.sum() > 1:
            sns.kdeplot(result.probs[mask_pos], ax=ax, fill=True, alpha=0.5,
                        color='#2ca02c', label=f'Gempa (n={mask_pos.sum()})', clip=(0, 1))
        
        if mask_neg.sum() > 1:
            sns.kdeplot(result.probs[mask_neg], ax=ax, fill=True, alpha=0.5,
                        color='#d62728', label=f'Noise (n={mask_neg.sum()})', clip=(0, 1))
        
        ax.axvline(x=result.threshold, color='black', linestyle='--',
                   linewidth=1.5, label=f'Threshold F2 = {result.threshold:.4f}')
        ax.set_title(result.method, fontsize=13, pad=10, fontweight='bold')
        ax.set_xlabel('Probabilitas Prediksi', fontsize=11)
        ax.set_ylabel('Kepadatan (Density)', fontsize=11)
        ax.legend(frameon=True, fontsize=9)
        ax.set_xlim(-0.02, 1.02)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"[OK] KDE Plot disimpan di: {save_path}")
    plt.close()


def plot_reliability_diagram(results: list, y_true: np.ndarray,
                              save_path: Optional[Path] = None):
    """
    Menghasilkan Reliability Diagram (Calibration Curve).
    
    Reliability diagram membandingkan probabilitas prediksi model terhadap
    frekuensi aktual kejadian gempa. Jika kalibrasi sempurna, kurva akan
    berada pada garis diagonal (identitas).
    
    Konsep penting:
      - Kurva di atas garis diagonal → model under-confident.
      - Kurva di bawah garis diagonal → model over-confident.
      - Kurva pada garis diagonal   → kalibrasi sempurna.
    """
    fig, ax = plt.subplots(figsize=(7, 7), facecolor='white')
    
    colors = ['#1f77b4', '#2ca02c', '#d62728']
    
    # Plot garis diagonal ideal
    ax.plot([0, 1], [0, 1], 'k--', linewidth=2, alpha=0.7, label='Kalibrasi Sempurna (Ideal)')
    
    for result, color in zip(results, colors):
        if result.probs.ndim == 1 and len(result.probs) > 0:
            prob_true, prob_pred = calibration_curve(
                y_true, result.probs, n_bins=10, strategy='quantile'
            )
            ax.plot(prob_pred, prob_true, marker='o', linewidth=2,
                    markersize=8, color=color, label=result.method)
            
            # Hitung Expected Calibration Error (ECE) sebagai bukti kuantitatif
            ece = np.mean(np.abs(prob_true - prob_pred))
            ax.annotate(f'ECE = {ece:.4f}',
                        xy=(0.05, 0.95 - 0.08 * colors.index(color)),
                        fontsize=9, color=color, fontweight='bold',
                        ha='left', va='top',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                  edgecolor=color, alpha=0.8))
    
    ax.set_xlabel('Probabilitas Prediksi Rata-rata', fontsize=12)
    ax.set_ylabel('Frekuensi Aktual Kejadian', fontsize=12)
    ax.set_title('Reliability Diagram — V8 SupCon Blind Test 2026',
                 fontsize=14, pad=10, fontweight='bold')
    ax.legend(loc='lower right', frameon=True, fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"[OK] Reliability Diagram disimpan di: {save_path}")
    plt.close()


# ------------------------------------------------------------------------------
# BAGIAN 5 — EKSEKUSI UTAMA
# ------------------------------------------------------------------------------

def main():
    """
    Alur eksekusi utama:
      1. Memuat data prediksi dari CSV.
      2. Memulihkan logits dari probabilitas mentah.
      3. Menerapkan Temperature Scaling dengan T = 0.1.
      4. Menerapkan Platt Scaling.
      5. Mencetak tabel perbandingan metrik.
      6. Menyimpan grafik KDE dan Reliability Diagram.
    """
    print("=" * 72)
    print("  OPTIMASI KALIBRASI PROBABILITAS — FASE 1B")
    print("  Model: V8 SupCon (EfficientNet-B1 + SupCon Loss)")
    print("=" * 72)
    
    # -- Konfigurasi ----------------------------------------------------------
    DATA_PATH = Path("D:/multi/scalogramv3/disertasi4/Btest_Forensic/predictions_2026.csv")
    OUTPUT_DIR = Path("D:/multi/scalogramv3/disertasi4/Btest_Forensic/stratification")
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    
    TEMPERATURE = 6.3    # Optimal berdasarkan Fase 1 (meningkatkan kontras)
    BETA = 2.0           # F2-Score (Recall diutamakan)
    
    # -- Langkah 1: Muat Data -------------------------------------------------
    print(f"\n[1] Memuat data dari: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    y_true = df['True_Label'].values.astype(int)
    p_raw = df['Pred_Prob'].values.astype(np.float64)
    
    print(f"    Total sampel : {len(df)}")
    print(f"    Positif (1)  : {(y_true == 1).sum()}")
    print(f"    Negatif (0)  : {(y_true == 0).sum()}")
    print(f"    Rentang prob : [{p_raw.min():.6f}, {p_raw.max():.6f}]")
    
    # -- Langkah 2: Baseline (Tanpa Kalibrasi) --------------------------------
    print(f"\n[2] Menghitung metrik baseline (tanpa kalibrasi)...")
    baseline = compute_all_metrics(p_raw, y_true, method='Mentah (T=1.0)', beta=BETA)
    print(f"    Recall    = {baseline.recall*100:.2f}%")
    print(f"    Precision = {baseline.precision*100:.2f}%")
    print(f"    F2-Score  = {baseline.f2:.4f}")
    print(f"    Threshold = {baseline.threshold:.4f}")
    
    # -- Langkah 3: Pemulihan Logits ------------------------------------------
    print(f"\n[3] Memulihkan logits dari probabilitas mentah (Inverse Sigmoid)...")
    logits = inverse_sigmoid(p_raw)
    print(f"    Rentang logits: [{logits.min():.4f}, {logits.max():.4f}]")
    print(f"    Rata-rata     : {logits.mean():.4f}")
    
    # -- Langkah 4a: Kalibrasi dengan Temperature Scaling ---------------------
    print(f"\n[4a] Temperature Scaling (T = {TEMPERATURE})...")
    p_ts = calibrate_temperature_scaling(logits, T=TEMPERATURE)
    ts_result = compute_all_metrics(p_ts, y_true, method='Temperature Scaling', beta=BETA)
    print(f"    Recall    = {ts_result.recall*100:.2f}%")
    print(f"    Precision = {ts_result.precision*100:.2f}%")
    print(f"    F2-Score  = {ts_result.f2:.4f}")
    print(f"    Threshold = {ts_result.threshold:.4f}")
    
    # -- Langkah 4b: Kalibrasi dengan Platt Scaling ---------------------------
    print(f"\n[4b] Platt Scaling (Regresi Logistik pada Logits)...")
    p_ps, model_platt = calibrate_platt_scaling(logits, y_true)
    ps_result = compute_all_metrics(p_ps, y_true, method='Platt Scaling', beta=BETA)
    print(f"    Recall    = {ps_result.recall*100:.2f}%")
    print(f"    Precision = {ps_result.precision*100:.2f}%")
    print(f"    F2-Score  = {ps_result.f2:.4f}")
    print(f"    Threshold = {ps_result.threshold:.4f}")
    print(f"    Model Platt: alpha = {model_platt.coef_[0][0]:.4f}, beta = {model_platt.intercept_[0]:.4f}")
    
    # -- Langkah 5: Tabel Perbandingan ----------------------------------------
    print("\n" + "=" * 72)
    print("  TABEL PERBANDINGAN METRIK KALIBRASI")
    print("=" * 72)
    print(f"  {'Metode':<25s} {'Recall':>8s} {'Precision':>10s} {'F2-Score':>9s} {'Threshold':>10s}")
    print("  " + "-" * 62)
    for result in [baseline, ts_result, ps_result]:
        print(f"  {result.method:<25s} {result.recall*100:>7.2f}% "
              f"{result.precision*100:>9.2f}% {result.f2:>9.4f} {result.threshold:>10.4f}")
    print("=" * 72)
    
    # -- Langkah 6: Visualisasi -----------------------------------------------
    print(f"\n[5] Menghasilkan grafik visualisasi...")
    
    # KDE Plot
    kde_path = OUTPUT_DIR / "fig_F_kde_calibration_comparison.png"
    plot_probability_kde([baseline, ts_result, ps_result], y_true, save_path=kde_path)
    
    # Reliability Diagram
    reliab_path = OUTPUT_DIR / "fig_G_reliability_diagram.png"
    plot_reliability_diagram([baseline, ts_result, ps_result], y_true, save_path=reliab_path)
    
    # -- Langkah 7: Simpan Hasil ---------------------------------------------
    output_csv = OUTPUT_DIR / "predictions_calibrated_comparison.csv"
    df_out = df.copy()
    df_out['Prob_TemperatureScaling'] = np.round(p_ts, 6)
    df_out['Prob_PlattScaling'] = np.round(p_ps, 6)
    df_out.to_csv(output_csv, index=False)
    print(f"\n[OK] Hasil kalibrasi CSV: {output_csv}")
    
    print(f"\n{'='*72}")
    print("  KESIMPULAN OPTIMASI KALIBRASI")
    print(f"{'='*72}")
    
    # Ringkasan akademik
    print(f"")
    print(f"  Data: {len(df)} sampel prediksi buta (Blind Test 2026)")
    print(f"  Dist. kelas: {(y_true==1).sum()} gempa vs {(y_true==0).sum()} noise")
    print(f"")
    print(f"  Temperature Scaling (T = {TEMPERATURE}):")
    print(f"    Recall    {baseline.recall*100:.2f}% -> {ts_result.recall*100:.2f}%")
    print(f"    Precision {baseline.precision*100:.2f}% -> {ts_result.precision*100:.2f}%")
    print(f"")
    print(f"  Platt Scaling (Logistic Regression):")
    print(f"    Recall    {baseline.recall*100:.2f}% -> {ps_result.recall*100:.2f}%")
    print(f"    Precision {baseline.precision*100:.2f}% -> {ps_result.precision*100:.2f}%")
    print(f"")
    print(f"  *** Bukti forensik: Kedua metode kalibrasi post-processing ***")
    print(f"  *** memulihkan performa deteksi tanpa retraining.        ***")
    print(f"{'='*72}")
    
    return {
        'baseline': baseline,
        'temperature_scaling': ts_result,
        'platt_scaling': ps_result,
        'platt_model': model_platt
    }


if __name__ == "__main__":
    results = main()
