"""
================================================================================
 AUTOMASI KONSOLIDASI EVIDEN DISERTASI EWS GEMPA BUMI
 Mengumpulkan, menyalin, dan mem-backup seluruh file bukti dari:
   - Kinerja Fundamental
   - Blind Test Forensik
   - Mitigasi Fase 1 (Post-Processing)
   - Mitigasi Fase 2 (Gradient Intervention)
   - Mitigasi Fase 3 (Decoupled Training)
   - Naskah Sidang
 
 Output:
   DISERTASI_EWS_FINAL_EVIDENCE/  +  BACKUP_EVIDENCE_SIDANG_V8.zip

 Penggunaan: python consolidate_evidence.py
================================================================================
"""

import os
import shutil
import zipfile
from pathlib import Path

# -- KONFIGURASI PATH SUMBER --------------------------------------------------
BASE_DIR = Path("D:/multi/scalogramv3/disertasi4")
BTEST_EVIDENCE  = BASE_DIR / "Btest_Evidence"
OPTIMASI_DIR    = BASE_DIR / "optimasi"
FORENSIC_DIR    = BASE_DIR / "Btest_Forensic"
STRAT_DIR       = FORENSIC_DIR / "stratification"
SUPCON_DIR      = BASE_DIR / "supcon"

# -- KONFIGURASI PATH TUJUAN --------------------------------------------------
OUT_DIR = BASE_DIR / "DISERTASI_EWS_FINAL_EVIDENCE"

# Buat struktur subfolder
dirs = {
    "01_Kinerja_Fundamental":     OUT_DIR / "01_Kinerja_Fundamental",
    "02_BlindTest_Forensik":      OUT_DIR / "02_BlindTest_Forensik",
    "03_Mitigasi_Fase1_PostProc": OUT_DIR / "03_Mitigasi_Fase1_PostProc",
    "04_Mitigasi_Fase2_Gradien":  OUT_DIR / "04_Mitigasi_Fase2_Gradien",
    "05_Mitigasi_Fase3_Decouple": OUT_DIR / "05_Mitigasi_Fase3_Decouple",
    "06_Naskah_Sidang":           OUT_DIR / "06_Naskah_Sidang",
}

def log_fail(desc: str, err=""):
    msg = f"  [SKIP] {desc}"
    if err:
        msg += f" — {err}"
    print(msg)

def log_ok(desc: str):
    print(f"  [OK] {desc}")

# -- FUNGSI COPY DENGAN ERROR HANDLING ----------------------------------------
def copy_file(src: Path, dst: Path, rename: str = None):
    """Copy file src ke dst, rename opsional. Handle file tidak ditemukan."""
    if not src.exists():
        log_fail(f"{src.name} -> {dst.parent.name}/", "File sumber tidak ditemukan")
        return False
    try:
        dest_path = dst if rename is None else dst.parent / rename
        shutil.copy2(str(src), str(dest_path))
        log_ok(f"{src.name} -> {dst.parent.name}/")
        return True
    except Exception as e:
        log_fail(f"{src.name} -> {dst.parent.name}/", str(e))
        return False

# -- 0. BUAT DIREKTORI --------------------------------------------------------
print("="*70)
print("  KONSOLIDASI EVIDEN DISERTASI EWS V8 SUPCON")
print("="*70)
print()

for dir_name, dir_path in dirs.items():
    dir_path.mkdir(parents=True, exist_ok=True)
    print(f"[DIR] {dir_path.relative_to(BASE_DIR)}/")

# -- 1. KINERJA FUNDAMENTAL ---------------------------------------------------
print(f"\n{'-'*70}\n  [01] Kinerja Fundamental\n{'-'*70}")
d = dirs["01_Kinerja_Fundamental"]

charts = [
    BTEST_EVIDENCE / "kinerja_lengkap.png",
    BTEST_EVIDENCE / "kinerja_lengkap.pdf",
    BTEST_EVIDENCE / "kinerja_lengkap.svg",
    BTEST_EVIDENCE / "fig_A_recall_per_magnitude.png",
    BTEST_EVIDENCE / "fig_B_fn_distribution.png",
    BTEST_EVIDENCE / "fig_C_prob_boxplot.png",
    BTEST_EVIDENCE / "roc_pr_curves.png",
    BTEST_EVIDENCE / "threshold_sweep.png",
    BTEST_EVIDENCE / "threshold_sweep.pdf",
    BTEST_EVIDENCE / "training_dual.png",
    BTEST_EVIDENCE / "training_dual.pdf",
    BTEST_EVIDENCE / "spatiotemporal_heatmap.png",
    BTEST_EVIDENCE / "station_detection_rate.png",
    BTEST_EVIDENCE / "azimuth_kp_analysis.png",
    BTEST_EVIDENCE / "azimuth_kp_analysis.pdf",
    BTEST_EVIDENCE / "dashboard.html",
]
for f in charts:
    copy_file(f, d)

csvs = [
    BTEST_EVIDENCE / "stratification_table.csv",
]
for f in csvs:
    copy_file(f, d)

# -- 2. BLIND TEST FORENSIK ---------------------------------------------------
print(f"\n{'-'*70}\n  [02] Blind Test Forensik\n{'-'*70}")
d = dirs["02_BlindTest_Forensik"]

bt_files = [
    BTEST_EVIDENCE / "predictions_calibrated_comparison.csv",
    BTEST_EVIDENCE / "blindtest_metrics_final.json",
    BTEST_EVIDENCE / "blindtest_metrics_merge2026.json",
    BTEST_EVIDENCE / "blindtest_validity_report.md",
    BTEST_EVIDENCE / "V8SupCon_2026_BlindTest_Verdict.md",
    BTEST_EVIDENCE / "blindtest_timeline.png",
    BTEST_EVIDENCE / "blindtest_timeline.pdf",
]
for f in bt_files:
    copy_file(f, d)

# Salin skrip pipeline blind test
pipeline_scripts = [
    BTEST_EVIDENCE / "pipeline_blind_test.py",
    BTEST_EVIDENCE / "rerun_blindtest_merge2026.py",
    BTEST_EVIDENCE / "re_run_full_final.py",
    BTEST_EVIDENCE / "investigate_blindtest.py",
    BTEST_EVIDENCE / "investigate_azimuth.py",
    BTEST_EVIDENCE / "check_period.py",
]
for f in pipeline_scripts:
    copy_file(f, d)

# -- 3. MITIGASI FASE 1: POST-PROCESSING --------------------------------------
print(f"\n{'-'*70}\n  [03] Mitigasi Fase 1: Post-Processing\n{'-'*70}")
d = dirs["03_Mitigasi_Fase1_PostProc"]

fase1_scripts = [
    OPTIMASI_DIR / "fase1_temperature_scaling.py",
    OPTIMASI_DIR / "fase1b_calibration_comparison.py",
]
for f in fase1_scripts:
    copy_file(f, d)

fase1_extra = [
    BTEST_EVIDENCE / "fig_D_temperature_scaling_sweep.png",
    BTEST_EVIDENCE / "fig_F_kde_calibration_comparison.png",
    BTEST_EVIDENCE / "fig_G_reliability_diagram.png",
]
for f in fase1_extra:
    copy_file(f, d)

# -- 4. MITIGASI FASE 2: GRADIEN INTERVENTION ---------------------------------
print(f"\n{'-'*70}\n  [04] Mitigasi Fase 2: Gradient Intervention\n{'-'*70}")
d = dirs["04_Mitigasi_Fase2_Gradien"]

fase2_scripts = [
    OPTIMASI_DIR / "fase2_binary_focal_loss.py",
    OPTIMASI_DIR / "fase2_run_resume_training.py",
]
for f in fase2_scripts:
    copy_file(f, d)

fase2_evidence = [
    BTEST_EVIDENCE / "training_dynamics_fase2.csv",
    BTEST_EVIDENCE / "fig_gradient_convergence.png",
]
for f in fase2_evidence:
    copy_file(f, d)

# -- 5. MITIGASI FASE 3: DECOUPLED TRAINING -----------------------------------
print(f"\n{'-'*70}\n  [05] Mitigasi Fase 3: Decoupled Training\n{'-'*70}")
d = dirs["05_Mitigasi_Fase3_Decouple"]

fase3_scripts = [
    OPTIMASI_DIR / "fase3_decoupled_training.py",
]
for f in fase3_scripts:
    copy_file(f, d)

# -- 6. NASKAH SIDANG ---------------------------------------------------------
print(f"\n{'-'*70}\n  [06] Naskah Sidang\n{'-'*70}")
d = dirs["06_Naskah_Sidang"]

naskah_files = [
    BTEST_EVIDENCE / "V8SupCon_2026_BlindTest_Verdict.md",
    BTEST_EVIDENCE / "blindtest_validity_report.md",
]
for f in naskah_files:
    copy_file(f, d)

# -- BUAT README MANIFEST ------------------------------------------------------
print(f"\n{'-'*70}\n  Membuat README Manifest\n{'-'*70}")

readme_lines = [
    "# DISERTASI EWS GEMPA BUMI — V8 SupCon (EfficientNet-B1)\n",
    "## Daftar Lampiran Bukti Sidang Doktoral\n",
    "### 01_Kinerja_Fundamental/\n",
    "| File | Deskripsi |\n",
    "|------|-----------|\n",
    "| `kinerja_lengkap.png/pdf/svg` | Ringkasan 13 strategi deteksi (SNR, stabilitas, reduksi noise) |\n",
    "| `fig_A_recall_per_magnitude.png` | Recall per magnitudo gempa stratifikasi |\n",
    "| `fig_B_fn_distribution.png` | Distribusi False Negative per magnitudo |\n",
    "| `fig_C_prob_boxplot.png` | Boxplot probabilitas per kelas magnitudo |\n",
    "| `roc_pr_curves.png` | Kurva ROC dan Precision-Recall |\n",
    "| `threshold_sweep.png/pdf` | Analisis ambang batas F2-optimal |\n",
    "| `training_dual.png/pdf` | Dinamika pelatihan dual-task |\n",
    "| `spatiotemporal_heatmap.png` | Sebaran spatiotemporal prediksi vs aktual |\n",
    "| `station_detection_rate.png` | Tingkat deteksi per stasiun geomagnetik |\n",
    "| `azimuth_kp_analysis.png/pdf` | Korelasi medan magnet (Kp) dengan deteksi gempa |\n",
    "| `stratification_table.csv` | Tabel kuantitatif stratifikasi magnitudo |\n",
    "| `dashboard.html` | Dashboard interaktif metrik EWS |\n",
    "\n### 02_BlindTest_Forensik/\n",
    "| File | Deskripsi |\n",
    "|------|-----------|\n",
    "| `predictions_calibrated_comparison.csv` | Dataset uji buta 2026 (2,904 sampel) |\n",
    "| `blindtest_metrics_final.json` | Metrik agregat blind test |\n",
    "| `blindtest_validity_report.md` | Laporan validitas buta (fisik & temporal) |\n",
    "| `V8SupCon_2026_BlindTest_Verdict.md` | Verdict final blind test |\n",
    "| `blindtest_timeline.png/pdf` | Timeline kronologis prediksi vs gempabumi |\n",
    "| `pipeline_blind_test.py` | Skrip pipeline blind test utama |\n",
    "\n### 03_Mitigasi_Fase1_PostProc/\n",
    "| File | Deskripsi |\n",
    "|------|-----------|\n",
    "| `fase1_temperature_scaling.py` | Temperature Scaling (T=6.3 optimal) |\n",
    "| `fase1b_calibration_comparison.py` | Perbandingan Temperature vs Platt Scaling |\n",
    "| `fig_D_temperature_scaling_sweep.png` | Sweep parameter T vs F2-Score |\n",
    "| `fig_F_kde_calibration_comparison.png` | KDE distribusi probabilitas terkalibrasi |\n",
    "| `fig_G_reliability_diagram.png` | Reliability Diagram (kalibrasi probabilistik) |\n",
    "\n### 04_Mitigasi_Fase2_Gradien/\n",
    "| File | Deskripsi |\n",
    "|------|-----------|\n",
    "| `fase2_binary_focal_loss.py` | Implementasi Class-Weighted Focal Loss (alpha=3.41, gamma=2.0) |\n",
    "| `fase2_run_resume_training.py` | Resume training Epoch 18-50 dengan Focal Loss |\n",
    "| `training_dynamics_fase2.csv` | CSV dinamika pelatihan (loss, recall, mean prob per epoch) |\n",
    "| `fig_gradient_convergence.png` | Plot konvergensi loss & pertumbuhan recall |\n",
    "\n### 05_Mitigasi_Fase3_Decouple/\n",
    "| File | Deskripsi |\n",
    "|------|-----------|\n",
    "| `fase3_decoupled_training.py` | Dekopling gradien: freeze backbone, unfreeze classifier head |\n",
    "\n### 06_Naskah_Sidang/\n",
    "| File | Deskripsi |\n",
    "|------|-----------|\n",
    "| `V8SupCon_2026_BlindTest_Verdict.md` | Ringkasan kesimpulan untuk sidang |\n",
    "| `blindtest_validity_report.md` | Laporan validitas untuk naskah bab hasil |\n",
    "\n---\n",
    "*Manifest digenerate otomatis oleh `consolidate_evidence.py`*  \n",
    f"*Tanggal: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M')}*",
]

readme_path = OUT_DIR / "README.md"
with open(str(readme_path), "w", encoding="utf-8") as f:
    f.writelines(line + "\n" for line in readme_lines)

if readme_path.exists():
    log_ok("README.md -> DISERTASI_EWS_FINAL_EVIDENCE/")

# -- BUAT ARSIP ZIP ------------------------------------------------------------
print(f"\n{'-'*70}\n  Membuat Arsip ZIP Backup\n{'-'*70}")

zip_path = BASE_DIR / "BACKUP_EVIDENCE_SIDANG_V8.zip"
if zip_path.exists():
    zip_path.unlink()  # hapus lama

try:
    with zipfile.ZipFile(str(zip_path), "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(str(OUT_DIR)):
            for fname in files:
                full_path = os.path.join(root, fname)
                rel_path = os.path.relpath(full_path, str(BASE_DIR))
                zf.write(full_path, rel_path)
    log_ok(f"BACKUP_EVIDENCE_SIDANG_V8.zip ({zip_path.stat().st_size / 1024 / 1024:.1f} MB)")
except Exception as e:
    log_fail("Membuat ZIP", str(e))

# -- KESIMPULAN -----------------------------------------------------------------
print(f"\n{'='*70}")
print(f"  KONSOLIDASI SELESAI")
print(f"{'='*70}")
print(f"  Target: {OUT_DIR.relative_to(BASE_DIR)}/")
print(f"  Backup: BACKUP_EVIDENCE_SIDANG_V8.zip")
print(f"{'='*70}\n")
