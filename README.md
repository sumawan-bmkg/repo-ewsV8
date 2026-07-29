<!-- markdownlint-disable MD041 MD033 MD036 -->
<div align="center">

<br>

# **Sistem Peringatan Dini Gempa Bumi Berbasis Deep Learning**
### EfficientNet-B1 + Supervised Contrastive Learning (V8 SupCon)

**Operational Geomagnetic Monitoring · BMKG**

<br>

[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![Python](https://img.shields.io/badge/Python-3.12+-3776AB?logo=python&logoColor=white)](https://python.org)
[![Status](https://img.shields.io/badge/Status-Dissertation%20Ready-success)]()

<br>

---

</div>

## Ringkasan

Repositori ini menyimpan seluruh artefak riset disertasi doktoral pada **Sistem Peringatan Dini (EWS) Gempa Bumi** menggunakan spektrum geomagnetik. Model inti adalah **EfficientNet-B1** yang dilatih dengan **Supervised Contrastive Learning (SupCon)** dan dioptimasi melalui 3 Fase mitigasi *Undercoupled Probability Emission*.

### Hasil Blind Test 2026

| Skenario | Recall | Precision | F2-Score | FPR | Mean Prob |
|:---------|:-----:|:---------:|:--------:|:---:|:---------:|
| **A: Baseline (Mentah)** | 12.89% | 97.10% | 0.1559 | 22.92% | 0.0033 |
| **B: Fase 1 (Temp. Scaling T=6.3)** | 90.06% | 98.39% | 0.9161 | 87.50% | 0.2848 |
| **C: Fase 2/3 (Focal Loss + Decoupled)** | **91.46%** | **99.96%** | **0.9304** | **2.08%** | **0.7869** |

> **Temuan Utama:** Optimasi 3 Fase berhasil menaikkan Recall dari **12.9% → 91.5%** dengan FPR hanya **2.08%**, membuktikan bahwa mitigasi *Undercoupled Probability Emission* efektif secara empiris.

---

## Struktur Repositori

```
disertasi4/
│
├── Btest/                          # Dataset uji Blind Test 2026
│   └── v3_v8_conv_fpr_best_weights.pth   # Bobot model final (36 MB)
│
├── Btest_Evidence/                 # Bukti evaluasi & forensik
│   ├── training_dynamics_fase2.csv # Dinamika konvergensi Focal Loss
│   └── ... (laporan metrik, grafik)
│
├── Btest_Forensic/                 # Investigasi forensik probabilitas
│   ├── predictions_2026.csv        # Prediksi baseline mentah
│   └── stratification/            # Analisis stratifikasi dan kalibrasi
│       └── predictions_calibrated_comparison.csv
│
├── DISERTASI_EWS_FINAL_EVIDENCE/   # Eviden final (13 strategi)
│   ├── 01_Kinerja_Fundamental/     # Laporan 13 strategi + dashboard
│   ├── 02_Bukti_Kalibrasi/         # Temperature Scaling, Platt Scaling
│   ├── ... (7 folder strategi)
│   └── dashboard.html              # Dashboard interaktif
│
├── DISERTASI_BLINDTEST_OPTIMIZED_EVIDENCE/   # Bukti optimasi 3 Fase
│   ├── data/                       # CSV prediksi 3 skenario
│   ├── plots/                      # Grafik komparasi (600 DPI)
│   │   ├── fig_0_graphical_abstract.png
│   │   ├── fig_1_recall_precision_comparison.png
│   │   ├── fig_2_roc_pr_curve_optimized.png
│   │   └── fig_3_reliability_diagram_after.png
│   └── logs/                       # Metrik JSON
│
├── EWS_Deployment_V8/              # Paket deployment produksi
│   ├── inference_ews.py            # Skrip inferensi siap-pakai
│   └── v3_v8_conv_fpr_best_weights.pth
│
├── optimasi/                       # Skrip optimasi 3 Fase
│   ├── build_deployment.py         # Builder deployment
│   ├── fase1_temperature_scaling.py
│   ├── fase2_focal_loss.py
│   ├── fase3_decoupled_training.py
│   ├── run_blindtest_optimized.py  # Runner komparasi 3 skenario
│   └── graphical_abstract_3fase.py # Generator graphical abstract
│
├── *.py                            # Skrip evaluasi & generator bukti
├── *.docx                          # Naskah disertasi bab per bab
├── .gitignore                      # Konfigurasi ignore file
├── backup_to_github.ps1            # Skrip backup ke GitHub
└── README.md                       # File ini
```

---

## Cara Reproduksi / Validasi

### 1. Prasyarat

- Python 3.12+
- PyTorch 2.x (CUDA optional — CPU sudah cukup untuk inferensi)
- Git

```bash
# Clone repositori
git clone https://github.com/<username>/repo-ews.git
cd repo-ews

# (Opsional) Buat virtual environment
python -m venv venv
source venv/bin/activate   # Linux/Mac
.\venv\Scripts\Activate    # Windows

# Install dependensi inti
pip install torch numpy pandas matplotlib scikit-learn
```

### 2. Menjalankan Evaluasi Blind Test

```bash
cd DISERTASI_BLINDTEST_OPTIMIZED_EVIDENCE
python ../optimasi/run_blindtest_optimized.py
```

Skrip akan:
1. Membaca dataset prediksi dari `Btest_Forensic/`
2. Menghitung metrik 3 skenario (Baseline, Fase 1, Fase 2/3)
3. Menyimpan CSV, JSON, dan grafik ke folder `plots/` dan `data/`

### 3. Menjalankan Inferensi Mandiri

```bash
cd EWS_Deployment_V8
python inference_ews.py --help
```

### 4. Membuat Graphical Abstract

```bash
python optimasi/graphical_abstract_3fase.py
```

Output: `DISERTASI_BLINDTEST_OPTIMIZED_EVIDENCE/plots/fig_0_graphical_abstract.png`

---

## Tentang Model

### Arsitektur

- **Backbone**: EfficientNet-B1 (pretrained ImageNet)
- **Head**: SupCon projection head (128-d embedding → 2 kelas)
- **Parameter**: ~36 MB (bobot akhir)

### Alur Optimasi 3 Fase

```
                         ┌─────────────────────┐
                         │  Baseline Model V8   │
                         │  Recall: 12.89%      │
                         └──────────┬──────────┘
                                    │
                         ┌──────────▼──────────┐
                         │  Fase 1              │
                         │  Temperature Scaling │
                         │  T = 6.3             │
                         │  Recall: 90.06%      │
                         └──────────┬──────────┘
                                    │
                         ┌──────────▼──────────┐
                         │  Fase 2              │
                         │  Focal Loss           │
                         │  γ=2.0, α=3.41       │
                         └──────────┬──────────┘
                                    │
                         ┌──────────▼──────────┐
                         │  Fase 3              │
                         │  Decoupled Training  │
                         │  Recall: 91.46%      │
                         │  Precision: 99.96%   │
                         │  FPR: 2.08%          │
                         └─────────────────────┘
```

---

## Git LFS (Large File Storage)

File bobot model `.pth` dalam repositori ini berukuran **~36–40 MB**, masih di bawah batas GitHub (100 MB) sehingga aman tanpa Git LFS.

**Jika di masa depan ada file > 100 MB:**

```bash
# 1. Install Git LFS
git lfs install

# 2. Track file .pth
git lfs track "*.pth"

# 3. Commit file .gitattributes
git add .gitattributes
git commit -m "chore: activate Git LFS for model weights"

# 4. Lakukan push normal
```

> **Catatan:** Git LFS memiliki kuota penyimpanan gratis 1 GB dan bandwidth 1 GB/bulan. Untuk file > 100 MB, pertimbangkan alternatif seperti Google Drive atau Zenodo.

---

## Lisensi & Sitasi

**Hak Cipta** © 2026 — Program Doktor Teknik Fisika

Data geomagnetik milik **BMKG (Badan Meteorologi, Klimatologi, dan Geofisika)**.



---

<div align="center">

**— Akhir README —**

*Repositori ini disusun sebagai lampiran bukti penelitian.*

</div>
