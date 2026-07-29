
# Ringkasan Eviden Kuantitatif: Validasi Pasca-Pemodelan & Pengujian Operasional EWS

## 1. Peningkatan Kinerja Fundamental (V3 vs V8)

| Metrik | V3 (Baseline) | V8 (SupCon) | Perubahan | Keterangan |
|--------|:------------:|:----------:|:---------:|------------|
| False Positive Rate (FPR) | 1.000 | **0.236** | **-76.4%** | 4.2x lebih baik |
| EWS Score (F2 - FPR) | -0.167 | **+0.708** | **+0.875** | Layak operasional |

## 2. Kepatuhan Fisis & Geografis

| Metrik | Nilai | Target | Status | Sumber |
|--------|:----:|:-----:|:------:|--------|
| Kp-Gate Correlation | **0.917** | -- | Signifikan | S1 |
| Saturation Rate | **0.000%** | <5% | Sempurna | S1 |
| COI Masking | **38.0%** | <50% | Tercapai | S4, S5 |
| Dobrovolsky Strain (Dekat) | **0.676** | > Far | Fisik | S1 |
| Dobrovolsky Strain (Jauh) | **0.064** | < Near | Fisik | S1 |
| Strain Ratio (Dekat/Jauh) | **10.6x** | -- | Kuat | S1 |

## 3. Analisis Kalibrasi Threshold

| Metrik | Static th=0.30 | Optimal th=0.25 | Proyeksi |
|--------|:-------------:|:---------------:|:--------:|
| Recall | 0.000 | **>0.60** | Lompatan >60% |
| Threshold | Old strategy (V4) | **F2-optimal dynamic** | S7, S9, S11 |

## 4. Efisiensi Operasional

| Metrik | Nilai | Target | Status | Sumber |
|--------|:----:|:-----:|:------:|--------|
| Latensi per sampel | **0.270 s** | -- | Cepat | S10 |
| Latensi batch (8) | **1.865 s** | -- | Cepat | S10 |
| Throughput | **13,319 pred/jam** | 100 pred/jam | **133x target** | S10 |
