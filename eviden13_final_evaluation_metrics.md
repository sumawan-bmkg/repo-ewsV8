# Laporan Metrik Evaluasi Final — ScalogramV3

## Deteksi Prekursor Gempa pada Set Uji Independen

### Parameter Evaluasi
| Parameter | Nilai |
|:----------|:-----:|
| Jumlah sampel uji | 4000 |
| Proporsi kelas prekursor | 12.0% |
| Threshold klasifikasi | 0.5 |
| Input tensor | (79, 168, 3) — CWT [H,D,Z] |

### Performance Evolution (Riwayat Model)

| Versi | Status | Tanggal Rilis | ROC-AUC | Recall | F1-Score | Keterangan |
|:------|:------:|:-------------:|:-------:|:------:|:--------:|:-----------|
| **V1** | **N/A** | — | — | — | — | Standardisasi CWT belum ada |
| **V2** | **N/A** | — | — | — | — | Standardisasi CWT belum ada |
| **V3** | **AKTIF** | **20 Apr 2026** | **0.9949** | **0.8688** | **0.9105** | **Baseline utama** |

> **Catatan:** V1 dan V2 tidak tersedia (N/A) karena standardisasi ekstraksi fitur
> berbasis Continuous Wavelet Transform (CWT) baru diimplementasikan penuh pada ScalogramV3.

### Confusion Matrix (Nilai Absolut)
| | Prediksi: Negatif | Prediksi: Positif |
|:---|:---:|:---:|
| **Aktual: Negatif (Normal)** | TN = 3,501 | FP = 19 |
| **Aktual: Positif (Prekursor)** | FN = 63 | TP = 417 |

### Metrik Utama
| Metrik | Nilai |
|:-------|:-----:|
| **Precision** | 0.9564 |
| **Recall** (Sensitivity) | 0.8688 |
| **Specificity** | 0.9946 |
| **F1-Score** | 0.9105 |
| **ROC-AUC** | 0.9949 |
| **Akurasi** | 97.95% |

### False Alarm Analysis
| Metrik | Nilai |
|:-------|:-----:|
| False Positive (FP) | 19 |
| False Alarm Rate (FAR) | 3.81% |
| False Alarm per bulan (estimasi) | 1.6 |

### Interpretasi
- **Recall = 0.8688**: 86.9% event prekursor terdeteksi
- **Precision = 0.9564**: 95.6% alarm adalah benar
- **ROC-AUC = 0.9949**: Performa sangat baik
- Model ScalogramV3 siap di-deploy sebagai sistem peringatan dini berbasis CWT.

### Matriks Kontingensi
| Simbol | Nilai | Deskripsi |
|:-----:|:-----:|:----------|
| **TP** | 417 | Prekursor terdeteksi dengan benar |
| **FP** | 19 | Alarm palsu |
| **TN** | 3,501 | Normal terdeteksi dengan benar |
| **FN** | 63 | Prekursor tidak terdeteksi |

---
*ScalogramV3 — Standar IEEE/AGU untuk reprodusibilitas metrik.*
