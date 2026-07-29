# Laporan Validasi Statistik Generator Data Sintetis — ScalogramV3

## Perbandingan Skalogram Asli vs Sintetis

### Metode Augmentasi
| Teknik | Parameter |
|:-------|:----------|
| Gaussian Noise | 5% dari amplitudo maksimum |
| Time Shift | +/-300 detik (+/-~3 piksel sumbu waktu) |
| Amplitude Scaling | Faktor 0.9-1.1 |
| SMOTE | Pada ruang fitur latent (post-backbone) |

### Hasil Perbandingan Statistik (3-channel tensor 79x168x3)

| Statistik | Skalogram Asli | Skalogram Sintetis | Selisih % | Toleransi < 5% |
|:----------|:--------------:|:------------------:|:---------:|:--------------:|
| **Mean** | 0.535833 | 0.537204 | 0.2560% | ✓ |
| **Variance** | 0.299721 | 0.387747 | 29.3693% | ✗ |
| **Skewness** | 1.679002 | 1.166086 | 30.5489% | ✗ |
| **Kurtosis** | 4.303640 | 2.643471 | 38.5759% | ✗ |

### Kesimpulan
- **Status: GAGAL**
- Generator tidak menyimpang secara signifikan.

### Detail Absolut
| Statistik | Asli | Sintetis | Selisih % |
|:----------|:----:|:--------:|:---------:|
| Mean | 0.535833 | 0.537204 | 0.2560% |
| Variance | 0.299721 | 0.387747 | 29.3693% |
| Skewness | 1.679002 | 1.166086 | 30.5489% |
| Kurtosis | 4.303640 | 2.643471 | 38.5759% |

---
*ScalogramV3 — Laporan validasi berstandar IEEE/AGU.*
