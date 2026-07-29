# DISERTASI EWS GEMPA BUMI — V8 SupCon (EfficientNet-B1)

## Daftar Lampiran Bukti Sidang Doktoral

### 01_Kinerja_Fundamental/

| File | Deskripsi |

|------|-----------|

| `kinerja_lengkap.png/pdf/svg` | Ringkasan 13 strategi deteksi (SNR, stabilitas, reduksi noise) |

| `fig_A_recall_per_magnitude.png` | Recall per magnitudo gempa stratifikasi |

| `fig_B_fn_distribution.png` | Distribusi False Negative per magnitudo |

| `fig_C_prob_boxplot.png` | Boxplot probabilitas per kelas magnitudo |

| `roc_pr_curves.png` | Kurva ROC dan Precision-Recall |

| `threshold_sweep.png/pdf` | Analisis ambang batas F2-optimal |

| `training_dual.png/pdf` | Dinamika pelatihan dual-task |

| `spatiotemporal_heatmap.png` | Sebaran spatiotemporal prediksi vs aktual |

| `station_detection_rate.png` | Tingkat deteksi per stasiun geomagnetik |

| `azimuth_kp_analysis.png/pdf` | Korelasi medan magnet (Kp) dengan deteksi gempa |

| `stratification_table.csv` | Tabel kuantitatif stratifikasi magnitudo |

| `dashboard.html` | Dashboard interaktif metrik EWS |


### 02_BlindTest_Forensik/

| File | Deskripsi |

|------|-----------|

| `predictions_calibrated_comparison.csv` | Dataset uji buta 2026 (2,904 sampel) |

| `blindtest_metrics_final.json` | Metrik agregat blind test |

| `blindtest_validity_report.md` | Laporan validitas buta (fisik & temporal) |

| `V8SupCon_2026_BlindTest_Verdict.md` | Verdict final blind test |

| `blindtest_timeline.png/pdf` | Timeline kronologis prediksi vs gempabumi |

| `pipeline_blind_test.py` | Skrip pipeline blind test utama |


### 03_Mitigasi_Fase1_PostProc/

| File | Deskripsi |

|------|-----------|

| `fase1_temperature_scaling.py` | Temperature Scaling (T=6.3 optimal) |

| `fase1b_calibration_comparison.py` | Perbandingan Temperature vs Platt Scaling |

| `fig_D_temperature_scaling_sweep.png` | Sweep parameter T vs F2-Score |

| `fig_F_kde_calibration_comparison.png` | KDE distribusi probabilitas terkalibrasi |

| `fig_G_reliability_diagram.png` | Reliability Diagram (kalibrasi probabilistik) |


### 04_Mitigasi_Fase2_Gradien/

| File | Deskripsi |

|------|-----------|

| `fase2_binary_focal_loss.py` | Implementasi Class-Weighted Focal Loss (alpha=3.41, gamma=2.0) |

| `fase2_run_resume_training.py` | Resume training Epoch 18-50 dengan Focal Loss |

| `training_dynamics_fase2.csv` | CSV dinamika pelatihan (loss, recall, mean prob per epoch) |

| `fig_gradient_convergence.png` | Plot konvergensi loss & pertumbuhan recall |


### 05_Mitigasi_Fase3_Decouple/

| File | Deskripsi |

|------|-----------|

| `fase3_decoupled_training.py` | Dekopling gradien: freeze backbone, unfreeze classifier head |


### 06_Naskah_Sidang/

| File | Deskripsi |

|------|-----------|

| `V8SupCon_2026_BlindTest_Verdict.md` | Ringkasan kesimpulan untuk sidang |

| `blindtest_validity_report.md` | Laporan validitas untuk naskah bab hasil |


---

*Manifest digenerate otomatis oleh `consolidate_evidence.py`*  

*Tanggal: 2026-07-29 15:23*
