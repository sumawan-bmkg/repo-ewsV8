
# Ringkasan Eviden Kuantitatif: Evaluasi Fase Perancangan & Dinamika Pelatihan

| Kategori | Metrik | Nilai | Keterangan |
|----------|--------|-------|------------|
| **Prapemrosesan (S8)** | Peningkatan SNR | **6.72×** | Signal-to-Noise Ratio |
| | Peningkatan Stabilitas | **11.58×** | Stabilitas sinyal |
| | Reduksi Noise | **11.58×** | Pengurangan noise |
| | Jumlah NaN | **0** | Data valid sepenuhnya |
| **Dataset (S3,S6,S13)** | Sampel Negatif | **7,312** | Pink noise (label_mag=0) |
| | Sampel Positif | **2,144** | Sinyal tektonik |
| | Rasio Ketidakseimbangan | **1:3.41** | Negative:Positive |
| | Rasio Imbalance Loss | **212.15** | Standard CE loss |
| **Konvergensi (S2,S12)** | Epoch Saat Ini | **17** | Dari target 50 epoch |
| | MAE Azimuth | **99.19°** | Belum konvergen penuh |
| | ECE (Sebelum) | **0.344** | Expected Calibration Error |
| | ECE (Sesudah) | **0.255** | **-25.9% perbaikan** |
| | Brier Score (Sebelum) | **0.330** | Probability calibration |
| | Brier Score (Sesudah) | **0.237** | **-28.2% perbaikan** |
