# Dokumentasi Metadata Repositori Cloud

## ScalogramV3 — Dataset Prekursor Gempa Berbasis CWT

| Item | Detail |
|:-----|:-------|
| **URL Repositori** | [https://its.id/m/datasetprekursor](https://its.id/m/datasetprekursor) |
| **Afiliasi** | Institut Teknologi Sepuluh Nopember (ITS) — BMKG |
| **Proyek** | ScalogramV3 — Deteksi Prekursor Gempa Berbasis Deep Learning |
| **Standarisasi** | Continuous Wavelet Transform (CWT) — Morlet Wavelet |
| **Model Final** | V3 (rilis 20 April 2026) |
| **Status Akses** | Terbuka untuk publik (open access) |
| **Format Data** | HDF5 (.h5) — tensor 3-channel |
| **Total Ukuran** | ~2.1 GB (terkompresi) |

---

## Riwayat Versi Proyek

| Versi | Status | Basis Fitur | Tanggal | Keterangan |
|:------|:------:|:------------|:--------|:-----------|
| **V1** | N/A | — | — | Tidak tersedia. Standardisasi CWT belum ada |
| **V2** | N/A | — | — | Tidak tersedia. Standardisasi CWT belum ada |
| **V3** | **AKTIF** | CWT Morlet (79×168×3) | **20 April 2026** | Baseline utama proyek |

> **Catatan Kritis:** Versi V1 dan V2 berstatus **N/A (Not Available)** karena standardisasi
> ekstraksi fitur berbasis Continuous Wavelet Transform (CWT) baru diimplementasikan
> secara penuh pada Model V3. Seluruh hasil yang diseminarkan merujuk pada ScalogramV3
> sebagai baseline tunggal proyek.

---

## Struktur Berkas HDF5

```
scalogram_v3_cosmic_final.h5
├── /train
│   ├── prekursor/       # Kelas 1 — 1-25 hari pra-gempa M>=5.0
│   └── normal/          # Kelas 0 — periode tenang >=30 hari
├── /validation
│   ├── prekursor/
│   └── normal/
└── /test
    ├── prekursor/
    └── normal/
```

### Format Tensor per Sampel

| Atribut | Spesifikasi |
|:--------|:------------|
| Format | HDF5 Dataset (float32) |
| Dimensi | (79, 168, 3) |
| Baris (79) | Skala frekuensi CWT (log-spaced) — setara 168–6 mHz |
| Kolom (168) | Rentang waktu jendela STFT referensi 1024 dtk (downsampled) |
| Kanal | 3 kanal: [H, D, Z] — komponen medan magnet |
| Nilai | Koefisien CWT ternormalisasi (|CWT|^2) |

### Komponen Medan Magnet

| Kanal | Komponen | Deskripsi Fisis |
|:-----:|:---------|:----------------|
| 0 | H | Horizontal (magnetic north-south) — sinyal utama untuk deteksi anomali ULF |
| 1 | D | Deklinasi (magnetic east-west) — referensi baseline |
| 2 | Z | Vertikal — digunakan untuk rasio polarisasi Z/H sebagai validasi fisis |

---

## Spesifikasi Scalogram (CWT Morlet)

| Parameter | Nilai | Keterangan |
|:----------|:-----:|:-----------|
| Wavelet | Morlet (cmor1-1.5) | cmorB-C: B=1, C=1.5 |
| Rentang skala | 1–128 | Log-spaced, 79 skala |
| Rentang frekuensi | ~168–6 mHz | Mencakup Pc3 (22-100 mHz) & Pc4 (7-22 mHz) |
| Window referensi | 1024 dtk | Overlap 50% |
| Normalisasi | Min-Max per sampel | [0, 1] float32 |

---

## Distribusi Sampel per Partisi

| Partition | Prekursor | Normal | Total | % Prekursor |
|:----------|----------:|-------:|------:|:------------|
| **Train** | 3,200 | 25,800 | 29,000 | 11.0% |
| **Validation** | 650 | 3,350 | 4,000 | 16.3% |
| **Test** | 650 | 3,350 | 4,000 | 16.3% |
| **Total** | **4,500** | **32,500** | **37,000** | **12.2%** |

---

## Atribut Pelabelan

| Atribut | Spesifikasi |
|:--------|:------------|
| Katalog gempa | BMKG — M ≥ 5.0, kedalaman ≤ 100 km |
| Jendela prekursor | 1–25 hari sebelum event |
| Buffer antar event | ±3 hari |
| Jarak stasiun-episenter | < 500 km |

---

*Dokumen metadata ScalogramV3 — Standar FAIR Data Principles*
*Referensi: IEEE Data Engineering / AGU JGR: Solid Earth*
