# Justifikasi Ilmiah Transisi Metodologi: Dari Spektrogram (STFT) ke Skalogram (CWT)

**— ScalogramV3: Standardisasi Continuous Wavelet Transform untuk Deteksi Prekursor Geomagnetik —**

---

> **Naskah Akademik ini disusun sebagai pembelaan ilmiah (*academic justification*)
> atas keputusan fundamental proyek ScalogramV3 untuk meninggalkan metode STFT
> dan mengadopsi CWT sebagai standar tunggal ekstraksi fitur.**
>
> *Disusun untuk: Sidang Disertasi Doktoral — Program Doktor Teknik Fisika*

---

## 1. Keterbatasan Fundamental STFT (Spektrogram)

### 1.1 Formulasi Matematis STFT

Transformasi Fourier Waktu-Pendek (STFT) didefinisikan sebagai:

$$
X(\tau, f) = \int_{-\infty}^{\infty} x(t) \, w(t - \tau) \, e^{-j 2\pi f t} \, dt
$$

dengan $w(t)$ adalah fungsi jendela (biasanya Hann/Hamming) yang kompak pada interval $[-T/2, T/2]$. Parameter $T$ (lebar jendela) bersifat **tetap** untuk seluruh analisis — inilah akar keterbatasan utama STFT.

### 1.2 Prinsip Ketidakpastian Heisenberg-Gabor

STFT terjebak dalam **Prinsip Ketidakpastian Waktu-Frekuensi** (*Heisenberg-Gabor Uncertainty Principle*):

$$
\Delta t \cdot \Delta f \ge \frac{1}{4\pi}
$$

di mana $\Delta t$ adalah resolusi temporal dan $\Delta f$ adalah resolusi frekuensi. Produk keduanya dibatasi oleh konstanta. Konsekuensinya:

| Kondisi | $\Delta t$ | $\Delta f$ | Akibat |
|:--------|:----------:|:----------:|:-------|
| Jendela **panjang** ($T$ besar) | Buruk (besar) | Baik (kecil) | Frekuensi akurat, onset waktu kabur |
| Jendela **pendek** ($T$ kecil) | Baik (kecil) | Buruk (besar) | Onset tajam, frekuensi tidak akurat |

STFT **tidak dapat** secara simultan memberikan resolusi waktu dan frekuensi yang baik. Ini adalah batasan *fisik*, bukan teknis.

### 1.3 Implikasi untuk Sinyal Prekursor Geomagnetik

Sinyal prekursor gempa bumi yang dihasilkan oleh mekanisme *microcracking* batuan (emisi piezoelektrik/fraktomagnetik) bersifat:

1. **Non-stasioner** — parameter statistik berubah terhadap waktu.
2. **Transient** — durasi singkat dengan onset mendadak.
3. **Multi-skala** — melibatkan interaksi antara variasi Sq harian (frekuensi ~11 $\mu$Hz) dan pulsasi Pc3 (22–100 mHz) yang berbeda 3–4 orde magnitudo.

Pada STFT, jendela tetap yang dioptimalkan untuk Pc3 akan **mengaburkan** (*smear*) onset transient, sementara jendela yang dioptimalkan untuk onset akan **kehilangan** informasi pita frekuensi sempit. Akibatnya, spektrogram 2D yang dihasilkan mengandung *ambiguity* spasio-temporal yang tidak dapat diatasi.

---

## 2. Superioritas CWT (Skalogram) melalui Multi-Resolusi

### 2.1 Definisi Matematis CWT

Transformasi Wavelet Kontinu (CWT) didefinisikan sebagai:

$$
W(a, b) = \frac{1}{\sqrt{a}} \int_{-\infty}^{\infty} x(t) \, \psi^*\left(\frac{t - b}{a}\right) dt
$$

dengan:
- $\psi(t)$ = *mother wavelet* (dalam kasus ini: Morlet, $\psi(t) = \pi^{-1/4} e^{j\omega_0 t} e^{-t^2/2}$)
- $a$ = parameter **dilasi** (skala, berbanding terbalik dengan frekuensi)
- $b$ = parameter **translasi** (waktu)

### 2.2 Multi-Resolution Analysis (MRA)

Keunggulan fundamental CWT dibanding STFT adalah **jendela adaptif**:

| Frekuensi | Skala $a$ | Jendela $T_{eff}$ | Resolusi Waktu | Resolusi Frekuensi |
|:----------|:---------:|:-----------------:|:--------------:|:------------------:|
| **Tinggi** (transient Pc3, ~80 mHz) | Kecil ($a \ll 1$) | Sempit ($T_{eff}$ kecil) | **Tajam** (presisi onset hingga detik) | Kurang presisi |
| **Rendah** (variasi Sq, ~0.01 mHz) | Besar ($a \gg 1$) | Lebar ($T_{eff}$ besar) | Kasar | **Tajam** (frekuensi akurat) |

Dengan kata lain, CWT **secara otomatis** menyesuaikan resolusi berdasarkan konten frekuensi sinyal — sesuatu yang secara fundamental tidak mungkin dilakukan oleh STFT. Hubungan ini memenuhi prinsip *wavelet Heisenberg boxes* yang optimal:

$$
\Delta t_a \cdot \Delta f_a \ge \frac{1}{4\pi}
$$

Namun $\Delta t_a$ dan $\Delta f_a$ **bervariasi secara proporsional terhadap skala** $a$, sehingga produk ketidakpastian tetap konstan namun distribusinya adaptif — berbeda dengan STFT yang kaku.

### 2.3 Keunggulan Empiris untuk Prekursor Gempa

Dalam konteks ScalogramV3, CWT memberikan:

1. **Deteksi Onset Presisi Tinggi:** Transien elektromagnetik akibat *stress activation* batuan yang terjadi dalam rentang menit hingga jam dapat ditangkap dengan presisi temporal tinggi oleh wavelet Morlet pada skala kecil.
2. **Diskriminasi Pita Frekuensi:** Variasi Sq harian (DC–0.01 mHz) dan pulsasi Pc3 (22–100 mHz) dapat direpresentasikan secara bersamaan dalam satu citra 2D tanpa kehilangan informasi.
3. **Robust terhadap Noise:** Koefisien CWT pada skala besar secara alami menekan noise frekuensi tinggi melalui efek *smoothing* bawaan wavelet.

---

## 3. Kronologi ScalogramV3: Mengapa V1 dan V2 Berstatus N/A

### 3.1 Fase Eksplorasi (Pre-V3)

| Fase | Metode Output | Status | Penyebab |
|:-----|:--------------|:------:|:---------|
| **V1** | Spektrogram STFT 128×128×3 | **N/A** | Resolusi temporal tidak memadai; onset transient mengalami *smearing* yang menghilangkan informasi fisis onset microcracking. Parameter jendela tetap tidak mampu mengakomodasi rentang Sq–Pc3. Tidak dapat direproduksi secara konsisten antar stasiun. |
| **V2** | Spektrogram STFT 256×256×3 (upscaled) | **N/A** | Peningkatan resolusi piksel tidak mengatasi keterbatasan fundamental STFT (Heisenberg). Upscaling hanya memperhalus *smearing* tanpa menyelesaikan ambiguitas waktu-frekuensi. Overparameterisasi model tanpa justifikasi fisis. |
| **V3** | **Skalogram CWT (79×168×3)** | **AKTIF** (20 Apr 2026) | Standardisasi CWT Morlet menyelesaikan seluruh keterbatasan V1/V2. Representasi multi-resolusi 79 skala frekuensi mampu menangkap onset transient dan variasi Sq secara simultan. Arsitektur EfficientNet-B1 dioptimalkan untuk tensor 3-channel [H, D, Z]. |

### 3.2 Justifikasi Keputusan N/A

Keputusan menetapkan status V1 = V2 = N/A didasarkan pada pertimbangan berikut:

1. **Integritas Ilmiah:** Menyajikan model dengan metode ekstraksi fitur yang memiliki keterbatasan fundamental (STFT) sebagai *baseline* yang valid akan menyesatkan komunitas ilmiah.
2. **Reprodusibilitas:** Parameter STFT yang bervariasi antar percobaan V1/V2 tidak menghasilkan konsistensi yang diperlukan untuk publikasi jurnal Q1.
3. **Homogenitas Dataset:** Standardisasi CWT pada V3 memastikan seluruh tensor input (79×168×3) memiliki interpretasi fisis yang seragam: sumbu-x adalah waktu (dengan resolusi adaptif), sumbu-y adalah skala frekuensi (log-spaced), dan kanal adalah [H, D, Z].
4. **Kepatuhan Standar:** ScalogramV3 dirilis sebagai *single baseline* untuk menghindari "cherry-picking" antar versi dan memenuhi prinsip FAIR (*Findable, Accessible, Interoperable, Reusable*).

### 3.3 Dataset `scalogram_v3_cosmic_final.h5`

File HDF5 ini mengemas tensor 3-channel dengan spesifikasi:

| Dimensi | Ukuran | Makna Fisis |
|:--------|:------:|:------------|
| Skala frekuensi | 79 | Log-spaced dari ~168 mHz hingga ~6 mHz — mencakup Pc3 (22–100 mHz) dan Pc4 (7–22 mHz) |
| Waktu | 168 | 168 titik waktu per jendela referensi CWT |
| Kanal | 3 | [H] Horizontal, [D] Deklinasi, [Z] Vertikal |

Total sampel: **37,000** tensor (4,500 prekursor, 32,500 normal) — terpartisi dalam train/validation/test.

---

## 4. Verifikasi Empiris: STFT vs CWT

### 4.1 Simulasi Komparasi

Sebagai verifikasi pendukung, telah dilakukan simulasi numerik (tersaji pada **Eviden 16** — `eviden16_stft_vs_cwt_plot.py`) yang membandingkan performa STFT dan CWT pada sinyal uji mengandung:

- Variasi frekuensi rendah (0.001 Hz) — analog variasi Sq harian
- Transien impulsif (0.08 Hz pada Pc3) — analog emisi microcracking pra-gempa

### 4.2 Hasil Observasi

| Aspek | STFT (Spektrogram) | CWT (Skalogram) |
|:------|:------------------:|:----------------:|
| Resolusi onset transient | **Terkaburkan** (*smearing* ~±20 dtk) | **Tajam** (<±3 dtk) |
| Diskriminasi frekuensi rendah | Baik (jendela panjang) | **Baik** (skala besar) |
| Representasi multi-skala | **Tidak mungkin** | **Otomatis via MRA** |
| Stabilitas terhadap noise | Sedang | **Tinggi** (smoothing wavelet) |

Grafik komparasi disajikan pada `eviden16_stft_vs_cwt_comparison.png` dan menunjukkan secara visual bahwa **efek *smearing* pada STFT menghilangkan informasi onset** yang justru merupakan fitur paling kritis untuk deteksi prekursor gempa.

---

## 5. Kesimpulan dan Implikasi

1. **STFT secara fundamental tidak memadai** untuk analisis sinyal prekursor geomagnetik yang non-stasioner dan multi-skala, karena terjebak dalam *trade-off* resolusi Heisenberg-Gabor yang kaku.

2. **CWT Morlet dengan Multi-Resolution Analysis** memberikan representasi waktu-frekuensi yang adaptif, memungkinkan deteksi onset transient presisi tinggi dan diskriminasi pita frekuensi secara simultan — sesuatu yang tidak dapat dicapai STFT.

3. **Status N/A untuk V1 dan V2 adalah keputusan ilmiah yang benar**: model dengan ekstraksi fitur yang tidak sesuai secara fisis tidak layak dijadikan *baseline* komparatif. ScalogramV3 (20 April 2026) adalah titik awal yang sahih untuk seluruh analisis disertasi ini.

4. **Implikasi untuk riset ke depan**: Standardisasi CWT pada ScalogramV3 membuka jalan untuk adopsi arsitektur vision transformer (ViT) dan *self-supervised learning* pada tensor skalogram, yang telah menunjukkan performa unggul pada data seismik dan geomagnetik di literatur terkini.

---

## Daftar Pustaka Kunci

1. Mallat, S. (2008). *A Wavelet Tour of Signal Processing: The Sparse Way*. 3rd ed. Academic Press.
2. Heil, C. E., & Walnut, D. F. (1989). Continuous and discrete wavelet transforms. *SIAM Review*, 31(4), 628–666.
3. Gabor, D. (1946). Theory of communication. *Journal of the Institution of Electrical Engineers*, 93(26), 429–457.
4. Daubechies, I. (1992). *Ten Lectures on Wavelets*. SIAM.
5. Hattori, K., et al. (2004). ULF geomagnetic changes associated with large earthquakes. *Terrestrial, Atmospheric and Oceanic Sciences*, 15(3), 329–360.
6. Masci, F., & Thomas, J. N. (2015). Are there new findings in the search for ULF magnetic precursors to earthquakes? *Journal of Geophysical Research: Space Physics*, 120(12), 10,289–10,304.

---

*Dokumen ini disusun sebagai bagian dari kelengkapan akademik Progress Report 4 — Disertasi Doktoral*
*Program Studi Teknik Fisika — Institut Teknologi Sepuluh Nopember*
*ScalogramV3 — 20 April 2026*
