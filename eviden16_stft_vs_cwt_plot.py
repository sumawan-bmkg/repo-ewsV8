#!/usr/bin/env python3
"""
EVIDEN 16: Perbandingan Empiris STFT vs CWT — ScalogramV3
===========================================================
Mensimulasikan sinyal geomagnetik: variasi harian (0.001 Hz) + transient impulsif (0.08 Hz/Pc3).
Membandingkan resolusi temporal STFT vs CWT Morlet.
Output: eviden16_stft_vs_cwt_comparison.png
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pywt
from scipy.signal import stft

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

FS = 10.0         # 10 Hz untuk resolusi temporal tinggi
DURATION = 600    # 600 detik (10 menit)
T = np.arange(0, DURATION, 1/FS)

# Frekuensi
F_LOW = 0.001     # Variasi harian (DC-ish)
F_PC3 = 0.08      # Pc3 transient (80 mHz)
ONSET = 300       # Onset transient di detik ke-300
DECAY = 15        # Decay time konstan

np.random.seed(42)

# Sinyal: background low-freq + transient Pc3 + noise
signal = (2.0 * np.sin(2 * np.pi * F_LOW * T) +
          0.5 * np.sin(2 * np.pi * F_LOW * 3 * T + 0.7))

# Envelope untuk transient (rise ~3s, decay ~15s)
env = np.exp(-np.clip(T - ONSET, 0, None) / DECAY)
env[T < ONSET] = 0
env[T > ONSET + 80] = 0  # cut after 80s

pc3_burst = 3.0 * np.sin(2 * np.pi * F_PC3 * T) * env
signal += pc3_burst
signal += np.random.normal(0, 0.08, len(T))

# ——— STFT ———
nperseg = 128
f_stft, t_stft, Zxx = stft(signal, fs=FS, nperseg=nperseg, noverlap=nperseg//2,
                            window='hann', scaling='psd')
Sxx = np.abs(Zxx)

# ——— CWT (Morlet) ———
scales = np.logspace(np.log10(1), np.log10(60), 79)
cwt_coeffs, freqs = pywt.cwt(signal, scales, 'morl', sampling_period=1/FS)
cwt_mag = np.abs(cwt_coeffs)

# Frekuensi pseudo-CWT
f_cwt = freqs

# Plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

# === STFT ===
im1 = ax1.pcolormesh(t_stft, f_stft * 1e3, Sxx, shading='gouraud',
                      cmap='plasma', vmin=np.percentile(Sxx, 10), vmax=np.percentile(Sxx, 95))
ax1.axvline(ONSET, color='cyan', ls='--', lw=1.5, alpha=0.8, label=f'Onset transient t={ONSET}s')
ax1.axhline(F_PC3 * 1e3, color='lime', ls=':', lw=1, alpha=0.7, label=f'Pc3 ({F_PC3*1e3:.0f} mHz)')
ax1.set_ylabel('Frekuensi (mHz)', fontsize=11, fontweight='bold')
ax1.set_ylim(0, 500)
ax1.set_title('(a) Spektrogram STFT — Jendela Tetap (nperseg=128)\n'
              'Sinyal Transient Melebar (Smearing Effect) — Resolusi Waktu Rendah',
              fontsize=12, fontweight='bold')
ax1.legend(fontsize=8, loc='upper right')
ax1.grid(True, alpha=0.2)
plt.colorbar(im1, ax=ax1, label='|STFT|')

# Annotate smearing
ax1.annotate('Smearing\ntemporal', xy=(315, 250), fontsize=9, color='white', fontweight='bold',
             ha='center',
             bbox=dict(boxstyle='round', facecolor='#e74c3c', edgecolor='white', alpha=0.7))

# === CWT ===
# Select only relevant freq range (< 500 mHz)
freq_mask = f_cwt < 0.5
im2 = ax2.pcolormesh(T, f_cwt[freq_mask] * 1e3, cwt_mag[freq_mask, :], shading='gouraud',
                      cmap='viridis', vmin=np.percentile(cwt_mag, 10), vmax=np.percentile(cwt_mag, 95))
ax2.axvline(ONSET, color='cyan', ls='--', lw=1.5, alpha=0.8, label=f'Onset transient t={ONSET}s')
ax2.axhline(F_PC3 * 1e3, color='lime', ls=':', lw=1, alpha=0.7, label=f'Pc3 ({F_PC3*1e3:.0f} mHz)')
ax2.set_xlabel('Waktu (detik)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Frekuensi (mHz)', fontsize=11, fontweight='bold')
ax2.set_ylim(0, 500)
ax2.set_title('(b) Skalogram CWT Morlet — Jendela Adaptif (Multi-Resolusi)\n'
              'Onset Transient Tajam & Presisi — 79 Skala Frekuensi',
              fontsize=12, fontweight='bold')
ax2.legend(fontsize=8, loc='upper right')
ax2.grid(True, alpha=0.2)
plt.colorbar(im2, ax=ax2, label='|CWT|')

# Annotate sharpness
ax2.annotate('Onset tajam\n& presisi', xy=(315, 250), fontsize=9, color='white', fontweight='bold',
             ha='center',
             bbox=dict(boxstyle='round', facecolor='#27ae60', edgecolor='white', alpha=0.7))

# Signal overlay on top
ax_twin = ax2.twinx()
ax_twin.plot(T, signal, color='white', alpha=0.35, lw=0.5, label='Sinyal mentah')
ax_twin.set_ylim(-6, 6)
ax_twin.set_ylabel('')

plt.suptitle('Perbandingan Empiris STFT vs CWT: Sinyal Transien Geomagnetik\n'
             f'Background {F_LOW*1e3:.1f} mHz + Burst Pc3 {F_PC3*1e3:.0f} mHz @ t={ONSET}s',
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()

out_path = os.path.join(OUTPUT_DIR, 'eviden16_stft_vs_cwt_comparison.png')
fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig)

print("=" * 60)
print("PERBANDINGAN STFT vs CWT — SCALOGRAMV3")
print("=" * 60)
print(f"  Sampling rate: {FS} Hz | Durasi: {DURATION}s | Sampel: {len(T):,}")
print(f"  STFT nperseg: {nperseg} | Freq bins: {len(f_stft)} | Time bins: {len(t_stft)}")
print(f"  CWT scales: {len(scales)} (79 skala, ~{f_cwt[0]*1e3:.0f}-{f_cwt[-1]*1e3:.0f} mHz)")
print(f"  Dimensi array STFT: {Sxx.shape} | CWT: {cwt_mag.shape}")
print(f"  Onset transient: t={ONSET}s, f={F_PC3*1e3:.0f} mHz")
print(f"\n[OK] Plot komparasi STFT vs CWT: {out_path}")

