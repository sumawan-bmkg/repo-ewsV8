#!/usr/bin/env python3
"""
EVIDEN 5: Perbandingan Skalogram Sebelum & Sesudah Filtering Pc3 — ScalogramV3
================================================================================
Mensimulasikan deret waktu geomagnetik (Sq + noise + Pc3 prekursor),
menerapkan filter Butterworth Pc3, lalu menghitung CWT (Morlet wavelet)
untuk menghasilkan skalogram 2D. Plot side-by-side RAW vs CLEAN.
Output: eviden5_scalogram_comparison.png
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pywt
from scipy.signal import butter, sosfilt

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
FS = 1.0; DURATION = 3600 * 48
LOW, HIGH = 0.022, 0.1
NPERSIST = 128  # Scalar untuk resolusi frekuensi skalogram

np.random.seed(42)

def main():
    t = np.arange(0, DURATION) / FS

    # Sq daily variation + harmonics
    signal = (25*np.sin(2*np.pi*t/86400) + 10*np.sin(4*np.pi*t/86400+0.3) + 5*np.sin(6*np.pi*t/86400+0.7))
    signal *= (0.8 + 0.3*np.sin(2*np.pi*t/86400 - np.pi/2))

    # Pc3 embedded in last 18h
    pc3_t = DURATION - 18*3600
    mask = t >= pc3_t
    pc3 = (3*np.sin(2*np.pi*0.045*t[mask]) + 2*np.sin(2*np.pi*0.070*t[mask]+1.2) + 1.5*np.sin(2*np.pi*0.088*t[mask]+2.1))
    signal[mask] += pc3 * np.linspace(0.1, 1.0, sum(mask))
    signal += np.random.normal(0, 2.0, DURATION)

    # Spikes
    for sp in np.random.choice(DURATION, 60, replace=False):
        signal[sp] += np.random.uniform(30, 80) * np.random.choice([-1, 1])

    # Apply Pc3 filter
    nyq = FS/2
    sos = butter(4, [LOW/nyq, HIGH/nyq], btype='band', output='sos')
    clean = sosfilt(sos, signal)

    # CWT Scalogram — Morlet wavelet
    scales = np.logspace(np.log10(1), np.log10(128), NPERSIST)
    coeffs_raw, freqs = pywt.cwt(signal, scales, 'morl', sampling_period=1/FS)
    coeffs_clean, _ = pywt.cwt(clean, scales, 'morl', sampling_period=1/FS)

    # Plot side-by-side
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    data = [(coeffs_raw, 'RAW (mentah + Sq + noise)', 'plasma'),
            (coeffs_clean, 'CLEAN (filter Butterworth Pc3 0.022-0.1 Hz)', 'viridis')]

    for ax, (coeff, title, cmap) in zip(axes, data):
        sc = np.abs(coeff)
        vmin, vmax = np.percentile(sc, 5), np.percentile(sc, 95)
        im = ax.imshow(sc, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax,
                       extent=[0, DURATION/3600, freqs[-1]*1e3, freqs[0]*1e3])
        ax.axhline(LOW*1e3, color='cyan', ls='--', lw=0.8, alpha=0.7)
        ax.axhline(HIGH*1e3, color='cyan', ls='--', lw=0.8, alpha=0.7, label=f'Pc3 band')
        ax.set_xlabel('Waktu (jam)', fontsize=10, fontweight='bold')
        ax.set_ylabel('Frekuensi (mHz)', fontsize=10, fontweight='bold')
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_ylim(freqs[-1]*1e3, freqs[0]*1e3)
        ax.legend(fontsize=7); ax.grid(True, alpha=0.15)
        plt.colorbar(im, ax=ax, label='|CWT Coeff|')

    plt.suptitle('Perbandingan Skalogram CWT — Morlet Wavelet\nScalogramV3: Sebelum & Sesudah Filtering Pc3',
                 fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, 'eviden5_scalogram_comparison.png')
    fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"\n[OK] Skalogram comparison tersimpan: {out_path}")
    print(f"     Shape CWT coeffs: {np.abs(coeffs_clean).shape} | Scales: {len(scales)} | Freq range: {freqs[-1]*1e3:.1f}-{freqs[0]*1e3:.1f} mHz")

if __name__ == '__main__':
    main()
