#!/usr/bin/env python3
"""
EVIDEN 3: Respon Filter Butterworth Bandpass Pc3 — ScalogramV3
===============================================================
Plot Magnitude & Phase Response filter Butterworth orde 4
pada pita Pc3 (0.022-0.1 Hz). Pra-kondisi sinyal sebelum CWT.
Roll-off >= 80 dB/decade (orde 4 x 20 dB/dec).
Output: eviden3_filter_response.png
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfreqz

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
ORDER, LOW, HIGH, FS = 4, 0.022, 0.1, 1.0

def main():
    nyq = FS / 2.0
    sos = butter(ORDER, [LOW/nyq, HIGH/nyq], btype='band', output='sos')
    w, h = sosfreqz(sos, worN=4096, fs=FS)
    mag = 20 * np.log10(np.abs(h) + 1e-12)
    phase = np.degrees(np.unwrap(np.angle(h)))

    # Roll-off high stopband
    hi = w > HIGH * 2
    if sum(hi) > 10:
        i1 = np.argmin(np.abs(w - HIGH*2))
        i2 = np.argmin(np.abs(w - HIGH*5))
        ro = (mag[i1] - mag[i2]) / np.log10(w[i2]/w[i1]) if w[i2]/w[i1] > 1 else 0
    else:
        ro = 0

    print(f"  Roll-off teoretis: {ORDER*20} dB/dec | Terukur: {ro:.1f} dB/dec | {'TERPENUHI' if ro>=75 else 'CEK'}")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    ax1.plot(w, mag, color='#2c3e50', lw=1.5, label='Magnitude Response')
    ax1.axvspan(LOW, HIGH, alpha=0.15, color='#27ae60', label=f'Passband Pc3 ({LOW*1e3:.0f}-{HIGH*1e3:.0f} mHz)')
    ax1.axhline(-3, color='#e74c3c', ls='--', lw=0.8, alpha=0.7, label='-3 dB')
    ax1.set_ylabel('Magnitude (dB)', fontsize=11, fontweight='bold')
    ax1.set_ylim(-120, 5)
    ax1.set_title(f'Respon Filter Butterworth Bandpass Orde {ORDER} — Pita Pc3 (Pra-CWT ScalogramV3)', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)

    ax2.plot(w, phase, color='#8e44ad', lw=1.5, label='Phase Response')
    ax2.axvspan(LOW, HIGH, alpha=0.15, color='#27ae60')
    ax2.set_xlabel('Frekuensi (Hz)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Phase (derajat)', fontsize=11, fontweight='bold')
    ax2.set_xlim(0, 0.3); ax2.grid(True, alpha=0.3); ax2.legend(fontsize=8)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, 'eviden3_filter_response.png')
    fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"[OK] Respon filter tersimpan: {out_path}")

if __name__ == '__main__':
    main()
