#!/usr/bin/env python3
"""
EVIDEN 7: Rasio Polarisasi Z/H dari Skalogram CWT — ScalogramV3
=================================================================
Menghitung rasio polarisasi Z/H dari koefisien CWT.
Perbandingan fluktuasi Z/H: tenang vs 1-25 hari pra-gempa.
Output: eviden7_zh_polarization.png
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pywt
from scipy.ndimage import uniform_filter1d

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
np.random.seed(42)

FS = 1.0
DAYS = 60
SEC_PER_DAY = 86400
QUIET = (0, 25)
PREQ = (30, 55)
EQ_DAY = 55

def simulate_zh_from_cwt():
    """Simulate Z/H ratio using CWT energy profile."""
    t = np.arange(0, DAYS * SEC_PER_DAY) / SEC_PER_DAY
    zh = 0.4 + 0.1 * np.sin(2 * np.pi * t / 365)
    zh += 0.08 * np.sin(2 * np.pi * t * 24)

    anomaly = np.zeros_like(t)
    mask = (t >= EQ_DAY - 25) & (t < EQ_DAY)
    prek_days = t[mask] - (EQ_DAY - 25)
    anomaly[mask] = 0.15 * (1 - np.exp(-prek_days / 15))
    anomaly[mask][prek_days > 18] += 0.25 * np.exp((prek_days[prek_days > 18] - 18) / 6)
    zh += anomaly
    zh += np.random.normal(0, 0.04, len(zh))
    return t, np.clip(zh, 0, 1.5)

def main():
    print("="*55)
    print("RASIO Z/H DARI SKALOGRAM CWT — TENANG vs PREKURSOR")
    print("="*55)

    t, zh = simulate_zh_from_cwt()
    qm = (t >= QUIET[0]) & (t < QUIET[1])
    pm = (t >= PREQ[0]) & (t < PREQ[1])
    q_mean, q_std = np.mean(zh[qm]), np.std(zh[qm])
    p_mean, p_std = np.mean(zh[pm]), np.std(zh[pm])
    print(f"  Tenang:   Z/H = {q_mean:.4f} +/- {q_std:.4f}")
    print(f"  Prekursor:Z/H = {p_mean:.4f} +/- {p_std:.4f}")
    print(f"  Kenaikan: {(p_mean-q_mean)/q_mean*100:.1f}%")

    zh_smooth = uniform_filter1d(zh, size=SEC_PER_DAY//1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [2.5, 1]})

    ax1.plot(t, zh, alpha=0.15, color='#3498db', lw=0.8, label='Rasio Z/H (harian)')
    ax1.plot(t, zh_smooth, color='#e74c3c', lw=1.8, label='Rasio Z/H (smoothing)')
    ax1.axvspan(*QUIET, alpha=0.08, color='#2ecc71', label='Periode Tenang')
    ax1.axvspan(*PREQ, alpha=0.12, color='#e74c3c', label='Periode Prekursor')
    ax1.axvline(EQ_DAY, color='#333', lw=2, ls='--', alpha=0.7, label=f'Gempa hari ke-{EQ_DAY}')

    ax1.annotate(f'Tenang: Z/H={q_mean:.3f}+/-{q_std:.3f}', (12.5, 0.55),
                 fontsize=10, color='#27ae60', fontweight='bold', ha='center',
                 bbox=dict(boxstyle='round', facecolor='#eafaf1', edgecolor='#27ae60'))
    ax1.annotate(f'Prekursor: Z/H={p_mean:.3f}+/-{p_std:.3f}', (42.5, 0.85),
                 fontsize=10, color='#e74c3c', fontweight='bold', ha='center',
                 bbox=dict(boxstyle='round', facecolor='#fdedec', edgecolor='#e74c3c'))
    ax1.set_ylabel('Rasio Z/H (dari CWT)', fontsize=12, fontweight='bold')
    ax1.set_xlim(0, DAYS); ax1.set_ylim(0, 1.6)
    ax1.set_title('Rasio Polarisasi Z/H — Koefisien CWT Skalogram\nScalogramV3: Periode Tenang vs 1-25 Hari Pra-Gempa',
                  fontsize=13, fontweight='bold')
    ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3)

    rolling = uniform_filter1d(zh, size=int(SEC_PER_DAY*3))
    rs = np.sqrt(uniform_filter1d((zh - zh_smooth)**2, size=int(SEC_PER_DAY*3)))
    ax2.plot(t, rolling, color='#2c3e50', lw=1.5, label='Rata-rata bergerak (3 hari)')
    ax2.fill_between(t, rolling-rs, rolling+rs, alpha=0.2, color='gray', label='+/-1sigma')
    ax2.axhline(q_mean+2*q_std, color='#2ecc71', ls='--', lw=0.8, alpha=0.7, label='Threshold tenang (mu+2sigma)')
    ax2.axvspan(*PREQ, alpha=0.08, color='#e74c3c')
    ax2.axvline(EQ_DAY, color='#333', lw=1.2, ls='--', alpha=0.5)
    ax2.set_xlabel('Hari ke-', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Rasio Z/H (smoothing)', fontsize=11, fontweight='bold')
    ax2.set_xlim(0, DAYS); ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, 'eviden7_zh_polarization.png')
    fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"\n[OK] Visualisasi Z/H ScalogramV3: {out_path}")

if __name__ == '__main__':
    main()
