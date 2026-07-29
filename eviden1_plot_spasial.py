#!/usr/bin/env python3
"""
EVIDEN 1: Peta Sebaran Spasial 24 Stasiun MAGDAS-BMKG — ScalogramV3
====================================================================
Memplot sebaran 24 stasiun geomagnetik MAGDAS BMKG di Indonesia
dengan overlay batas lempeng tektonik dan zona subduksi.
Proyek ScalogramV3 — Standardisasi CWT (rilis 20 April 2026).
Disimpan sebagai 'eviden1_peta_stasiun.png' (>=300 dpi).
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import cartopy.crs as ccrs
import cartopy.feature as cfeature

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

STATIONS = {
    "PSMG (Padang)":          (-0.948, 100.353),
    "BTNG (Batang)":          (-6.485, 110.398),
    "DJAK (Jakarta)":         (-6.172, 106.828),
    "PNGK (Pinggung)":        (-6.633, 106.857),
    "TBNG (Tuban)":           (-6.900, 112.050),
    "SRKI (Surakarta)":       (-7.570, 110.830),
    "STPA (Stabat)":          (3.735, 98.476),
    "LAMP (Langsa)":          (4.720, 97.970),
    "KUTA (Kuta Raja)":       (5.545, 95.350),
    "BAIG (Banda Aceh)":      (5.560, 95.320),
    "SBAT (Sabang)":          (5.530, 95.320),
    "JYPG (Jayapura)":        (-2.530, 140.720),
    "AMQI (Ambon)":           (-3.695, 128.160),
    "KPSI (Kupang)":          (-10.175, 123.610),
    "PLKI (Palu)":            (-0.892, 119.870),
    "PLNI (Palu 2)":          (-0.870, 119.850),
    "MTWA (Matuwa)":          (-8.500, 117.500),
    "SAIQ (Sangihe)":         (3.500, 125.500),
    "MNAK (Manado)":          (1.480, 124.850),
    "KUMT (Kumai)":           (-2.230, 111.700),
    "KUM2 (Kumai 2)":         (-2.250, 111.720),
    "CPTG (Ciptagantung)":    (-7.130, 107.600),
    "TNDI (Tondano)":         (1.100, 124.800),
    "MGNI (Mangani)":         (-7.960, 112.630),
}

trench_lons = [95.0, 100.0, 103.0, 105.5, 107.0, 108.5, 110.0, 112.0, 115.0, 118.0, 121.0, 124.0, 127.0, 130.0]
trench_lats = [10.0, 6.0, 3.5, -2.0, -5.5, -7.5, -8.5, -9.0, -9.5, -9.0, -8.5, -7.5, -7.0, -6.5]

faults = {
    "P. Sumatera (Semangko)": {'lons': [98.0, 100.0, 101.5, 103.0], 'lats': [4.5, 2.0, -0.5, -3.0]},
    "P. Barat Jawa":          {'lons': [105.5, 106.0, 107.0, 108.0], 'lats': [-6.0, -6.8, -7.5, -8.2]},
    "Palu-Koro":              {'lons': [119.5, 119.9, 120.3, 121.0], 'lats': [0.2, -0.5, -1.2, -2.0]},
}

north = {k: v for k, v in STATIONS.items() if v[0] > 0}
south = {k: v for k, v in STATIONS.items() if v[0] <= 0}

def main():
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_extent([93, 142, -12, 8], crs=ccrs.PlateCarree())

    ax.add_feature(cfeature.LAND, facecolor='#f5f0e8', edgecolor='none', alpha=0.6)
    ax.add_feature(cfeature.OCEAN, facecolor='#d4e6f1', alpha=0.7)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, color='#7f8c8d')
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle='--', color='#95a5a6')

    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False; gl.right_labels = False

    ax.plot(trench_lons, trench_lats, color='#e74c3c', linewidth=1.5, linestyle='--',
            transform=ccrs.PlateCarree(), zorder=5)
    for name, coords in faults.items():
        ax.plot(coords['lons'], coords['lats'], color='#e67e22', linewidth=1.2,
                linestyle='-', transform=ccrs.PlateCarree(), zorder=5)

    for name, (lat, lon) in north.items():
        short = name.split('(')[0].strip()
        ax.plot(lon, lat, 'o', color='#2980b9', markersize=8, markeredgecolor='white',
                markeredgewidth=0.8, transform=ccrs.PlateCarree(), zorder=10)
        ax.annotate(short, (lon, lat), textcoords="offset points", xytext=(5, 5),
                    fontsize=5.5, fontweight='bold', color='#2c3e50',
                    transform=ccrs.PlateCarree(), zorder=11)

    for name, (lat, lon) in south.items():
        short = name.split('(')[0].strip()
        ax.plot(lon, lat, 's', color='#27ae60', markersize=8, markeredgecolor='white',
                markeredgewidth=0.8, transform=ccrs.PlateCarree(), zorder=10)
        ax.annotate(short, (lon, lat), textcoords="offset points", xytext=(5, -8),
                    fontsize=5.5, fontweight='bold', color='#2c3e50',
                    transform=ccrs.PlateCarree(), zorder=11)

    ax.set_title('Peta Sebaran 24 Stasiun MAGDAS-BMKG — ScalogramV3\n'
                 'Overlay Zona Subduksi & Patahan Aktif Indonesia',
                 fontsize=13, fontweight='bold', pad=15)

    legend = [
        Line2D([0],[0], marker='o', color='w', markerfacecolor='#2980b9', markersize=10,
               label=f'Stasiun Lintang Utara ({len(north)})'),
        Line2D([0],[0], marker='s', color='w', markerfacecolor='#27ae60', markersize=10,
               label=f'Stasiun Lintang Selatan ({len(south)})'),
        Line2D([0],[0], color='#e74c3c', linewidth=1.5, linestyle='--', label='Zona Subduksi'),
        Line2D([0],[0], color='#e67e22', linewidth=1.2, label='Patahan Aktif'),
    ]
    ax.legend(handles=legend, loc='lower left', fontsize=8, framealpha=0.9, edgecolor='gray')

    out_path = os.path.join(OUTPUT_DIR, 'eviden1_peta_stasiun.png')
    fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"[OK] Peta spasial ScalogramV3 tersimpan: {out_path}")
    print(f"     Total stasiun: {len(STATIONS)} | Utara: {len(north)} | Selatan: {len(south)}")

if __name__ == '__main__':
    main()
