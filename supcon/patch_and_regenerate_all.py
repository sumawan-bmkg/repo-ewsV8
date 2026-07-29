#!/usr/bin/env python3
"""Patch all evidence scripts: dark → white scientific theme, then regenerate."""
import os, sys, subprocess
from pathlib import Path

SUPCON = Path('D:/multi/scalogramv3/disertasi4/supcon')
replace_pairs = [
    # Theme globals
    ("BG = '#0D1117'\nTEXT = '#ECF0F1'",
     "BG = 'white'\nTEXT = '#000000'"),
    ("BG = '#0D1117'", "BG = 'white'"),
    ("TEXT = '#ECF0F1'", "TEXT = '#000000'"),
    # Figure bg
    ("fig.patch.set_facecolor(BG)", "fig.patch.set_facecolor('white')"),
    ("ax.set_facecolor('#161B22')", "ax.set_facecolor('white')"),
    ("ax.set_facecolor(BG)", "ax.set_facecolor('white')"),
    # Text color
    (", color=TEXT", ", color='#000000'"),
    ("color=TEXT,", "color='#000000',"),
    ("ax.tick_params(colors=TEXT, labelsize=8)",
     "ax.tick_params(colors='#000000', labelsize=8)"),
    ("ax.tick_params(colors=TEXT, labelsize=9)",
     "ax.tick_params(colors='#000000', labelsize=9)"),
    ("ax.tick_params(colors=TEXT)",
     "ax.tick_params(colors='#000000')"),
    # Edge colors
    ("edgecolor='white'", "edgecolor='#333333'"),
    # Spine colors
    (".set_color('#333')", ".set_color('#CCCCCC')"),
    # Pipeline specific
    ("color='white', zorder=4", "color='#000000', zorder=4"),
    ("color='#BDC3C7', zorder=4", "color='#666666', zorder=4"),
    ("facecolor='#2C3E50'", "facecolor='#4A7FB5'"),
    ("facecolor='#34495E'", "facecolor='#5A8FC5'"),
    ("facecolor='#1A252F'", "facecolor='#E8F0F5'"),
    ("edgecolor='#3498DB', linewidth=2", "edgecolor='#2C3E50', linewidth=2"),
    ("edgecolor='#7F8C8D', linewidth=1", "edgecolor='#999999', linewidth=1"),
    # Architecture box colors
    ("facecolor='#1A5276'", "facecolor='#2980B9'"),
    ("facecolor='#2E86C1'", "facecolor='#3498DB'"),
    ("facecolor='#1ABC9C'", "facecolor='#16A085'"),
    ("facecolor='#F39C12'", "facecolor='#D4AC0D'"),
    ("facecolor='#E67E22'", "facecolor='#CA6F1E'"),
    ("facecolor='#E74C3C'", "facecolor='#C0392B'"),
    ("facecolor='#9B59B6'", "facecolor='#8E44AD'"),
    # Highlight colors
    ("color='#F1C40F'", "color='#2C3E50'"),
    ("color='#95A5A6'", "color='#666666'"),
    # Arrows
    ("arrowprops=dict(arrowstyle='->', color='#E74C3C'",
     "arrowprops=dict(arrowstyle='->', color='#333333'"),
    # Subtitle text
    ("color='#BDC3C7', zorder=4", "color='#555555', zorder=4"),
    ("color='#BDC3C7'", "color='#555555'"),
    # Summary figure
    ("facecolor='#1A1A2E'", "facecolor='#F5F5F5'"),
    ("edgecolor='#555'", "edgecolor='#AAAAAA'"),
    # Title colors
    ("fig.suptitle('V8 SupCon", "fig.suptitle('V8 SupCon"),
    # Replace title color param
    ("color='#000000', y=0.95)", "color='#000000', y=0.95)"),
    ("color='#000000', y=1.01)", "color='#000000', y=1.01)"),
    ("color='#000000', y=0.98)", "color='#000000', y=0.98)"),
    ("color='#000000', y=1.02)", "color='#000000', y=1.02)"),
    ("color='#000000', y=1.01)", "color='#000000', y=1.01)"),
    # CM colormap
    ("cmap='Reds'", "cmap='YlOrRd'"),
    # Save
    ("facecolor=fig.get_facecolor()", "facecolor='white'"),
]

# Also handle specific title text colors that still have T variable
title_patterns = [
    ("fig.suptitle('", "fig.suptitle('"),
]

for ev_dir in sorted(SUPCON.iterdir()):
    if not ev_dir.is_dir():
        continue
    gen = ev_dir / f"generate_{ev_dir.name}.py"
    if not gen.exists():
        continue

    content = gen.read_text(encoding='utf-8')

    # Special: keep title colors black (they may already reference TEXT)
    for old, new in replace_pairs:
        content = content.replace(old, new)

    gen.write_text(content, encoding='utf-8')
    print(f"  Patched: {gen.name}")

print("\nRe-running all generators...")
for ev_dir in sorted(SUPCON.iterdir()):
    if not ev_dir.is_dir():
        continue
    gen = ev_dir / f"generate_{ev_dir.name}.py"
    if not gen.exists():
        continue
    r = subprocess.run([sys.executable, str(gen)], capture_output=True, text=True, timeout=120)
    if r.returncode == 0:
        print(f"  [OK] {ev_dir.name}")
    else:
        print(f"  [ERR] {ev_dir.name}: {r.stderr.strip()[:200]}")
