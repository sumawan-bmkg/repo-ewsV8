import subprocess, sys, os
BASE = 'D:/multi/scalogramv3/disertasi4/supcon'
scripts = [
    '01_pipeline/generate_evidence1.py',
    '02_dataset/generate_evidence2.py',
    '03_architecture/generate_evidence3.py',
    '04_training/generate_evidence4.py',
    '06_evaluation/generate_evidence6.py',
    '07_latent/generate_evidence7.py',
    '09_blindtest/generate_evidence9.py',
    '11_comparison/generate_evidence11.py',
    '12_summary/generate_evidence12.py',
]
os.chdir('D:/multi/scalogramv3')
for s in scripts:
    print(f'\n{"="*60}')
    print(f'Running: {s}')
    r = subprocess.run([sys.executable, os.path.join(BASE, s)],
                       capture_output=True, text=True, timeout=120)
    print(r.stdout)
    if r.returncode != 0:
        print(f'ERROR: {r.stderr}')
    else:
        print(f'OK: {s}')
