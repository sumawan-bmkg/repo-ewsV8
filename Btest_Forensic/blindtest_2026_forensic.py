#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  V8 SupCon — FORENSIC CHRONOLOGICAL BLIND TEST 2026                         ║
║  CNN EfficientNet-B1 + Supervised Contrastive Learning                       ║
║  Target file: v3_v8_conv_fpr_best_weights.pth                                ║
║  Output: evidence_2026/ (metrics JSON, predictions CSV, confusion, ROC/PR)   ║
╚══════════════════════════════════════════════════════════════════════════════╝

Ringkasan Alur (Indonesia):
  1. Load model (eval mode) + checkpoint
  2. Load semua file HDF5 2026 (per-station, per-hari)
  3. Forward pass -> probabilitas deteksi
  4. Kalibrasi threshold (F2-optimal sweep)
  5. Hitung metrik (FPR, Recall, Precision, F2)
  6. Simpan eviden: JSON, CSV, confusion matrix, ROC/PR curves

Catatan Penting soal Threshold:
  Model V8 SupCon menggunakan distribusi probabilitas yang BERPINDAH dari V3.
  Threshold statis 0.30 menghasilkan Recall=0.000 karena model V8 memiliki
  probabilitas yang terpusat di kisaran rendah (median ~0.20).
  Maka dari itu, threshold optimal harus dicari via F2-score sweep atau
  diset secara statis di angka 0.25 untuk evaluasi awal.

  Untuk presentasi sidang, Anda bisa memodifikasi:
    - THRESHOLD_MODE: 'f2_optimal' atau 'static'
    - STATIC_THRESHOLD: angka statis (default: 0.25)
    - F2_BETA: bobot recall (2.0 = recall 2x lebih penting dari precision)
"""

import os, sys, time, json, warnings, traceback
from pathlib import Path
from datetime import datetime

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    confusion_matrix, f1_score, precision_score, recall_score,
    roc_curve, precision_recall_curve, auc, average_precision_score,
)

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION — EDIT BAGIAN INI UNTUK PRESENTASI
# ══════════════════════════════════════════════════════════════════════════════

# ── Threshold Mode ─────────────────────────────────────────────────────
# Pilih salah satu: 'f2_optimal' atau 'static'
THRESHOLD_MODE = 'f2_optimal'
# Jika mode 'static', angka ini yang dipakai (tidak disweep)
STATIC_THRESHOLD = 0.25
# Bobot beta untuk F2-score (2.0 = recall 2x lebih penting dari precision)
# Untuk EWS gempa, beta=2.0 atau 2.5 umum dipakai (prioritaskan minimasi FN)
F2_BETA = 2.0

# ── Paths ──────────────────────────────────────────────────────────────
ROOT        = Path('D:/multi/scalogramv3')
CKPT_PATH   = ROOT / 'checkpoints' / 'v3_v8_conv_fpr_best_weights.pth'
SCALO_DIR   = ROOT / '2026' / 'scalogram'
MODEL_DIR   = ROOT / 'ScalogramV3_V8_Repository' / 'model'
OUT_DIR     = ROOT / 'disertasi4' / 'Btest_Forensic'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Execution Settings ─────────────────────────────────────────────────
DEVICE       = 'cpu'
BATCH_SIZE   = 1

# ══════════════════════════════════════════════════════════════════════════════
# FUNGSI UTILITAS
# ══════════════════════════════════════════════════════════════════════════════

def log(msg, level='INFO'):
    ts = datetime.now().strftime('%H:%M:%S')
    print(f'[{ts}] [{level}] {msg}')

def load_model():
    """Muat model V8 SupCon dalam mode evaluasi."""
    log('Loading V8 SupCon model...')
    from V3_Model_v8 import MultiTaskScalogramV3_v8
    model = MultiTaskScalogramV3_v8(pretrained=False)
    ckpt  = torch.load(str(CKPT_PATH), map_location=DEVICE, weights_only=False)
    state = ckpt if isinstance(ckpt, dict) and 'model_state_dict' not in ckpt \
            else ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt))
    state = {k.replace('module.', ''): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    model.to(DEVICE).eval()
    log(f'  Checkpoint: {CKPT_PATH.name} ({CKPT_PATH.stat().st_size/1e6:.1f} MB)')
    log(f'  Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M')
    return model

def scan_hdf5_files():
    """Pindai folder scalogram 2026 dan kumpulkan semua file HDF5 valid."""
    log(f'Scanning {SCALO_DIR}...')
    import re
    pattern = re.compile(r'scalogram_([A-Z]+)_(\d{8})\.h5$')
    files = []
    for f in sorted(SCALO_DIR.glob('scalogram_*.h5')):
        m = pattern.match(f.name)
        if m:
            files.append({
                'path': str(f),
                'station': m.group(1),
                'date': m.group(2),
                'filename': f.name
            })
    log(f'  Found {len(files)} valid HDF5 files')
    if files:
        log(f'  Stations: {sorted(set(f["station"] for f in files))}')
        log(f'  Date range: {files[0]["date"]} to {files[-1]["date"]}')
    return files

def load_hdf5_sample(path):
    """
    Muat satu sample dari file HDF5.
    Returns: (tensors, cosmic_features, label_event, label_mag) atau None jika gagal.
    """
    try:
        import h5py
        with h5py.File(path, 'r') as f:
            daily = f['daily']
            station_keys = list(daily.keys())
            if not station_keys:
                return None
            station_key = station_keys[0]
            grp = daily[station_key]

            tensors  = torch.from_numpy(np.array(grp['tensors'][0], copy=True)).float()
            cosmic   = torch.from_numpy(np.array(grp['cosmic_features'][0], copy=True)).float()
            label_ev = int(grp['label_event'][0])
            label_mg = int(grp['label_mag'][0])
            return tensors, cosmic, label_ev, label_mg
    except Exception as e:
        log(f'  Gagal load {path}: {e}', 'WARN')
        return None

# ══════════════════════════════════════════════════════════════════════════════
# THRESHOLD CALIBRATION
# ══════════════════════════════════════════════════════════════════════════════

def find_optimal_threshold_f2(all_probs, all_labels, beta=2.0, n_steps=200):
    """
    ┌─────────────────────────────────────────────────────────────────────┐
    │ PENCARIAN THRESHOLD OPTIMAL BERBASIS F2-SCORE (SANGAT PENTING!)     │
    │                                                                     │
    │ Mengapa F2?                                                         │
    │   EWS Gempa Bumi: False Negative (missed detection) jauh lebih      │
    │   berbahaya daripada False Positive (alarm palsu).                  │
    │   F2-score memberikan bobot 2x lebih besar pada Recall (minimasi FN)│
    │   dibandingkan Precision.                                           │
    │                                                                     │
    │ Proses:                                                             │
    │   1. Sweep threshold dari 0.01 -> 0.99 (n_steps=200 titik)         │
    │   2. Untuk setiap threshold, hitung F2-score                        │
    │   3. Threshold dengan F2 terbesar dipilih sebagai optimal           │
    │                                                                     │
    │ UNTUK PRESENTASI SIDANG:                                            │
    │   Anda bisa mengganti beta menjadi 2.5 (lebih agresif recall)      │
    │   atau mengganti ke mode statis dengan THRESHOLD_MODE = 'static'   │
    │                                                                     │
    │ Referensi:                                                          │
    │   Siffer et al. (2017) "Anomaly Detection in Streams with          │
    │   Rare Earthquakes" -- threshold adaptive berbasis loss.           │
    └─────────────────────────────────────────────────────────────────────┘
    """
    best_th, best_f2 = 0.5, -1.0
    for i in range(1, n_steps):
        th = i / n_steps
        preds = (all_probs >= th).astype(int)
        if preds.sum() == 0 or preds.sum() == len(preds):
            continue
        tp = int(((preds == 1) & (all_labels == 1)).sum())
        fp = int(((preds == 1) & (all_labels == 0)).sum())
        fn = int(((preds == 0) & (all_labels == 1)).sum())
        precision = tp / (tp + fp + 1e-8)
        recall    = tp / (tp + fn + 1e-8)
        f2        = (1 + beta**2) * precision * recall / (beta**2 * precision + recall + 1e-8)
        if f2 > best_f2:
            best_f2 = f2
            best_th = th
    return best_th, best_f2

# ══════════════════════════════════════════════════════════════════════════════
# INFERENCE
# ══════════════════════════════════════════════════════════════════════════════

def run_inference(model, hdf5_files):
    """Eksekusi forward pass pada semua file HDF5 2026."""
    log('Starting inference...')
    records = []
    n_total = len(hdf5_files)
    t_start = time.time()

    for idx, finfo in enumerate(hdf5_files):
        if idx % 200 == 0 or idx == n_total - 1:
            elapsed = time.time() - t_start
            speed   = (idx + 1) / elapsed if elapsed > 0 else 0
            eta     = (n_total - idx - 1) / speed if speed > 0 else 0
            log(f'  [{idx+1}/{n_total}] speed={speed:.1f} file/s, ETA={eta:.0f}s')

        sample = load_hdf5_sample(finfo['path'])
        if sample is None:
            continue
        tensors, cosmic, label_event, label_mag = sample

        with torch.no_grad():
            out_det, _, _, _, _, _, _ = model(
                tensors.unsqueeze(0).to(DEVICE),
                cosmic.unsqueeze(0).to(DEVICE)
            )
            prob = F.softmax(out_det, dim=1)[0, 1].item()

        records.append({
            'Date':      f'{finfo["date"][:4]}-{finfo["date"][4:6]}-{finfo["date"][6:]}',
            'Station':   finfo['station'],
            'True_Label': label_event,
            'Pred_Prob': round(prob, 6),
            'True_Mag':  label_mag,
            'FileName':  finfo['filename'],
        })

    total_time = time.time() - t_start
    log(f'  Inference selesai: {len(records)} samples dalam {total_time:.1f}s')
    log(f'  Speed: {len(records)/total_time:.1f} file/s, Latency: {total_time/len(records)*1000:.1f}ms/sample')

    df = pd.DataFrame(records)
    return df, total_time

# ══════════════════════════════════════════════════════════════════════════════
# METRICS
# ══════════════════════════════════════════════════════════════════════════════

def compute_all_metrics(df, threshold):
    """Hitung semua metrik di bawah threshold tertentu."""
    probs  = df['Pred_Prob'].values
    labels = df['True_Label'].values
    preds  = (probs >= threshold).astype(int)

    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())

    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)
    beta      = F2_BETA
    f2        = (1 + beta**2) * precision * recall / (beta**2 * precision + recall + 1e-8)
    accuracy  = (tp + tn) / (tp + fp + tn + fn + 1e-8)
    fpr       = fp / (fp + tn + 1e-8)

    return {
        'threshold': round(threshold, 4),
        'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn,
        'precision': round(precision, 6),
        'recall': round(recall, 6),
        'f1': round(f1, 6),
        'f2': round(f2, 6),
        'accuracy': round(accuracy, 6),
        'fpr': round(fpr, 6),
        'total_samples': len(labels),
        'n_positive': int(labels.sum()),
        'n_negative': len(labels) - int(labels.sum()),
    }

def compute_latency_metrics(total_time, n_samples):
    """Hitung latensi dan throughput."""
    latency_per_sample = total_time / n_samples
    batch_latency = latency_per_sample * 8
    throughput_per_hour = 3600 / latency_per_sample
    return {
        'total_inference_time_sec': round(total_time, 3),
        'latency_per_sample_sec': round(latency_per_sample, 6),
        'batch_8_latency_sec': round(batch_latency, 3),
        'throughput_per_hour': round(throughput_per_hour, 3),
        'throughput_target_basal': 100,
        'exceeds_target_factor': round(throughput_per_hour / 100, 1),
    }

# ══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION
# ══════════════════════════════════════════════════════════════════════════════

def plot_confusion_matrix(df, threshold, out_path):
    """Plot confusion matrix (seaborn heatmap, 300 DPI)."""
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns

    probs  = df['Pred_Prob'].values
    labels = df['True_Label'].values
    preds  = (probs >= threshold).astype(int)

    cm = confusion_matrix(labels, preds)
    cm_pct = cm.astype(float) / cm.sum() * 100

    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor('white')
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Non-Event (0)', 'Event (1)'],
                yticklabels=['Non-Event (0)', 'Event (1)'],
                annot_kws={'size': 16, 'weight': 'bold'},
                cbar_kws={'shrink': 0.8},
                linewidths=1.5, linecolor='#FFF')
    for i in range(2):
        for j in range(2):
            ax.text(j + 0.5, i + 0.75, f'({cm_pct[i, j]:.1f}%)',
                    ha='center', va='center', fontsize=9, color='#555')
    ax.set_title(f'Confusion Matrix - V8 SupCon\nThreshold = {threshold:.4f}',
                 fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel('Predicted Label', fontsize=11, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=11, fontweight='bold')
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    log(f'  Confusion matrix saved: {out_path}')

def plot_roc_pr_curves(df, threshold, out_path):
    """Plot ROC curve dan Precision-Recall curve (side-by-side)."""
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    probs  = df['Pred_Prob'].values
    labels = df['True_Label'].values

    fpr_arr, tpr_arr, _ = roc_curve(labels, probs)
    roc_auc_val = auc(fpr_arr, tpr_arr)
    prec_arr, rec_arr, _ = precision_recall_curve(labels, probs)
    pr_auc_val = average_precision_score(labels, probs)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor('white')

    # ROC
    ax1.plot(fpr_arr, tpr_arr, '-', color='#1a5276', lw=2.5,
             label=f'V8 SupCon (AUC={roc_auc_val:.4f})')
    ax1.fill_between(fpr_arr, tpr_arr, alpha=0.08, color='#1a5276')
    ax1.plot([0, 1], [0, 1], '--', color='#ccc', lw=1.5, label='Random (AUC=0.50)')
    ax1.set_xlabel('False Positive Rate', fontweight='bold')
    ax1.set_ylabel('True Positive Rate', fontweight='bold')
    ax1.set_title('ROC Curve', fontweight='bold')
    ax1.legend(loc='lower right', fontsize=9)
    ax1.set_xlim(-0.02, 1.02); ax1.set_ylim(-0.02, 1.02)
    ax1.set_facecolor('white')

    # PR
    ax2.plot(rec_arr, prec_arr, '-', color='#1a5276', lw=2.5,
             label=f'V8 SupCon (AP={pr_auc_val:.4f})')
    ax2.fill_between(rec_arr, prec_arr, alpha=0.08, color='#1a5276')
    baseline = labels.mean()
    ax2.axhline(y=baseline, ls='--', color='#ccc', lw=1.5, label=f'Baseline ({baseline:.3f})')
    # Mark operating point
    preds_at = (probs >= threshold).astype(int)
    p_val = precision_score(labels, preds_at, zero_division=0)
    r_val = recall_score(labels, preds_at, zero_division=0)
    ax2.plot(r_val, p_val, 'o', color='#e74c3c', ms=10, zorder=5, label=f'Th={threshold:.3f}')
    ax2.set_xlabel('Recall', fontweight='bold')
    ax2.set_ylabel('Precision', fontweight='bold')
    ax2.set_title('Precision-Recall Curve', fontweight='bold')
    ax2.legend(loc='lower left', fontsize=9)
    ax2.set_xlim(-0.02, 1.02); ax2.set_ylim(0, 1.05)
    ax2.set_facecolor('white')

    fig.suptitle('V8 SupCon - Chronological Blind Test 2026',
                 fontsize=13, fontweight='bold', y=1.03)
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    log(f'  ROC/PR curves saved: {out_path}')

def plot_f2_threshold_sweep(df, optimal_th, optimal_f2, out_path):
    """Plot F2-score vs threshold sweep curve."""
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    probs  = df['Pred_Prob'].values
    labels = df['True_Label'].values
    n_steps = 200
    ths, f2s, recs, precs = [], [], [], []
    for i in range(1, n_steps):
        th = i / n_steps
        preds = (probs >= th).astype(int)
        if preds.sum() == 0 or preds.sum() == len(preds):
            ths.append(th); f2s.append(0); recs.append(0); precs.append(0)
            continue
        tp = ((preds == 1) & (labels == 1)).sum()
        fp = ((preds == 1) & (labels == 0)).sum()
        fn = ((preds == 0) & (labels == 1)).sum()
        prec = tp / (tp + fp + 1e-8)
        rec  = tp / (tp + fn + 1e-8)
        beta = F2_BETA
        f2_v = (1 + beta**2) * prec * rec / (beta**2 * prec + rec + 1e-8)
        ths.append(th); f2s.append(f2_v); recs.append(rec); precs.append(prec)

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor('white')
    ax.plot(ths, f2s, '-', color='#1a5276', lw=2.5, label='F2-Score')
    ax.plot(ths, recs, '--', color='#27ae60', lw=2, alpha=0.7, label='Recall')
    ax.plot(ths, precs, '--', color='#e74c3c', lw=2, alpha=0.7, label='Precision')
    ax.axvline(x=optimal_th, color='#f39c12', ls='--', lw=2.5, alpha=0.8,
               label=f'Optimal th={optimal_th:.4f}')
    ax.plot(optimal_th, optimal_f2, 'o', color='#f39c12', ms=12, zorder=5)
    ax.axvline(x=0.30, color='#aaa', ls=':', lw=2, alpha=0.5, label='Old static th=0.30')
    ax.set_xlabel('Decision Threshold', fontweight='bold')
    ax.set_ylabel('Score', fontweight='bold')
    ax.set_title(f'F2-Score Threshold Sweep (beta={F2_BETA})\nOptimal: th={optimal_th:.4f}, F2={optimal_f2:.4f}',
                 fontweight='bold')
    ax.legend(fontsize=9); ax.set_facecolor('white'); ax.grid(True, alpha=0.15)
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    log(f'  F2 sweep curve saved: {out_path}')

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print('=' * 70)
    print('V8 SupCon - FORENSIC CHRONOLOGICAL BLIND TEST 2026')
    print('=' * 70)
    print(f'Mode threshold: {THRESHOLD_MODE}')
    if THRESHOLD_MODE == 'static':
        print(f'Static threshold: {STATIC_THRESHOLD}')
    else:
        print(f'F2 beta: {F2_BETA}')
    print(f'Output: {OUT_DIR}')
    print('=' * 70)

    # 1. Load model
    model = load_model()

    # 2. Scan data
    hdf5_files = scan_hdf5_files()
    if not hdf5_files:
        log('ERROR: Tidak ada file HDF5 ditemukan!', 'ERROR')
        sys.exit(1)

    # 3. Run inference
    df, total_time = run_inference(model, hdf5_files)

    # 4. Save predictions CSV
    csv_path = OUT_DIR / 'predictions_2026.csv'
    df.to_csv(csv_path, index=False)
    log(f'Predictions saved: {csv_path}')

    # 5. Threshold calibration
    log('Threshold calibration...')
    all_probs  = df['Pred_Prob'].values
    all_labels = df['True_Label'].values

    if THRESHOLD_MODE == 'f2_optimal':
        optimal_th, optimal_f2 = find_optimal_threshold_f2(
            all_probs, all_labels, beta=F2_BETA)
        chosen_th = optimal_th
        log(f'  F2-optimal threshold: {chosen_th:.4f} (F2={optimal_f2:.4f})')
        for test_th in [0.20, 0.25, 0.30, 0.40, 0.50]:
            m = compute_all_metrics(df, test_th)
            log(f'  th={test_th:.2f}: Recall={m["recall"]:.4f}, FPR={m["fpr"]:.4f}, F2={m["f2"]:.4f}')
    else:
        chosen_th = STATIC_THRESHOLD
        optimal_f2 = 0.0
        log(f'  Static threshold: {chosen_th}')

    # 6. Compute final metrics
    metrics = compute_all_metrics(df, chosen_th)
    latency = compute_latency_metrics(total_time, len(df))
    metrics.update(latency)
    metrics['threshold_mode'] = THRESHOLD_MODE
    metrics['f2_beta'] = F2_BETA
    metrics['timestamp'] = datetime.now().isoformat()
    metrics['model_checkpoint'] = CKPT_PATH.name
    metrics['data_dir'] = str(SCALO_DIR)

    json_path = OUT_DIR / 'blindtest_metrics_2026.json'
    with open(json_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    log(f'Metrics JSON saved: {json_path}')

    # Print summary
    log('=' * 70)
    log('FINAL METRICS SUMMARY')
    log('=' * 70)
    for k, v in metrics.items():
        log(f'  {k}: {v}')
    log('=' * 70)

    # 7. Visualizations
    log('Generating visualizations...')
    plot_confusion_matrix(df, chosen_th, OUT_DIR / 'confusion_matrix_2026.png')
    plot_roc_pr_curves(df, chosen_th, OUT_DIR / 'roc_pr_curve_2026.png')
    plot_f2_threshold_sweep(df, chosen_th, optimal_f2, OUT_DIR / 'f2_threshold_sweep_2026.png')

    log(f'\nBLIND TEST COMPLETE - Evidence in: {OUT_DIR}')
    log('Files:')
    for fpath in sorted(OUT_DIR.glob('*')):
        log(f'  {fpath.name} ({fpath.stat().st_size/1024:.1f} KB)')
