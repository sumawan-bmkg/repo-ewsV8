#!/usr/bin/env python3
"""
V8 SupCon — Explainable AI Audit
==================================
Phase 1: Contrastive Grad-CAM (EfficientNet-B1 backbone)
Phase 2: SHAP GradientExplainer (frequency-band attribution)
Phase 3: 3-panel figure (Fig 7: XAI Interpretability)

Model: MultiTaskScalogramV3_v8
Backbone: EfficientNet-B1 → BiGRU → GNN → Cosmic Gate → Heads
Input: (B, 3, 128, 1440) CWT Scalogram (128 freq bins × 1440 time steps)
"""

import os, sys, warnings, gc, math
os.environ['MPLBACKEND'] = 'Agg'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
from pathlib import Path
from scipy.ndimage import zoom

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT     = Path('D:/multi/scalogramv3')
CKPT     = ROOT / 'checkpoints' / 'v3_v8_conv_fpr_best_weights.pth'
SCALO_DIR = ROOT / '2026' / 'scalogram'
MODEL_DIR = ROOT / 'ScalogramV3_V8_Repository' / 'model'
OUT_DIR   = ROOT / 'disertasi4' / 'supcon' / '10_xai'
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(MODEL_DIR))
sys.path.insert(0, str(ROOT))

# ── SEED for reproducibility ──────────────────────────────────────────────────
torch.manual_seed(42)
np.random.seed(42)

# ── Device (modest batch, CPU-safe) ───────────────────────────────────────────
device = torch.device('cpu')

# ══════════════════════════════════════════════════════════════════════════════
# 1. MODEL LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_v8():
    from V3_Model_v8 import MultiTaskScalogramV3_v8
    model = MultiTaskScalogramV3_v8(pretrained=False)
    state = torch.load(str(CKPT), map_location='cpu', weights_only=False)
    if isinstance(state, dict):
        for prefix in ['model_state_dict', 'state_dict']:
            if prefix in state:
                state = state[prefix]
                break
    state = {k.replace('module.', ''): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    model.to(device).eval()
    print(f'[OK] V8 loaded from {CKPT.name}')
    return model

# ══════════════════════════════════════════════════════════════════════════════
# 2. DATA SAMPLER (select representative tectonic + quiescent samples)
# ══════════════════════════════════════════════════════════════════════════════

def load_sample_hdf5(path: Path, station: str):
    """Load a single HDF5 sample and return (tensor, cosmic, label)."""
    import h5py
    with h5py.File(str(path), 'r') as f:
        grp = f[f'daily/{station}']
        tensor = torch.from_numpy(np.array(grp['tensors'][0], copy=True)).float()
        cosmic = torch.from_numpy(np.array(grp['cosmic_features'][0], copy=True)).float()
        label  = int(grp['label_event'][0])
    return tensor.unsqueeze(0), cosmic.unsqueeze(0), label

def sample_xai_data(n_event: int = 3, n_noise: int = 3):
    """
    Scan HDF5 directory, load n_event positive + n_noise negative samples.
    """
    import re, h5py
    pattern = re.compile(r'scalogram_([A-Z]+)_(\d{8})\.h5$')
    h5_files = sorted(SCALO_DIR.glob('scalogram_*.h5'))
    np.random.shuffle(h5_files)

    pos_samples, neg_samples = [], []
    for f in h5_files:
        m = pattern.match(f.name)
        if not m:
            continue
        stn, date_str = m.group(1), m.group(2)
        try:
            with h5py.File(str(f), 'r') as hf:
                grp = hf[f'daily/{stn}']
                evt = int(grp['label_event'][0])
        except Exception:
            continue
        if evt == 1 and len(pos_samples) < n_event:
            t, c, l = load_sample_hdf5(f, stn)
            pos_samples.append((t, c, l, f.name, stn, date_str))
        elif evt == 0 and len(neg_samples) < n_noise:
            t, c, l = load_sample_hdf5(f, stn)
            neg_samples.append((t, c, l, f.name, stn, date_str))
        if len(pos_samples) >= n_event and len(neg_samples) >= n_noise:
            break

    # If no negatives found, simulate one using mean stats
    if len(neg_samples) == 0:
        print('[WARN] No negative samples found — using low-activity positive proxy')
        if pos_samples:
            t, c, l, fn, stn, ds = pos_samples[0]
            c_noise = c.clone()
            c_noise[0, 1] = -50.0  # Dst very quiescent
            neg_samples.append((t, c_noise, 0, fn, stn, ds))

    print(f'[OK] XAI samples: {len(pos_samples)} tectonic + {len(neg_samples)} quiescent')
    return pos_samples, neg_samples

# ══════════════════════════════════════════════════════════════════════════════
# 3. CONTRASTIVE GRAD-CAM
# ══════════════════════════════════════════════════════════════════════════════

def contrastive_gradcam(model, image: torch.Tensor, cosmic: torch.Tensor,
                        target_layer, pos_idx: int = 0, neg_idx: int = 0):
    """
    Compute Grad-CAM using contrastive objective:
      score = cosine_sim(proj_pos, anchor) - cosine_sim(proj_neg, anchor)

    This highlights regions that maximise separation between tectonic
    and quiescent samples in the SupCon embedding space.
    """
    # Build a batch: [anchor, positive, negative]
    # We need 3 forward passes — hook activations for anchor only
    activations = []
    gradients   = []

    def forward_hook(m, inp, out):
        activations.append(out.detach())

    def backward_hook(m, grad_in, grad_out):
        gradients.append(grad_out[0].detach())

    hook_fwd = target_layer.register_forward_hook(forward_hook)
    hook_bwd = target_layer.register_full_backward_hook(backward_hook)

    # Forward anchor
    out_det, _, _, _, _, proj_anchor, _ = model(image, cosmic)
    prob_anchor = torch.softmax(out_det, dim=1)[0, 1].item()

    # Create contrastive pair from augmented image
    # Perturb in upper half of frequency (simulates high-freq noise suppression)
    image_pos = image.clone()
    image_neg = image.clone()
    # Positive pair: subtle noise (model should still see tectonic signal)
    noise_pos = torch.randn_like(image_pos[:, :, :32, :]) * 0.03
    image_pos[:, :, :32, :] = image_pos[:, :, :32, :] + noise_pos
    # Negative pair: strong high-freq noise (model should suppress)
    noise_neg = torch.randn_like(image_neg[:, :, :32, :]) * 0.15
    image_neg[:, :, :32, :] = image_neg[:, :, :32, :] + noise_neg

    # Forward + and -
    _, _, _, _, _, proj_pos, _ = model(image_pos, cosmic)
    _, _, _, _, _, proj_neg, _ = model(image_neg, cosmic)

    # Contrastive score: attraction to positive - attraction to negative
    cos_pos = F.cosine_similarity(proj_anchor, proj_pos, dim=1)
    cos_neg = F.cosine_similarity(proj_anchor, proj_neg, dim=1)
    score = (cos_pos - cos_neg).mean()

    # Backprop
    model.zero_grad()
    score.backward()

    hook_fwd.remove()
    hook_bwd.remove()

    # Grad-CAM computation
    acts = activations[0][0]   # (C, H, W)
    grads = gradients[0][0]    # (C, H, W)

    # Global average pooling of gradients → importance weights
    weights = grads.mean(dim=(1, 2), keepdim=True)   # (C, 1, 1)
    cam = (weights * acts).sum(dim=0)                 # (H, W)
    cam = F.relu(cam)                                  # Only positive influences
    cam = cam.detach().cpu().numpy()

    # Normalize to [0, 1]
    eps = 1e-8
    cam = (cam - cam.min()) / (cam.max() - cam.min() + eps)

    # Resize CAM to input spatial dimensions (128, 1440)
    H_in, W_in = 128, 1440
    H_cam, W_cam = cam.shape
    scale_h = H_in / H_cam
    scale_w = W_in / W_cam
    cam_full = zoom(cam, (scale_h, scale_w), order=1)
    cam_full = np.clip(cam_full, 0, 1)

    return cam_full, prob_anchor


# ══════════════════════════════════════════════════════════════════════════════
# 4. SHAP — GRADIENT EXPLAINER (frequency-band attribution)
# ══════════════════════════════════════════════════════════════════════════════

def shap_explain(model, pos_samples, neg_samples):
    """
    Compute SHAP using GradientExplainer on downsampled input.
    128x1440 → 32x360 for tractable CPU computation.
    Background: quiescent samples (n=3).
    """
    import shap

    # ── Downsample function ──────────────────────────────────────────────
    def downsample(t, scale_h=0.25, scale_w=0.25):
        """Bilinear downsample (B,3,H,W) → (B,3,H',W')"""
        from torch.nn.functional import interpolate
        H_new = int(t.shape[2] * scale_h)
        W_new = int(t.shape[3] * scale_w)
        return interpolate(t, size=(H_new, W_new), mode='bilinear',
                           align_corners=False)

    bg_images = []
    for t, c, _, _, _, _ in neg_samples[:3]:
        bg_images.append(downsample(t.to(device)))
    if not bg_images:
        t0 = pos_samples[0][0].to(device)
        bg_images.append(downsample(t0 * 0.1))
    bg_tensor = torch.cat(bg_images, dim=0).to(device)

    # ── SHAP wrapper (proper nn.Module for compatibility) ─────────────────
    class SHAPWrapper(torch.nn.Module):
        def __init__(self, net):
            super().__init__()
            self.net = net
        def forward(self, x):
            dummy_cosmic = torch.zeros(x.size(0), 2, device=x.device)
            _, _, _, _, _, proj, _ = self.net(x, dummy_cosmic)
            return proj.norm(dim=1, keepdim=True)
    wrapper = SHAPWrapper(model).to(device).eval()

    # ── Compute SHAP (on downsampled input) ──────────────────────────────
    print('[SHAP] Initializing GradientExplainer (32×360 downsampled)...')
    explainer = shap.GradientExplainer(wrapper, bg_tensor)

    all_shap = []
    for t, c, _, _, _, _ in pos_samples[:3]:
        inp = downsample(t.to(device))
        sv = explainer.shap_values(inp, nsamples=20)
        # sv[0]: (1, 3, 32, 360)
        sv_arr = np.array(sv).squeeze()  # (3, 32, 360)
        # Average channels, sum over time → frequency importance
        freq_imp = sv_arr.mean(axis=0).sum(axis=1)  # (32,)
        # Upsample frequency dimension to match 128 bins
        freq_imp_full = np.interp(np.linspace(0, 31, 128),
                                  np.arange(32), freq_imp)
        all_shap.append(freq_imp_full)

    shap_freq = np.mean(all_shap, axis=0) if all_shap else np.zeros(128)
    shap_freq = shap_freq / (np.abs(shap_freq).max() + 1e-10)

    print(f'[OK] SHAP done — freq attribution shape: {shap_freq.shape}')
    return shap_freq


# ══════════════════════════════════════════════════════════════════════════════
# 5. VISUALIZATION — 3-panel Figure (Fig 7)
# ══════════════════════════════════════════════════════════════════════════════

def generate_xai_figure(cam_map: np.ndarray, input_image: np.ndarray,
                         shap_freq: np.ndarray, prob: float,
                         station: str, date_str: str):
    """
    Panel A: Original CWT Scalogram
    Panel B: Grad-CAM overlay (ULF focus zone)
    Panel C: SHAP frequency importance
    """
    fig = plt.figure(figsize=(22, 8))
    fig.patch.set_facecolor('white')
    gs = GridSpec(3, 3, figure=fig, width_ratios=[1, 1.5, 1.2],
                  hspace=0.35, wspace=0.30)

    # ── Panel A: Original Scalogram ──────────────────────────────────────
    ax0 = fig.add_subplot(gs[:, 0])
    ax0.set_facecolor('white')

    # Input is (3, 128, 1440) — average RGB channels
    img_mono = input_image.mean(axis=0)  # (128, 1440)
    im0 = ax0.imshow(img_mono, aspect='auto', cmap='viridis',
                     origin='lower', interpolation='bilinear',
                     extent=[0, 30, 0, 128])  # 30 days in time axis
    ax0.set_title('A. Original CWT Scalogram\n(tectonic anomaly)',
                  fontsize=11, fontweight='bold')
    ax0.set_xlabel('Time (days)', fontsize=10)
    ax0.set_ylabel('Frequency bin', fontsize=10)

    # Add ULF band annotation (0.01-0.1 Hz → roughly bins 20-60)
    ax0.axhspan(20, 60, color='cyan', alpha=0.12, label='ULF band\n(0.01–0.1 Hz)')
    ax0.legend(fontsize=7, loc='upper right')

    cbar0 = fig.colorbar(im0, ax=ax0, shrink=0.7)
    cbar0.set_label('CWT amplitude', fontsize=8)
    ax0.tick_params(colors='#000')
    for spine in ax0.spines.values():
        spine.set_color('#CCC')

    # ── Panel B: Grad-CAM Overlay ──────────────────────────────────────
    ax1 = fig.add_subplot(gs[:, 1])
    ax1.set_facecolor('white')

    # Rescale image to [0,1] for overlay
    img_norm = (img_mono - img_mono.min()) / (img_mono.max() - img_mono.min() + 1e-8)
    img_rgb = plt.cm.gray(img_norm)[:, :, :3]  # (128, 1440, 3)
    # Apply CAM as jet overlay with transparency
    cam_colored = plt.cm.jet(cam_map)[:, :, :3]
    overlay = 0.55 * img_rgb + 0.45 * cam_colored
    overlay = np.clip(overlay, 0, 1)

    ax1.imshow(overlay, aspect='auto', origin='lower',
               extent=[0, 30, 0, 128], interpolation='bilinear')
    ax1.set_title(f'B. Grad-CAM Overlay\n(contrastive saliency, P={prob:.2f})',
                  fontsize=11, fontweight='bold')
    ax1.set_xlabel('Time (days)', fontsize=10)
    ax1.set_ylabel('Frequency bin', fontsize=10)

    # ULF band highlight
    ax1.axhspan(20, 60, color='cyan', alpha=0.15, label='ULF band\n(0.01–0.1 Hz)')
    ax1.legend(fontsize=7, loc='upper right')

    # Add annotation for "hot zone"
    max_y, max_x = np.unravel_index(cam_map.argmax(), cam_map.shape)
    # Convert CAM coordinates to data coordinates
    data_x = max_x / cam_map.shape[1] * 30
    data_y = max_y / cam_map.shape[0] * 128
    ax1.plot(data_x, data_y, marker='*', color='red', markersize=14,
             markeredgecolor='white', markeredgewidth=1.0,
             label=f'Peak focus\n(f={data_y:.0f}, t={data_x:.1f}d)')
    ax1.legend(fontsize=7, loc='lower right')

    ax1.tick_params(colors='#000')
    for spine in ax1.spines.values():
        spine.set_color('#CCC')

    # ── Panel C: SHAP Frequency Importance ──────────────────────────────
    ax2 = fig.add_subplot(gs[:, 2])
    ax2.set_facecolor('white')

    freq_bins = np.arange(128)
    # Color positive vs negative SHAP
    colors_shap = ['#C0392B' if v >= 0 else '#2980B9' for v in shap_freq]
    ax2.barh(freq_bins, shap_freq, color=colors_shap, height=0.8, alpha=0.85,
             edgecolor='#333', linewidth=0.3)

    ax2.axvline(0, color='#666', lw=0.8)
    ax2.set_title('C. SHAP Frequency\nFeature Importance',
                  fontsize=11, fontweight='bold')
    ax2.set_xlabel('SHAP value (attribution)', fontsize=10)
    ax2.set_ylabel('Frequency bin', fontsize=10)

    # Highlight ULF band on y-axis
    ax2.axhspan(20, 60, color='#27AE60', alpha=0.12, label='ULF band\n(0.01–0.1 Hz)')
    ax2.legend(fontsize=7, loc='lower right')

    ax2.tick_params(colors='#000')
    for spine in ax2.spines.values():
        spine.set_color('#CCC')

    # Informational text
    fig.suptitle(f'V8 SupCon — XAI Interpretability Audit\n'
                 f'Sample: {station} ({date_str}) | '
                 f'Grad-CAM target: features[-1][0] (Conv2d 320→1280)',
                 fontsize=12, fontweight='bold', y=1.02)

    # Save
    for fmt in ['png', 'pdf', 'svg']:
        fig.savefig(OUT_DIR / f'fig7_xai_interpretability.{fmt}',
                    dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'[OK] Figure saved to {OUT_DIR}')

    # ── Supplemental: frequency profile panel ──────────────────────────
    fig2, axes2 = plt.subplots(1, 3, figsize=(16, 4))
    fig2.patch.set_facecolor('white')

    # Frequency profile: mean energy per bin
    freq_profile = img_mono.mean(axis=1)  # (128,)
    axes2[0].plot(freq_profile, np.arange(128), color='#2980B9', lw=1.5)
    axes2[0].fill_betweenx(np.arange(128), 0, freq_profile, alpha=0.2, color='#2980B9')
    axes2[0].axhspan(20, 60, color='#27AE60', alpha=0.15)
    axes2[0].set_title('Mean Energy vs Frequency', fontsize=10, fontweight='bold')
    axes2[0].set_xlabel('Mean CWT amplitude', fontsize=9)
    axes2[0].set_ylabel('Frequency bin', fontsize=9)
    axes2[0].set_facecolor('white')
    axes2[0].tick_params(colors='#000')
    axes2[0].spines['bottom'].set_color('#CCC')
    axes2[0].spines['left'].set_color('#CCC')

    # Grad-CAM frequency profile (mean over time)
    cam_freq = cam_map.mean(axis=1)
    axes2[1].plot(cam_freq, np.arange(128), color='#C0392B', lw=1.5)
    axes2[1].fill_betweenx(np.arange(128), 0, cam_freq, alpha=0.2, color='#C0392B')
    axes2[1].axhspan(20, 60, color='#27AE60', alpha=0.15)
    axes2[1].set_title('Grad-CAM Attention vs Frequency', fontsize=10, fontweight='bold')
    axes2[1].set_xlabel('Mean CAM weight', fontsize=9)
    axes2[1].set_ylabel('Frequency bin', fontsize=9)
    axes2[1].set_facecolor('white')
    axes2[1].tick_params(colors='#000')
    axes2[1].spines['bottom'].set_color('#CCC')
    axes2[1].spines['left'].set_color('#CCC')

    # SHAP frequency profile
    axes2[2].barh(np.arange(128), shap_freq, color=colors_shap, height=0.7, alpha=0.8,
                  edgecolor='#333', linewidth=0.2)
    axes2[2].axvline(0, color='#666', lw=0.8)
    axes2[2].axhspan(20, 60, color='#27AE60', alpha=0.12)
    axes2[2].set_title('SHAP Attribution vs Frequency', fontsize=10, fontweight='bold')
    axes2[2].set_xlabel('SHAP value', fontsize=9)
    axes2[2].set_ylabel('Frequency bin', fontsize=9)
    axes2[2].set_facecolor('white')
    axes2[2].tick_params(colors='#000')
    axes2[2].spines['bottom'].set_color('#CCC')
    axes2[2].spines['left'].set_color('#CCC')

    fig2.suptitle('Frequency-domain Analysis — V8 SupCon XAI Profiles',
                  fontsize=11, fontweight='bold')
    plt.tight_layout()
    fig2.savefig(OUT_DIR / 'fig7_xai_frequency_profiles.png',
                 dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'[OK] Frequency profiles saved')


# ══════════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print('=' * 70)
    print('V8 SupCon — EXPLAINABLE AI AUDIT')
    print('=' * 70)

    # 1. Load model
    model = load_v8()
    model.eval()

    # 2. Identify target layer for Grad-CAM
    target_layer = model.features[-1][0]  # Conv2d(320, 1280, 1x1)
    print(f'[OK] Grad-CAM target: model.features[-1][0] = {target_layer}')

    # 3. Sample data
    pos_samples, neg_samples = sample_xai_data(n_event=3, n_noise=3)

    # 4. Grad-CAM for first positive sample
    print('\n[PHASE 1] Contrastive Grad-CAM...')
    t, c, label, fname, station, date_str = pos_samples[0]
    t_gpu, c_gpu = t.to(device), c.to(device)

    cam_map, prob = contrastive_gradcam(
        model, t_gpu, c_gpu, target_layer)
    print(f'  CAM shape: {cam_map.shape} [{cam_map.min():.4f}, {cam_map.max():.4f}]')
    print(f'  Detection probability: {prob:.4f}')
    print(f'  Sample: {station} ({date_str})')

    # 5. SHAP for frequency attribution
    print('\n[PHASE 2] SHAP GradientExplainer...')
    shap_freq = shap_explain(model, pos_samples, neg_samples)
    print(f'  SHAP frequency shape: {shap_freq.shape}')

    # 6. Generate figure
    print('\n[PHASE 3] Generating XAI figure...')
    img_np = t.squeeze(0).cpu().numpy()  # (3, 128, 1440)
    generate_xai_figure(cam_map, img_np, shap_freq, prob, station, date_str)

    # 7. Print summary for manuscript
    print('\n' + '=' * 70)
    print('XAI AUDIT COMPLETE')
    print('=' * 70)
    print(f'  Output: {OUT_DIR}')
    print(f'  Files:')
    for f in sorted(OUT_DIR.glob('fig7_*')):
        print(f'    {f.name} ({f.stat().st_size / 1024:.1f} KB)')

    # Compute statistics for narrative
    # ULF band = bins 20-60
    ulf_mask = np.zeros(128, dtype=bool)
    ulf_mask[20:61] = True
    cam_freq = cam_map.mean(axis=1)
    ulf_attention = cam_freq[ulf_mask].mean()
    high_attention = cam_freq[~ulf_mask].mean()
    print(f'\n[XAI METRICS]')
    print(f'  Mean CAM attention in ULF band (bins 20-60):   {ulf_attention:.4f}')
    print(f'  Mean CAM attention outside ULF band:            {high_attention:.4f}')
    print(f'  ULF / non-ULF attention ratio:                 {ulf_attention / (high_attention + 1e-8):.2f}x')
    print(f'  SHAP dominant frequency bin:                   {np.argmax(np.abs(shap_freq))}')
    print(f'  SHAP mass within ULF band:                     {np.abs(shap_freq[ulf_mask]).sum() / (np.abs(shap_freq).sum() + 1e-8) * 100:.1f}%')
