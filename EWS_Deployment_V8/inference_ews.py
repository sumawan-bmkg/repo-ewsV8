#!/usr/bin/env python3
"""
======================================================================
  inference_ews.py — Skrip Inferensi Produksi EWS Gempa Bumi
======================================================================

Penulis       : Tim Disertasi EWS Gempa Bumi
Model         : V8 SupCon (EfficientNet-B1 + Supervised Contrastive Loss)
Mitigasi      : Temperature Scaling (T = 6.3) — Fase 1 Post-Processing
Tanggal Build : Auto-generated oleh build_deployment.py

Cara Pakai:
  python inference_ews.py --input <path_to_scalogram.h5>
  python inference_ews.py --input <path_to_scalogram.h5> --threshold 0.5
  python inference_ews.py --input <folder_path> --batch

Output:
  - Gempa (1): Alarm aktif jika probabilitas >= threshold
  - Noise  (0): Aman jika probabilitas < threshold

Catatan untuk Operator BMKG:
  Threshold default sudah disetel pada 0.5 karena probabilitas output
  sudah terkalibrasi oleh Temperature Scaling (T = 6.3). Tidak perlu
  mengubah threshold kecuali ada instruksi dari tim peneliti.
======================================================================
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.models as models
import h5py


# =====================================================================
# KONFIGURASI PRODUSI — JANGAN DIUBAH TANPA INSTRUKSI PENELITI
# =====================================================================

# Temperatur kalibrasi (Fase 1 Post-Processing)
TEMPERATURE = 6.3

# Threshold klasifikasi (sudah terkalibrasi)
DEFAULT_THRESHOLD = 0.5

# Label kelas
LABELS = {0: "NOISE (Aman)", 1: "GEMPA (Alarm!)"}


# =====================================================================
# FUNGSI: Memuat Model EfficientNet-B1
# =====================================================================

def load_model(weights_path: str, device: str = "cpu") -> torch.nn.Module:
    """
    Memuat arsitektur EfficientNet-B1 dan memasukkan bobot dari file .pth.

    Parameter:
        weights_path : str — Path ke file bobot model (.pth)
        device       : str — Perangkat komputasi ('cpu' atau 'cuda')

    Mengembalikan:
        model : torch.nn.Module — Model dalam mode evaluasi (inference)
    """
    # Bangun arsitektur EfficientNet-B1 tanpa bobot pretrained ImageNet
    model = models.efficientnet_b1(pretrained=False)
    
    # Sesuaikan classifier head untuk binary classification
    # EfficientNet-B1 default: classifier = Linear(1280, 1000)
    # Kita ganti: classifier = Linear(1280, 1) untuk output logit tunggal
    num_features = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(num_features, 1)
    
    # Muat bobot model dari file checkpoint
    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    
    # Handle dua kemungkinan format checkpoint:
    # 1. State dict langsung
    # 2. Dictionary dengan kunci 'model_state_dict'
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    # Pindahkan ke perangkat yang ditentukan dan set mode evaluasi
    model.to(device)
    model.eval()
    
    print(f"[OK] Model berhasil dimuat dari: {weights_path}")
    print(f"    Perangkat: {device}")
    print(f"    Jumlah parameter: {sum(p.numel() for p in model.parameters()):,}")
    
    return model


# =====================================================================
# FUNGSI: Prediksi dengan Temperature Scaling
# =====================================================================

def predict(model: torch.nn.Module, input_tensor: torch.Tensor,
            threshold: float = DEFAULT_THRESHOLD) -> dict:
    """
    Melakukan prediksi pada satu sampel input dengan Temperature Scaling.

    ALGORITMA:
      1. Forward pass menghasilkan logit mentah (raw logit).
      2. Logit dibagi oleh Temperatur (T = 6.3) untuk "melembutkan"
         distribusi probabilitas yang terkompresi oleh SupCon Loss.
      3. Hasil dibagi Temperatur dimasukkan ke fungsi Sigmoid untuk
         menghasilkan probabilitas terkalibrasi.
      4. Probabilitas dibandingkan dengan threshold untuk menentukan alarm.

    Parameter:
        model        : torch.nn.Module — Model EWS yang sudah dimuat
        input_tensor : torch.Tensor    — Tensor input (1, 3, 224, 224)
        threshold    : float           — Ambang batas klasifikasi (default: 0.5)

    Mengembalikan:
        dict dengan kunci:
            - probability  : float — Probabilitas terkalibrasi [0, 1]
            - raw_logit    : float — Logit mentah sebelum scaling
            - label        : str   — Label kelas (GEMPA atau NOISE)
            - label_int    : int   — Label numerik (1 atau 0)
            - alarm        : bool  — True jika alarm aktif
    """
    model.eval()
    
    with torch.no_grad():
        # === LANGKAH 1: Forward pass -> dapatkan logit mentah ===
        raw_logit = model(input_tensor)  # Output: tensor dengan 1 logit
        
        # Pastikan output berbentuk scalar
        if raw_logit.dim() > 1:
            raw_logit = raw_logit.squeeze()
        
        raw_logit_val = raw_logit.item()
        
        # === LANGKAH 2: Temperature Scaling ===
        # Bagi logit dengan T untuk "melembutkan" distribusi
        # Tanpa scaling, model memancarkan probabilitas ~0.001-0.014
        # (terlalu konservatif akibat SupCon Loss)
        calibrated_logit = raw_logit_val / TEMPERATURE
        
        # === LANGKAH 3: Sigmoid -> probabilitas terkalibrasi ===
        calibrated_prob = 1.0 / (1.0 + np.exp(-calibrated_logit))
        
        # === LANGKAH 4: Klasifikasi berdasarkan threshold ===
        label_int = 1 if calibrated_prob >= threshold else 0
        label = LABELS[label_int]
        alarm = (label_int == 1)
    
    return {
        "probability": round(calibrated_prob, 6),
        "raw_logit": round(raw_logit_val, 6),
        "label": label,
        "label_int": label_int,
        "alarm": alarm
    }


# =====================================================================
# FUNGSI: Pra-pemrosesan Input (Scalogram)
# =====================================================================

def preprocess_scalogram(h5_path: str, target_size: int = 224) -> torch.Tensor:
    """
    Membaca file scalogram (.h5) dan mengubahnya menjadi tensor PyTorch.

    Parameter:
        h5_path     : str — Path ke file scalogram HDF5
        target_size : int — Target ukuran spasial (default: 224x224)

    Mengembalikan:
        tensor : torch.Tensor — Tensor berbentuk (1, 3, 224, 224)
    """
    with h5py.File(h5_path, 'r') as f:
        # Ambil data scalogram dari file HDF5
        data_key = list(f.keys())[0]
        scalogram = np.array(f[data_key])
    
    # Pastikan berbentuk 3 channel (RGB)
    if scalogram.ndim == 2:
        scalogram = np.stack([scalogram] * 3, axis=0)
    elif scalogram.ndim == 3:
        if scalogram.shape[0] not in (1, 3):
            scalogram = np.transpose(scalogram, (2, 0, 1))
        if scalogram.shape[0] == 1:
            scalogram = np.repeat(scalogram, 3, axis=0)
    
    # Normalisasi ke rentang [0, 1]
    scalogram = scalogram.astype(np.float32)
    if scalogram.max() > 1.0:
        scalogram = scalogram / 255.0
    
    # Resize jika diperlukan
    if scalogram.shape[1] != target_size or scalogram.shape[2] != target_size:
        from scipy.ndimage import zoom
        factors = (1, target_size / scalogram.shape[1], target_size / scalogram.shape[2])
        scalogram = zoom(scalogram, factors, order=1)
    
    # Konversi ke tensor dan tambah dimensi batch
    tensor = torch.tensor(scalogram, dtype=torch.float32).unsqueeze(0)
    
    return tensor


# =====================================================================
# FUNGSI UTAMA: CLI Entry Point
# =====================================================================

def main():
    """
    Entry point utama untuk inferensi EWS produksi.
    
    Mendukung dua mode:
      1. Satu file: python inference_ews.py --input scalogram.h5
      2. Batch folder: python inference_ews.py --input folder/ --batch
    """
    parser = argparse.ArgumentParser(
        description="Sistem Peringatan Dini Gempa Bumi — Inferensi Produksi",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Contoh:
  python inference_ews.py --input scalogram_ALR_20260101.h5
  python inference_ews.py --input scalogram_ALR_20260101.h5 --threshold 0.3
  python inference_ews.py --input ./scalograms/ --batch
        """
    )
    parser.add_argument("--input", type=str, required=True,
                        help="Path ke file scalogram (.h5) atau folder")
    parser.add_argument("--weights", type=str,
                        default=str(Path(__file__).parent / "v3_v8_conv_fpr_best_weights.pth"),
                        help="Path ke file bobot model (.pth)")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help=f"Ambang batas klasifikasi (default: {DEFAULT_THRESHOLD})")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Perangkat komputasi: 'cpu' atau 'cuda'")
    parser.add_argument("--batch", action="store_true",
                        help="Mode batch: proses semua .h5 dalam folder")
    
    args = parser.parse_args()
    
    # Header
    print("=" * 60)
    print("  EWS GEMPA BUMI — INFERENSI PRODUKSI")
    print("  Model: V8 SupCon (EfficientNet-B1)")
    print(f"  Temperatur Kalibrasi: T = {TEMPERATURE}")
    print(f"  Threshold: {args.threshold}")
    print(f"  Perangkat: {args.device}")
    print("=" * 60)
    
    # Muat model
    model = load_model(args.weights, device=args.device)
    
    # Kumpulkan file yang akan diproses
    input_path = Path(args.input)
    if args.batch:
        if not input_path.is_dir():
            print(f"[ERROR] '{args.input}' bukan direktori untuk mode batch.")
            sys.exit(1)
        files = sorted(input_path.glob("*.h5"))
        if not files:
            print(f"[WARNING] Tidak ada file .h5 ditemukan di '{args.input}'")
            sys.exit(0)
        print(f"\n[INFO] Ditemukan {len(files)} file scalogram untuk diproses.\n")
    else:
        if not input_path.exists():
            print(f"[ERROR] File tidak ditemukan: {args.input}")
            sys.exit(1)
        files = [input_path]
    
    # Proses setiap file
    alarm_count = 0
    safe_count = 0
    
    for fpath in files:
        print(f"[..] Memproses: {fpath.name}")
        
        try:
            input_tensor = preprocess_scalogram(str(fpath))
            input_tensor = input_tensor.to(args.device)
            
            result = predict(model, input_tensor, threshold=args.threshold)
            
            status_icon = "[ALARM]" if result["alarm"] else "[AMAN]"
            print(f"    {status_icon} Hasil  : {result['label']}")
            print(f"       Prob   : {result['probability']:.6f}")
            print(f"       Logit  : {result['raw_logit']:.6f}")
            print(f"       Alarm  : {'AKTIF' if result['alarm'] else 'TIDAK AKTIF'}")
            print()
            
            if result["alarm"]:
                alarm_count += 1
            else:
                safe_count += 1
                
        except Exception as e:
            print(f"    [ERROR] Gagal memproses: {e}")
            print()
    
    # Ringkasan untuk mode batch
    if args.batch:
        print("-" * 60)
        print(f"  RINGKASAN: {alarm_count} ALARM | {safe_count} AMAN")
        print(f"  Total diproses: {len(files)} file")
        print("-" * 60)


if __name__ == "__main__":
    main()
