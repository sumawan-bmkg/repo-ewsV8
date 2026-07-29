"""
================================================================================
 FASE 3 — Decoupled Training untuk Pemisahan Representasi & Klasifikasi
 Model: V8 SupCon (EfficientNet-B1 + Supervised Contrastive Learning)

 Metode Decoupled Training membagi pelatihan menjadi dua tahap terpisah:
 1. Stage 1: Melatih representasi fitur yang kokoh (SupCon Loss).
 2. Stage 2: Membekukan (freeze) seluruh encoder/extractor fitur 
    (requires_grad = False), dan hanya melatih (requires_grad = True) 
    linear classifier head dengan klasifikasi standar.
 
 Program ini menunjukkan cara membekukan backbone model V8 SupCon secara 
 selektif dan menyiapkan optimizer Adam khusus untuk Stage 2.

 Lampiran Disertasi Doktoral — Program Studi Teknik Fisika
================================================================================
"""

import sys
import torch
import torch.nn as nn

# Impor model V8 asli dari repositori
sys.path.insert(0, 'D:/multi/scalogramv3/ScalogramV3_V8_Repository/model')
try:
    from V3_Model_v8 import MultiTaskScalogramV3_v8
    MODEL_AVAILABLE = True
except ImportError:
    MODEL_AVAILABLE = False


def freeze_encoder_for_stage2(model: nn.Module) -> nn.Module:
    """
    Membekukan seluruh parameter pengekstraksi fitur (Backbone, GRU, GNN, Cosmic Gating)
    dan hanya membuka gradien (requires_grad = True) pada detection head.
    
    Argumen:
        model (nn.Module): Arsitektur MultiTaskScalogramV3_v8.
        
    Kembalian:
        nn.Module: Model yang siap untuk pelatihan Stage 2.
    """
    print("[INFO] Memulai pembekuan parameter untuk Decoupled Training (Stage 2)...")
    
    # -- Langkah 1: Bekukan seluruh parameter model ----------------------------
    for param in model.parameters():
        param.requires_grad = False
        
    # -- Langkah 2: Buka kembali parameter classification head ------------------
    # head_detection merujuk pada Linear Layer deteksi biner di ujung jaringan
    if hasattr(model, 'head_detection'):
        for param in model.head_detection.parameters():
            param.requires_grad = True
        print("[OK] Gradien pada 'head_detection' berhasil dibuka.")
    else:
        print("[WARN] Atribut 'head_detection' tidak ditemukan pada model!")
        
    # -- Langkah 3: Verifikasi status parameter untuk Audit Akademik -----------
    print("\n" + "-" * 72)
    print("  STATUS PARAMETER AKHIR (AUDIT GRADIENT)")
    print("-" * 72)
    
    trainable_count = 0
    frozen_count = 0
    
    for name, param in model.named_parameters():
        status = "TRAINABLE" if param.requires_grad else "FROZEN"
        if param.requires_grad:
            print(f"  [{status}] {name:<50} | Shape: {list(param.shape)}")
            trainable_count += param.numel()
        else:
            frozen_count += param.numel()
            
    total_params = trainable_count + frozen_count
    print("-" * 72)
    print(f"  Total Parameter     : {total_params:,}")
    print(f"  Parameter Dibekukan : {frozen_count:,} ({frozen_count/total_params*100:.2f}%)")
    print(f"  Parameter Dilatih   : {trainable_count:,} ({trainable_count/total_params*100:.2f}%)")
    print("-" * 72 + "\n")
    
    return model


def main():
    print("=" * 72)
    print("  FASE 3: Inisialisasi Decoupled Training (Stage 2)")
    print("  Model: V8 SupCon (EfficientNet-B1 + SupCon Loss)")
    print("=" * 72)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Running on: {device}")
    
    # 1. Inisialisasi Model V8
    if MODEL_AVAILABLE:
        model = MultiTaskScalogramV3_v8(
            pretrained=False
        ).to(device)
        print("[OK] Model MultiTaskScalogramV3_v8 berhasil diinstansiasi.")
    else:
        print("[ERROR] Repositori model tidak terdeteksi pada path!")
        return
        
    # 2. Terapkan Pembekuan Parameter (Stage 2 Transition)
    model = freeze_encoder_for_stage2(model)
    
    # 3. Kumpulkan parameter yang aktif untuk optimizer
    # Ini memastikan optimizer tidak membuang memori untuk menghitung gradien parameter beku
    trainable_parameters = [p for p in model.parameters() if p.requires_grad]
    
    # Gunakan Adam dengan learning rate kecil (misal 1e-4) khusus untuk fine-tuning classifier
    optimizer_stage2 = torch.optim.Adam(
        trainable_parameters, 
        lr=1e-4, 
        weight_decay=1e-5
    )
    
    print("  Optimizer Stage 2 siap digunakan:")
    print(f"  - Tipe optimizer : Adam")
    print(f"  - Learning Rate  : {optimizer_stage2.param_groups[0]['lr']}")
    print(f"  - Weight Decay   : {optimizer_stage2.param_groups[0]['weight_decay']}")
    print("=" * 72)


if __name__ == "__main__":
    main()
