"""
================================================================================
 FASE 3: OPTIMASI ARSITEKTUR — DECOUPLED TRAINING (CODE SNIPPET)
 Model: V8 SupCon (EfficientNet-B1 + Supervised Contrastive Learning)
 
 Lampiran Disertasi Doktoral — Program Studi Teknik Fisika
================================================================================
"""

import torch
import torch.nn as nn
from typing import Tuple, Iterator

def setup_decoupled_training(
    model: nn.Module, 
    learning_rate: float = 1e-4, 
    weight_decay: float = 1e-4
) -> Tuple[nn.Module, torch.optim.Optimizer]:
    """
    Mengonfigurasi model untuk 'Decoupled Training' (Stage-2 Linear Evaluation)
    dengan membekukan (freeze) seluruh parameter representasi spasial (backbone) 
    dan hanya melatih lapisan klasifikasi (classifier head).

    Landasan Teoretis:
    ------------------
    Model berbasis Supervised Contrastive Learning (SupCon) dilatih dalam dua tahap:
    1. Stage 1 (Representation Learning): Model belajar memetakan ruang fitur agar
       sampel dari kelas yang sama saling berdekatan (clustering) dan sampel beda kelas 
       saling berjauhan. Pada tahap ini, fitur dilatih menggunakan Contrastive Loss.
    2. Stage 2 (Linear Evaluation / Decoupled Training): Pembentukan decision boundary.
    
    Mengapa Dekopling Gradien Ini Wajib Dilakukan?
    Jika kita melatih ulang seluruh jaringan (backbone + head) secara bersamaan menggunakan 
    Cross Entropy (atau Focal Loss) pada fase klasifikasi, gradien dari head akan menjalar 
    kembali ke backbone. Hal ini memicu 'Representation Collapse' atau 'Catastrophic 
    Forgetting', di mana representasi manifold spasial yang sudah tersusun rapi oleh 
    SupCon menjadi rusak akibat tabrakan objektif antara Contrastive Loss (yang 
    menjaga margin jarak antar fitur) dan Focal Loss (yang berfokus pada probabilitas 
    skalar batas linier).

    Dengan membekukan encoder (requires_grad = False), kita menjamin bahwa ruang 
    fitur ULF geomagnetik tetap invarian secara fisik, sementara classifier head bebas 
    mengkalibrasi ulang distribusinya untuk mengatasi 'Undercoupled Probability Emission'.

    Parameter:
    ----------
    model : nn.Module
        Arsitektur V8 SupCon. Diasumsikan memiliki atribut `.head_detection` sebagai 
        layer klasifikasi linier terakhir.
    learning_rate : float
        Laju pembelajaran khusus untuk classifier head.
    weight_decay : float
        Regularisasi L2 (AdamW) untuk mencegah overfitting pada classifier head.

    Kembalian:
    ----------
    Tuple[nn.Module, torch.optim.Optimizer]
        Model dengan backbone beku, dan optimizer yang hanya memantau classifier head.
    """
    
    # 1. Bekukan (Freeze) seluruh parameter di dalam model (Backbone/Encoder)
    for param in model.parameters():
        param.requires_grad = False
        
    # 2. Cairkan (Unfreeze) hanya pada bagian Classifier Head.
    # Pada arsitektur V8 SupCon, layer klasifikasi target dinamakan 'head_detection'.
    if hasattr(model, 'head_detection'):
        for param in model.head_detection.parameters():
            param.requires_grad = True
    else:
        # Fallback jika penamaan layer berbeda, cari layer linier terakhir
        print("[WARNING] 'head_detection' tidak ditemukan. Membuka blok parameter fallback.")
        # Asumsikan layer terakhir ada di model.fc atau model.classifier
        if hasattr(model, 'fc'):
            for param in model.fc.parameters():
                param.requires_grad = True
        elif hasattr(model, 'classifier'):
            for param in model.classifier.parameters():
                param.requires_grad = True

    # 3. Filter hanya parameter yang aktif (requires_grad == True) untuk dimasukkan ke Optimizer
    trainable_parameters: Iterator[nn.Parameter] = filter(lambda p: p.requires_grad, model.parameters())
    
    # 4. Inisialisasi AdamW (Adam dengan Decoupled Weight Decay yang lebih stabil)
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # [OPSIONAL] Logika diagnostik forensik untuk sidang:
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    print("=" * 60)
    print("  DIAGNOSTIK DECOUPLED TRAINING FASE 3")
    print("=" * 60)
    print(f"Total Parameter     : {total_params:,}")
    print(f"Parameter Beku      : {frozen_params:,} ({frozen_params/total_params*100:.2f}% - Backbone)")
    print(f"Parameter Terbuka   : {trainable_params:,} ({trainable_params/total_params*100:.2f}% - Head)")
    print(f"Optimizer Terpilih  : AdamW (lr={learning_rate})")
    print("=" * 60)
    
    return model, optimizer

