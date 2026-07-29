"""
================================================================================
 FASE 2 — Binary Focal Loss untuk Resume Training (Epoch 18-50)
 Model: V8 SupCon (EfficientNet-B1 + Supervised Contrastive Learning)

 Focal Loss mengatasi ketidakseimbangan kelas ekstrem dengan memfokuskan 
 gradien pada sampel sulit (positif yang terlewat) dan mengabaikan sampel 
 mudah (negatif yang sudah benar). Parameter alpha dan gamma memberikan 
 kontrol terhadap bobot kelas dan derajat penalti prediksi konservatif.

 Formula:  FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

 Lampiran Disertasi Doktoral — Program Studi Teknik Fisika
================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryFocalLoss(nn.Module):
    """
    Binary Focal Loss untuk klasifikasi biner tidak seimbang.
    
    Implementasi ini dirancang khusus untuk mengatasi Undercoupled 
    Probability Emission pada model V8 SupCon yang dilatih dengan 
    distribusi kelas ekstrem (98.3% positif).
    
    Argumen Constructor:
        alpha     : Bobot kelas positif. alpha > 0.5 memprioritaskan recall.
                    Default: 0.75 (memberi bobot lebih pada event gempa).
        gamma     : Parameter pemfokus. gamma > 0 menekan loss sampel mudah.
                    Default: 2.0 (standard untuk deteksi sinyal lemah).
        reduction : Metode reduksi tensor ('mean', 'sum', atau 'none').
    
    Input:
        logits   : Tensor (B,) atau (B, 1) — output mentah sebelum sigmoid.
        targets  : Tensor (B,) atau (B, 1) — label biner {0, 1}.
    
    Output:
        Tensor scalar (jika reduction='mean' atau 'sum'), atau tensor (B,) jika 'none'.
    """
    
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0, reduction: str = 'mean'):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
        # Validasi parameter
        assert 0.0 <= alpha <= 1.0, "alpha harus berada di [0, 1]"
        assert gamma >= 0.0, "gamma harus >= 0 (0 = identik dengan BCE)"

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass untuk Focal Loss.
        
        Proses komputasi:
          1. Hitung BCE loss mentah per sampel (logaritma prediksi terhadap target).
          2. Hitung probabilitas prediksi p_t = sigmoid(logits) jika target=1,
             atau 1-sigmoid(logits) jika target=0.
          3. Terapkan modulator (1 - p_t)^gamma untuk menekan loss sampel mudah.
          4. Terapkan bobot kelas alpha_t untuk mengoreksi bias distribusi.
        """
        # Pastikan targets memiliki tipe float dan shape yang konsisten
        targets = targets.float().view_as(logits)
        
        # Komponen 1: Binary Cross Entropy tanpa reduksi (per-sampel)
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        
        # Komponen 2: Probabilitas prediksi (p_t) untuk kelas target aktual
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
        
        # Komponen 3: Modulator pemfokus — menekan loss dari sampel yang mudah
        # (1 - p_t)^gamma: jika p_t tinggi (mudah), modulator mendekati 0
        modulating_factor = torch.pow(1.0 - p_t, self.gamma)
        
        # Komponen 4: Bobot kelas — alpha_t menyesuaikan bobot per kelas
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        
        # Gabungan: FL = -alpha_t * (1-p_t)^gamma * BCE
        focal_loss = alpha_t * modulating_factor * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


# ------------------------------------------------------------------------------
# DEMONSTRASI PENGGUNAAN DALAM TRAINING LOOP
# ------------------------------------------------------------------------------

def demonstrate_training_loop():
    """
    Contoh integrasi BinaryFocalLoss ke dalam training loop V8 SupCon.
    Fungsi ini mensimulasikan satu epoch pelatihan untuk verifikasi konsistensi.
    """
    print("=" * 72)
    print("  DEMONSTRASI: Integrasi Focal Loss ke Training Loop V8 SupCon")
    print("=" * 72)
    
    # -- Konfigurasi ----------------------------------------------------------
    batch_size = 32
    input_channels = 3
    height, width = 128, 360  # Ukuran input CWT scalogram
    n_epochs = 1
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Device: {device}")
    
    # -- Inisialisasi Model & Loss --------------------------------------------
    # Gunakan model V8 SupCon dari repositori
    try:
        import sys
        sys.path.insert(0, 'D:/multi/scalogramv3/ScalogramV3_V8_Repository/model')
        from V3_Model_v8 import MultiTaskScalogramV3_v8
        
        model = MultiTaskScalogramV3_v8(
            backbone_name='efficientnet-b1',
            pretrained=False,
            fusion_dim=512
        ).to(device)
        print("[OK] Model V8 SupCon dimuat")
    except ImportError:
        print("[WARN] Model V8 tidak tersedia, menggunakan placeholder")
        # Placeholder untuk demonstrasi standalone
        model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_channels * height * width, 512),
            nn.ReLU(),
            nn.Linear(512, 2)
        ).to(device)
    
    # -- Inisialisasi Focal Loss ----------------------------------------------
    # alpha=0.75: Memberi bobot 75% pada kelas positif (gempa)
    # gamma=2.0:  Penalti kuat pada prediksi konservatif (sudah benar tapi yakin)
    criterion = BinaryFocalLoss(alpha=0.75, gamma=2.0, reduction='mean')
    
    # Optimizer: Adam dengan learning rate kecil untuk resume training
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-5)
    
    print(f"[OK] BinaryFocalLoss: alpha={criterion.alpha}, gamma={criterion.gamma}")
    print(f"[OK] Optimizer: Adam, lr=1e-5, weight_decay=1e-5")
    
    # -- Simulasi Data --------------------------------------------------------
    # Mencerminkan ketidakseimbangan kelas aktual: 98.3% positif, 1.7% negatif
    n_pos = 30  # Gempa
    n_neg = 2   # Non-event
    n_total = n_pos + n_neg
    
    x_img = torch.randn(n_total, input_channels, height, width).to(device)
    x_cosmic = torch.randn(n_total, 2).to(device)
    y_event = torch.cat([
        torch.ones(n_pos),    # Label gempa
        torch.zeros(n_neg)    # Label non-event
    ]).to(device)
    
    print(f"\n[DATA] Batch size: {n_total} (Positif={n_pos}, Negatif={n_neg})")
    
    # -- Training Loop (Resume dari Epoch 18) ---------------------------------
    print(f"\n{'-'*72}")
    print(f"  Resume Training — Epoch 18 (menggunakan Focal Loss)")
    print(f"{'-'*72}")
    
    model.train()
    
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        
        try:
            # Forward pass pada model V8
            outputs = model(x_img, x_cosmic)
            # V8 mengembalikan tuple: (out_detection, out_magnitude, out_azimuth, ...)
            out_detection = outputs[0]  # Shape: (B, 2) — raw logits
        except Exception:
            # Fallback untuk model placeholder
            out_detection = model(x_img)
        
        # -- Konversi logits untuk Focal Loss biner ---------------------------
        # out_detection berisi logits [logit_negatif, logit_positif]
        # Kita butuh skor relatif: logit_positif - logit_negatif
        if out_detection.dim() == 2 and out_detection.shape[-1] == 2:
            binary_logits = out_detection[:, 1] - out_detection[:, 0]  # (B,)
        else:
            binary_logits = out_detection.squeeze(-1)
        
        # Hitung Focal Loss
        loss = criterion(binary_logits, y_event)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        print(f"\n  Epoch 18 | Focal Loss = {loss.item():.6f}")
        
        # Hitung metrik prediksi
        with torch.no_grad():
            probs = torch.sigmoid(binary_logits)
            preds = (probs >= 0.25).float()
            tp = ((preds == 1) & (y_event == 1)).sum().item()
            fn = ((preds == 0) & (y_event == 1)).sum().item()
            fp = ((preds == 1) & (y_event == 0)).sum().item()
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            
            print(f"  Predictions  : TP={tp}, FN={fn}, FP={fp}")
            print(f"  Recall={recall*100:.1f}%  Precision={precision*100:.1f}%")
            print(f"  Prob range   : [{probs.min().item():.4f}, {probs.max().item():.4f}]")
            print(f"  Mean prob    : {probs.mean().item():.4f}")
    
    print(f"\n{'='*72}")
    print("  KESIMPULAN FASE 2")
    print(f"{'='*72}")
    print("  BinaryFocalLoss dengan alpha=0.75, gamma=2.0:")
    print("  - Memfokuskan gradien pada sampel positif yang terlewat model")
    print("  - Menekan dominasi gradien dari sampel negatif mudah")
    print("  - Mengatasi Undercoupled Probability Emission dari akarnya")
    print(f"{'='*72}")
    
    return criterion


if __name__ == "__main__":
    demonstrate_training_loop()
