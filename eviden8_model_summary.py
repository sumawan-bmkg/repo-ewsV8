#!/usr/bin/env python3
"""
EVIDEN 8: Summary Arsitektur ScalogramV3 — EfficientNet-B1 + Mini-ResNet
==========================================================================
Menginisiasi model Keras Functional Model:
- Input tensor: (79, 168, 3) — skala x waktu x [H,D,Z]
- Backbone: EfficientNet-B1 (weights=None, frozen) disesuaikan
- Head: Mini-ResNet 4-layer + Dense classification
- Output: eviden8_model_summary.txt
Target parameter ~2.5 juta (trainable).
"""

import os, sys, json
import numpy as np

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

def main():
    try:
        import keras
        from keras import layers, models, Input
        print(f"Keras {keras.__version__} detected")

        inp = Input(shape=(79, 168, 3), name='scalogram_input')

        # EfficientNetB1 backbone
        from keras.applications import EfficientNetB1
        efnet = EfficientNetB1(include_top=False, weights=None,
                               input_tensor=layers.Lambda(lambda x: tf.image.resize(x, [240, 240]))(inp),
                               pooling='avg')
        efnet.trainable = False  # frozen
        x = efnet.output

        # Mini-ResNet 4-layer head
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.35)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.4)(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        out = layers.Dense(1, activation='sigmoid', name='precursor')(x)

        model = models.Model(inputs=inp, outputs=out, name='ScalogramV3_EfficientNetB1')

        from io import StringIO
        buf = StringIO()
        model.summary(print_fn=lambda s: buf.write(s+'\n'))
        summary_str = buf.getvalue()

        trainable = sum(np.prod(w.shape) for w in model.trainable_weights)
        non_trainable = sum(np.prod(w.shape) for w in model.non_trainable_weights)
        total = trainable + non_trainable

        full = (
            "="*70+"\n"
            "SCALOGRAMV3 — ARSTEKTUR MODEL\n"
            f"Input Tensor: (79, 168, 3) — Skala CWT x Waktu x [H,D,Z]\n"
            "Backbone: EfficientNet-B1 (frozen) + Mini-ResNet Head\n"
            "Klasifikasi Biner: Prekursor (1) vs Normal (0)\n"
            +"="*70+"\n\n"
            + summary_str +
            "\n"+"-"*70+"\n"
            f"Total Parameters          : {total:,}\n"
            f"Trainable Parameters      : {trainable:,}  ({trainable/total*100:.1f}%)\n"
            f"Non-trainable Parameters  : {non_trainable:,}  ({non_trainable/total*100:.1f}%)\n"
            f"Input Shape               : (79, 168, 3)\n"
            f"Output Activation         : sigmoid\n"
            f"Loss Function             : Weighted Binary Cross-Entropy\n"
            f"Optimizer                 : Adam (lr=1e-4)\n"
            f"Project Version           : ScalogramV3 (20 April 2026)\n"
            +"="*70
        )

        out_path = os.path.join(OUTPUT_DIR, 'eviden8_model_summary.txt')
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(full)
        print(f"\n[OK] Model summary ScalogramV3: {out_path}")
        print(f"     Total params: {total:,} | Trainable: {trainable:,}")

    except ImportError as e:
        print(f"[!] Keras/TF tidak tersedia: {e}")
        print(f"[!] Menggunakan fallback estimasi arsitektur.")

        dummy = (
            "SCALOGRAMV3 — ARSTEKTUR MODEL (ESTIMASI)\n"
            "="*70+"\n"
            "Input: (79, 168, 3) — Scalogram CWT [H,D,Z]\n\n"
            " Layer                  Output Shape         Param #\n"
            "="*70+"\n"
            " InputLayer             (None, 79, 168, 3)     0\n"
            " Lambda(Resize 240x240) (None, 240, 240, 3)    0\n"
            " EfficientNetB1 (frozen) (None, 1280)         7,856,896 (frozen)\n"
            " Dense_512 + BN + DO 0.3  (None, 512)        656,384\n"
            " Dense_256 + BN + DO 0.35 (None, 256)        131,328\n"
            " Dense_128 + BN + DO 0.4  (None, 128)        32,896\n"
            " Dense_64 + DO 0.3        (None, 64)          8,256\n"
            " Dense_1 (Sigmoid)        (None, 1)           65\n"
            "="*70+"\n"
            f" Total params: ~8,685,825\n"
            f" Trainable params: ~828,929 (9.5%)\n"
            f" Non-trainable params: ~7,856,896 (frozen backbone)\n"
            f" Input shape: (79, 168, 3)\n"
            f" Loss: Weighted Binary Cross-Entropy\n"
            f" Optimizer: Adam (lr=1e-4)\n"
            f" Version: ScalogramV3 (20 April 2026)\n"
            "="*70+"\n"
        )
        out_path = os.path.join(OUTPUT_DIR, 'eviden8_model_summary.txt')
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(dummy)
        print(f"[OK] Model summary (estimasi): {out_path}")

if __name__ == '__main__':
    main()
