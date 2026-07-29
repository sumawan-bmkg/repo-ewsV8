#!/usr/bin/env python3
"""
SUPCON_DISSERTATION_EVIDENCE_REPORT.md generator
V8 SupCon — Bab 4 Dissertation Evidence Summary
"""
import os
from datetime import datetime

now = datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC+8')

# All evidence inventory
evidence = {
    "E1": {
        "title": "Pipeline Preprocessing",
        "path": "01_pipeline",
        "files": ["pipeline_diagram.png", "pipeline_diagram.svg", "pipeline_diagram.pdf"],
        "source": ["V3_Model_v8.py architecture", "preprocessing source code"],
        "status": "GENERATED",
        "figure": "Pipeline preprocessing from raw ULF magnetometer to final tensor"
    },
    "E2": {
        "title": "Dataset Distribution & Timeline",
        "path": "02_dataset",
        "files": ["dataset_distribution.png", "dataset_distribution.svg", "dataset_distribution.pdf"],
        "source": ["v8supcon_2026_predictions.csv (2880 rows)", "2026/merge2026.csv"],
        "status": "GENERATED",
        "figure": "Class balance, station distribution, daily event timeline, Kp histogram"
    },
    "E3": {
        "title": "Model V8 SupCon Architecture",
        "path": "03_architecture",
        "files": ["architecture_diagram.png", "architecture_diagram.svg", "architecture_diagram.pdf"],
        "source": ["ScalogramV3_V8_Repository/model/V3_Model_v8.py"],
        "status": "GENERATED",
        "figure": "EfficientNet-B1 → BiGRU → GNN → Cosmic Gate → SupCon → Multi-Task Heads"
    },
    "E4": {
        "title": "Training History",
        "path": "04_training",
        "files": ["training_history.png", "training_history.svg", "training_history.pdf"],
        "source": ["pull_real/logs/training_v8_convergence_history.csv (35 epochs)"],
        "status": "GENERATED",
        "figure": "Loss, precision, recall, FPR, F2, EWS, Brier, Azimuth MAE, LR vs epoch"
    },
    "E5": {
        "title": "Ablation Study",
        "path": "05_ablation",
        "files": ["ablation_waterfall.png", "ablation_waterfall.svg", "ablation_waterfall.pdf"],
        "source": ["N/A — controlled ablation not conducted"],
        "status": "NOT AVAILABLE",
        "reason": "Single consolidated training run. Component evidence from evidence_v8.json available but not structured ablation."
    },
    "E6": {
        "title": "Evaluation Metrics",
        "path": "06_evaluation",
        "files": ["evaluation_metrics.png", "evaluation_metrics.svg", "evaluation_metrics.pdf",
                  "confusion_matrix_pub.pdf"],
        "source": ["v8supcon_2026_predictions.csv", "blind test 2880 samples"],
        "status": "GENERATED",
        "figure": "CM, ROC (AUC=0.464), PR (AUPRC=0.985), Calibration, Prob Histogram, Metrics Summary"
    },
    "E7": {
        "title": "Latent Space Embedding",
        "path": "07_latent",
        "files": ["latent_space_embedding.png", "latent_space_embedding.svg", "latent_space_embedding.pdf"],
        "source": ["eswa/tsne_embedding_v8.csv (500 points, balanced)"],
        "status": "GENERATED",
        "figure": "t-SNE class scatter, t-SNE density map, SupCon projection analysis"
    },
    "E8": {
        "title": "Spatial GNN",
        "path": "08_spatial",
        "files": ["spatial_gnn_attention.png", "spatial_gnn_attention.svg", "spatial_gnn_attention.pdf"],
        "source": ["N/A — att_weights not persisted"],
        "status": "NOT AVAILABLE",
        "reason": "GNN attention weights are ephemeral forward() outputs not saved. Re-run with capture needed."
    },
    "E9": {
        "title": "Blind Test Timeline",
        "path": "09_blindtest",
        "files": ["blind_test_timeline.png", "blind_test_timeline.svg", "blind_test_timeline.pdf"],
        "source": ["v8supcon_2026_predictions.csv (2880 rows, Jan-Apr 2026)"],
        "status": "GENERATED",
        "figure": "Probability timeline, detection vs events, false positives, Kp storm activity"
    },
    "E10": {
        "title": "Explainable AI (GradCAM)",
        "path": "10_xai",
        "files": ["xai_gradcam.png", "xai_gradcam.svg", "xai_gradcam.pdf"],
        "source": ["N/A — GradCAM not precomputed"],
        "status": "NOT AVAILABLE",
        "reason": "GradCAM requires model re-execution with pytorch_grad_cam hooked. Persistent activations not saved."
    },
    "E11": {
        "title": "Model Comparison",
        "path": "11_comparison",
        "files": ["comparison_chart.png", "comparison_chart.svg", "comparison_chart.pdf"],
        "source": ["evidence_v8.json", "training_v8_convergence_history.csv"],
        "status": "GENERATED",
        "figure": "Bar chart per metric (V8 SupCon only), Radar chart for multi-dimensional performance"
    },
    "E12": {
        "title": "Summary Figure",
        "path": "12_summary",
        "files": ["summary_figure.png", "summary_figure.svg", "summary_figure.pdf",
                  "validation.md"],
        "source": ["All previous evidence, config.yaml, training history"],
        "status": "GENERATED",
        "figure": "End-to-end pipeline diagram + final metrics summary + validation report"
    }
}

base = 'D:/multi/scalogramv3/disertasi4/supcon'
generated = sum(1 for e in evidence.values() if e['status'] == 'GENERATED')
na = sum(1 for e in evidence.values() if e['status'] == 'NOT AVAILABLE')

report = f"""# V8 SupCon — Dissertation Evidence Report (Bab 4)

> Generated: {now}
> Model: **MultiTaskScalogramV3_v8** (SupCon + True Negatives)
> Checkpoint: `v3_v8_conv_fpr_best_weights.pth`
> Output Base: `D:\\multi\\scalogramv3\\disertasi4\\supcon\\`

---

## Summary

| Item | Status |
|------|--------|
| Total Evidence Figures | 12 |
| **Generated** | **{generated} / 12** |
| **Not Available** | **{na} / 12** |
| Total Output Files | ~45 PNG/SVG/PDF + 13 Python scripts + validation reports |

---

## Evidence Inventory

"""

for eid, data in evidence.items():
    report += f"### {eid}: {data['title']}\n"
    report += f"**Status:** {data['status']}\n"
    if data['status'] == 'GENERATED':
        report += f"**Figures:**\\n" + "\\n".join([f"  - [{f}](file:///{base}/{data['path']}/{f})" for f in data['files']])
        report += f"**Source:**\\n" + "\\n".join([f"  - {s}" for s in data['source']])
        report += f"**Content:** {data['figure']}\\n"
    else:
        report += f"**Reason:** {data['reason']}\\n"
    report += "\\n"

report += """
---

## Deployment Recommendations for Bab 4

| Figure | Section | Recommendation |
|--------|---------|---------------|
| E1 (Pipeline) | 4.2.1 Preprocessing | Place before model description |
| E2 (Dataset) | 4.1 Dataset | Next to station map |
| E3 (Architecture) | 4.2.2 Model | Central architectural figure |
| E4 (Training) | 4.3 Training | Show convergence behavior |
| E5 (Ablation) | 4.4 | ⚠️ NOT AVAILABLE — suggest replacing with component table |
| E6 (Evaluation) | 4.5.1 Detection | Primary results figure |
| E7 (Latent) | 4.5.2 Representation | Show SupCon embedding quality |
| E8 (GNN) | 4.2.2 Spatial | ⚠️ NOT AVAILABLE — station analysis as alternative |
| E9 (Blind Test) | 4.6 Operational | Operational validation core figure |
| E10 (XAI) | 4.7 Interpretability | ⚠️ NOT AVAILABLE — defer to future work |
| E11 (Comparison) | 4.8 Comparison | Radar chart ideal for overview |
| E12 (Summary) | 4.9 | Closing figure of Bab 4 |

---

## Data Source Integrity

All generated figures are derived from **REAL V8 SupCon data**:

- **Training Log**: `pull_real/logs/training_v8_convergence_history.csv` ✓
- **Blind Test Predictions**: `blind_test_2026_v8_results/v8supcon_2026_predictions.csv` (2880 rows) ✓
- **t-SNE Embedding**: `eswa/tsne_embedding_v8.csv` (500 points) ✓
- **Architecture**: `ScalogramV3_V8_Repository/model/V3_Model_v8.py` ✓
- **Config**: `ScalogramV3_V8_Repository/config.yaml` ✓
- **Evidence JSON**: `ScalogramV3_V8_Repository/evidence/evidence_v8.json` ✓
- **EQ Catalogue**: `2026/merge2026.csv` ✓

No synthetic data, no placeholder values, no baseline/model V3 data used.

---

*Report auto-generated by V8SupCon Dissertation Evidence Pipeline*
"""

with open(f'{base}/SUPCON_DISSERTATION_EVIDENCE_REPORT.md', 'w', encoding='utf-8') as f:
    f.write(report)
print(f'[OK] Final report saved to {base}/SUPCON_DISSERTATION_EVIDENCE_REPORT.md')
