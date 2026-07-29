# V8 SupCon — Dissertation Evidence Report (Bab 4)

> Generated: 2026-07-14 16:47:47 UTC+8
> Model: **MultiTaskScalogramV3_v8** (SupCon + True Negatives)
> Checkpoint: `v3_v8_conv_fpr_best_weights.pth`
> Output Base: `D:\multi\scalogramv3\disertasi4\supcon\`

---

## Summary

| Item | Status |
|------|--------|
| Total Evidence Figures | 12 |
| **Generated** | **9 / 12** |
| **Not Available** | **3 / 12** |
| Total Output Files | ~45 PNG/SVG/PDF + 13 Python scripts + validation reports |

---

## Evidence Inventory

### E1: Pipeline Preprocessing
**Status:** GENERATED
**Figures:**\n  - [pipeline_diagram.png](file:///D:/multi/scalogramv3/disertasi4/supcon/01_pipeline/pipeline_diagram.png)\n  - [pipeline_diagram.svg](file:///D:/multi/scalogramv3/disertasi4/supcon/01_pipeline/pipeline_diagram.svg)\n  - [pipeline_diagram.pdf](file:///D:/multi/scalogramv3/disertasi4/supcon/01_pipeline/pipeline_diagram.pdf)**Source:**\n  - V3_Model_v8.py architecture\n  - preprocessing source code**Content:** Pipeline preprocessing from raw ULF magnetometer to final tensor\n\n### E2: Dataset Distribution & Timeline
**Status:** GENERATED
**Figures:**\n  - [dataset_distribution.png](file:///D:/multi/scalogramv3/disertasi4/supcon/02_dataset/dataset_distribution.png)\n  - [dataset_distribution.svg](file:///D:/multi/scalogramv3/disertasi4/supcon/02_dataset/dataset_distribution.svg)\n  - [dataset_distribution.pdf](file:///D:/multi/scalogramv3/disertasi4/supcon/02_dataset/dataset_distribution.pdf)**Source:**\n  - v8supcon_2026_predictions.csv (2880 rows)\n  - 2026/merge2026.csv**Content:** Class balance, station distribution, daily event timeline, Kp histogram\n\n### E3: Model V8 SupCon Architecture
**Status:** GENERATED
**Figures:**\n  - [architecture_diagram.png](file:///D:/multi/scalogramv3/disertasi4/supcon/03_architecture/architecture_diagram.png)\n  - [architecture_diagram.svg](file:///D:/multi/scalogramv3/disertasi4/supcon/03_architecture/architecture_diagram.svg)\n  - [architecture_diagram.pdf](file:///D:/multi/scalogramv3/disertasi4/supcon/03_architecture/architecture_diagram.pdf)**Source:**\n  - ScalogramV3_V8_Repository/model/V3_Model_v8.py**Content:** EfficientNet-B1 → BiGRU → GNN → Cosmic Gate → SupCon → Multi-Task Heads\n\n### E4: Training History
**Status:** GENERATED
**Figures:**\n  - [training_history.png](file:///D:/multi/scalogramv3/disertasi4/supcon/04_training/training_history.png)\n  - [training_history.svg](file:///D:/multi/scalogramv3/disertasi4/supcon/04_training/training_history.svg)\n  - [training_history.pdf](file:///D:/multi/scalogramv3/disertasi4/supcon/04_training/training_history.pdf)**Source:**\n  - pull_real/logs/training_v8_convergence_history.csv (35 epochs)**Content:** Loss, precision, recall, FPR, F2, EWS, Brier, Azimuth MAE, LR vs epoch\n\n### E5: Ablation Study
**Status:** NOT AVAILABLE
**Reason:** Single consolidated training run. Component evidence from evidence_v8.json available but not structured ablation.\n\n### E6: Evaluation Metrics
**Status:** GENERATED
**Figures:**\n  - [evaluation_metrics.png](file:///D:/multi/scalogramv3/disertasi4/supcon/06_evaluation/evaluation_metrics.png)\n  - [evaluation_metrics.svg](file:///D:/multi/scalogramv3/disertasi4/supcon/06_evaluation/evaluation_metrics.svg)\n  - [evaluation_metrics.pdf](file:///D:/multi/scalogramv3/disertasi4/supcon/06_evaluation/evaluation_metrics.pdf)\n  - [confusion_matrix_pub.pdf](file:///D:/multi/scalogramv3/disertasi4/supcon/06_evaluation/confusion_matrix_pub.pdf)**Source:**\n  - v8supcon_2026_predictions.csv\n  - blind test 2880 samples**Content:** CM, ROC (AUC=0.464), PR (AUPRC=0.985), Calibration, Prob Histogram, Metrics Summary\n\n### E7: Latent Space Embedding
**Status:** GENERATED
**Figures:**\n  - [latent_space_embedding.png](file:///D:/multi/scalogramv3/disertasi4/supcon/07_latent/latent_space_embedding.png)\n  - [latent_space_embedding.svg](file:///D:/multi/scalogramv3/disertasi4/supcon/07_latent/latent_space_embedding.svg)\n  - [latent_space_embedding.pdf](file:///D:/multi/scalogramv3/disertasi4/supcon/07_latent/latent_space_embedding.pdf)**Source:**\n  - eswa/tsne_embedding_v8.csv (500 points, balanced)**Content:** t-SNE class scatter, t-SNE density map, SupCon projection analysis\n\n### E8: Spatial GNN
**Status:** NOT AVAILABLE
**Reason:** GNN attention weights are ephemeral forward() outputs not saved. Re-run with capture needed.\n\n### E9: Blind Test Timeline
**Status:** GENERATED
**Figures:**\n  - [blind_test_timeline.png](file:///D:/multi/scalogramv3/disertasi4/supcon/09_blindtest/blind_test_timeline.png)\n  - [blind_test_timeline.svg](file:///D:/multi/scalogramv3/disertasi4/supcon/09_blindtest/blind_test_timeline.svg)\n  - [blind_test_timeline.pdf](file:///D:/multi/scalogramv3/disertasi4/supcon/09_blindtest/blind_test_timeline.pdf)**Source:**\n  - v8supcon_2026_predictions.csv (2880 rows, Jan-Apr 2026)**Content:** Probability timeline, detection vs events, false positives, Kp storm activity\n\n### E10: Explainable AI (GradCAM)
**Status:** NOT AVAILABLE
**Reason:** GradCAM requires model re-execution with pytorch_grad_cam hooked. Persistent activations not saved.\n\n### E11: Model Comparison
**Status:** GENERATED
**Figures:**\n  - [comparison_chart.png](file:///D:/multi/scalogramv3/disertasi4/supcon/11_comparison/comparison_chart.png)\n  - [comparison_chart.svg](file:///D:/multi/scalogramv3/disertasi4/supcon/11_comparison/comparison_chart.svg)\n  - [comparison_chart.pdf](file:///D:/multi/scalogramv3/disertasi4/supcon/11_comparison/comparison_chart.pdf)**Source:**\n  - evidence_v8.json\n  - training_v8_convergence_history.csv**Content:** Bar chart per metric (V8 SupCon only), Radar chart for multi-dimensional performance\n\n### E12: Summary Figure
**Status:** GENERATED
**Figures:**\n  - [summary_figure.png](file:///D:/multi/scalogramv3/disertasi4/supcon/12_summary/summary_figure.png)\n  - [summary_figure.svg](file:///D:/multi/scalogramv3/disertasi4/supcon/12_summary/summary_figure.svg)\n  - [summary_figure.pdf](file:///D:/multi/scalogramv3/disertasi4/supcon/12_summary/summary_figure.pdf)\n  - [validation.md](file:///D:/multi/scalogramv3/disertasi4/supcon/12_summary/validation.md)**Source:**\n  - All previous evidence, config.yaml, training history**Content:** End-to-end pipeline diagram + final metrics summary + validation report\n\n
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
