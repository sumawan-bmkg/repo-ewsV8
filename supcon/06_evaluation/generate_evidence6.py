#!/usr/bin/env python3
"""EVIDENCE 6: Evaluation — Print-ready white theme, 300 DPI"""
import os, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (confusion_matrix, roc_curve, precision_recall_curve,
                             average_precision_score, roc_auc_score, ConfusionMatrixDisplay)
from sklearn.calibration import calibration_curve

OUT = 'D:/multi/scalogramv3/disertasi4/supcon/06_evaluation'
os.makedirs(OUT, exist_ok=True)
df = pd.read_csv('D:/multi/scalogramv3/blind_test_2026_v8_results/v8supcon_2026_predictions.csv')
y_true = df['True_Label'].values
y_prob = df['Pred_Prob'].values
y_pred = (y_prob >= 0.5).astype(int)

fig, axes = plt.subplots(2, 3, figsize=(22, 14))
fig.patch.set_facecolor('white')

# 1. Confusion Matrix
ax = axes[0, 0]; ax.set_facecolor('white')
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=['Negative', 'Positive'])
disp.plot(ax=ax, cmap='YlOrRd', colorbar=False, values_format='d', text_kw={'color':'#000000'})
ax.set_title('Confusion Matrix (th=0.50)', fontsize=12, fontweight='bold', color='#000000')

# 2. ROC
ax = axes[0, 1]; ax.set_facecolor('white')
fpr, tpr, _ = roc_curve(y_true, y_prob)
auc = roc_auc_score(y_true, y_prob)
ax.plot(fpr, tpr, color='#C0392B', linewidth=2, label=f'AUC = {auc:.4f}')
ax.plot([0,1],[0,1],'--', color='#888', linewidth=1)
ax.fill_between(fpr, tpr, alpha=0.1, color='#C0392B')
ax.set_title('ROC Curve', fontsize=12, fontweight='bold', color='#000000')
ax.set_xlabel('False Positive Rate', fontsize=10, color='#000000')
ax.set_ylabel('True Positive Rate', fontsize=10, color='#000000')
ax.legend(fontsize=9)
ax.tick_params(colors='#000000')
ax.spines['bottom'].set_color('#CCCCCC'); ax.spines['left'].set_color('#CCCCCC')

# 3. PR Curve
ax = axes[0, 2]; ax.set_facecolor('white')
prec, rec, _ = precision_recall_curve(y_true, y_prob)
auprc = average_precision_score(y_true, y_prob)
ax.plot(rec, prec, color='#16A085', linewidth=2, label=f'AUPRC = {auprc:.4f}')
ax.fill_between(rec, prec, alpha=0.1, color='#16A085')
ax.set_title('Precision-Recall Curve', fontsize=12, fontweight='bold', color='#000000')
ax.set_xlabel('Recall', fontsize=10, color='#000000')
ax.set_ylabel('Precision', fontsize=10, color='#000000')
ax.legend(fontsize=9)
ax.tick_params(colors='#000000')
ax.spines['bottom'].set_color('#CCCCCC'); ax.spines['left'].set_color('#CCCCCC')

# 4. Prob Histogram
ax = axes[1, 0]; ax.set_facecolor('white')
ax.hist([y_prob[y_true==1], y_prob[y_true==0]], bins=50,
        color=['#C0392B', '#2980B9'], label=['Event=1', 'Event=0'],
        alpha=0.7, edgecolor='#333', linewidth=0.3)
ax.axvline(0.5, color='#000000', linestyle='--', linewidth=1.5, label='Threshold=0.5')
ax.set_title('Prediction Probability Distribution', fontsize=12, fontweight='bold', color='#000000')
ax.set_xlabel('Predicted Probability', fontsize=10, color='#000000')
ax.set_ylabel('Count', fontsize=10, color='#000000')
ax.legend(fontsize=8)
ax.tick_params(colors='#000000')
ax.spines['bottom'].set_color('#CCCCCC'); ax.spines['left'].set_color('#CCCCCC')

# 5. Calibration
ax = axes[1, 1]; ax.set_facecolor('white')
pt, pp = calibration_curve(y_true, y_prob, n_bins=10)
ax.plot(pp, pt, marker='o', color='#D4AC0D', linewidth=2, markersize=8)
ax.plot([0,1],[0,1],'--', color='#888', linewidth=1, label='Perfect')
ax.set_title('Calibration Curve', fontsize=12, fontweight='bold', color='#000000')
ax.set_xlabel('Mean Predicted Probability', fontsize=10, color='#000000')
ax.set_ylabel('Fraction of Positives', fontsize=10, color='#000000')
ax.legend(fontsize=9)
ax.tick_params(colors='#000000')
ax.spines['bottom'].set_color('#CCCCCC'); ax.spines['left'].set_color('#CCCCCC')

# 6. Summary text
ax = axes[1, 2]; ax.axis('off')
TP=int(((y_pred==1)&(y_true==1)).sum())
TN=int(((y_pred==0)&(y_true==0)).sum())
FP=int(((y_pred==1)&(y_true==0)).sum())
FN=int(((y_pred==0)&(y_true==1)).sum())
rec_val=TP/(TP+FN) if(TP+FN)>0 else 0
prec_val=TP/(TP+FP) if(TP+FP)>0 else 0
f2_val=(5*prec_val*rec_val)/(4*prec_val+rec_val) if(4*prec_val+rec_val)>0 else 0
txt=(f"DETECTION METRICS (th=0.50)\n\n"
     f"True Positives:  {TP}\nTrue Negatives:  {TN}\n"
     f"False Positives: {FP}\nFalse Negatives: {FN}\n\n"
     f"Accuracy:   {(TP+TN)/(TP+TN+FP+FN):.4f}\n"
     f"Precision:  {prec_val:.4f}\nRecall:     {rec_val:.4f}\n"
     f"Specificity:{TN/(TN+FP):.4f}\n"
     f"FPR:        {FP/(FP+TN) if(FP+TN)>0 else 0:.4f}\n"
     f"F2 Score:   {f2_val:.4f}\n"
     f"AUPRC:      {auprc:.4f}\nAUC-ROC:    {auc:.4f}")
ax.text(0.1, 0.9, txt, fontsize=9, color='#000000', fontfamily='monospace', va='top')

fig.suptitle('V8 SupCon — Evaluation Metrics', fontsize=15, fontweight='bold', color='#000000', y=1.01)
plt.tight_layout()
for fmt in ['png', 'svg', 'pdf']:
    fig.savefig(f'{OUT}/evaluation_metrics.{fmt}', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

# CM pub
fig, ax = plt.subplots(1, 1, figsize=(8, 7))
fig.patch.set_facecolor('white'); ax.set_facecolor('white')
sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd', ax=ax,
            xticklabels=['Neg','Pos'], yticklabels=['Neg','Pos'],
            annot_kws={'fontsize':14,'fontweight':'bold','color':'#000000'})
ax.set_title('V8 SupCon — Confusion Matrix', fontsize=13, fontweight='bold', color='#000000')
ax.tick_params(colors='#000000')
plt.tight_layout()
fig.savefig(f'{OUT}/confusion_matrix_pub.pdf', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f'[OK] Evidence 6 saved to {OUT}/')
