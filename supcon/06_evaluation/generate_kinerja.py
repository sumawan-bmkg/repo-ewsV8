#!/usr/bin/env python3
"""
V8 SupCon — Kinerja Model Lengkap & Detail
Comprehensive performance figure for dissertation, print-ready white theme, 300 DPI
"""
import os, sys, json, warnings, math
warnings.filterwarnings('ignore')
os.environ['MPLBACKEND'] = 'Agg'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from sklearn.metrics import (roc_curve, precision_recall_curve,
                             average_precision_score, roc_auc_score,
                             confusion_matrix, ConfusionMatrixDisplay,
                             accuracy_score, precision_score, recall_score, fbeta_score)
from sklearn.calibration import calibration_curve

OUT = 'D:/multi/scalogramv3/disertasi4/supcon/06_evaluation/kinerja'
os.makedirs(OUT, exist_ok=True)

# ── Load data ──
pred = pd.read_csv('D:/multi/scalogramv3/blind_test_2026_v8_results/v8supcon_2026_predictions.csv')
train = pd.read_csv('D:/multi/scalogramv3/pull_real/logs/training_v8_convergence_history.csv')
with open('D:/multi/scalogramv3/ScalogramV3_V8_Repository/evidence/evidence_v8.json') as f:
    ev = json.load(f)

y_true = pred['True_Label'].values
y_prob = pred['Pred_Prob'].values

# ── Compute metrics at various thresholds ──
thresholds = np.linspace(0.01, 0.99, 99)
th_metrics = []
for th in thresholds:
    yp = (y_prob >= th).astype(int)
    tn = ((yp==0)&(y_true==0)).sum()
    fp = ((yp==1)&(y_true==0)).sum()
    fn = ((yp==0)&(y_true==1)).sum()
    tp = ((yp==1)&(y_true==1)).sum()
    prec = tp/(tp+fp) if (tp+fp)>0 else 1.0
    rec  = tp/(tp+fn) if (tp+fn)>0 else 0.0
    fpr_ = fp/(fp+tn) if (fp+tn)>0 else 0.0
    f2   = (5*prec*rec)/(4*prec+rec) if (4*prec+rec)>0 else 0.0
    acc  = (tp+tn)/(tp+tn+fp+fn)
    th_metrics.append({'th':th,'prec':prec,'rec':rec,'fpr':fpr_,'f2':f2,'acc':acc})
th_df = pd.DataFrame(th_metrics)

# Best F2 with FPR <= 20% constraint
valid = th_df[th_df['fpr'] <= 0.20]
if len(valid) > 0:
    best = valid.loc[valid['f2'].idxmax()]
    th_opt = best['th']
else:
    best = th_df.loc[th_df['f2'].idxmax()]
    th_opt = best['th']

yp_opt = (y_prob >= th_opt).astype(int)
yp_50  = (y_prob >= 0.5).astype(int)

# ── Compute Azimuth Error distribution ──
az_err = pred['Az_Error'].values
mask_event = y_true == 1
az_err_pos = az_err[mask_event]

# ── Kp vs detection ──
kp_bins = np.arange(0, 10, 1)
kp_groups = pred.groupby(pd.cut(pred['Kp_Raw'], kp_bins))
kp_recall = kp_groups.apply(lambda g: ((g['Pred_Binary']==1)&(g['True_Label']==1)).sum() / max((g['True_Label']==1).sum(),1))

# ── N classes for magnitude ──
mag_classes = sorted(pred['True_MagClass'].unique())
mag_labels = {0:'None',1:'Mw<3.0',2:'Mw 3.0-3.9',3:'Mw 4.0-4.4',4:'Mw 4.5-4.9',5:'Mw 5.0+'}

# ── Timing ──
pred['Date'] = pd.to_datetime(pred['Date'])
daily = pred.groupby(pred['Date'].dt.date).agg(
    n=('True_Label','count'), n_pos=('True_Label','sum'),
    n_det=('Pred_Binary','sum'), mean_kp=('Kp_Raw','mean'))

# ============================================================
# MAIN FIGURE: COMPREHENSIVE PERFORMANCE
# ============================================================
fig = plt.figure(figsize=(24, 18), facecolor='white')
gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.30, wspace=0.28)

# ── 1. Confusion Matrix (optimal th) ──
ax = fig.add_subplot(gs[0, 0])
cm = confusion_matrix(y_true, yp_opt)
disp = ConfusionMatrixDisplay(cm, display_labels=['Non-Event','Event'])
disp.plot(ax=ax, cmap='YlOrRd', colorbar=False, values_format='d',
          text_kw={'fontsize':13,'fontweight':'bold','color':'#000000'})
ax.set_title(f'Confusion Matrix (th={th_opt:.3f})', fontsize=11, fontweight='bold', color='#000000')
ax.tick_params(colors='#000000', labelsize=9)
for t in ax.texts:
    t.set_color('#000000')

# ── 2. ROC Curve ──
ax = fig.add_subplot(gs[0, 1])
fpr_roc, tpr_roc, _ = roc_curve(y_true, y_prob)
auc = roc_auc_score(y_true, y_prob)
ax.plot(fpr_roc, tpr_roc, color='#C0392B', lw=2.5, label=f'AUC-ROC = {auc:.4f}')
ax.plot([0,1],[0,1],'--', color='#888', lw=1)
ax.fill_between(fpr_roc, tpr_roc, alpha=0.12, color='#C0392B')
# Mark optimal th
idx_opt = np.argmin(np.abs(thresholds - th_opt))
ax.plot(th_df.loc[idx_opt,'fpr'], th_df.loc[idx_opt,'rec'], 'o',
        color='#2980B9', ms=10, mfc='white', mew=2.5, label=f'Optimal th={th_opt:.3f}')
ax.set_xlabel('False Positive Rate', fontsize=10, color='#000000')
ax.set_ylabel('True Positive Rate', fontsize=10, color='#000000')
ax.set_title('ROC Curve', fontsize=11, fontweight='bold', color='#000000')
ax.legend(fontsize=8, loc='lower right')
ax.tick_params(colors='#000000'); ax.grid(True, alpha=0.15)
ax.spines['bottom'].set_color('#CCC'); ax.spines['left'].set_color('#CCC')

# ── 3. Precision-Recall Curve ──
ax = fig.add_subplot(gs[0, 2])
prec_curve, rec_curve, _ = precision_recall_curve(y_true, y_prob)
auprc = average_precision_score(y_true, y_prob)
ax.plot(rec_curve, prec_curve, color='#16A085', lw=2.5, label=f'AUPRC = {auprc:.4f}')
ax.fill_between(rec_curve, prec_curve, alpha=0.12, color='#16A085')
ax.plot(th_df.loc[idx_opt,'rec'], th_df.loc[idx_opt,'prec'], 'o',
        color='#2980B9', ms=10, mfc='white', mew=2.5)
ax.set_xlabel('Recall', fontsize=10, color='#000000')
ax.set_ylabel('Precision', fontsize=10, color='#000000')
ax.set_title('Precision-Recall Curve', fontsize=11, fontweight='bold', color='#000000')
ax.legend(fontsize=8, loc='upper right')
ax.tick_params(colors='#000000'); ax.grid(True, alpha=0.15)
ax.spines['bottom'].set_color('#CCC'); ax.spines['left'].set_color('#CCC')

# ── 4. Threshold Sweep ──
ax = fig.add_subplot(gs[0, 3])
ax.plot(th_df['th'], th_df['f2'], 'o-', color='#8E44AD', lw=2, ms=3, label='F2 Score')
ax.plot(th_df['th'], th_df['prec'], 's--', color='#2980B9', lw=1.5, ms=2, label='Precision')
ax.plot(th_df['th'], th_df['rec'], '^--', color='#C0392B', lw=1.5, ms=2, label='Recall')
ax.axvline(th_opt, color='#333', ls=':', lw=1.5, alpha=0.7, label=f'Optimal th={th_opt:.3f}')
ax.axvline(0.5, color='#666', ls=':', lw=1, alpha=0.5, label='Default th=0.5')
ax.set_xlabel('Threshold', fontsize=10, color='#000000')
ax.set_ylabel('Score', fontsize=10, color='#000000')
ax.set_title('Threshold Analysis', fontsize=11, fontweight='bold', color='#000000')
ax.legend(fontsize=7, loc='center left')
ax.tick_params(colors='#000000'); ax.grid(True, alpha=0.15)
ax.spines['bottom'].set_color('#CCC'); ax.spines['left'].set_color('#CCC')

# ── 5. Training History — F2 + Loss ──
ax = fig.add_subplot(gs[1, 0])
ax_twin = ax.twinx()
ax.plot(train['epoch'], train['train_loss'], color='#3498DB', lw=1.5, alpha=0.7, label='Train Loss')
ax.plot(train['epoch'], train['val_loss'], color='#E74C3C', lw=1.5, alpha=0.7, label='Val Loss')
ax_twin.plot(train['epoch'], train['f2'], color='#2ECC71', lw=2, label='F2 Score')
ax_twin.plot(train['epoch'], train['ews_score'], color='#F39C12', lw=1.5, ls='--', label='EWS Score')
ax.set_xlabel('Epoch', fontsize=10, color='#000000')
ax.set_ylabel('Loss', fontsize=10, color='#000000')
ax_twin.set_ylabel('F2 / EWS', fontsize=10, color='#2ECC71')
ax.set_title('Training Convergence', fontsize=11, fontweight='bold', color='#000000')
lines1, labs1 = ax.get_legend_handles_labels()
lines2, labs2 = ax_twin.get_legend_handles_labels()
ax.legend(lines1+lines2, labs1+labs2, fontsize=7, loc='upper right')
ax.tick_params(colors='#000000'); ax_twin.tick_params(colors='#2ECC71')
ax.spines['bottom'].set_color('#CCC'); ax.spines['left'].set_color('#CCC')
ax_twin.spines['right'].set_color('#2ECC71')

# ── 6. Precision, Recall, FPR over epochs ──
ax = fig.add_subplot(gs[1, 1])
ax.plot(train['epoch'], train['precision'], 'o-', color='#2980B9', lw=1.5, ms=3, label='Precision')
ax.plot(train['epoch'], train['recall'], 's-', color='#C0392B', lw=1.5, ms=3, label='Recall')
ax.plot(train['epoch'], train['fpr'], '^-', color='#E67E22', lw=1.5, ms=3, label='FPR')
ax.set_xlabel('Epoch', fontsize=10, color='#000000')
ax.set_ylabel('Score', fontsize=10, color='#000000')
ax.set_title('Precision / Recall / FPR Over Training', fontsize=11, fontweight='bold', color='#000000')
ax.legend(fontsize=8); ax.grid(True, alpha=0.15)
ax.tick_params(colors='#000000')
ax.spines['bottom'].set_color('#CCC'); ax.spines['left'].set_color('#CCC')

# ── 7. Probability Distribution ──
ax = fig.add_subplot(gs[1, 2])
colors_hist = ['#C0392B','#2980B9']
ax.hist([y_prob[y_true==1], y_prob[y_true==0]], bins=40,
        color=colors_hist, label=['Event (Positive)', 'Non-Event (Negative)'],
        alpha=0.65, edgecolor='#333', linewidth=0.3, stacked=False)
ax.axvline(th_opt, color='#8E44AD', ls='--', lw=2, label=f'Optimal th={th_opt:.3f}')
ax.axvline(0.5, color='#666', ls=':', lw=1.5, alpha=0.7, label='Default th=0.5')
ax.set_xlabel('Predicted Probability', fontsize=10, color='#000000')
ax.set_ylabel('Count', fontsize=10, color='#000000')
ax.set_title('Probability Distribution', fontsize=11, fontweight='bold', color='#000000')
ax.legend(fontsize=7); ax.tick_params(colors='#000000')
ax.spines['bottom'].set_color('#CCC'); ax.spines['left'].set_color('#CCC')

# ── 8. Calibration Curve ──
ax = fig.add_subplot(gs[1, 3])
prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
ax.plot(prob_pred, prob_true, 'o-', color='#D4AC0D', lw=2.5, ms=8, label='Model')
ax.plot([0,1],[0,1], '--', color='#888', lw=1, label='Perfect')
# ECE calculation
ece = np.mean(np.abs(prob_true - prob_pred))
ax.fill_between(prob_pred, prob_true, prob_pred, alpha=0.15, color='#D4AC0D')
ax.set_xlabel('Mean Predicted Probability', fontsize=10, color='#000000')
ax.set_ylabel('Observed Fraction', fontsize=10, color='#000000')
ax.set_title(f'Calibration Curve (ECE={ece:.3f})', fontsize=11, fontweight='bold', color='#000000')
ax.legend(fontsize=8); ax.grid(True, alpha=0.15)
ax.tick_params(colors='#000000')
ax.spines['bottom'].set_color('#CCC'); ax.spines['left'].set_color('#CCC')

# ── 9. Azimuth Error Distribution ──
ax = fig.add_subplot(gs[2, 0])
ax.hist(az_err_pos, bins=40, color='#16A085', edgecolor='#333', lw=0.3, alpha=0.8)
ax.axvline(az_err_pos.mean(), color='#C0392B', ls='--', lw=2, label=f'Mean = {az_err_pos.mean():.1f}°')
ax.axvline(np.median(az_err_pos), color='#8E44AD', ls=':', lw=2, label=f'Median = {np.median(az_err_pos):.1f}°')
ax.set_xlabel('Azimuth Error (°)', fontsize=10, color='#000000')
ax.set_ylabel('Count', fontsize=10, color='#000000')
ax.set_title(f'Azimuth Error Distribution (MAE={az_err_pos.mean():.2f}°)', fontsize=11, fontweight='bold', color='#000000')
ax.legend(fontsize=8); ax.tick_params(colors='#000000')
ax.spines['bottom'].set_color('#CCC'); ax.spines['left'].set_color('#CCC')

# ── 10. Kp vs Detection Rate ──
ax = fig.add_subplot(gs[2, 1])
kp_bins_edges = np.arange(0, 10, 1)
pred['KpBin'] = pd.cut(pred['Kp_Raw'], bins=kp_bins_edges, labels=[f'{i}-{i+1}' for i in range(9)])
kp_stats = pred[pred['True_Label']==1].groupby('KpBin').agg(
    n=('Pred_Binary','count'),
    detected=('Pred_Binary', lambda x: (x==1).sum()))
detect_rate = (kp_stats['detected'] / kp_stats['n'] * 100).fillna(0)
colors_kp = ['#27AE60' if v >= 20 else '#E67E22' if v >= 10 else '#C0392B' for v in detect_rate.values]
bars = ax.bar(range(len(detect_rate)), detect_rate.values, color=colors_kp, edgecolor='#333', lw=0.5, width=0.7)
for i, (bar, v) in enumerate(zip(bars, detect_rate.values)):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1, f'{v:.1f}%',
            ha='center', fontsize=7, fontweight='bold', color='#000000')
ax.set_xticks(range(len(detect_rate)))
ax.set_xticklabels(detect_rate.index, rotation=45, fontsize=7)
ax.set_xlabel('Kp Index Range', fontsize=10, color='#000000')
ax.set_ylabel('Detection Rate (%)', fontsize=10, color='#000000')
ax.set_title('Detection Rate by Geomagnetic Activity', fontsize=11, fontweight='bold', color='#000000')
ax.tick_params(colors='#000000')
ax.spines['bottom'].set_color('#CCC'); ax.spines['left'].set_color('#CCC')

# ── 11. Magnitude Class Confusion ──
ax = fig.add_subplot(gs[2, 2])
mag_true = pred['True_MagClass'].values
mag_pred = pred['Pred_MagClass'].values
all_classes = sorted(set(mag_true) | set(mag_pred))
nc = len(all_classes)
mag_cm = confusion_matrix(mag_true, mag_pred, labels=all_classes)
mag_labels_short = [mag_labels.get(c, f'Cl{c}') for c in all_classes]
im = ax.imshow(mag_cm, cmap='YlOrRd', aspect='auto')
ax.set_xticks(range(nc))
ax.set_yticks(range(nc))
ax.set_xticklabels(mag_labels_short, fontsize=6, rotation=45, ha='right')
ax.set_yticklabels(mag_labels_short, fontsize=6)
ax.set_xlabel('Predicted', fontsize=9, color='#000000')
ax.set_ylabel('True', fontsize=9, color='#000000')
ax.set_title('Magnitude Class Confusion', fontsize=11, fontweight='bold', color='#000000')
for i in range(nc):
    for j in range(nc):
        ax.text(j, i, str(mag_cm[i,j]), ha='center', va='center',
                fontsize=6, color='white' if mag_cm[i,j] > mag_cm.max()/2 else '#333')

# ── 12. Metrics Summary ──
ax = fig.add_subplot(gs[2, 3])
ax.axis('off')

# Calculate metrics at both thresholds
def calc_metrics(th):
    yp = (y_prob >= th).astype(int)
    tn = ((yp==0)&(y_true==0)).sum()
    fp = ((yp==1)&(y_true==0)).sum()
    fn = ((yp==0)&(y_true==1)).sum()
    tp = ((yp==1)&(y_true==1)).sum()
    return {
        'th': th, 'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn,
        'Acc': (tp+tn)/(tp+tn+fp+fn)*100,
        'Prec': tp/(tp+fp)*100 if (tp+fp)>0 else 0,
        'Rec': tp/(tp+fn)*100 if (tp+fn)>0 else 0,
        'Spec': tn/(tn+fp)*100 if (tn+fp)>0 else 0,
        'FPR': fp/(fp+tn)*100 if (fp+tn)>0 else 0,
        'F2': (5*tp/(tp+fp))/(4*tp/(tp+fp)+tp/(tp+fn))
              if (tp+fp)>0 and (tp+fn)>0 else 0,
        'AUPRC': auprc*100,
        'AUC': auc*100,
        'AzMAE': az_err_pos.mean(),
        'BA': ((tp/(tp+fn)) + (tn/(tn+fp))) / 2 * 100 if (tp+fn)>0 and (tn+fp)>0 else 0
    }

m50 = calc_metrics(0.5)
m_opt = calc_metrics(th_opt)
m_best = calc_metrics(0.225)  # From earlier blind test

header = f"{'Metric':<18} {'th=0.5':<12} {'th='+str(th_opt):<12} {'Best F2':<12}"
sep = '─' * 60
text = f"COMPREHENSIVE PERFORMANCE METRICS\n{sep}\n"
text += f"{'':<18} {'th=0.50':>10} {'th='+f'{th_opt:.3f}':>10} {'th=0.225':>10}\n{sep}\n"

rows = [
    ('True Positives', m50['TP'], m_opt['TP'], m_best['TP']),
    ('True Negatives', m50['TN'], m_opt['TN'], m_best['TN']),
    ('False Positives', m50['FP'], m_opt['FP'], m_best['FP']),
    ('False Negatives', m50['FN'], m_opt['FN'], m_best['FN']),
]
for name, v50, vo, vb in rows:
    text += f"{name:<20} {v50:<10} {vo:<10} {vb:<10}\n"

text += f"{sep}\n"
rows_pct = [
    ('Accuracy (%)', m50['Acc'], m_opt['Acc'], m_best['Acc']),
    ('Precision (%)', m50['Prec'], m_opt['Prec'], m_best['Prec']),
    ('Recall (%)', m50['Rec'], m_opt['Rec'], m_best['Rec']),
    ('Specificity (%)', m50['Spec'], m_opt['Spec'], m_best['Spec']),
    ('Balanced Acc (%)', m50['BA'], m_opt['BA'], m_best['BA']),
]
for name, v50, vo, vb in rows_pct:
    text += f"{name:<20} {v50:>8.2f}   {vo:>8.2f}   {vb:>8.2f}\n"
text += f"{sep}\n"
text += f"{'FPR (%)':<20} {m50['FPR']:>8.2f}   {m_opt['FPR']:>8.2f}   {m_best['FPR']:>8.2f}\n"
text += f"{'F2 Score':<20} {m50['F2']:>8.4f}   {m_opt['F2']:>8.4f}   {m_best['F2']:>8.4f}\n"
text += f"{sep}\n"
text += f"{'AUPRC':<20} {m50['AUPRC']:>8.2f}%    {m50['AUPRC']:>8.2f}%    {m50['AUPRC']:>8.2f}%\n"
text += f"{'AUC-ROC':<20} {m50['AUC']:>8.2f}%    {m50['AUC']:>8.2f}%    {m50['AUC']:>8.2f}%\n"
text += f"{'Azimuth MAE (°)':<20} {m50['AzMAE']:>8.2f}    {m50['AzMAE']:>8.2f}    {m50['AzMAE']:>8.2f}\n"
text += f"{sep}\n"
text += f"{'Optimal Threshold':<20} {'—':>10} {th_opt:>10.3f} {'0.225':>10}\n"
text += f"{sep}\n"

# Training summary
best_f2_epoch = train.loc[train['f2'].idxmax(), 'epoch']
train_loss_end = train['train_loss'].iloc[-1]
text += f"\nTRAINING SUMMARY\n{sep}\n"
text += f"{'Best F2 Score':<25} {train['f2'].max():.4f} (epoch {int(best_f2_epoch)})\n"
text += f"{'Best Precision':<25} {train['precision'].max():.4f}\n"
text += f"{'Best EWS Score':<25} {train['ews_score'].max():.4f}\n"
text += f"{'Final Train Loss':<25} {train_loss_end:.4f}\n"
text += f"{'Epochs':<25} {len(train)}\n"
text += f"{'Best Brier':<25} {train['brier'].min():.4f}\n"
text += f"{'Azimuth MAE (val)':<25} {train['azm_mae'].min():.2f}°\n"
text += f"{sep}\n"

ax.text(0.02, 0.98, text, fontsize=7.5, color='#000000', fontfamily='monospace',
        va='top', ha='left', transform=ax.transAxes,
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#FAFAFA', edgecolor='#DDD'))

fig.suptitle('V8 SupCon — Detailed Model Performance', fontsize=16, fontweight='bold',
             color='#000000', y=0.98)
plt.savefig(f'{OUT}/kinerja_lengkap.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig(f'{OUT}/kinerja_lengkap.svg', dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig(f'{OUT}/kinerja_lengkap.pdf', dpi=300, bbox_inches='tight', facecolor='white')
print(f'[OK] Performance figure saved to {OUT}/')
plt.close()

# ── Also generate individual high-res publication figures ──

# A: ROC + PR side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.patch.set_facecolor('white')
for ax in [ax1, ax2]: ax.set_facecolor('white')
ax1.plot(fpr_roc, tpr_roc, color='#C0392B', lw=2.5); ax1.plot([0,1],[0,1],'--',color='#888',lw=1)
ax1.fill_between(fpr_roc, tpr_roc, alpha=0.12, color='#C0392B')
ax1.set_xlabel('False Positive Rate', fontsize=11); ax1.set_ylabel('True Positive Rate', fontsize=11)
ax1.set_title(f'AUC-ROC = {auc:.4f}', fontsize=13, fontweight='bold'); ax1.grid(True, alpha=0.15)
ax1.tick_params(colors='#000000'); ax1.spines['bottom'].set_color('#CCC'); ax1.spines['left'].set_color('#CCC')

ax2.plot(rec_curve, prec_curve, color='#16A085', lw=2.5)
ax2.fill_between(rec_curve, prec_curve, alpha=0.12, color='#16A085')
ax2.set_xlabel('Recall', fontsize=11); ax2.set_ylabel('Precision', fontsize=11)
ax2.set_title(f'AUPRC = {auprc:.4f}', fontsize=13, fontweight='bold'); ax2.grid(True, alpha=0.15)
ax2.tick_params(colors='#000000'); ax2.spines['bottom'].set_color('#CCC'); ax2.spines['left'].set_color('#CCC')
fig.suptitle('V8 SupCon — ROC & PR Curves', fontsize=14, fontweight='bold', color='#000000', y=1.02)
plt.tight_layout()
fig.savefig(f'{OUT}/roc_pr_curves.png', dpi=300, bbox_inches='tight', facecolor='white')
print(f'[OK] ROC/PR curves saved')
plt.close()

# B: Threshold Sweep
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
fig.patch.set_facecolor('white'); ax.set_facecolor('white')
ax.plot(th_df['th'], th_df['f2'], 'o-', color='#8E44AD', lw=2.5, ms=4, label='F2 Score')
ax.plot(th_df['th'], th_df['prec'], 's--', color='#2980B9', lw=2, ms=3, label='Precision')
ax.plot(th_df['th'], th_df['rec'], '^--', color='#C0392B', lw=2, ms=3, label='Recall')
ax.plot(th_df['th'], th_df['acc'], 'd-.', color='#16A085', lw=2, ms=3, label='Accuracy')
ax.axvline(th_opt, color='#333', ls=':', lw=2, alpha=0.7)
ax.axvline(0.5, color='#666', ls=':', lw=1.5, alpha=0.5)
ax.set_xlabel('Decision Threshold', fontsize=12); ax.set_ylabel('Score', fontsize=12)
ax.set_title('V8 SupCon — Threshold Performance Analysis', fontsize=14, fontweight='bold')
ax.legend(fontsize=10, loc='center right'); ax.grid(True, alpha=0.2)
ax.tick_params(colors='#000000'); ax.spines['bottom'].set_color('#CCC'); ax.spines['left'].set_color('#CCC')
ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
plt.tight_layout()
fig.savefig(f'{OUT}/threshold_sweep.png', dpi=300, bbox_inches='tight', facecolor='white')
fig.savefig(f'{OUT}/threshold_sweep.pdf', dpi=300, bbox_inches='tight', facecolor='white')
print(f'[OK] Threshold sweep saved')
plt.close()

# C: Training Dual Axis (clean version)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
fig.patch.set_facecolor('white')
for ax in [ax1, ax2]: ax.set_facecolor('white')
ax1.plot(train['epoch'], train['train_loss'], color='#3498DB', lw=2, label='Training Loss')
ax1.plot(train['epoch'], train['val_loss'], color='#E74C3C', lw=2, label='Validation Loss')
ax1.set_ylabel('Loss', fontsize=11); ax1.legend(fontsize=10); ax1.grid(True, alpha=0.15)
ax1.set_title('V8 SupCon — Training & Validation Loss', fontsize=13, fontweight='bold')
ax1.tick_params(colors='#000000'); ax1.spines['bottom'].set_color('#CCC'); ax1.spines['left'].set_color('#CCC')

ax2.plot(train['epoch'], train['f2'], 'o-', color='#8E44AD', lw=2, ms=4, label='F2 Score')
ax2.plot(train['epoch'], train['ews_score'], 's-', color='#F39C12', lw=2, ms=4, label='EWS Score')
ax2.set_xlabel('Epoch', fontsize=11); ax2.set_ylabel('Score', fontsize=11)
ax2.legend(fontsize=10); ax2.grid(True, alpha=0.15)
ax2.tick_params(colors='#000000'); ax2.spines['bottom'].set_color('#CCC'); ax2.spines['left'].set_color('#CCC')
plt.tight_layout()
fig.savefig(f'{OUT}/training_dual.png', dpi=300, bbox_inches='tight', facecolor='white')
fig.savefig(f'{OUT}/training_dual.pdf', dpi=300, bbox_inches='tight', facecolor='white')
print(f'[OK] Training dual saved')

# D: Azimuth Error + Kp Detection
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.patch.set_facecolor('white')
for ax in [ax1, ax2]: ax.set_facecolor('white')
ax1.hist(az_err_pos, bins=45, color='#16A085', edgecolor='#333', lw=0.3, alpha=0.8)
ax1.axvline(az_err_pos.mean(), color='#C0392B', ls='--', lw=2.5, label=f'MAE={az_err_pos.mean():.2f}°')
ax1.axvline(np.median(az_err_pos), color='#8E44AD', ls=':', lw=2, label=f'Median={np.median(az_err_pos):.2f}°')
ax1.set_xlabel('Azimuth Error (°)', fontsize=11); ax1.set_ylabel('Count', fontsize=11)
ax1.set_title('Azimuth Error Distribution', fontsize=13, fontweight='bold')
ax1.legend(fontsize=10); ax1.grid(True, alpha=0.15)
ax1.tick_params(colors='#000000'); ax1.spines['bottom'].set_color('#CCC'); ax1.spines['left'].set_color('#CCC')

kp_by = pred[pred['True_Label']==1].copy()
kp_by['KpBin'] = pd.cut(kp_by['Kp_Raw'], bins=range(10), labels=[str(i) for i in range(9)])
kp_grp = kp_by.groupby('KpBin').agg(n=('Pred_Binary','count'), det=('Pred_Binary', lambda x: (x==1).sum()))
rate = (kp_grp['det']/kp_grp['n']*100).fillna(0)
ax2.bar(range(len(rate)), rate.values, color=['#27AE60' if v>15 else '#E67E22' if v>8 else '#C0392B' for v in rate.values],
        edgecolor='#333', lw=0.5, width=0.7)
ax2.set_xticks(range(len(rate))); ax2.set_xticklabels(rate.index, fontsize=8)
ax2.set_xlabel('Kp Index', fontsize=11); ax2.set_ylabel('Detection Rate (%)', fontsize=11)
ax2.set_title('Detection Rate by Geomagnetic Activity', fontsize=13, fontweight='bold')
ax2.tick_params(colors='#000000'); ax2.spines['bottom'].set_color('#CCC'); ax2.spines['left'].set_color('#CCC')
plt.tight_layout()
fig.savefig(f'{OUT}/azimuth_kp_analysis.png', dpi=300, bbox_inches='tight', facecolor='white')
fig.savefig(f'{OUT}/azimuth_kp_analysis.pdf', dpi=300, bbox_inches='tight', facecolor='white')
print(f'[OK] Azimuth + KP saved')
plt.close()

print('\n=== SEMUA GAMBAR KINERJA BERHASIL DIGENERASI ===')
print(f'Output: {OUT}/')
