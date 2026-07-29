#!/usr/bin/env python3
"""
Diagram Arsitektur V8 SUPCON — Paper Internasional
=====================================================
Output: bab4_diagram_arsitektur_v8.png (300 dpi, 16:9)
"""
import os, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# Colors
W = '#FFFFFF'; BG='#FAFBFC'; TXT='#1A1A2E'; GR='#6B7280'
BLU='#2563EB'; BLUL='#DBEAFE'; GRN='#059669'; GRNL='#D1FAE5'
ORG='#EA580C'; ORGL='#FFEDD5'; PPL='#7C3AED'; PPLL='#EDE9FE'
RED='#DC2626'; REDL='#FEE2E2'; TQL='#CCFBF1'; TQM='#0D9488'
CYNL='#CFFAFE'; GLD='#D97706'; GLDL='#FEF3C7'; GRYL='#F3F4F6'

def b(ax,x,y,w,h,t,c,e,tc,fs=8,bs=6):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.08",
        facecolor=c,edgecolor=e,linewidth=1.6,zorder=3))
    ax.text(x+w/2,y+h-0.12,t,ha='center',va='top',fontsize=fs,
            fontweight='bold',color=tc,zorder=4)
    ax.text(x+w/2,y+h/2-0.06,c if not isinstance(c,str) else '',ha='center',va='center')
    ax.text(x+w/2,y+h/2-0.08,s if (s:=c) else '',ha='center',va='center',
            fontsize=bs,color='#374151',zorder=4,linespacing=1.3)

def b2(ax,x,y,w,h,title,body,fc,ec,tc,fs=8,bs=6):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.08",
        facecolor=fc,edgecolor=ec,linewidth=1.6,zorder=3))
    ax.text(x+w/2,y+h-0.12,title,ha='center',va='top',fontsize=fs,
            fontweight='bold',color=tc,zorder=4)
    ax.text(x+w/2,y+h/2-0.08,body,ha='center',va='center',
            fontsize=bs,color='#374151',zorder=4,linespacing=1.3)

def ar(ax,x1,y1,x2,y2,c='#6B7280',lw=1.4,r=0):
    ax.annotate('',xy=(x2,y2),xytext=(x1,y1),
        arrowprops=dict(arrowstyle='->',color=c,lw=lw,connectionstyle=f'arc3,rad={r}'),zorder=2)

def main():
    fig,ax = plt.subplots(1,1,figsize=(18,12))
    ax.set_xlim(0,18); ax.set_ylim(0,12); ax.axis('off')
    ax.set_facecolor(BG)

    # TITLE
    ax.text(9,11.6,'ScalogramV3 V8 — Multi-Task Deep Learning Architecture',
        ha='center',fontsize=16,fontweight='bold',color='#1A237E')
    ax.text(9,11.25,'Supervised Contrastive Learning (SupCon) + Spatial GNN + EfficientNet-B1 Backbone',
        ha='center',fontsize=9,color=GR,style='italic')

    # ========== ROW 1: INPUT ==========
    y1=10.2; h1=0.8
    b2(ax,0.5,y1,3.5,h1,'INPUT: CWT Skalogram',
       'x_img: (B, 3, 128, 1440)\nKanal [H] Horizontal  [D] Deklinasi  [Z] Vertikal\n24 jam data magnetik @ 1 Hz sampling rate',
       PPLL,PPL,PPL,8.5,6)
    b2(ax,4.5,y1,2.8,h1,'INPUT: Indeks Kosmik',
       'x_cosmic: (B, 2)\n[Kp_raw, Dst_raw]\nIndeks aktivitas geomagnet global',
       PPLL,PPL,PPL,8.5,6)
    ar(ax,2.25,y1,2.25,9.6)
    ar(ax,5.9,y1,5.9,9.6)

    # ========== ROW 2: BACKBONE ==========
    y2=8.8; h2=1.1
    b2(ax,0.5,y2,3.5,h2,'EfficientNet-B1 (Frozen)',
       'ImageNet Pre-trained — ~7,86M param (non-trainable)\nTransfer Learning: Visual → Skalogram\nFeature Map Output: (B, 1280, H, W)',
       BLUL,BLU,BLU,9,6.5)

    # ========== ROW 3: AdaptivePool ==========
    y3=7.6; h3=0.55
    b2(ax,0.5,y3,3.5,h3,'AdaptiveAvgPool2d',
       'Spatial Squeeze → (B, 1280, 1, 45)',
       BLUL,BLU,BLU,8,6)
    ar(ax,2.25,8.8,2.25,8.15)

    # ========== ROW 4: GRU Projection ==========
    y4=6.7; h4=0.55
    b2(ax,0.5,y4,3.5,h4,'GRU Projection Layer',
       'Linear 1280 → 256 — VRAM optimization layer',
       BLUL,BLU,BLU,8,6)
    ar(ax,2.25,7.6,2.25,7.25)

    # ========== ROW 5: BiGRU ==========
    y5=5.4; h5=1.0
    b2(ax,0.5,y5,3.5,h5,'BiGRU — Temporal Encoding',
       '2 Layers Bidirectional Gated Recurrent Unit\nHidden 256 → 512  |  Dropout 0.2\nOutput: (B, 45, 512)\nMemodelkan dependensi temporal sinyal geomagnetik',
       BLUL,BLU,BLU,9,6.5)
    ar(ax,2.25,6.7,2.25,6.4)

    # ========== ROW 6: Global Pooling ==========
    y6=4.5; h6=0.5
    b2(ax,0.5,y6,3.5,h6,'Global Temporal Pooling',
       'mean(dim=1) → v_img: (B, 512)',
       BLUL,BLU,BLU,8,6)
    ar(ax,2.25,5.4,2.25,5.0)

    # ========== COSMIC BRANCH ==========
    b2(ax,4.5,8.4,2.8,0.9,'Cosmic MLP',
       'Linear 2 → 32 → 512\nLayerNorm + ReLU + Dropout(0.1)\nSigmoid → Attention Gate',
       GRNL,GRN,GRN,8.5,6)
    ar(ax,5.9,10.2,5.9,9.3)

    # ========== SPATIAL GNN ==========
    b2(ax,4.5,5.4,3.2,1.6,'Spatial Graph Neural Network (GNN)',
       '8 Virtual Stations (Regional Node Features)\nMulti-Head Attention — 4 Heads\nGraph Attention Network (GAT) + Residual\nin=512 → hidden=256 → out=512\nOutput: consensus_feat (B, 512)',
       ORGL,ORG,ORG,9,6.5)
    ar(ax,4.0,5.2,4.5,5.2,r=0.12)
    ar(ax,5.9,8.4,5.9,7.2)
    ax.text(6.15,7.7,'attention',fontsize=6,color=GRN,fontweight='bold')

    # ========== FUSION (SoftPhysicsGate) ==========
    b2(ax,4.5,3.8,3.2,0.7,'SoftPhysicsGate — Element-wise Fusion',
       'v_fusion = consensus_feat ⊙ cosmic_attention\nOutput: (B, 512)',
       TQL,TQM,GRN,8.5,6)
    ar(ax,4.0,4.0,4.5,4.0)
    ar(ax,5.9,5.4,5.9,4.5)

    # ========== ROW 7: TASK HEADS ==========
    yh=2.6; hh=0.85; wh=2.0; gap=0.18
    heads=[
        ('Detection Head','512→128→2 (logits)\n[normal, prekursor]\nSigmoid + Threshold 0.5'),
        ('Magnitude Head','512→128→5 (logits)\n5 Bin Magnitude\nCross-Entropy Loss'),
        ('Azimuth Head','512→128→2\n[sin(θ), cos(θ)]\nUnit Vector Regression'),
        ('Projection Head\n(SupCon)','512→128→L2-norm\nEmbedding 128-d\nSupervised Contrastive\nNT-Xent Loss'),
    ]
    xs=[]
    for i,(nm,body) in enumerate(heads):
        x0=0.5+i*(wh+gap); xs.append(x0+wh/2)
        b2(ax,x0,yh,wh,hh,nm,body,REDL,RED,RED,8,6)

    for xc in xs: ar(ax,xc,3.8,xc,yh+hh)

    # ========== LOSS FUNCTIONS ==========
    yl=1.3; hl=0.7
    losses=['L_det\nλ_det=1.0\nWeighted BCE',
            'L_mag\nλ_mag=0.5\nCross-Entropy',
            'L_az\nλ_az=0.3\nMSE',
            'L_con\nλ_con=1.0\nSupCon (NT-Xent)']
    for i,(xc,ls) in enumerate(zip(xs,losses)):
        b2(ax,xc-0.85,yl,1.7,hl,'',ls,GLDL,GLD,GLD,6,5)
        ar(ax,xc,yl+hl,xc,yl+hl-0.15,GR,0.8)

    # TOTAL LOSS
    b2(ax,xs[-1]+1.1,yl-0.15,2.6,hl+0.15,'TOTAL LOSS',
       'L_total = λ_det L_det + λ_mag L_mag\n+ λ_az L_az + λ_con L_con',
       GLDL,GLD,GLD,8,6)
    ar(ax,xs[-1]+0.85,yl+hl/2,xs[-1]+1.1,yl+hl/2-0.15)

    # ========== RIGHT PANEL: SPECS ==========
    spec_body = (
        'Parameter:\n'
        '  Total: ~8,69 Juta\n'
        '  Trainable: ~0,83 Juta\n'
        '  Frozen Backbone: ~7,86 Juta\n\n'
        'Input Tensor:\n'
        '  (79, 168, 3) → resize 240×240\n'
        '  CWT Morlet — 79 skala, 168 waktu\n\n'
        'Dataset:\n'
        '  37.000 sampel (12,2% prekursor)\n'
        '  HDF5 — scalogram_v3_cosmic_final.h5\n\n'
        'Performa (Test Set):\n'
        '  ROC-AUC:  0,9949\n'
        '  Recall:    0,8688\n'
        '  Precision: 0,9564\n'
        '  F1-Score:  0,9105\n'
        '  FAR:       1,6/bulan'
    )
    b2(ax,10.5,7.8,3.5,3.2,'Spesifikasi Model V8 SUPCON',
       spec_body,'#F5F3FF','#7C3AED',PPL,9,6.5)

    # ========== DEPLOYMENT ==========
    dep_body = (
        'Inferensi Real-time:\n'
        '  CPU:  < 100 ms/sampel\n'
        '  GPU:  batch 32 → 15 ms (T4)\n\n'
        'Operational:\n'
        '  Self-updating: Rolling Window 48 bln\n'
        '  Trigger: AUC turun >5% selama 3 hr\n'
        '  24 stasiun MAGDAS-BMKG real-time\n\n'
        'Docker: FastAPI + Monitoring Dashboard\n'
        'Target: BMKG Indonesia — EEWS'
    )
    b2(ax,10.5,5.1,3.5,2.5,'Deployment Operasional',
       dep_body,GRNL,GRN,GRN,9,6)

    ar(ax,4.0,6.5,10.5,6.5,GR,0.8,0.12)
    ar(ax,10.5,2.6,10.0,2.6,GR,0.8,0.15)

    # ========== REGIONAL & ATTENTION OUTPUTS ==========
    b2(ax,8.5,5.4,1.8,0.55,'Regional Consensus',
       'reg_score: ≥3 stasiun → alarm',CYNL,TQM,TQM,7,5.5)
    b2(ax,8.5,4.6,1.8,0.55,'Attention Weights',
       'att_weights: (B, 8)',CYNL,TQM,TQM,7,5.5)
    ar(ax,7.7,5.5,8.5,5.65,r=0.1)
    ar(ax,7.7,4.7,8.5,4.85,r=0.1)

    # ========== LEGEND ==========
    leg=[
        mpatches.Patch(facecolor=BLUL,edgecolor=BLU,label='Spatial Feature Extractor (EfficientNet-B1)'),
        mpatches.Patch(facecolor=GRNL,edgecolor=GRN,label='Cosmic Injection / Fusion (SoftPhysicsGate)'),
        mpatches.Patch(facecolor=ORGL,edgecolor=ORG,label='Graph Neural Network (Spatial GNN)'),
        mpatches.Patch(facecolor=PPLL,edgecolor=PPL,label='Input Layer (CWT Skalogram + Cosmic)'),
        mpatches.Patch(facecolor=REDL,edgecolor=RED,label='Task-Specific Head (Multi-Task Learning)'),
        mpatches.Patch(facecolor=CYNL,edgecolor=TQM,label='Operational Output'),
        mpatches.Patch(facecolor=GLDL,edgecolor=GLD,label='Loss Function'),
    ]
    ax.legend(handles=leg,loc='lower center',bbox_to_anchor=(9,0.65),
              ncol=2,fontsize=7,framealpha=0.95,edgecolor='#D1D5DB',
              title='Komponen Arsitektur V8 SUPCON',title_fontsize=8)

    ax.text(9,0.05,
        'ScalogramV3 V8 — Efektif 20 April 2026 | Teknik Fisika ITS × BMKG | IEEE/AGU JGR: Solid Earth',
        ha='center',fontsize=6.5,color='#9CA3AF',style='italic')

    out=os.path.join(OUTPUT_DIR,'bab4_diagram_arsitektur_v8.png')
    fig.subplots_adjust(left=0.02,right=0.98,top=0.97,bottom=0.05)
    plt.savefig(out,dpi=300,facecolor=BG,edgecolor='none')
    plt.close()
    print(f"[OK] Diagram arsitektur V8: {out}")

if __name__=='__main__':
    main()
