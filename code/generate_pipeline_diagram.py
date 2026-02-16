#!/usr/bin/env python3
"""
Generate the Overall Thesis Pipeline Diagram (Figure 1).
Visualizes: Raw Traffic -> BiMamba Teacher -> Knowledge Distillation -> UniMamba Student -> TED Inference
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

def create_pipeline_diagram():
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # Styles
    box_style = dict(boxstyle="round,pad=0.3", ec="black", lw=1.5)
    
    # ── 1. Input Data ──
    rect_input = patches.Rectangle((0.5, 3), 2, 2, linewidth=1.5, edgecolor='#34495E', facecolor='#ECF0F1')
    ax.add_patch(rect_input)
    ax.text(1.5, 4.5, "Raw Traffic\n(PCAP)", ha="center", va="center", fontsize=10, fontweight='bold')
    # Packets
    for i in range(4):
        p = patches.Rectangle((0.8 + i*0.35, 3.5), 0.25, 0.5, facecolor='#3498DB', edgecolor='black')
        ax.add_patch(p)
    ax.text(1.5, 3.2, "32-Packet Window", ha="center", fontsize=9, style='italic')

    # Arrow Input -> Models
    ax.annotate("", xy=(3, 5.5), xytext=(2.5, 4), arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.annotate("", xy=(3, 2.5), xytext=(2.5, 4), arrowprops=dict(arrowstyle="->", lw=1.5))

    # ── 2. Teacher (BiMamba) - Top Path ──
    # Training Loop Box
    rect_teacher = patches.FancyBboxPatch((3, 5), 3, 2.5, boxstyle="round,pad=0.1", ec='#E67E22', fc='#FDEBD0')
    ax.add_patch(rect_teacher)
    ax.text(4.5, 7.2, "TEACHER (BiMamba)", ha="center", fontweight='bold', color='#D35400')
    
    # Stage 1: SSL
    ax.text(4.5, 6.5, "1. SSL Pre-training\n(Masked Modeling)", ha="center", fontsize=9, 
            bbox=dict(boxstyle="round", fc="white", ec="#D35400"))
    # Stage 2: Finetune
    ax.text(4.5, 5.5, "2. Fine-tuning\n(100% Labeled Data)", ha="center", fontsize=9, 
            bbox=dict(boxstyle="round", fc="white", ec="#D35400"))

    # ── 3. Student (UniMamba) - Bottom Path ──
    rect_student = patches.FancyBboxPatch((3, 0.5), 3, 2.5, boxstyle="round,pad=0.1", ec='#27AE60', fc='#D5F5E3')
    ax.add_patch(rect_student)
    ax.text(4.5, 2.7, "STUDENT (UniMamba)", ha="center", fontweight='bold', color='#1E8449')
    
    # Architecture
    ax.text(4.5, 1.8, "UniMamba Encoder\n(Causal SSM)", ha="center", fontsize=9, 
            bbox=dict(boxstyle="round", fc="white", ec="#1E8449"))
    ax.text(4.5, 1.0, "Input: 10% Labeled Data", ha="center", fontsize=8, style='italic')

    # ── 4. Knowledge Distillation (The Link) ──
    ax.annotate("Knowledge Distillation\n(Soft Targets)", xy=(4.5, 3.1), xytext=(4.5, 4.9),
                ha="center", fontsize=9, fontweight='bold', color='purple',
                arrowprops=dict(arrowstyle="->", lw=2, color='purple', linestyle='dashed'))

    # Arrow Student -> Inference
    ax.annotate("", xy=(6.5, 1.75), xytext=(6.1, 1.75), arrowprops=dict(arrowstyle="->", lw=1.5))

    # ── 5. TED Inference (Blockwise) ──
    rect_ted = patches.FancyBboxPatch((6.5, 0.5), 5.5, 2.5, boxstyle="round,pad=0.1", ec='#8E44AD', fc='#EBDEF0')
    ax.add_patch(rect_ted)
    ax.text(9.25, 2.7, "Inference: Blockwise TED", ha="center", fontweight='bold', color='#6C3483')
    
    # Block 1 (Packet 8)
    b1 = patches.Rectangle((6.8, 1.2), 1.2, 1, fc='white', ec='black')
    ax.add_patch(b1)
    ax.text(7.4, 1.7, "Block 1\n(8 pkts)", ha="center", fontsize=8)
    ax.text(7.4, 0.9, "Confidence?\n(95% Exit)", ha="center", fontsize=7, color='red')
    
    # Block 2 (Packet 16)
    b2 = patches.Rectangle((8.6, 1.2), 1.2, 1, fc='white', ec='black')
    ax.add_patch(b2)
    ax.text(9.2, 1.7, "Block 2\n(16 pkts)", ha="center", fontsize=8)
    ax.text(9.2, 0.9, "Confidence?\n(1.3% Exit)", ha="center", fontsize=7, color='red')

    # Block 3 (Packet 32)
    b3 = patches.Rectangle((10.4, 1.2), 1.2, 1, fc='white', ec='black')
    ax.add_patch(b3)
    ax.text(11.0, 1.7, "Block 3\n(32 pkts)", ha="center", fontsize=8)
    ax.text(11.0, 0.9, "Final Exit\n(3.7% Exit)", ha="center", fontsize=7, color='red')

    # Arrows between blocks
    ax.annotate("", xy=(8.6, 1.7), xytext=(8.0, 1.7), arrowprops=dict(arrowstyle="->"))
    ax.annotate("", xy=(10.4, 1.7), xytext=(9.8, 1.7), arrowprops=dict(arrowstyle="->"))

    # ── 6. Final Output ──
    ax.annotate("", xy=(12.5, 1.7), xytext=(12.0, 1.7), arrowprops=dict(arrowstyle="->", lw=1.5))
    
    # Output Box
    bbox_props = dict(boxstyle="darrow,pad=0.3", fc="#F1C40F", ec="black", lw=1)
    ax.text(13.2, 1.7, "Attack / Benign", ha="center", va="center", rotation=0,
            size=10, bbox=bbox_props, fontweight='bold')

    plt.title("Full Thesis Pipeline: From SSL Teacher to Efficient Blockwise Student", fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, '00_overall_pipeline.png'), dpi=150)
    print("✅ Pipeline Diagram Saved")

if __name__ == "__main__":
    create_pipeline_diagram()
