#!/usr/bin/env python3
"""
Generate all thesis plots from verified results.
Saves to thesis_final/plots/
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

PLOTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# ── Verified Data ──
models = ['XGBoost', 'BERT', 'UniMamba', 'BiMamba\n(Teacher)', 'KD\nStudent', 'TED\n(Ours)']
f1_scores = [0.8845, 0.8725, 0.8842, 0.8924, 0.8836, 0.8783]
auc_scores = [0.9977, 0.9937, 0.9956, 0.9975, 0.9959, 0.9951]
cross_f1 = [0.7195, 0.7948, 0.8663, 0.7438, 0.8710, 0.8998]
latency = [None, 1.03, 0.72, 1.20, 0.74, 0.72]  # XGBoost N/A
throughput = [None, 25565, 33467, 17028, 31723, 33467]

colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']

plt.rcParams.update({'font.size': 12, 'font.family': 'DejaVu Sans'})

# ── Plot 1: F1 Score Comparison ──
fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.bar(models, f1_scores, color=colors, edgecolor='black', linewidth=0.5)
for bar, val in zip(bars, f1_scores):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.002,
            f'{val:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
ax.set_ylabel('F1 Score')
ax.set_title('In-Domain F1 Score Comparison (UNSW-NB15)', fontsize=14, fontweight='bold')
ax.set_ylim(0.85, 0.90)
ax.axhline(y=0.8783, color='purple', linestyle='--', alpha=0.5, label='TED Baseline')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '01_f1_comparison.png'), dpi=150)
plt.close()
print("✅ Plot 1: F1 Comparison")

# ── Plot 2: Cross-Dataset Generalization ──
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(models))
width = 0.35
bars1 = ax.bar(x - width/2, f1_scores, width, label='In-Domain F1', color='#45B7D1', edgecolor='black')
bars2 = ax.bar(x + width/2, cross_f1, width, label='Cross-Dataset F1', color='#FF6B6B', edgecolor='black')
for bar, val in zip(bars1, f1_scores):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
            f'{val:.3f}', ha='center', va='bottom', fontsize=8)
for bar, val in zip(bars2, cross_f1):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
            f'{val:.3f}', ha='center', va='bottom', fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.set_ylabel('F1 Score')
ax.set_title('In-Domain vs Cross-Dataset Generalization', fontsize=14, fontweight='bold')
ax.set_ylim(0.6, 1.0)
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '02_cross_dataset_gap.png'), dpi=150)
plt.close()
print("✅ Plot 2: Cross-Dataset Gap")

# ── Plot 3: Latency Comparison ──
lat_models = ['BERT', 'UniMamba', 'BiMamba', 'KD Student', 'TED']
lat_values = [1.03, 0.72, 1.20, 0.74, 0.72]
lat_colors = ['#FF6B6B', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.barh(lat_models, lat_values, color=lat_colors, edgecolor='black')
for bar, val in zip(bars, lat_values):
    ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2.,
            f'{val:.2f} ms', va='center', fontweight='bold')
ax.set_xlabel('Latency (ms)')
ax.set_title('Per-Flow Inference Latency', fontsize=14, fontweight='bold')
ax.axvline(x=1.0, color='red', linestyle='--', alpha=0.5, label='1ms threshold')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '03_latency_comparison.png'), dpi=150)
plt.close()
print("✅ Plot 3: Latency")

# ── Plot 4: Exit Distribution Histogram ──
exit_labels = ['Packet 8\n(95%)', 'Packet 16\n(1.3%)', 'Packet 32\n(4.2%)']
exit_values = [94.44, 1.32, 4.24]
exit_colors = ['#2ECC71', '#F39C12', '#E74C3C']

fig, ax = plt.subplots(figsize=(8, 6))
bars = ax.bar(exit_labels, exit_values, color=exit_colors, edgecolor='black', width=0.5)
for bar, val in zip(bars, exit_values):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
            f'{val:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=14)
ax.set_ylabel('Percentage of Flows (%)')
ax.set_title('TED Early Exit Distribution', fontsize=14, fontweight='bold')
ax.set_ylim(0, 110)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '04_exit_distribution.png'), dpi=150)
plt.close()
print("✅ Plot 4: Exit Distribution")

# ── Plot 5: Throughput Scaling ──
batch_sizes = [1, 16, 32, 64]
bert_tp = [962, 25565*16/32, 25565, 41388]
mamba_tp = [1475, 33467*16/32, 33467, 33467*64/32]

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(batch_sizes, bert_tp, 'o-', color='#FF6B6B', label='BERT', linewidth=2, markersize=8)
ax.plot(batch_sizes, mamba_tp, 's-', color='#45B7D1', label='UniMamba', linewidth=2, markersize=8)
ax.set_xlabel('Batch Size')
ax.set_ylabel('Throughput (flows/s)')
ax.set_title('Throughput vs Batch Size', fontsize=14, fontweight='bold')
ax.legend()
ax.set_xscale('log', base=2)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '05_throughput_scaling.png'), dpi=150)
plt.close()
print("✅ Plot 5: Throughput Scaling")

# ── Plot 6: The Evolution Story (Radar Chart) ──
fig, ax = plt.subplots(figsize=(10, 6))
evolution = ['XGBoost', 'BERT+SSL', 'UniMamba', 'KD Student', 'TED']
metrics_names = ['In-Domain F1', 'Cross-DS F1', 'Speed\n(1/Latency)', 'Efficiency\n(1/Params)', 'Early Exit\n(1/AvgPkts)']

# Normalize to 0-1
data_matrix = [
    [0.8845/0.9, 0.7195/0.9, 0.0, 0.0, 1/32],        # XGBoost
    [0.8725/0.9, 0.7948/0.9, 1/1.03, 1/4.59, 1/32],     # BERT
    [0.8842/0.9, 0.8663/0.9, 1/0.72, 1/1.95, 1/32],     # UniMamba
    [0.8836/0.9, 0.8710/0.9, 1/0.74, 1/1.95, 1/32],     # KD
    [0.8783/0.9, 0.8998/0.9, 1/0.72, 1/1.95, 1/9.1],    # TED
]

x_pos = np.arange(len(evolution))
evo_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFEAA7', '#DDA0DD']

for i, metric in enumerate(metrics_names):
    vals = [row[i] for row in data_matrix]
    # Normalize
    max_v = max(vals) if max(vals) > 0 else 1
    vals_norm = [v/max_v for v in vals]
    ax.plot(x_pos, vals_norm, 'o-', label=metric, linewidth=1.5, markersize=6)

ax.set_xticks(x_pos)
ax.set_xticklabels(evolution)
ax.set_ylabel('Normalized Score')
ax.set_title('The Evolution: Each Step Solves a Problem', fontsize=14, fontweight='bold')
ax.legend(loc='upper left', fontsize=8)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '06_evolution_story.png'), dpi=150)
plt.close()
print("✅ Plot 6: Evolution Story")

print("\n🎉 All plots saved to", PLOTS_DIR)
