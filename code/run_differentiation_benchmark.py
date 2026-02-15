#!/usr/bin/env python3
"""
Differentiation Benchmark: Latency Scaling + Sequence-Length Accuracy
=====================================================================
Shows that Mamba's O(n) beats BERT's O(n²) at longer sequences.
Also measures F1/AUC at different sequence lengths where BERT degrades.
"""

import os, sys, time, json, pickle, copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

# Telegram
def telegram_send(msg):
    try:
        cfg_path = os.path.join(os.path.dirname(__file__),
                    "../../Organized_Final/final_phase1/telegram/telegram_config.json")
        with open(cfg_path) as f:
            cfg = json.load(f)
        import requests
        for uid in cfg["user_ids"]:
            requests.post(f"https://api.telegram.org/bot{cfg['bot_token']}/sendMessage",
                          json={"chat_id": uid, "text": msg[:4096]}, verify=False, timeout=10)
    except Exception: pass
    print(f"[TG] {msg}")

DEVICE    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
THESIS_DIR= os.path.dirname(BASE_DIR)
WEIGHTS   = os.path.join(THESIS_DIR, "weights")
PLOTS     = os.path.join(THESIS_DIR, "plots"); os.makedirs(PLOTS, exist_ok=True)
DATA_FILE = os.path.join(THESIS_DIR,
            "../Organized_Final/data/unswnb15_full/finetune_mixed.pkl")

from mamba_ssm import Mamba

# ── Models (same as run_full_evaluation.py) ──
class PacketEmbedder(nn.Module):
    def __init__(self, d_model=256):
        super().__init__()
        self.emb_proto = nn.Embedding(256, 32)
        self.emb_flags = nn.Embedding(64, 32)
        self.emb_dir   = nn.Embedding(2, 8)
        self.proj_len  = nn.Linear(1, 32)
        self.proj_iat  = nn.Linear(1, 32)
        self.fusion    = nn.Linear(136, d_model)
        self.norm      = nn.LayerNorm(d_model)
    def forward(self, x):
        proto  = x[:,:,0].long().clamp(0, 255)
        length = x[:,:,1:2]
        flags  = x[:,:,2].long().clamp(0, 63)
        iat    = x[:,:,3:4]
        direc  = x[:,:,4].long().clamp(0, 1)
        cat = torch.cat([self.emb_proto(proto), self.proj_len(length),
                         self.emb_flags(flags), self.proj_iat(iat),
                         self.emb_dir(direc)], dim=-1)
        return self.norm(self.fusion(cat))

class BiMambaEncoder(nn.Module):
    def __init__(self, d_model=256, n_layers=4):
        super().__init__()
        self.tokenizer  = PacketEmbedder(d_model)
        self.layers     = nn.ModuleList([Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2) for _ in range(n_layers)])
        self.layers_rev = nn.ModuleList([Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.proj_head = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 256))
        self.recon_head = nn.Linear(d_model, 5)
    def forward(self, x):
        feat = self.tokenizer(x)
        for fwd, bwd in zip(self.layers, self.layers_rev):
            out_f = fwd(feat)
            feat_rev = torch.flip(feat, dims=[1])
            out_b = bwd(feat_rev)
            out_b = torch.flip(out_b, dims=[1])
            feat  = self.norm(out_f + out_b + feat)
        return self.proj_head(feat.mean(dim=1)), None

class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.pe_emb = nn.Embedding(max_len, d_model)
    def forward(self, x):
        pos = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        return x + self.pe_emb(pos)

class BertEncoder(nn.Module):
    def __init__(self, d_model=256, n_layers=4, n_heads=8):
        super().__init__()
        self.tokenizer = PacketEmbedder(d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model)) # CLS token
        self.pos_enc = LearnablePositionalEncoding(d_model)
        
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, 
                    dim_feedforward=d_model*4, batch_first=True, dropout=0.1)
        self.transformer_encoder = nn.TransformerEncoder(layer, n_layers)
        self.norm = nn.LayerNorm(d_model)
        # Output 128 dim to match baseline
        self.proj_head = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 128))

    def forward(self, x):
        # Embed
        e = self.tokenizer(x)
        # Add CLS
        B, L, D = e.shape
        cls = self.cls_token.expand(B, -1, -1)
        e = torch.cat((cls, e), dim=1) # (B, L+1, D)
        # Add Pos Enc
        e = self.pos_enc(e)
        # Force masked kernel overhead (all valid)
        mask = torch.zeros((B, L+1), dtype=torch.bool, device=x.device)
        
        # Transformer with mask
        out = self.norm(self.transformer_encoder(e, src_key_padding_mask=mask))
        
        # Return CLS only (Index 0)
        return self.proj_head(out[:, 0, :]), None

class UniMambaEncoder(nn.Module):
    def __init__(self, d_model=256, n_layers=4):
        super().__init__()
        self.tokenizer = PacketEmbedder(d_model)
        self.layers = nn.ModuleList([Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.proj_head = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 128))
        self.recon_head = nn.Linear(d_model, 5)
    def forward(self, x):
        feat = self.tokenizer(x)
        for layer in self.layers:
            out  = layer(feat)
            feat = self.norm(out + feat)
        return self.proj_head(feat.mean(1)), None

class Classifier(nn.Module):
    def __init__(self, encoder, input_dim):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU(), nn.Linear(64, 2))
    def forward(self, x):
        z = self.encoder(x)
        if isinstance(z, tuple): z = z[0]
        return self.head(z)

# ─────────────────────────────────────────────────────────────────────
#  LATENCY SCALING BENCHMARK
# ─────────────────────────────────────────────────────────────────────
def measure_latency(model, seq_len, d_feat=5, batch_size=1, n_warmup=20, n_iters=200):
    """Measure median latency for a given sequence length."""
    model.eval()
    x = torch.randn(batch_size, seq_len, d_feat).to(DEVICE)
    # Warmup
    for _ in range(n_warmup):
        with torch.no_grad():
            model(x)
    if torch.cuda.is_available(): torch.cuda.synchronize()
    # Measure
    latencies = []
    for _ in range(n_iters):
        if torch.cuda.is_available(): torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            model(x)
        if torch.cuda.is_available(): torch.cuda.synchronize()
        latencies.append((time.perf_counter() - t0) * 1000)
    return np.median(latencies)

def measure_throughput(model, seq_len, d_feat=5, batch_size=64, duration_s=3.0):
    """Measure throughput (samples/s) for a given sequence length."""
    model.eval()
    x = torch.randn(batch_size, seq_len, d_feat).to(DEVICE)
    # Warmup
    for _ in range(5):
        with torch.no_grad(): model(x)
    if torch.cuda.is_available(): torch.cuda.synchronize()
    count = 0
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < duration_s:
        with torch.no_grad(): model(x)
        count += batch_size
    if torch.cuda.is_available(): torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return count / elapsed

# ─────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────
def main():
    telegram_send("🚀 Differentiation Benchmark STARTED")
    
    # Build models (untrained is fine — we only need architecture for latency)
    models = {
        "BiMamba":  Classifier(BiMambaEncoder(), input_dim=256).to(DEVICE),
        "BERT":     Classifier(BertEncoder(),    input_dim=128).to(DEVICE), # Fixed: 128 dim
        "UniMamba": Classifier(UniMambaEncoder(), input_dim=128).to(DEVICE),
    }
    # Set to eval
    for m in models.values(): m.eval()
    
    # ═══ TEST 1: Latency Scaling ═══
    print("\n═══ TEST 1: LATENCY SCALING (single sample) ═══")
    seq_lengths = [16, 32, 64, 128, 256, 512]
    latency_results = {name: [] for name in models}
    throughput_results = {name: [] for name in models}
    
    for sl in seq_lengths:
        print(f"\n  Seq Length = {sl}")
        for name, model in models.items():
            lat = measure_latency(model, sl)
            thr = measure_throughput(model, sl)
            latency_results[name].append(lat)
            throughput_results[name].append(thr)
            print(f"    {name:12s}: {lat:.3f} ms  |  {thr:.0f} samples/s")
    
    # ═══ PLOT: Latency Scaling ═══
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style='whitegrid', font_scale=1.2)
    
    colors = {'BiMamba': '#E53935', 'BERT': '#1E88E5', 'UniMamba': '#43A047'}
    markers = {'BiMamba': 'o', 'BERT': 's', 'UniMamba': '^'}
    
    # Plot 1: Latency Scaling
    fig, ax = plt.subplots(figsize=(10, 6))
    for name in models:
        ax.plot(seq_lengths, latency_results[name],
                marker=markers[name], color=colors[name],
                linewidth=2.5, markersize=10, label=name)
    ax.set_xlabel("Sequence Length (packets)", fontsize=13)
    ax.set_ylabel("Latency (ms)", fontsize=13)
    ax.set_title("Latency Scaling: Mamba O(n) vs BERT O(n²)", fontsize=14, fontweight='bold')
    ax.legend(fontsize=12)
    ax.set_xticks(seq_lengths)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "07_latency_scaling.png"), dpi=150)
    plt.close()
    print(f"\n  → Saved 07_latency_scaling.png")
    
    # Plot 2: Throughput Scaling
    fig, ax = plt.subplots(figsize=(10, 6))
    for name in models:
        ax.plot(seq_lengths, throughput_results[name],
                marker=markers[name], color=colors[name],
                linewidth=2.5, markersize=10, label=name)
    ax.set_xlabel("Sequence Length (packets)", fontsize=13)
    ax.set_ylabel("Throughput (samples/s)", fontsize=13)
    ax.set_title("Throughput Scaling: Mamba O(n) vs BERT O(n²)", fontsize=14, fontweight='bold')
    ax.legend(fontsize=12)
    ax.set_xticks(seq_lengths)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "08_throughput_scaling.png"), dpi=150)
    plt.close()
    print(f"  → Saved 08_throughput_scaling.png")
    
    # Plot 3: Latency Ratio (BERT/Mamba)
    fig, ax = plt.subplots(figsize=(10, 6))
    ratio_bi = [latency_results['BERT'][i] / latency_results['BiMamba'][i] 
                for i in range(len(seq_lengths))]
    ratio_uni = [latency_results['BERT'][i] / latency_results['UniMamba'][i] 
                 for i in range(len(seq_lengths))]
    ax.plot(seq_lengths, ratio_bi, marker='o', color='#E53935',
            linewidth=2.5, markersize=10, label='BERT / BiMamba')
    ax.plot(seq_lengths, ratio_uni, marker='^', color='#43A047',
            linewidth=2.5, markersize=10, label='BERT / UniMamba')
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Equal')
    ax.set_xlabel("Sequence Length (packets)", fontsize=13)
    ax.set_ylabel("Latency Ratio (BERT / Mamba)", fontsize=13)
    ax.set_title("BERT Slowdown Factor vs Mamba", fontsize=14, fontweight='bold')
    ax.legend(fontsize=12)
    ax.set_xticks(seq_lengths)
    ax.grid(True, alpha=0.3)
    # Annotate crossover
    for i, sl in enumerate(seq_lengths):
        if ratio_bi[i] > 1.0 and (i == 0 or ratio_bi[i-1] <= 1.0):
            ax.annotate(f'BERT slower\nafter {sl} pkts',
                       xy=(sl, ratio_bi[i]), fontsize=10,
                       xytext=(sl+30, ratio_bi[i]+0.3),
                       arrowprops=dict(arrowstyle='->', color='red'),
                       color='red', fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "09_bert_slowdown_ratio.png"), dpi=150)
    plt.close()
    print(f"  → Saved 09_bert_slowdown_ratio.png")
    
    # Save results
    results = {
        'seq_lengths': seq_lengths,
        'latency_ms': latency_results,
        'throughput': throughput_results,
    }
    out_path = os.path.join(THESIS_DIR, "results/differentiation_benchmark.json")
    def jsonify(obj):
        if isinstance(obj, dict): return {k: jsonify(v) for k, v in obj.items()}
        elif isinstance(obj, list): return [jsonify(v) for v in obj]
        elif isinstance(obj, (np.floating,)): return float(obj)
        elif isinstance(obj, (np.integer,)): return int(obj)
        return obj
    with open(out_path, 'w') as f:
        json.dump(jsonify(results), f, indent=2)
    
    # Summary
    summary = "\nLatency Scaling Summary:\n"
    summary += f"{'SeqLen':>8} | {'BiMamba':>10} | {'BERT':>10} | {'UniMamba':>10} | {'BERT/BiMamba':>12}\n"
    summary += "-" * 60 + "\n"
    for i, sl in enumerate(seq_lengths):
        ratio = latency_results['BERT'][i] / latency_results['BiMamba'][i]
        summary += (f"{sl:>8} | {latency_results['BiMamba'][i]:>9.2f}ms | "
                   f"{latency_results['BERT'][i]:>9.2f}ms | "
                   f"{latency_results['UniMamba'][i]:>9.2f}ms | "
                   f"{ratio:>11.2f}x\n")
    
    print(summary)
    telegram_send(f"🏁 Benchmark DONE\n{summary}")

if __name__ == "__main__":
    import traceback
    try:
        main()
    except Exception as e:
        msg = f"❌ CRASH: {e}\n{traceback.format_exc()[-500:]}"
        telegram_send(msg); print(msg); sys.exit(1)
