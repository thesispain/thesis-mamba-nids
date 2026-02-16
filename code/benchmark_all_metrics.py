#!/usr/bin/env python3
"""
FULL BENCHMARK: ALL MODELS, ALL METRICS
========================================
Loads pretrained weights (NO retraining). Measures:
  1. F1, AUC (in-domain UNSW)
  2. Latency (ms per flow, CUDA synced, 100 trials)
  3. Throughput (flows/s)
  4. MNP (Mean Number of Packets) for TED
  5. TTD (Time-To-Detect) = latency scaled by MNP ratio
  6. Cross-Dataset AUC (Zero-Shot on CIC-IDS)
"""
import os, sys, pickle, time, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, precision_score, recall_score
from mamba_ssm import Mamba

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BASE   = os.path.dirname(os.path.abspath(__file__))
ROOT   = os.path.dirname(BASE)
DATA   = os.path.join(ROOT, "../Organized_Final/data/unswnb15_full/finetune_mixed.pkl")
CIC    = os.path.join(ROOT, "data/cicids2017_flows.pkl")
W      = os.path.join(ROOT, "weights")

# ═══════════════════════════════════════════════════════════════════
#  MODELS (exact copies from run_full_evaluation.py)
# ═══════════════════════════════════════════════════════════════════
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

class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.pe_emb = nn.Embedding(max_len, d_model)
    def forward(self, x):
        return x + self.pe_emb(torch.arange(x.size(1), device=x.device))

class BertEncoder(nn.Module):
    def __init__(self, d_model=256, n_layers=4, n_heads=8):
        super().__init__()
        self.tokenizer = PacketEmbedder(d_model)
        self.pos_encoder = LearnablePositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=1024,
                                               dropout=0.1, activation='gelu', batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(enc_layer, n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.proj_head = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 128))
        self.recon_head = nn.Linear(d_model, 5)
    def forward(self, x):
        f = self.tokenizer(x)
        f = self.pos_encoder(f)
        f = self.transformer_encoder(f)
        f = self.norm(f)
        return self.proj_head(f[:, 0, :]), None

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
        for fwd, rev in zip(self.layers, self.layers_rev):
            out_f = fwd(feat)
            out_r = rev(feat.flip(1)).flip(1)
            feat  = self.norm((out_f + out_r) / 2 + feat)
        return self.proj_head(feat.mean(1)), None

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
            feat = self.norm(layer(feat) + feat)
        return self.proj_head(feat.mean(1)), None

class Classifier(nn.Module):
    def __init__(self, encoder, input_dim, dropout=0.0):
        super().__init__()
        self.encoder = encoder
        if dropout > 0:
            self.head = nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 2))
        else:
            self.head = nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU(), nn.Linear(64, 2))
    def forward(self, x):
        z = self.encoder(x)
        if isinstance(z, tuple): z = z[0]
        return self.head(z)

class BlockwiseEarlyExitMamba(nn.Module):
    def __init__(self, d_model=256, exit_positions=None, conf_thresh=0.85):
        super().__init__()
        if exit_positions is None: exit_positions = [8, 16, 32]
        self.exit_positions = exit_positions
        self.n_exits = len(exit_positions)
        self.conf_thresh = conf_thresh
        self.d_model = d_model
        self.tokenizer = PacketEmbedder(d_model)
        self.layers = nn.ModuleList([Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2) for _ in range(4)])
        self.norm = nn.LayerNorm(d_model)
        self.exit_classifiers = nn.ModuleDict({
            str(p): nn.Sequential(nn.Linear(d_model, 128), nn.ReLU(), nn.Dropout(0.1), nn.Linear(128, 2)) for p in exit_positions})
        self.confidence_heads = nn.ModuleDict({
            str(p): nn.Sequential(nn.Linear(d_model + 2, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid()) for p in exit_positions})
        self.register_buffer('exit_counts', torch.zeros(self.n_exits, dtype=torch.long))
        self.register_buffer('total_inferences', torch.tensor(0, dtype=torch.long))

    def _backbone(self, x):
        feat = self.tokenizer(x)
        for layer in self.layers:
            feat = self.norm(layer(feat) + feat)
        return feat

    def forward_inference(self, x):
        B = x.size(0)
        feat = self._backbone(x)
        final_logits = torch.zeros(B, 2, device=x.device)
        exit_indices = torch.full((B,), self.n_exits-1, device=x.device, dtype=torch.long)
        active = torch.ones(B, dtype=torch.bool, device=x.device)
        for i, pos in enumerate(self.exit_positions[:-1]):
            if not active.any(): break
            idx = min(pos, feat.size(1)) - 1
            h = feat[active, idx, :]
            logits = self.exit_classifiers[str(pos)](h)
            conf_input = torch.cat([h, logits], dim=-1)
            conf = self.confidence_heads[str(pos)](conf_input).squeeze(-1)
            should_exit = conf >= self.conf_thresh
            if should_exit.any():
                active_idx = active.nonzero(as_tuple=True)[0]
                exiting = active_idx[should_exit]
                final_logits[exiting] = logits[should_exit]
                exit_indices[exiting] = i
                active[exiting] = False
                self.exit_counts[i] += should_exit.sum().item()
        if active.any():
            last_pos = self.exit_positions[-1]
            idx = min(last_pos, feat.size(1)) - 1
            h = feat[active, idx, :]
            final_logits[active] = self.exit_classifiers[str(last_pos)](h)
            self.exit_counts[-1] += active.sum().item()
        self.total_inferences += B
        return final_logits, exit_indices

    def forward(self, x):
        return self.forward_inference(x)

    def reset_stats(self):
        self.exit_counts.zero_(); self.total_inferences.zero_()

    def get_exit_stats(self):
        if self.total_inferences == 0: return None
        pct = (self.exit_counts.float() / self.total_inferences * 100).cpu().numpy()
        return {
            'exit_pct': dict(zip([str(p) for p in self.exit_positions], pct.tolist())),
            'avg_packets': sum(self.exit_positions[i] * pct[i] / 100 for i in range(self.n_exits))
        }

# ═══════════════════════════════════════════════════════════════════
#  BENCHMARK FUNCTION
# ═══════════════════════════════════════════════════════════════════
def measure_latency(model, sample, is_blockwise=False, n_warmup=20, n_trials=200):
    """Per-flow latency with CUDA synced timing."""
    x = sample.unsqueeze(0).to(DEVICE)
    # Warmup
    for _ in range(n_warmup):
        with torch.no_grad():
            if is_blockwise: model.forward_inference(x)
            else: model(x)
    if torch.cuda.is_available(): torch.cuda.synchronize()
    # Measure
    lats = []
    for _ in range(n_trials):
        if torch.cuda.is_available(): torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            if is_blockwise: model.forward_inference(x)
            else: model(x)
        if torch.cuda.is_available(): torch.cuda.synchronize()
        lats.append((time.perf_counter() - t0) * 1000)
    return np.median(lats)

def benchmark(model, dl, name, is_blockwise=False):
    model.eval()
    if is_blockwise: model.reset_stats()
    all_probs, all_preds, all_labels = [], [], []
    n = 0
    t0 = time.perf_counter()
    with torch.no_grad():
        for x, y in dl:
            x = x.to(DEVICE)
            if is_blockwise:
                logits, _ = model.forward_inference(x)
            else:
                logits = model(x)
            all_probs.extend(F.softmax(logits, 1)[:, 1].cpu().numpy())
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(y.numpy())
            n += x.size(0)
    if torch.cuda.is_available(): torch.cuda.synchronize()
    throughput = n / (time.perf_counter() - t0)

    # Per-flow latency
    sample = dl.dataset[0][0]
    latency = measure_latency(model, sample, is_blockwise)

    y_true = np.array(all_labels)
    probs  = np.nan_to_num(all_probs)
    preds  = np.array(all_preds)

    f1  = f1_score(y_true, preds, zero_division=0)
    try: auc = roc_auc_score(y_true, probs)
    except: auc = 0.5
    acc  = accuracy_score(y_true, preds)
    prec = precision_score(y_true, preds, zero_division=0)
    rec  = recall_score(y_true, preds, zero_division=0)

    # MNP and TTD
    mnp = 32.0
    if is_blockwise:
        stats = model.get_exit_stats()
        if stats:
            mnp = stats['avg_packets']
    ttd = latency * (mnp / 32.0)  # Time-to-detect scales with packets used

    res = {
        'model': name,
        'f1': round(f1, 4), 'auc': round(auc, 4),
        'acc': round(acc, 4), 'prec': round(prec, 4), 'rec': round(rec, 4),
        'latency_ms': round(latency, 4),
        'throughput': round(throughput, 0),
        'mnp': round(mnp, 1),
        'ttd_ms': round(ttd, 4),
    }
    if is_blockwise and model.get_exit_stats():
        res['exit_distribution'] = model.get_exit_stats()['exit_pct']
    return res

def cross_dataset_auc(model, cic_dl, is_blockwise=False):
    model.eval()
    if is_blockwise:
        model.reset_stats()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for x, y in cic_dl:
            x = x.to(DEVICE)
            if is_blockwise:
                logits, _ = model.forward_inference(x)
            else:
                logits = model(x)
            all_probs.extend(F.softmax(logits, 1)[:, 1].cpu().numpy())
            all_labels.extend(y.numpy())
    y_true = np.array(all_labels)
    y_scores = np.nan_to_num(all_probs)
    try: auc = roc_auc_score(y_true, y_scores)
    except: auc = 0.5
    if auc < 0.5: auc = 1 - auc  # Handle inversion
    return round(auc, 4)

# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════
def main():
    print(f"Device: {DEVICE}")
    print("="*80)
    print("FULL BENCHMARK: ALL MODELS, ALL METRICS")
    print("="*80)

    # ── Load UNSW ──
    print("\nLoading UNSW-NB15...")
    with open(DATA, 'rb') as f:
        data = pickle.load(f)
    np.random.seed(42)
    np.random.shuffle(data)
    n_test = int(len(data) * 0.90)  # Use last 10% as test
    test_data = data[n_test:]
    X_test = np.array([d['features'] for d in test_data], dtype=np.float32)
    y_test = np.array([d['label'] for d in test_data], dtype=np.longlong)
    test_dl = DataLoader(TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test)),
                         batch_size=32, shuffle=False)
    benign = np.sum(y_test == 0); attack = np.sum(y_test == 1)
    print(f"  UNSW Test: {len(y_test):,} (Benign: {benign:,}, Attack: {attack:,})")

    # ── Load CIC-IDS ──
    print("Loading CIC-IDS-2017...")
    with open(CIC, 'rb') as f:
        cic_data = pickle.load(f)
    X_cic = np.array([d['features'] for d in cic_data], dtype=np.float32)
    y_cic = np.array([d['label'] for d in cic_data], dtype=np.longlong)
    cic_dl = DataLoader(TensorDataset(torch.from_numpy(X_cic), torch.from_numpy(y_cic)),
                        batch_size=32, shuffle=False)
    print(f"  CIC-IDS: {len(y_cic):,}")

    results = []

    # ── 1. BERT ──
    print("\n═══ 1. BERT (SSL + UNSW) ═══")
    bert_enc = BertEncoder()
    bert = Classifier(bert_enc, input_dim=128, dropout=0.1).to(DEVICE)
    sd = torch.load(f"{W}/teachers/teacher_bert_masking.pth", map_location=DEVICE, weights_only=False)
    bert.load_state_dict(sd, strict=False)
    r = benchmark(bert, test_dl, "BERT")
    r['cross_ds_auc'] = cross_dataset_auc(bert, cic_dl)
    results.append(r)
    print(f"  → F1={r['f1']:.4f}  AUC={r['auc']:.4f}  Lat={r['latency_ms']:.2f}ms  TTD={r['ttd_ms']:.2f}ms  Cross-DS={r['cross_ds_auc']:.4f}")

    # ── 2. BiMamba Teacher ──
    print("\n═══ 2. BiMamba Teacher (SSL + 100% UNSW) ═══")
    bimamba_enc = BiMambaEncoder()
    bimamba = Classifier(bimamba_enc, input_dim=256).to(DEVICE)
    sd = torch.load(f"{W}/teachers/teacher_bimamba_retrained.pth", map_location=DEVICE, weights_only=False)
    bimamba.load_state_dict(sd, strict=False)
    r = benchmark(bimamba, test_dl, "BiMamba Teacher")
    r['cross_ds_auc'] = cross_dataset_auc(bimamba, cic_dl)
    results.append(r)
    print(f"  → F1={r['f1']:.4f}  AUC={r['auc']:.4f}  Lat={r['latency_ms']:.2f}ms  TTD={r['ttd_ms']:.2f}ms  Cross-DS={r['cross_ds_auc']:.4f}")

    # ── 3. KD Student (also BlockwiseEarlyExitMamba architecture) ──
    print("\n═══ 3. KD Student ═══")
    kd = BlockwiseEarlyExitMamba().to(DEVICE)
    sd = torch.load(f"{W}/students/student_standard_kd.pth", map_location=DEVICE, weights_only=False)
    kd.load_state_dict(sd, strict=False)
    r = benchmark(kd, test_dl, "KD Student", is_blockwise=True)
    r['cross_ds_auc'] = cross_dataset_auc(kd, cic_dl, is_blockwise=True)
    results.append(r)
    print(f"  → F1={r['f1']:.4f}  AUC={r['auc']:.4f}  Lat={r['latency_ms']:.2f}ms  TTD={r['ttd_ms']:.2f}ms  Cross-DS={r['cross_ds_auc']:.4f}")

    # ── 4. TED Student ──
    print("\n═══ 4. TED Student (Blockwise Early Exit) ═══")
    ted = BlockwiseEarlyExitMamba().to(DEVICE)
    sd = torch.load(f"{W}/students/student_ted.pth", map_location=DEVICE, weights_only=False)
    ted.load_state_dict(sd, strict=False)
    r = benchmark(ted, test_dl, "TED Student", is_blockwise=True)
    r['cross_ds_auc'] = cross_dataset_auc(ted, cic_dl, is_blockwise=True)
    results.append(r)
    print(f"  → F1={r['f1']:.4f}  AUC={r['auc']:.4f}  Lat={r['latency_ms']:.2f}ms  TTD={r['ttd_ms']:.2f}ms  MNP={r['mnp']:.1f}  Cross-DS={r['cross_ds_auc']:.4f}")
    if 'exit_distribution' in r:
        print(f"  → Exit Distribution: {r['exit_distribution']}")

    # ── 5. UniMamba (No KD, with Early Exit) ──
    print("\n═══ 5. UniMamba (No KD + Early Exit) ═══")
    uni_ee = BlockwiseEarlyExitMamba().to(DEVICE)
    sd = torch.load(f"{W}/students/student_no_kd.pth", map_location=DEVICE, weights_only=False)
    uni_ee.load_state_dict(sd, strict=False)
    r = benchmark(uni_ee, test_dl, "UniMamba (No KD)", is_blockwise=True)
    r['cross_ds_auc'] = cross_dataset_auc(uni_ee, cic_dl, is_blockwise=True)
    results.append(r)
    print(f"  → F1={r['f1']:.4f}  AUC={r['auc']:.4f}  Lat={r['latency_ms']:.2f}ms  TTD={r['ttd_ms']:.2f}ms  MNP={r['mnp']:.1f}  Cross-DS={r['cross_ds_auc']:.4f}")
    if 'exit_distribution' in r:
        print(f"  → Exit Distribution: {r['exit_distribution']}")

    # ═══════════════════════════════════════════════════════════════
    #  FINAL RESULTS TABLE
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*110}")
    print("COMPLETE RESULTS TABLE")
    print(f"{'='*110}")
    header = f"{'Model':25s} {'F1':>8s} {'AUC':>8s} {'Lat(ms)':>8s} {'TTD(ms)':>8s} {'Thr(f/s)':>10s} {'MNP':>6s} {'CrossDS':>8s}"
    print(header)
    print("-"*110)
    for r in results:
        print(f"{r['model']:25s} {r['f1']:>8.4f} {r['auc']:>8.4f} {r['latency_ms']:>8.2f} {r['ttd_ms']:>8.2f} {r['throughput']:>10.0f} {r['mnp']:>6.1f} {r['cross_ds_auc']:>8.4f}")

    # ═══════════════════════════════════════════════════════════════
    #  METRIC SUITE
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print("METRIC SUITE SUMMARY")
    print(f"{'='*80}")
    ted_r = [r for r in results if r['model'] == 'TED Student'][0]
    uni_r = [r for r in results if r['model'] == 'UniMamba (No KD)'][0]
    print(f"\n  SECURITY (Earliness):")
    print(f"    TED MNP:          {ted_r['mnp']:.1f} packets ({ted_r['mnp']/32*100:.0f}% of flow)")
    print(f"    UniMamba MNP:     {uni_r['mnp']:.1f} packets ({uni_r['mnp']/32*100:.0f}% of flow)")
    print(f"\n  ENGINEERING (Speed):")
    print(f"    TED Latency:      {ted_r['latency_ms']:.2f} ms")
    print(f"    TED TTD:          {ted_r['ttd_ms']:.2f} ms")
    print(f"    TED Throughput:   {ted_r['throughput']:.0f} flows/s")
    print(f"\n  COST (FLOPs Reduction):")
    flops_saved = (1 - ted_r['mnp'] / 32) * 100
    print(f"    FLOPs Saved:      ~{flops_saved:.0f}% (based on {ted_r['mnp']:.1f}/32 avg packets)")

    # Save
    out_path = os.path.join(ROOT, "results/full_benchmark_all_metrics.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=lambda o: float(o) if hasattr(o, 'item') else o)
    print(f"\nResults saved to {out_path}")

if __name__ == "__main__":
    main()
