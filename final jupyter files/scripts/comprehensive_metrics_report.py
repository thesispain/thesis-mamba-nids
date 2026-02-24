"""
COMPREHENSIVE METRICS REPORT
- Per-attack type: AUC, F1, Precision, Recall, Accuracy
- Overall metrics
- Latency, Throughput, Time-to-detect
- @8 vs @32 comparison
"""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, pickle, json
from collections import Counter, defaultdict
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (roc_auc_score, f1_score, accuracy_score,
                              precision_score, recall_score, confusion_matrix)
from mamba_ssm import Mamba
import time

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── Paths ──────────────────────────────────────────────────────────────
HERE     = Path('/home/T2510596/Downloads/totally fresh/thesis_final/final jupyter files')
ROOT     = Path('/home/T2510596/Downloads/totally fresh')
CIC_PATH = ROOT / 'thesis_final' / 'data' / 'cicids2017_flows.pkl'
CTU_PATH = ROOT / 'thesis_final' / 'data' / 'ctu13_flows.pkl'
UNSW_DIR = ROOT / 'Organized_Final' / 'data' / 'unswnb15_full'
SSL_PATH = HERE / 'weights' / 'self_distill' / 'unimamba_ssl_v2.pth'

# ── Model Architecture ─────────────────────────────────────────────────
class PacketEmbedder(nn.Module):
    def __init__(self, d=256, de=32):
        super().__init__()
        self.emb_proto = nn.Embedding(256, de)
        self.emb_flags = nn.Embedding(64, de)
        self.emb_dir   = nn.Embedding(2, de // 4)
        self.proj_len  = nn.Linear(1, de)
        self.proj_iat  = nn.Linear(1, de)
        self.fusion    = nn.Linear(de * 4 + de // 4, d)
        self.norm      = nn.LayerNorm(d)
    def forward(self, x):
        c = torch.cat([self.emb_proto(x[:,:,0].long().clamp(0,255)),
                       self.proj_len(x[:,:,1:2]),
                       self.emb_flags(x[:,:,2].long().clamp(0,63)),
                       self.proj_iat(x[:,:,3:4]),
                       self.emb_dir(x[:,:,4].long().clamp(0,1))], dim=-1)
        return self.norm(self.fusion(c))

class UniMambaSSL(nn.Module):
    def __init__(self, d=256, de=32, n_layers=4, proj_dim=128):
        super().__init__()
        self.tokenizer = PacketEmbedder(d, de)
        self.layers    = nn.ModuleList([Mamba(d, d_state=16, d_conv=4, expand=2)
                                        for _ in range(n_layers)])
        self.norm      = nn.LayerNorm(d)
        self.proj_head = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, proj_dim))
        self.recon_head = nn.Linear(d, 5)
    def forward(self, x):
        feat = self.tokenizer(x)
        for layer in self.layers:
            feat = self.norm(layer(feat) + feat)
        return feat

# ── Dataset ────────────────────────────────────────────────────────────
class FlowDatasetWithLabels(Dataset):
    def __init__(self, data):
        self.features = torch.tensor(np.array([d['features'] for d in data]), dtype=torch.float32)
        self.labels   = torch.tensor(np.array([d['label'] for d in data]), dtype=torch.long)
        self.attack_types = [d.get('attack_type', 'Benign' if d['label']==0 else 'Unknown') for d in data]
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx): return self.features[idx], self.labels[idx], self.attack_types[idx]

def load_pkl(path, fix_iat=False):
    with open(path, 'rb') as f: data = pickle.load(f)
    if fix_iat:
        for d in data: d['features'][:, 3] = np.log1p(d['features'][:, 3])
    return data

# ── Evaluation Functions ───────────────────────────────────────────────
@torch.no_grad()
def extract_reps(loader, model, exit_point):
    """Extract representations at given exit point."""
    reps, labels, attack_types_list = [], [], []
    for x, y, atypes in loader:
        feat = model(x.to(DEVICE))
        rep  = feat[:, :exit_point, :].mean(dim=1)
        reps.append(rep.cpu())
        labels.append(y)
        attack_types_list.extend(atypes)
    return torch.cat(reps), torch.cat(labels).numpy(), attack_types_list

def knn_scores(test_reps, train_reps, k=10, chunk=512):
    """Anomaly scores: higher = more anomalous."""
    db = F.normalize(train_reps.to(DEVICE), dim=1)
    scores = []
    for s in range(0, len(test_reps), chunk):
        q   = F.normalize(test_reps[s:s+chunk].to(DEVICE), dim=1)
        sim = torch.mm(q, db.T)
        topk_sim = sim.topk(k, dim=1).values.mean(dim=1)
        scores.append(topk_sim.cpu())
    return 1.0 - torch.cat(scores).numpy()

def compute_metrics(scores, labels):
    """Return dict with all metrics."""
    auc = roc_auc_score(labels, scores)
    if auc < 0.5:
        scores, auc = -scores, 1.0 - auc
    
    # Optimal threshold via Youden's J
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(labels, scores)
    j = tpr - fpr
    best_thresh = thresholds[np.argmax(j)]
    
    preds = (scores >= best_thresh).astype(int)
    
    return {
        'auc': float(auc),
        'f1': float(f1_score(labels, preds, zero_division=0)),
        'accuracy': float(accuracy_score(labels, preds)),
        'precision': float(precision_score(labels, preds, zero_division=0)),
        'recall': float(recall_score(labels, preds, zero_division=0)),
        'threshold': float(best_thresh),
        'tn_fp_fn_tp': confusion_matrix(labels, preds).ravel().tolist() if len(np.unique(labels)) > 1 else [0,0,0,0]
    }

def per_attack_metrics(scores, labels, attack_types):
    """Compute metrics for each attack type."""
    results = {}
    unique_attacks = sorted(set(attack_types))
    
    for attack in unique_attacks:
        if attack == 'Benign':
            continue
        
        # Create binary labels: this attack vs rest
        attack_mask = np.array([at == attack for at in attack_types])
        if attack_mask.sum() == 0:
            continue
        
        binary_labels = attack_mask.astype(int)
        attack_scores = scores.copy()
        
        # Compute metrics if we have both classes
        if len(np.unique(binary_labels)) > 1:
            metrics = compute_metrics(attack_scores, binary_labels)
            metrics['count'] = int(attack_mask.sum())
            results[attack] = metrics
    
    return results

# ── Performance Metrics ────────────────────────────────────────────────
def measure_latency(model, batch_size=1, n_warmup=100, n_iter=1000):
    """Measure inference latency."""
    dummy = torch.randn(batch_size, 32, 5, device=DEVICE)
    
    # Warmup
    for _ in range(n_warmup):
        _ = model(dummy)
    
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(n_iter):
        _ = model(dummy)
    torch.cuda.synchronize()
    end = time.time()
    
    avg_latency_ms = (end - start) / n_iter * 1000
    throughput_fps = 1000 / avg_latency_ms
    
    return avg_latency_ms, throughput_fps

# ══════════════════════════════════════════════════════════════════════
# MAIN EVALUATION
# ══════════════════════════════════════════════════════════════════════

print("=" * 80)
print("COMPREHENSIVE METRICS REPORT: UniMamba SSL @8 vs @32")
print("=" * 80)
print(f"Device: {DEVICE}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}\n")

# Load model
print("Loading model...")
model = UniMambaSSL().to(DEVICE)
sd = torch.load(SSL_PATH, map_location='cpu', weights_only=False)
model.load_state_dict(sd, strict=False)
model.eval()
print(f"  Model params: {sum(p.numel() for p in model.parameters()):,}\n")

# Load datasets
print("Loading datasets...")
cic_data = load_pkl(CIC_PATH, fix_iat=True)
ctu_data = load_pkl(CTU_PATH)
unsw_data = load_pkl(UNSW_DIR / 'finetune_mixed.pkl')
pretrain_data = load_pkl(UNSW_DIR / 'pretrain_50pct_benign.pkl')

cic_loader = DataLoader(FlowDatasetWithLabels(cic_data), batch_size=512, shuffle=False)
ctu_loader = DataLoader(FlowDatasetWithLabels(ctu_data), batch_size=512, shuffle=False)
unsw_loader = DataLoader(FlowDatasetWithLabels(unsw_data), batch_size=512, shuffle=False)

# Training refs (benign only, sample 20K for speed)
N = 20000
idx = np.random.choice(len(pretrain_data), min(N, len(pretrain_data)), replace=False)
pretrain_subset = [pretrain_data[i] for i in idx]
pretrain_loader = DataLoader(FlowDatasetWithLabels(pretrain_subset), batch_size=512, shuffle=False)

print(f"  CIC-IDS-2017: {len(cic_data):,} flows")
print(f"  CTU-13: {len(ctu_data):,} flows")
print(f"  UNSW-NB15: {len(unsw_data):,} flows")
print(f"  Training refs: {len(pretrain_subset):,} flows\n")

# Extract training representations
print("Extracting training representations...")
train_reps_8  = extract_reps(pretrain_loader, model, 8)[0]
train_reps_32 = extract_reps(pretrain_loader, model, 32)[0]
print(f"  @8: {train_reps_8.shape}  @32: {train_reps_32.shape}\n")

# ── Performance Metrics ────────────────────────────────────────────────
print("=" * 80)
print("PERFORMANCE METRICS")
print("=" * 80)
latency_8, throughput_8 = measure_latency(model, batch_size=1)
print(f"Latency (batch=1):    {latency_8:.3f} ms")
print(f"Throughput:           {throughput_8:.1f} flows/sec")
print(f"Time-to-detect:       {latency_8:.3f} ms  (= latency for single flow)")
print()

# ── Evaluate Each Dataset ──────────────────────────────────────────────
all_results = {}

for ds_name, loader in [('CIC-IDS-2017', cic_loader), ('CTU-13', ctu_loader), ('UNSW-NB15', unsw_loader)]:
    print("=" * 80)
    print(f"DATASET: {ds_name}")
    print("=" * 80)
    
    # Extract test reps
    reps_8,  labels, attack_types = extract_reps(loader, model, 8)
    reps_32, _, _ = extract_reps(loader, model, 32)
    
    # Attack distribution
    attack_counter = Counter(attack_types)
    print(f"\nAttack Distribution (n={len(labels):,}):")
    for atype, count in sorted(attack_counter.items(), key=lambda x: -x[1])[:15]:
        pct = 100.0 * count / len(labels)
        print(f"  {atype:<30} {count:>8,}  ({pct:>5.2f}%)")
    
    # Compute scores
    scores_8  = knn_scores(reps_8,  train_reps_8)
    scores_32 = knn_scores(reps_32, train_reps_32)
    
    # Overall metrics
    print(f"\n{'-'*80}")
    print("OVERALL METRICS")
    print("-" * 80)
    
    m8_overall  = compute_metrics(scores_8, labels)
    m32_overall = compute_metrics(scores_32, labels)
    
    print(f"{'Exit Point':<12} {'AUC':>8} {'F1':>8} {'Acc':>8} {'Prec':>8} {'Rec':>8}")
    print("-" * 80)
    print(f"{'@8 packets':<12} {m8_overall['auc']:>8.4f} {m8_overall['f1']:>8.4f} "
          f"{m8_overall['accuracy']:>8.4f} {m8_overall['precision']:>8.4f} {m8_overall['recall']:>8.4f}")
    print(f"{'@32 packets':<12} {m32_overall['auc']:>8.4f} {m32_overall['f1']:>8.4f} "
          f"{m32_overall['accuracy']:>8.4f} {m32_overall['precision']:>8.4f} {m32_overall['recall']:>8.4f}")
    print(f"{'Δ (@8 - @32)':<12} {m8_overall['auc']-m32_overall['auc']:>+8.4f} "
          f"{m8_overall['f1']-m32_overall['f1']:>+8.4f} "
          f"{m8_overall['accuracy']-m32_overall['accuracy']:>+8.4f} "
          f"{m8_overall['precision']-m32_overall['precision']:>+8.4f} "
          f"{m8_overall['recall']-m32_overall['recall']:>+8.4f}")
    
    # Per-attack metrics
    print(f"\n{'-'*80}")
    print("PER-ATTACK TYPE METRICS (@8 packets)")
    print("-" * 80)
    
    per_attack_8  = per_attack_metrics(scores_8, labels, attack_types)
    per_attack_32 = per_attack_metrics(scores_32, labels, attack_types)
    
    if per_attack_8:
        print(f"{'Attack Type':<30} {'Count':>8} {'AUC':>8} {'F1':>8} {'Prec':>8} {'Rec':>8}")
        print("-" * 80)
        for attack in sorted(per_attack_8.keys(), key=lambda x: per_attack_8[x]['count'], reverse=True):
            m = per_attack_8[attack]
            print(f"{attack:<30} {m['count']:>8,} {m['auc']:>8.4f} {m['f1']:>8.4f} "
                  f"{m['precision']:>8.4f} {m['recall']:>8.4f}")
    
    print(f"\n{'-'*80}")
    print("PER-ATTACK TYPE METRICS (@32 packets)")
    print("-" * 80)
    
    if per_attack_32:
        print(f"{'Attack Type':<30} {'Count':>8} {'AUC':>8} {'F1':>8} {'Prec':>8} {'Rec':>8}")
        print("-" * 80)
        for attack in sorted(per_attack_32.keys(), key=lambda x: per_attack_32[x]['count'], reverse=True):
            m = per_attack_32[attack]
            print(f"{attack:<30} {m['count']:>8,} {m['auc']:>8.4f} {m['f1']:>8.4f} "
                  f"{m['precision']:>8.4f} {m['recall']:>8.4f}")
    
    all_results[ds_name] = {
        'overall_@8': m8_overall,
        'overall_@32': m32_overall,
        'per_attack_@8': per_attack_8,
        'per_attack_@32': per_attack_32,
        'attack_distribution': dict(attack_counter)
    }
    print()

# ── Save Results ───────────────────────────────────────────────────────
output_file = HERE / 'results' / 'comprehensive_metrics.json'
output_file.parent.mkdir(parents=True, exist_ok=True)

all_results['performance'] = {
    'latency_ms': latency_8,
    'throughput_fps': throughput_8,
    'time_to_detect_ms': latency_8,
    'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'
}

with open(output_file, 'w') as f:
    json.dump(all_results, f, indent=2)

print("=" * 80)
print(f"✅ Results saved to: {output_file}")
print("=" * 80)

# ── Summary ────────────────────────────────────────────────────────────
print("\nKEY FINDINGS:")
print("-" * 80)
for ds_name in ['CIC-IDS-2017', 'CTU-13', 'UNSW-NB15']:
    m8  = all_results[ds_name]['overall_@8']
    m32 = all_results[ds_name]['overall_@32']
    delta_auc = m8['auc'] - m32['auc']
    
    better = "@8 BETTER" if delta_auc > 0 else "@32 better"
    print(f"{ds_name:<15}  Δ AUC = {delta_auc:>+.4f}  ({better})")

print(f"\nPerformance: {latency_8:.3f} ms latency, {throughput_8:.1f} flows/sec")
print("\n✅ Done!")
