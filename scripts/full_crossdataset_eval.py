"""
FULL Cross-Dataset Evaluation — All metrics (AUC, F1, Precision, Recall, Accuracy)
Pre-trained on UNSW-NB15 only → evaluated on UNSW, CTU-13, CIC-IDS 2017, CIC-DDoS 2019
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, pickle, json, gc
from sklearn.metrics import (roc_auc_score, f1_score, precision_score, 
                             recall_score, accuracy_score, classification_report,
                             roc_curve)
from sklearn.neighbors import NearestNeighbors
try:
    from mamba_ssm import Mamba
except ImportError:
    print("Mamba not found")

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(os.path.dirname(BASE), "Organized_Final/data/unswnb15_full")
PRETRAIN_PKL = os.path.join(DATA_DIR, "pretrain_50pct_benign.pkl")
FINETUNE_PKL = os.path.join(DATA_DIR, "finetune_mixed.pkl")
CTU_PKL      = os.path.join(BASE, "data/ctu13_flows.pkl")
CICIDS_PKL   = os.path.join(BASE, "data/cicids2017_flows.pkl")
CICDOS_PKL   = os.path.join(BASE, "data/cicdos2019_flows.pkl")
SSL_DIR      = os.path.join(BASE, "weights/ssl")

# ═══════════════════════════════════════════════════════════════════
#  ARCHITECTURE (identical to training)
# ═══════════════════════════════════════════════════════════════════

class PacketEmbedder(nn.Module):
    def __init__(self, d_model=256):
        super().__init__()
        self.proto_emb  = nn.Embedding(256, 32, padding_idx=0)
        self.flags_emb  = nn.Embedding(64, 32, padding_idx=0)
        self.dir_emb    = nn.Embedding(2, 8, padding_idx=0)
        self.loglen_proj = nn.Linear(1, 32)
        self.iat_proj    = nn.Linear(1, 32)
        self.fusion = nn.Linear(32+32+32+32+8, d_model)
        self.norm = nn.LayerNorm(d_model)
    def forward(self, x):
        proto  = x[:,:,0].long().clamp(0,255)
        loglen = x[:,:,1:2]
        flags  = x[:,:,2].long().clamp(0,63)
        iat    = x[:,:,3:4]
        direction = x[:,:,4].long().clamp(0,1)
        cat = torch.cat([self.proto_emb(proto), self.loglen_proj(loglen),
                         self.flags_emb(flags), self.iat_proj(iat),
                         self.dir_emb(direction)], dim=-1)
        return self.norm(self.fusion(cat))

class BiMambaEncoder(nn.Module):
    def __init__(self, d_model=256, n_layers=4):
        super().__init__()
        self.embedder = PacketEmbedder(d_model)
        self.forward_layers = nn.ModuleList([Mamba(d_model=d_model) for _ in range(n_layers)])
        self.reverse_layers = nn.ModuleList([Mamba(d_model=d_model) for _ in range(n_layers)])
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(d_model, 128)
        self.recon_head = nn.Linear(d_model, 5)
    def forward(self, x, padding_mask=None):
        emb = self.embedder(x)
        fwd = emb
        for layer in self.forward_layers: fwd = layer(fwd) + fwd
        rev = torch.flip(emb, dims=[1])
        for layer in self.reverse_layers: rev = layer(rev) + rev
        rev = torch.flip(rev, dims=[1])
        combined = fwd + rev
        if padding_mask is not None:
            combined = combined * (~padding_mask).unsqueeze(-1).float()
        pooled = self.pool(combined.transpose(1,2)).squeeze(-1)
        return self.proj(pooled)

class BertEncoder(nn.Module):
    def __init__(self, d_model=256, n_layers=4, n_heads=8):
        super().__init__()
        self.embedder = PacketEmbedder(d_model)
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        layer = TransformerEncoderLayer(d_model=d_model, nhead=n_heads, 
                                       dim_feedforward=d_model*4, dropout=0.1,
                                       activation='gelu', batch_first=True)
        self.transformer = TransformerEncoder(layer, num_layers=n_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(d_model, 128)
        self.recon_head = nn.Linear(d_model, 5)
    
    def forward(self, x, padding_mask=None):
        emb = self.embedder(x)
        if padding_mask is not None:
            attn_mask = padding_mask
        else:
            attn_mask = None
        out = self.transformer(emb, src_key_padding_mask=attn_mask)
        
        if padding_mask is not None:
            out = out.masked_fill(padding_mask.unsqueeze(-1), 0)
        
        pooled = self.pool(out.transpose(1, 2)).squeeze(-1)
        z = F.normalize(self.proj(pooled), dim=-1)
        return z

# ═══════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════
def prepare_eval_subset(data, max_per_class=20000):
    benign = [f for f in data if f['label'] == 0]
    attack = [f for f in data if f['label'] == 1]
    np.random.seed(42)
    if len(benign) > max_per_class:
        benign = [benign[i] for i in np.random.choice(len(benign), max_per_class, replace=False)]
    if len(attack) > max_per_class:
        attack = [attack[i] for i in np.random.choice(len(attack), max_per_class, replace=False)]
    subset = benign + attack
    np.random.shuffle(subset)
    return subset

@torch.no_grad()
def encode_flows(encoder, flows, batch_size=512):
    encoder.eval()
    all_embs = []
    for i in range(0, len(flows), batch_size):
        batch = flows[i:i+batch_size]
        feats = np.stack([f['features'] for f in batch])
        x = torch.tensor(feats, dtype=torch.float32, device=DEVICE)
        pad_mask = (x.sum(dim=-1) == 0)
        emb = encoder(x, pad_mask)
        all_embs.append(emb.cpu().numpy())
    return np.concatenate(all_embs, axis=0)

def full_eval(encoder, ref_flows, eval_flows, method='cosine_top10'):
    """Full evaluation: AUC, F1, Precision, Recall, Accuracy."""
    ref_embs  = encode_flows(encoder, ref_flows)
    eval_embs = encode_flows(encoder, eval_flows)
    labels = np.array([f['label'] for f in eval_flows])
    
    # Compute anomaly scores
    if method == 'knn':
        nn_model = NearestNeighbors(n_neighbors=1, metric='euclidean')
        nn_model.fit(ref_embs)
        distances, _ = nn_model.kneighbors(eval_embs)
        scores = distances.mean(axis=1)
    elif method == 'cosine_top10':
        ref_n = ref_embs / (np.linalg.norm(ref_embs, axis=1, keepdims=True) + 1e-8)
        scores = []
        for i in range(0, len(eval_embs), 512):
            chunk = eval_embs[i:i+512]
            chunk_n = chunk / (np.linalg.norm(chunk, axis=1, keepdims=True) + 1e-8)
            sim = chunk_n @ ref_n.T
            topk = np.sort(sim, axis=1)[:, -10:]
            scores.extend((1 - topk.mean(axis=1)).tolist())
        scores = np.array(scores)
    elif method == 'cosine_max':
        ref_n = ref_embs / (np.linalg.norm(ref_embs, axis=1, keepdims=True) + 1e-8)
        scores = []
        for i in range(0, len(eval_embs), 512):
            chunk = eval_embs[i:i+512]
            chunk_n = chunk / (np.linalg.norm(chunk, axis=1, keepdims=True) + 1e-8)
            sim = chunk_n @ ref_n.T
            scores.extend((1 - sim.max(axis=1)).tolist())
        scores = np.array(scores)
    
    # AUC
    auc = roc_auc_score(labels, scores)
    
    # Optimal threshold via Youden's J
    fpr, tpr, thresholds = roc_curve(labels, scores)
    youden_j = tpr - fpr
    best_idx = np.argmax(youden_j)
    best_threshold = thresholds[best_idx]
    
    # Binary predictions
    preds = (scores >= best_threshold).astype(int)
    
    f1  = f1_score(labels, preds)
    prec = precision_score(labels, preds, zero_division=0)
    rec  = recall_score(labels, preds)
    acc  = accuracy_score(labels, preds)
    
    return {
        'auc': auc, 'f1': f1, 'precision': prec, 'recall': rec,
        'accuracy': acc, 'threshold': float(best_threshold),
        'tpr': float(tpr[best_idx]), 'fpr': float(fpr[best_idx])
    }

# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════
def main():
    print("=" * 80)
    print("  COMPREHENSIVE CROSS-DATASET EVALUATION")
    print("  Pre-trained on UNSW-NB15 → Tested on all datasets")
    print("=" * 80)
    
    # Reference: benign-only pretrain data
    print("\n[1] Loading UNSW-NB15 reference (benign pretrain)...")
    with open(PRETRAIN_PKL, 'rb') as f:
        pretrain = pickle.load(f)
    ref_subset = pretrain[:50000]
    print(f"    Reference pool: {len(ref_subset)} benign UNSW flows")
    del pretrain; gc.collect()
    
    # Load all eval datasets
    print("\n[2] Loading evaluation datasets...")
    datasets = {}
    
    with open(FINETUNE_PKL, 'rb') as f:
        d = pickle.load(f)
    datasets['UNSW-NB15'] = prepare_eval_subset(d)
    del d; gc.collect()
    
    for name, path in [('CTU-13', CTU_PKL), ('CIC-IDS-2017', CICIDS_PKL), ('CIC-DDoS-2019', CICDOS_PKL)]:
        if os.path.exists(path):
            with open(path, 'rb') as f:
                d = pickle.load(f)
            datasets[name] = prepare_eval_subset(d)
            del d; gc.collect()
    
    for name, subset in datasets.items():
        b = sum(1 for f in subset if f['label'] == 0)
        a = sum(1 for f in subset if f['label'] == 1)
        print(f"    {name}: {len(subset)} flows ({b} benign, {a} attack)")
    
    # Models
    models = {
        'BiMamba+Masking (v3)': os.path.join(SSL_DIR, 'bimamba_masking_v2.pth'),
        'BiMamba+CutMix  (v3)': os.path.join(SSL_DIR, 'bimamba_cutmix_v2.pth'),
        'BERT+Masking    (v2)': os.path.join(SSL_DIR, 'bert_masking_v2.pth'),
    }
    
    # Best method per previous analysis
    method = 'cosine_top10'
    
    all_results = {}
    
    print(f"\n[3] Running evaluation (method: {method})...")
    print()
    
    for model_name, ckpt_path in models.items():
        if not os.path.exists(ckpt_path):
            print(f"  SKIP {model_name}: not found"); continue
        
        print(f"  ╔══ {model_name} ══╗")
        
        if 'BERT' in model_name:
             encoder = BertEncoder().to(DEVICE)
        else:
             encoder = BiMambaEncoder().to(DEVICE)

        state = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        try:
             encoder.load_state_dict(state, strict=False)
        except Exception as e:
             print(f"Error loading {model_name}: {e}")
             continue

        encoder.eval()
        
        model_results = {}
        
        for ds_name, ds_flows in datasets.items():
            r = full_eval(encoder, ref_subset, ds_flows, method=method)
            model_results[ds_name] = r
            
            print(f"  ║ {ds_name:<16} │ AUC={r['auc']:.4f}  F1={r['f1']:.4f}  "
                  f"Prec={r['precision']:.4f}  Rec={r['recall']:.4f}  Acc={r['accuracy']:.4f}")
        
        all_results[model_name] = model_results
        print(f"  ╚{'═'*70}╝\n")
        
        del encoder; torch.cuda.empty_cache(); gc.collect()
    
    # Save results
    results_path = os.path.join(BASE, "results/crossdataset_full_results.json")
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {results_path}")
    
    # Print summary table
    print("\n" + "=" * 80)
    print("  SUMMARY TABLE (Best method per dataset)")
    print("=" * 80)
    print(f"\n{'Model':<30} {'Dataset':<18} {'AUC':>7} {'F1':>7} {'Prec':>7} {'Rec':>7} {'Acc':>7}")
    print("-" * 85)
    
    for model_name in models.keys():
        if model_name not in all_results: continue
        r_map = all_results[model_name]
        
        for ds_name in datasets.keys():
            r = r_map.get(ds_name, {})
            print(f"  {model_name:<28} {ds_name:<18} {r['auc']:>6.4f} "
                  f"{r['f1']:>6.4f} {r['precision']:>6.4f} {r['recall']:>6.4f} {r['accuracy']:>6.4f}")
        print()
    
    print("=" * 80)

if __name__ == "__main__":
    main()
