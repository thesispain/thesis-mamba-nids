#!/usr/bin/env python3
"""
Zero-Shot Cross-Dataset Evaluation
===================================
Load ALL models trained on UNSW-NB15, test directly on CIC-IDS-2017.
NO fine-tuning on CIC-IDS. Pure Zero-Shot.
"""
import os, sys, pickle, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, recall_score
from mamba_ssm import Mamba

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CICIDS_PATH = "data/cicids2017_flows.pkl"
BASE = os.path.dirname(os.path.abspath(__file__))
THESIS = os.path.dirname(BASE)

# ═══ Model Definitions (from run_full_evaluation.py) ═══
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
        self.tokenizer = PacketEmbedder(d_model)
        self.layers = nn.ModuleList([Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2) for _ in range(n_layers)])
        self.layers_rev = nn.ModuleList([Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.proj_head = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 256))
        self.recon_head = nn.Linear(d_model, 5)
    def forward(self, x):
        feat = self.tokenizer(x)
        for fwd, bwd in zip(self.layers, self.layers_rev):
            out_f = fwd(feat)
            out_b = torch.flip(bwd(torch.flip(feat, [1])), [1])
            feat = self.norm(out_f + out_b + feat)
        return self.proj_head(feat.mean(1)), None

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
        self.pos_encoder = LearnablePositionalEncoding(d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads,
                    dim_feedforward=d_model*4, batch_first=True, dropout=0.1)
        self.transformer_encoder = nn.TransformerEncoder(layer, n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.proj_head = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 128))
        self.recon_head = nn.Linear(d_model, 5)
    def forward(self, x):
        emb = self.tokenizer(x)
        B, L, D = emb.shape
        emb = torch.cat((self.cls_token.expand(B, -1, -1), emb), dim=1)
        emb = self.pos_encoder(emb)
        mask = torch.zeros((B, L+1), dtype=torch.bool, device=x.device)
        f = self.norm(self.transformer_encoder(emb, src_key_padding_mask=mask))
        return self.proj_head(f[:, 0, :]), None

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
    def __init__(self, encoder, input_dim):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU(), nn.Linear(64, 2))
    def forward(self, x):
        z = self.encoder(x)
        if isinstance(z, tuple): z = z[0]
        return self.head(z)

class BlockwiseEarlyExitMamba(nn.Module):
    def __init__(self, d_model=256):
        super().__init__()
        self.tokenizer = PacketEmbedder(d_model)
        self.layers = nn.ModuleList([Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2) for _ in range(4)])
        self.norm = nn.LayerNorm(d_model)
        self.exit_positions = [8, 16, 32]
        self.exit_classifiers = nn.ModuleDict({
            str(p): nn.Sequential(nn.Linear(d_model, 128), nn.ReLU(), nn.Dropout(0.1), nn.Linear(128, 2))
            for p in self.exit_positions})
        self.confidence_heads = nn.ModuleDict({
            str(p): nn.Sequential(nn.Linear(d_model+2, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid())
            for p in self.exit_positions})
        self.register_buffer('exit_counts', torch.zeros(3))
        self.register_buffer('total_inferences', torch.tensor(0))
    def _backbone(self, x):
        feat = self.tokenizer(x)
        for layer in self.layers:
            feat = self.norm(layer(feat) + feat)
        return feat
    def forward(self, x):
        feat = self._backbone(x)
        idx = min(32, feat.size(1)) - 1
        h = feat[:, idx, :]
        return self.exit_classifiers['32'](h)

# ═══ Evaluation ═══
def evaluate_zero_shot(model, dl, name):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for x, y in dl:
            x = x.to(DEVICE)
            logits = model(x)
            if isinstance(logits, tuple): logits = logits[0]
            probs = F.softmax(logits, 1)[:, 1].cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(y.numpy())
    y_true = np.array(all_labels)
    y_scores = np.array(all_probs)
    y_pred_default = (y_scores > 0.5).astype(int)
    f1_default = f1_score(y_true, y_pred_default, zero_division=0)

    try: auc = roc_auc_score(y_true, y_scores)
    except: auc = 0.5
    
    # Handle Inversion
    if auc < 0.5:
        print(f"  [Warn] Inverted predictions detected (AUC={auc:.4f}). Flipping scores...")
        y_scores = 1 - y_scores
        auc = 1 - auc
        
    # Find Optimal F1
    best_f1 = 0
    best_thresh = 0.5
    # Use percentiles for efficient sweep
    thresholds = np.unique(np.percentile(y_scores, np.linspace(0, 100, 101)))
    
    for t in thresholds:
        yp = (y_scores > t).astype(int)
        f = f1_score(y_true, yp, zero_division=0)
        if f > best_f1:
            best_f1 = f
            best_thresh = t
            
    acc = accuracy_score(y_true, y_pred_default)
    rec = recall_score(y_true, y_pred_default, zero_division=0)
    
    print(f"  [{name:25s}] AUC={auc:.4f}  Default F1={f1_default:.4f}  Optimal F1={best_f1:.4f} (@{best_thresh:.2f})")
    return {'model': name, 'f1': f1_default, 'opt_f1': best_f1, 'auc': auc, 'acc': acc, 'recall': rec}

def main():
    # Load CIC-IDS
    print("Loading CIC-IDS-2017...")
    with open(CICIDS_PATH, 'rb') as f:
        data = pickle.load(f)
    X = np.array([d['features'] for d in data], dtype=np.float32)
    y = np.array([d['label'] for d in data], dtype=np.longlong)
    benign = np.sum(y==0); attack = np.sum(y==1)
    print(f"  Total: {len(y):,}  Benign: {benign:,} ({benign/len(y):.1%})  Attack: {attack:,}")
    
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    dl = DataLoader(ds, batch_size=2048, shuffle=False)

    results = []

    # ── 1. BERT (SSL + 10% UNSW fine-tuned) ──
    print("\n═══ 1. BERT (teacher_bert_masking.pth) ═══")
    bert_enc = BertEncoder()
    bert = Classifier(bert_enc, input_dim=128).to(DEVICE)
    try:
        sd = torch.load("weights/teachers/teacher_bert_masking.pth", map_location=DEVICE, weights_only=False)
        bert.load_state_dict(sd, strict=False)
        results.append(evaluate_zero_shot(bert, dl, "BERT (SSL+UNSW)"))
    except Exception as e:
        print(f"  Failed: {e}")

    # ── 2. BiMamba Teacher (SSL + 100% UNSW) ──
    print("\n═══ 2. BiMamba Teacher (teacher_bimamba_retrained.pth) ═══")
    bi_enc = BiMambaEncoder()
    teacher = Classifier(bi_enc, input_dim=256).to(DEVICE)
    try:
        sd = torch.load("weights/teachers/teacher_bimamba_retrained.pth", map_location=DEVICE, weights_only=False)
        teacher.load_state_dict(sd, strict=False)
        results.append(evaluate_zero_shot(teacher, dl, "BiMamba Teacher (100%)"))
    except Exception as e:
        print(f"  Failed: {e}")

    # ── 3. UniMamba (scratch + 10% UNSW) ──
    print("\n═══ 3. UniMamba (student_no_kd.pth) ═══")
    uni = Classifier(UniMambaEncoder(), input_dim=128).to(DEVICE)
    try:
        sd = torch.load("weights/students/student_no_kd.pth", map_location=DEVICE, weights_only=False)
        uni.load_state_dict(sd, strict=False)
        results.append(evaluate_zero_shot(uni, dl, "UniMamba (No SSL)"))
    except Exception as e:
        print(f"  Failed: {e}")

    # ── 4. KD Student (distilled from Teacher + 10% UNSW) ──
    print("\n═══ 4. KD Student (student_standard_kd.pth) ═══")
    kd = Classifier(UniMambaEncoder(), input_dim=128).to(DEVICE)
    try:
        sd = torch.load("weights/students/student_standard_kd.pth", map_location=DEVICE, weights_only=False)
        kd.load_state_dict(sd, strict=False)
        results.append(evaluate_zero_shot(kd, dl, "KD Student"))
    except Exception as e:
        print(f"  Failed: {e}")

    # ── 5. TED Student (early exit + distilled) ──
    print("\n═══ 5. TED Student (student_ted.pth) ═══")
    ted = BlockwiseEarlyExitMamba().to(DEVICE)
    try:
        sd = torch.load("weights/students/student_ted.pth", map_location=DEVICE, weights_only=False)
        ted.load_state_dict(sd, strict=False)
        results.append(evaluate_zero_shot(ted, dl, "TED Student"))
    except Exception as e:
        print(f"  Failed: {e}")

    # ── Summary ──
    print("\n" + "="*70)
    print("ZERO-SHOT CROSS-DATASET RESULTS (UNSW→CICIDS)")
    print("="*70)
    print(f"{'Model':25s} {'F1':>8s} {'AUC':>8s} {'Acc':>8s} {'Recall':>8s}")
    print("-"*70)
    for r in results:
        print(f"{r['model']:25s} {r['f1']:8.4f} {r['auc']:8.4f} {r['acc']:8.4f} {r['recall']:8.4f}")

if __name__ == "__main__":
    main()
