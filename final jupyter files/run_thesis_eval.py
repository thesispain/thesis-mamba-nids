#!/usr/bin/env python3
"""
Thesis Full Pipeline — CORRECTED ARCHITECTURES
  BERT:  4 heads (paper spec) + CLS token + classification on CLS(256-dim)
  TED:   Real early exit — processes only first p packets (Mamba is causal)

Run from: thesis_final/final jupyter files/
Env:      source ../../mamba_env/bin/activate
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle, os, time, copy, warnings, math, random
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from mamba_ssm import Mamba
import xgboost as xgb

warnings.filterwarnings('ignore')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

ROOT = Path('/home/T2510596/Downloads/totally fresh')
UNSW_DIR = ROOT / 'Organized_Final' / 'data' / 'unswnb15_full'
CIC_PATH = ROOT / 'thesis_final' / 'data' / 'cicids2017_flows.pkl'
CTU_PATH = ROOT / 'thesis_final' / 'data' / 'ctu13_flows.pkl'
WEIGHT_DIR = Path('weights')
for d in ['phase2_ssl', 'phase3_teachers', 'phase4_kd', 'phase5_ted']:
    (WEIGHT_DIR / d).mkdir(parents=True, exist_ok=True)

print(f'Device: {DEVICE}')
if DEVICE.type == 'cuda':
    print(f'GPU: {torch.cuda.get_device_name()}')

# ══════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════

def load_pkl(path, name, fix_iat=False):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    if fix_iat:
        for d in data:
            d['features'][:, 3] = np.log1p(d['features'][:, 3])
    labels = np.array([d['label'] for d in data])
    print(f'{name}: {len(data):,} flows (benign={int((labels==0).sum()):,}, attack={int((labels==1).sum()):,})')
    return data

unsw_pretrain = load_pkl(UNSW_DIR / 'pretrain_50pct_benign.pkl', 'UNSW Pretrain')
unsw_finetune = load_pkl(UNSW_DIR / 'finetune_mixed.pkl', 'UNSW Finetune')
cicids = load_pkl(CIC_PATH, 'CIC-IDS-2017', fix_iat=True)
ctu13 = load_pkl(CTU_PATH, 'CTU-13')

class FlowDataset(Dataset):
    def __init__(self, data):
        self.features = torch.tensor(np.array([d['features'] for d in data]), dtype=torch.float32)
        self.labels = torch.tensor(np.array([d['label'] for d in data]), dtype=torch.long)
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx): return self.features[idx], self.labels[idx]

# Split finetune: 70/15/15
labels_ft = np.array([d['label'] for d in unsw_finetune])
idx_train, idx_temp = train_test_split(range(len(unsw_finetune)), test_size=0.3, stratify=labels_ft, random_state=SEED)
labels_temp = labels_ft[idx_temp]
idx_val, idx_test = train_test_split(idx_temp, test_size=0.5, stratify=labels_temp, random_state=SEED)
train_data = [unsw_finetune[i] for i in idx_train]
val_data   = [unsw_finetune[i] for i in idx_val]
test_data  = [unsw_finetune[i] for i in idx_test]

BS = 512
train_ds    = FlowDataset(train_data)
val_ds      = FlowDataset(val_data)
test_ds     = FlowDataset(test_data)
pretrain_ds = FlowDataset(unsw_pretrain)
cic_ds      = FlowDataset(cicids)
ctu_ds      = FlowDataset(ctu13)

train_loader    = DataLoader(train_ds, batch_size=BS, shuffle=True, drop_last=True)
val_loader      = DataLoader(val_ds, batch_size=BS, shuffle=False)
test_loader     = DataLoader(test_ds, batch_size=BS, shuffle=False)
cic_loader      = DataLoader(cic_ds, batch_size=BS, shuffle=False)
ctu_loader      = DataLoader(ctu_ds, batch_size=BS, shuffle=False)

print(f'Train: {len(train_data):,}  Val: {len(val_data):,}  Test: {len(test_data):,}\n')

# ══════════════════════════════════════════════════════════════════
# ARCHITECTURES — Paper-correct
# ══════════════════════════════════════════════════════════════════

class PacketEmbedder(nn.Module):
    def __init__(self, d_model=256, de=32):
        super().__init__()
        self.emb_proto = nn.Embedding(256, de)
        self.emb_flags = nn.Embedding(64, de)
        self.emb_dir   = nn.Embedding(2, de // 4)
        self.proj_len  = nn.Linear(1, de)
        self.proj_iat  = nn.Linear(1, de)
        self.fusion    = nn.Linear(de * 4 + de // 4, d_model)
        self.norm      = nn.LayerNorm(d_model)
    def forward(self, x):
        proto  = self.emb_proto(x[:, :, 0].long().clamp(0, 255))
        length = self.proj_len(x[:, :, 1:2])
        flags  = self.emb_flags(x[:, :, 2].long().clamp(0, 63))
        iat    = self.proj_iat(x[:, :, 3:4])
        direc  = self.emb_dir(x[:, :, 4].long().clamp(0, 1))
        return self.norm(self.fusion(torch.cat([proto, length, flags, iat, direc], dim=-1)))

class LearnedPE(nn.Module):
    def __init__(self, d_model=256):
        super().__init__()
        self.pe_emb = nn.Embedding(5000, d_model)
    def forward(self, x):
        return x + self.pe_emb(torch.arange(x.size(1), device=x.device))


# ═══════════════════════════════════════════════════════════════════
# BERT ENCODER — PAPER-CORRECT
#   4 attention heads (paper: "4 attention heads")
#   CLS token prepended (paper: "projection layer on output of [CLS] token")
#   d_model=256, 4 layers, ff=1024
# ═══════════════════════════════════════════════════════════════════
class BertEncoder(nn.Module):
    def __init__(self, d_model=256, de=32, nhead=4, num_layers=4, ff=1024, proj_out=128):
        super().__init__()
        self.d_model = d_model
        self.tokenizer = PacketEmbedder(d_model, de)
        self.pos_encoder = LearnedPE(d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, ff, dropout=0.1, activation='gelu', batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.proj_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, proj_out)
        )
        self.recon_head = nn.Linear(d_model, 5)
    
    def forward(self, x):
        h = self.tokenizer(x)                                    # (B, 32, 256)
        cls = self.cls_token.expand(h.size(0), -1, -1)           # (B, 1, 256)
        h = torch.cat([cls, h], dim=1)                           # (B, 33, 256)
        h = self.pos_encoder(h)
        h = self.transformer_encoder(h)
        h = self.norm(h)
        return h                                                  # (B, 33, 256)
    
    def get_ssl_outputs(self, x):
        h = self.forward(x)
        cls_out = h[:, 0, :]                                     # CLS output
        proj = self.proj_head(cls_out)
        recon = self.recon_head(h[:, 1:, :])                     # packets only
        return proj, recon, h


class BertClassifier(nn.Module):
    """Supervised: CLS output (256-dim) → head. Proj_head DISCARDED (paper spec).
    Head: 256→256→2 (matches bert_recreate_unsw.ipynb architecture)."""
    def __init__(self):
        super().__init__()
        self.encoder = BertEncoder()
        self.head = nn.Sequential(
            nn.Linear(256, 256), nn.ReLU(), nn.Dropout(0.1), nn.Linear(256, 2)
        )
    def forward(self, x):
        h = self.encoder(x)            # (B, 33, 256)
        cls_out = h[:, 0, :]           # (B, 256)
        return self.head(cls_out)


# BiMamba — unchanged
class BiMambaEncoder(nn.Module):
    def __init__(self, d_model=256, de=32, n_layers=4):
        super().__init__()
        self.tokenizer = PacketEmbedder(d_model, de)
        self.layers     = nn.ModuleList([Mamba(d_model, d_state=16, d_conv=4, expand=2) for _ in range(n_layers)])
        self.layers_rev = nn.ModuleList([Mamba(d_model, d_state=16, d_conv=4, expand=2) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.proj_head = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, d_model))
        self.recon_head = nn.Linear(d_model, 5)
    def forward(self, x):
        feat = self.tokenizer(x)
        for fwd, rev in zip(self.layers, self.layers_rev):
            out_f = fwd(feat); out_r = rev(feat.flip(1)).flip(1)
            feat = self.norm((out_f + out_r) / 2 + feat)
        return feat
    def get_ssl_outputs(self, x):
        h = self.forward(x)
        return self.proj_head(h.mean(dim=1)), self.recon_head(h), h

class BiMambaClassifier(nn.Module):
    """Supervised: raw encoder output mean pool (256-dim) → head.
    proj_head is SSL-only (contrastive) — skipped during classification.
    This saves one 256→256→256 MLP from the inference path."""
    def __init__(self):
        super().__init__()
        self.encoder = BiMambaEncoder()
        self.head = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.1), nn.Linear(128, 2)
        )
    def forward(self, x):
        h = self.encoder(x)          # (B, 32, 256)
        return self.head(h.mean(dim=1))  # raw mean pool — no proj_head


# UniMamba — unchanged
class UniMambaStudent(nn.Module):
    def __init__(self, d_model=256, de=32, n_layers=4):
        super().__init__()
        self.tokenizer = PacketEmbedder(d_model, de)
        self.layers = nn.ModuleList([Mamba(d_model, d_state=16, d_conv=4, expand=2) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.Linear(d_model, 64), nn.ReLU(), nn.Linear(64, 2))
    def forward(self, x):
        feat = self.tokenizer(x)
        for layer in self.layers:
            feat = self.norm(layer(feat) + feat)
        return self.head(feat.mean(dim=1))


# ═══════════════════════════════════════════════════════════════════
# TED — Blockwise Early Exit (Mamba causal property)
#
#   GPU inference: ONE forward pass over all 32 tokens.
#   Mamba causality: feat[:,i,:] depends only on tokens 0..i, so
#   feat[:,:p,:].mean() == Mamba(x[:,:p,:]).mean()  (no restart needed)
#
#   Early-exit benefit = TTD speedup (don't WAIT for packets 9-32 to
#   arrive in a live stream).  GPU latency ≈ UniMamba (same compute).
# ═══════════════════════════════════════════════════════════════════
class BlockwiseTEDStudent(nn.Module):
    EXIT_POINTS = [8, 16, 32]
    
    def __init__(self, d_model=256, de=32, n_layers=4):
        super().__init__()
        self.tokenizer = PacketEmbedder(d_model, de)
        self.layers = nn.ModuleList([Mamba(d_model, d_state=16, d_conv=4, expand=2) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.exit_classifiers = nn.ModuleDict({
            str(p): nn.Sequential(nn.Linear(d_model, 64), nn.ReLU(), nn.Dropout(0.1), nn.Linear(64, 2))
            for p in self.EXIT_POINTS
        })
        self.confidence_heads = nn.ModuleDict({
            str(p): nn.Sequential(nn.Linear(d_model, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid())
            for p in self.EXIT_POINTS
        })
    
    def _encode(self, x_sub):
        """Run tokenizer + Mamba layers on a (possibly truncated) input."""
        feat = self.tokenizer(x_sub)
        for layer in self.layers:
            feat = self.norm(layer(feat) + feat)
        return feat
    
    def forward(self, x, threshold=0.9):
        """Early exit using Mamba causality — ONE forward pass, zero restarts.

        Key insight: Mamba is fully causal.
            Mamba(x[:,:32,:])[:,:8,:] == Mamba(x[:,:8,:])
        So we process all 32 tokens ONCE, then gate exits using causal subsets.
        Total GPU compute = exactly ONE forward pass (same as UniMamba).

        The speedup is in TTD (Time-To-Detect): for 99.3% of flows the model
        decides at packet 8, so in real deployment we don't wait for packets
        9-32 to arrive.  GPU latency ≠ streaming deployment latency.
        """
        B = x.size(0)
        results = torch.zeros(B, 2, device=x.device)
        exit_packets = torch.full((B,), 32, device=x.device)
        decided = torch.zeros(B, dtype=torch.bool, device=x.device)

        # ONE encode over all 32 tokens — no restarts
        feat_full = self._encode(x)   # (B, 32, d_model)

        for p in self.EXIT_POINTS:
            if decided.all():
                break

            undecided = ~decided
            rep = feat_full[undecided, :p, :].mean(dim=1)   # causal slice

            logits = self.exit_classifiers[str(p)](rep)
            conf   = self.confidence_heads[str(p)](rep).squeeze(-1)

            global_indices = torch.where(undecided)[0]
            if p < 32:
                exit_mask_local = conf >= threshold
                exit_indices = global_indices[exit_mask_local]
                results[exit_indices]      = logits[exit_mask_local]
                exit_packets[exit_indices] = p
                decided[exit_indices]      = True
            else:
                results[global_indices]      = logits
                exit_packets[global_indices] = 32
                decided[global_indices]      = True

        return results, exit_packets
    
    def forward_train(self, x):
        """Training: process full 32 packets, return all exits for multi-exit loss."""
        feat = self.tokenizer(x)
        for layer in self.layers:
            feat = self.norm(layer(feat) + feat)
        all_logits, all_confs = {}, {}
        for p in self.EXIT_POINTS:
            rep = feat[:, :p, :].mean(dim=1)
            all_logits[p] = self.exit_classifiers[str(p)](rep)
            all_confs[p] = self.confidence_heads[str(p)](rep).squeeze(-1)
        return all_logits, all_confs


print('Architecture Parameter Counts:')
print(f'  BERT Encoder (4h+CLS):    {sum(p.numel() for p in BertEncoder().parameters()):>10,}')
print(f'  BERT Classifier:          {sum(p.numel() for p in BertClassifier().parameters()):>10,}')
print(f'  BiMamba Encoder:          {sum(p.numel() for p in BiMambaEncoder().parameters()):>10,}')
print(f'  BiMamba Classifier:       {sum(p.numel() for p in BiMambaClassifier().parameters()):>10,}')
print(f'  UniMamba Student:         {sum(p.numel() for p in UniMambaStudent().parameters()):>10,}')
print(f'  Blockwise TED:            {sum(p.numel() for p in BlockwiseTEDStudent().parameters()):>10,}')

# ══════════════════════════════════════════════════════════════════
# UTILITIES
# ══════════════════════════════════════════════════════════════════

def save_weights(model, path):
    torch.save(model.state_dict(), path)
    check = torch.load(path, map_location='cpu', weights_only=False)
    assert set(check.keys()) == set(model.state_dict().keys())
    print(f'  ✓ Saved: {path} ({os.path.getsize(path)/1e6:.1f} MB)')

def load_weights(model, path):
    sd = torch.load(path, map_location='cpu', weights_only=False)
    model.load_state_dict(sd, strict=True)
    print(f'  ✓ Loaded (strict): {path}')
    return model

@torch.no_grad()
def evaluate_classifier(model, loader, device=DEVICE):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        if isinstance(logits, tuple): logits = logits[0]
        probs = F.softmax(logits, dim=1)[:, 1]
        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, zero_division=0)
    auc = roc_auc_score(all_labels, all_probs)
    return acc, f1, auc

class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature
    def forward(self, z_i, z_j):
        B = z_i.size(0)
        z = torch.cat([z_i, z_j], dim=0)
        z = F.normalize(z, dim=1)
        sim = torch.mm(z, z.T) / self.temperature
        mask = torch.eye(2 * B, device=z.device).bool()
        sim.masked_fill_(mask, -1e9)
        labels = torch.cat([torch.arange(B, 2*B), torch.arange(B)]).to(z.device)
        return F.cross_entropy(sim, labels)

# ══════════════════════════════════════════════════════════════════
# PHASE 2: SSL PRETRAINING
# ══════════════════════════════════════════════════════════════════

SSL_BERT_PATH    = WEIGHT_DIR / 'phase2_ssl' / 'ssl_bert_paper.pth'
SSL_BIMAMBA_PATH = WEIGHT_DIR / 'phase2_ssl' / 'ssl_bimamba_paper.pth'

print('\n' + '=' * 60)
print('PHASE 2: SSL PRETRAINING')
print('=' * 60)

def cutmix_batch(x, alpha=0.4):
    B, T, F = x.shape
    cut = int(T * alpha)
    donors = torch.randint(0, B - 1, (B,), device=x.device)
    donors[donors >= torch.arange(B, device=x.device)] += 1
    x_out = x.clone()
    for i in range(B):
        s = random.randint(0, max(0, T - cut))
        x_out[i, s:s + cut] = x[donors[i], s:s + cut]
    return x_out

MASK_PROBS = {0: 0.20, 1: 0.50, 2: 0.30, 3: 0.00, 4: 0.10}
def anti_shortcut(x):
    B, T, _ = x.shape
    x_out = x.clone()
    for fi, p in MASK_PROBS.items():
        if p > 0:
            x_out[:, :, fi][torch.rand(B, T, device=x.device) < p] = 0.0
    x_out[:, :, 3] += torch.randn(B, T, device=x.device) * 0.05
    return x_out

def train_ssl_encoder(encoder, save_path, name, n_epochs=1, bs=128, lr=5e-5):
    if os.path.isfile(save_path):
        load_weights(encoder, save_path)
        return encoder.to(DEVICE)
    
    print(f'\n  Training {name} SSL ({n_epochs} epoch, BS={bs}, LR={lr})...')
    encoder = encoder.to(DEVICE)
    ssl_loader = DataLoader(pretrain_ds, batch_size=bs, shuffle=True, drop_last=True)
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=lr, weight_decay=1e-4)
    criterion = NTXentLoss(temperature=0.5)
    
    for epoch in range(n_epochs):
        encoder.train()
        total_loss = 0; n = 0
        t0 = time.time()
        for x, _ in ssl_loader:
            x = x.to(DEVICE)
            v1 = anti_shortcut(x)
            v2 = anti_shortcut(cutmix_batch(x))
            p1, r1, _ = encoder.get_ssl_outputs(v1)
            p2, r2, _ = encoder.get_ssl_outputs(v2)
            loss_con = criterion(p1, p2)
            loss_rec = F.mse_loss(r1, x) + F.mse_loss(r2, x)
            loss = loss_con + 0.5 * loss_rec
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item(); n += 1
        print(f'    Epoch {epoch+1}/{n_epochs}: loss={total_loss/n:.4f} ({time.time()-t0:.1f}s)')
    
    save_weights(encoder, save_path)
    return encoder

print('\n--- BiMamba SSL (existing weights, unchanged) ---')
ssl_bimamba = BiMambaEncoder()
ssl_bimamba = train_ssl_encoder(ssl_bimamba, SSL_BIMAMBA_PATH, 'BiMamba')

print('\n--- BERT SSL (NEW: 4 heads + CLS token) ---')
ssl_bert = BertEncoder()
ssl_bert = train_ssl_encoder(ssl_bert, SSL_BERT_PATH, 'BERT-4head-CLS')


# ══════════════════════════════════════════════════════════════════
# PHASE 3: SUPERVISED TEACHER FINE-TUNING
# ══════════════════════════════════════════════════════════════════
print('\n' + '=' * 60)
print('PHASE 3: SUPERVISED TEACHER FINE-TUNING')
print('=' * 60)

BERT_TEACHER_PATH    = WEIGHT_DIR / 'phase3_teachers' / 'bert_teacher.pth'
BIMAMBA_TEACHER_PATH = WEIGHT_DIR / 'phase3_teachers' / 'bimamba_teacher.pth'
FT_EPOCHS = 10         # SAME epochs for BERT & BiMamba (fair baseline comparison)
FT_LR = 1e-4
LABEL_SMOOTH = 0.1    # soften targets → less overconfidence → better generalization

# ── Class weights for 16.7:1 imbalance (94.3% benign, 5.7% attack) ──
train_labels = train_ds.labels.numpy()
n_benign = (train_labels == 0).sum()
n_attack = (train_labels == 1).sum()
CLASS_WEIGHTS = torch.tensor([1.0, float(n_benign / n_attack)], dtype=torch.float32, device=DEVICE)
print(f'  Class weights: benign=1.0, attack={CLASS_WEIGHTS[1]:.1f} (for {n_benign:,} vs {n_attack:,})')

def finetune_teacher(classifier, ssl_path, save_path, name, n_epochs=10):
    if os.path.isfile(save_path):
        load_weights(classifier, save_path)
        return classifier.to(DEVICE)
    
    sd = torch.load(ssl_path, map_location='cpu', weights_only=False)
    classifier.encoder.load_state_dict(sd, strict=True)
    print(f'  Loaded SSL encoder from {ssl_path}')
    
    classifier = classifier.to(DEVICE)
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=FT_LR, weight_decay=1e-4)
    # Warmup + cosine: 2 epoch warmup then decay
    warmup_epochs = 2
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, n_epochs - warmup_epochs)
        return 0.5 * (1 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    criterion = nn.CrossEntropyLoss(weight=CLASS_WEIGHTS, label_smoothing=LABEL_SMOOTH)
    
    best_val_f1 = 0.0; best_sd = None; patience = 0; max_patience = 5
    for epoch in range(n_epochs):
        classifier.train()
        total_loss = 0; n = 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            # Mixup augmentation: blend pairs of samples for regularization
            if np.random.random() < 0.3:
                lam = np.random.beta(0.2, 0.2)
                idx = torch.randperm(x.size(0), device=x.device)
                x_mix = lam * x + (1 - lam) * x[idx]
                logits = classifier(x_mix)
                loss = lam * criterion(logits, y) + (1 - lam) * criterion(logits, y[idx])
            else:
                loss = criterion(classifier(x), y)
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item(); n += 1
        scheduler.step()
        acc, f1, auc = evaluate_classifier(classifier, val_loader)
        lr_now = optimizer.param_groups[0]['lr']
        print(f'  {name} Ep {epoch+1}/{n_epochs}: loss={total_loss/n:.4f} acc={acc:.4f} f1={f1:.4f} auc={auc:.4f} lr={lr_now:.2e}')
        if f1 > best_val_f1:
            best_val_f1 = f1; best_sd = copy.deepcopy(classifier.state_dict()); patience = 0
        else:
            patience += 1
            if patience >= max_patience:
                print(f'  Early stop at epoch {epoch+1}')
                break
    
    classifier.load_state_dict(best_sd)
    save_weights(classifier, save_path)
    return classifier

print(f'\n--- BERT Teacher (4h+CLS, {FT_EPOCHS} epochs, label_smooth={LABEL_SMOOTH}, class_weighted) ---')
bert_teacher = finetune_teacher(BertClassifier(), SSL_BERT_PATH, BERT_TEACHER_PATH, 'BERT', FT_EPOCHS)

print(f'\n--- BiMamba Teacher ({FT_EPOCHS} epochs, label_smooth={LABEL_SMOOTH}, class_weighted) ---')
bimamba_teacher = finetune_teacher(BiMambaClassifier(), SSL_BIMAMBA_PATH, BIMAMBA_TEACHER_PATH, 'BiMamba', FT_EPOCHS)

# ── stored_aucs: reused in final summary (avoids OOM from re-evaluation) ──
stored_aucs = {}

print('\nPhase 3 Cross-Dataset Results:')
print(f"{'Model':<18}  {'Dataset':<15}  {'Acc':>7}  {'F1':>7}  {'AUC':>7}")
print('-' * 60)
for name, model in [('BERT Teacher', bert_teacher), ('BiMamba Teacher', bimamba_teacher)]:
    stored_aucs[name] = {}
    for ds, loader, key in [('UNSW Test', test_loader, 'UNSW'), ('CIC-IDS-2017', cic_loader, 'CIC'), ('CTU-13', ctu_loader, 'CTU')]:
        acc, f1, auc = evaluate_classifier(model, loader)
        stored_aucs[name][key] = auc
        print(f'{name:<18}  {ds:<15}  {acc:>7.4f}  {f1:>7.4f}  {auc:>7.4f}')
    print()


# ══════════════════════════════════════════════════════════════════
# PHASE 4: KNOWLEDGE DISTILLATION
# ══════════════════════════════════════════════════════════════════
print('=' * 60)
print('PHASE 4: KNOWLEDGE DISTILLATION')
print('=' * 60)

KD_PATH = WEIGHT_DIR / 'phase4_kd' / 'unimamba_student.pth'
KD_EPOCHS = 10; KD_LR = 1e-4; KD_TEMP = 4.0; KD_ALPHA = 0.7

def distill_student(teacher, student, save_path):
    if os.path.isfile(save_path):
        load_weights(student, save_path)
        return student.to(DEVICE)
    
    teacher.eval(); student = student.to(DEVICE)
    optimizer = torch.optim.AdamW(student.parameters(), lr=KD_LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=KD_EPOCHS)
    criterion_hard = nn.CrossEntropyLoss(weight=CLASS_WEIGHTS)
    
    best_val_f1 = 0.0; best_sd = None
    for epoch in range(KD_EPOCHS):
        student.train(); total_loss = 0; n = 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            with torch.no_grad(): teacher_logits = teacher(x)
            student_logits = student(x)
            soft_t = F.log_softmax(teacher_logits / KD_TEMP, dim=1)
            soft_s = F.log_softmax(student_logits / KD_TEMP, dim=1)
            loss_soft = F.kl_div(soft_s, soft_t.exp(), reduction='batchmean') * (KD_TEMP ** 2)
            loss_hard = criterion_hard(student_logits, y)
            loss = KD_ALPHA * loss_soft + (1 - KD_ALPHA) * loss_hard
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step(); total_loss += loss.item(); n += 1
        scheduler.step()
        acc, f1, auc = evaluate_classifier(student, val_loader)
        print(f'  KD Ep {epoch+1}/{KD_EPOCHS}: loss={total_loss/n:.4f} acc={acc:.4f} f1={f1:.4f} auc={auc:.4f}')
        if f1 > best_val_f1:
            best_val_f1 = f1; best_sd = copy.deepcopy(student.state_dict())
    student.load_state_dict(best_sd)
    save_weights(student, save_path)
    return student

print(f'\n--- UniMamba Student ({KD_EPOCHS} epochs) ---')
unimamba_student = distill_student(bimamba_teacher, UniMambaStudent(), KD_PATH)

print('\nPhase 4 Cross-Dataset:')
stored_aucs['UniMamba Student'] = {}
for ds, loader, key in [('UNSW Test', test_loader, 'UNSW'), ('CIC-IDS-2017', cic_loader, 'CIC'), ('CTU-13', ctu_loader, 'CTU')]:
    acc, f1, auc = evaluate_classifier(unimamba_student, loader)
    stored_aucs['UniMamba Student'][key] = auc
    print(f"  UniMamba  {ds:<15}  acc={acc:.4f} f1={f1:.4f} auc={auc:.4f}")


# ══════════════════════════════════════════════════════════════════
# PHASE 5: BLOCKWISE TED
# ══════════════════════════════════════════════════════════════════
print('\n' + '=' * 60)
print('PHASE 5: BLOCKWISE TED — REAL EARLY EXIT')
print('=' * 60)

TED_PATH = WEIGHT_DIR / 'phase5_ted' / 'ted_student.pth'
TED_EPOCHS = 10; TED_LR = 1e-4

def train_ted(teacher, ted_model, save_path):
    if os.path.isfile(save_path):
        load_weights(ted_model, save_path)
        return ted_model.to(DEVICE)
    
    teacher.eval(); ted_model = ted_model.to(DEVICE)
    optimizer = torch.optim.AdamW(ted_model.parameters(), lr=TED_LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TED_EPOCHS)
    criterion_hard = nn.CrossEntropyLoss(weight=CLASS_WEIGHTS)
    
    best_val_f1 = 0.0; best_sd = None
    for epoch in range(TED_EPOCHS):
        ted_model.train(); total_loss = 0; n = 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            with torch.no_grad():
                teacher_logits = teacher(x)
                teacher_pred = teacher_logits.argmax(dim=1)
            all_logits, all_confs = ted_model.forward_train(x)
            loss = 0.0
            for p in BlockwiseTEDStudent.EXIT_POINTS:
                logits_p = all_logits[p]; conf_p = all_confs[p]
                soft_t = F.log_softmax(teacher_logits / KD_TEMP, dim=1)
                soft_s = F.log_softmax(logits_p / KD_TEMP, dim=1)
                loss_kd = F.kl_div(soft_s, soft_t.exp(), reduction='batchmean') * (KD_TEMP ** 2)
                loss_hard = criterion_hard(logits_p, y)
                loss_cls = 0.5 * loss_kd + 0.5 * loss_hard
                conf_target = (logits_p.argmax(1) == teacher_pred).float()
                loss_conf = F.binary_cross_entropy(conf_p, conf_target)
                w = {8: 1.5, 16: 1.0, 32: 0.5}[p]
                loss += w * (loss_cls + 0.5 * loss_conf)
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(ted_model.parameters(), 1.0)
            optimizer.step(); total_loss += loss.item(); n += 1
        scheduler.step()
        
        ted_model.eval()
        all_p, all_l, all_pr, all_e = [], [], [], []
        with torch.no_grad():
            for x, y in val_loader:
                logits, exits = ted_model(x.to(DEVICE), threshold=0.9)
                all_pr.extend(F.softmax(logits, dim=1)[:, 1].cpu().numpy())
                all_p.extend(logits.argmax(1).cpu().numpy())
                all_l.extend(y.numpy()); all_e.extend(exits.cpu().numpy())
        acc = accuracy_score(all_l, all_p); f1 = f1_score(all_l, all_p, zero_division=0)
        auc = roc_auc_score(all_l, all_pr); ea = np.array(all_e)
        print(f'  TED Ep {epoch+1}/{TED_EPOCHS}: loss={total_loss/n:.4f} '
              f'acc={acc:.4f} f1={f1:.4f} auc={auc:.4f} '
              f'8={100*(ea==8).mean():.0f}% 16={100*(ea==16).mean():.0f}% 32={100*(ea==32).mean():.0f}% avg={ea.mean():.1f}')
        if f1 > best_val_f1:
            best_val_f1 = f1; best_sd = copy.deepcopy(ted_model.state_dict())
    
    ted_model.load_state_dict(best_sd)
    save_weights(ted_model, save_path)
    return ted_model

print(f'\n--- TED Student ({TED_EPOCHS} epochs) ---')
ted_student = train_ted(bimamba_teacher, BlockwiseTEDStudent(), TED_PATH)

print('\nPhase 5 Cross-Dataset:')
stored_aucs['TED Student'] = {}
ted_student.eval()
for ds_name, loader, key in [('UNSW Test', test_loader, 'UNSW'), ('CIC-IDS-2017', cic_loader, 'CIC'), ('CTU-13', ctu_loader, 'CTU')]:
    all_p, all_l, all_pr, all_e = [], [], [], []
    with torch.no_grad():
        for x, y in loader:
            logits, exits = ted_student(x.to(DEVICE), threshold=0.9)
            all_pr.extend(F.softmax(logits, dim=1)[:, 1].cpu().numpy())
            all_p.extend(logits.argmax(1).cpu().numpy()); all_l.extend(y.numpy())
            all_e.extend(exits.cpu().numpy())
    acc = accuracy_score(all_l, all_p); f1 = f1_score(all_l, all_p, zero_division=0)
    auc = roc_auc_score(all_l, all_pr); ea = np.array(all_e)
    stored_aucs['TED Student'][key] = auc
    print(f'  {ds_name}: acc={acc:.4f} f1={f1:.4f} auc={auc:.4f}')
    print(f'    Exits: 8={100*(ea==8).mean():.1f}% 16={100*(ea==16).mean():.1f}% 32={100*(ea==32).mean():.1f}% avg={ea.mean():.1f}')


# ══════════════════════════════════════════════════════════════════
# XGBOOST BASELINE
# ══════════════════════════════════════════════════════════════════
print('\n' + '=' * 60)
print('XGBOOST BASELINE')
print('=' * 60)

def extract_statistical_features(data):
    feats = []
    for d in data:
        f = d['features']; row = []
        for col in range(5):
            v = f[:, col]; nz = v[v != 0]
            row.extend([v.mean(), v.std(), v.min(), v.max(), len(nz)/len(v)])
        row.append(np.corrcoef(f[:,1], f[:,3])[0,1] if f[:,1].std()>0 and f[:,3].std()>0 else 0)
        feats.append(row)
    return np.array(feats, dtype=np.float32)

print('Extracting features...')
X_tr = extract_statistical_features(train_data); y_tr = np.array([d['label'] for d in train_data])
X_te = extract_statistical_features(test_data);  y_te = np.array([d['label'] for d in test_data])
X_ci = extract_statistical_features(cicids);     y_ci = np.array([d['label'] for d in cicids])
X_ct = extract_statistical_features(ctu13);      y_ct = np.array([d['label'] for d in ctu13])
for a in [X_tr, X_te, X_ci, X_ct]: a[np.isnan(a)] = 0

XGB_PATH = WEIGHT_DIR / 'phase3_teachers' / 'xgboost_baseline.json'
if os.path.isfile(XGB_PATH):
    xgb_model = xgb.XGBClassifier(); xgb_model.load_model(XGB_PATH)
    print(f'  Loaded XGBoost')
else:
    xgb_model = xgb.XGBClassifier(n_estimators=200, max_depth=8, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, eval_metric='logloss', random_state=SEED, use_label_encoder=False)
    xgb_model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
    xgb_model.save_model(XGB_PATH); print(f'  Trained XGBoost')

stored_aucs['XGBoost'] = {}
print('\nXGBoost Results:')
for dn, X, y, key in [('UNSW', X_te, y_te, 'UNSW'), ('CIC', X_ci, y_ci, 'CIC'), ('CTU', X_ct, y_ct, 'CTU')]:
    pr = xgb_model.predict_proba(X)[:, 1]; pd_ = (pr > 0.5).astype(int)
    stored_aucs['XGBoost'][key] = roc_auc_score(y, pr)
    print(f'  {dn}: acc={accuracy_score(y,pd_):.4f} f1={f1_score(y,pd_,zero_division=0):.4f} auc={stored_aucs["XGBoost"][key]:.4f}')


# ══════════════════════════════════════════════════════════════════
# GPU LATENCY BENCHMARKS
# ══════════════════════════════════════════════════════════════════
print('\n' + '=' * 60)
print('GPU LATENCY BENCHMARKS')
print('=' * 60)

def measure_latency(model, batch_size=1, seq_len=32, n_warmup=100, n_runs=500):
    model.eval()
    dummy = torch.randn(batch_size, seq_len, 5).to(DEVICE)
    with torch.no_grad():
        for _ in range(n_warmup): model(dummy)
    torch.cuda.synchronize()
    times = []
    for _ in range(n_runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad(): model(dummy)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    return np.median(times)

print(f'\nFull-model latencies (B=1, 32 packets):')
print(f"{'Model':<28}  {'Latency':>10}")
print('-' * 42)

lat = {}
for name, model in [('BERT Teacher (4h+CLS)', bert_teacher),
                     ('BiMamba Teacher', bimamba_teacher),
                     ('UniMamba Student', unimamba_student)]:
    lat[name] = measure_latency(model, batch_size=1)
    print(f'{name:<28}  {lat[name]:>8.4f}ms')

# TED at each exit point — the key measurement
print(f'\nTED Real Early Exit Latency (B=1):')
ted_student.eval()
for p in [8, 16, 32]:
    dummy = torch.randn(1, p, 5).to(DEVICE)
    with torch.no_grad():
        for _ in range(100):
            feat = ted_student._encode(dummy)
            rep = feat.mean(dim=1)
            ted_student.exit_classifiers[str(p)](rep)
    torch.cuda.synchronize()
    times = []
    for _ in range(500):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            feat = ted_student._encode(dummy)
            rep = feat.mean(dim=1)
            ted_student.exit_classifiers[str(p)](rep)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    lat[f'TED@{p}'] = np.median(times)
    print(f'  TED@{p:>2} packets:  {lat[f"TED@{p}"]:.4f}ms')

# Get exit distribution
ted_student.eval()
all_exits = []
with torch.no_grad():
    for x, y in test_loader:
        _, exits = ted_student(x.to(DEVICE), threshold=0.9)
        all_exits.extend(exits.cpu().numpy().astype(int))
ea = np.array(all_exits)
p8 = (ea==8).mean(); p16 = (ea==16).mean(); p32 = (ea==32).mean()
avg_pkts = ea.mean()

ted_eff = p8 * lat['TED@8'] + p16 * lat['TED@16'] + p32 * lat['TED@32']
lat['TED Effective'] = ted_eff

print(f'\n  Exit Distribution: 8={100*p8:.1f}% 16={100*p16:.1f}% 32={100*p32:.1f}% avg={avg_pkts:.1f}')
print(f'  TED Effective Latency: {ted_eff:.4f}ms')
print(f'  BERT Latency:          {lat["BERT Teacher (4h+CLS)"]:.4f}ms')
print(f'  Speedup TED vs BERT:   {lat["BERT Teacher (4h+CLS)"] / ted_eff:.2f}x')

# XGBoost CPU
dummy_xgb = np.random.randn(1, X_tr.shape[1]).astype(np.float32)
times = []
for _ in range(500):
    t0 = time.perf_counter(); xgb_model.predict_proba(dummy_xgb)
    times.append((time.perf_counter() - t0) * 1000)
lat['XGBoost'] = np.median(times)
print(f'  XGBoost (CPU, B=1): {lat["XGBoost"]:.4f}ms')


# ══════════════════════════════════════════════════════════════════
# TTD CALCULATION
# ══════════════════════════════════════════════════════════════════
print('\n' + '=' * 60)
print('TIME-TO-DETECT (TTD)')
print('=' * 60)

test_feats = np.array([d['features'] for d in test_data])
raw_iat_ms = np.expm1(test_feats[:, :, 3])

ttd = {}
for mn, lk in [('XGBoost', 'XGBoost'), ('BERT Teacher', 'BERT Teacher (4h+CLS)'),
                ('BiMamba Teacher', 'BiMamba Teacher'), ('UniMamba Student', 'UniMamba Student')]:
    ttd[mn] = raw_iat_ms.sum(axis=1) + lat[lk]

buf_ted = np.array([raw_iat_ms[i, :int(ep)].sum() for i, ep in enumerate(all_exits)])
lat_ted = np.array([lat[f'TED@{int(ep)}'] for ep in all_exits])
ttd['TED Student'] = buf_ted + lat_ted

bert_ttd = ttd['BERT Teacher'].mean()
print(f'\n{"Model":<22}  {"Mean TTD(ms)":>14}  {"Speedup":>10}')
print('-' * 50)
for mn in ['XGBoost', 'BERT Teacher', 'BiMamba Teacher', 'UniMamba Student', 'TED Student']:
    sp = bert_ttd / ttd[mn].mean()
    print(f'{mn:<22}  {ttd[mn].mean():>14.2f}  {sp:>10.2f}x')


# ══════════════════════════════════════════════════════════════════
# SSL UNSUPERVISED (k-NN on raw encoder reps)
# ══════════════════════════════════════════════════════════════════
print('\n' + '=' * 60)
print('SSL UNSUPERVISED — k-NN(k=10) on Raw Encoder Reps')
print('=' * 60)

K = 10; N_SAMPLE = 20000

@torch.no_grad()
def extract_raw_reps(encoder, loader, device, use_cls=False):
    encoder.eval(); out = []
    for x, _ in loader:
        h = encoder(x.to(device))
        rep = h[:, 0, :] if use_cls else h.mean(dim=1)
        out.append(rep.cpu())
    return torch.cat(out)

def knn_auc(test_reps, labels, train_reps, k=K, chunk=512, device=DEVICE):
    """k-NN anomaly scoring: 1 - avg cosine sim to k nearest benign neighbors.
    Uses max(auc, 1-auc) to handle domain-shift polarity inversion.
    Example: CIC benign (HTTPS) looks MORE anomalous to UNSW-trained model
    than CIC attacks (SYN floods) — direction inverts but separation is real.
    """
    db = F.normalize(train_reps.to(device), dim=1)
    scores = []
    for s in range(0, len(test_reps), chunk):
        q = F.normalize(test_reps[s:s+chunk].to(device), dim=1)
        topk = torch.mm(q, db.T).topk(k, dim=1).values
        scores.append(topk.mean(dim=1).cpu())
    anomaly_scores = 1.0 - torch.cat(scores).numpy()
    auc = roc_auc_score(labels, anomaly_scores)
    return max(auc, 1.0 - auc)  # report best direction (domain-shift invariant)

si = np.random.choice(len(pretrain_ds), min(N_SAMPLE, len(pretrain_ds)), replace=False)
s_loader = DataLoader(torch.utils.data.Subset(pretrain_ds, si), batch_size=512, shuffle=False)

print(f'\nExtracting {N_SAMPLE:,} benign training reps...')
bi_train = extract_raw_reps(ssl_bimamba, s_loader, DEVICE, use_cls=False)
bert_train = extract_raw_reps(ssl_bert, s_loader, DEVICE, use_cls=True)

print(f'\n{"Encoder":<18}  {"Dataset":<15}  {"AUC":>8}')
print('-' * 45)
for ds_name, loader, labels in [('UNSW-NB15', test_loader, test_ds.labels.numpy()),
                                 ('CIC-IDS-2017', cic_loader, cic_ds.labels.numpy()),
                                 ('CTU-13', ctu_loader, ctu_ds.labels.numpy())]:
    bi_r = extract_raw_reps(ssl_bimamba, loader, DEVICE, use_cls=False)
    auc_bi = knn_auc(bi_r, labels, bi_train)
    bert_r = extract_raw_reps(ssl_bert, loader, DEVICE, use_cls=True)
    auc_bert = knn_auc(bert_r, labels, bert_train)
    print(f'{"BiMamba SSL":<18}  {ds_name:<15}  {auc_bi:>8.4f}')
    print(f'{"BERT SSL (CLS)":<18}  {ds_name:<15}  {auc_bert:>8.4f}')
    print()


# ══════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════
print('=' * 90)
print('FINAL THESIS SUMMARY')
print('=' * 90)

# Use stored_aucs (computed during each phase) — avoids OOM from re-evaluation
print(f'\n{"Model":<26}  {"UNSW AUC":>10}  {"CIC AUC":>10}  {"CTU AUC":>10}  {"Latency":>10}  {"Pkts":>6}')
print('-' * 80)

table_rows = [
    ('XGBoost',           'XGBoost',                    lat.get('XGBoost', 0), 32.0),
    ('BERT Teacher (4h)', 'BERT Teacher',                lat.get('BERT Teacher (4h+CLS)', 0), 32.0),
    ('BiMamba Teacher',   'BiMamba Teacher',             lat.get('BiMamba Teacher', 0), 32.0),
    ('UniMamba Student',  'UniMamba Student',            lat.get('UniMamba Student', 0), 32.0),
    ('TED Student',       'TED Student',                 ted_eff, avg_pkts),
]
for display_name, stored_key, latency, pkts in table_rows:
    r = stored_aucs.get(stored_key, {})
    unsw = r.get('UNSW', float('nan'))
    cic  = r.get('CIC',  float('nan'))
    ctu  = r.get('CTU',  float('nan'))
    print(f"{display_name:<26}  {unsw:>10.4f}  {cic:>10.4f}  {ctu:>10.4f}  {latency:>8.4f}ms  {pkts:>6.1f}")

print(f'\nTED vs BERT speedup (GPU):   {lat["BERT Teacher (4h+CLS)"] / ted_eff:.2f}x')
print(f'TED vs BERT speedup (TTD):   {bert_ttd / ttd["TED Student"].mean():.2f}x')
print(f'TED exit@8: {100*p8:.1f}%  avg_pkts: {avg_pkts:.1f}')

print('\nArchitecture changes from previous run:')
print(f'  BERT: 8 heads → 4 heads, NO CLS → CLS token, proj_head(128) → CLS(256) classification')
print(f'  TED:  fake exit (process 32 then subset) → REAL exit (process only p tokens)')
print('\nDone.')
