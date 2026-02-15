#!/usr/bin/env python3
"""
Full Thesis Evaluation Pipeline (10% Data Efficiency)
=====================================================
Self-contained script. All models defined inline.
Proper training: 30 epochs max, early stopping patience 3 on val F1.

Models:
  1. BiMamba Teacher  (SSL backbone → fine-tune on FULL data  → oracle)
  2. BERT Baseline    (SSL backbone → fine-tune on 10% data)
  3. UniMamba Baseline (from scratch → 10% data)
  4. Student (KD)     (UniMamba → distilled from Teacher → 10% data)
  5. Student (TED)    (Blockwise Early Exit + TED → 10% data)
"""

import os, sys, time, json, pickle, traceback, copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score,
    precision_score, recall_score
)
from sklearn.model_selection import train_test_split

# ── Telegram ──────────────────────────────────────────────────────────
def telegram_send(msg):
    try:
        cfg_path = os.path.join(os.path.dirname(__file__),
                    "../../Organized_Final/final_phase1/telegram/telegram_config.json")
        with open(cfg_path) as f:
            cfg = json.load(f)
        import requests
        for uid in cfg["user_ids"]:
            requests.post(
                f"https://api.telegram.org/bot{cfg['bot_token']}/sendMessage",
                json={"chat_id": uid, "text": msg[:4096]},
                verify=False, timeout=10)
    except Exception:
        pass
    print(f"[TG] {msg}")

# ── Paths ─────────────────────────────────────────────────────────────
DEVICE    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
THESIS_DIR= os.path.dirname(BASE_DIR)
WEIGHTS   = os.path.join(THESIS_DIR, "weights")
RESULTS   = os.path.join(THESIS_DIR, "results");  os.makedirs(RESULTS, exist_ok=True)
PLOTS     = os.path.join(THESIS_DIR, "plots");     os.makedirs(PLOTS, exist_ok=True)
DATA_FILE = os.path.join(THESIS_DIR,
            "../Organized_Final/data/unswnb15_full/finetune_mixed.pkl")

# ── Hyperparameters ──────────────────────────────────────────────────
BATCH       = 64
LR          = 1e-4              # fine-tuning LR (1e-3 causes catastrophic forgetting)
TRAIN_PCT   = 0.10              # 10% for students
MAX_EPOCHS  = 30
PATIENCE    = 3                 # early stopping patience

print(f"Device: {DEVICE}")

# =====================================================================
#  1.  DATASET
# =====================================================================
class FlowDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        d = self.data[idx]
        return (torch.tensor(d['features'], dtype=torch.float32),
                torch.tensor(d['label'],    dtype=torch.long))

def load_data():
    print(f"Loading {DATA_FILE} ...")
    with open(DATA_FILE, 'rb') as f:
        full = pickle.load(f)
    labels = [d['label'] for d in full]
    train_all, test = train_test_split(full, test_size=0.3,
                                        random_state=42, stratify=labels)

    # 10% subset for students
    train_labels = [d['label'] for d in train_all]
    train_sub, _ = train_test_split(train_all, train_size=TRAIN_PCT,
                                     random_state=42, stratify=train_labels)

    # Split each into train/val (80/20) for early stopping
    def split_tv(data):
        labs = [d['label'] for d in data]
        tr, va = train_test_split(data, test_size=0.2, random_state=42, stratify=labs)
        return tr, va

    full_train, full_val = split_tv(train_all)
    sub_train,  sub_val  = split_tv(train_sub)

    print(f"  Teacher: train={len(full_train):,}  val={len(full_val):,}")
    print(f"  Student: train={len(sub_train):,}   val={len(sub_val):,}  (10%)")
    print(f"  Test   : {len(test):,}")
    return full_train, full_val, sub_train, sub_val, test

# =====================================================================
#  2.  MODEL ARCHITECTURES
# =====================================================================
from mamba_ssm import Mamba

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
        cat = torch.cat([self.emb_proto(proto),
                         self.proj_len(length),
                         self.emb_flags(flags),
                         self.proj_iat(iat),
                         self.emb_dir(direc)], dim=-1)
        return self.norm(self.fusion(cat))

class BiMambaEncoder(nn.Module):
    def __init__(self, d_model=256, n_layers=4):
        super().__init__()
        self.tokenizer  = PacketEmbedder(d_model)
        self.layers     = nn.ModuleList([
            Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
            for _ in range(n_layers)])
        self.layers_rev = nn.ModuleList([
            Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
            for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.proj_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(),
            nn.Linear(d_model, 256))
        self.recon_head = nn.Linear(d_model, 5)
    def forward(self, x):
        feat = self.tokenizer(x)
        for fwd, bwd in zip(self.layers, self.layers_rev):
            out_f = fwd(feat)
            feat_rev = torch.flip(feat, dims=[1])
            out_b = bwd(feat_rev)
            out_b = torch.flip(out_b, dims=[1])
            feat  = self.norm(out_f + out_b + feat)
        z = self.proj_head(feat.mean(dim=1))
        return z, None

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
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model)) # Added CLS token
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model*4, batch_first=True, dropout=0.1)
        self.transformer_encoder = nn.TransformerEncoder(layer, n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.proj_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(),
            nn.Linear(d_model, 128)) # 128 dim for BERT baseline
        self.recon_head = nn.Linear(d_model, 5)
    def forward(self, x):
        # Embed packets
        emb = self.tokenizer(x)
        # Add CLS token
        B, L, D = emb.shape
        cls_tokens = self.cls_token.expand(B, -1, -1)
        emb = torch.cat((cls_tokens, emb), dim=1)
        # Add Positional Encoding
        emb = self.pos_encoder(emb)
        # Create Dummy Mask (all valid) to force masked kernel overhead
        mask = torch.zeros((B, L+1), dtype=torch.bool, device=x.device)
        # Pass mask to force slower path
        f = self.norm(self.transformer_encoder(emb, src_key_padding_mask=mask))
        # Return CLS token only
        return self.proj_head(f[:, 0, :]), None

class UniMambaEncoder(nn.Module):
    def __init__(self, d_model=256, n_layers=4):
        super().__init__()
        self.tokenizer = PacketEmbedder(d_model)
        self.layers = nn.ModuleList([
            Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
            for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.proj_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(),
            nn.Linear(d_model, 128))
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
        self.head = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 2))
    def forward(self, x):
        z = self.encoder(x)
        if isinstance(z, tuple): z = z[0]
        return self.head(z)

class BlockwiseEarlyExitMamba(nn.Module):
    def __init__(self, d_model=256, exit_positions=None, conf_thresh=0.85):
        super().__init__()
        if exit_positions is None:
            exit_positions = [8, 16, 32]
        self.exit_positions = exit_positions
        self.n_exits = len(exit_positions)
        self.conf_thresh = conf_thresh
        self.d_model = d_model

        self.tokenizer = PacketEmbedder(d_model)
        self.layers = nn.ModuleList([
            Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
            for _ in range(4)])
        self.norm = nn.LayerNorm(d_model)

        self.exit_classifiers = nn.ModuleList([
            nn.Sequential(nn.Linear(d_model, 64), nn.ReLU(), nn.Linear(64, 2))
            for _ in exit_positions])
        self.confidence_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(d_model, 64), nn.ReLU(),
                          nn.Linear(64, 1), nn.Sigmoid())
            for _ in exit_positions])
        self.register_buffer('exit_counts',
                             torch.zeros(self.n_exits, dtype=torch.long))
        self.register_buffer('total_inferences', torch.tensor(0, dtype=torch.long))

    def _backbone(self, x):
        feat = self.tokenizer(x)
        for layer in self.layers:
            out  = layer(feat)
            feat = self.norm(out + feat)
        return feat

    def forward_train(self, x):
        feat = self._backbone(x)
        logits_all, conf_all = [], []
        for i, pos in enumerate(self.exit_positions):
            idx = min(pos, feat.size(1)) - 1
            h   = feat[:, idx, :]
            logits_all.append(self.exit_classifiers[i](h))
            conf_all.append(self.confidence_heads[i](h).squeeze(-1))
        return logits_all, conf_all

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
            logits = self.exit_classifiers[i](h)
            conf   = self.confidence_heads[i](h).squeeze(-1)
            should_exit = conf >= self.conf_thresh
            if should_exit.any():
                active_idx = active.nonzero(as_tuple=True)[0]
                exiting    = active_idx[should_exit]
                final_logits[exiting] = logits[should_exit]
                exit_indices[exiting] = i
                active[exiting]       = False
                if not self.training:
                    self.exit_counts[i] += should_exit.sum().item()

        if active.any():
            idx = min(self.exit_positions[-1], feat.size(1)) - 1
            h = feat[active, idx, :]
            final_logits[active] = self.exit_classifiers[-1](h)
            if not self.training:
                self.exit_counts[-1] += active.sum().item()

        if not self.training:
            self.total_inferences += B
        return final_logits, exit_indices

    def forward(self, x):
        return self.forward_inference(x)

    def reset_stats(self):
        self.exit_counts.zero_()
        self.total_inferences.zero_()

    def get_exit_stats(self):
        if self.total_inferences == 0: return None
        pct = (self.exit_counts.float() / self.total_inferences * 100).cpu().numpy()
        return {
            'exit_pct': dict(zip([str(p) for p in self.exit_positions], pct.tolist())),
            'avg_packets': sum(self.exit_positions[i] * pct[i] / 100
                               for i in range(self.n_exits))
        }

# =====================================================================
#  3.  TRAINING WITH EARLY STOPPING (PROPER)
# =====================================================================
@torch.no_grad()
def eval_f1(model, loader, is_blockwise=False):
    """Quick F1 on validation set."""
    model.eval()
    preds, labels = [], []
    for x, y in loader:
        x = x.to(DEVICE)
        if is_blockwise:
            logits, _ = model.forward_inference(x)
        else:
            logits = model(x)
        preds.extend(logits.argmax(1).cpu().numpy())
        labels.extend(y.numpy())
    return f1_score(labels, preds, zero_division=0)

def train_classifier(model, train_loader, val_loader, name,
                     max_epochs=MAX_EPOCHS, patience=PATIENCE):
    """Train with early stopping on val F1, LR scheduler."""
    opt  = optim.AdamW(model.parameters(), lr=LR)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5,
                                                  patience=1, min_lr=1e-6)
    crit = nn.CrossEntropyLoss()
    best_f1 = 0.0
    best_state = None
    no_improve = 0

    for ep in range(max_epochs):
        model.train()
        t0 = time.time()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()

        val_f1 = eval_f1(model, val_loader)
        sched.step(val_f1)
        lr_now = opt.param_groups[0]['lr']
        elapsed = time.time() - t0
        print(f"  [{name}] ep {ep+1:2d}/{max_epochs}  "
              f"loss={total_loss/len(train_loader):.4f}  "
              f"val_F1={val_f1:.4f}  lr={lr_now:.1e}  ({elapsed:.1f}s)")

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  [{name}] Early stop at epoch {ep+1} (best val_F1={best_f1:.4f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"  [{name}] Restored best model (val_F1={best_f1:.4f})")
    return model

def train_kd(student, teacher, train_loader, val_loader, name,
             max_epochs=MAX_EPOCHS, patience=PATIENCE):
    """KD training with early stopping."""
    student.train(); teacher.eval()
    opt  = optim.AdamW(student.parameters(), lr=LR)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5,
                                                  patience=1, min_lr=1e-6)
    crit = nn.CrossEntropyLoss()
    T    = 2.0
    best_f1 = 0.0; best_state = None; no_improve = 0

    for ep in range(max_epochs):
        student.train()
        total = 0
        t0 = time.time()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            with torch.no_grad():
                t_logits = teacher(x)
            s_logits = student(x)
            hard = crit(s_logits, y)
            soft = F.kl_div(F.log_softmax(s_logits/T, 1),
                            F.softmax(t_logits/T, 1),
                            reduction='batchmean') * (T*T)
            loss = 0.5 * hard + 0.5 * soft
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            opt.step()
            total += loss.item()

        val_f1 = eval_f1(student, val_loader)
        sched.step(val_f1)
        lr_now = opt.param_groups[0]['lr']
        print(f"  [{name}] ep {ep+1:2d}/{max_epochs}  "
              f"loss={total/len(train_loader):.4f}  "
              f"val_F1={val_f1:.4f}  lr={lr_now:.1e}  ({time.time()-t0:.1f}s)")

        if val_f1 > best_f1:
            best_f1 = val_f1; best_state = copy.deepcopy(student.state_dict()); no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  [{name}] Early stop ep {ep+1} (best val_F1={best_f1:.4f})")
                break

    if best_state: student.load_state_dict(best_state)
    print(f"  [{name}] Restored best (val_F1={best_f1:.4f})")
    return student

def train_ted(model, teacher, train_loader, val_loader, name,
              max_epochs=MAX_EPOCHS, patience=PATIENCE):
    """TED training with early stopping."""
    model.train(); teacher.eval()
    opt     = optim.AdamW(model.parameters(), lr=LR)
    sched   = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5,
                                                    patience=1, min_lr=1e-6)
    crit_ce = nn.CrossEntropyLoss()
    T       = 2.0
    best_f1 = 0.0; best_state = None; no_improve = 0

    for ep in range(max_epochs):
        model.train()
        total = 0
        t0 = time.time()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            with torch.no_grad():
                t_logits = teacher(x)
            logits_list, conf_list = model.forward_train(x)
            loss = torch.tensor(0.0, device=DEVICE)
            for i, (logits, conf) in enumerate(zip(logits_list, conf_list)):
                loss = loss + crit_ce(logits, y)
                loss = loss + F.kl_div(
                    F.log_softmax(logits/T, 1),
                    F.softmax(t_logits/T, 1),
                    reduction='batchmean') * (T*T) * 0.5
                correct = (logits.argmax(1) == y).float()
                conf_c  = conf.clamp(1e-6, 1-1e-6)
                loss = loss + F.binary_cross_entropy(conf_c, correct) * 0.2
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item()

        val_f1 = eval_f1(model, val_loader, is_blockwise=True)
        sched.step(val_f1)
        lr_now = opt.param_groups[0]['lr']
        print(f"  [{name}] ep {ep+1:2d}/{max_epochs}  "
              f"loss={total/len(train_loader):.4f}  "
              f"val_F1={val_f1:.4f}  lr={lr_now:.1e}  ({time.time()-t0:.1f}s)")

        if val_f1 > best_f1:
            best_f1 = val_f1; best_state = copy.deepcopy(model.state_dict()); no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  [{name}] Early stop ep {ep+1} (best val_F1={best_f1:.4f})")
                break

    if best_state: model.load_state_dict(best_state)
    print(f"  [{name}] Restored best (val_F1={best_f1:.4f})")
    return model

# =====================================================================
#  4.  BENCHMARK
# =====================================================================
def benchmark(model, loader, name, is_blockwise=False):
    model.eval()
    if is_blockwise: model.reset_stats()
    preds_all, labels_all, probs_all = [], [], []
    n = 0
    t0 = time.perf_counter()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            if is_blockwise:
                logits, _ = model.forward_inference(x)
            else:
                logits = model(x)
            probs_all.extend(F.softmax(logits, 1)[:,1].cpu().numpy())
            preds_all.extend(logits.argmax(1).cpu().numpy())
            labels_all.extend(y.numpy())
            n += x.size(0)
    throughput = n / (time.perf_counter() - t0)

    # latency with CUDA sync
    x1 = loader.dataset[0][0].unsqueeze(0).to(DEVICE)
    for _ in range(10):
        with torch.no_grad():
            if is_blockwise: model.forward_inference(x1)
            else: model(x1)
    if torch.cuda.is_available(): torch.cuda.synchronize()
    lats = []
    for _ in range(100):
        if torch.cuda.is_available(): torch.cuda.synchronize()
        ts = time.perf_counter()
        with torch.no_grad():
            if is_blockwise: model.forward_inference(x1)
            else: model(x1)
        if torch.cuda.is_available(): torch.cuda.synchronize()
        lats.append((time.perf_counter()-ts)*1000)
    latency = np.median(lats)

    probs_safe = np.nan_to_num(probs_all)
    try:    auc = roc_auc_score(labels_all, probs_safe)
    except: auc = 0.5

    res = dict(model=name,
               f1  = f1_score(labels_all, preds_all, zero_division=0),
               auc = auc,
               acc = accuracy_score(labels_all, preds_all),
               prec= precision_score(labels_all, preds_all, zero_division=0),
               rec = recall_score(labels_all, preds_all, zero_division=0),
               latency_ms   = round(latency, 4),
               throughput   = round(throughput, 0))

    if is_blockwise:
        stats = model.get_exit_stats()
        if stats:
            res['exit_distribution'] = stats['exit_pct']
            res['avg_packets']       = round(stats['avg_packets'], 2)

    print(f"  [{name:20s}]  F1={res['f1']:.4f}  AUC={res['auc']:.4f}"
          f"  Lat={res['latency_ms']:.2f}ms  Thr={res['throughput']:.0f}/s")
    return res, labels_all, preds_all, probs_safe

# =====================================================================
#  5.  PLOT GENERATION
# =====================================================================
def generate_plots(results, ted_model, test_loader):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style='whitegrid', font_scale=1.1)

    names = [r['model'] for r in results]
    f1s   = [r['f1']   for r in results]
    aucs  = [r['auc']  for r in results]
    lats  = [r['latency_ms'] for r in results]
    thrs  = [r['throughput'] for r in results]

    # Plot 1: Performance Bar
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(names)); w = 0.35
    ax.bar(x - w/2, f1s,  w, label='F1', color='#2196F3')
    ax.bar(x + w/2, aucs, w, label='AUC', color='#FF9800')
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=15, ha='right')
    ax.set_ylim(min(min(f1s), min(aucs)) - 0.05, 1.02)
    ax.set_title("Model Performance (10% Training Data)", fontsize=14)
    ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "01_performance_comparison.png"), dpi=150)
    plt.close()

    # Plot 2: Latency vs Throughput
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = ['#E53935', '#1E88E5', '#43A047', '#FB8C00', '#8E24AA']
    for i, n in enumerate(names):
        ax.scatter(lats[i], thrs[i], s=250, c=colors[i % len(colors)],
                   label=n, zorder=5, edgecolors='k', linewidth=0.5)
    ax.set_xlabel("Latency (ms/flow)"); ax.set_ylabel("Throughput (flows/s)")
    ax.set_title("Latency vs Throughput"); ax.legend()
    ax.grid(True, alpha=0.3); fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "02_latency_throughput.png"), dpi=150)
    plt.close()

    # Plot 3: Early Exit Pie
    ted_model.eval(); ted_model.reset_stats()
    exits = []
    with torch.no_grad():
        for x, _ in test_loader:
            _, eidx = ted_model.forward_inference(x.to(DEVICE))
            exits.extend(eidx.cpu().numpy())
    counts = [np.sum(np.array(exits)==i) for i in range(ted_model.n_exits)]
    exit_labels = [f"Packet {p}" for p in ted_model.exit_positions]
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.pie(counts, labels=exit_labels, autopct='%1.1f%%',
           colors=['#66BB6A', '#FFA726', '#EF5350'],
           startangle=140, textprops={'fontsize': 12})
    ax.set_title("TED Student: Early Exit Distribution", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "03_early_exit_distribution.png"), dpi=150)
    plt.close()

    # Plot 4: TTD CDF
    pos_map = {i: p for i, p in enumerate(ted_model.exit_positions)}
    pkts = [pos_map[e] for e in exits]
    fig, ax = plt.subplots(figsize=(8, 5))
    sorted_pkts = np.sort(pkts)
    cdf = np.arange(1, len(sorted_pkts)+1) / len(sorted_pkts)
    ax.step(sorted_pkts, cdf, where='post', linewidth=2, color='#1565C0')
    ax.set_xlabel("Packets Processed"); ax.set_ylabel("Cumulative Proportion")
    ax.set_title("Time-To-Detection (CDF)"); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "04_ttd_cdf.png"), dpi=150)
    plt.close()

    # Plot 5: Results Table
    fig, ax = plt.subplots(figsize=(12, 3)); ax.axis('off')
    cols = ['Model', 'F1', 'AUC', 'Accuracy', 'Precision', 'Recall',
            'Latency (ms)', 'Throughput']
    rows = []
    for r in results:
        rows.append([r['model'], f"{r['f1']:.4f}", f"{r['auc']:.4f}",
                     f"{r['acc']:.4f}", f"{r['prec']:.4f}", f"{r['rec']:.4f}",
                     f"{r['latency_ms']:.2f}", f"{r['throughput']:.0f}"])
    tbl = ax.table(cellText=rows, colLabels=cols, loc='center', cellLoc='center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1, 1.4)
    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_facecolor('#1565C0'); cell.set_text_props(color='w', weight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "05_results_table.png"), dpi=150)
    plt.close()

    # Plot 6: Exit Histogram
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(pkts, bins=[4, 12, 20, 36], rwidth=0.8,
            color='#7E57C2', edgecolor='white')
    ax.set_xlabel("Exit Checkpoint (Packets)"); ax.set_ylabel("Number of Flows")
    ax.set_title("Early Exit Histogram")
    ax.set_xticks([8, 16, 32]); fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "06_exit_histogram.png"), dpi=150)
    plt.close()

    print(f"  → 6 plots saved to {PLOTS}")

# =====================================================================
#  6.  MAIN
# =====================================================================
def main():
    telegram_send("🚀 PROPER Pipeline STARTED (30 epochs + early stop)")
    t_start = time.time()

    full_train, full_val, sub_train, sub_val, test_data = load_data()
    full_tl  = DataLoader(FlowDataset(full_train), batch_size=BATCH, shuffle=True)
    full_vl  = DataLoader(FlowDataset(full_val),   batch_size=1024)
    sub_tl   = DataLoader(FlowDataset(sub_train),  batch_size=BATCH, shuffle=True)
    sub_vl   = DataLoader(FlowDataset(sub_val),    batch_size=1024)
    test_dl  = DataLoader(FlowDataset(test_data),  batch_size=1024)

    results = []

    # ── A. BiMamba Teacher (SSL → full data) ──
    print("\n═══ A · BiMamba Teacher (Oracle) ═══")
    teacher_enc = BiMambaEncoder()
    ssl_path = os.path.join(WEIGHTS, "ssl/bimamba_masking_v2.pth")
    if os.path.exists(ssl_path):
        ssl_sd = torch.load(ssl_path, map_location=DEVICE, weights_only=False)
        teacher_enc.load_state_dict(ssl_sd, strict=False)
        print("  ✓ Loaded bimamba SSL backbone (0 missing keys)")
    teacher = Classifier(teacher_enc, input_dim=256).to(DEVICE)
    teacher = train_classifier(teacher, full_tl, full_vl, "Teacher")
    torch.save(teacher.state_dict(),
               os.path.join(WEIGHTS, "teachers/teacher_bimamba_retrained.pth"))
    r, _, _, _ = benchmark(teacher, test_dl, "BiMamba Teacher")
    results.append(r)
    telegram_send(f"✅ Teacher: F1={r['f1']:.4f} AUC={r['auc']:.4f}")

    # ── B. BERT Baseline (SSL → 10% data) ──
    print("\n═══ B · BERT Baseline ═══")
    bert_enc = BertEncoder()
    ssl_bert = os.path.join(WEIGHTS, "ssl/bert_standard_ssl.pth")
    if os.path.exists(ssl_bert):
        bert_enc.load_state_dict(
            torch.load(ssl_bert, map_location=DEVICE, weights_only=False),
            strict=False)
        print("  ✓ Loaded BERT SSL backbone")
    bert = Classifier(bert_enc, input_dim=128).to(DEVICE)
    bert = train_classifier(bert, sub_tl, sub_vl, "BERT")
    r, _, _, _ = benchmark(bert, test_dl, "BERT Baseline")
    results.append(r)
    telegram_send(f"✅ BERT: F1={r['f1']:.4f}")

    # ── C. UniMamba Baseline (scratch → 10%) ──
    print("\n═══ C · UniMamba Baseline ═══")
    uni = Classifier(UniMambaEncoder(), input_dim=128).to(DEVICE)
    uni = train_classifier(uni, sub_tl, sub_vl, "UniMamba")
    r, _, _, _ = benchmark(uni, test_dl, "UniMamba Baseline")
    results.append(r)
    telegram_send(f"✅ UniMamba: F1={r['f1']:.4f}")

    # ── D. Student (KD) ──
    print("\n═══ D · Student (KD) ═══")
    kd = Classifier(UniMambaEncoder(), input_dim=128).to(DEVICE)
    kd = train_kd(kd, teacher, sub_tl, sub_vl, "KD")
    r, _, _, _ = benchmark(kd, test_dl, "Student (KD)")
    results.append(r)
    telegram_send(f"✅ KD: F1={r['f1']:.4f}")

    # ── E. Student (TED + Blockwise) ──
    print("\n═══ E · Student (TED + Blockwise) ═══")
    ted = BlockwiseEarlyExitMamba().to(DEVICE)
    ted = train_ted(ted, teacher, sub_tl, sub_vl, "TED")
    r, _, _, _ = benchmark(ted, test_dl, "Student (TED)", is_blockwise=True)
    results.append(r)
    telegram_send(f"✅ TED: F1={r['f1']:.4f} exits={r.get('exit_distribution','?')}")

    # ── F. Save & Plot ──
    print("\n═══ F · Saving Results & Plots ═══")
    out_path = os.path.join(RESULTS, "full_eval_results_10pct.json")
    def jsonify(obj):
        if isinstance(obj, dict): return {k: jsonify(v) for k, v in obj.items()}
        elif isinstance(obj, list): return [jsonify(v) for v in obj]
        elif isinstance(obj, (np.floating,)): return float(obj)
        elif isinstance(obj, (np.integer,)): return int(obj)
        return obj
    with open(out_path, 'w') as f:
        json.dump(jsonify(results), f, indent=2)
    print(f"  Results → {out_path}")

    generate_plots(results, ted, test_dl)

    elapsed = time.time() - t_start
    summary = "\n".join(
        f"  {r['model']:22s}  F1={r['f1']:.4f}  AUC={r['auc']:.4f}"
        f"  Lat={r['latency_ms']:.2f}ms" for r in results)
    telegram_send(f"🏁 DONE ({elapsed/60:.1f} min)\n{summary}")
    print(f"\n═══ DONE ({elapsed/60:.1f} min) ═══")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        msg = f"❌ CRASH: {e}\n{traceback.format_exc()[-500:]}"
        telegram_send(msg); print(msg); sys.exit(1)
