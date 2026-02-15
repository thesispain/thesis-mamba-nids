#!/usr/bin/env python3
"""
CIC-IDS-2017 Full Pipeline Replication
======================================
Replicates the thesis methodology on CIC-IDS-2017:
1. Train BiMamba Teacher (Full Data) - Supervised
2. Train BERT Baseline (10% Data)
3. Train UniMamba Student (10% Data)
4. Train TED Student (10% Data) - Early Exit
5. Evaluate on CIC-IDS Test Set
6. Cross-Evaluate on UNSW-NB15
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
from mamba_ssm import Mamba

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
WEIGHTS   = os.path.join(THESIS_DIR, "weights_cicids"); os.makedirs(WEIGHTS, exist_ok=True)
RESULTS   = os.path.join(THESIS_DIR, "results_cicids"); os.makedirs(RESULTS, exist_ok=True)
PLOTS     = os.path.join(THESIS_DIR, "plots_cicids");   os.makedirs(PLOTS, exist_ok=True)

# Primary Data: CIC-IDS
DATA_FILE = os.path.join(THESIS_DIR, "data/cicids2017_flows.pkl")
# Cross Data: UNSW-NB15
CROSS_FILE = os.path.join(THESIS_DIR, "../Organized_Final/data/unswnb15_full/finetune_mixed.pkl")

# ── Hyperparameters ──────────────────────────────────────────────────
BATCH       = 64
LR          = 1e-4
TRAIN_PCT   = 0.10  # 10% for students
MAX_EPOCHS  = 15    # Faster run
PATIENCE    = 3

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
    print(f"Loading Primary: {DATA_FILE} ...")
    with open(DATA_FILE, 'rb') as f:
        full = pickle.load(f)
    labels = [d['label'] for d in full]
    print(f"  Total Flows: {len(full):,}")
    
    # Check max label
    max_label = max(labels)
    print(f"  Max Label: {max_label} (Binary assumption: {max_label==1})")
    
    train_all, test = train_test_split(full, test_size=0.3,
                                        random_state=42, stratify=labels)

    # 10% subset for students
    train_labels = [d['label'] for d in train_all]
    train_sub, _ = train_test_split(train_all, train_size=TRAIN_PCT,
                                     random_state=42, stratify=train_labels)

    # Split each into train/val (80/20)
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

def load_cross_data():
    print(f"Loading Cross: {CROSS_FILE} ...")
    with open(CROSS_FILE, 'rb') as f:
        data = pickle.load(f)
    print(f"  Cross Size: {len(data):,}")
    # Just return a loader for evaluation
    return DataLoader(FlowDataset(data), batch_size=1024)

# =====================================================================
#  2.  MODEL ARCHITECTURES (Same as UNSW)
# =====================================================================
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
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model*4, batch_first=True, dropout=0.1)
        self.transformer_encoder = nn.TransformerEncoder(layer, n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.proj_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(),
            nn.Linear(d_model, 128))
    def forward(self, x):
        emb = self.tokenizer(x)
        B, L, D = emb.shape
        cls_tokens = self.cls_token.expand(B, -1, -1)
        emb = torch.cat((cls_tokens, emb), dim=1)
        emb = self.pos_encoder(emb)
        mask = torch.zeros((B, L+1), dtype=torch.bool, device=x.device)
        f = self.norm(self.transformer_encoder(emb, src_key_padding_mask=mask))
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
        self.register_buffer('exit_counts', torch.zeros(self.n_exits, dtype=torch.long))
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
#  3.  TRAINING (Reuse from run_full_evaluation.py logic)
# =====================================================================
@torch.no_grad()
def eval_f1(model, loader, is_blockwise=False):
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
    opt  = optim.AdamW(model.parameters(), lr=LR)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=1, min_lr=1e-6)
    crit = nn.CrossEntropyLoss()
    best_f1 = 0.0; best_state = None; no_improve = 0

    for ep in range(max_epochs):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
            total_loss += loss.item()

        val_f1 = eval_f1(model, val_loader)
        sched.step(val_f1)
        print(f"  [{name}] ep {ep+1} loss={total_loss/len(train_loader):.4f} val_F1={val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1; best_state = copy.deepcopy(model.state_dict()); no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience: break

    if best_state: model.load_state_dict(best_state)
    return model

def train_kd(student, teacher, train_loader, val_loader, name,
             max_epochs=MAX_EPOCHS, patience=PATIENCE):
    student.train(); teacher.eval()
    opt = optim.AdamW(student.parameters(), lr=LR)
    crit = nn.CrossEntropyLoss()
    T = 2.0
    best_f1 = 0.0; best_state = None; no_improve = 0
    
    for ep in range(max_epochs):
        student.train()
        total = 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            with torch.no_grad(): t_logits = teacher(x)
            s_logits = student(x)
            hard = crit(s_logits, y)
            soft = F.kl_div(F.log_softmax(s_logits/T, 1), F.softmax(t_logits/T, 1), reduction='batchmean') * (T*T)
            loss = 0.5 * hard + 0.5 * soft
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item()
        
        val_f1 = eval_f1(student, val_loader)
        print(f"  [{name}] ep {ep+1} loss={total/len(train_loader):.4f} val_F1={val_f1:.4f}")
        if val_f1 > best_f1:
            best_f1 = val_f1; best_state = copy.deepcopy(student.state_dict()); no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience: break
            
    if best_state: student.load_state_dict(best_state)
    return student

def train_ted(model, teacher, train_loader, val_loader, name,
              max_epochs=MAX_EPOCHS, patience=PATIENCE):
    model.train(); teacher.eval()
    opt = optim.AdamW(model.parameters(), lr=LR)
    crit_ce = nn.CrossEntropyLoss()
    T = 2.0
    best_f1 = 0.0; best_state = None; no_improve = 0

    for ep in range(max_epochs):
        model.train()
        total = 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            with torch.no_grad(): t_logits = teacher(x)
            logits_list, conf_list = model.forward_train(x)
            loss = torch.tensor(0.0, device=DEVICE)
            for i, (logits, conf) in enumerate(zip(logits_list, conf_list)):
                loss += crit_ce(logits, y)
                loss += F.kl_div(F.log_softmax(logits/T, 1), F.softmax(t_logits/T, 1), reduction='batchmean') * (T*T) * 0.5
                correct = (logits.argmax(1) == y).float()
                loss += F.binary_cross_entropy(conf.clamp(1e-6, 1-1e-6), correct) * 0.2
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item()

        val_f1 = eval_f1(model, val_loader, is_blockwise=True)
        print(f"  [{name}] ep {ep+1} loss={total/len(train_loader):.4f} val_F1={val_f1:.4f}")
        if val_f1 > best_f1:
             best_f1 = val_f1; best_state = copy.deepcopy(model.state_dict()); no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience: break
            
    if best_state: model.load_state_dict(best_state)
    return model

def benchmark(model, loader, name, is_blockwise=False):
    model.eval()
    if is_blockwise: model.reset_stats()
    preds, labels = [], []
    t0 = time.time()
    n = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            if is_blockwise: logits, _ = model.forward_inference(x)
            else: logits = model(x)
            preds.extend(logits.argmax(1).cpu().numpy())
            labels.extend(y.numpy())
            n += x.size(0)
    
    elapsed = time.time() - t0
    f1 = f1_score(labels, preds, zero_division=0)
    acc = accuracy_score(labels, preds)
    thr = n / elapsed
    
    print(f"  [{name:20s}] F1={f1:.4f} Acc={acc:.4f} Thr={thr:.0f}/s")
    res = {"model": name, "f1": f1, "acc": acc, "throughput": thr}
    if is_blockwise:
        stats = model.get_exit_stats()
        if stats: res.update(stats)
    return res

# =====================================================================
#  4.  MAIN
# =====================================================================
def main():
    telegram_send("🚀 CIC-IDS Full Replication STARTED")
    full_train, full_val, sub_train, sub_val, test_data = load_data()
    
    full_tl = DataLoader(FlowDataset(full_train), batch_size=BATCH, shuffle=True)
    full_vl = DataLoader(FlowDataset(full_val),   batch_size=1024)
    sub_tl  = DataLoader(FlowDataset(sub_train),  batch_size=BATCH, shuffle=True)
    sub_vl  = DataLoader(FlowDataset(sub_val),    batch_size=1024)
    test_dl = DataLoader(FlowDataset(test_data),  batch_size=1024)
    cross_dl = load_cross_data()

    results = []

    # A. Teacher (BiMamba) - Full Data
    print("\n--- A. Teacher (BiMamba) ---")
    teacher = Classifier(BiMambaEncoder(), input_dim=256).to(DEVICE)
    teacher = train_classifier(teacher, full_tl, full_vl, "Teacher")
    r = benchmark(teacher, test_dl, "BiMamba Teacher")
    results.append(r)

    # B. BERT Baseline - 10%
    print("\n--- B. BERT Baseline ---")
    bert = Classifier(BertEncoder(), input_dim=128).to(DEVICE)
    bert = train_classifier(bert, sub_tl, sub_vl, "BERT")
    r = benchmark(bert, test_dl, "BERT Baseline")
    results.append(r)

    # C. UniMamba Baseline - 10%
    print("\n--- C. UniMamba Baseline ---")
    uni = Classifier(UniMambaEncoder(), input_dim=128).to(DEVICE)
    uni = train_classifier(uni, sub_tl, sub_vl, "UniMamba")
    r = benchmark(uni, test_dl, "UniMamba Baseline")
    results.append(r)

    # D. KD Student - 10%
    print("\n--- D. KD Student ---")
    kd = Classifier(UniMambaEncoder(), input_dim=128).to(DEVICE)
    kd = train_kd(kd, teacher, sub_tl, sub_vl, "KD Student")
    r = benchmark(kd, test_dl, "KD Student")
    results.append(r)

    # E. TED Student - 10%
    print("\n--- E. TED Student ---")
    ted = BlockwiseEarlyExitMamba().to(DEVICE)
    ted = train_ted(ted, teacher, sub_tl, sub_vl, "TED Student")
    r = benchmark(ted, test_dl, "TED Student", is_blockwise=True)
    results.append(r)
    
    # F. Cross-Eval on UNSW
    print("\n--- F. Cross-Eval (UNSW-NB15) ---")
    benchmark(teacher, cross_dl, "Teacher (UNSW)")
    benchmark(ted, cross_dl, "TED (UNSW)", is_blockwise=True)

    # Save
    with open(os.path.join(RESULTS, "cicids_results.json"), 'w') as f:
        json.dump(results, f, indent=2)
    
    telegram_send(f"✅ CIC-IDS Done. TED F1={results[-1]['f1']:.4f}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        telegram_send(f"❌ CIC-IDS Crash: {e}")
        traceback.print_exc()
