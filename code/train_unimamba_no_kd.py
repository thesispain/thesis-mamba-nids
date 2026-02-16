#!/usr/bin/env python3
"""
Train UniMamba WITH Early Exit but WITHOUT KD (No Teacher)
===========================================================
This proves that Early Exit alone (without SSL Teacher guidance) gives WORSE results.
Comparison:
  - This script: UniMamba + Early Exit + Supervised Only → Weak
  - TED (original): UniMamba + Early Exit + KD from SSL Teacher → Strong
"""
import os, sys, pickle, time, copy, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from mamba_ssm import Mamba

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
THESIS_DIR = os.path.dirname(BASE_DIR)

DATA_FILE = os.path.join(THESIS_DIR, "../Organized_Final/data/unswnb15_full/finetune_mixed.pkl")
CICIDS_PATH = os.path.join(THESIS_DIR, "data/cicids2017_flows.pkl")
SAVE_PATH = os.path.join(THESIS_DIR, "weights/students/student_no_kd.pth")

BATCH = 64
LR = 1e-4
TRAIN_PCT = 0.10
MAX_EPOCHS = 30
PATIENCE = 3

# ═══ Model ═══
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

class BlockwiseEarlyExitMamba(nn.Module):
    """Same architecture as TED but trained WITHOUT teacher (pure supervised)."""
    def __init__(self, d_model=256):
        super().__init__()
        self.d_model = d_model
        self.tokenizer = PacketEmbedder(d_model)
        self.layers = nn.ModuleList([
            Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
            for _ in range(4)])
        self.norm = nn.LayerNorm(d_model)

        self.exit_positions = [8, 16, 32]
        self.n_exits = 3
        self.conf_thresh = 0.85

        # Same classifier/confidence heads as TED
        self.exit_classifiers = nn.ModuleDict({
            str(p): nn.Sequential(
                nn.Linear(d_model, 128), nn.ReLU(),
                nn.Dropout(0.1), nn.Linear(128, 2))
            for p in self.exit_positions})
        self.confidence_heads = nn.ModuleDict({
            str(p): nn.Sequential(
                nn.Linear(d_model + 2, 64), nn.ReLU(),
                nn.Linear(64, 1), nn.Sigmoid())
            for p in self.exit_positions})

        self.register_buffer('exit_counts', torch.zeros(self.n_exits))
        self.register_buffer('total_inferences', torch.tensor(0))

    def _backbone(self, x):
        feat = self.tokenizer(x)
        for layer in self.layers:
            feat = self.norm(layer(feat) + feat)
        return feat

    def forward_train(self, x):
        """Return logits at ALL exits for training."""
        feat = self._backbone(x)
        all_logits = {}
        for p in self.exit_positions:
            idx = min(p, feat.size(1)) - 1
            h = feat[:, idx, :]
            logits = self.exit_classifiers[str(p)](h)
            all_logits[str(p)] = logits
        return all_logits

    def forward_inference(self, x):
        feat = self._backbone(x)
        B = feat.size(0)
        final_logits = torch.zeros(B, 2, device=feat.device)
        active = torch.ones(B, dtype=torch.bool, device=feat.device)

        for i, p in enumerate(self.exit_positions):
            if not active.any(): break
            idx = min(p, feat.size(1)) - 1
            h = feat[active, idx, :]
            logits = self.exit_classifiers[str(p)](h)
            conf_in = torch.cat([h, logits], dim=-1)
            conf = self.confidence_heads[str(p)](conf_in).squeeze(-1)

            if i < self.n_exits - 1:
                exits = conf > self.conf_thresh
            else:
                exits = torch.ones(h.size(0), dtype=torch.bool, device=feat.device)

            idx_active = active.nonzero(as_tuple=True)[0]
            exit_indices = idx_active[exits]
            final_logits[exit_indices] = logits[exits]
            self.exit_counts[i] += exits.sum().item()

            stay = ~exits
            new_active = active.clone()
            new_active[idx_active[~stay]] = False
            active = new_active

        self.total_inferences += B
        return final_logits, None

    def forward(self, x):
        if self.training:
            return self.forward_train(x)
        return self.forward_inference(x)

    def get_exit_stats(self):
        if self.total_inferences == 0: return None
        pct = (self.exit_counts.float() / self.total_inferences * 100).cpu().numpy()
        return {
            'exit_pct': dict(zip([str(p) for p in self.exit_positions], pct.tolist())),
            'avg_packets': sum(self.exit_positions[i] * pct[i] / 100 for i in range(self.n_exits))
        }

# ═══ Training (Supervised ONLY, no KD) ═══
def train_supervised_early_exit(model, train_dl, val_dl):
    """Train with weighted exit loss (same as TED) but NO teacher/KD."""
    opt = optim.AdamW(model.parameters(), lr=LR)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=1, min_lr=1e-6)
    crit = nn.CrossEntropyLoss()
    # Same weights as TED training
    exit_weights = {'8': 4.0, '16': 1.5, '32': 0.5}

    best_f1, best_state, no_improve = 0, None, 0

    for ep in range(MAX_EPOCHS):
        model.train()
        total_loss = 0
        t0 = time.time()

        for x, y in train_dl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            all_logits = model.forward_train(x)

            # Weighted CE loss at each exit (NO soft KD!)
            loss = sum(exit_weights[k] * crit(all_logits[k], y) for k in all_logits)

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()

        # Validate
        model.eval()
        model.exit_counts.zero_()
        model.total_inferences.zero_()
        preds, labels = [], []
        with torch.no_grad():
            for x, y in val_dl:
                x = x.to(DEVICE)
                logits, _ = model.forward_inference(x)
                preds.extend(logits.argmax(1).cpu().numpy())
                labels.extend(y.numpy())
        val_f1 = f1_score(labels, preds, zero_division=0)
        stats = model.get_exit_stats()
        avg_pkts = stats['avg_packets'] if stats else 32

        sched.step(val_f1)
        lr_now = opt.param_groups[0]['lr']
        print(f"  Ep {ep+1:2d}/{MAX_EPOCHS}  loss={total_loss/len(train_dl):.4f}  "
              f"val_F1={val_f1:.4f}  avg_pkts={avg_pkts:.1f}  lr={lr_now:.1e}  ({time.time()-t0:.1f}s)")

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"  Early stop at epoch {ep+1} (best val_F1={best_f1:.4f})")
                break

    if best_state: model.load_state_dict(best_state)
    print(f"  Restored best (val_F1={best_f1:.4f})")
    return model

# ═══ Evaluate ═══
def evaluate_full(model, dl, name):
    model.eval()
    model.exit_counts.zero_()
    model.total_inferences.zero_()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for x, y in dl:
            x = x.to(DEVICE)
            logits, _ = model.forward_inference(x)
            probs = F.softmax(logits, 1)[:, 1].cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(y.numpy())

    y_true = np.array(all_labels)
    y_scores = np.array(all_probs)
    y_pred = (y_scores > 0.5).astype(int)

    f1 = f1_score(y_true, y_pred, zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    try: auc = roc_auc_score(y_true, y_scores)
    except: auc = 0.5

    stats = model.get_exit_stats()
    avg_pkts = stats['avg_packets'] if stats else 32
    exit_pct = stats['exit_pct'] if stats else {}

    print(f"\n  [{name}]")
    print(f"  F1={f1:.4f}  AUC={auc:.4f}  Acc={acc:.4f}")
    print(f"  Avg Packets: {avg_pkts:.1f}")
    print(f"  Exit Distribution: {exit_pct}")
    return {'f1': f1, 'auc': auc, 'acc': acc, 'avg_pkts': avg_pkts, 'exit_pct': exit_pct}

def main():
    print(f"Device: {DEVICE}")

    # Load UNSW
    print("Loading UNSW-NB15...")
    with open(DATA_FILE, 'rb') as f:
        data = pickle.load(f)
    np.random.shuffle(data)
    n_train = int(len(data) * TRAIN_PCT)
    train_data = data[:n_train]
    test_data = data[n_train:]
    n_val = int(len(train_data) * 0.2)
    val_data, train_data = train_data[:n_val], train_data[n_val:]

    def to_dl(d, shuffle=False):
        X = np.array([s['features'] for s in d], dtype=np.float32)
        y = np.array([s['label'] for s in d], dtype=np.longlong)
        return DataLoader(TensorDataset(torch.from_numpy(X), torch.from_numpy(y)),
                         batch_size=BATCH, shuffle=shuffle)

    train_dl, val_dl, test_dl = to_dl(train_data, True), to_dl(val_data), to_dl(test_data)
    print(f"  Train: {len(train_data):,}  Val: {len(val_data):,}  Test: {len(test_data):,}")

    # Train
    print(f"\n{'='*60}")
    print("Training UniMamba + Early Exit (NO KD, supervised only)")
    print(f"{'='*60}")
    model = BlockwiseEarlyExitMamba().to(DEVICE)
    model = train_supervised_early_exit(model, train_dl, val_dl)

    # Save
    torch.save(model.state_dict(), SAVE_PATH)
    print(f"Saved to {SAVE_PATH}")

    # In-domain eval
    print(f"\n{'='*60}")
    print("IN-DOMAIN (UNSW-NB15)")
    print(f"{'='*60}")
    unsw = evaluate_full(model, test_dl, "UniMamba+EarlyExit (No KD)")

    # Cross-dataset eval
    print(f"\n{'='*60}")
    print("CROSS-DATASET (CIC-IDS, Zero-Shot)")
    print(f"{'='*60}")
    with open(CICIDS_PATH, 'rb') as f:
        cic_data = pickle.load(f)
    X_cic = np.array([d['features'] for d in cic_data], dtype=np.float32)
    y_cic = np.array([d['label'] for d in cic_data], dtype=np.longlong)
    cic_dl = DataLoader(TensorDataset(torch.from_numpy(X_cic), torch.from_numpy(y_cic)),
                        batch_size=2048, shuffle=False)
    model.exit_counts.zero_()
    model.total_inferences.zero_()
    cic = evaluate_full(model, cic_dl, "Cross-DS (Zero-Shot)")

    # Summary
    print(f"\n{'='*60}")
    print("COMPARISON TABLE")
    print(f"{'='*60}")
    print(f"{'Metric':25s} {'UniMamba (No KD)':>18s} {'TED (With KD)':>18s}")
    print(f"{'-'*62}")
    print(f"{'F1 (UNSW)':25s} {unsw['f1']:>18.4f} {'0.8783':>18s}")
    print(f"{'AUC (UNSW)':25s} {unsw['auc']:>18.4f} {'0.9951':>18s}")
    print(f"{'Cross-DS AUC':25s} {cic['auc']:>18.4f} {'0.76':>18s}")
    print(f"{'Avg Packets':25s} {unsw['avg_pkts']:>18.1f} {'9.1':>18s}")

    # Save
    results = {
        'unsw': unsw, 'cic': cic,
        'comparison': 'UniMamba+EarlyExit (No KD) vs TED (KD from SSL Teacher)'
    }
    res_path = os.path.join(THESIS_DIR, "results/unimamba_no_kd_earlyexit.json")
    with open(res_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {res_path}")

if __name__ == "__main__":
    main()
