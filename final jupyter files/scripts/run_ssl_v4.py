#!/usr/bin/env python3
"""
UniMamba SSL v4 — clean design:
  • ONE augmentation style (CutMix + mild feature dropout) applied
    independently to both views — no asymmetric aug strategies.
  • LOSS = NT-Xent (contrastive) + λ * next-packet prediction (autoregressive)
    No MSE reconstruction head.
  • Next-packet loss: h[t] predicts raw x[t+1] via a small Linear head.
    This is the natural SSL objective for a causal sequential model.

Produces weights/self_distill/unimamba_ssl_v4.pth
Then compares v2, v3, v4 at @8 and @32 on CIC-IDS cross-dataset eval.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle, os, time, copy, random, warnings
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

try:
    from mamba_ssm import Mamba
except ImportError:
    raise ImportError("activate mamba_env first: source mamba_env/bin/activate")

warnings.filterwarnings('ignore')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED   = 42
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

ROOT     = Path('/home/T2510596/Downloads/totally fresh')
UNSW_DIR = ROOT / 'Organized_Final' / 'data' / 'unswnb15_full'
CIC_PATH = ROOT / 'thesis_final' / 'data' / 'cicids2017_flows.pkl'
WD       = Path('weights')
(WD / 'self_distill').mkdir(parents=True, exist_ok=True)

print(f'Device: {DEVICE}')

# ── Hyperparameters ────────────────────────────────────────────────
SSL_EPOCHS   = 10
SSL_BS       = 128
SSL_LR       = 5e-5
LAMBDA_NEXT  = 0.5    # weight of next-packet prediction loss
TEMP         = 0.5    # NT-Xent temperature
K            = 10     # k-NN neighbors
N_SAMPLE     = 20000  # benign reference DB size

SSL_V4_PATH = WD / 'self_distill' / 'unimamba_ssl_v4.pth'

# ── Data ──────────────────────────────────────────────────────────
def load_pkl(path, name, fix_iat=False):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    if fix_iat:
        for d in data:
            d['features'][:, 3] = np.log1p(d['features'][:, 3])
    labels = np.array([d['label'] for d in data])
    print(f'  {name}: {len(data):,} flows  '
          f'(benign={int((labels==0).sum()):,}, attack={int((labels==1).sum()):,})')
    return data

print('\nLoading data...')
unsw_pretrain = load_pkl(UNSW_DIR / 'pretrain_50pct_benign.pkl', 'UNSW Pretrain')
unsw_finetune = load_pkl(UNSW_DIR / 'finetune_mixed.pkl',        'UNSW Finetune')
cicids        = load_pkl(CIC_PATH,                                'CIC-IDS-2017', fix_iat=True)

class FlowDataset(Dataset):
    def __init__(self, data):
        self.features = torch.tensor(
            np.array([d['features'] for d in data]), dtype=torch.float32)
        self.labels = torch.tensor(
            np.array([d['label'] for d in data]), dtype=torch.long)
    def __len__(self):  return len(self.labels)
    def __getitem__(self, i): return self.features[i], self.labels[i]

labels_ft = np.array([d['label'] for d in unsw_finetune])
idx_tr, idx_tmp = train_test_split(
    range(len(unsw_finetune)), test_size=0.3, stratify=labels_ft, random_state=SEED)
idx_val, idx_tst = train_test_split(
    idx_tmp, test_size=0.5, stratify=labels_ft[idx_tmp], random_state=SEED)

pretrain_ds = FlowDataset(unsw_pretrain)
test_ds     = FlowDataset([unsw_finetune[i] for i in idx_tst])
cic_ds      = FlowDataset(cicids)

pretrain_loader = DataLoader(pretrain_ds, batch_size=512, shuffle=False)
test_loader     = DataLoader(test_ds,     batch_size=512, shuffle=False)
cic_loader      = DataLoader(cic_ds,      batch_size=512, shuffle=False)

# ── Model ─────────────────────────────────────────────────────────
class PacketEmbedder(nn.Module):
    """
    Embeds raw 5-feature packet [proto, pkt_len, flags, iat, direction]
    into d-dimensional vectors using mixed embeddings + projections.
    """
    def __init__(self, d=256, de=32):
        super().__init__()
        self.emb_proto = nn.Embedding(256, de)
        self.emb_flags = nn.Embedding(64,  de)
        self.emb_dir   = nn.Embedding(2,   de // 4)
        self.proj_len  = nn.Linear(1, de)
        self.proj_iat  = nn.Linear(1, de)
        self.fusion    = nn.Linear(de * 4 + de // 4, d)
        self.norm      = nn.LayerNorm(d)

    def forward(self, x):
        return self.norm(self.fusion(torch.cat([
            self.emb_proto(x[:, :, 0].long().clamp(0, 255)),
            self.proj_len( x[:, :, 1:2]),
            self.emb_flags(x[:, :, 2].long().clamp(0, 63)),
            self.proj_iat( x[:, :, 3:4]),
            self.emb_dir(  x[:, :, 4].long().clamp(0, 1)),
        ], dim=-1)))


class UniMambaSSLv4(nn.Module):
    """
    Causal (forward-only) Mamba encoder with:
      - proj_head  : for NT-Xent contrastive loss
      - next_head  : predicts next packet raw features (h[t] → x_raw[t+1])
                     No reconstruction head (MSE dropped).
    """
    def __init__(self, d=256, de=32, nl=4):
        super().__init__()
        self.tokenizer = PacketEmbedder(d, de)
        self.layers    = nn.ModuleList([
            Mamba(d, d_state=16, d_conv=4, expand=2) for _ in range(nl)
        ])
        self.norm      = nn.LayerNorm(d)

        # Contrastive projection head (NT-Xent)
        self.proj_head = nn.Sequential(
            nn.Linear(d, d), nn.GELU(), nn.Linear(d, 128)
        )

        # Next-packet prediction head  h[t] → x_raw[t+1]
        # Predicts all 5 raw features (float)
        self.next_head = nn.Sequential(
            nn.Linear(d, d // 2), nn.GELU(), nn.Linear(d // 2, 5)
        )

    def encode(self, x):
        """Causal encoding: h[t] sees only packets 0..t."""
        h = self.tokenizer(x)
        for layer in self.layers:
            h = self.norm(layer(h) + h)
        return h   # [B, T, d]

    def forward(self, x):
        """
        Returns:
          proj : [B, 128]     — mean-pooled projection for NT-Xent
          h    : [B, T, d]    — full hidden states (for next-packet loss)
        """
        h    = self.encode(x)
        proj = self.proj_head(h.mean(1))
        return proj, h


# ── Loss functions ────────────────────────────────────────────────
class NTXentLoss(nn.Module):
    """Standard NT-Xent / InfoNCE with temperature τ."""
    def __init__(self, temperature=0.5):
        super().__init__()
        self.t = temperature

    def forward(self, z1, z2):
        B  = z1.size(0)
        z  = F.normalize(torch.cat([z1, z2], dim=0), dim=1)   # [2B, 128]
        sim = torch.mm(z, z.T) / self.t                        # [2B, 2B]
        sim.masked_fill_(torch.eye(2 * B, device=z.device).bool(), -1e9)
        labels = torch.cat([
            torch.arange(B, 2 * B, device=z.device),
            torch.arange(B,        device=z.device),
        ])
        return F.cross_entropy(sim, labels)


def next_packet_loss(h, x_raw):
    """
    Autoregressive next-packet prediction.

    h_pred  = model.next_head(h[:, :-1, :])   shape [B, T-1, 5]
    target  = x_raw[:, 1:, :]                 shape [B, T-1, 5]

    Uses the ORIGINAL (un-augmented) x_raw as the target — cleaner signal.
    """
    return h


# ── Augmentation ─────────────────────────────────────────────────
def cutmix(x, alpha=0.4):
    """
    CutMix: for each sample, swap a contiguous window of ~alpha*T packets
    with the matching window from a randomly chosen other sample.
    """
    B, T, _ = x.shape
    cut  = int(T * alpha)
    idx  = torch.randperm(B, device=x.device)           # random pairing
    x_out = x.clone()
    for i in range(B):
        s = random.randint(0, max(0, T - cut))
        x_out[i, s:s + cut] = x[idx[i], s:s + cut]
    return x_out


def augment(x):
    """
    Single augmentation strategy applied to BOTH views (independently).

    Steps:
      1. CutMix (40% window swapped from random partner)
      2. Mild per-feature dropout across ALL packets
         proto 15%, pkt_len 30%, flags 20%, iat 0%, direction 5%
      3. Gaussian noise on IAT (×0.05)

    Both views call this same function with different random states
    → symmetric augmentation.
    """
    x_out = cutmix(x, alpha=0.4)
    B, T, _ = x_out.shape

    # Feature-specific dropout
    drop_probs = {0: 0.15, 1: 0.30, 2: 0.20, 4: 0.05}   # feature_idx → prob
    for fi, p in drop_probs.items():
        mask = torch.rand(B, T, device=x_out.device) < p
        x_out[:, :, fi][mask] = 0.0

    # IAT noise
    x_out[:, :, 3] += torch.randn(B, T, device=x_out.device) * 0.05

    return x_out


# ── k-NN evaluation ───────────────────────────────────────────────
@torch.no_grad()
def extract_reps(encode_fn, loader, exit_pt, device=DEVICE):
    """Mean-pool hidden states up to exit_pt packet."""
    reps = []
    for x, _ in loader:
        h = encode_fn(x.to(device))          # [B, T, d]
        if h.size(1) == 33:                   # BERT: strip CLS token
            h = h[:, 1:, :]
        reps.append(h[:, :exit_pt, :].mean(1).cpu())
    return torch.cat(reps)


def knn_auc(test_reps, test_labels, db_reps, device=DEVICE):
    """k-NN anomaly score: 1 - avg cosine similarity to nearest benign neighbours."""
    db     = F.normalize(db_reps.to(device), dim=1)
    scores = []
    for s in range(0, len(test_reps), 512):
        q = F.normalize(test_reps[s:s + 512].to(device), dim=1)
        topk = torch.mm(q, db.T).topk(K, dim=1).values
        scores.append(topk.mean(1).cpu())
    anomaly = 1.0 - torch.cat(scores).numpy()
    auc = roc_auc_score(test_labels, anomaly)
    return max(auc, 1.0 - auc)


def eval_at_exits(name, encode_fn, exits=(8, 32)):
    """Evaluate a model at multiple exit points, print table row."""
    idx = np.random.choice(len(pretrain_ds), min(N_SAMPLE, len(pretrain_ds)), replace=False)
    sub = torch.utils.data.Subset(pretrain_ds, idx)
    sub_loader = DataLoader(sub, batch_size=512, shuffle=False)

    print(f'\n  {name}')
    print(f'  {"Dataset":<14}  {"@pkt":>4}  {"AUC":>8}')
    print('  ' + '─' * 30)
    for p in exits:
        db   = extract_reps(encode_fn, sub_loader, p, DEVICE)
        test = extract_reps(encode_fn, test_loader, p, DEVICE)
        cic  = extract_reps(encode_fn, cic_loader,  p, DEVICE)
        print(f'  {"UNSW":<14}  {p:>4}  {knn_auc(test, test_ds.labels.numpy(), db):>8.4f}')
        print(f'  {"CIC-IDS":<14}  {p:>4}  {knn_auc(cic, cic_ds.labels.numpy(), db):>8.4f}')


# ══════════════════════════════════════════════════════════════════
# TRAINING
# ══════════════════════════════════════════════════════════════════
model = UniMambaSSLv4().to(DEVICE)
n_params = sum(p.numel() for p in model.parameters())
print(f'\nUniMambaSSLv4 params: {n_params:,}')

if os.path.isfile(SSL_V4_PATH):
    print(f'Found existing v4 weights → loading {SSL_V4_PATH}')
    model.load_state_dict(torch.load(SSL_V4_PATH, map_location='cpu', weights_only=False))
    model.eval()
else:
    print(f'\nTraining SSL v4 ({SSL_EPOCHS} epochs)...')
    print(f'  Aug:  cutmix(40%) + feature dropout — SAME for both views')
    print(f'  Loss: NT-Xent (τ={TEMP}) + {LAMBDA_NEXT}×next-packet prediction')
    print(f'  No MSE reconstruction\n')

    ssl_loader = DataLoader(pretrain_ds, batch_size=SSL_BS, shuffle=True, drop_last=True)
    opt   = torch.optim.AdamW(model.parameters(), lr=SSL_LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=SSL_EPOCHS, eta_min=SSL_LR / 100)
    nt_xent   = NTXentLoss(temperature=TEMP)
    best_loss = float('inf')
    best_sd   = None

    for ep in range(SSL_EPOCHS):
        model.train()
        total_loss = total_cl = total_np = 0.0
        n   = 0
        t0  = time.time()

        for x_raw, _ in ssl_loader:
            x_raw = x_raw.to(DEVICE)

            # ── Two views via SAME augmentation (different random draws) ──
            x1 = augment(x_raw)   # view 1
            x2 = augment(x_raw)   # view 2 — same function, independent randomness

            # ── Forward pass ──────────────────────────────────────────────
            proj1, h1 = model(x1)
            proj2, h2 = model(x2)

            # ── Contrastive loss (NT-Xent) ────────────────────────────────
            loss_cl = nt_xent(proj1, proj2)

            # ── Next-packet prediction loss (on both views) ───────────────
            # Predict ORIGINAL x_raw[t+1] from h_view[t]
            # h[:, :-1, :] has hidden states for packets 0..T-2
            # x_raw[:, 1:, :] has raw features for packets 1..T-1
            np_pred1 = model.next_head(h1[:, :-1, :])   # [B, T-1, 5]
            np_pred2 = model.next_head(h2[:, :-1, :])   # [B, T-1, 5]
            target   = x_raw[:, 1:, :]                  # [B, T-1, 5] — original, not aug

            loss_np = 0.5 * F.mse_loss(np_pred1, target) \
                    + 0.5 * F.mse_loss(np_pred2, target)

            # ── Total loss ────────────────────────────────────────────────
            loss = loss_cl + LAMBDA_NEXT * loss_np

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            total_loss += loss.item()
            total_cl   += loss_cl.item()
            total_np   += loss_np.item()
            n += 1

        sched.step()
        avg_total = total_loss / n
        avg_cl    = total_cl / n
        avg_np    = total_np / n
        lr_now    = sched.get_last_lr()[0]

        print(f'  Ep {ep+1:>2}/{SSL_EPOCHS}: '
              f'total={avg_total:.4f}  '
              f'NT-Xent={avg_cl:.4f}  '
              f'next={avg_np:.4f}  '
              f'lr={lr_now:.2e}  '
              f'[{time.time()-t0:.1f}s]')

        if avg_total < best_loss:
            best_loss = avg_total
            best_sd   = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_sd)
    torch.save(model.state_dict(), SSL_V4_PATH)
    print(f'\nSaved → {SSL_V4_PATH}')
    model.eval()


# ══════════════════════════════════════════════════════════════════
# EVALUATION — compare v2, v3, v4
# ══════════════════════════════════════════════════════════════════
print('\n' + '=' * 70)
print('EVALUATION — UniMamba v2, v3, v4 at @8 and @32 (CIC-IDS cross-dataset AUC)')
print('=' * 70)

model.eval()
eval_at_exits('UniMamba SSL v4  (1-aug + NT-Xent + next-pkt predict)',
              lambda x: model.encode(x))

# ── Also reload v2 and v3 for comparison ─────────────────────────
print('\nLoading v2 and v3 for comparison...')

class UniMambaSSL(nn.Module):
    """Reference: v2 / v3 architecture (with recon_head)."""
    def __init__(self, d=256, de=32, nl=4):
        super().__init__()
        self.tokenizer = PacketEmbedder(d, de)
        self.layers    = nn.ModuleList([Mamba(d, d_state=16, d_conv=4, expand=2) for _ in range(nl)])
        self.norm      = nn.LayerNorm(d)
        self.proj_head = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, 128))
        self.recon_head = nn.Linear(d, 5)
    def encode(self, x):
        h = self.tokenizer(x)
        for layer in self.layers:
            h = self.norm(layer(h) + h)
        return h

v2 = UniMambaSSL().to(DEVICE)
v2.load_state_dict(torch.load(WD/'self_distill'/'unimamba_ssl_v2.pth',
                               map_location='cpu', weights_only=False), strict=True)
v2.eval()

v3 = UniMambaSSL().to(DEVICE)
v3.load_state_dict(torch.load(WD/'self_distill'/'unimamba_ssl_v3_late_mask.pth',
                               map_location='cpu', weights_only=False), strict=True)
v3.eval()

eval_at_exits('UniMamba SSL v2  (2-aug + NT-Xent + MSE recon)',      lambda x: v2.encode(x))
eval_at_exits('UniMamba SSL v3  (late-mask + 2-aug + NT-Xent + MSE)', lambda x: v3.encode(x))


# ── Summary table ─────────────────────────────────────────────────
print('\n' + '=' * 70)
print('FINAL TABLE — CIC-IDS AUC')
print('=' * 70)
print(f'  {"Model":<48}  {"@8":>8}  {"@32":>8}')
print('  ' + '─' * 68)

idx_ref = np.random.choice(len(pretrain_ds), N_SAMPLE, replace=False)
sub_ref = torch.utils.data.Subset(pretrain_ds, idx_ref)
sub_loader = DataLoader(sub_ref, batch_size=512, shuffle=False)

rows = [
    ('v2  (2-aug, NT-Xent+MSE)',         lambda x: v2.encode(x)),
    ('v3  (late-mask, 2-aug, NT-Xent+MSE)', lambda x: v3.encode(x)),
    ('v4  (1-aug, NT-Xent+next-pkt)  ←NEW', lambda x: model.encode(x)),
]

for label, enc_fn in rows:
    aucs = {}
    for p in [8, 32]:
        db  = extract_reps(enc_fn, sub_loader, p, DEVICE)
        cic = extract_reps(enc_fn, cic_loader, p, DEVICE)
        aucs[p] = knn_auc(cic, cic_ds.labels.numpy(), db)
    print(f'  {label:<48}  {aucs[8]:>8.4f}  {aucs[32]:>8.4f}')

print('\nDone.')
