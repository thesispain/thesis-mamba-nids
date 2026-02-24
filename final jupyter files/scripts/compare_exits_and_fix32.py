#!/usr/bin/env python3
"""
Compare BERT SSL, BiMamba SSL, UniMamba SSL at @8 vs @32 packet exits.
Then retrain UniMamba SSL v3 with late-packet masking to fix @32.

KEY NOTE:
  BERT and BiMamba are NOT causal — self-attention / bidirectional Mamba
  means h[7] already "saw" all 32 packets. So @8 for them is NOT a true
  early-exit — it's just the mean of the first 8 hidden states from a model
  that had full context. Only UniMamba (forward-only Mamba) gives a true
  early-exit where h[7] literally only saw packets 0..7.
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
    raise ImportError("activate mamba_env first")

warnings.filterwarnings('ignore')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

ROOT     = Path('/home/T2510596/Downloads/totally fresh')
UNSW_DIR = ROOT / 'Organized_Final' / 'data' / 'unswnb15_full'
CIC_PATH = ROOT / 'thesis_final' / 'data' / 'cicids2017_flows.pkl'
WD       = Path('weights')
(WD / 'self_distill').mkdir(parents=True, exist_ok=True)

print(f'Device: {DEVICE}')

# ── Data ──────────────────────────────────────────────────────────
def load_pkl(path, name, fix_iat=False):
    with open(path, 'rb') as f: data = pickle.load(f)
    if fix_iat:
        for d in data: d['features'][:, 3] = np.log1p(d['features'][:, 3])
    labels = np.array([d['label'] for d in data])
    print(f'  {name}: {len(data):,} flows (benign={int((labels==0).sum()):,}, attack={int((labels==1).sum()):,})')
    return data

print('\nLoading data...')
unsw_pretrain = load_pkl(UNSW_DIR / 'pretrain_50pct_benign.pkl', 'UNSW Pretrain')
unsw_finetune = load_pkl(UNSW_DIR / 'finetune_mixed.pkl', 'UNSW Finetune')
cicids        = load_pkl(CIC_PATH, 'CIC-IDS-2017', fix_iat=True)

class FlowDataset(Dataset):
    def __init__(self, data):
        self.features = torch.tensor(np.array([d['features'] for d in data]), dtype=torch.float32)
        self.labels   = torch.tensor(np.array([d['label'] for d in data]), dtype=torch.long)
    def __len__(self): return len(self.labels)
    def __getitem__(self, i): return self.features[i], self.labels[i]

labels_ft = np.array([d['label'] for d in unsw_finetune])
idx_tr, idx_tmp = train_test_split(range(len(unsw_finetune)), test_size=0.3, stratify=labels_ft, random_state=SEED)
idx_val, idx_tst = train_test_split(idx_tmp, test_size=0.5, stratify=labels_ft[idx_tmp], random_state=SEED)

BS = 512
pretrain_ds = FlowDataset(unsw_pretrain)
test_ds     = FlowDataset([unsw_finetune[i] for i in idx_tst])
train_ds    = FlowDataset([unsw_finetune[i] for i in idx_tr])
cic_ds      = FlowDataset(cicids)

pretrain_loader = DataLoader(pretrain_ds, batch_size=BS, shuffle=False)
test_loader     = DataLoader(test_ds,     batch_size=BS, shuffle=False)
train_loader    = DataLoader(train_ds,    batch_size=BS, shuffle=False)
cic_loader      = DataLoader(cic_ds,      batch_size=BS, shuffle=False)

# ── Architectures ──────────────────────────────────────────────────
class PacketEmbedder(nn.Module):
    def __init__(self, d=256, de=32):
        super().__init__()
        self.emb_proto = nn.Embedding(256, de); self.emb_flags = nn.Embedding(64, de)
        self.emb_dir   = nn.Embedding(2, de//4)
        self.proj_len  = nn.Linear(1, de);      self.proj_iat = nn.Linear(1, de)
        self.fusion    = nn.Linear(de*4 + de//4, d)
        self.norm      = nn.LayerNorm(d)
    def forward(self, x):
        return self.norm(self.fusion(torch.cat([
            self.emb_proto(x[:,:,0].long().clamp(0,255)),
            self.proj_len(x[:,:,1:2]),
            self.emb_flags(x[:,:,2].long().clamp(0,63)),
            self.proj_iat(x[:,:,3:4]),
            self.emb_dir(x[:,:,4].long().clamp(0,1))], dim=-1)))

class LearnedPE(nn.Module):
    def __init__(self, d=256): super().__init__(); self.pe = nn.Embedding(5000, d)
    def forward(self, x): return x + self.pe(torch.arange(x.size(1), device=x.device))

# BERT SSL encoder (as originally trained)
class BertSSLEnc(nn.Module):
    def __init__(self, d=256, de=32, nh=4, nl=4, ff=1024):
        super().__init__()
        self.tok = PacketEmbedder(d, de); self.pos = LearnedPE(d)
        self.cls_tok = nn.Parameter(torch.zeros(1, 1, d))
        enc = nn.TransformerEncoderLayer(d, nh, ff, 0.1, 'gelu', batch_first=True)
        self.enc = nn.TransformerEncoder(enc, nl); self.norm = nn.LayerNorm(d)
        self.proj_head = nn.Sequential(nn.Linear(d,d), nn.ReLU(), nn.Linear(d,128))
        self.recon_head = nn.Linear(d, 5)
    def encode(self, x):
        h = self.tok(x)
        cls = self.cls_tok.expand(h.size(0), -1, -1)
        h = torch.cat([cls, h], dim=1)   # [B, 33, d]
        h = self.pos(h); return self.norm(self.enc(h))

# BiMamba SSL encoder
class BiMambaSSLEnc(nn.Module):
    def __init__(self, d=256, de=32, nl=4):
        super().__init__()
        self.tok = PacketEmbedder(d, de)
        self.layers = nn.ModuleList([Mamba(d, d_state=16, d_conv=4, expand=2) for _ in range(nl)])
        self.layers_rev = nn.ModuleList([Mamba(d, d_state=16, d_conv=4, expand=2) for _ in range(nl)])
        self.norm = nn.LayerNorm(d)
        self.proj_head = nn.Sequential(nn.Linear(d,d), nn.ReLU(), nn.Linear(d,d))
        self.recon_head = nn.Linear(d, 5)
    def encode(self, x):
        h = self.tok(x)
        for f, r in zip(self.layers, self.layers_rev):
            h = self.norm(f(h) + r(h.flip(1)).flip(1) + h)
        return h  # [B, 32, d]

# UniMamba SSL encoder
class UniMambaSSL(nn.Module):
    def __init__(self, d=256, de=32, nl=4):
        super().__init__()
        self.tokenizer = PacketEmbedder(d, de)  # must match saved weight keys
        self.layers = nn.ModuleList([Mamba(d, d_state=16, d_conv=4, expand=2) for _ in range(nl)])
        self.norm = nn.LayerNorm(d)
        self.proj_head = nn.Sequential(nn.Linear(d,d), nn.ReLU(), nn.Linear(d,128))
        self.recon_head = nn.Linear(d, 5)
    def encode(self, x):
        h = self.tokenizer(x)
        for layer in self.layers: h = self.norm(layer(h) + h)
        return h
    def get_ssl_outputs(self, x):
        h = self.encode(x)
        return self.proj_head(h.mean(1)), self.recon_head(h), h

# ── k-NN eval ─────────────────────────────────────────────────────
K = 10
N_SAMPLE = 20000

@torch.no_grad()
def extract_reps(encode_fn, loader, exit_pt, device):
    reps = []
    for x, _ in loader:
        h = encode_fn(x.to(device))          # [B, T, d]  (or [B, T+1, d] for BERT)
        # For BERT: h[:,0,:] is CLS → skip it, packets start at index 1
        if h.size(1) == 33:                   # BERT has CLS prepended
            h = h[:, 1:, :]                   # strip CLS → [B, 32, d]
        reps.append(h[:, :exit_pt, :].mean(1).cpu())
    return torch.cat(reps)

def knn_auc(test_reps, test_labels, db_reps, device=DEVICE):
    db = F.normalize(db_reps.to(device), dim=1)
    scores = []
    for s in range(0, len(test_reps), 512):
        q = F.normalize(test_reps[s:s+512].to(device), dim=1)
        topk = torch.mm(q, db.T).topk(K, dim=1).values
        scores.append(topk.mean(1).cpu())
    anomaly = 1.0 - torch.cat(scores).numpy()
    auc = roc_auc_score(test_labels, anomaly)
    return max(auc, 1.0 - auc)

def eval_model(name, encode_fn, exit_pts=[8, 32]):
    print(f'\n  {name}')
    print(f'  {"Dataset":<14}  {"Exit":>4}  {"AUC":>8}')
    print('  ' + '─'*32)
    # Sample from pretrain (benign only) as reference DB
    idx = np.random.choice(len(pretrain_ds), min(N_SAMPLE, len(pretrain_ds)), replace=False)
    sub = torch.utils.data.Subset(pretrain_ds, idx)
    sub_loader = DataLoader(sub, batch_size=512, shuffle=False)

    for p in exit_pts:
        db_reps   = extract_reps(encode_fn, sub_loader, p, DEVICE)
        test_reps = extract_reps(encode_fn, test_loader, p, DEVICE)
        cic_reps  = extract_reps(encode_fn, cic_loader, p, DEVICE)
        unsw_auc  = knn_auc(test_reps, test_ds.labels.numpy(), db_reps)
        cic_auc   = knn_auc(cic_reps,  cic_ds.labels.numpy(),  db_reps)
        print(f'  {"UNSW":<14}  {p:>4}  {unsw_auc:>8.4f}')
        print(f'  {"CIC-IDS":<14}  {p:>4}  {cic_auc:>8.4f}')
    print()


# ══════════════════════════════════════════════════════════════════
# PART 1: Compare BERT SSL, BiMamba SSL, UniMamba SSL at @8 and @32
# ══════════════════════════════════════════════════════════════════
print('\n' + '='*70)
print('PART 1: k-NN comparison — BERT, BiMamba, UniMamba at @8 and @32')
print('='*70)
print('\n  NOTE: BERT and BiMamba are NOT causal — h[0:8] already saw all')
print('  32 packets via attention/bidirectional Mamba. Only UniMamba gives')
print('  a TRUE early-exit where h[7] saw only packets 0-7.')

# Load BERT SSL
bert_enc = BertSSLEnc().to(DEVICE)
sd = torch.load(WD/'phase2_ssl'/'ssl_bert_paper.pth', map_location='cpu', weights_only=False)
bert_enc.load_state_dict(sd, strict=False)
bert_enc.eval()
print('\n  Loaded: ssl_bert_paper.pth')

# Load BiMamba SSL
bimamba_enc = BiMambaSSLEnc().to(DEVICE)
sd2 = torch.load(WD/'phase2_ssl'/'ssl_bimamba_paper.pth', map_location='cpu', weights_only=False)
bimamba_enc.load_state_dict(sd2, strict=False)
bimamba_enc.eval()
print('  Loaded: ssl_bimamba_paper.pth')

# Load UniMamba SSL v2
unimamba = UniMambaSSL().to(DEVICE)
sd3 = torch.load(WD/'self_distill'/'unimamba_ssl_v2.pth', map_location='cpu', weights_only=False)
unimamba.load_state_dict(sd3, strict=True)
unimamba.eval()
print('  Loaded: unimamba_ssl_v2.pth (10-epoch)')

eval_model('BERT SSL (1ep, bidirectional — @8 NOT a true early exit)',
           lambda x: bert_enc.encode(x))
eval_model('BiMamba SSL (1ep, bidirectional — @8 NOT a true early exit)',
           lambda x: bimamba_enc.encode(x))
eval_model('UniMamba SSL v2 (10ep, causal — @8 IS a true early exit)',
           lambda x: unimamba.encode(x))


# ══════════════════════════════════════════════════════════════════
# PART 2: Fix @32 — train UniMamba SSL v3 with late-packet masking
#
# Problem: @32 CIC AUC = 0.62 because late packets (9-32) encode
#          UNSW-specific patterns that hurt cross-dataset generalization.
#
# Fix: During SSL, apply HEAVY masking on positions 8-31 (70% chance).
#      This forces the model to build a @32 global representation that
#      ignores late-packet specifics → @32 should generalize like @8.
#
# Loss is same NT-Xent + MSE but augmentation now specifically targets
# late packets instead of random positions.
# ══════════════════════════════════════════════════════════════════
print('='*70)
print('PART 2: UniMamba SSL v3 — Late-Packet Masking to fix @32')
print('='*70)

SSL_V3_PATH = WD / 'self_distill' / 'unimamba_ssl_v3_late_mask.pth'
SSL_EPOCHS = 10
SSL_BS     = 128
SSL_LR     = 5e-5

class NTXentLoss(nn.Module):
    def __init__(self, t=0.5): super().__init__(); self.t = t
    def forward(self, z1, z2):
        B = z1.size(0)
        z = F.normalize(torch.cat([z1, z2], 0), dim=1)
        sim = torch.mm(z, z.T) / self.t
        sim.masked_fill_(torch.eye(2*B, device=z.device).bool(), -1e9)
        labels = torch.cat([torch.arange(B, 2*B), torch.arange(B)]).to(z.device)
        return F.cross_entropy(sim, labels)

def augment_late_mask(x, early_end=8, late_mask_prob=0.70):
    """
    Mask late packets (positions early_end..31) with prob late_mask_prob.
    Also apply standard random feature dropout on early packets.
    This forces model to NOT rely on late packets → @32 rep becomes universal.
    """
    x_out = x.clone()
    B, T, _ = x.shape
    # Late packet masking (the key change vs v2)
    if late_mask_prob > 0 and T > early_end:
        late_mask = torch.rand(B, T - early_end, device=x.device) < late_mask_prob
        x_out[:, early_end:, :][late_mask] = 0.0
    # Early packet: mild feature dropout (same as before)
    for fi, p in {0: 0.15, 1: 0.30, 2: 0.20, 3: 0.00, 4: 0.05}.items():
        if p > 0:
            m = torch.rand(B, early_end, device=x.device) < p
            x_out[:, :early_end, fi][m] = 0.0
    x_out[:, :early_end, 3] += torch.randn(B, early_end, device=x.device) * 0.05
    return x_out

def cutmix(x, alpha=0.4):
    B, T, _ = x.shape
    cut = int(T * alpha)
    donors = torch.randint(0, B-1, (B,), device=x.device)
    donors[donors >= torch.arange(B, device=x.device)] += 1
    x_out = x.clone()
    for i in range(B):
        s = random.randint(0, max(0, T - cut))
        x_out[i, s:s+cut] = x[donors[i], s:s+cut]
    return x_out

model_v3 = UniMambaSSL().to(DEVICE)
print(f'  UniMambaSSL v3 params: {sum(p.numel() for p in model_v3.parameters()):,}')

if os.path.isfile(SSL_V3_PATH):
    print(f'  Found existing v3 weights, loading...')
    model_v3.load_state_dict(torch.load(SSL_V3_PATH, map_location='cpu', weights_only=False))
    model_v3.eval()
else:
    print(f'  Training SSL v3 from scratch ({SSL_EPOCHS} epochs)...')
    print(f'  Augmentation: 70% late-packet masking (positions 8-31) per view')
    ssl_loader = DataLoader(pretrain_ds, batch_size=SSL_BS, shuffle=True, drop_last=True)
    opt = torch.optim.AdamW(model_v3.parameters(), lr=SSL_LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=SSL_EPOCHS, eta_min=0)
    loss_fn = NTXentLoss()
    best_loss, best_sd = float('inf'), None

    for ep in range(SSL_EPOCHS):
        model_v3.train()
        total, n = 0.0, 0
        t0 = time.time()
        for x, _ in ssl_loader:
            x = x.to(DEVICE)
            # Both views get late-packet masking
            x1 = augment_late_mask(x,            late_mask_prob=0.70)
            x2 = augment_late_mask(cutmix(x),    late_mask_prob=0.70)

            p1, r1, h1 = model_v3.get_ssl_outputs(x1)
            p2, r2, h2 = model_v3.get_ssl_outputs(x2)

            loss = loss_fn(p1, p2) + 0.5 * F.mse_loss(r1, x1) + 0.5 * F.mse_loss(r2, x2)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model_v3.parameters(), 1.0)
            opt.step()
            total += loss.item(); n += 1

        sched.step()
        avg = total / n
        print(f'    Ep {ep+1}/{SSL_EPOCHS}: loss={avg:.4f} lr={sched.get_last_lr()[0]:.2e} [{time.time()-t0:.1f}s]')
        if avg < best_loss: best_loss = avg; best_sd = copy.deepcopy(model_v3.state_dict())

    model_v3.load_state_dict(best_sd)
    torch.save(model_v3.state_dict(), SSL_V3_PATH)
    print(f'  Saved: {SSL_V3_PATH}')
    model_v3.eval()

print()
eval_model('UniMamba SSL v3 (10ep, causal, 70% late-mask — @32 should now generalise)',
           lambda x: model_v3.encode(x))


# ══════════════════════════════════════════════════════════════════
# FINAL COMPARISON TABLE
# ══════════════════════════════════════════════════════════════════
print('='*70)
print('FINAL SUMMARY — CIC AUC at @8 and @32')
print('='*70)
print(f'  {"Model":<42}  {"@8":>8}  {"@32":>8}')
print('  ' + '─'*62)

results = {}
models_to_compare = [
    ('BERT SSL (bidirectional, NOT true early-exit)', lambda x: bert_enc.encode(x)),
    ('BiMamba SSL (bidirectional, NOT true early-exit)', lambda x: bimamba_enc.encode(x)),
    ('UniMamba SSL v2 (causal, 10ep)', lambda x: unimamba.encode(x)),
    ('UniMamba SSL v3 (causal, late-mask fix)', lambda x: model_v3.encode(x)),
]

idx = np.random.choice(len(pretrain_ds), min(N_SAMPLE, len(pretrain_ds)), replace=False)
sub = torch.utils.data.Subset(pretrain_ds, idx)
sub_loader = DataLoader(sub, batch_size=512, shuffle=False)

for name, enc_fn in models_to_compare:
    aucs = {}
    for p in [8, 32]:
        db   = extract_reps(enc_fn, sub_loader, p, DEVICE)
        cic  = extract_reps(enc_fn, cic_loader, p, DEVICE)
        aucs[p] = knn_auc(cic, cic_ds.labels.numpy(), db)
    print(f'  {name:<42}  {aucs[8]:>8.4f}  {aucs[32]:>8.4f}')

print()
print('Done.')
