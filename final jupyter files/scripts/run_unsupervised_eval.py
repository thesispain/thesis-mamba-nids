#!/usr/bin/env python3
"""
UNSUPERVISED-ONLY EVAL — Phase 2 Similarity AUC
Waits for Cell 10 to finish, then evaluates all SSL weight files.
NO fine-tuning. Pure unsupervised results.

BERT NOTE: ssl_bert_cutmix was trained with broken h[:,0,:] pooling (stuck at 5.5413=random).
           ssl_bert_anti was trained with fixed mean pooling but LR=1e-3 caused divergence.
           BiMamba weights are clean (mean pool throughout).
"""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, pickle, os, sys, time, random, warnings, subprocess
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
from mamba_ssm import Mamba
warnings.filterwarnings('ignore')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED   = 42
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

ROOT     = Path('/home/T2510596/Downloads/totally fresh')
UNSW_DIR = ROOT / 'Organized_Final' / 'data' / 'unswnb15_full'
CIC_PATH = ROOT / 'thesis_final' / 'data' / 'cicids2017_flows.pkl'
CTU_PATH = ROOT / 'thesis_final' / 'data' / 'ctu13_flows.pkl'
SSL_DIR  = Path('weights/phase2_ssl')
RESULTS  = Path('UNSUPERVISED_EVAL_RESULTS.txt')

lines = []
def log(msg=''):
    print(msg, flush=True)
    lines.append(msg)
    RESULTS.write_text('\n'.join(lines))

# ══════════════════════════════════════════════════════════════════════════
# ARCHITECTURES — exact match to Cell 4 (mean pooling everywhere)
# ══════════════════════════════════════════════════════════════════════════

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
        c = torch.cat([
            self.emb_proto(x[:,:,0].long().clamp(0,255)),
            self.proj_len(x[:,:,1:2]),
            self.emb_flags(x[:,:,2].long().clamp(0,63)),
            self.proj_iat(x[:,:,3:4]),
            self.emb_dir(x[:,:,4].long().clamp(0,1))
        ], dim=-1)
        return self.norm(self.fusion(c))

class LearnedPE(nn.Module):
    def __init__(self, d=256):
        super().__init__()
        self.pe_emb = nn.Embedding(5000, d)
    def forward(self, x):
        return x + self.pe_emb(torch.arange(x.size(1), device=x.device))

class BertEncoder(nn.Module):
    """FIXED: uses mean pooling (not h[:,0,:]). Matches Cell 4."""
    def __init__(self, d=256, de=32, nh=8, nl=4, ff=1024, po=128):
        super().__init__()
        self.tokenizer = PacketEmbedder(d, de)
        self.pos_encoder = LearnedPE(d)
        enc_layer = nn.TransformerEncoderLayer(d, nh, ff, 0.1, 'gelu', batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(enc_layer, nl)
        self.norm = nn.LayerNorm(d)
        self.proj_head = nn.Sequential(nn.Linear(d,d), nn.ReLU(), nn.Linear(d, po))
        self.recon_head = nn.Linear(d, 5)
    def forward(self, x):
        h = self.tokenizer(x); h = self.pos_encoder(h)
        h = self.transformer_encoder(h); return self.norm(h)
    def get_ssl_outputs(self, x):
        h = self.forward(x)
        return self.proj_head(h.mean(dim=1)), self.recon_head(h), h

class BiMambaEncoder(nn.Module):
    def __init__(self, d=256, de=32, nl=4):
        super().__init__()
        self.tokenizer  = PacketEmbedder(d, de)
        self.layers     = nn.ModuleList([Mamba(d, d_state=16, d_conv=4, expand=2) for _ in range(nl)])
        self.layers_rev = nn.ModuleList([Mamba(d, d_state=16, d_conv=4, expand=2) for _ in range(nl)])
        self.norm       = nn.LayerNorm(d)
        self.proj_head  = nn.Sequential(nn.Linear(d,d), nn.ReLU(), nn.Linear(d, d))
        self.recon_head = nn.Linear(d, 5)
    def forward(self, x):
        feat = self.tokenizer(x)
        for fwd, rev in zip(self.layers, self.layers_rev):
            feat = self.norm((fwd(feat) + rev(feat.flip(1)).flip(1)) / 2 + feat)
        return feat
    def get_ssl_outputs(self, x):
        h = self.forward(x)
        return self.proj_head(h.mean(dim=1)), self.recon_head(h), h

# ══════════════════════════════════════════════════════════════════════════
# DATA
# ══════════════════════════════════════════════════════════════════════════

class FlowDataset(Dataset):
    def __init__(self, data):
        self.features = torch.tensor(np.array([d['features'] for d in data]), dtype=torch.float32)
        self.labels   = torch.tensor(np.array([d['label']    for d in data]), dtype=torch.long)
    def __len__(self): return len(self.labels)
    def __getitem__(self, i): return self.features[i], self.labels[i]

log(f'Started: {time.strftime("%Y-%m-%d %H:%M:%S")}')
log(f'Device : {DEVICE}')
log()
log('Loading datasets ...')
with open(UNSW_DIR/'pretrain_50pct_benign.pkl','rb') as f: unsw_pre = pickle.load(f)
with open(UNSW_DIR/'finetune_mixed.pkl','rb') as f:        unsw_ft  = pickle.load(f)
with open(CIC_PATH,'rb') as f: cic = pickle.load(f)
with open(CTU_PATH,'rb') as f: ctu = pickle.load(f)

# quick 85/15 split for test labels (same seed as notebook)
from sklearn.model_selection import train_test_split
labels_ft = np.array([d['label'] for d in unsw_ft])
_, idx_temp = train_test_split(range(len(unsw_ft)), test_size=0.3, stratify=labels_ft, random_state=42)
_, idx_test  = train_test_split(idx_temp, test_size=0.5,
                                stratify=labels_ft[np.array(idx_temp)], random_state=42)
test_data = [unsw_ft[i] for i in idx_test]

pre_ds  = FlowDataset(unsw_pre)
test_ds = FlowDataset(test_data)
cic_ds  = FlowDataset(cic)
ctu_ds  = FlowDataset(ctu)

BS = 512
pre_loader  = DataLoader(pre_ds,  batch_size=BS, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=BS, shuffle=False)
cic_loader  = DataLoader(cic_ds,  batch_size=BS, shuffle=False)
ctu_loader  = DataLoader(ctu_ds,  batch_size=BS, shuffle=False)

log(f'  UNSW pretrain (benign): {len(unsw_pre):,}')
log(f'  UNSW test:              {len(test_data):,}')
log(f'  CIC-IDS-2017:           {len(cic):,}')
log(f'  CTU-13:                 {len(ctu):,}')
log()

# ══════════════════════════════════════════════════════════════════════════
# WAIT FOR CELL 10 TO FINISH
# Polls every 30s; breaks when GPU util <15% for 2 consecutive checks
# ══════════════════════════════════════════════════════════════════════════

def gpu_util():
    try:
        r = subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu','--format=csv,noheader'],
                           capture_output=True, text=True, timeout=5)
        return int(r.stdout.strip().replace('%','').strip())
    except: return 100

WAIT_FILES = [
    SSL_DIR / 'ssl_bimamba_cutmix.pth',
    SSL_DIR / 'ssl_bimamba_anti.pth',
]
EXPECTED_BOTH = SSL_DIR / 'ssl_bimamba_both.pth'

log('Waiting for Cell 10 to finish (polling GPU every 30s)...')
low_count = 0
while True:
    u = gpu_util()
    log(f'  [{time.strftime("%H:%M:%S")}] GPU={u}%   weights: { [f.name for f in SSL_DIR.glob("*.pth")] }')
    if u < 15:
        low_count += 1
        if low_count >= 2:
            log('  GPU idle — training complete!')
            break
    else:
        low_count = 0
    time.sleep(30)

log()

# Final listing of all weight files
log('='*60)
log('ALL WEIGHT FILES ON DISK')
log('='*60)
for d in sorted(Path('weights').rglob('*.pth')):
    sz = d.stat().st_size / 1e6
    mt = time.strftime('%Y-%m-%d %H:%M', time.localtime(d.stat().st_mtime))
    log(f'  {mt}  {sz:5.1f}MB  {d}')
log()

# ══════════════════════════════════════════════════════════════════════════
# EVAL HELPERS
# ══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def extract_reps(enc, loader):
    enc.eval()
    out = []
    for x, _ in loader:
        h = enc(x.to(DEVICE))
        out.append(enc.proj_head(h.mean(dim=1)).cpu())
    return torch.cat(out)

def sim_auc(test_reps, labels, train_reps, chunk=512):
    db = F.normalize(train_reps.to(DEVICE), dim=1)
    sims = []
    for s in range(0, len(test_reps), chunk):
        q = F.normalize(test_reps[s:s+chunk].to(DEVICE), dim=1)
        sims.append(torch.mm(q, db.T).max(dim=1)[0].cpu())
    return roc_auc_score(labels, 1.0 - torch.cat(sims).numpy())

def load_enc(Cls, path):
    m = Cls()
    m.load_state_dict(torch.load(path, map_location='cpu', weights_only=False), strict=True)
    return m.to(DEVICE)

# ══════════════════════════════════════════════════════════════════════════
# PHASE 2 — UNSUPERVISED SIMILARITY AUC (all variants)
# ══════════════════════════════════════════════════════════════════════════

log('='*60)
log('PHASE 2 — UNSUPERVISED SIMILARITY AUC')
log('Method: max cosine similarity to benign train set')
log('Higher = more benign → AUC reported as anomaly score')
log('='*60)
log()

# Define all possible weight pairs
ALL_VARIANTS = {
    'bimamba_cutmix':  (BiMambaEncoder, SSL_DIR/'ssl_bimamba_cutmix.pth'),
    'bimamba_anti':    (BiMambaEncoder, SSL_DIR/'ssl_bimamba_anti.pth'),
    'bimamba_both':    (BiMambaEncoder, SSL_DIR/'ssl_bimamba_both.pth'),
    'bimamba_paper':   (BiMambaEncoder, SSL_DIR/'ssl_bimamba_paper.pth'),
    'bert_anti':       (BertEncoder,    SSL_DIR/'ssl_bert_anti.pth'),
    'bert_paper':      (BertEncoder,    SSL_DIR/'ssl_bert_paper.pth'),
    'bert_cutmix':     (BertEncoder,    SSL_DIR/'ssl_bert_cutmix.pth'),
}

BERT_NOTES = {
    'bert_cutmix': 'BROKEN — trained with h[:,0,:] (stuck loss=5.54)',
    'bert_anti':   'PARTIAL — LR too high, diverged ep3 (4.69→4.98→5.54)',
    'bert_paper':  'old 1-epoch paper params',
}

eval_sets = [
    ('UNSW-NB15',    test_loader,  test_ds.labels.numpy()),
    ('CIC-IDS-2017', cic_loader,   cic_ds.labels.numpy()),
    ('CTU-13',       ctu_loader,   ctu_ds.labels.numpy()),
]

# Header
log(f"{'Variant':<22}  {'UNSW-NB15':>10}  {'CIC-IDS':>10}  {'CTU-13':>10}  Note")
log('─' * 80)

best_bi_cic = 0.0
all_results = {}

for vname, (Cls, wpath) in ALL_VARIANTS.items():
    if not wpath.exists():
        log(f'{vname:<22}  {"—":>10}  {"—":>10}  {"—":>10}  (weights not found)')
        continue

    note = BERT_NOTES.get(vname, '')
    try:
        enc = load_enc(Cls, wpath)
        train_reps = extract_reps(enc, pre_loader)
        aucs = []
        for ds_name, loader, labels in eval_sets:
            reps = extract_reps(enc, loader)
            aucs.append(sim_auc(reps, labels, train_reps))
        all_results[vname] = aucs
        row = f'{vname:<22}  {aucs[0]:>10.4f}  {aucs[1]:>10.4f}  {aucs[2]:>10.4f}  {note}'
        log(row)
        if 'bimamba' in vname and aucs[1] > best_bi_cic:
            best_bi_cic = aucs[1]
    except Exception as e:
        log(f'{vname:<22}  ERROR: {e}')

log('─' * 80)
log()

# Summary
log('='*60)
log('SUMMARY')
log('='*60)

bi_variants = {k:v for k,v in all_results.items() if 'bimamba' in k}
if bi_variants:
    best_k = max(bi_variants, key=lambda k: bi_variants[k][1])
    log(f'Best BiMamba variant (CIC AUC): {best_k}')
    log(f'  UNSW={bi_variants[best_k][0]:.4f}  CIC={bi_variants[best_k][1]:.4f}  CTU={bi_variants[best_k][2]:.4f}')
    log()
    target = 0.84
    log(f'Paper target CIC AUC = {target:.2f}')
    log(f'Best achieved         = {best_k}: {bi_variants[best_k][1]:.4f}  ({"✓ REACHED" if bi_variants[best_k][1] >= target else f"gap = {target - bi_variants[best_k][1]:.4f}"})')

log()
log('='*60)
log('ALL .pth FILES (final listing)')
log('='*60)
for f in sorted(Path('weights').rglob('*.pth')):
    sz = f.stat().st_size / 1e6
    mt = time.strftime('%H:%M', time.localtime(f.stat().st_mtime))
    log(f'  [{mt}]  {sz:5.1f}MB  {f}')

log()
log(f'Completed: {time.strftime("%Y-%m-%d %H:%M:%S")}')
log(f'Results  : {RESULTS.resolve()}')
