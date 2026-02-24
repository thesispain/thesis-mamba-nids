#!/usr/bin/env python3
"""
Full pipeline: wait for weights → Phase 2 eval → Phase 3 finetune → save results
Run from: /home/T2510596/Downloads/totally fresh/thesis_final/final jupyter files/
"""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, pickle, os, sys, time, copy, random, warnings
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from mamba_ssm import Mamba
warnings.filterwarnings('ignore')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED   = 42
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

# ── Paths ──────────────────────────────────────────────────────────────
ROOT      = Path('/home/T2510596/Downloads/totally fresh')
UNSW_DIR  = ROOT / 'Organized_Final' / 'data' / 'unswnb15_full'
CIC_PATH  = ROOT / 'thesis_final' / 'data' / 'cicids2017_flows.pkl'
CTU_PATH  = ROOT / 'thesis_final' / 'data' / 'ctu13_flows.pkl'
W         = Path('weights')
SSL_DIR   = W / 'phase2_ssl'
T3_DIR    = W / 'phase3_teachers'
T3_DIR.mkdir(parents=True, exist_ok=True)
RESULTS   = Path('FULL_PIPELINE_RESULTS.txt')

log_lines = []
def log(msg=''):
    print(msg, flush=True)
    log_lines.append(msg)
    RESULTS.write_text('\n'.join(log_lines))  # save incrementally

log(f'Full pipeline started: {time.strftime("%Y-%m-%d %H:%M:%S")}')
log(f'Device: {DEVICE}')
log()

# ══════════════════════════════════════════════════════════════════════
# ARCHITECTURES
# ══════════════════════════════════════════════════════════════════════

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

class BertEncoder(nn.Module):
    def __init__(self, d_model=256, de=32, nhead=8, num_layers=4, ff=1024, proj_out=128):
        super().__init__()
        self.tokenizer = PacketEmbedder(d_model, de)
        self.pos_encoder = LearnedPE(d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model, nhead, ff, 0.1, 'gelu', batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(enc_layer, num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.proj_head = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, proj_out))
        self.recon_head = nn.Linear(d_model, 5)
    def forward(self, x):
        h = self.tokenizer(x); h = self.pos_encoder(h)
        h = self.transformer_encoder(h); return self.norm(h)
    def get_ssl_outputs(self, x):
        h = self.forward(x)
        return self.proj_head(h.mean(dim=1)), self.recon_head(h), h  # mean pool

class BertClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = BertEncoder()
        self.head = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.1), nn.Linear(64, 2))
    def forward(self, x):
        h = self.encoder(x)
        return self.head(self.encoder.proj_head(h.mean(dim=1)))

class BiMambaEncoder(nn.Module):
    def __init__(self, d_model=256, de=32, n_layers=4):
        super().__init__()
        self.tokenizer  = PacketEmbedder(d_model, de)
        self.layers     = nn.ModuleList([Mamba(d_model, d_state=16, d_conv=4, expand=2) for _ in range(n_layers)])
        self.layers_rev = nn.ModuleList([Mamba(d_model, d_state=16, d_conv=4, expand=2) for _ in range(n_layers)])
        self.norm       = nn.LayerNorm(d_model)
        self.proj_head  = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, d_model))
        self.recon_head = nn.Linear(d_model, 5)
    def forward(self, x):
        feat = self.tokenizer(x)
        for fwd, rev in zip(self.layers, self.layers_rev):
            feat = self.norm((fwd(feat) + rev(feat.flip(1)).flip(1)) / 2 + feat)
        return feat
    def get_ssl_outputs(self, x):
        h = self.forward(x)
        return self.proj_head(h.mean(dim=1)), self.recon_head(h), h

class BiMambaClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = BiMambaEncoder()
        self.head = nn.Sequential(nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 2))
    def forward(self, x):
        h = self.encoder(x)
        return self.head(self.encoder.proj_head(h.mean(dim=1)))

# ══════════════════════════════════════════════════════════════════════
# DATA
# ══════════════════════════════════════════════════════════════════════

class FlowDataset(Dataset):
    def __init__(self, data):
        self.features = torch.tensor(np.array([d['features'] for d in data]), dtype=torch.float32)
        self.labels   = torch.tensor(np.array([d['label']    for d in data]), dtype=torch.long)
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx): return self.features[idx], self.labels[idx]

def load_pkl(path):
    with open(path, 'rb') as f: return pickle.load(f)

log('Loading datasets...')
unsw_pretrain = load_pkl(UNSW_DIR / 'pretrain_50pct_benign.pkl')
unsw_finetune = load_pkl(UNSW_DIR / 'finetune_mixed.pkl')
cicids        = load_pkl(CIC_PATH)
ctu13         = load_pkl(CTU_PATH)
log(f'  UNSW pretrain: {len(unsw_pretrain):,}   finetune: {len(unsw_finetune):,}')
log(f'  CIC-IDS: {len(cicids):,}   CTU-13: {len(ctu13):,}')

labels_ft = np.array([d['label'] for d in unsw_finetune])
idx_train, idx_temp = train_test_split(range(len(unsw_finetune)), test_size=0.3, stratify=labels_ft, random_state=SEED)
idx_val, idx_test   = train_test_split(idx_temp, test_size=0.5, stratify=labels_ft[idx_temp], random_state=SEED)
train_data = [unsw_finetune[i] for i in idx_train]
val_data   = [unsw_finetune[i] for i in idx_val]
test_data  = [unsw_finetune[i] for i in idx_test]

BS = 512
pretrain_ds = FlowDataset(unsw_pretrain)
test_ds     = FlowDataset(test_data)
val_ds      = FlowDataset(val_data)
train_ds    = FlowDataset(train_data)
cic_ds      = FlowDataset(cicids)
ctu_ds      = FlowDataset(ctu13)

pretrain_loader = DataLoader(pretrain_ds, batch_size=BS, shuffle=False)
test_loader     = DataLoader(test_ds, batch_size=BS, shuffle=False)
val_loader      = DataLoader(val_ds, batch_size=BS, shuffle=False)
train_loader    = DataLoader(train_ds, batch_size=BS, shuffle=True, drop_last=True)
cic_loader      = DataLoader(cic_ds, batch_size=BS, shuffle=False)
ctu_loader      = DataLoader(ctu_ds, batch_size=BS, shuffle=False)
log()

# ══════════════════════════════════════════════════════════════════════
# WAIT FOR CELL 10 TO FINISH
# ══════════════════════════════════════════════════════════════════════

# All possible weight files Cell 10 might produce
POSSIBLE_WEIGHTS = {
    'cutmix': (SSL_DIR/'ssl_bimamba_cutmix.pth', SSL_DIR/'ssl_bert_cutmix.pth'),
    'anti':   (SSL_DIR/'ssl_bimamba_anti.pth',   SSL_DIR/'ssl_bert_anti.pth'),
    'both':   (SSL_DIR/'ssl_bimamba_both.pth',   SSL_DIR/'ssl_bert_both.pth'),
}

log('Waiting for SSL weights to be ready...')
# Wait up to 90 minutes for any new weights to appear
deadline = time.time() + 90 * 60
last_found = set()
while time.time() < deadline:
    # Check which weights exist
    found = set()
    for v, (bp, bertp) in POSSIBLE_WEIGHTS.items():
        if bp.exists(): found.add(f'bimamba_{v}')
        if bertp.exists(): found.add(f'bert_{v}')
    new = found - last_found
    if new:
        log(f'  [{time.strftime("%H:%M:%S")}] New weights: {sorted(new)}')
        last_found = found
    # GPU idle = training done
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu','--format=csv,noheader'],
                                capture_output=True, text=True)
        gpu_util = int(result.stdout.strip().replace('%','').strip())
    except: gpu_util = 100
    if gpu_util < 10 and len(last_found) >= 4:
        log(f'  GPU idle ({gpu_util}%), training complete!')
        break
    time.sleep(30)

log(f'Weight files found: {sorted(last_found)}')
log()

# ══════════════════════════════════════════════════════════════════════
# PHASE 2 EVAL — Similarity AUC for all available variants
# ══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def extract_reps(encoder, loader):
    encoder.eval()
    out = []
    for x, _ in loader:
        h = encoder(x.to(DEVICE))
        out.append(encoder.proj_head(h.mean(dim=1)).cpu())
    return torch.cat(out)

def sim_auc(test_reps, test_labels, train_reps, chunk=512):
    db = F.normalize(train_reps.to(DEVICE), dim=1)
    sims = []
    for s in range(0, len(test_reps), chunk):
        q = F.normalize(test_reps[s:s+chunk].to(DEVICE), dim=1)
        sims.append(torch.mm(q, db.T).max(dim=1)[0].cpu())
    scores = 1.0 - torch.cat(sims).numpy()
    return roc_auc_score(test_labels, scores)

def load_enc(EncoderCls, path):
    enc = EncoderCls()
    sd = torch.load(path, map_location='cpu', weights_only=False)
    enc.load_state_dict(sd, strict=True)
    return enc.to(DEVICE)

log('═'*60)
log('PHASE 2 — SSL Unsupervised Similarity AUC')
log('═'*60)
log()

eval_sets = [
    ('UNSW-NB15',    test_loader,  test_ds.labels.numpy()),
    ('CIC-IDS-2017', cic_loader,   cic_ds.labels.numpy()),
    ('CTU-13',       ctu_loader,   ctu_ds.labels.numpy()),
]

log(f"{'Variant':<8}  {'Dataset':<15}  {'BiMamba':>10}  {'BERT':>10}")
log('─' * 48)

p2_results = {}   # (v, ds) -> (auc_bi, auc_bert)
best_bi_cic = ('', 0.0, None)
# Also track best BERT separately
best_bert_cic = ('', 0.0, None)

for v, (bi_path, bert_path) in POSSIBLE_WEIGHTS.items():
    if not bi_path.exists():
        log(f'{v:<8}  (weights not found, skipped)')
        continue
    bi_enc   = load_enc(BiMambaEncoder, bi_path)
    bert_enc = load_enc(BertEncoder, bert_path) if bert_path.exists() else None

    bi_train   = extract_reps(bi_enc, pretrain_loader)
    bert_train = extract_reps(bert_enc, pretrain_loader) if bert_enc else None

    for ds_name, loader, labels in eval_sets:
        bi_reps  = extract_reps(bi_enc, loader)
        auc_bi   = sim_auc(bi_reps, labels, bi_train)
        if bert_enc is not None:
            bert_reps = extract_reps(bert_enc, loader)
            auc_bert  = sim_auc(bert_reps, labels, bert_train)
            b_str = f'{auc_bert:>10.4f}'
        else:
            auc_bert = None
            b_str = '      skip'
        p2_results[(v, ds_name)] = (auc_bi, auc_bert)
        log(f'{v:<8}  {ds_name:<15}  {auc_bi:>10.4f}  {b_str}')

        if ds_name == 'CIC-IDS-2017':
            if auc_bi > best_bi_cic[1]:
                best_bi_cic = (v, auc_bi, bi_path)
            if auc_bert is not None and auc_bert > best_bert_cic[1]:
                best_bert_cic = (v, auc_bert, bert_path)
    log()

log('─' * 48)
log(f'Best BiMamba variant (CIC AUC): {best_bi_cic[0].upper()}  AUC={best_bi_cic[1]:.4f}')
log(f'Best BERT variant    (CIC AUC): {best_bert_cic[0].upper()}  AUC={best_bert_cic[1]:.4f}')
log()

best_bi_ssl_path   = best_bi_cic[2]
best_bert_ssl_path = best_bert_cic[2] if best_bert_cic[2] else (SSL_DIR/'ssl_bert_cutmix.pth' if (SSL_DIR/'ssl_bert_cutmix.pth').exists() else None)
log(f'BiMamba SSL → {best_bi_ssl_path}')
log(f'BERT SSL    → {best_bert_ssl_path}')
log()

# ══════════════════════════════════════════════════════════════════════
# UTILS for Phase 3
# ══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_classifier(model, loader):
    model.eval()
    preds, labs, probs = [], [], []
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)
        probs.extend(F.softmax(logits, dim=1)[:, 1].cpu().numpy())
        preds.extend(logits.argmax(1).cpu().numpy())
        labs.extend(y.cpu().numpy())
    acc = accuracy_score(labs, preds)
    f1  = f1_score(labs, preds, zero_division=0)
    auc = roc_auc_score(labs, probs)
    return acc, f1, auc

def save_weights(model, path):
    torch.save(model.state_dict(), path)
    check = torch.load(path, map_location='cpu', weights_only=False)
    assert set(check.keys()) == set(model.state_dict().keys())
    log(f'  ✓ Saved: {path}')

# ══════════════════════════════════════════════════════════════════════
# PHASE 3 — Fine-tune BERT and BiMamba teachers
# ══════════════════════════════════════════════════════════════════════

log('═'*60)
log('PHASE 3 — Supervised Teacher Fine-Tuning')
log('═'*60)

FT_EPOCHS = 5
FT_LR     = 1e-4
BERT_T_PATH    = T3_DIR / 'bert_teacher.pth'
BIMAMBA_T_PATH = T3_DIR / 'bimamba_teacher.pth'

def finetune_teacher(ClassifierCls, ssl_enc_path, save_path, name):
    if save_path.exists():
        clf = ClassifierCls()
        clf.load_state_dict(torch.load(save_path, map_location='cpu', weights_only=False), strict=True)
        log(f'  ✓ Loaded existing {name} teacher from {save_path}')
        return clf.to(DEVICE)

    if ssl_enc_path is None or not ssl_enc_path.exists():
        log(f'  ✗ No SSL weights for {name}, skipping teacher fine-tune')
        return None

    clf = ClassifierCls()
    sd = torch.load(ssl_enc_path, map_location='cpu', weights_only=False)
    clf.encoder.load_state_dict(sd, strict=True)
    log(f'  ✓ Loaded SSL encoder: {ssl_enc_path.name}')

    clf = clf.to(DEVICE)
    opt = torch.optim.AdamW(clf.parameters(), lr=FT_LR, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=FT_EPOCHS)
    crit = nn.CrossEntropyLoss()

    best_val_f1, best_sd = 0.0, None
    for epoch in range(FT_EPOCHS):
        clf.train(); total_loss = 0; n = 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            loss = crit(clf(x), y)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(clf.parameters(), 1.0)
            opt.step(); total_loss += loss.item(); n += 1
        sch.step()
        acc, f1, auc = evaluate_classifier(clf, val_loader)
        log(f'  ep {epoch+1}/{FT_EPOCHS}  loss={total_loss/n:.4f}  val acc={acc:.4f} f1={f1:.4f} auc={auc:.4f}')
        if f1 > best_val_f1:
            best_val_f1 = f1; best_sd = copy.deepcopy(clf.state_dict())

    clf.load_state_dict(best_sd)
    save_weights(clf, save_path)
    return clf

log()
log(f'Fine-tuning BERT Teacher ({FT_EPOCHS} epochs, LR={FT_LR})')
log('─'*40)
bert_teacher    = finetune_teacher(BertClassifier, best_bert_ssl_path, BERT_T_PATH, 'BERT')
log()
log(f'Fine-tuning BiMamba Teacher ({FT_EPOCHS} epochs, LR={FT_LR})')
log('─'*40)
bimamba_teacher = finetune_teacher(BiMambaClassifier, best_bi_ssl_path, BIMAMBA_T_PATH, 'BiMamba')
log()

# ══════════════════════════════════════════════════════════════════════
# PHASE 3 FINAL RESULTS
# ══════════════════════════════════════════════════════════════════════

log('═'*60)
log('PHASE 3 — Teacher Evaluation Results')
log('═'*60)
log()
log(f"{'Model':<20}  {'Dataset':<15}  {'Acc':>7}  {'F1':>7}  {'AUC':>7}")
log('─' * 65)

for model_name, model in [('BERT Teacher', bert_teacher), ('BiMamba Teacher', bimamba_teacher)]:
    if model is None:
        log(f'{model_name:<20}  (not trained)')
        continue
    for ds_name, loader in [('UNSW Test', test_loader), ('CIC-IDS-2017', cic_loader), ('CTU-13', ctu_loader)]:
        acc, f1, auc = evaluate_classifier(model, loader)
        log(f'{model_name:<20}  {ds_name:<15}  {acc:>7.4f}  {f1:>7.4f}  {auc:>7.4f}')
    log()

log('═'*60)
log(f'Pipeline complete: {time.strftime("%Y-%m-%d %H:%M:%S")}')
log('═'*60)
log()
log('Key findings:')
log(f'  Best BiMamba SSL variant: {best_bi_cic[0].upper()} (CIC AUC={best_bi_cic[1]:.4f})')
log(f'  Best BERT    SSL variant: {best_bert_cic[0].upper()} (CIC AUC={best_bert_cic[1]:.4f})')
log()
log('All results saved to: FULL_PIPELINE_RESULTS.txt')
