
import os, sys, pickle, time, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score
from mamba_ssm import Mamba
import xgboost as xgb

print("Imports successful.")

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
W = "weights"
UNSW = "../Organized_Final/data/unswnb15_full/finetune_mixed.pkl"
CIC = "data/cicids2017_flows.pkl"

print(f"Device: {DEVICE}")
print(f"UNSW Path: {UNSW}")
print(f"CIC Path: {CIC}")

# --- Standard Packet Embedder (For BiMamba / TED / Standard BERT) ---
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

# --- Standard BERT Encoder (Matches Part 1 Training) ---
class BertEncoder(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_layers=2):
        super().__init__()
        self.emb_proto = nn.Embedding(256, 16)
        self.emb_flags = nn.Embedding(64, 16)
        self.emb_dir   = nn.Embedding(2, 4)
        self.proj_len  = nn.Linear(1, 16)
        self.proj_iat  = nn.Linear(1, 16)
        self.fusion    = nn.Linear(68, d_model)
        self.norm      = nn.LayerNorm(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=1024, dropout=0.1, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.proj_head = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 128))
    def forward(self, x):
        proto  = x[:,:,0].long().clamp(0, 255)
        length = x[:,:,1:2]
        flags  = x[:,:,2].long().clamp(0, 63)
        iat    = x[:,:,3:4]
        direc  = x[:,:,4].long().clamp(0, 1)
        cat = torch.cat([self.emb_proto(proto), self.proj_len(length),
                         self.emb_flags(flags), self.proj_iat(iat),
                         self.emb_dir(direc)], dim=-1)
        feat = self.norm(self.fusion(cat))
        feat = self.transformer_encoder(feat)
        return self.proj_head(feat[:, 0, :]), None

# --- Custom BERT Packet Embedder (Matches bert_cutmix_v5_partial.pth) ---
class PacketEmbedderBert(nn.Module):
    def __init__(self, d_model=256):
        super().__init__()
        self.proto_emb = nn.Embedding(256, 32)
        self.flags_emb = nn.Embedding(64, 32)
        self.dir_emb   = nn.Embedding(2, 8)
        self.loglen_proj = nn.Linear(1, 32)
        self.iat_proj  = nn.Linear(1, 32)
        self.fusion    = nn.Linear(136, d_model)
        self.norm      = nn.LayerNorm(d_model)
    def forward(self, x):
        proto  = x[:,:,0].long().clamp(0, 255)
        length = x[:,:,1:2]
        flags  = x[:,:,2].long().clamp(0, 63)
        iat    = x[:,:,3:4]
        direc  = x[:,:,4].long().clamp(0, 1)
        cat = torch.cat([self.proto_emb(proto), self.loglen_proj(length),
                         self.flags_emb(flags), self.iat_proj(iat),
                         self.dir_emb(direc)], dim=-1)
        return self.norm(self.fusion(cat))

# --- Custom BERT Wrapper (Matches bert_cutmix_v5_partial.pth) ---
class BertWrapper(nn.Module):
    def __init__(self, d_model=256, nhead=8):
        super().__init__()
        self.embedder = PacketEmbedderBert(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=1024, dropout=0.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
        self.proj = nn.Linear(d_model, 128)
        self.recon_head = nn.Linear(d_model, 5) # Matches checkpoint [5, 256]
    def forward(self, x):
        feat = self.embedder(x)
        feat = self.transformer(feat)
        return self.proj(feat[:, 0, :]), None

# --- BiMamba ---
class BiMambaEncoder(nn.Module):
    def __init__(self, d_model=256):
        super().__init__()
        self.tokenizer = PacketEmbedder(d_model) # Standard
        self.layers = nn.ModuleList([Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2) for _ in range(4)])
        self.layers_rev = nn.ModuleList([Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2) for _ in range(4)])
        self.norm = nn.LayerNorm(d_model)
        self.proj_head = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 256))
    def forward(self, x):
        feat = self.tokenizer(x)
        for fwd, rev in zip(self.layers, self.layers_rev):
             out_f = fwd(feat)
             out_r = rev(feat.flip(1)).flip(1)
             feat = self.norm((out_f + out_r) / 2 + feat)
        return self.proj_head(feat.mean(1)), None

class Classifier(nn.Module):
    def __init__(self, encoder, d_model=256):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(nn.Linear(d_model, 64), nn.ReLU(), nn.Dropout(0.1), nn.Linear(64, 2))
    def forward(self, x):
        z = self.encoder(x)
        if isinstance(z, tuple): z = z[0]
        return self.head(z)

class BlockwiseEarlyExitMamba(nn.Module):
    def __init__(self, d_model=256, exit_positions=None, conf_thresh=0.85):
        super().__init__()
        if exit_positions is None: exit_positions = [8, 16, 32]
        self.exit_positions = exit_positions
        self.n_exits = len(exit_positions)
        self.conf_thresh = conf_thresh
        self.tokenizer = PacketEmbedder(d_model) # Standard
        self.layers = nn.ModuleList([Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2) for _ in range(4)])
        self.norm = nn.LayerNorm(d_model)
        self.exit_classifiers = nn.ModuleDict({
            str(p): nn.Sequential(nn.Linear(d_model, 128), nn.ReLU(), nn.Dropout(0.1), nn.Linear(128, 2)) for p in exit_positions})
        self.confidence_heads = nn.ModuleDict({
            str(p): nn.Sequential(nn.Linear(d_model + 2, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid()) for p in exit_positions})
    def _backbone(self, x):
        feat = self.tokenizer(x)
        for layer in self.layers:
            feat = self.norm(layer(feat) + feat)
        return feat
    def forward_inference(self, x):
         feat = self._backbone(x)
         last_pos = self.exit_positions[-1]
         idx = min(last_pos, feat.size(1)) - 1
         h = feat[:, idx, :]\n         logits = self.exit_classifiers[str(last_pos)](h)
         return logits, None

print("Models defined.")

def train_xgboost(X, y, depth=32, name="Full"):
    print(f"Training XGBoost ({name})...")
    X_flat = X.reshape(X.shape[0], 32, -1)[:, :depth, :].reshape(X.shape[0], -1)
    clf = xgb.XGBClassifier(n_estimators=100, max_depth=6, n_jobs=-1, device='cuda')
    clf.fit(X_flat, y)
    return clf

def eval_per_class(true_y, pred_probs, attack_types, model_name):
    unique_types = np.unique(attack_types)
    targets = ['Benign', 'FTP-Patator', 'SSH-Patator', 'DoS Hulk', 'DoS GoldenEye', 'PortScan']
    benign_mask = (attack_types == 'Benign') 
    benign_probs = pred_probs[benign_mask]
    print(f"\\n--- {model_name} Breakdown ---\")
    print(f"{'Attack Class':<25} |Count   | AUC (vs Benign)\")
    print("-" * 50)
    for atk in targets:
        if atk not in unique_types: continue
        if atk == 'Benign': continue
        atk_mask = (attack_types == atk)
        atk_probs = pred_probs[atk_mask]
        combined_probs = np.concatenate([benign_probs, atk_probs])
        combined_y = np.concatenate([np.zeros(len(benign_probs)), np.ones(len(atk_probs))])
        try: auc = roc_auc_score(combined_y, combined_probs)
        except: auc = 0.5
        print(f"{str(atk):<25} |{len(atk_probs):<8} | {auc:.4f}")

def eval_model(model, dl, name):
    model.eval()
    probs = []
    with torch.no_grad():
        for x, _ in dl:
            if hasattr(model, 'forward_inference'): logits, _ = model.forward_inference(x)
            else: logits = model(x)
            if isinstance(logits, tuple): logits = logits[0]
            probs.extend(F.softmax(logits, 1)[:, 1].cpu().numpy())
    probs = np.array(probs)
    return probs

# Load Data
print("Loading UNSW (Train/Test In-Domain)...")
with open(UNSW, 'rb') as f: unsw_data = pickle.load(f)
np.random.seed(42)
np.random.shuffle(unsw_data)
unsw_sub = unsw_data[:200000]
unsw_train = unsw_sub[:100000]
unsw_test  = unsw_sub[100000:]
X_unsw_train = np.array([d['features'] for d in unsw_train], dtype=np.float32)
y_unsw_train = np.array([d['label'] for d in unsw_train], dtype=np.int64)
X_unsw_test  = np.array([d['features'] for d in unsw_test], dtype=np.float32)
y_unsw_test  = np.array([d['label'] for d in unsw_test], dtype=np.int64)

print("Loading CIC-IDS (Zero-Day Test)...")
with open(CIC, 'rb') as f: cic_data = pickle.load(f)
np.random.shuffle(cic_data)
cic_sub = cic_data[:200000]
X_cic_test  = np.array([d['features'] for d in cic_sub], dtype=np.float32)
y_cic_test  = np.array([d['label'] for d in cic_sub], dtype=np.int64)
cats_cic_test = np.array([d.get('attack_type', 'Unknown') for d in cic_sub])

unsw_dl = DataLoader(TensorDataset(torch.from_numpy(X_unsw_test).to(DEVICE), torch.from_numpy(y_unsw_test).to(DEVICE)), batch_size=512)
cic_dl  = DataLoader(TensorDataset(torch.from_numpy(X_cic_test).to(DEVICE), torch.from_numpy(y_cic_test).to(DEVICE)), batch_size=512)
print("Data Loaded.")

# XGBoost Evaluation
xgb_unsw = train_xgboost(X_unsw_train, y_unsw_train, 32, "In-Domain UNSW")
y_prob_unsw = xgb_unsw.predict_proba(X_unsw_test.reshape(X_unsw_test.shape[0], -1))[:, 1]
print(f"XGBoost In-Domain AUC: {roc_auc_score(y_unsw_test, y_prob_unsw):.4f}")

y_prob_zero = xgb_unsw.predict_proba(X_cic_test.reshape(X_cic_test.shape[0], -1))[:, 1]
print(f"XGBoost Zero-Day AUC: {roc_auc_score(y_cic_test, y_prob_zero):.4f}")
eval_per_class(y_cic_test, y_prob_zero, cats_cic_test, "XGBoost Zero-Day")

# RESULTS
results = {}

# BiMamba
enc = BiMambaEncoder()
bimamba = Classifier(enc).to(DEVICE)
sd = torch.load(f"{W}/teachers/teacher_bimamba_retrained.pth", map_location=DEVICE, weights_only=False)
bimamba.load_state_dict(sd, strict=False)
results['BiMamba'] = {'In': roc_auc_score(y_unsw_test, eval_model(bimamba, unsw_dl, "BiMamba")),
                      'Zero': roc_auc_score(y_cic_test, eval_model(bimamba, cic_dl, "BiMamba"))}
print(f"BiMamba: {results['BiMamba']}")

# TED
ted = BlockwiseEarlyExitMamba(conf_thresh=0.85).to(DEVICE)
sd = torch.load(f"{W}/students/student_ted.pth", map_location=DEVICE, weights_only=False)
ted.load_state_dict(sd, strict=False)
results['TED'] = {'In': roc_auc_score(y_unsw_test, eval_model(ted, unsw_dl, "TED")),
                  'Zero': roc_auc_score(y_cic_test, eval_model(ted, cic_dl, "TED"))}
print(f"TED: {results['TED']}")
eval_per_class(y_cic_test, eval_model(ted, cic_dl, "TED"), cats_cic_test, "TED")

# BERT
# Check for optimized weights first
if os.path.exists(f"{W}/ssl/bert_standard_ssl_optimized.pth"):
    print(f"Loading Optimized Weights: {W}/ssl/bert_standard_ssl_optimized.pth")
    bert_enc = BertEncoder(d_model=256, nhead=8) # Standard Class
    bert = Classifier(bert_enc, d_model=128).to(DEVICE)
    sd = torch.load(f"{W}/ssl/bert_standard_ssl_optimized.pth", map_location=DEVICE, weights_only=False)
    bert.encoder.load_state_dict(sd, strict=False)
else:
    print(f"Loading Checkpoint Weights: {W}/ssl/bert_cutmix_v5_partial.pth")
    bert_enc = BertWrapper(d_model=256, nhead=8) # Custom Class
    bert = Classifier(bert_enc, d_model=128).to(DEVICE)
    sd = torch.load(f"{W}/ssl/bert_cutmix_v5_partial.pth", map_location=DEVICE, weights_only=False)
    bert.encoder.load_state_dict(sd, strict=False)

results['BERT'] = {'In': roc_auc_score(y_unsw_test, eval_model(bert, unsw_dl, "BERT")),
                   'Zero': roc_auc_score(y_cic_test, eval_model(bert, cic_dl, "BERT"))}
print(f"BERT: {results['BERT']}")
