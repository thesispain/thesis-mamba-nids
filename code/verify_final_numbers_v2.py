
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

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
W = "weights"
UNSW = "/home/T2510596/Downloads/totally fresh/Organized_Final/data/unswnb15_full/finetune_mixed.pkl"
CIC = "/home/T2510596/Downloads/totally fresh/thesis_final/data/cicids2017_flows.pkl"

# --- Models ---
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
        self.proj_head = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 128)) # 128 head
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
        return self.proj_head(feat.mean(1)), None

class BiMambaEncoder(nn.Module):
    def __init__(self, d_model=256):
        super().__init__()
        self.tokenizer = PacketEmbedder(d_model)
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
        self.tokenizer = PacketEmbedder(d_model)
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
         # Standard Output (Last Layer) for Benchmark
         feat = self._backbone(x)
         last_pos = self.exit_positions[-1]
         idx = min(last_pos, feat.size(1)) - 1
         h = feat[:, idx, :]
         logits = self.exit_classifiers[str(last_pos)](h)
         return logits, None

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
    
    print(f"\n--- {model_name} Breakdown ---")
    print(f"{'Attack Class':<25} |Count   | AUC (vs Benign)")
    print("-" * 50)
    
    res = {}

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
        res[atk] = auc
    return res

def main():
    print("=== FINAL COMPLETE VERIFICATION V3 (IN-DOMAIN + ZERO-DAY) ===")
    
    # 1. Load Data
    print("Loading UNSW (Train/Test In-Domain)...")
    with open(UNSW, 'rb') as f: unsw_data = pickle.load(f)
    np.random.seed(42)
    np.random.shuffle(unsw_data) # Shuffle UNSW
    unsw_sub = unsw_data[:200000] 
    
    unsw_train = unsw_sub[:100000]
    unsw_test  = unsw_sub[100000:]
    
    X_unsw_train = np.array([d['features'] for d in unsw_train], dtype=np.float32)
    y_unsw_train = np.array([d['label'] for d in unsw_train], dtype=np.int64)
    
    X_unsw_test  = np.array([d['features'] for d in unsw_test], dtype=np.float32)
    y_unsw_test  = np.array([d['label'] for d in unsw_test], dtype=np.int64)

    print("Loading CIC-IDS (Zero-Day Test)...")
    with open(CIC, 'rb') as f: cic_data = pickle.load(f)
    np.random.shuffle(cic_data) # Shuffle CIC
    cic_sub = cic_data[:200000] 
    
    X_cic_test  = np.array([d['features'] for d in cic_sub], dtype=np.float32)
    y_cic_test  = np.array([d['label'] for d in cic_sub], dtype=np.int64)
    cats_cic_test = np.array([d.get('attack_type', 'Unknown') for d in cic_sub])
    
    # 2. XGBoost In-Domain (Train UNSW -> Test UNSW)
    # NOT CIC->CIC. User asked for Train on This -> Test on This.
    # But usually In-Domain means UNSW Train -> UNSW Test.
    xgb_unsw = train_xgboost(X_unsw_train, y_unsw_train, 32, "In-Domain UNSW")
    
    X_unsw_test_flat = X_unsw_test.reshape(X_unsw_test.shape[0], -1)
    y_prob_unsw = xgb_unsw.predict_proba(X_unsw_test_flat)[:, 1]
    auc_unsw = roc_auc_score(y_unsw_test, y_prob_unsw)
    print(f"XGBoost In-Domain (UNSW->UNSW) AUC: {auc_unsw:.4f}")
    
    # 3. XGBoost Zero-Day (Train UNSW -> Test CIC)
    X_cic_test_flat = X_cic_test.reshape(X_cic_test.shape[0], -1)
    y_prob_zero = xgb_unsw.predict_proba(X_cic_test_flat)[:, 1]
    auc_zero = roc_auc_score(y_cic_test, y_prob_zero)
    print(f"XGBoost Zero-Day (UNSW->CIC) AUC: {auc_zero:.4f}")
    eval_per_class(y_cic_test, y_prob_zero, cats_cic_test, "XGBoost Zero-Day")
    
    # 3b. XGBoost Zero-Day (8 Pkts)
    xgb_unsw_8 = train_xgboost(X_unsw_train, y_unsw_train, 8, "In-Domain UNSW 8")
    X_cic_test_8 = X_cic_test.reshape(X_cic_test.shape[0], 32, -1)[:, :8, :].reshape(X_cic_test.shape[0], -1)
    y_prob_zero_8 = xgb_unsw_8.predict_proba(X_cic_test_8)[:, 1]
    auc_zero_8 = roc_auc_score(y_cic_test, y_prob_zero_8)
    print(f"XGBoost Zero-Day (8 Pkt) AUC: {auc_zero_8:.4f}")
    eval_per_class(y_cic_test, y_prob_zero_8, cats_cic_test, "XGBoost Zero-Day (8 Pkt)")

    # DL Loaders
    unsw_dl = DataLoader(TensorDataset(torch.from_numpy(X_unsw_test).to(DEVICE), torch.from_numpy(y_unsw_test).to(DEVICE)), batch_size=512)
    cic_dl  = DataLoader(TensorDataset(torch.from_numpy(X_cic_test).to(DEVICE), torch.from_numpy(y_cic_test).to(DEVICE)), batch_size=512)

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

    # 4. BiMamba
    enc = BiMambaEncoder()
    bimamba = Classifier(enc).to(DEVICE)
    sd_t = torch.load(f"{W}/teachers/teacher_bimamba_retrained.pth", map_location=DEVICE, weights_only=False)
    bimamba.load_state_dict(sd_t, strict=False)
    
    probs_in = eval_model(bimamba, unsw_dl, "BiMamba")
    auc_in = roc_auc_score(y_unsw_test, probs_in)
    print(f"BiMamba In-Domain (UNSW->UNSW) AUC: {auc_in:.4f}")
    
    probs_zero = eval_model(bimamba, cic_dl, "BiMamba")
    auc_zero = roc_auc_score(y_cic_test, probs_zero)
    print(f"BiMamba Zero-Day AUC: {auc_zero:.4f}")
    eval_per_class(y_cic_test, probs_zero, cats_cic_test, "BiMamba")

    # 5. TED Zero-Day
    ted = BlockwiseEarlyExitMamba(conf_thresh=0.85).to(DEVICE)
    sd = torch.load(f"{W}/students/student_ted.pth", map_location=DEVICE, weights_only=False)
    ted.load_state_dict(sd, strict=False)
    
    probs_in = eval_model(ted, unsw_dl, "TED")
    auc_in = roc_auc_score(y_unsw_test, probs_in)
    print(f"TED In-Domain (UNSW->UNSW) AUC: {auc_in:.4f}")
    
    probs_zero = eval_model(ted, cic_dl, "TED")
    auc_zero = roc_auc_score(y_cic_test, probs_zero)
    print(f"TED Zero-Day AUC: {auc_zero:.4f}")
    eval_per_class(y_cic_test, probs_zero, cats_cic_test, "TED")
    
    # 6. UniMamba (No SSL)
    uni = BlockwiseEarlyExitMamba().to(DEVICE)
    sd = torch.load(f"{W}/students/student_no_kd.pth", map_location=DEVICE, weights_only=False)
    uni.load_state_dict(sd, strict=False)
    
    probs_in = eval_model(uni, unsw_dl, "UniMamba")
    auc_in = roc_auc_score(y_unsw_test, probs_in)
    print(f"UniMamba In-Domain AUC: {auc_in:.4f}")
    
    probs_zero = eval_model(uni, cic_dl, "UniMamba")
    auc_zero = roc_auc_score(y_cic_test, probs_zero)
    print(f"UniMamba Zero-Day AUC: {auc_zero:.4f}")

    # 7. KD Student
    kd = BlockwiseEarlyExitMamba().to(DEVICE)
    sd = torch.load(f"{W}/students/student_standard_kd.pth", map_location=DEVICE, weights_only=False)
    kd.load_state_dict(sd, strict=False)
    
    probs_in = eval_model(kd, unsw_dl, "KD")
    auc_in = roc_auc_score(y_unsw_test, probs_in)
    print(f"KD Student In-Domain AUC: {auc_in:.4f}")
    
    probs_zero = eval_model(kd, cic_dl, "KD")
    auc_zero = roc_auc_score(y_cic_test, probs_zero)
    print(f"KD Student Zero-Day AUC: {auc_zero:.4f}")

    # 8. BERT
    print("\n--- BERT ---")
    bert_enc = BertEncoder()
    bert = Classifier(bert_enc, d_model=128).to(DEVICE)
    sd = torch.load(f"{W}/teachers/teacher_bert_masking.pth", map_location=DEVICE, weights_only=False)
    bert.load_state_dict(sd, strict=False)
    
    probs_in = eval_model(bert, unsw_dl, "BERT")
    auc_in = roc_auc_score(y_unsw_test, probs_in)
    print(f"BERT In-Domain AUC: {auc_in:.4f}")
    
    probs_zero = eval_model(bert, cic_dl, "BERT")
    auc_zero = roc_auc_score(y_cic_test, probs_zero)
    print(f"BERT Zero-Day AUC: {auc_zero:.4f}")
    
if __name__ == "__main__":
    main()
