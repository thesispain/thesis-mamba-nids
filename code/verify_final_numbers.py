
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
        features = [] # To accumulate if needed, but BiMamba usually sums
        # Re-implement exact logic from benchmark script:
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
         # Adaptive Early Exit Logic
         B = x.size(0)
         feat = self._backbone(x)
         final_logits = torch.zeros(B, 2, device=x.device)
         exit_indices = torch.full((B,), self.n_exits-1, device=x.device, dtype=torch.long)
         active = torch.ones(B, dtype=torch.bool, device=x.device)
         
         for i, pos in enumerate(self.exit_positions[:-1]):
             if not active.any(): break
             idx = min(pos, feat.size(1)) - 1
             h = feat[active, idx, :] # Only process active flows
             logits = self.exit_classifiers[str(pos)](h)
             
             # Calculate Confidence
             conf_input = torch.cat([h, logits], dim=-1)
             conf = self.confidence_heads[str(pos)](conf_input).squeeze(-1)
             
             should_exit = conf >= self.conf_thresh
             if should_exit.any():
                 active_idx = active.nonzero(as_tuple=True)[0]
                 exiting = active_idx[should_exit]
                 
                 final_logits[exiting] = logits[should_exit]
                 exit_indices[exiting] = i
                 active[exiting] = False
         
         if active.any():
             last_pos = self.exit_positions[-1]
             idx = min(last_pos, feat.size(1)) - 1
             h = feat[active, idx, :]
             final_logits[active] = self.exit_classifiers[str(last_pos)](h)
         
         return final_logits, exit_indices

def train_xgboost(X, y):
    print("Training XGBoost...")
    X_flat = X.reshape(X.shape[0], -1)
    clf = xgb.XGBClassifier(n_estimators=100, max_depth=6, n_jobs=-1, device='cuda')
    clf.fit(X_flat, y)
    return clf

def eval_per_class(true_y, pred_probs, attack_types, model_name):
    unique_types = np.unique(attack_types)
    targets = ['Benign', 'FTP-Patator', 'SSH-Patator', 'DoS Hulk', 'DoS GoldenEye']
    
    # Identify Benign
    benign_mask = (attack_types == 'Benign') 
    benign_probs = pred_probs[benign_mask]
    
    print(f"\n--- {model_name} Breakdown ---")
    print(f"{'Attack Class':<25} |Count   | AUC (vs Benign)")
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

def main():
    print("=== FINAL VERIFICATION & REDO ===")
    
    # 1. Load Data
    print("Loading UNSW (Train for Zero-Day)...")
    with open(UNSW, 'rb') as f: unsw_data = pickle.load(f)
    unsw_sub = unsw_data[:100000] 
    X_unsw = np.array([d['features'] for d in unsw_sub], dtype=np.float32)
    y_unsw = np.array([d['label'] for d in unsw_sub], dtype=np.int64)

    print("Loading CIC-IDS (Test for Zero-Day & Train for In-Domain)...")
    with open(CIC, 'rb') as f: cic_data = pickle.load(f)
    np.random.seed(42)
    np.random.shuffle(cic_data) # <--- CRITICAL FIX
    cic_sub = cic_data[:200000] # 200k
    
    # Split CIC for In-Domain Training
    cic_train_data = cic_sub[:100000]
    cic_test_data  = cic_sub[100000:]
    
    X_cic_train = np.array([d['features'] for d in cic_train_data], dtype=np.float32)
    y_cic_train = np.array([d['label'] for d in cic_train_data], dtype=np.int64)
    
    X_cic_test  = np.array([d['features'] for d in cic_test_data], dtype=np.float32)
    y_cic_test  = np.array([d['label'] for d in cic_test_data], dtype=np.int64)
    cats_cic_test = np.array([d.get('attack_type', 'Unknown') for d in cic_test_data])
    
    # 2. XGBoost In-Domain (Train on CIC, Test on CIC)
    print("\n--- XGBoost In-Domain (CIC->CIC) ---")
    xgb_in = train_xgboost(X_cic_train, y_cic_train)
    X_cic_test_flat = X_cic_test.reshape(X_cic_test.shape[0], -1)
    y_prob_in = xgb_in.predict_proba(X_cic_test_flat)[:, 1]
    auc_in = roc_auc_score(y_cic_test, y_prob_in)
    print(f"XGBoost In-Domain AUC: {auc_in:.4f}")
    
    # 3. XGBoost Zero-Day (Train on UNSW, Test on CIC)
    print("\n--- XGBoost Zero-Day (UNSW->CIC) ---")
    xgb_zero = train_xgboost(X_unsw, y_unsw)
    y_prob_zero = xgb_zero.predict_proba(X_cic_test_flat)[:, 1]
    auc_zero = roc_auc_score(y_cic_test, y_prob_zero)
    print(f"XGBoost Zero-Day AUC: {auc_zero:.4f}")
    eval_per_class(y_cic_test, y_prob_zero, cats_cic_test, "XGBoost Zero-Day")
    
    # 4. BiMamba Zero-Day
    print("\n--- BiMamba Zero-Day ---")
    enc = BiMambaEncoder()
    bimamba = Classifier(enc).to(DEVICE)
    sd_t = torch.load(f"{W}/teachers/teacher_bimamba_retrained.pth", map_location=DEVICE, weights_only=False)
    bimamba.load_state_dict(sd_t, strict=False)
    bimamba.eval()
    
    dl = DataLoader(TensorDataset(torch.from_numpy(X_cic_test).to(DEVICE), torch.from_numpy(y_cic_test).to(DEVICE)), batch_size=512)
    
    probs_bi = []
    with torch.no_grad():
        for x, _ in dl:
            logits = bimamba(x)
            probs_bi.extend(F.softmax(logits, 1)[:, 1].cpu().numpy())
    probs_bi = np.array(probs_bi)
    auc_bi = roc_auc_score(y_cic_test, probs_bi)
    print(f"BiMamba Zero-Day AUC: {auc_bi:.4f}")
    eval_per_class(y_cic_test, probs_bi, cats_cic_test, "BiMamba")
    
    # 5. TED Zero-Day
    print("\n--- TED Zero-Day ---")
    ted = BlockwiseEarlyExitMamba(conf_thresh=0.85).to(DEVICE)
    sd = torch.load(f"{W}/students/student_ted.pth", map_location=DEVICE, weights_only=False)
    ted.load_state_dict(sd, strict=False)
    ted.eval()
    
    probs_ted = []
    with torch.no_grad():
        for x, _ in dl:
            logits, _ = ted.forward_inference(x) # Adaptive
            probs_ted.extend(F.softmax(logits, 1)[:, 1].cpu().numpy())
    probs_ted = np.array(probs_ted)
    auc_ted = roc_auc_score(y_cic_test, probs_ted)
    print(f"TED Zero-Day AUC: {auc_ted:.4f}")
    eval_per_class(y_cic_test, probs_ted, cats_cic_test, "TED Adaptive")

if __name__ == "__main__":
    main()
