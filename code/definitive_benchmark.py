
import os, sys, pickle, time, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import recall_score, roc_auc_score, f1_score
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
        # Corrected per benchmark script: 256->256
        self.proj_head = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 256))
        
    def forward(self, x):
        feat = self.tokenizer(x)
        feat_f, feat_b = feat, torch.flip(feat, [1]) 
        for fwd, rev in zip(self.layers, self.layers_rev):
             out_f = fwd(feat_f)
             out_r = rev(feat_b) 
             # Complex residual logic:
             feat_b_flipped = out_r.flip(1)
             feat = self.norm((out_f + feat_b_flipped) / 2 + feat)
             feat_f, feat_b = feat, torch.flip(feat, [1]) # Update inputs for next layer? No, usually feat is updated.
             # Wait, benchmark script:
             # out_f = fwd(feat)
             # out_r = rev(feat.flip(1)).flip(1)
             # feat = norm(...)
             # This implies feat is shared state.
             pass
        # Re-implement loop exactly:
        feat = self.tokenizer(x)
        for fwd, rev in zip(self.layers, self.layers_rev):
            out_f = fwd(feat)
            out_r = rev(feat.flip(1)).flip(1)
            feat = self.norm((out_f + out_r) / 2 + feat)
            
        # Return MEAN pooled + calc head
        return self.proj_head(feat.mean(1)), None

class Classifier(nn.Module):
    def __init__(self, encoder, d_model=256): 
        super().__init__()
        self.encoder = encoder
        # Corrected: 256 -> 64 -> 2 with Dropout
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
         # FORCE LAST PACKET (32) for fair comparison or Normal Logic?
         # User asked for "full test".
         return self.forward_inference_manual_force_last(x)

    def forward_inference_manual_force_last(self, x):
         B = x.size(0)
         feat = self._backbone(x)
         last_pos = self.exit_positions[-1]
         idx = min(last_pos, feat.size(1)) - 1
         h = feat[:, idx, :]
         logits = self.exit_classifiers[str(last_pos)](h)
         return logits, None

def train_xgboost(X, y, depth=32, name="Full"):
    print(f"Training XGBoost ({name})...")
    # Truncate features
    # Each packet has ~5 features? 256 dim embeddings? No, raw features.
    # Dataset has 'features': (32, 5).
    # Flatten: 160 dim.
    # If depth=8: (8, 5) -> 40 dim.
    
    X_flat = X.reshape(X.shape[0], 32, -1)[:, :depth, :].reshape(X.shape[0], -1)
    
    clf = xgb.XGBClassifier(n_estimators=100, max_depth=6, n_jobs=-1, device='cuda')
    clf.fit(X_flat, y)
    return clf

def eval_per_class(true_y, pred_probs, attack_types, model_name):
    unique_types = np.unique(attack_types)
    # Filter for interesting ones
    targets = ['Benign', 'FTP-Patator', 'SSH-Patator', 'DoS Hulk', 'DoS GoldenEye']
    
    res = {}
    
    benign_mask = (attack_types == 'Benign')
    benign_probs = pred_probs[benign_mask]
    
    for atk in targets:
        if atk not in unique_types: continue
        
        if atk == 'Benign':
            res[atk] = "N/A"
            continue
            
        atk_mask = (attack_types == atk)
        atk_probs = pred_probs[atk_mask]
        
        combined_probs = np.concatenate([benign_probs, atk_probs])
        combined_y = np.concatenate([np.zeros(len(benign_probs)), np.ones(len(atk_probs))])
        
        try: auc = roc_auc_score(combined_y, combined_probs)
        except: auc = 0.5
        
        res[atk] = auc
        
    return res

def main():
    print("=== FINAL DEFINITIVE BENCHMARK ===")
    
    # 1. Load Data
    print("Loading UNSW (Train)...")
    with open(UNSW, 'rb') as f: unsw_data = pickle.load(f)
    unsw_sub = unsw_data[:200000] # Subsample 200k for training XGB
    X_train = np.array([d['features'] for d in unsw_sub], dtype=np.float32)
    y_train = np.array([d['label'] for d in unsw_sub], dtype=np.int64)

    print("Loading CIC-IDS (Test)...")
    with open(CIC, 'rb') as f: cic_data = pickle.load(f)
    cic_sub = cic_data[:500000] # 500k for testing
    X_test = np.array([d['features'] for d in cic_sub], dtype=np.float32)
    y_test = np.array([d['label'] for d in cic_sub], dtype=np.int64)
    attack_types = np.array([d.get('attack_type', 'Unknown') for d in cic_sub])

    # 2. Train/Test XGBoost
    xgb_full = train_xgboost(X_train, y_train, 32, "Full-32")
    xgb_8    = train_xgboost(X_train, y_train, 8,  "Limited-8")
    
    X_test_32 = X_test.reshape(X_test.shape[0], -1)
    X_test_8  = X_test.reshape(X_test.shape[0], 32, -1)[:, :8, :].reshape(X_test.shape[0], -1)
    
    y_prob_xgb_32 = xgb_full.predict_proba(X_test_32)[:, 1]
    y_prob_xgb_8  = xgb_8.predict_proba(X_test_8)[:, 1]
    
    auc_xgb_32 = roc_auc_score(y_test, y_prob_xgb_32)
    auc_xgb_8  = roc_auc_score(y_test, y_prob_xgb_8)
    
    res_xgb_32 = eval_per_class(y_test, y_prob_xgb_32, attack_types, "XGBoost-32")
    res_xgb_8  = eval_per_class(y_test, y_prob_xgb_8,  attack_types, "XGBoost-8")
    
    # 3. Test DL Models
    dl = DataLoader(TensorDataset(torch.from_numpy(X_test).to(DEVICE), torch.from_numpy(y_test).to(DEVICE)), batch_size=512)
    
    # TED
    print("\nEvaluating TED...")
    ted = BlockwiseEarlyExitMamba().to(DEVICE)
    sd = torch.load(f"{W}/students/student_ted.pth", map_location=DEVICE, weights_only=False)
    ted.load_state_dict(sd, strict=False)
    ted.eval()
    
    probs_ted = []
    with torch.no_grad():
        for x, _ in dl:
            logits, _ = ted.forward_inference(x)
            probs_ted.extend(F.softmax(logits, 1)[:, 1].cpu().numpy())
    probs_ted = np.array(probs_ted)
    auc_ted = roc_auc_score(y_test, probs_ted)
    res_ted = eval_per_class(y_test, probs_ted, attack_types, "TED-Student")
    
    # BiMamba
    print("Evaluating BiMamba...")
    enc = BiMambaEncoder()
    bimamba = Classifier(enc).to(DEVICE)
    sd_t = torch.load(f"{W}/teachers/teacher_bimamba_retrained.pth", map_location=DEVICE, weights_only=False)
    bimamba.load_state_dict(sd_t, strict=False)
    bimamba.eval()
    
    probs_bi = []
    with torch.no_grad():
        for x, _ in dl:
            logits = bimamba(x)
            probs_bi.extend(F.softmax(logits, 1)[:, 1].cpu().numpy())
    probs_bi = np.array(probs_bi)
    auc_bi = roc_auc_score(y_test, probs_bi)
    res_bi = eval_per_class(y_test, probs_bi, attack_types, "BiMamba-Teacher")
    
    # 4. Print Table
    print(f"\n{'Model':<15} | {'Global AUC':<10} | {'FTP-Patator':<12} | {'SSH-Patator':<12} | {'DoS Hulk':<10}")
    print("-" * 75)
    
    rows = [
        ("XGBoost-32", auc_xgb_32, res_xgb_32),
        ("XGBoost-8",  auc_xgb_8,  res_xgb_8),
        ("BiMamba",    auc_bi,     res_bi),
        ("TED",        auc_ted,    res_ted)
    ]
    
    for name, g_auc, res in rows:
        ftp = res.get('FTP-Patator', 0.0)
        ssh = res.get('SSH-Patator', 0.0)
        hulk = res.get('DoS Hulk', 0.0)
        print(f"{name:<15} | {g_auc:.4f}     | {ftp:.4f}       | {ssh:.4f}       | {hulk:.4f}")
        
    # Save Report
    with open("DEFINITIVE_RESULTS.md", 'w') as f:
        f.write("# Definitive Benchmark Results (CIC-IDS)\n\n")
        f.write("| Model | Global AUC | FTP-Patator | SSH-Patator | DoS Hulk |\n")
        f.write("| --- | --- | --- | --- | --- |\n")
        for name, g_auc, res in rows:
            ftp = res.get('FTP-Patator', 0.0)
            ssh = res.get('SSH-Patator', 0.0)
            hulk = res.get('DoS Hulk', 0.0)
            f.write(f"| {name} | {g_auc:.4f} | {ftp:.4f} | {ssh:.4f} | {hulk:.4f} |\n")

if __name__ == "__main__":
    main()
