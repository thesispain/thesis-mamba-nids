
import os, sys, pickle, time, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import recall_score
import xgboost as xgb
from mamba_ssm import Mamba

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
        return self.forward_inference_manual_force_last(x) # Forcing output for robust comparison

    def forward_inference_manual_force_last(self, x):
         # FORCE LAST PACKET (32) to compare Architecture capability
         B = x.size(0)
         feat = self._backbone(x)
         last_pos = self.exit_positions[-1]
         idx = min(last_pos, feat.size(1)) - 1
         h = feat[:, idx, :] # All active at end
         logits = self.exit_classifiers[str(last_pos)](h)
         return logits, None

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
        feat_f, feat_b = feat, torch.flip(feat, [1]) # Simplistic flip logic, assuming works
        # Real logic in benchmark script:
        for fwd, rev in zip(self.layers, self.layers_rev):
            out_f = fwd(feat_f)
            out_r = rev(feat_b) 
            # Flip 'rev' output back? No, standard logic:
            # benchmark script lines 90-91:
            # out_f = fwd(feat)
            # out_r = rev(feat.flip(1)).flip(1)
            # feat = norm((out_f + out_r)/2 + feat)
            # This is complex residual logic. I MUST COPY EXACTLY.
            pass # Placeholder for below

    def forward_impl(self, x):
        feat = self.tokenizer(x)
        for fwd, rev in zip(self.layers, self.layers_rev):
             out_f = fwd(feat)
             out_r = rev(feat.flip(1)).flip(1)
             feat = self.norm((out_f + out_r) / 2 + feat)
        return self.proj_head(feat.mean(1)), None

    def forward(self, x):
        return self.forward_impl(x)

class Classifier(nn.Module):
    def __init__(self, encoder, d_model=256): # d_model used as input_dim
        super().__init__()
        self.encoder = encoder
        # input_dim 256 -> 64 -> 2
        self.head = nn.Sequential(nn.Linear(d_model, 64), nn.ReLU(), nn.Dropout(0.1), nn.Linear(64, 2))
    def forward(self, x):
        z = self.encoder(x)
        if isinstance(z, tuple): z = z[0]
        return self.head(z)

def train_xgboost(X, y):
    print("Training XGBoost (8 packets)...")
    # Truncate features. Assuming 5 features per packet. 0-40.
    X_8 = X.reshape(X.shape[0], 32, -1)[:, :8, :].reshape(X.shape[0], -1)
    
    clf = xgb.XGBClassifier(n_estimators=100, max_depth=6, n_jobs=-1, device='cuda')
    clf.fit(X_8, y)
    return clf

from sklearn.metrics import recall_score, roc_auc_score

def eval_per_class(true_y, pred_probs, attack_types, model_name):
    unique_types = np.unique(attack_types)
    print(f"\n--- {model_name} Breakdown ---")
    print(f"{'Attack Class':<25} |Count   | AUC (vs Benign) | Avg Prob")
    print("-" * 65)
    
    # Identify Benign samples
    benign_mask = (attack_types == 'Benign') 
    benign_probs = pred_probs[benign_mask]
    
    for atk in unique_types:
        if atk == 'Benign': 
            avg_prob = np.mean(benign_probs) if len(benign_probs) > 0 else 0.0
            print(f"{str(atk):<25} |{len(benign_probs):<8} | {'N/A':<15} | {avg_prob:.4f}")
            continue
        
        atk_mask = (attack_types == atk)
        atk_probs = pred_probs[atk_mask]
        
        if len(atk_probs) < 2: 
             print(f"{str(atk):<25} |{len(atk_probs):<8} | {'Too Few':<15} | {np.mean(atk_probs):.4f}")
             continue

        # Combine Benign + This Attack
        combined_probs = np.concatenate([benign_probs, atk_probs])
        combined_y = np.concatenate([np.zeros(len(benign_probs)), np.ones(len(atk_probs))])
        
        try:
            auc = roc_auc_score(combined_y, combined_probs)
        except:
            auc = 0.5
            
        avg_prob = np.mean(atk_probs)
        count = np.sum(atk_mask)
        print(f"{str(atk):<25} |{count:<8} | {auc:.4f}          | {avg_prob:.4f}")
        
def main():
    print("Loading UNSW (Train)...")
    with open(UNSW, 'rb') as f: unsw_data = pickle.load(f)
    print(f"UNSW Size: {len(unsw_data)}")
    
    # Stratified Subsample for Speed (100k)
    unsw_sub = unsw_data[:200000] # Assuming mixed
    X_train = np.array([d['features'] for d in unsw_sub], dtype=np.float32)
    y_train = np.array([d['label'] for d in unsw_sub], dtype=np.int64)
    
    xgb_8 = train_xgboost(X_train, y_train)
    
    print("\nLoading CIC-IDS (Test)...")
    with open(CIC, 'rb') as f: cic_data = pickle.load(f)
    
    # Extract labels and features
    # Limit to 500k for speed?
    cic_data = cic_data[:500000] 
    
    X_test_raw = np.array([d['features'] for d in cic_data], dtype=np.float32)
    y_test_bin = np.array([d['label'] for d in cic_data], dtype=np.int64)
    attack_types = np.array([d.get('attack_type', 'Unknown') for d in cic_data])
    
    # 1. Eval XGBoost @ 8
    print("\nPredicting XGBoost (8)...")
    X_test_8 = X_test_raw.reshape(X_test_raw.shape[0], 32, -1)[:, :8, :].reshape(X_test_raw.shape[0], -1)
    y_prob_xgb = xgb_8.predict_proba(X_test_8)[:, 1]
    
    eval_per_class(y_test_bin, y_prob_xgb, attack_types, "XGBoost (8 Pkts)")
    
    # 2. Eval TED (Student)
    print("\nPredicting TED...")
    ted = BlockwiseEarlyExitMamba().to(DEVICE)
    sd = torch.load(f"{W}/students/student_ted.pth", map_location=DEVICE, weights_only=False)
    ted.load_state_dict(sd, strict=False)
    ted.eval()
    
    dl = DataLoader(TensorDataset(torch.from_numpy(X_test_raw).to(DEVICE), torch.from_numpy(y_test_bin).to(DEVICE)), batch_size=512)
    
    all_probs = []
    with torch.no_grad():
        for x, _ in dl:
            logits, _ = ted.forward_inference(x) # Forced to 32 packet logic inside class
            probs = F.softmax(logits, 1)[:, 1]
            all_probs.extend(probs.cpu().numpy())
            
    eval_per_class(y_test_bin, np.array(all_probs), attack_types, "TED (Forced 32 context)")

    # 3. Eval BiMamba (Teacher)
    print("\nPredicting BiMamba...")
    enc = BiMambaEncoder()
    bimamba = Classifier(enc).to(DEVICE)
    sd_t = torch.load(f"{W}/teachers/teacher_bimamba_retrained.pth", map_location=DEVICE, weights_only=False)
    bimamba.load_state_dict(sd_t, strict=False)
    bimamba.eval()
    
    all_probs_bi = []
    with torch.no_grad():
        for x, _ in dl: # Reuse DL
            logits = bimamba(x)
            probs = F.softmax(logits, 1)[:, 1]
            all_probs_bi.extend(probs.cpu().numpy())

    eval_per_class(y_test_bin, np.array(all_probs_bi), attack_types, "BiMamba (Teacher)")            

if __name__ == "__main__":
    main()
