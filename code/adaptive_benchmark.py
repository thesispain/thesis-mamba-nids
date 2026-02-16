
import os, sys, pickle, time, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
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
             # exit_indices[active] stays as last index (2)
         
         return final_logits, exit_indices

def eval_TED_adaptive(X, y, attack_types, dataset_name):
    print(f"\n=== Evaluating TED (Adaptive) on {dataset_name} ===")
    
    ted = BlockwiseEarlyExitMamba(conf_thresh=0.85).to(DEVICE)
    sd = torch.load(f"{W}/students/student_ted.pth", map_location=DEVICE, weights_only=False)
    ted.load_state_dict(sd, strict=False)
    ted.eval()
    
    dl = DataLoader(TensorDataset(torch.from_numpy(X).to(DEVICE), torch.from_numpy(y).to(DEVICE)), batch_size=512)
    
    all_preds = []
    all_exits = []
    
    with torch.no_grad():
        for x, _ in dl:
            logits, exit_idxs = ted.forward_inference(x)
            preds = (F.softmax(logits, 1)[:, 1] > 0.5).long().cpu().numpy()
            all_preds.extend(preds)
            all_exits.extend(exit_idxs.cpu().numpy())
            
    all_preds = np.array(all_preds)
    all_exits = np.array(all_exits)
    
    # Calculate Exit Stats
    n_flows = len(all_exits)
    counts = np.bincount(all_exits, minlength=3)
    p8 = counts[0] / n_flows * 100
    p16 = counts[1] / n_flows * 100
    p32 = counts[2] / n_flows * 100
    mnp = (8*counts[0] + 16*counts[1] + 32*counts[2]) / n_flows
    
    print(f"Global MNP: {mnp:.2f} packets")
    print(f"Exit Dist: 8pkts={p8:.1f}% | 16pkts={p16:.1f}% | 32pkts={p32:.1f}%")
    
    # Per-Class Accuracy & Exit Depth
    unique_cats = np.unique(attack_types)
    print(f"\n{'Attack Class':<25} | {'Count':<8} | {'Accuracy':<10} | {'Exit@8(%)':<10}")
    print("-" * 65)
    
    results = [] # List of dicts for summary
    
    for cat in unique_cats:
        mask = (attack_types == cat)
        sub_preds = all_preds[mask]
        sub_y = y[mask]
        sub_exits = all_exits[mask]
        
        acc = np.mean(sub_preds == sub_y)
        
        # Invert Accuracy if Inverted (Naive inversion for display if global < 0.5?)
        # Let's just report raw for now.
        
        n_sub = len(sub_preds)
        c_sub = np.bincount(sub_exits, minlength=3)
        p8_sub = c_sub[0] / n_sub * 100
        
        print(f"{str(cat):<25} | {n_sub:<8} | {acc:.4f}     | {p8_sub:.1f}")
        
        results.append({
            "class": str(cat),
            "count": n_sub,
            "acc": acc,
            "exit8": p8_sub
        })
        
    return results

def main():
    # 1. Load Data
    print("Loading UNSW (Train Subset)...")
    with open(UNSW, 'rb') as f: unsw_data = pickle.load(f)
    unsw_sub = unsw_data[:100000] # 100k
    X_train = np.array([d['features'] for d in unsw_sub], dtype=np.float32)
    y_train = np.array([d['label'] for d in unsw_sub], dtype=np.int64)
    # Reconstruct attack category? UNSW 'attack_cat'?
    # Need to check keys.
    # Assuming 'attack_cat' key exists in unsw_sub dicts.
    cats_train = np.array([d.get('attack_cat', 'Normal' if d['label'] == 0 else 'Unknown') for d in unsw_sub])

    print("Loading CIC-IDS (Test Subset)...")
    with open(CIC, 'rb') as f: cic_data = pickle.load(f)
    cic_sub = cic_data[:100000] # 100k
    X_test = np.array([d['features'] for d in cic_sub], dtype=np.float32)
    y_test = np.array([d['label'] for d in cic_sub], dtype=np.int64)
    cats_test = np.array([d.get('attack_type', 'Benign' if d['label'] == 0 else 'Unknown') for d in cic_sub])

    # 2. Evaluate
    res_unsw = eval_TED_adaptive(X_train, y_train, cats_train, "UNSW-In-Domain")
    res_cic  = eval_TED_adaptive(X_test,  y_test,  cats_test,  "CIC-Cross-Dataset")

if __name__ == "__main__":
    main()
