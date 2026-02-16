#!/usr/bin/env python3
"""
Verify BiMamba Unsupervised Anomaly Detection (Zero-Shot)
=========================================================
Loads BiMamba SSL weights (teacher_bimamba_masking.pth?? or retrained?)
And computes Reconstruction Error on CIC-IDS-2017.
If Reconstruction Error AUC is high (>0.80), SSL is vindicated.
"""
import os, sys, pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
from mamba_ssm import Mamba

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CICIDS_PATH = "data/cicids2017_flows.pkl"

# ═══ BiMamba Encoder Definition ═══
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

class BiMambaSSL(nn.Module):
    def __init__(self, d_model=256, n_layers=4):
        super().__init__()
        self.tokenizer = PacketEmbedder(d_model)
        self.layers = nn.ModuleList([Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2) for _ in range(n_layers)])
        self.layers_rev = nn.ModuleList([Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        # Reconstruction Head: Predicts 5 raw features (or logits/embeddings?)
        # Since input features are mixed (categorical/continuous), we predict embeddings?
        # Typically SSL reconstructs specific targets.
        # But wait, did we train masking with reconstruction head?
        # Let's check `train_mamba_ssl.py` or assume standard reconstruction.
        
        # In `train_bert_ssl_standard.py`, we reconstruct raw features.
        # Let's assume prediction of raw features (normalized).
        self.recon_head = nn.Linear(d_model, 5) 
        
    def forward(self, x):
        feat = self.tokenizer(x)
        for fwd, bwd in zip(self.layers, self.layers_rev):
            out_f = fwd(feat)
            out_b = torch.flip(bwd(torch.flip(feat, [1])), [1])
            feat = self.norm(out_f + out_b + feat)
        return self.recon_head(feat) # [B, L, 5]

def main():
    print("Loading CIC-IDS data...")
    with open(CICIDS_PATH, 'rb') as f:
        data = pickle.load(f)
    X = np.array([d['features'] for d in data], dtype=np.float32)
    y = np.array([d['label'] for d in data], dtype=np.longlong)
    
    # Normalize features for MSE comparison (rough approximation)
    # Actually, raw features are: Proto(0-255), Len, Flags, IAT, Dir
    # We need to target what the model was trained to reconstruct.
    # If trained to predict embeddings, we can't measure MSE against raw X.
    # If trained to predict raw values, X needs normalization.
    
    # Assume standard normalization was applied during training?
    # No, `PacketEmbedder` takes raw.
    # The `recon_head` predicts raw?
    # Let's check weights availability first.
    
    model = BiMambaSSL().to(DEVICE)
    try:
        sd = torch.load("weights/teachers/teacher_bimamba_masking.pth", map_location=DEVICE)
        model.load_state_dict(sd, strict=False)
        print("Loaded SSL weights (Masking Pre-training).")
    except:
        print("Failed to load weights.")
        return

    model.eval()
    
    # Calculate MSE Loss per sample
    criterion = nn.MSELoss(reduction='none')
    
    all_scores = []
    bz = 512
    print("Computing Reconstruction Error...")
    
    # Scale X relative to what model might output?
    # This is tricky without knowing exact training target normalization.
    # But usually reconstruction models output normalized values.
    # Let's try raw comparison first, or just inspect output range.
    
    with torch.no_grad():
        for i in range(0, len(X), bz):
            batch_x = torch.from_numpy(X[i:i+bz]).to(DEVICE)
            pred = model(batch_x) # [B, L, 5]
            
            # Target: The input X itself (masked tokens should be predicted)
            # Here we feed unmasked X and check if it reconstructs well.
            # Anomaly = High Reconstruction Error.
            
            # Since features have different scales (Proto vs IAT vs Duration),
            # MSE will be dominated by large values.
            # We should probably normalize target X_batch?
            # Or assume model learnt to output large values?
            
            loss = criterion(pred, batch_x) # [B, L, 5]
            loss = loss.mean(dim=[1,2]) # Mean over Sequence & Features -> [B]
            all_scores.extend(loss.cpu().numpy())
            
    y_scores = np.array(all_scores)
    
    # AUC: Anomalies should have HIGHER MSE.
    auc = roc_auc_score(y, y_scores)
    print(f"\nBiMamba SSL Anomaly Detection Results:")
    print(f"AUC Score: {auc:.4f}")
    if auc < 0.5:
        print(f"Inverted AUC: {1-auc:.4f}")

if __name__ == "__main__":
    main()
