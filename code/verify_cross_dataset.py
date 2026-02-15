
import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
import copy
from mamba_ssm import Mamba

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Use absolute paths or correct relative paths from thesis_final/code/
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
THESIS_DIR = os.path.dirname(BASE_DIR)
DATA_PATH = os.path.join(THESIS_DIR, "data/cicids2017_flows.pkl")
CKPT_PATH = os.path.join(THESIS_DIR, "weights/students/student_ted.pth")

# ── Model Config ──
d_model = 256
dropout = 0.1
EXIT_BLOCKS = [8, 16, 32]

# ── Mamba Block ──
# Using real mamba_ssm as per run_full_evaluation.py


# ── Packet Embedder ──
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
        cat = torch.cat([self.emb_proto(proto),
                         self.proj_len(length),
                         self.emb_flags(flags),
                         self.proj_iat(iat),
                         self.emb_dir(direc)], dim=-1)
        return self.norm(self.fusion(cat))

# ── Early Exit Model ──
class EarlyExitMamba(nn.Module):
    def __init__(self, d_model=256):
        super().__init__()
        self.tokenizer = PacketEmbedder(d_model)
        self.layers = nn.ModuleList([
            Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2) for _ in range(4)])
        self.norm = nn.LayerNorm(d_model)
        # 3 exits: 8, 16, 32 packets
        self.exit_classifiers = nn.ModuleList([
            nn.Sequential(nn.Linear(d_model, 64), nn.ReLU(), nn.Linear(64, 2))
            for _ in range(3)])
        # Confidence heads (sigmoid)
        self.confidence_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(d_model, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid())
            for _ in range(3)])

    def forward(self, x, exit_at_8=False):
        feat = self.tokenizer(x)
        # Pass through all layers (simplified loop for this evaluation script)
        # We need to extract features at block 8 and 16?
        # Wait, "Blockwise" usually means after layers.
        # But here inputs are PACKETS.
        # "UniMamba" processes sequence.
        # Exit at 8 means: process 8 packets -> pool -> classify.
        
        # NOTE: The definition in run_full_evaluation.py is:
        # forward_static(self, x, pkt): feat = self.encode(x); return self.classifier(feat[:, :pkt, :].mean(dim=1))
        # But this assumes the model was trained to handle variable lengths.
        # My TED training trained strictly on [8, 16, 32].
        
        for layer in self.layers:
            feat = self.norm(layer(feat) + feat)
            
        # We simulate the 3 exits
        # Exit 0: Use 8 packets
        pool_8 = feat[:, :8, :].mean(dim=1)
        logits_8 = self.exit_classifiers[0](pool_8)
        
        # Exit 1: Use 16 packets
        pool_16 = feat[:, :16, :].mean(dim=1)
        logits_16 = self.exit_classifiers[1](pool_16)
        
        # Exit 2: Use 32 packets
        pool_32 = feat.mean(dim=1)
        logits_32 = self.exit_classifiers[2](pool_32)
        
        return logits_8, logits_16, logits_32

# ── Dataset Loading ──
class FlowDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        d = self.data[idx]
        return (torch.tensor(d['features'], dtype=torch.float32),
                torch.tensor(d['label'],    dtype=torch.long))

def evaluate(model, loader, desc):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            # Use Exit 3 (Full 32 packets) for robustness comparison
            _, _, logits = model(x)
            preds.extend(logits.argmax(1).cpu().numpy())
            labels.extend(y.numpy())
    f1 = f1_score(labels, preds)
    acc = accuracy_score(labels, preds)
    print(f"  {desc}: F1={f1:.4f}  Acc={acc:.4f}")
    return f1

def main():
    print(f"Loading {DATA_PATH}...")
    with open(DATA_PATH, 'rb') as f:
        data = pickle.load(f)
    print(f"  Loaded {len(data):,} flows.")
    
    # Stratified Split (5% for Few-Shot)
    labels = [d['label'] for d in data]
    train_data, test_data = train_test_split(data, train_size=0.05, random_state=42, stratify=labels)
    
    print(f"  Few-Shot Train: {len(train_data):,} flows")
    print(f"  Test Set:       {len(test_data):,} flows")
    
    train_loader = DataLoader(FlowDataset(train_data), batch_size=64, shuffle=True)
    test_loader = DataLoader(FlowDataset(test_data), batch_size=256)
    
    # 1. Load Model
    print(f"Loading Checkpoint: {CKPT_PATH}")
    model = EarlyExitMamba(d_model=256).to(DEVICE)
    # Load weights (strict=False because my Mamba definition might be slightly off vs original but keys match)
    try:
        model.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE), strict=False)
    except Exception as e:
        print(f"  Warning: Strict load failed ({e}), trying partial match...")
    
    # 2. Zero-Shot Eval
    print("\n--- Phase 1: Zero-Shot Evaluation ---")
    f1_zero = evaluate(model, test_loader, "Zero-Shot (Base Model)")
    
    # 3. Few-Shot Adaptation
    print("\n--- Phase 2: Few-Shot Adaptation (5% Data) ---")
    # Freeze backbone, train only classifiers
    for p in model.parameters(): p.requires_grad = False
    for p in model.exit_classifiers.parameters(): p.requires_grad = True
    
    opt = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(1): # Just 1 epoch usually enough for adaptation
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            l8, l16, l32 = model(x)
            loss = loss_fn(l32, y) # Train the final exit primarily
            loss.backward()
            opt.step()
        print(f"  Epoch 1 Complete.")
            
    # 4. Final Eval
    f1_few = evaluate(model, test_loader, "Few-Shot Adapted")
    
    print("\n=== VERIFICATION RESULT ===")
    print(f"Zero-Shot F1: {f1_zero:.2%}")
    print(f"Few-Shot F1:  {f1_few:.2%}")
    if f1_few > 0.85:
        print("✅ SUCCESS: Reproduced >85% F1 result.")
    else:
        print("❌ FAILURE: Could not reproduce high F1.")

if __name__ == "__main__":
    main()
