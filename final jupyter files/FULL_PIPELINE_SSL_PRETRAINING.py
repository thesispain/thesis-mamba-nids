======================================================================
STAGE 0: IMPORTS & SETUP
======================================================================

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import os
import time
from pathlib import Path

# Mamba imports
from mamba_ssm import Mamba

print("✅ Mamba imported")

# Device setup
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print(f"✅ Device: {DEVICE}")
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print(f"⚠️ Device: {DEVICE}")

======================================================================
STAGE 1: PATHS & DATA
======================================================================

# Absolute paths (no more ../../../)
WORKSPACE_ROOT = Path('/home/T2510596/Downloads/totally fresh')
DATA_DIR = WORKSPACE_ROOT / 'thesis_final' / 'data'
WEIGHTS_DIR = WORKSPACE_ROOT / 'thesis_final' / 'weights' / 'ssl'
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

pretrain_path = DATA_DIR / 'pretrain_data.pt'
finetune_path = DATA_DIR / 'finetune_data.pt'
final_weight_path = WEIGHTS_DIR / 'bimamba_masking_ssl_final.pth'

print(f"📂 Loading data...")
pretrain_data = torch.load(pretrain_path)
finetune_data = torch.load(finetune_path)

print(f"✅ Pretrain: {len(pretrain_data)} flows")
print(f"✅ Finetune: {len(finetune_data)} flows")

======================================================================
STAGE 2: MODEL DEFINITION
======================================================================

class PacketEmbedder(nn.Module):
    def __init__(self, d_embed=256):
        super().__init__()
        self.embed = nn.Linear(5, d_embed)
    
    def forward(self, x):
        return self.embed(x)

class BiMambaEncoder(nn.Module):
    def __init__(self, d_model=256, n_layers=4, d_state=32):
        super().__init__()
        self.embedder = PacketEmbedder(d_model)
        self.mamba_fwd = nn.Sequential(*[Mamba(d_model, d_state) for _ in range(n_layers)])
        self.mamba_bwd = nn.Sequential(*[Mamba(d_model, d_state) for _ in range(n_layers)])
        self.projection = nn.Linear(d_model * 2, 128)
    
    def forward(self, x):
        emb = self.embedder(x)
        fwd = self.mamba_fwd(emb)
        bwd = self.mamba_bwd(torch.flip(emb, [0]))
        bwd = torch.flip(bwd, [0])
        combined = torch.cat([fwd, bwd], dim=-1)
        return self.projection(combined[-1])

print("✅ Models defined")

print("\n" + "="*70)
print("⚡ SMART WEIGHT DETECTION ⚡".center(70))
print("="*70)

print(f"\n🔍 Checking: {final_weight_path}\n")

if os.path.exists(final_weight_path):
    size_mb = os.path.getsize(final_weight_path) / 1e6
    print(f"✅ WEIGHTS FOUND ({size_mb:.0f} MB)!")
    print(f"\n🚀 LOADING WEIGHTS (skipping 30-minute training!)\n")
    print(f"⏱️ Time saved: ~30 minutes ⏭️\n")
    
    encoder = BiMambaEncoder(d_model=256, n_layers=4).to(DEVICE)
    encoder.load_state_dict(torch.load(final_weight_path, map_location=DEVICE, weights_only=False))
    encoder.eval()
    
    total_params = sum(p.numel() for p in encoder.parameters())
    print(f"✅ Model loaded: {total_params:,} parameters")
    SKIP_TRAINING = True
else:
    print(f"❌ Weights not found - training needed")
    SKIP_TRAINING = False

print("\n" + "="*70)

print("\n" + "="*70)
print("STAGE 3: ANOMALY DETECTION EVAL".center(70))
print("="*70 + "\n")

# Extract benign reference from finetune data
finetune_benign = finetune_data['benign'].cpu().numpy()
benign_ref_indices = np.random.choice(len(finetune_benign), size=2000, replace=False)
benign_ref = torch.tensor(finetune_benign[benign_ref_indices], dtype=torch.float32).to(DEVICE)

# Test sets
remaining_indices = np.setdiff1d(np.arange(len(finetune_benign)), benign_ref_indices)
benign_test = torch.tensor(finetune_benign[remaining_indices[:2500]], dtype=torch.float32).to(DEVICE)
attack_test = torch.tensor(finetune_data['attack'].cpu().numpy()[:312], dtype=torch.float32).to(DEVICE)

print("Preparing test set...")
print(f"   Benign reference: {len(benign_ref)}")
print(f"   Benign test: {len(benign_test)}")
print(f"   Attack test: {len(attack_test)}\n")

print("🧠 Encoding samples...")
with torch.no_grad():
    ref_encodings = encoder(benign_ref).cpu().numpy()
    test_encodings = torch.cat([
        encoder(benign_test),
        encoder(attack_test)
    ]).cpu().numpy()

print(f"   ✅ Encodings computed")
print(f"      Reference: {ref_encodings.shape}")
print(f"      Test: {test_encodings.shape}\n")

print("Computing anomaly scores...\n")

# Normalize
ref_norm = ref_encodings / (np.linalg.norm(ref_encodings, axis=1, keepdims=True) + 1e-8)
test_norm = test_encodings / (np.linalg.norm(test_encodings, axis=1, keepdims=True) + 1e-8)

# Score: similarity to top-10 closest references
scores = np.zeros(len(test_norm))
for i, test_vec in enumerate(test_norm):
    sims = np.dot(ref_norm, test_vec)
    scores[i] = -np.mean(np.sort(sims)[-10:])  # Negative because lower is more anomalous

# Labels
y_true = np.concatenate([np.zeros(len(benign_test)), np.ones(len(attack_test))])

# AUC
auc = roc_auc_score(y_true, scores)

print("="*70)
print(f"🎯 UNSUPERVISED AUC: {auc:.4f}".center(70))
print("="*70)

if auc >= 0.7:
    print("\n✅ Excellent - AUC >= 0.7")
elif auc >= 0.6:
    print("\n⚠️ Good - AUC >= 0.6")
elif auc >= 0.5:
    print("\n⚠️ Moderate - AUC >= 0.5")
else:
    print("\n❌ Poor - AUC < 0.5")

print("\n" + "="*70)
print("📋 FINAL SUMMARY".center(70))
print("="*70)

print(f"\n✅ COMPLETE!")

print(f"\n📁 Weights:")
print(f"   {final_weight_path}")
if os.path.exists(final_weight_path):
    size_mb = os.path.getsize(final_weight_path) / 1e6
    print(f"   Size: {size_mb:.0f} MB")

print(f"\n🎯 Performance:")
print(f"   Unsupervised AUC: {auc:.4f}")
if SKIP_TRAINING:
    print(f"   Training Status: SKIPPED ✅ (Weights loaded)")
    print(f"   Time Saved: ~30 minutes ⏭️")
else:
    print(f"   Training Status: FRESH TRAINED")

print(f"\n💡 Key Answer to Your Question:")
print(f"   ❓ Why 30 mins if we have weights?")
print(f"   ✅ Answer: We don't! This notebook detects weights and loads them instantly!")

print(f"\n✨ Absolute Path Solution:")
print(f"   ✅ All paths are absolute (no more ../../../)")
print(f"   ✅ Works from any subdirectory")
print(f"   ✅ No relative path breaking")

print(f"\n{'='*70}")
print("✅ PIPELINE COMPLETE!".center(70))
print("="*70)

print("="*70)
print("STAGE 0: IMPORTS & SETUP")
print("="*70)

import os, sys, pickle, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

try:
    from mamba_ssm import Mamba
    print("✅ Mamba imported")
except:
    os.system('pip install mamba-ssm -q')
    from mamba_ssm import Mamba
    print("✅ Mamba installed & imported")

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"✅ Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name()}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

print("\n" + "="*70)
print("STAGE 1: PATHS & DATA")
print("="*70)

WORKSPACE = "/home/T2510596/Downloads/totally fresh"
WEIGHTS_DIR = os.path.join(WORKSPACE, "thesis_final/weights/ssl")
DATA_DIR = os.path.join(WORKSPACE, "Organized_Final/data/unswnb15_full")
PRETRAIN_PKL = os.path.join(DATA_DIR, "pretrain_50pct_benign.pkl")
FINETUNE_PKL = os.path.join(DATA_DIR, "finetune_mixed.pkl")
final_weight_path = os.path.join(WEIGHTS_DIR, "bimamba_masking_ssl_final.pth")

print(f"\n📂 Loading data...")
with open(PRETRAIN_PKL, 'rb') as f:
    pretrain_data = pickle.load(f)
with open(FINETUNE_PKL, 'rb') as f:
    finetune_data = pickle.load(f)

print(f"✅ Pretrain: {len(pretrain_data):,} flows")
print(f"✅ Finetune: {len(finetune_data):,} flows")

print("\n" + "="*70)
print("STAGE 2: MODEL DEFINITION")
print("="*70)

class PacketEmbedder(nn.Module):
    def __init__(self, d_model=256):
        super().__init__()
        self.emb_proto = nn.Embedding(256, 32)
        self.emb_flags = nn.Embedding(64, 32)
        self.emb_dir = nn.Embedding(2, 8)
        self.proj_len = nn.Linear(1, 32)
        self.proj_iat = nn.Linear(1, 32)
        self.fusion = nn.Linear(136, d_model)
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x):
        proto = x[:,:,0].long().clamp(0, 255)
        length = x[:,:,1:2]
        flags = x[:,:,2].long().clamp(0, 63)
        iat = x[:,:,3:4]
        direction = x[:,:,4].long().clamp(0, 1)
        
        e_p = self.emb_proto(proto)
        e_f = self.emb_flags(flags)
        e_d = self.emb_dir(direction)
        e_l = self.proj_len(length)
        e_i = self.proj_iat(iat)
        
        cat = torch.cat([e_p, e_f, e_d, e_l, e_i], dim=-1)
        return self.norm(self.fusion(cat))

class BiMambaEncoder(nn.Module):
    def __init__(self, d_model=256, n_layers=4):
        super().__init__()
        self.tokenizer = PacketEmbedder(d_model)
        self.layers = nn.ModuleList([Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2) for _ in range(n_layers)])
        self.layers_rev = nn.ModuleList([Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.proj_head = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 128))
        self.recon_head = nn.Linear(d_model, 5)
    
    def forward(self, x):
        x_emb = self.tokenizer(x)
        feat = x_emb
        for fwd, bwd in zip(self.layers, self.layers_rev):
            out_f = fwd(feat)
            feat_rev = torch.flip(feat, dims=[1])
            out_b = bwd(feat_rev)
            out_b = torch.flip(out_b, dims=[1])
            feat = self.norm(out_f + out_b + feat)
        z = self.proj_head(feat.mean(dim=1))
        recon = self.recon_head(feat)
        return z, recon

print("✅ Models defined")

print("\n" + "="*70)
print("⚡ SMART WEIGHT DETECTION ⚡")
print("="*70)

print(f"\n🔍 Checking: {final_weight_path}")

if os.path.exists(final_weight_path):
    size_mb = os.path.getsize(final_weight_path) / 1e6
    print(f"\n✅ WEIGHTS FOUND ({size_mb:.0f} MB)!")
    print(f"\n🚀 LOADING WEIGHTS (skipping 30-minute training!)")
    print(f"\n⏱️ Time saved: ~30 minutes ⏭️")
    
    encoder = BiMambaEncoder(d_model=256, n_layers=4).to(DEVICE)
    encoder.load_state_dict(torch.load(final_weight_path, map_location=DEVICE, weights_only=False))
    encoder.eval()
    
    params = sum(p.numel() for p in encoder.parameters())
    print(f"\n✅ Model loaded: {params:,} parameters")
    SKIP_TRAINING = True
else:
    print(f"\n❌ Weights not found")
    print(f"Training will be needed (~30 mins)")
    SKIP_TRAINING = False

print("\n" + "="*70)
print("STAGE 3: ANOMALY DETECTION EVAL")
print("="*70)

print("\nPreparing test set...")
test_sample = finetune_data[:5000]
benign_test = [d for d in test_sample if d['label'] == 0][:2500]
attack_test = [d for d in test_sample if d['label'] == 1][:2500]

test_flows = benign_test + attack_test
test_labels = [0]*len(benign_test) + [1]*len(attack_test)
benign_ref = pretrain_data[:2000]

print(f"   Benign reference: {len(benign_ref):,}")
print(f"   Benign test: {len(benign_test):,}")
print(f"   Attack test: {len(attack_test):,}")

print("\n🧠 Encoding samples...")
with torch.no_grad():
    # Reference
    ref_reps = []
    for i in range(0, len(benign_ref), 256):
        batch = benign_ref[i:i+256]
        feats = np.stack([d['features'] if isinstance(d, dict) else d for d in batch])
        x = torch.from_numpy(feats).float().to(DEVICE)
        z, _ = encoder(x)
        ref_reps.append(z.cpu().numpy())
    ref_reps = np.concatenate(ref_reps, axis=0)
    ref_reps = ref_reps / (np.linalg.norm(ref_reps, axis=1, keepdims=True) + 1e-8)
    
    # Test
    test_reps = []
    for i in range(0, len(test_flows), 256):
        batch = test_flows[i:i+256]
        feats = np.stack([d['features'] if isinstance(d, dict) else d for d in batch])
        x = torch.from_numpy(feats).float().to(DEVICE)
        z, _ = encoder(x)
        test_reps.append(z.cpu().numpy())
    test_reps = np.concatenate(test_reps, axis=0)
    test_reps = test_reps / (np.linalg.norm(test_reps, axis=1, keepdims=True) + 1e-8)
    
    print(f"   ✅ Encodings computed")
    print(f"      Reference: {ref_reps.shape}")
    print(f"      Test: {test_reps.shape}")

print("\nComputing anomaly scores...")
with torch.no_grad():
    scores = []
    for i in range(0, len(test_reps), 200):
        chunk = test_reps[i:i+200]
        sim = chunk @ ref_reps.T
        topk = np.sort(sim, axis=1)[:, -10:]
        scores.extend(topk.mean(axis=1).tolist())
    
    auc = roc_auc_score(test_labels, scores)

print(f"\n{'='*70}")
print(f"🎯 UNSUPERVISED AUC: {auc:.4f}")
print(f"{'='*70}")

if auc > 0.85:
    print(f"\n✨ Excellent! AUC > 0.85")
elif auc > 0.75:
    print(f"\n✅ Good! AUC > 0.75")
elif auc > 0.5:
    print(f"\n⚠️ Fair - AUC > 0.5 (random is 0.5)")
else:
    print(f"\n❌ Poor - AUC < 0.5")

print(f"\n{'='*70}")
print("📋 FINAL SUMMARY")
print(f"{'='*70}")

print(f"""
✅ COMPLETE!

📁 Weights:
   {final_weight_path}
   Size: {os.path.getsize(final_weight_path)/1e6:.0f} MB

🎯 Performance:
   Unsupervised AUC: {auc:.4f}
   Training Status: {'SKIPPED ✅ (Weights loaded)' if SKIP_TRAINING else 'Would need training'}
   Time Saved: {'~30 minutes ⏭️' if SKIP_TRAINING else 'N/A'}

💡 Key Answer to Your Question:
   ❓ Why 30 mins if we have weights?
   ✅ Answer: We don't! This notebook detects weights and loads them instantly!
   
✨ Absolute Path Solution:
   ✅ All paths are absolute (no more ../../../)
   ✅ Works from any subdirectory
   ✅ No relative path breaking
""")

print(f"{'='*70}")
print("✅ PIPELINE COMPLETE!")
print(f"{'='*70}")

encoder.cpu()
torch.cuda.empty_cache()

import os
import sys
import pickle
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from collections import Counter
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

print("✅ Core imports complete")

# Mamba-SSM for state space models
try:
    from mamba_ssm import Mamba
    print("✅ Mamba SSM imported")
except ImportError:
    print("❌ Missing mamba_ssm. Install: pip install mamba-ssm")
    sys.exit(1)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n{'='*70}")
print(f"Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print(f"{'='*70}\n")

# SOLUTION TO PATH ISSUE:
# All paths are absolute, rooted at WORKSPACE_ROOT
# Works from any subdirectory without breaking relative path chains

WORKSPACE_ROOT = "/home/T2510596/Downloads/totally fresh"
ORGANIZED_FINAL = os.path.join(WORKSPACE_ROOT, "Organized_Final")
THESIS_FINAL = os.path.join(WORKSPACE_ROOT, "thesis_final")

DATA_DIR = os.path.join(ORGANIZED_FINAL, "data", "unswnb15_full")
PRETRAIN_PKL = os.path.join(DATA_DIR, "pretrain_50pct_benign.pkl")
FINETUNE_PKL = os.path.join(DATA_DIR, "finetune_mixed.pkl")

OUTPUT_DIR = os.path.join(THESIS_FINAL, "results_ssl_pipeline")
WEIGHTS_DIR = os.path.join(THESIS_FINAL, "weights", "ssl")
LOGS_DIR = os.path.join(OUTPUT_DIR, "logs")

for d in [OUTPUT_DIR, WEIGHTS_DIR, LOGS_DIR]:
    os.makedirs(d, exist_ok=True)

print("📁 Paths configured")
print(f"   Data: {DATA_DIR}")
print(f"   Weights: {WEIGHTS_DIR}")
print(f"   Logs: {LOGS_DIR}\n")

# Verify data exists
for fpath, fname in [(PRETRAIN_PKL, "Pretrain"), (FINETUNE_PKL, "Finetune")]:
    if os.path.exists(fpath):
        size_mb = os.path.getsize(fpath) / 1e6
        print(f"✅ {fname}: {size_mb:.0f} MB")
    else:
        print(f"❌ {fname}: NOT FOUND at {fpath}")

print(f"\n{'='*70}\nSTAGE 3: Load Preprocessed Flow Data\n{'='*70}")

print("📂 Loading pretrain data...")
with open(PRETRAIN_PKL, 'rb') as f:
    pretrain_data = pickle.load(f)

print(f"✅ Loaded {len(pretrain_data):,} flows")
print(f"\nSample flow structure:")
if isinstance(pretrain_data[0], dict):
    print(f"   Keys: {list(pretrain_data[0].keys())}")
    print(f"   Features shape: {pretrain_data[0]['features'].shape}")
    print(f"   Label: {pretrain_data[0]['label']}")
else:
    print(f"   Type: {type(pretrain_data[0])}")
    print(f"   Shape: {pretrain_data[0].shape if hasattr(pretrain_data[0], 'shape') else 'N/A'}")

print(f"\n{'='*70}\nSTAGE 4: Define Model Architectures\n{'='*70}")

class PacketEmbedder(nn.Module):
    """Embed 5 packet features [Proto, LogLen, Flags, LogIAT, Direction] → 256-d"""
    def __init__(self, d_model=256):
        super().__init__()
        self.emb_proto = nn.Embedding(256, 32)  # Protocol: 0-255 → 32-d
        self.emb_flags = nn.Embedding(64, 32)   # Flags: 0-63 → 32-d
        self.emb_dir = nn.Embedding(2, 8)       # Direction: 0-1 → 8-d
        self.proj_len = nn.Linear(1, 32)        # LogLen: 1-d → 32-d
        self.proj_iat = nn.Linear(1, 32)        # LogIAT: 1-d → 32-d
        # Total: 32+32+8+32+32 = 136-d → fuse to 256-d
        self.fusion = nn.Linear(136, d_model)
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x):
        # x: (B, T, 5) where B=batch, T=seq_len, 5=features
        proto = x[:,:,0].long().clamp(0, 255)
        length = x[:,:,1:2]
        flags = x[:,:,2].long().clamp(0, 63)
        iat = x[:,:,3:4]
        direction = x[:,:,4].long().clamp(0, 1)
        
        e_p = self.emb_proto(proto)    # (B, T, 32)
        e_f = self.emb_flags(flags)    # (B, T, 32)
        e_d = self.emb_dir(direction)  # (B, T, 8)
        e_l = self.proj_len(length)    # (B, T, 32)
        e_i = self.proj_iat(iat)       # (B, T, 32)
        
        cat = torch.cat([e_p, e_f, e_d, e_l, e_i], dim=-1)  # (B, T, 136)
        return self.norm(self.fusion(cat))  # (B, T, 256)

print("✅ PacketEmbedder defined")

class BiMambaEncoder(nn.Module):
    """Bidirectional Mamba encoder with contrastive projection head"""
    def __init__(self, d_model=256, n_layers=4):
        super().__init__()
        self.tokenizer = PacketEmbedder(d_model)
        
        # Bidirectional layers
        self.layers = nn.ModuleList([
            Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
            for _ in range(n_layers)
        ])
        self.layers_rev = nn.ModuleList([
            Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
            for _ in range(n_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        
        # Contrastive projection head
        self.proj_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 128)  # Project to 128-d for NT-Xent
        )
        
        # Reconstruction head (for auxiliary loss, optional)
        self.recon_head = nn.Linear(d_model, 5)
    
    def forward(self, x):
        x_emb = self.tokenizer(x)  # (B, T, 256)
        feat = x_emb
        
        # Bidirectional processing
        for fwd, bwd in zip(self.layers, self.layers_rev):
            out_f = fwd(feat)                              # Forward
            feat_rev = torch.flip(feat, dims=[1])          # Reverse sequence
            out_b = bwd(feat_rev)                          # Backward
            out_b = torch.flip(out_b, dims=[1])            # Flip back
            feat = self.norm(out_f + out_b + feat)         # Residual + fuse
        
        # Global representation (mean pooling)
        global_rep = feat.mean(dim=1)  # (B, 256)
        
        z = self.proj_head(global_rep)  # (B, 128) - for contrastive loss
        recon = self.recon_head(feat)   # (B, T, 5) - reconstruction
        
        return z, recon

print("✅ BiMambaEncoder defined")

# Test model architecture
model_test = BiMambaEncoder(d_model=256, n_layers=4).to(DEVICE)
params = sum(p.numel() for p in model_test.parameters())
print(f"✅ BiMambaEncoder: {params:,} parameters (~{params/1e6:.1f}M)")

# Test forward pass
x_dummy = torch.randn(2, 32, 5).to(DEVICE)  # Move to device
z, recon = model_test(x_dummy)
print(f"   Input: {x_dummy.shape}")
print(f"   Contrastive output (z): {z.shape}")
print(f"   Reconstruction output: {recon.shape}")
print(f"   ✅ Architecture test passed")
del model_test

print(f"\n{'='*70}\nSTAGE 5: Anti-Shortcut Masking Augmentation\n{'='*70}")

class AntiShortcutAugmentation:
    """
    Selective masking to prevent model from learning shortcuts:
    - Pro(col 0): 20% mask rate
    - LogLen(col 1): 50% mask rate ← PRIMARY DISCRIMINATOR
    - Flags(col 2): 30% mask rate
    - LogIAT(col 3): 0% mask rate ← NEVER mask (temporal dynamics)
    - Direction(col 4): 10% mask rate
    
    Plus jitter on LogIAT to simulate timing noise.
    """
    def __init__(self):
        self.mask_probs = {
            0: 0.20,  # Proto
            1: 0.50,  # LogLen (HEAVY!)
            2: 0.30,  # Flags
            3: 0.00,  # LogIAT (NEVER mask)
            4: 0.10,  # Dir
        }
        self.jitter_scale = 0.05
    
    def __call__(self, features):
        """Apply masking to features (numpy array, shape=(T,5))"""
        aug = features.copy()
        
        # Apply column-wise masking
        for col, prob in self.mask_probs.items():
            if prob > 0 and np.random.random() < prob:
                aug[:, col] = 0.0
        
        # Jitter on LogIAT to simulate timing variations
        aug[:, 3] += np.random.randn(aug.shape[0]) * self.jitter_scale
        
        return aug

print("✅ AntiShortcutAugmentation class defined")

# Test augmentation
aug = AntiShortcutAugmentation()
features_test = np.random.randn(32, 5)
features_aug = aug(features_test)
print(f"\n   Original shape: {features_test.shape}")
print(f"   Augmented shape: {features_aug.shape}")
print(f"   Columns masked (showing sparsity):")
for col in range(5):
    sparsity = (features_aug[:, col] == 0).sum() / len(features_aug)
    print(f"      Col {col}: {sparsity:.1%}")

class ContrastiveDataset(Dataset):
    """Returns (x_original, x_augmented) pairs for NT-Xent loss"""
    def __init__(self, data, augmentor):
        self.data = data
        self.augmentor = augmentor
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data[idx]
        # Handle both dict and array formats
        f = row['features'] if isinstance(row, dict) else row
        
        # Original
        x = torch.from_numpy(f).float()
        
        # Augmented
        x_aug = torch.from_numpy(self.augmentor(f)).float()
        
        return x, x_aug

print("✅ ContrastiveDataset defined")

print(f"\n{'='*70}\nSTAGE 6: NT-Xent Contrastive Loss\n{'='*70}")

def nt_xent_loss(z_i, z_j, temperature=0.5):
    """
    Normalized Temperature-scaled Cross Entropy Loss
    From SimCLR paper.
    
    Args:
        z_i: Projections from augmented view 1, shape (B, d)
        z_j: Projections from augmented view 2, shape (B, d)
        temperature: Scaling factor (lower = sharper distribution)
    
    Returns:
        loss: Scalar contrastive loss
    """
    # Normalize on unit sphere
    z_i = F.normalize(z_i, dim=1)
    z_j = F.normalize(z_j, dim=1)
    
    # Cosine similarity matrix
    logits = torch.matmul(z_i, z_j.T) / temperature  # (B, B)
    
    # Labels are identity: sample i should match with sample i
    labels = torch.arange(z_i.size(0), device=z_i.device)
    
    # Cross entropy
    return F.cross_entropy(logits, labels)

print("✅ NT-Xent loss function defined")

# Test loss
z_test = torch.randn(8, 128)
z_aug_test = torch.randn(8, 128)
loss_test = nt_xent_loss(z_test, z_aug_test)
print(f"   Test loss: {loss_test.item():.4f} ✅")

print(f"\n{'='*70}\nSTAGE 7: Prepare Data for Training\n{'='*70}")

augmentor = AntiShortcutAugmentation()
pretrain_dataset = ContrastiveDataset(pretrain_data, augmentor)

BATCH_SIZE = 64
pretrain_loader = DataLoader(
    pretrain_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

print(f"✅ DataLoader ready")
print(f"   Total flows: {len(pretrain_dataset):,}")
print(f"   Batch size: {BATCH_SIZE}")
print(f"   Batches per epoch: {len(pretrain_loader):,}")

print(f"\n{'='*70}\nSTAGE 8: SSL Pretraining with Checkpointing\n{'='*70}")

SSL_EPOCHS = 3
LR = 5e-4
TEMPERATURE = 0.5

# Initialize logging
log_file = os.path.join(LOGS_DIR, "ssl_pretraining.log")
with open(log_file, 'w') as f:
    f.write(f"SSL Pretraining Log\n")
    f.write(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Config: Epochs={SSL_EPOCHS}, LR={LR}, Batch={BATCH_SIZE}, Temp={TEMPERATURE}\n")
    f.write(f"Data: {len(pretrain_data):,} flows\n")
    f.write("="*70 + "\n\n")

print(f"✅ Config:")
print(f"   Epochs: {SSL_EPOCHS}")
print(f"   Learning Rate: {LR}")
print(f"   Temperature: {TEMPERATURE}")
print(f"   Batch Size: {BATCH_SIZE}")
print(f"   Log file: {log_file}")

# Check for existing weights (checkpoint logic)
final_weight_path = os.path.join(WEIGHTS_DIR, "bimamba_masking_ssl_final.pth")
last_epoch_file = os.path.join(LOGS_DIR, "last_epoch.txt")

encoder = BiMambaEncoder(d_model=256, n_layers=4).to(DEVICE)
optimizer = torch.optim.AdamW(encoder.parameters(), lr=LR)

start_epoch = 0

if os.path.exists(final_weight_path):
    print(f"\n✅ Final model exists: {final_weight_path}")
    encoder.load_state_dict(torch.load(final_weight_path, map_location=DEVICE, weights_only=False))
    print("   Skipping training (already complete)")
    start_epoch = SSL_EPOCHS
elif os.path.exists(last_epoch_file):
    with open(last_epoch_file, 'r') as f:
        start_epoch = int(f.read().strip()) + 1
    checkpoint_path = os.path.join(WEIGHTS_DIR, f"bimamba_masking_ssl_epoch_{start_epoch}.pth")
    print(f"\n🔄 Resuming from epoch {start_epoch}...")
    encoder.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE, weights_only=False))
else:
    print(f"\n🆕 Starting fresh training...")

print(f"   Starting from epoch {start_epoch}")

# Training loop
if start_epoch < SSL_EPOCHS:
    print(f"\n🔄 Training epochs {start_epoch+1}-{SSL_EPOCHS}\n")
    
    for epoch in range(start_epoch, SSL_EPOCHS):
        encoder.train()
        total_loss = 0
        steps = 0
        start_time = time.time()
        
        print(f"\n{'='*70}")
        print(f"Epoch {epoch+1}/{SSL_EPOCHS}")
        print(f"{'='*70}")
        
        for step, (x, x_aug) in enumerate(pretrain_loader):
            x = x.to(DEVICE)
            x_aug = x_aug.to(DEVICE)
            
            # Forward pass (both augmented views)
            optimizer.zero_grad()
            z_i, _ = encoder(x)
            z_j, _ = encoder(x_aug)
            
            # Contrastive loss
            loss = nt_xent_loss(z_i, z_j, temperature=TEMPERATURE)
            
            # Backward
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            steps += 1
            
            # Log progress
            if (step + 1) % 500 == 0:
                elapsed = time.time() - start_time
                eta = (elapsed / (step + 1)) * (len(pretrain_loader) - step - 1)
                msg = f"  E{epoch+1} S{step+1:>5d}/{len(pretrain_loader)} | Loss:{loss.item():.4f} | ETA:{eta/60:.1f}m"
                print(msg)
                with open(log_file, 'a') as f:
                    f.write(msg + "\n")
        
        # Epoch summary
        avg_loss = total_loss / steps
        elapsed = time.time() - start_time
        msg = f"✅ Epoch {epoch+1} complete: avg_loss={avg_loss:.4f} ({elapsed/60:.1f}m)"
        print(f"\n{msg}")
        with open(log_file, 'a') as f:
            f.write(msg + "\n")
        
        # Save checkpoint
        ckpt_path = os.path.join(WEIGHTS_DIR, f"bimamba_masking_ssl_epoch_{epoch+1}.pth")
        torch.save(encoder.state_dict(), ckpt_path)
        print(f"   💾 Saved: {ckpt_path}")
        
        # Update last epoch
        with open(last_epoch_file, 'w') as f:
            f.write(str(epoch))
    
    # Save final
    torch.save(encoder.state_dict(), final_weight_path)
    print(f"\n✅ Training complete!")
    print(f"   💾 Final: {final_weight_path}")
    
    # Cleanup
    if os.path.exists(last_epoch_file):
        os.remove(last_epoch_file)
else:
    print(f"\n✅ Training already complete (all {SSL_EPOCHS} epochs done)")

# Cleanup GPU
encoder.cpu()
torch.cuda.empty_cache()

print(f"\n{'='*70}\nSTAGE 9: Verification & Testing\n{'='*70}")

print(f"\n📂 Loading final model...")
encoder = BiMambaEncoder(d_model=256, n_layers=4).to(DEVICE)
encoder.load_state_dict(torch.load(final_weight_path, map_location=DEVICE, weights_only=False))
encoder.eval()
print("✅ Model loaded")

print(f"\n🧪 Testing on sample batch...")
with torch.no_grad():
    x_test, x_aug_test = next(iter(pretrain_loader))
    x_test = x_test.to(DEVICE)
    z, recon = encoder(x_test)
    print(f"   Input: {x_test.shape}")
    print(f"   Contrastive output: {z.shape} ✅")
    print(f"   Reconstruction output: {recon.shape} ✅")

# Compute unsupervised AUC
print(f"\n📊 Computing unsupervised AUC...")
with open(FINETUNE_PKL, 'rb') as f:
    finetune_sample = pickle.load(f)[:5000]

# Split benign vs attack
benign_test = [d for d in finetune_sample if d['label'] == 0][:2500]
attack_test = [d for d in finetune_sample if d['label'] == 1][:2500]
test_flows = benign_test + attack_test
test_labels = [0]*len(benign_test) + [1]*len(attack_test)
benign_ref = pretrain_data[:2000]

with torch.no_grad():
    # Encode reference (benign)
    ref_reps = []
    for i in range(0, len(benign_ref), 256):
        batch = benign_ref[i:i+256]
        feats = np.stack([d['features'] if isinstance(d, dict) else d for d in batch])
        x = torch.from_numpy(feats).float().to(DEVICE)
        z, _ = encoder(x)
        ref_reps.append(z.cpu().numpy())
    ref_reps = np.concatenate(ref_reps, axis=0)
    ref_reps = ref_reps / (np.linalg.norm(ref_reps, axis=1, keepdims=True) + 1e-8)
    
    # Encode test
    test_reps = []
    for i in range(0, len(test_flows), 256):
        batch = test_flows[i:i+256]
        feats = np.stack([d['features'] if isinstance(d, dict) else d for d in batch])
        x = torch.from_numpy(feats).float().to(DEVICE)
        z, _ = encoder(x)
        test_reps.append(z.cpu().numpy())
    test_reps = np.concatenate(test_reps, axis=0)
    test_reps = test_reps / (np.linalg.norm(test_reps, axis=1, keepdims=True) + 1e-8)
    
    # Compute anomaly scores (distance to benign reference)
    scores = []
    for i in range(0, len(test_reps), 200):
        chunk = test_reps[i:i+200]
        sim = chunk @ ref_reps.T  # Cosine similarity
        topk = np.sort(sim, axis=1)[:, -10:]  # Top 10 similarity
        scores.extend(topk.mean(axis=1).tolist())  # Average
    
    auc = roc_auc_score(test_labels, scores)
    print(f"   Unsupervised AUC: {auc:.4f} ✅")

# Cleanup
encoder.cpu()
torch.cuda.empty_cache()

print(f"\n{'='*70}")
print("PIPELINE COMPLETION SUMMARY")
print(f"{'='*70}")

print(f"""
✅ Complete SSL Pretraining Pipeline Done!

📁 Outputs Saved:
   Weights:  {final_weight_path}
   Logs:     {log_file}

🎯 Ready for:
   1. Supervised fine-tuning
   2. Student distillation
   3. Cross-dataset evaluation

⏱️ Performance:
   Unsupervised AUC: {auc:.4f}
   Model size: ~3.65M parameters
   
✨ Checkpoint Features:
   ✅ Skips re-processing if outputs exist
   ✅ Resumes from last epoch if interrupted
   ✅ Full logging included
   ✅ Absolute paths (no relative path issues)
   ✅ Centralized weight storage (ssl/ folder)
""")

print("="*70)
print("✅ FULL PIPELINE COMPLETE!")
print("="*70)

