import torch
import time
import os
import sys
import numpy as np
from torch.utils.data import DataLoader
import json

# Import definitions from run_full_evaluation
sys.path.append(os.path.join(os.path.dirname(__file__), '..')) # Add thesis_final to path?
# Actually, run_full_evaluation is in thesis_final/code.
# I will copy the minimal imports and definitions or try to import from run_full_evaluation if possible.
# But run_full_evaluation is a script.

# Better: Just define the models again or use the classes if they were in a module. 
# They are inline. I'll copy the classes.

import torch.nn as nn
from mamba_ssm import Mamba

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ----------------- COPIED ARCHITECTURES -----------------
class PacketEmbedder(nn.Module):
    def __init__(self, d_model=256):
        super().__init__()
        self.emb_proto = nn.Embedding(256, 32)
        self.emb_flags = nn.Embedding(64, 32)
        self.emb_dir   = nn.Embedding(2, 8)
        self.proj_len  = nn.Linear(1, 32)
        self.proj_iat  = nn.Linear(1, 32) # Normalized IAT
        self.fusion    = nn.Linear(136, d_model)
        self.norm      = nn.LayerNorm(d_model)
    def forward(self, x): # (B, L, 5)
        # 0=proto, 1=len, 2=iat, 3=dir, 4=flags
        proto  = x[:,:,0].long().clamp(0, 255)
        length = x[:,:,1:2]
        iat    = x[:,:,2:3]
        dirc   = x[:,:,3].long().clamp(0, 1)
        flags  = x[:,:,4].long().clamp(0, 63)
        
        e_p = self.emb_proto(proto)
        e_f = self.emb_flags(flags)
        e_d = self.emb_dir(dirc)
        e_l = self.proj_len(length)
        e_i = self.proj_iat(iat)
        
        cat = torch.cat([e_p, e_f, e_d, e_l, e_i], dim=-1) # 32+32+8+32+32 = 136
        return self.norm(self.fusion(cat))

class BiMambaEncoder(nn.Module):
    def __init__(self, d_model=256, n_layers=4):
        super().__init__()
        self.tokenizer  = PacketEmbedder(d_model)
        self.layers     = nn.ModuleList([
            Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
            for _ in range(n_layers)])
        self.layers_rev = nn.ModuleList([
            Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
            for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.proj_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(),
            nn.Linear(d_model, 256))
        self.recon_head = nn.Linear(d_model, 5)
    def forward(self, x):
        feat = self.tokenizer(x)
        for fwd, bwd in zip(self.layers, self.layers_rev):
            out_f = fwd(feat)
            feat_rev = torch.flip(feat, dims=[1])
            out_b = bwd(feat_rev)
            out_b = torch.flip(out_b, dims=[1])
            feat  = self.norm(out_f + out_b + feat)
        z = self.proj_head(feat.mean(dim=1))
        return z, None

class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.pe_emb = nn.Embedding(max_len, d_model)
    def forward(self, x):
        B, L, D = x.shape
        pos = torch.arange(L, device=x.device).unsqueeze(0).expand(B, L)
        return x + self.pe_emb(pos)

class BertEncoder(nn.Module):
    def __init__(self, d_model=256, n_layers=4, n_heads=8):
        super().__init__()
        self.tokenizer = PacketEmbedder(d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model)) # Added CLS token
        self.pos_enc = LearnablePositionalEncoding(d_model)
        
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, 
                    dim_feedforward=d_model*4, batch_first=True, dropout=0.1)
        self.transformer_encoder = nn.TransformerEncoder(layer, n_layers)
        self.norm = nn.LayerNorm(d_model)
        # Output 128 dim to match baseline
        self.proj_head = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(),
            nn.Linear(d_model, 128))
    def forward(self, x):
        B, L, _ = x.shape
        # 1. Embed
        e = self.tokenizer(x) # (B, L, D)
        # 2. Add CLS Token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        e = torch.cat((cls_tokens, e), dim=1) # (B, L+1, D)
        # 3. Pos Enc
        e = self.pos_enc(e)
        # 4. Transformer
        mask = torch.zeros((B, L+1), dtype=torch.bool, device=x.device) # No masking for now (full attention)
        f = self.transformer_encoder(e, src_key_padding_mask=mask)
        f = self.norm(f)
        # 5. Extract CLS
        cls_out = f[:, 0, :]
        return self.proj_head(cls_out), None

class UniMambaEncoder(nn.Module):
    def __init__(self, d_model=256, n_layers=4):
        super().__init__()
        self.tokenizer = PacketEmbedder(d_model)
        self.layers = nn.ModuleList([
            Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
            for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.proj_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(),
            nn.Linear(d_model, 128))
    def forward(self, x):
        feat = self.tokenizer(x)
        for layer in self.layers:
            feat = layer(feat) + feat
        feat = self.norm(feat)
        return self.proj_head(feat.mean(dim=1)), None # UniMamba uses Mean (or Last)

class Classifier(nn.Module):
    def __init__(self, encoder, input_dim):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 2))
    def forward(self, x):
        z, _ = self.encoder(x)
        return self.head(z)

# ----------------- BENCHMARK -----------------

def measure_throughput(model, batch_size=1):
    model.eval()
    dummy_input = torch.randn(batch_size, 32, 5).to(DEVICE) # (B, 32, 5) features
    
    # Warmup
    for _ in range(10):
        with torch.no_grad(): model(dummy_input)
    if torch.cuda.is_available(): torch.cuda.synchronize()
    
    # Measure
    t0 = time.perf_counter()
    N = 1000
    if batch_size == 1: N = 2000 # More samples for stability
    
    with torch.no_grad():
        for _ in range(N):
            out = model(dummy_input)
    
    if torch.cuda.is_available(): torch.cuda.synchronize()
    t1 = time.perf_counter()
    
    total_samples = N * batch_size
    duration = t1 - t0
    thr = total_samples / duration
    lat = (duration / N) * 1000 # ms per batch (which is ms per flow if B=1)
    
    return thr, lat

def main():
    print(f"Benchmarking Real-Time Throughput (Batch Size = 1)")
    print("==================================================")
    
    models = {
        "Teacher (BiMamba)": Classifier(BiMambaEncoder(), input_dim=256).to(DEVICE),
        "Standard BERT":     Classifier(BertEncoder(),    input_dim=128).to(DEVICE),
        "UniMamba (Ours)":   Classifier(UniMambaEncoder(),input_dim=128).to(DEVICE)
    }
    
    # We don't need to load weights for throughput test, just architecture.
    # Architecture determines speed.
    
    print(f"{'Model':<20} | {'Batch':<5} | {'Throughput (flow/s)':<20} | {'Latency (ms)':<15}")
    print("-" * 70)
    
    for name, model in models.items():
        thr, lat = measure_throughput(model, batch_size=1)
        print(f"{name:<20} | {1:<5} | {thr:<20.0f} | {lat:<15.4f}")
        
    print("\nBenchmarking 'NIDS Buffer' Throughput (Batch Size = 32)")
    print("=======================================================")
    print(f"{'Model':<20} | {'Batch':<5} | {'Throughput (flow/s)':<20} | {'Latency (ms)':<15}")
    print("-" * 70)
    
    for name, model in models.items():
        thr, lat = measure_throughput(model, batch_size=32)
        print(f"{name:<20} | {32:<5} | {thr:<20.0f} | {lat:<15.4f}")

if __name__ == "__main__":
    main()
