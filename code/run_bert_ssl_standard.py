#!/usr/bin/env python3
import os, sys, torch, pickle, time
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset

# Add path to import from run_full_evaluation
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from run_full_evaluation import BertEncoder, PacketEmbedder, FlowDataset, load_data

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_FILE = "/home/T2510596/Downloads/totally fresh/Organized_Final/data/unswnb15_full/finetune_mixed.pkl"

# --- REPLICATING NOTEBOOK LOGIC EXACTLY ---

class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, z_i, z_j):
        batch_size = z_i.shape[0]
        z = torch.cat([z_i, z_j], dim=0)
        z_norm = torch.nn.functional.normalize(z, dim=1)
        sim_matrix = torch.mm(z_norm, z_norm.t()) / self.temperature
        mask = torch.eye(2 * batch_size, dtype=torch.bool).to(z_i.device)
        sim_matrix.masked_fill_(mask, -9e15)
        labels = torch.cat([torch.arange(batch_size, 2*batch_size), torch.arange(0, batch_size)]).to(z_i.device)
        return self.criterion(sim_matrix, labels)

def fast_aug(tensor, perm, start_indices, lam=0.4):
    aug = tensor.clone()
    source = tensor[perm]
    batch_size, seq_len = tensor.shape
    patch_len = int(seq_len * lam)
    
    for b in range(batch_size):
        s = start_indices[b]
        aug[b, s : s+patch_len] = source[b, s : s+patch_len] 
        
    return aug

def main():
    print("🚀 Starting Standard BERT SSL Pre-training (Notebook Replication)")
    
    # 1. Load Data
    print(f"Loading {DATA_FILE}...")
    with open(DATA_FILE, 'rb') as f:
        flows = pickle.load(f)
    print(f"Total flows: {len(flows)}")
    
    # 2. Filter Benign Only (Paper Section V-A)
    # Label 0 is Benign (usually). Checking dataset convention.
    # In processed_flows.pkl, label is often string or int.
    # We assume 'label' key exists.
    
    benign_flows = [f for f in flows if f['label'] == 0 or f['label'] == 'Benign']
    print(f"Benign flows filtered: {len(benign_flows)}")
    
    dataset = FlowDataset(benign_flows)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True, drop_last=True)
    
    # 3. Model Setup
    # Use exact same encoder as evaluation script
    model = BertEncoder().to(DEVICE)
    model.train()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5) # Notebook LR
    loss_fn = NTXentLoss(temperature=0.5)
    
    LAMBDA = 0.4
    
    print("--- Starting 1-Epoch Contrastive Pre-Training ---")
    t0 = time.time()
    total_loss = 0
    
    for i, (x, _) in enumerate(dataloader):
        # Move to device
        x = x.to(DEVICE) # (B, L, D_feat)
        
        # We need to unpack x to apply augmentation per feature
        # FlowDataset in run_full_evaluation returns valid float tensors.
        # But fast_aug acts on indices (long) or floats?
        # PacketEmbedder expects: proto, len, flags, iat, direc.
        # x shape: (B, 32, 5)
        
        proto = x[:,:,0].long()
        length = x[:,:,1]
        flags = x[:,:,2].long()
        iat = x[:,:,3]
        direc = x[:,:,4].long()
        
        # 2. Forward Anchor (Original)
        # BertEncoder.forward takes 'x' directly.
        # But to be consistent with augmentation, we should pass augmented features?
        # BertEncoder expects (B, 32, 5).
        
        z_i, _ = model(x) # Output is (B, 128), None
        
        # 3. Create Augmentation
        B, L, _ = x.shape
        perm = torch.randperm(B).to(DEVICE)
        start_indices = torch.randint(0, L - int(L * LAMBDA), (B,)).to(DEVICE)
        
        # Augment each feature
        aug_proto = fast_aug(proto, perm, start_indices, LAMBDA)
        aug_len   = fast_aug(length, perm, start_indices, LAMBDA)
        aug_flags = fast_aug(flags, perm, start_indices, LAMBDA)
        aug_iat   = fast_aug(iat, perm, start_indices, LAMBDA)
        aug_dir   = fast_aug(direc, perm, start_indices, LAMBDA)
        
        # Stack back to (B, L, 5)
        x_aug = torch.stack([aug_proto.float(), aug_len, aug_flags.float(), aug_iat, aug_dir.float()], dim=-1)
        
        z_j, _ = model(x_aug)
        
        # 4. Loss
        loss = loss_fn(z_i, z_j)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if i % 100 == 0:
            print(f"Step {i}/{len(dataloader)} Loss: {loss.item():.4f}")
            
    print(f"Epoch 1 Complete. Avg Loss: {total_loss/len(dataloader):.4f}")
    print(f"Time Taken: {time.time() - t0:.1f}s")
    
    # Save Weights
    out_path = "/home/T2510596/Downloads/totally fresh/thesis_final/weights/ssl/bert_standard_ssl.pth"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(model.state_dict(), out_path)
    print(f"✅ Saved Standard BERT SSL weights to {out_path}")

if __name__ == "__main__":
    main()
