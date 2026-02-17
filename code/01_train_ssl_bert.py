#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os, sys, torch, pickle, time
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")


# In[ ]:


DATA_FILE = "../Organized_Final/data/unswnb15_full/finetune_mixed.pkl"
WEIGHTS_DIR = "weights/ssl"
os.makedirs(WEIGHTS_DIR, exist_ok=True)
print(f"Data Path: {DATA_FILE}")
print(f"Weights Dir: {WEIGHTS_DIR}")


# In[ ]:


class FlowDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        # Return features as tensor
        x = torch.from_numpy(self.data[idx]['features']).float()
        y = torch.tensor(self.data[idx]['label']).long()
        return x, y

print(f"Loading {DATA_FILE}...")
with open(DATA_FILE, 'rb') as f:
    flows = pickle.load(f)
print(f"Total flows: {len(flows)}")

# Filter Benign
benign_flows = [f for f in flows if f['label'] == 0 or f['label'] == 'Benign']
print(f"Benign flows filtered: {len(benign_flows)}")

dataset = FlowDataset(benign_flows)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True, drop_last=True)
print("DataLoader ready.")


# In[ ]:


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
        # USE CLS TOKEN (First element)
        return self.proj_head(feat[:, 0, :]), None

print("BertEncoder (CLS) Defined.")


# In[ ]:


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

print("Helpers Defined.")


# In[ ]:


model = BertEncoder().to(DEVICE)
model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
loss_fn = NTXentLoss(temperature=0.5)
LAMBDA = 0.4

print("--- Starting Pre-Training ---")
t0 = time.time()
total_loss = 0

for i, (x, _) in enumerate(dataloader):
    x = x.to(DEVICE)
    z_i, _ = model(x)

    # Augmentation
    proto, length, flags, iat, direc = x[:,:,0].long(), x[:,:,1], x[:,:,2].long(), x[:,:,3], x[:,:,4].long()
    B, L, _ = x.shape
    perm = torch.randperm(B).to(DEVICE)
    start_indices = torch.randint(0, L - int(L * LAMBDA), (B,)).to(DEVICE)

    aug_proto = fast_aug(proto, perm, start_indices, LAMBDA)
    aug_len   = fast_aug(length, perm, start_indices, LAMBDA)
    aug_flags = fast_aug(flags, perm, start_indices, LAMBDA)
    aug_iat   = fast_aug(iat, perm, start_indices, LAMBDA)
    aug_dir   = fast_aug(direc, perm, start_indices, LAMBDA)
    x_aug = torch.stack([aug_proto.float(), aug_len, aug_flags.float(), aug_iat, aug_dir.float()], dim=-1)

    z_j, _ = model(x_aug)

    loss = loss_fn(z_i, z_j)
    optimizer.zero_grad(); loss.backward(); optimizer.step()

    total_loss += loss.item()
    if i % 100 == 0: print(f"Step {i}/{len(dataloader)} Loss: {loss.item():.4f}")

print(f"Training Complete. Avg Loss: {total_loss/len(dataloader):.4f}")

# Save Weights
out_path = os.path.join(WEIGHTS_DIR, "bert_standard_ssl_optimized.pth")
torch.save(model.state_dict(), out_path)
print(f"Weights saved to {out_path}")

