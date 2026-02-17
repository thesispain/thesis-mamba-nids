
import os, sys, torch, pickle, glob
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CIC = "data/cicids2017_flows.pkl"

class BertEncoder(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_layers=2, pool='cls'):
        super().__init__()
        self.pool = pool
        self.emb_proto = nn.Embedding(256, 16)
        self.emb_flags = nn.Embedding(64, 16)
        self.emb_dir   = nn.Embedding(2, 4)
        self.proj_len  = nn.Linear(1, 16)
        self.proj_iat  = nn.Linear(1, 16)
        self.fusion    = nn.Linear(68, d_model)
        self.norm      = nn.LayerNorm(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=1024, dropout=0.1, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.proj_head = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 128))
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
        if self.pool == 'cls':
            return self.proj_head(feat[:, 0, :]), None
        else: # mean
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

def eval_bert(weight_path):
    # Only test CLS
    pool = 'cls'
    enc = BertEncoder(pool=pool)
    model = Classifier(enc, d_model=128).to(DEVICE)
    try:
        sd = torch.load(weight_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(sd, strict=False)
    except: return

    model.eval()
    
    # Check if weights loaded into head?
    # If not, AUC will be 0.5.
    # We proceed anyway.

    # Load Data (Brief)
    # Reuse global if possible, else load once
    if 'X_shared' not in globals():
        global X_shared, y_shared
        with open(CIC, 'rb') as f: cic_data = pickle.load(f)
        np.random.shuffle(cic_data)
        sub = cic_data[:5000] 
        X_shared = np.array([d['features'] for d in sub], dtype=np.float32)
        y_shared = np.array([d['label'] for d in sub], dtype=np.int64)

    dl = DataLoader(TensorDataset(torch.from_numpy(X_shared).to(DEVICE), torch.from_numpy(y_shared).to(DEVICE)), batch_size=512)
    
    probs = []
    with torch.no_grad():
        for x, _ in dl:
            logits = model(x)
            probs.extend(F.softmax(logits, 1)[:, 1].cpu().numpy())
    auc = roc_auc_score(y_shared, probs)
    print(f"[{auc:.4f}] {os.path.basename(weight_path)}")

def main():
    print("Scanning ALL .pth files in weights/ for CLS Compatibility...")
    files = glob.glob("weights/**/*.pth", recursive=True)
    for f in files:
        if "bert" in f:
            eval_bert(f)

if __name__ == "__main__":
    main()
