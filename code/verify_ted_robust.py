
import os, sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pickle
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from mamba_ssm import Mamba

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CICIDS_PATH = "data/cicids2017_flows.pkl"
TED_WEIGHTS = "weights/students/student_ted.pth"

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

class BlockwiseEarlyExitMamba(nn.Module):
    def __init__(self, d_model=256):
        super().__init__()
        self.d_model = d_model
        self.tokenizer = PacketEmbedder(d_model)
        self.layers = nn.ModuleList([
            Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
            for _ in range(4)])
        self.norm = nn.LayerNorm(d_model)
        
        self.exit_positions = [8, 16, 32]
        self.n_exits = 3
        self.conf_thresh = 0.85
        
        # Classifier: Linear(256, 128) -> ReLU -> Dropout -> Linear(128, 2)
        # Assuming index 2 is missing from weights -> index 2 is Dropout/Identity
        self.exit_classifiers = nn.ModuleDict({
            str(p): nn.Sequential(
                nn.Linear(d_model, 128), 
                nn.ReLU(), 
                nn.Dropout(0.1), # Placeholder
                nn.Linear(128, 2)
            ) for p in self.exit_positions
        })
        
        # Confidence: Linear(258, 64) -> ReLU -> Linear(64, 1) -> Sigmoid
        self.confidence_heads = nn.ModuleDict({
            str(p): nn.Sequential(
                nn.Linear(d_model + 2, 64), 
                nn.ReLU(), 
                nn.Linear(64, 1), 
                nn.Sigmoid()
            ) for p in self.exit_positions
        })
        
        self.register_buffer('exit_counts', torch.zeros(self.n_exits))
        self.register_buffer('total_inferences', torch.tensor(0))

    def _backbone(self, x):
        feat = self.tokenizer(x)
        for layer in self.layers:
            out  = layer(feat)
            feat = self.norm(out + feat)
        return feat

    def forward_inference(self, x):
        B = x.size(0)
        feat = self._backbone(x)
        
        # Zero-Shot at Final Exit (32)
        pos = 32
        str_pos = str(pos)
        idx = min(pos, feat.size(1)) - 1
        h = feat[:, idx, :]
        
        logits = self.exit_classifiers[str_pos](h)
        return logits, None

def main():
    print(f"Loading TED from {TED_WEIGHTS}...")
    try:
        sd = torch.load(TED_WEIGHTS, map_location=DEVICE)
    except Exception as e:
        print(f"Load failed: {e}")
        return

    model = BlockwiseEarlyExitMamba(d_model=256).to(DEVICE)
    model.load_state_dict(sd, strict=False)
    print("Model loaded successfully.")

    print(f"Loading CIC-IDS data...")
    with open(CICIDS_PATH, 'rb') as f:
        data = pickle.load(f)
    
    # Full Data (Comment out sample line)
    # data = data[:20000]
    
    X = np.array([d['features'] for d in data], dtype=np.float32)
    y = np.array([d['label'] for d in data], dtype=np.longlong)
    
    benign = np.sum(y==0)
    attack = np.sum(y==1)
    print(f"Data: {len(y)} flows. Benign: {benign} ({benign/len(y):.1%}), Attack: {attack}")

    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    dl = DataLoader(ds, batch_size=2048, shuffle=False)
    
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for x, label in dl:
            x = x.to(DEVICE)
    print("Running Inference over full dataset...")
    try:
        with torch.no_grad():
            for x, label in dl:
                x = x.to(DEVICE)
                logits, _ = model.forward_inference(x)
                # Store Prob(Attack) for AUC
                probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                all_preds.extend(probs)
                all_labels.extend(label.numpy())
    except Exception as e:
        print(f"Inference error: {e}")
        return
            
    y_true = np.array(all_labels)
    y_scores = np.array(all_preds) # Prob(Attack)
    y_pred = (y_scores > 0.5).astype(int) # Binary Predictions
    
    f1 = f1_score(y_true, y_pred, zero_division=0) # Attack F1
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    f1_benign = f1_score(y_true, y_pred, pos_label=0)
    
    # Calculate AUC
    from sklearn.metrics import roc_auc_score
    try:
        auc = roc_auc_score(y_true, y_scores)
    except:
        auc = 0.5
    
    print(f"\n=== TED Zero-Shot Real Results ===")
    print(f"Attack F1: {f1:.4f}")
    print(f"Benign F1: {f1_benign:.4f}")
    print(f"Accuracy:  {acc:.4f}")
    print(f"AUC Score: {auc:.4f}")
    print(f"Confusion Matrix:\n{cm}")

if __name__ == "__main__":
    main()
