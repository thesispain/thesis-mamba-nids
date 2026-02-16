
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pickle
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from mamba_ssm import Mamba

# ── Config ──
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CICIDS_PATH = "data/cicids2017_flows.pkl"
TED_WEIGHTS = "weights/students/student_ted.pth"

# ── Model Definitions (Aligned with run_full_evaluation.py) ──
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
    def __init__(self, d_model=256, exit_positions=None, conf_thresh=0.85):
        super().__init__()
        if exit_positions is None:
            exit_positions = [8, 16, 32]
        self.exit_positions = exit_positions
        self.n_exits = len(exit_positions)
        self.conf_thresh = conf_thresh
        self.d_model = d_model

        self.tokenizer = PacketEmbedder(d_model)
        self.layers = nn.ModuleList([
            Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
            for _ in range(4)])
        self.norm = nn.LayerNorm(d_model)

        self.exit_classifiers = nn.ModuleList([
            nn.Sequential(nn.Linear(d_model, 64), nn.ReLU(), nn.Linear(64, 2))
            for _ in exit_positions])
        self.confidence_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(d_model, 64), nn.ReLU(),
                          nn.Linear(64, 1), nn.Sigmoid())
            for _ in exit_positions])
        self.register_buffer('exit_counts',
                             torch.zeros(self.n_exits, dtype=torch.long))
        self.register_buffer('total_inferences', torch.tensor(0, dtype=torch.long))

    def _backbone(self, x):
        feat = self.tokenizer(x)
        for layer in self.layers:
            out  = layer(feat)
            feat = self.norm(out + feat)
        return feat

    def forward_inference(self, x):
        B = x.size(0)
        feat = self._backbone(x)
        final_logits = torch.zeros(B, 2, device=x.device)
        exit_indices = torch.full((B,), self.n_exits-1, device=x.device, dtype=torch.long)
        active = torch.ones(B, dtype=torch.bool, device=x.device)

        for i, pos in enumerate(self.exit_positions[:-1]):
            if not active.any(): break
            idx = min(pos, feat.size(1)) - 1
            h = feat[active, idx, :]
            logits = self.exit_classifiers[i](h)
            conf   = self.confidence_heads[i](h).squeeze(-1)
            should_exit = conf >= self.conf_thresh
            if should_exit.any():
                active_idx = active.nonzero(as_tuple=True)[0]
                exiting    = active_idx[should_exit]
                final_logits[exiting] = logits[should_exit]
                exit_indices[exiting] = i
                active[exiting]       = False
                if not self.training:
                    self.exit_counts[i] += should_exit.sum().item()

        if active.any():
            idx = min(self.exit_positions[-1], feat.size(1)) - 1
            h = feat[active, idx, :]
            final_logits[active] = self.exit_classifiers[-1](h)
        
        if not self.training:
            self.total_inferences += B
        return final_logits, exit_indices

def main():
    print(f"Loading CIC-IDS from {CICIDS_PATH}...")
    with open(CICIDS_PATH, 'rb') as f:
        data = pickle.load(f)
    
    # Take a sample if too large for quick verification
    # data = data[:50000] # Optional sample
    
    X = np.array([d['features'] for d in data], dtype=np.float32)
    y = np.array([d['label'] for d in data], dtype=np.longlong) # Use long for label
    
    # Class stats
    benign = np.sum(y==0)
    attack = np.sum(y==1)
    total = len(y)
    print(f"Total: {total}, Benign: {benign} ({benign/total:.1%}), Attack: {attack}")
    
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    dl = DataLoader(ds, batch_size=1024, shuffle=False)
    
    print("Loading TED Model (d_model=128)...") 
    # WAIT! run_full_evaluation.py uses d_model=128 for students!
    # "uni = Classifier(UniMambaEncoder(), input_dim=128)" 
    # But wait, UniMamba definition: 
    # "class UniMambaEncoder(nn.Module): def __init__(self, d_model=256, n_layers=4):" 
    # Default is 256. But run_full_evaluation.py says:
    # "uni = Classifier(UniMambaEncoder(), input_dim=128)" <- Wait, input_dim for classifier head is 128?
    # No, UniMamba projection head outputs 128.
    # Ah, let's check Blockwise definition usage.
    # "ted = BlockwiseEarlyExitMamba().to(DEVICE)" -> uses defaults (d_model=256)?
    
    # Let's checked saved weights size to be sure.
    # "model.tokenizer.fusion.weight" shape is (256, 136) -> d_model=256.
    # So d_model=256 is correct.
    
    model = BlockwiseEarlyExitMamba(d_model=256).to(DEVICE)
    try:
        sd = torch.load(TED_WEIGHTS, map_location=DEVICE)
        model.load_state_dict(sd)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Model load failed: {e}")
        # Try d_model=128 just in case
        try:
            print("Retrying with d_model=128...")
            model = BlockwiseEarlyExitMamba(d_model=128).to(DEVICE)
            model.load_state_dict(sd)
            print("Model loaded with d_model=128!")
        except Exception as e2:
            print(f"Model load failed again: {e2}")
            return

    model.eval()
    all_preds = []
    all_labels = []
    
    print("Running Inference...")
    with torch.no_grad():
        for feat, label in dl:
            feat = feat.to(DEVICE)
            logits, _ = model.forward_inference(feat)
            preds = logits.argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(label.numpy())
            
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    
    f1 = f1_score(y_true, y_pred, zero_division=0) # Attack F1
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    rec = cm[1,1] / (cm[1,1] + cm[1,0]) if (cm[1,1]+cm[1,0]) > 0 else 0
    prec = cm[1,1] / (cm[1,1] + cm[0,1]) if (cm[1,1]+cm[0,1]) > 0 else 0
    
    print(f"\n=== TED Zero-Shot (Original Weights) ===")
    print(f"F1 Score: {f1:.4f}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Recall:   {rec:.4f}")
    print(f"Precision:{prec:.4f}")
    print(f"Confusion Matrix:\n{cm}")
    
if __name__ == "__main__":
    main()
