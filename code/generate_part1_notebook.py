
import json
import os

NOTEBOOK_PATH = "Part1_SSL_Pretraining.ipynb"

cells = []

# Cell 1: Imports
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "import os, sys, torch, pickle, time\n",
        "import numpy as np\n",
        "import torch.nn as nn\n",
        "from torch.utils.data import DataLoader, Dataset\n",
        "\n",
        "DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
        "print(f\"Device: {DEVICE}\")"
    ]
})

# Cell 2: Config & Constants
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "DATA_FILE = \"../Organized_Final/data/unswnb15_full/finetune_mixed.pkl\"\n",
        "WEIGHTS_DIR = \"weights/ssl\"\n",
        "os.makedirs(WEIGHTS_DIR, exist_ok=True)\n",
        "print(f\"Data Path: {DATA_FILE}\")\n",
        "print(f\"Weights Dir: {WEIGHTS_DIR}\")"
    ]
})

# Cell 3: Data Loading (Benign Only)
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "class FlowDataset(Dataset):\n",
        "    def __init__(self, data):\n",
        "        self.data = data\n",
        "    def __len__(self):\n",
        "        return len(self.data)\n",
        "    def __getitem__(self, idx):\n",
        "        # Return features as tensor\n",
        "        x = torch.from_numpy(self.data[idx]['features']).float()\n",
        "        y = torch.tensor(self.data[idx]['label']).long()\n",
        "        return x, y\n",
        "\n",
        "print(f\"Loading {DATA_FILE}...\")\n",
        "with open(DATA_FILE, 'rb') as f:\n",
        "    flows = pickle.load(f)\n",
        "print(f\"Total flows: {len(flows)}\")\n",
        "\n",
        "# Filter Benign\n",
        "benign_flows = [f for f in flows if f['label'] == 0 or f['label'] == 'Benign']\n",
        "print(f\"Benign flows filtered: {len(benign_flows)}\")\n",
        "\n",
        "dataset = FlowDataset(benign_flows)\n",
        "dataloader = DataLoader(dataset, batch_size=128, shuffle=True, drop_last=True)\n",
        "print(\"DataLoader ready.\")"
    ]
})

# Cell 4: Model Definition (BERT with CLS Token)
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "class PacketEmbedder(nn.Module):\n",
        "    def __init__(self, d_model=256):\n",
        "        super().__init__()\n",
        "        self.emb_proto = nn.Embedding(256, 32)\n",
        "        self.emb_flags = nn.Embedding(64, 32)\n",
        "        self.emb_dir   = nn.Embedding(2, 8)\n",
        "        self.proj_len  = nn.Linear(1, 32)\n",
        "        self.proj_iat  = nn.Linear(1, 32)\n",
        "        self.fusion    = nn.Linear(136, d_model)\n",
        "        self.norm      = nn.LayerNorm(d_model)\n",
        "    def forward(self, x):\n",
        "        proto  = x[:,:,0].long().clamp(0, 255)\n",
        "        length = x[:,:,1:2]\n",
        "        flags  = x[:,:,2].long().clamp(0, 63)\n",
        "        iat    = x[:,:,3:4]\n",
        "        direc  = x[:,:,4].long().clamp(0, 1)\n",
        "        cat = torch.cat([self.emb_proto(proto), self.proj_len(length),\n",
        "                         self.emb_flags(flags), self.proj_iat(iat),\n",
        "                         self.emb_dir(direc)], dim=-1)\n",
        "        return self.norm(self.fusion(cat))\n",
        "\n",
        "class BertEncoder(nn.Module):\n",
        "    def __init__(self, d_model=256, nhead=8, num_layers=2):\n",
        "        super().__init__()\n",
        "        self.emb_proto = nn.Embedding(256, 16)\n",
        "        self.emb_flags = nn.Embedding(64, 16)\n",
        "        self.emb_dir   = nn.Embedding(2, 4)\n",
        "        self.proj_len  = nn.Linear(1, 16)\n",
        "        self.proj_iat  = nn.Linear(1, 16)\n",
        "        self.fusion    = nn.Linear(68, d_model)\n",
        "        self.norm      = nn.LayerNorm(d_model)\n",
        "        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=1024, dropout=0.1, batch_first=True)\n",
        "        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)\n",
        "        self.proj_head = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 128)) # 128 head\n",
        "    def forward(self, x):\n",
        "        proto  = x[:,:,0].long().clamp(0, 255)\n",
        "        length = x[:,:,1:2]\n",
        "        flags  = x[:,:,2].long().clamp(0, 63)\n",
        "        iat    = x[:,:,3:4]\n",
        "        direc  = x[:,:,4].long().clamp(0, 1)\n",
        "        cat = torch.cat([self.emb_proto(proto), self.proj_len(length),\n",
        "                         self.emb_flags(flags), self.proj_iat(iat),\n",
        "                         self.emb_dir(direc)], dim=-1)\n",
        "        feat = self.norm(self.fusion(cat))\n",
        "        feat = self.transformer_encoder(feat)\n",
        "        # USE CLS TOKEN (First element)\n",
        "        return self.proj_head(feat[:, 0, :]), None\n",
        "\n",
        "print(\"BertEncoder (CLS) Defined.\")"
    ]
})

# Cell 5: Loss and Augmentation
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "class NTXentLoss(nn.Module):\n",
        "    def __init__(self, temperature=0.5):\n",
        "        super().__init__()\n",
        "        self.temperature = temperature\n",
        "        self.criterion = nn.CrossEntropyLoss()\n",
        "    def forward(self, z_i, z_j):\n",
        "        batch_size = z_i.shape[0]\n",
        "        z = torch.cat([z_i, z_j], dim=0)\n",
        "        z_norm = torch.nn.functional.normalize(z, dim=1)\n",
        "        sim_matrix = torch.mm(z_norm, z_norm.t()) / self.temperature\n",
        "        mask = torch.eye(2 * batch_size, dtype=torch.bool).to(z_i.device)\n",
        "        sim_matrix.masked_fill_(mask, -9e15)\n",
        "        labels = torch.cat([torch.arange(batch_size, 2*batch_size), torch.arange(0, batch_size)]).to(z_i.device)\n",
        "        return self.criterion(sim_matrix, labels)\n",
        "\n",
        "def fast_aug(tensor, perm, start_indices, lam=0.4):\n",
        "    aug = tensor.clone()\n",
        "    source = tensor[perm]\n",
        "    batch_size, seq_len = tensor.shape\n",
        "    patch_len = int(seq_len * lam)\n",
        "    for b in range(batch_size):\n",
        "        s = start_indices[b]\n",
        "        aug[b, s : s+patch_len] = source[b, s : s+patch_len] \n",
        "    return aug\n",
        "\n",
        "print(\"Helpers Defined.\")"
    ]
})

# Cell 6: Training Loop
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "model = BertEncoder().to(DEVICE)\n",
        "model.train()\n",
        "optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)\n",
        "loss_fn = NTXentLoss(temperature=0.5)\n",
        "LAMBDA = 0.4\n",
        "\n",
        "print(\"--- Starting Pre-Training ---\")\n",
        "t0 = time.time()\n",
        "total_loss = 0\n",
        "\n",
        "for i, (x, _) in enumerate(dataloader):\n",
        "    x = x.to(DEVICE)\n",
        "    z_i, _ = model(x)\n",
        "    \n",
        "    # Augmentation\n",
        "    proto, length, flags, iat, direc = x[:,:,0].long(), x[:,:,1], x[:,:,2].long(), x[:,:,3], x[:,:,4].long()\n",
        "    B, L, _ = x.shape\n",
        "    perm = torch.randperm(B).to(DEVICE)\n",
        "    start_indices = torch.randint(0, L - int(L * LAMBDA), (B,)).to(DEVICE)\n",
        "    \n",
        "    aug_proto = fast_aug(proto, perm, start_indices, LAMBDA)\n",
        "    aug_len   = fast_aug(length, perm, start_indices, LAMBDA)\n",
        "    aug_flags = fast_aug(flags, perm, start_indices, LAMBDA)\n",
        "    aug_iat   = fast_aug(iat, perm, start_indices, LAMBDA)\n",
        "    aug_dir   = fast_aug(direc, perm, start_indices, LAMBDA)\n",
        "    x_aug = torch.stack([aug_proto.float(), aug_len, aug_flags.float(), aug_iat, aug_dir.float()], dim=-1)\n",
        "    \n",
        "    z_j, _ = model(x_aug)\n",
        "    \n",
        "    loss = loss_fn(z_i, z_j)\n",
        "    optimizer.zero_grad(); loss.backward(); optimizer.step()\n",
        "    \n",
        "    total_loss += loss.item()\n",
        "    if i % 100 == 0: print(f\"Step {i}/{len(dataloader)} Loss: {loss.item():.4f}\")\n",
        "\n",
        "print(f\"Training Complete. Avg Loss: {total_loss/len(dataloader):.4f}\")\n",
        "\n",
        "# Save Weights\n",
        "out_path = os.path.join(WEIGHTS_DIR, \"bert_standard_ssl_optimized.pth\")\n",
        "torch.save(model.state_dict(), out_path)\n",
        "print(f\"Weights saved to {out_path}\")"
    ]
})

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.8.10"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

with open(NOTEBOOK_PATH, 'w') as f:
    json.dump(nb, f, indent=2)

print(f"Created {NOTEBOOK_PATH}")
