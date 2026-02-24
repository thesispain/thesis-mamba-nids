#!/usr/bin/env python3
"""
Predictive Self-Distillation for Early Exit (UniMamba)
Step 1: Train/Load SSL UniMamba (No projection head used for inference).
Step 2: Calculate Benign/Attack Centroids in pure representations space.
Step 3: Train Early Exit layers using MSE (Predictive Self-Distillation).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle, os, time, sys
from pathlib import Path
from mamba_ssm import Mamba
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

# ==========================================
# CONFIGURATION
# ==========================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

ROOT = Path('/home/T2510596/Downloads/totally fresh')
UNSW_DIR = ROOT / 'Organized_Final' / 'data' / 'unswnb15_full'
WEIGHT_DIR = Path('weights/predictive_ted')
WEIGHT_DIR.mkdir(parents=True, exist_ok=True)

# Define dataset loading... (to be added)
