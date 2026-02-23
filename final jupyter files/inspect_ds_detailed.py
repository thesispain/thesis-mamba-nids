import pickle
from pathlib import Path
import gc
import numpy as np

try:
    print("Inspecting detailed feature distribution of UNSW-NB15.")
    UNSW_PATH = '/home/T2510596/Downloads/totally fresh/Organized_Final/data/unswnb15_full/pretrain_50pct_benign.pkl'
    with open(UNSW_PATH, 'rb') as f:
        unsw = pickle.load(f)
        
    sample_indices_unsw = np.random.choice(len(unsw), min(100000, len(unsw)), replace=False)
    sub_unsw = [unsw[i] for i in sample_indices_unsw]
    del unsw
    gc.collect()
    
    # We might have 'input_ids' or 'packet_features' or 'features'
    feat_key = 'features'
    if 'input_ids' in sub_unsw[0]:
        feat_key = 'input_ids'
        
    feats_unsw = np.array([x[feat_key] for x in sub_unsw])
    if len(feats_unsw.shape) == 3:
        flat_feats_unsw = feats_unsw.reshape(-1, feats_unsw.shape[2])
    else:
        flat_feats_unsw = feats_unsw
        
    means_unsw = np.mean(flat_feats_unsw, axis=0)
    stds_unsw = np.std(flat_feats_unsw, axis=0)
    mins_unsw = np.min(flat_feats_unsw, axis=0)
    maxs_unsw = np.max(flat_feats_unsw, axis=0)
    
    print("\n--- UNSW-NB15 100k Sample Feature Distribution ---")
    print(f"{'Feat':<5} | {'Mean':>12} | {'Std':>12} | {'Min':>12} | {'Max':>12}")
    print("-" * 65)
    for i in range(len(means_unsw)):
        print(f"{i:<5} | {means_unsw[i]:12.4f} | {stds_unsw[i]:12.4f} | {mins_unsw[i]:12.4f} | {maxs_unsw[i]:12.4f}")

except Exception as e:
    print("Error:", e)
