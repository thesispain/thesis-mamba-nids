import pickle
import numpy as np
from pathlib import Path

ROOT = Path('/home/T2510596/Downloads/totally fresh')
CIC_PATH = ROOT / 'thesis_final' / 'data' / 'cicids2017_flows.pkl'

print(f"Loading {CIC_PATH}...")
try:
    # Use mmap_mode if supported, or just read a chunk
    with open(CIC_PATH, 'rb') as f:
        cic = pickle.load(f)
    print(f"Successfully loaded. Total flows: {len(cic)}\n")
    
    first_flow = cic[0]
    print("Format of the first flow:")
    keys_to_use = None
    for k, v in first_flow.items():
        print(f"  {k}: {type(v)} - {v.shape if hasattr(v, 'shape') else v}")
        if hasattr(v, 'shape') and len(v.shape) > 0 and (keys_to_use is None or v.shape[0] > 1):
            keys_to_use = k
    print(f"\nUsing key '{keys_to_use}' for analysis.")

    # Only analyze the first 10,000 flows to be fast
    sample_size = min(10000, len(cic))
    print(f"Analyzing {sample_size} flows...")
    
    if keys_to_use:
        sample_features = np.array([flow[keys_to_use] for flow in cic[:sample_size]])
        print(f"Sample features array shape: {sample_features.shape}")
        
        if len(sample_features.shape) == 3:
            num_flows, seq_len, num_feats = sample_features.shape
            flat_features = sample_features.reshape(-1, num_feats)
        else:
            num_feats = sample_features.shape[1]
            flat_features = sample_features
            
        means = np.mean(flat_features, axis=0)
        stds = np.std(flat_features, axis=0)
        mins = np.min(flat_features, axis=0)
        maxs = np.max(flat_features, axis=0)
        zeros_pct = np.mean(flat_features == 0, axis=0) * 100
        
        print("\n--- Feature Distribution Statistics (Sample) ---")
        print(f"{'Feat':<5} | {'Mean':>10} | {'Std':>10} | {'Min':>10} | {'Max':>10} | {'Zeros %':>8}")
        print("-" * 65)
        for i in range(num_feats):
            print(f"{i:<5} | {means[i]:10.4f} | {stds[i]:10.4f} | {mins[i]:10.4f} | {maxs[i]:10.4f} | {zeros_pct[i]:7.2f}%")
            
except Exception as e:
    print(f"Error: {e}")
