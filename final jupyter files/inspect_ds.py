import pickle
from pathlib import Path

ROOT = Path('/home/T2510596/Downloads/totally fresh')
CIC_PATH = ROOT / 'thesis_final' / 'data' / 'cicids2017_flows.pkl'

print("Using ijson or partial loads to inspect dataset")
import gc

class _LoadOne:
    pass

try:
    with open(CIC_PATH, 'rb') as f:
        print("Wait, just unpacking minimal features via standard load to see memory impact...")
        cic = pickle.load(f)
        
    print(f"Total rows: {len(cic)}")
    print(f"Sample dict keys: {list(cic[0].keys())}")
    
    # Take a subset right away and delete original to free RAM
    sub_cic = cic[:5000]
    del cic
    gc.collect()
    
    import numpy as np
    
    print("\nFeature inspection on first 5000 samples...")
    if 'features' in sub_cic[0]:
        feat_key = 'features'
    elif 'input_ids' in sub_cic[0]:
        feat_key = 'input_ids'
    else:
        feat_key = list(sub_cic[0].keys())[0]
        
    feats = np.array([x[feat_key] for x in sub_cic])
    if len(feats.shape) == 3:
        flat_feats = feats.reshape(-1, feats.shape[2])
    else:
        flat_feats = feats
        
    print(f"Flattened features shape: {flat_feats.shape}")
    means = np.mean(flat_feats, axis=0)
    mins = np.min(flat_feats, axis=0)
    maxs = np.max(flat_feats, axis=0)
    
    print("\n--- Feature Distribution Statistics (Sample) ---")
    print(f"{'Feat':<5} | {'Mean':>10} | {'Min':>10} | {'Max':>10}")
    print("-" * 45)
    for i in range(len(means)):
        print(f"{i:<5} | {means[i]:10.4f} | {mins[i]:10.4f} | {maxs[i]:10.4f}")
        
    if 'label' in sub_cic[0]:
        labels = [x['label'] for x in sub_cic]
        from collections import Counter
        print(f"\nLabel distribution in sample: {Counter(labels)}")

except Exception as e:
    print("Error:", e)
