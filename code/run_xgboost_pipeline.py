
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import time

# ── Config ──
DATA_PATH = "data/cicids2017_flows.pkl"
CSV_OUT = "data/xgboost_32pkt.csv"
DEVICE = "cuda" # or "cpu"

def main():
    print(f"Loading {DATA_PATH}...")
    with open(DATA_PATH, 'rb') as f:
        data = pickle.load(f)
    print(f"Loaded {len(data):,} flows")

    # ── 1. Flatten Data (The "32 Packet Cap") ──
    print("Flattening features (32 packets * 5 features = 160 columns)...")
    
    X_list = []
    y_list = []
    
    # Feature columns: 32 * [IAT, Size, Direction, Flags, Protocol]
    # Actually Protocol is usually constant, but in this array it's per packet? 
    # Based on inspection: (32, 5). 
    # Let's assume features are [IAT, Size, Direction, Flags, Protocol] or similar.
    # We will just treat them as generic Feature_0..Feature_159
    
    cols = []
    for i in range(32):
        cols.extend([f"pkt_{i}_eat", f"pkt_{i}_size", f"pkt_{i}_dir", f"pkt_{i}_flags", f"pkt_{i}_proto"])

    batch = []
    labels = []
    
    start_t = time.time()
    for idx, d in enumerate(data):
        feat = d['features'] # (32, 5)
        flat = feat.flatten() # (160,)
        batch.append(flat)
        labels.append(d['label'])
        
        if len(batch) >= 10000:
            X_list.append(np.array(batch, dtype=np.float32))
            y_list.extend(labels)
            batch = []
            labels = []
            
    if batch:
        X_list.append(np.array(batch, dtype=np.float32))
        y_list.extend(labels)
        
    X = np.concatenate(X_list)
    y = np.array(y_list)
    
    print(f"Data Shape: {X.shape}") 
    print(f"Flattening took {time.time() - start_t:.2f}s")

    # ── 2. Train/Test Split ──
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    # ── 3. XGBoost ──
    print("Training XGBoost...")
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'tree_method': 'hist', # 'gpu_hist' if GPU available
        'device': DEVICE,
        'max_depth': 6,
        'eta': 0.1
    }
    
    # Check for imbalance
    pos = np.sum(y_train)
    neg = len(y_train) - pos
    scale = neg / pos if pos > 0 else 1.0
    print(f"Pos: {pos}, Neg: {neg}, Scale: {scale:.2f}")
    params['scale_pos_weight'] = scale

    bst = xgb.train(params, dtrain, num_boost_round=100)
    
    # ── 4. Evaluate ──
    preds_prob = bst.predict(dtest)
    preds = (preds_prob > 0.5).astype(int)
    
    acc = (preds == y_test).mean()
    f1 = f1_score(y_test, preds)
    
    print(f"\n=== XGBoost Results (Tabular 32-Pkt) ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, preds))
    
    # Save CSV? User asked to "MAKE THE CSV"
    # Saving 1M rows * 160 cols might be huge (1GB+). 
    # We will save a sample or the whole thing if requested.
    # User said "MAKE THE CSV FILE", implying output.
    
    print(f"Saving DataFrame to {CSV_OUT}...")
    # Construct DataFrame
    df = pd.DataFrame(X, columns=cols)
    df['label'] = y
    
    # Save optimized parquet or csv? User said CSV.
    df.to_csv(CSV_OUT, index=False)
    print("Done.")

if __name__ == "__main__":
    main()
