import pickle
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score

# Paths
UNSW_PATH = "../Organized_Final/data/unswnb15_full/finetune_mixed.pkl"
CICIDS_PATH = "data/cicids2017_flows.pkl"

def flatten_subset(data, num_packets=8):
    # Features are [Proto, Flags, Dir, Length, IAT] (5 features per packet)
    # Total features = 5 * 32 = 160.
    # We want first 8 packets -> 5 * 8 = 40 features.
    
    X_list = []
    y_list = []
    
    for d in data:
        # Get full 32-packet features (Sequence Length, 5)
        feats = d['features'] 
        # Truncate to first num_packets
        feats_subset = feats[:num_packets, :]
        # Flatten
        X_list.append(feats_subset.flatten())
        y_list.append(d['label'])
        
    return np.array(X_list, dtype=np.float32), np.array(y_list)

def main():
    print(f"=== Experiment: XGBoost Limited to First 8 Packets ===")
    
    # 1. Load UNSW and Limit to 8 Packets
    print("Loading UNSW-NB15...")
    with open(UNSW_PATH, 'rb') as f:
        unsw = pickle.load(f)
    
    X, y = flatten_subset(unsw, num_packets=8)
    print(f"UNSW (8 Pkts) Shape: {X.shape}")
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y)
        
    # Train
    neg = np.sum(y_train == 0)
    pos = np.sum(y_train == 1)
    scale = neg / pos if pos > 0 else 1.0
    
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'tree_method': 'hist',
        'device': 'cuda',
        'max_depth': 6,
        'eta': 0.1,
        'scale_pos_weight': scale
    }
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    
    print("Training XGBoost@8...")
    bst = xgb.train(params, dtrain, num_boost_round=100)
    
    # Eval In-Domain
    preds = bst.predict(dtest)
    auc_unsw = roc_auc_score(y_test, preds)
    print(f"UNSW In-Domain AUC (@8 pkts): {auc_unsw:.4f}")
    
    # 2. Load CIC-IDS and Limited to 8 Packets
    print("Loading CIC-IDS-2017...")
    with open(CICIDS_PATH, 'rb') as f:
        cic = pickle.load(f)
        
    X_cic, y_cic = flatten_subset(cic, num_packets=8)
    print(f"CIC-IDS (8 Pkts) Shape: {X_cic.shape}")
    
    dcic = xgb.DMatrix(X_cic)
    preds_cic = bst.predict(dcic)
    
    auc_cic = roc_auc_score(y_cic, preds_cic)
    print(f"\n=== CRITICAL RESULT ===")
    print(f"XGBoost Cross-DS AUC (@8 pkts): {auc_cic:.4f}")
    
    # Compare with TED
    print(f"TED Cross-DS AUC (@8 pkts):     0.7637 (Benchmark)")
    
    if auc_cic > 0.7637:
        print("RESULT: XGBoost wins. Thesis argument is flawed.")
    else:
        print("RESULT: TED wins. Thesis argument is VALIDATED.")

if __name__ == "__main__":
    main()
