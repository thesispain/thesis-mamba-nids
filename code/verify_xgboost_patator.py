
import os, sys, pickle, time, json
import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score

CIC = "/home/T2510596/Downloads/totally fresh/thesis_final/data/cicids2017_flows.pkl"
MODEL_PATH = "weights/xgboost_32.json"

def eval_per_class(true_y, pred_probs, attack_types, model_name):
    unique_types = np.unique(attack_types)
    targets = ['Benign', 'FTP-Patator', 'SSH-Patator', 'DoS Hulk', 'DoS GoldenEye', 'PortScan']
    
    print(f"\n--- {model_name} Breakdown ---")
    print(f"{'Attack Class':<25} |Count   | AUC (vs Benign) | Avg Prob")
    print("-" * 65)
    
    benign_mask = (attack_types == 'Benign')
    benign_probs = pred_probs[benign_mask]
    
    for atk in targets:
        if atk not in unique_types: continue
        
        if atk == 'Benign':
            print(f"{str(atk):<25} |{len(benign_probs):<8} | {'N/A':<15} | {np.mean(benign_probs):.4f}")
            continue
            
        atk_mask = (attack_types == atk)
        atk_probs = pred_probs[atk_mask]
        
        combined_probs = np.concatenate([benign_probs, atk_probs])
        combined_y = np.concatenate([np.zeros(len(benign_probs)), np.ones(len(atk_probs))])
        
        try: auc = roc_auc_score(combined_y, combined_probs)
        except: auc = 0.5
        
        avg_prob = np.mean(atk_probs)
        count = np.sum(atk_mask)
        print(f"{str(atk):<25} |{count:<8} | {auc:.4f}          | {avg_prob:.4f}")

def main():
    print(f"Loading XGBoost Model from {MODEL_PATH}...")
    if not os.path.exists(MODEL_PATH):
        print("Model file not found!")
        if os.path.exists("weights/xgboost_full.json"): # Alternative Check
             MODEL_PATH = "weights/xgboost_full.json"
             print(f"Found {MODEL_PATH}")
        else:
             print("Cannot find XGBoost weights.")
             return

    # Load Model
    clf = xgb.XGBClassifier()
    clf.load_model(MODEL_PATH)
    
    print("Loading CIC-IDS (Test 100k)...")
    with open(CIC, 'rb') as f: cic_data = pickle.load(f)
    cic_sub = cic_data[:100000] 
    X_test = np.array([d['features'] for d in cic_sub], dtype=np.float32)
    y_test = np.array([d['label'] for d in cic_sub], dtype=np.int64)
    attack_types = np.array([d.get('attack_type', 'Unknown') for d in cic_sub])
    
    # Flatten Features (32 pkts * 5 feats?) -> 160 dim?
    # Check if model expects 160 dim. Or 40.
    # The 'xgboost_32.json' implies 32 packets.
    X_flat = X_test.reshape(X_test.shape[0], -1) 
    
    print("Predicting...")
    try:
        y_prob = clf.predict_proba(X_flat)[:, 1]
    except Exception as e:
        print(f"Prediction Failed: {e}")
        # Maybe feature mismatch. Try raw if not flattened?
        # Re-try with known flattening logic from benchmark_all_metrics.py
        pass
        
    global_auc = roc_auc_score(y_test, y_prob)
    print(f"\nGlobal AUC: {global_auc:.4f}")
    
    eval_per_class(y_test, y_prob, attack_types, "XGBoost-32 (Saved)")

if __name__ == "__main__":
    main()
