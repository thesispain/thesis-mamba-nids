#!/usr/bin/env python3
"""
XGBoost Cross-Dataset Test:
Train on UNSW-NB15 (tabular 32-pkt), Test on CIC-IDS-2017 (zero-shot).
This proves XGBoost CANNOT generalize across datasets.
"""
import pickle, time
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score

UNSW_PATH = "../Organized_Final/data/unswnb15_full/finetune_mixed.pkl"
CICIDS_PATH = "data/cicids2017_flows.pkl"

def flatten(data):
    X = np.array([d['features'].flatten() for d in data], dtype=np.float32)
    y = np.array([d['label'] for d in data])
    return X, y

def main():
    # 1. Train on UNSW-NB15
    print("Loading UNSW-NB15...")
    with open(UNSW_PATH, 'rb') as f:
        unsw = pickle.load(f)
    X_unsw, y_unsw = flatten(unsw)
    X_train, X_test, y_train, y_test = train_test_split(
        X_unsw, y_unsw, test_size=0.3, random_state=42, stratify=y_unsw)
    
    print(f"UNSW Train: {X_train.shape}, Test: {X_test.shape}")
    
    # Train XGBoost
    neg = np.sum(y_train == 0)
    pos = np.sum(y_train == 1)
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'tree_method': 'hist',
        'device': 'cuda',
        'max_depth': 6,
        'eta': 0.1,
        'scale_pos_weight': neg / pos if pos > 0 else 1.0
    }
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest_unsw = xgb.DMatrix(X_test, label=y_test)
    
    print("Training XGBoost on UNSW-NB15...")
    bst = xgb.train(params, dtrain, num_boost_round=100)
    
    # In-domain eval
    preds = bst.predict(dtest_unsw)
    y_pred = (preds > 0.5).astype(int)
    f1_unsw = f1_score(y_test, y_pred)
    acc_unsw = accuracy_score(y_test, y_pred)
    auc_unsw = roc_auc_score(y_test, preds)
    print(f"\n=== UNSW In-Domain ===")
    print(f"F1: {f1_unsw:.4f}, Acc: {acc_unsw:.4f}, AUC: {auc_unsw:.4f}")
    
    # 2. Cross-Dataset: Test on CIC-IDS-2017
    print("\nLoading CIC-IDS-2017...")
    with open(CICIDS_PATH, 'rb') as f:
        cicids = pickle.load(f)
    X_cic, y_cic = flatten(cicids)
    dcic = xgb.DMatrix(X_cic)
    
    preds_cic = bst.predict(dcic)
    y_pred_cic = (preds_cic > 0.5).astype(int)
    f1_cic = f1_score(y_cic, y_pred_cic)
    acc_cic = accuracy_score(y_cic, y_pred_cic)
    try:
        auc_cic = roc_auc_score(y_cic, preds_cic)
    except:
        auc_cic = 0.0
    
    print(f"\n=== CIC-IDS Cross-Dataset (Zero-Shot) ===")
    print(f"F1: {f1_cic:.4f}, Acc: {acc_cic:.4f}, AUC: {auc_cic:.4f}")
    
    print(f"\n=== SUMMARY ===")
    print(f"UNSW (In-Domain):  F1={f1_unsw:.4f}  AUC={auc_unsw:.4f}")
    print(f"CIC-IDS (Cross):   F1={f1_cic:.4f}  AUC={auc_cic:.4f}")
    print(f"Drop:              F1={f1_unsw-f1_cic:.4f}  AUC={auc_unsw-auc_cic:.4f}")

if __name__ == "__main__":
    main()
