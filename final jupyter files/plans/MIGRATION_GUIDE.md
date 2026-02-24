# System Migration & Context Recovery Guide

This guide is designed to help you (or another AI assistant) immediately resume work on the NIDS Predictive Self-Distillation thesis project when you migrate to a new PC. 

This repository contains the code, but **the large dataset files (`.pkl`) and weights (`.pth`) were ignored via `.gitignore` to keep the git history clean.** You must manually copy those files or recreate them.

## 1. Required Data Files & Locations
When setting up on the new PC, ensure you recreate or copy the following dataset pickle files matching these exact paths (relative to `/home/T2510596/Downloads/totally fresh` or whatever your new `ROOT` directory becomes):

### UNSW-NB15 Data (In-Domain)
*   `Organized_Final/data/unswnb15_full/pretrain_50pct_benign.pkl` (Used for SSL Pre-training)
*   `Organized_Final/data/unswnb15_full/finetune_mixed.pkl` (Used for few-shot supervised tasks and evaluation splits)

### Cross-Dataset Benchmarks 
*(Note: As of Feb 2026, the CIC-IDS pre-processing is flawed/unscaled and needs a fix!)*
*   `thesis_final/data/cicids2017_flows.pkl` 
*   `thesis_final/data/ctu13_flows.pkl`

## 2. Python Environment Setup
You will need the `mamba_env` virtual environment containing the state-space model dependencies.
```bash
# Recommended command to restart environment:
source ../../mamba_env/bin/activate
# Required libraries:
pip install torch mamba_ssm xgboost scikit-learn numpy pandas
```

## 3. Core Working Directory
All execution happens within:
`thesis_final/final jupyter files/`

## 4. Historical Scripts (The "Old" Pipeline)
If you need to reproduce the old results (before the architectural pivot):
1.  **`FULL_PIPELINE_SSL_PRETRAINING.py`**: The original full execution script ported from the monolithic Jupyter Notebook.
2.  **`run_thesis_eval.py`**: Contains the hard-label Classification, Knowledge Distillation, and the sequence-restarting Blockwise TED Model. (Shows the ~1.2ms Mamba vs ~0.5ms BERT latency numbers).
3.  **`run_unsupervised_eval.py`**: Calculates the unsupervised cosine-similarity baseline (0.88 AUC on UNSW, 0.44 AUC on CIC-IDS due to the scaling flaw).
4.  **`inspect_ds_detailed.py`**: The script used to physically verify that `cicids2017_flows.pkl` Feature 3 (Payload Size) was unscaled (max ~372,000) compared to UNSW (max ~14).

## 5. The Active Pivot: Predictive Self-Distillation
**Current state of the project:** We realized that Knowledge Distillation + Hard-Label Cross Entropy destroys the generalized representations formed during SSL.

We have agreed to pivot to **Predictive Self-Distillation for Early Exit**:
1.  **Unsupervised SSL on UniMamba** (Avoid using the BiMamba teacher).
2.  **Centroid Distance Classification:** Use the distance to `Benign` vs `Attack` cluster centers in the pure SSL representation space to detect anomalies (No linear classification layers).
3.  **MSE Predictive Exits:** Train the 8-packet and 16-packet layers to minimize Mean Squared Error (MSE) compared to the uncorrupted 32-packet embedding. 

A starter script named `run_predictive_ted.py` has been created. **This is where the new AI should begin its work upon migration.**
