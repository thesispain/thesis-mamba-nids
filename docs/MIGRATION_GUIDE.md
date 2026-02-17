# Migration Guide: Thesis Project Setup

This guide details exactly what files you need to copy to your new PC and how to set them up to run the experiments.

## 1. Essential Files Checklist
Copy these files/folders from your current directory (`thesis_final`) to your new machine:

### 📂 Code (Notebooks)
- **`Part1_SSL_Pretraining.ipynb`**:
  - *Purpose*: Pre-trains the BERT model from scratch using CLS token pooling.
  - *Use*: Run this **first** on your new PC if you want to generate fresh, guaranteed-correct weights.
- **`Part3_Comprehensive_Evaluation.ipynb`**:
  - *Purpose*: Loads all models (XGBoost, BiMamba, TED, BERT) and runs the full evaluation suite.
  - *Use*: Run this to generate the final tables/results.

### 📂 Weights (Model Checkpoints)
Copy the entire `weights/` folder. Specifically, ensure these 3 key files are present:
1.  **`weights/ssl/bert_cutmix_v5_partial.pth`**: 
    - The BERT CLS-compatible checkpoint. (Note: Only SSL pre-trained, needs head training for max performance).
2.  **`weights/teachers/teacher_bimamba_retrained.pth`**: 
    - The verified BiMamba model (0.99 In-Domain, 0.76 Zero-Day).
3.  **`weights/students/student_ted.pth`**: 
    - The TED (Early Exit) model.

### 📂 Data (Datasets)
You need two specific `.pkl` files.
1.  **CIC-IDS-2017 (Zero-Day Test)**:
    - Current Location: `data/cicids2017_flows.pkl`
    - Action: Copy `data/cicids2017_flows.pkl` to your new `data/` folder.
2.  **UNSW-NB15 (Train/In-Domain)**:
    - Current Location: `../Organized_Final/data/unswnb15_full/finetune_mixed.pkl`
    - Action: **Find this file** and copy it to your new `data/` folder (e.g., rename it to `data/unsw_finetune_mixed.pkl`).

---

## 2. Setup Instructions on New PC

### Step 1: Install Dependencies
Ensure you have Python (3.8+) and CUDA installed. Run:
```bash
pip install torch numpy pandas scikit-learn xgboost mamba-ssm jupyter
```
*Note: `mamba-ssm` installation can be tricky; ensure you have `nvcc` (CUDA compiler) available.*

### Step 2: Update File Paths
Open `Part3_Comprehensive_Evaluation.ipynb` in Jupyter.
Find the **Config Cell** (usually Cell 2) and update the paths:
```python
# OLD PATH (Current PC)
UNSW = "../Organized_Final/data/unswnb15_full/finetune_mixed.pkl"

# NEW PATH (Example)
UNSW = "data/unsw_finetune_mixed.pkl"  # <--- Change this to match where you put the file
```

---

## 3. Recommended Workflow

### 1️⃣ Train BERT (Recommended)
Since the current BERT weights are SSL-only (no classifier head), best performance comes from training fresh.
1. Open `Part1_SSL_Pretraining.ipynb`.
2. Update the Data Path (`DATA_FILE`).
3. Run All Cells.
4. This will create optimal weights at `weights/ssl/bert_standard_ssl_optimized.pth`.

### 2️⃣ Evaluate All Models
1. Open `Part3_Comprehensive_Evaluation.ipynb`.
2. Update the Data Paths (`UNSW`, `CIC`).
3. (Optional) Update the BERT weight path to point to your *newly trained* weights from Step 1 (`weights/ssl/bert_standard_ssl_optimized.pth`).
4. Run All Cells.
5. You should see:
   - **BiMamba**: ~0.99 (In) / ~0.76 (Zero)
   - **BERT**: ~0.84+ (Zero) [If using trained weights]
   - **TED**: ~0.95+ (Efficiency)
