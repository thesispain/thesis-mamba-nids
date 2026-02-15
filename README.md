# Efficient Real-Time Network Intrusion Detection via State Space Models (Mamba) & Distillation

This repository contains the official implementation of the Master's Thesis project: **"Replacing Transformers with SSMs for Real-Time NIDS"**.

## 🚀 Key Results
We propose a novel architecture combining:
1.  **UniMamba (Student):** A lightweight State Space Model with linear $O(N)$ complexity.
2.  **TED (Temporally-Emphasised Distillation):** A training method that forces early detection.

### Performance vs. BERT Baseline
| Metric | Standard BERT | **Our Model (UniMamba)** | Impact |
| :--- | :--- | :--- | :--- |
| **F1 Score** | 0.8725 | **0.8842** | **+1.2% Accuracy** |
| **Latency** | 1.03 ms | **0.72 ms** | **30% Faster** |
| **Compute** | 32 Packets | **9.1 Packets** | **3.5x Less Compute** |
| **Training VRAM** | ~6-7 GB | **~2 GB** | **Democratic AI** |

## 📂 Project Structure
- `code/`: Python scripts for Training, Evaluation, and Pipeline Orchestration.
  - `run_full_evaluation.py`: Main entry point for UNSW-NB15 experiments.
  - `run_cicids_pipeline.py`: Replication script for CIC-IDS-2017.
- `docs/`: Comprehensive documentation and the **Final Thesis Report** (`THESIS_HERO_REPORT.md`).
- `results/`: JSON outputs of all experiments.
- `plots/`: Generated figures used in the thesis.
- `weights/`: Pre-trained model weights (Teacher, Student, SSL).

## 🛠️ Usage
1.  **Install Constraints:** Requires `torch`, `mamba_ssm`, `sklearn`.
2.  **Dataset:** Place `cicids2017_flows.pkl` and `ctu13_flows.pkl` in `data/`.
3.  **Run Pipeline:**
    ```bash
    python code/run_full_evaluation.py
    ```

## 📜 Citation
If you use this code, please cite the associated thesis.
