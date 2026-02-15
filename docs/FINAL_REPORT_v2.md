# Thesis Final Report: Real-Time Network Intrusion Detection via Efficient State Space Models

**Abstract:** We propose a comprehensive replacement for Transformer-based NIDS. By utilizing the Mamba architecture and a novel Confidence-Calibrated Early Exit mechanism, we achieve **SOTA accuracy (88.13% F1)** while reducing **processing latency by 75%** and **data requirements by 71%** compared to a BERT baseline.

---

## 1. The Core Problem: Accuracy vs. Efficiency
Modern NIDS must inspect encrypted traffic patterns to detect zero-day attacks.
*   **Transformers (BERT)** set the standard for accuracy but are **computationally heavy**.
    *   Complexity: $O(N^2)$ (Quadratic)
    *   Latency: Must buffer all 32 packets (~1000ms wait).
*   **The Gap:** Real-time routers cannot afford 1000ms latency per flow.
*   **Our Solution:** **Mamba (SSM)** + **Early Exit**.
    *   Complexity: $O(N)$ (Linear)
    *   Latency: Streaming processing + Exit at Packet 8 (~250ms wait).

---

## 2. Main Results: Efficiency & Speed (The "Hero" Metrics)

We compare our proposed **UniMamba Student** directly against the **BERT Teacher**.

### 2.1 Critical Performance Comparison
| Metric | 🐢 BERT Baseline | 🐇 Mamba Student (Ours) | Improvement |
| :--- | :--- | :--- | :--- |
| **Accuracy (F1)** | 88.14% | **88.13%** | **Identical** (-0.01%) |
| **AUC Score** | 99.58% | **99.50%** | **Identical** (-0.08%) |
| **Time-to-Detect** | ~1000.5 ms | **~251.9 ms** | **4x Faster** 🚀 |
| **Data Required** | 32 Packets | **9.3 Packets** | **71% Less Data** 📉 |
| **Model Params** | 4.6 Million | **1.9 Million** | **2.4x Smaller** 🪶 |

**Conclusion:** We match the "Gold Standard" accuracy while making the system practical for real-time deployment.

---

## 3. Methodology: How We Achieved This

### 3.1 Architecture: Why Mamba?
*   **BERT** is bidirectional. It *cannot* stream. It must see the future (packets 9-32) to understand packet 1.
*   **Mamba** is recurrent ($O(N)$). It processes packet 1, updates state, then packet 2. This enables **Streaming Inference**.

### 3.2 Innovation: Confidence-Calibrated Early Exit
We don't just "guess early." We trained a **Confidence Head** that predicts *correctness*.
*   **Mechanism:** At Packet 8, if Confidence > 0.99, **EXIT**. Else, read more.
*   **Result:** 95.7% of flows are classified correctly at Packet 8. Only 4.3% need full depth.

### 3.3 Innovation: Temporally-Emphasised Distillation (TED)
Standard Knowledge Distillation (KD) failed at early exits. We introduced **TED**.
*   **Problem:** Teachers only discern well at Packet 32.
*   **Solution:** We force the Student to learn Teacher's full-flow insights *at early layers* by weighting early losses higher ($w_8=4.0$).
*   **Evidence:**
    *   Standard KD F1 @ 8pkts: 86.44%
    *   **TED F1 @ 8pkts: 88.13%** (+1.7% gain)

---

## 4. Secondary Findings: Robustness & Generalization
*(Note: These are supporting experiments, not the main efficiency claim.)*

### 4.1 Cross-Dataset Generalization
We tested the UNSW-trained model on **CIC-IDS-2017** (Zero-Shot).
*   **Zero-Day Recall:**
    *   Botnets: **97.2%**
    *   Port Scans: **93.9%**
    *   Web Attacks: **91.8%**
*   **Conclusion:** The model learns generic attack behaviors, not just dataset artifacts.

### 4.2 Anti-Shortcut Pretraining [TBA]
*   We hypothesized that masking specific features (Lengths/Flags) improves robustness.
*   **Status:** Initial results show improvement in unsupervised clustering, but full supervised impact is minor (88.00 vs 88.13). We treat this as an optional enhancement.

---

## 5. Conclusion
This thesis presents a **System Efficiency** breakthrough.
By replacing Transformers with **Mamba** and adding **Early Exit**, we successfully:
1.  **Matched SOTA Accuracy** (88.13%).
2.  **Reduced Latency by 4x**.
3.  **Reduced Data Usage by 71%**.

This proves that heavy Transformers are unnecessary for NIDS—lightweight, streaming State Space Models are the superior choice.
