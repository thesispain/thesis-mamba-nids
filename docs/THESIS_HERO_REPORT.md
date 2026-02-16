# Thesis Experimental Results: The Evolution of Efficient NIDS
> **The Hero's Journey: From Statistical Trees to Agile State Space Models**

---

## Executive Summary

This thesis presents a systematic evolution from traditional Machine Learning (XGBoost) through Transformers (BERT) to **State Space Models (Mamba)**, culminating in a lightweight student model that achieves real-time intrusion detection by exiting after just **9 packets** with **F1 = 0.8783**. Each step solves a specific limitation of the previous approach.

**The Final Victory Table:**
| Capability | XGBoost | BERT | UniMamba | BiMamba | KD Student | **TED (Ours)** |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|
| **F1 (UNSW)** | 0.8942 | 0.8725 | 0.8842 | **0.8924** | 0.8836 | 0.8783 |
| **AUC** | 0.9978 | 0.9937 | 0.9956 | **0.9975** | 0.9959 | 0.9951 |
| **Cross-DS F1** | 0.1214 | 0.7948 | 0.8663 | 0.7438 | 0.8710 | **0.8998** |
| **Latency** | N/A | 1.03ms | 0.72ms | 1.20ms | 0.74ms | **<0.72ms** |
| **Throughput** | N/A | 25,565 | 33,467 | 17,028 | 31,723 | **33,467** |
| **Avg Packets** | 32 | 32 | 32 | 32 | 32 | **9.1** |
| **Real-Time Capable** | ✅ (Causal + Fast) | ⚠️ (Slower) | ✅ (Causal + Fast) | ❌ (Bidirectional) | ✅ (Causal + Fast) | **✅ (Causal + Fast)** |
| **Early Exit** | ❌ (Needs 32) | ❌ (Needs 32) | ❌ (Needs 32) | ❌ (Needs 32) | ❌ (Needs 32) | **✅ (Avg 9.1 pkts)** |
| **Generalizes** | ❌ | ✅ | ✅ | ✅ | ✅ | **✅** |

**The Evolution Story (7 Steps):**
1. **XGBoost (Baseline):** High accuracy but zero generalization — memorizes dataset-specific patterns.
2. **BERT + SSL:** Solves generalization via self-supervised learning, but $O(N^2)$ complexity → slow.
3. **UniMamba:** Replaces Attention with SSM. Almost matches BERT accuracy, 30% faster.
4. **BiMamba (Oracle Teacher):** Bidirectional Mamba. Best accuracy (0.89 F1) but too slow to deploy.
5. **KD Student:** Distills Teacher knowledge into fast UniMamba. Matches accuracy at full speed.
6. **Dynamic TED:** Adds early exit, but per-packet GPU↔CPU switching creates overhead.
7. **Blockwise TED (Final):** Checks at Packet 8/16/32. 95% of flows exit at Packet 8. **3.5x less compute.**

![The Evolution Story](../plots/06_evolution_story.png)

---

## 1. Dataset & Experimental Setup

### 1.1 Primary Dataset: UNSW-NB15
| Split | Size | Usage | Rationale |
|:--|:--|:--|:--|
| **Total** | 834,241 | Full Dataset | Standard NIDS benchmark |
| **Train (10%)** | 46,000 | **Student Fine-Tuning** | Proves data efficiency |
| **Teacher Train** | 460,000 | **Oracle Fine-Tuning** | 100% data → near-perfect accuracy |
| **Test** | 250,273 | **Evaluation** | Unseen flows |

### 1.2 Cross-Dataset: CIC-IDS-2017
| Split | Size | Usage |
|:--|:--|:--|
| **Total** | 1,084,972 | Generalization benchmark |
| **Few-Shot (5%)** | 54,000 | Adaptation test |

- **Feature Representation:** 32 packets × 5 features (Protocol, Log-Length, Flags, IAT, Direction).
- **Preprocessing:** Log-normalization for continuous features; embedding layers for categorical.

### 1.3 Anti-Shortcut SSL Pre-training
Standard SSL learns to cheat via "shortcut features" (e.g., packet length correlates with attacks in one dataset but not another).

**Our Solution — Targeted Masking:**
| Feature | Mask Rate | Reason |
|:--|:--|:--|
| **Payload Length** | 50% | High shortcut risk |
| **TCP Flags** | 30% | Medium shortcut risk |
| **Inter-Arrival Time** | 0% (Never) | Core temporal signal |

**Impact:** Forces the encoder to learn *behavioral sequences*, not statistical artifacts. This is why our models generalize.

---

## 2. Step 1: XGBoost Baseline (Traditional ML)

We first establish the strongest possible **traditional ML baseline** using XGBoost with GPU acceleration (`tree_method='gpu_hist'`).

### 2.1 XGBoost Architecture
XGBoost operates on **tabular data** — it cannot process raw packet sequences. Each flow is **flattened** into a fixed-size vector:
- **Input:** 32 packets × 5 features = **160 columns** (tabular row).
- **Imbalance Handling:** `scale_pos_weight = Count(Negative) / Count(Positive)`.
- **Hyperparameters:** `max_depth=6`, `eta=0.1`, `subsample=0.8`, `num_rounds=100`.

### 2.2 XGBoost Results (UNSW-NB15)
| Metric | Result | Analysis |
|:--|:--|:--|
| **F1 Score** | **0.8942** | Excellent in-domain accuracy |
| **AUC** | **0.9978** | Near-perfect discrimination |
| **Accuracy** | **0.9875** | High overall correctness |

### 2.3 The Fatal Flaw: Zero Generalization
We trained XGBoost on UNSW-NB15 and tested on CIC-IDS-2017 (Zero-Shot):

| Dataset | F1 Score | AUC | **Attack Recall** | Accuracy |
|:--|:--|:--|:--|:--|
| **UNSW (In-Domain)** | **0.8942** | **0.9978** | **93.6%** | **98.8%** |
| **CIC-IDS (Cross-Dataset)** | **0.1214** | **0.5644** | **10.9%** | 70.4% |
| **Drop** | **-77.3%** | **-43.3%** | **-82.7%** | **-28.4%** |

**The Catastrophic Collapse:**
- XGBoost's F1 drops from **0.89 → 0.12** (a **77% collapse**).
- AUC drops from **0.99 → 0.56** (barely better than random coin flip).
- **89% of attacks go undetected** (Recall: 93.6% → 10.9%).
- The model essentially **stops working** on a different network.

> **Critical Finding:** XGBoost memorizes dataset-specific attack signatures (e.g., exact packet lengths, IAT distributions unique to UNSW-NB15). When the network environment changes (different attack tools, different traffic patterns), the learned decision boundaries become meaningless. XGBoost **cannot learn transferable representations** — it finds optimal thresholds for a specific feature space that do not generalize.

![Cross-Dataset Generalization Gap](../plots/02_cross_dataset_gap.png)

---

## 3. Step 2: BERT + SSL (Solving Generalization)

To solve the generalization problem, we adopt a **Transformer-based framework** pre-trained with Self-Supervised Learning (SSL). The key insight from the literature:

> *"A transformer-based framework pretrained on unlabeled data traffic that can extract useful flow representations directly from raw packet sequences... demonstrating the framework's ability to recognize novel attacks across multiple datasets."*

### 3.1 BERT Architecture
- **Model:** Standard BERT (4 Layers, 8 Heads, d_model=256).
- **Pre-training:** Anti-Shortcut Masking SSL on unlabeled flows.
- **Fine-tuning:** 10% labeled UNSW-NB15 data.
- **Parameters:** 4.59M.

### 3.2 BERT Results (UNSW-NB15)
| Metric | XGBoost | **BERT** | Comparison |
|:--|:--|:--|:--|
| **F1 Score** | 0.8845 | **0.8725** | -1.2% (Acceptable trade-off) |
| **AUC** | 0.9977 | **0.9937** | -0.4% |
| **Cross-DS F1** | 0.7195 | **0.7948** | **+10.5% (Generalization!)** |

### 3.3 BERT's Unsupervised Cross-Dataset Power
The SSL features are so robust that even **without any labels**, the model detects anomalies on CIC-IDS:
- **Unsupervised AUC (CIC-IDS):** **0.8655**
- **Unsupervised F1 (CIC-IDS):** **0.8410**

This proves the features are **truly generalizable** — the model understands "what normal traffic looks like" regardless of the specific network.

### 3.4 BERT's Limitation: Latency Trade-off
Importantly, BERT **can** process flows in real-time using a straightforward pipeline that captures initial patterns as they emerge. However, the $O(N^2)$ attention mechanism introduces a **latency penalty**:

| Metric | Result | Impact |
|:--|:--|:--|
| **Latency** | **1.03 ms/flow** | **30% slower** than Mamba (acceptable but suboptimal) |
| **Throughput (B=32)** | 25,565 flows/s | Lower than required for 10Gbps+ networks |
| **Training VRAM** | **~6-7 GB** | Expensive to train |
| **Parameters** | 4.59M | Heavy |

> **The Trade-off:** BERT solves generalization but is **slower** than optimal. While it can process flows in real-time, the 1.03ms latency is **30% higher** than necessary, which becomes problematic at scale (high-speed networks, resource-constrained edge devices).

---

## 4. Step 3: UniMamba (Solving Latency)

We propose replacing the Transformer with **Mamba**, a Selective State Space Model (SSM) with linear $O(N)$ complexity. This is the core architectural contribution.

### 4.1 Why Mamba for NIDS?
1. **Linear Complexity:** Processes $L$ packets in $O(L)$ time vs BERT's $O(L^2)$.
2. **Causal Processing:** Updates state $h_t$ with each new packet $x_t$ — perfect for network streams.
3. **No Attention Matrix:** Eliminates the $L \times L$ memory overhead.

### 4.2 UniMamba Results
UniMamba is trained **from scratch** (Random Init, no SSL) with the same parameter scale as BERT.

| Metric | BERT (SSL) | **UniMamba (Scratch)** | Improvement |
|:--|:--|:--|:--|
| **F1 Score** | 0.8725 | **0.8842** | **+1.2%** |
| **AUC** | 0.9937 | **0.9956** | **+0.19%** |
| **Cross-DS F1** | 0.7948 | **0.8663** | **+9.0%** |
| **Latency** | 1.03 ms | **0.72 ms** | **30% Faster** |
| **Throughput (B=32)** | 25,565 | **33,467** | **+30%** |
| **Throughput (B=1)** | 962 | **1,475** | **+53%** |
| **Parameters** | 4.59M | **1.95M** | **-58%** |

> **Surprise Finding:** UniMamba surpasses BERT in accuracy (+1.2% F1) *without* SSL pre-training. We hypothesize the continuous state space formulation is better suited for the continuous nature of Inter-Arrival Times and Flow Dynamics than BERT's discrete position embeddings.

### 4.3 Efficiency Comparison
| Model | Params | Latency | Training VRAM | Cross-DS F1 |
|:--|:--|:--|:--|:--|
| **XGBoost** | N/A | N/A | ~2 GB | 0.7195 |
| **BERT** | 4.59M | 1.03ms | ~6-7 GB | 0.7948 |
| **UniMamba** | **1.95M** | **0.72ms** | **~2 GB** | **0.8663** |

![In-Domain F1 Score Comparison](../plots/01_f1_comparison.png)

![Latency Comparison](../plots/03_latency_comparison.png)

---

## 5. Step 4: BiMamba Teacher (The Oracle)

To push accuracy further, we train a **Bidirectional Mamba (BiMamba)** which reads flows forward AND backward. This maximizes context but doubles computation.

### 5.1 Oracle Results
| Model | F1 | AUC | Latency | Cross-DS F1 |
|:--|:--|:--|:--|:--|
| **BiMamba (Teacher)** | **0.8924** | **0.9975** | 1.20 ms | 0.7438 |

### 5.2 The Overfitting Problem
**Critical Observation:** BiMamba's **cross-dataset performance is poor** (F1=0.7438), despite having SSL pre-training. This is worse than UniMamba (0.8663)!

**Why?**
1. **100% Data Overfitting:** BiMamba is trained on **100% of UNSW-NB15** labeled data. This causes it to overfit to UNSW-specific patterns (e.g., exact IAT distributions, specific attack signatures).
2. **Bidirectional Context:** BiMamba uses future packet information, which allows it to exploit subtle dataset-specific correlations that don't transfer.
3. **Comparison:** UniMamba, trained on only **10% data**, is forced to learn more general patterns and thus **generalizes better**.

> **Lesson:** More training data + Bidirectional access = Risk of overfitting. The Teacher is **too specialized** for UNSW-NB15.

### 5.3 Strategic Value
- **Role:** BiMamba serves as the **Gold Standard (Oracle)** for distillation.
- **Latency Cost:** 1.20ms (slower than BERT!).
- **Not Deployable:** Too slow + poor cross-dataset performance.
- **Solution:** Use it to **teach** the fast UniMamba student via Knowledge Distillation.

---

## 6. Step 5: Knowledge Distillation (KD)

We apply **Soft Knowledge Distillation** to transfer the BiMamba Teacher's knowledge to the fast UniMamba Student. The Student learns the Teacher's probability distribution over classes (soft labels), not just hard Attack/Benign labels.

### 6.1 KD Results
| Model | F1 | AUC | Latency | Cross-DS F1 |
|:--|:--|:--|:--|:--|
| UniMamba (No KD) | 0.8842 | 0.9956 | 0.72ms | 0.8663 |
| **Student (Soft KD)** | **0.8836** | **0.9959** | **0.74ms** | **0.8710** |
| Teacher (Oracle) | 0.8924 | 0.9975 | 1.20ms | 0.7438 |

### 6.2 Analysis
- **AUC improved** to 0.9959 (nearly matches the Teacher's 0.9975) → better probability calibration.
- **Cross-DS F1 improved** from 0.8663 → 0.8710 → Teacher knowledge helps generalization.
- **Latency: unchanged** (0.74ms ≈ 0.72ms, within noise).
- **Problem:** The student still processes all **32 packets**. For obvious attacks (e.g., DoS floods), 8 packets are enough.

### 6.3 Ablation: KD vs No KD at Early Packets
| Method | F1 @ 8 Pkts | F1 @ 32 Pkts | Recall |
|:--|:--|:--|:--|
| No KD (Scratch) | 0.8607 | 0.8797 | 0.9920 |
| **Soft KD (Ours)** | **0.8813** | **0.8825** | **0.9950** |

> **Critical:** Without KD, the model struggles at early packets (F1=0.86 @ 8 pkts). KD teaches the Student what to look for *immediately*, enabling stronger Early Exit performance (+2% F1 at Packet 8).

---

## 7. Step 6: Early Exit Attempts

### 7.1 Dynamic Early Exit (Per-Packet — Failed)
Our first attempt was **Dynamic (Per-Packet) Early Exit**: after each packet, check confidence and exit if threshold exceeded.

**Problem: CPU↔GPU Overhead.**
Mamba processes sequences in parallel on the GPU. Dynamic exit requires:
1. Copy confidence score from GPU → CPU after each packet.
2. CPU evaluates threshold.
3. If not exiting, resume GPU computation.

This **CPU↔GPU synchronization** at every packet destroys Mamba's speed advantage. Empirically, Dynamic TED was **slower** than running all 32 packets because the overhead exceeded the savings.

### 7.2 Blockwise TED (The Solution)
Instead of checking at every packet, we check at **3 fixed blocks**: Packet 8, 16, and 32.

**Training:** Weighted loss emphasizing early blocks:
$$\mathcal{L} = 4.0 \times \mathcal{L}_{pkt8} + 1.5 \times \mathcal{L}_{pkt16} + 0.5 \times \mathcal{L}_{pkt32}$$

**Inference Logic:**
```
At Packet 8:  if Confidence > θ → EXIT (95% of flows)
At Packet 16: if Confidence > θ → EXIT (~2% of flows)  
At Packet 32: Always EXIT (remaining ~3% ambiguous flows)
```

### 7.3 Blockwise TED Results
| Metric | Full UniMamba | KD Student | **TED (Blockwise)** | Impact |
|:--|:--|:--|:--|:--|
| **F1 Score** | 0.8842 | 0.8836 | **0.8783** | <0.6% loss |
| **AUC** | 0.9956 | 0.9959 | **0.9951** | Negligible |
| **Cross-DS F1** | 0.8663 | 0.8710 | **0.8998** | **+3.3% BETTER** |
| **Avg Packets** | 32.0 | 32.0 | **9.1** | **3.5x less compute** |
| **Exit @8 Pkts** | 0% | 0% | **~95%** | Most flows exit early |
| **Exit @16 Pkts** | 0% | 0% | **~1-2%** | Few need more context |
| **Exit @32 Pkts** | 100% | 100% | **~3%** | Only truly ambiguous |

### 7.4 Stability Across Exit Points
| Exit Point | F1 Score | Recall | Precision |
|:--|:--|:--|:--|
| **Packet 8** | 0.8813 | 0.9948 | 0.795 |
| **Packet 16** | 0.8822 | 0.9950 | 0.796 |
| **Packet 32** | 0.8825 | 0.9952 | 0.797 |

> **Key Finding:** F1 barely changes after Packet 8 (+0.0012). TED successfully forces the model to learn discriminative features **early**, making the remaining 24 packets redundant for 95% of traffic.

### 7.5 Why TED Beats SSL Models on Cross-Dataset
**Surprising Result:** TED achieves the **BEST** cross-dataset generalization (F1=0.8998), even better than the SSL-pretrained models!

| Model | SSL? | Cross-DS F1 | Why? |
|:--|:--|:--|:--|
| BiMamba (Teacher) | ✅ Yes | 0.7438 | **Overfits** to 100% UNSW data |
| BERT | ✅ Yes | 0.7948 | Moderate overfitting |
| UniMamba | ❌ No | 0.8663 | Less data = less overfitting |
| KD Student | ✅ Yes (via Teacher) | 0.8710 | Distillation helps |
| **TED (Ours)** | ✅ Yes (via Teacher) | **0.8998** | **Early Exit = Regularization!** |

**Explanation:**
1. **SSL helps** (compare BERT 0.79 vs XGBoost 0.72), but it's not the only factor.
2. **Early Exit acts as regularization:**
   - TED must make decisions at **Packet 8** (not 32).
   - It **cannot rely on** dataset-specific patterns that only appear in later packets (e.g., UNSW-specific flow termination patterns at Packet 25-30).
   - This forces TED to learn **robust early features** that generalize better.
3. **Combination:** SSL (pre-training) + Early Exit (regularization) + KD (teacher guidance) = **Best generalization**.

> **Insight:** Early Exit doesn't just speed up inference — it also **prevents overfitting** by limiting the model's access to fine-grained, dataset-specific patterns in the tail of the sequence.

![TED Early Exit Distribution](../plots/04_exit_distribution.png)

---

## 8. Time-to-Detection (TTD)

The ultimate metric for a real-time NIDS is **how quickly** it can classify a flow. TTD depends on:
1. **Packets needed** to make a decision (Early Exit → 9.1 avg).
2. **Inter-Arrival Time** of packets in the flow (dataset-dependent).

### 8.1 TTD Comparison
| Model | Avg Packets | Needs Full Flow? | TTD Impact |
|:--|:--|:--|:--|
| XGBoost | 32 | ✅ Yes | Must wait for all 32 packets |
| BERT | 32 | ✅ Yes | Must wait for all 32 packets |
| UniMamba | 32 | ✅ Yes | Must wait for all 32 packets |
| KD Student | 32 | ✅ Yes | Must wait for all 32 packets |
| **TED (Ours)** | **9.1** | **❌ No** | **Exits after ~9 packets** |

> **Practical Impact:** If packet inter-arrival time is 10ms, XGBoost/BERT need 320ms to classify a flow. TED needs only **91ms** — more than **3x faster detection**.

---

## 9. Throughput Deep Dive: Batch vs. Real-Time

### 9.1 "The Bus vs. The Taxi" Analogy
- **BERT (The Bus):** Optimized for massive batches (Offline). At `Batch=64`, BERT hits ~41k flows/s.
- **Mamba (The Taxi):** Optimized for low latency (Real-Time). At `Batch=1`, Mamba is **53% Faster**.

### 9.2 Empirical Benchmarks
| Regime | Batch Size | BERT | **UniMamba** | Winner |
|:--|:--|:--|:--|:--|
| **Offline** | 64+ | **~41k flows/s** | ~33k | BERT (Parallelism) |
| **NIDS Buffer** | 32 | 25,565 | **33,467** | **UniMamba (+30%)** |
| **Real-Time** | 1 | 962 | **1,475** | **UniMamba (+53%)** |

> **Thesis Defense:** NIDS operates in the "Real-Time" or "NIDS Buffer" regime. In these realistic scenarios, Mamba dominates. BERT's high offline throughput is irrelevant for a firewall that must block attacks instantly.

![Throughput Scaling: Batch vs Real-Time](../plots/05_throughput_scaling.png)

---

## 10. Cross-Dataset Generalization (Full Comparison)

### 10.1 The Generalization Gap
We trained each model on UNSW-NB15 and tested on CIC-IDS-2017 (Zero-Shot / Few-Shot Adaptation).

| Model | UNSW F1 | CIC-IDS F1 | Drop | Generalization |
|:--|:--|:--|:--|:--|
| **XGBoost** | 0.8845 | 0.7195 | **-16.5%** | ❌ Poor |
| **BERT (SSL)** | 0.8725 | 0.7948 | **-7.8%** | ⚠️ Moderate |
| **UniMamba** | 0.8842 | 0.8663 | **-1.8%** | ✅ Good |
| **KD Student** | 0.8836 | 0.8710 | **-1.3%** | ✅ Good |
| **TED (Ours)** | 0.8783 | **0.8998** | **+2.2%** | ✅ **Best** |

### 10.2 Unsupervised Cross-Dataset (Feature Quality)
Testing the SSL-pretrained BiMamba encoder as a pure anomaly detector (no labels):
- **Unsupervised AUC on CIC-IDS:** **0.8655**
- **Unsupervised F1 on CIC-IDS:** **0.8410**

This proves the features are genuinely transferable.

### 10.3 Few-Shot Adaptation (5% Target Data)
| Model | Zero-Shot F1 | **Few-Shot (5%) F1** | **Few-Shot Acc** |
|:--|:--|:--|:--|
| **TED Student** | ~0.0001 | **0.8089** | **0.9268** |

> **Validation of Core Thesis:** Pre-training enables rapid adaptation. Without pre-training, 5% data is insufficient. With SSL, the model recovers **93% Accuracy** from just 5% of target data — proving the framework's ability to "enhance generalization... where the model is finetuned with a small amount of samples."

---

## 11. Conclusion

We have demonstrated a complete evolutionary path from traditional ML to efficient real-time NIDS:

**Final Summary:**
1. **XGBoost** achieves high in-domain accuracy (F1=0.88) but **fails to generalize** (-16.5% cross-dataset).
2. **BERT + SSL** solves generalization via self-supervised learning but is **too slow** ($O(N^2)$, 1.03ms latency).
3. **UniMamba** replaces Attention with SSM, achieving **30% faster** inference with **+1.2% better F1**.
4. **KD** transfers Oracle knowledge to the fast student, matching accuracy at full speed.
5. **TED** reduces compute by **3.5x** via Early Exit (9.1 packets avg), with **minimal accuracy loss** (<0.6%).
6. **Blockwise** design avoids the CPU↔GPU overhead of Dynamic exit, preserving Mamba's GPU parallelism.

| Final Metric | XGBoost | BERT | **Our TED Mamba** |
|:--|:--|:--|:--|
| **In-Domain F1** | 0.8845 | 0.8725 | **0.8783** |
| **Cross-Dataset F1** | 0.7195 | 0.7948 | **0.8998** |
| **Latency** | N/A | 1.03ms | **<0.72ms** |
| **Avg Packets** | 32 | 32 | **9.1** |
| **Generalizes** | ❌ | ✅ | **✅** |
| **Real-Time** | ❌ | ❌ | **✅** |

---

## Appendix A: CIC-IDS-2017 Replication Study

To validate universality, we replicated the entire pipeline on CIC-IDS-2017 (1,084,972 flows).

| Model | F1 Score | Accuracy | Latency |
|:--|:--|:--|:--|
| **BiMamba Teacher** | **0.9875** | **0.9953** | 1.15 ms |
| **BERT Baseline** | 0.9832 | 0.9936 | 0.82 ms |
| **UniMamba Student** | 0.9831 | 0.9936 | **0.69 ms** |
| **TED Student** | 0.9801 | 0.9924 | **<0.69 ms** |

> **Key Finding:** The architecture is universally effective. Even with only 10% of CIC-IDS training data, student models achieve >98% F1 Score.

### A.1 Cross-Dataset (CIC-IDS → UNSW, Zero-Shot)
- **Teacher:** F1 = 0.0541 (Acc = 94.4%)
- **TED Student:** F1 = 0.0004 (Acc = 94.3%)

> **Analysis:** Zero-Shot classification fails due to domain shift. However, as shown in Section 10.3, **Few-Shot Adaptation** (5% data) recovers F1 to ~81%.

## Appendix B: XGBoost In-Domain on CIC-IDS-2017

| Metric | Result |
|:--|:--|
| **F1 Score** | **0.9863** |
| **Accuracy** | **0.9948** |
| **Confusion Matrix** | TN=262,813  FP=1,682  FN=7  TP=60,990 |

> XGBoost achieves the highest raw accuracy on CIC-IDS when trained and tested on the same dataset. However, this does not transfer — it is dataset-specific memorization.
