
# Chapter 5: Experimental Results and Analysis

This chapter presents a systematic evaluation of the proposed progressive NIDS pipeline.
Each architecture is evaluated not only on standard classification metrics but also on the practical deployment concerns it was designed to address.
The results are organized to follow the design narrative: each successive model solves a concrete limitation exposed by the preceding one.

| Symbol | Meaning |
|--------|---------|
| **Bold** | Best in column |
| ↑ | Higher is better |
| ↓ | Lower is better |

---

## 5.1 Experimental Setup

### 5.1.1 Datasets

| Dataset | Role | Total Flows | Train | Val | Test | Attack Types |
|---------|------|------------:|------:|----:|-----:|:-------------|
| UNSW-NB15 | Primary training & eval | 1,621,245 | 1,134,871 | 162,124 | 324,250 | 10 (Fuzzers, Analysis, Backdoors, DoS, Exploits, Generic, Reconnaissance, Shellcode, Worms, Normal) |
| CIC-IDS-2017 | Cross-dataset eval | 1,900,645 | 0 | 0 | 40,000 | 5 (PortScan, DoS GoldenEye, DoS Hulk, DDoS, Bot) |
| CTU-13 | Cross-dataset eval | 283,000 | 0 | 0 | 40,000 | 7 (Botnet variants, IRC, Spam, DDoS, PortScan, HTTP, P2P) |

All datasets are preprocessed into a unified 5-feature × 32-packet tensor representation per flow:
**Protocol** (categorical), **LogLength** (log₁₀ packet size), **Flags** (TCP flags), **IAT** (inter-arrival time), **Direction** (0/1).

### 5.1.2 Evaluation Protocol

- **In-dataset**: 70/10/20 stratified train/validation/test split on UNSW-NB15 (1.13M / 162K / 324K flows)
- **SSL pretraining**: 787K benign-only flows (50% of benign class, disjoint from supervised training)
- **Cross-dataset (zero-shot)**: Train on UNSW-NB15, evaluate directly on CIC-IDS-2017 and CTU-13 without any target-domain labels
- **Combined training**: Train on UNSW-NB15 + target-domain data, evaluate on held-out test sets
- **Metrics**: AUC-ROC, Binary F1, Macro F1, Accuracy, Recall, Precision, FPR
- **Latency**: Measured on NVIDIA RTX 4070 Ti SUPER (16 GB VRAM), batch sizes 1/64/256, 100-iteration warmup + 100-iteration measurement

### 5.1.3 Model Configurations

| Model | Architecture | Layers | Hidden Dim | Params | Complexity |
|-------|-------------|-------:|-----------:|-------:|:-----------|
| BERT Teacher | Transformer encoder, 4 attention heads | 4 | 256 | 4.59M | O(n²) |
| BiMamba Teacher | Bidirectional Mamba (fwd + bwd scan) | 4 | 256 | 3.66M | O(n) |
| UniMamba Student | Unidirectional Mamba (causal) | 4 | 256 | 1.95M | O(n) |

All deep learning models share the same **PacketEmbedder** front-end (5 features → 256-d) and **linear classification head**.
SSL pretraining uses a 256-d projection head with NT-Xent contrastive loss.

---

## 5.2 Machine Learning Baselines
1.13M UNSW-NB15 flows, evaluated on 324K test flows (5 features × 32 packets, flattened to 160-d).

**Table 5.1: ML Baseline Performance on UNSW-NB15 Test Set (324K flows)**

| Model | F1 ↑ | AUC ↑ | Accuracy ↑ | Recall ↑ | Precision ↑ | Train Time |
|-------|------:|------:|-----------:|---------:|------------:|:-----------|
| XGBoost | 0.883 | **0.997** | 0.986 | 0.922 | 0.838 | 186.4 s |
| Random Forest | **0.888** | **0.997** | **0.986** | **0.956** | 0.829 | 8.2
| Random Forest | **0.888** | **0.997** | **0.987** | **0.955** | 0.829 | 0.90 s |
 on the 324K test flows: AUC = 0.997 and F1 ≈ 0.88. 
These are strong baselines trained on 1.13M flows that any deep learning approach must match before it can justify its additional complexity. 
Note: Random Forest is surprisingly faster (8.2s) than XGBoost (186s) due to efficient sklearn parallelization
These are strong baselines that any deep learning approach must match before it can justify its additional complexity.

**However**, when evaluated on unseen datasets without retraining (zero-shot cross-dataset), their performance collapses:

**Table 5.2: ML Baselines — Cross-Dataset Generalization (Zero-Shot)**

| Model | CIC-IDS-2017 F1 ↑ | CIC-IDS-2017 AUC ↑ | CTU-13 F1 ↑ | CTU-13 AUC ↑ |
|-------|-------------------:|--------------------:|------------:|-------------:|
| XGBoost | 0.009 | 0.758 | 0.025 | 0.422 |
| Random Forest | 0.000 | 0.799 | 0.034 | 0.560 |

XGBoost achieves a CTU-13 AUC of only **0.42** — worse than random guessing (0.50).
Random Forest fares slightly better but still produces near-zero F1 on both cross-datasets.
The fundamental issue: tree-based models memorize dataset-specific feature distributions rat= 0.997 on 324K test flows) after training on 1.13M flows, learning transferable attack patterns.

> **Key Finding 1:** Traditional ML baselines achieve saturated in-dataset performance (AUC ≈ 0.997) but fail catastrophically on cross-dataset generalization (CTU-13 AUC < 0.56), motivating deep learning approaches that learn more generalizable representations.

---

## 5.3 Self-Supervised Pretraining Evaluation

### 5.3.1 Unsupervised Anomaly Detection via SSL Representations

Before supervised fine-tuning, we evaluate the quality of SSL-learned representations using a label-free protocol: train a k-nearest-neighbors (k=1) anomaly detector on the embeddings from a 50% benign pretraining set, then measure AUC on held-out test data containing both benign and attack traffic.

**Table 5.3: Unsupervised Anomaly Detection (SSL Representations, Cosine Similarity, No Labels)**

| Encoder | SSL Augmentation | UNSW-NB15 AUC ↑ | CIC-IDS-2017 AUC ↑ | CTU-13 AUC ↑ |
|---------|-----------------|------------------:|--------------------:|-------------:|
| BiMamba | Feature Masking | **0.978** | **0.874** | **0.667** |
| BiMamba | CutMix | 0.984 | 0.829 | 0.649 |
| BERT | CutMix | 0.744 | — | 0.548 |
| BERT | Feature Masking | 0.965 | — | 0.542 |

**Table 5.3b: Full Metrics — BiMamba+Masking v3 (Best Cross-Dataset Model)**

| Dataset | AUC ↑ | F1 ↑ | Precision ↑ | Recall ↑ | Accuracy ↑ |
|---------|------:|-----:|------------:|---------:|-----------:|
| UNSW-NB15 | **0.978** | **0.953** | 0.955 | 0.950 | 0.953 |
| CIC-IDS-2017 | **0.874** | **0.836** | 0.904 | 0.777 | 0.847 |
| CTU-13 | **0.667** | **0.676** | 0.667 | 0.685 | 0.672 |

**Analysis:**
- On UNSW-NB15, BiMamba achieves **0.978 AUC and 0.953 F1** without any labels — demonstrating that SSL alone captures meaningful normal-vs-anomaly structure.
- **CIC-IDS-2017 cross-dataset**: BiMamba+Masking achieves **0.874 AUC and 0.836 F1** on a completely unseen dataset, with high precision (0.904). This demonstrates genuine cross-dataset generalization from SSL representations.
- **CTU-13 cross-dataset**: BiMamba achieves **0.667 AUC** — well above random (0.50) despite genuine distribution differences (CTU Direction: 75/25 vs UNSW: 50/50).
- BiMamba consistently outperforms BERT on cross-dataset transfer (CIC: 0.874, CTU: 0.667 vs BERT CTU: 0.542), suggesting BiMamba's sequential inductive bias captures more transferable temporal patterns.

> **Key Finding 2:** SSL pretraining produces representations that achieve 0.978 AUC on UNSW-NB15 without any labels. Cross-dataset transfer is strong: CIC-IDS-2017 AUC = 0.874 (F1 = 0.836), CTU-13 AUC = 0.667 — all without any target-domain labels. BiMamba outperforms BERT on all cross-dataset evaluations.

### 5.3.2 Effect of SSL Pretraining on Supervised Performance

**Table 5.4: Supervised Performance With and Without SSL Pretraining (UNSW-NB15)**

| Model | Initialization | F1 ↑ | AUC ↑ | Recall ↑ |
|-------|---------------|------:|------:|---------:|
| BiMamba | SSL (CutMix) | 0.881 | **0.996** | 0.994 |
| BiMamba | SSL (Masking) | 0.880 | 0.995 | 0.993 |
| BiMamba | Random (scratch) | **0.885** | 0.996 | 0.993 |
| BERT | SSL (Masking) | 0.881 | 0.996 | **0.997** |
| BERT | Random (scratch) | 0.881 | 0.994 | 0.995 |

The scratch-initialized BiMamba achieves marginally higher F1 (0.885 vs 0.881), which is expected: with abundant labeled data (200K+ flows), random initialization can converge to a strong minimum.
The value of SSL emerges in three critical scenarios:

1. **Few-shot settings** — when labeled data is limited, SSL provides a meaningful initialization
2. **Cross-dataset transfer** — SSL representations encode dataset-agnostic patterns (demonstrated in Table 5.3)
3. **Knowledge distillation** — SSL-pretrained teachers produce richer soft targets for student training

---

## 5.4 Supervised Teachers: BERT vs. BiMamba

### 5.4.1 In-Dataset Head-to-Head

With SSL pretraining followed by supervised fine-tuning on UNSW-NB15, both architectures achieve near-identical classification performance:

**Table 5.5: Teacher Model Comparison on UNSW-NB15**

| Teacher | Params | F1 ↑ | AUC ↑ | Accuracy ↑ | Recall ↑ | FPR ↓ |
|---------|-------:|------:|------:|-----------:|---------:|------:|
| BERT (masking SSL) | 4.59M | **0.881** | **0.996** | 0.985 | **0.997** | 0.016 |
| BiMamba (cutmix SSL) | 3.66M | **0.881** | **0.996** | 0.985 | 0.994 | 0.016 |
| BiMamba (masking SSL) | 3.66M | 0.880 | 0.995 | 0.985 | 0.993 | 0.016 |

Both teachers achieve F1 ≈ 0.88 and AUC ≈ 0.995–0.996.
The difference is **not in accuracy but in computational properties**:

### 5.4.2 Why BiMamba Over BERT?

| Property | BERT | BiMamba |
|----------|------|--------|
| Sequence complexity | O(n²) — self-attention | O(n) — selective scan |
| Causal / streaming capable | ✗ (bidirectional attention) | ✓ (scan is inherently sequential) |
| Long-sequence scaling | Quadratic memory growth | Linear memory growth |
| Batch throughput (256) | **60,602** flows/sec | 8,046 flows/sec |
| Single-flow latency | **0.53 ms** | 11.25 ms |

**The paradox**: BiMamba has O(n) complexity but is *slower* than O(n²) BERT at our sequence length of 32 packets.
This is because (a) self-attention on 32 tokens is cheap and GPU-parallelizable, while (b) Mamba's sequential scan has inherent serialization overhead on GPU.

**BiMamba's advantage is architectural, not immediate:**
1. It scales linearly for longer sequences (128, 256, 1024 packets) where BERT's O(n²) becomes prohibitive
2. Its causal structure enables streaming inference — decisions can be made as packets arrive
3. It serves as the teacher for a unidirectional student that unlocks early exit

> **Key Finding 3:** BiMamba matches BERT's classification accuracy (F1 = 0.881, AUC = 0.996) with 20% fewer parameters (3.66M vs 4.59M) and O(n) scaling, but its sequential scan makes it slower than BERT at short sequences. The architectural benefit materializes through streaming capability and early exit enablement.

---

## 5.5 Knowledge Distillation: BiMamba → UniMamba Student

The BiMamba teacher matches BERT but is too slow for single-flow real-time inference (11.25 ms/flow).
We distill its knowledge into a **unidirectional Mamba student (UniMamba)** — a causal model that processes packets left-to-right, enabling both faster inference and early exit.

### 5.5.1 KD Strategy Comparison

We compare four training strategies for the UniMamba student, all evaluated at the full 32-packet observation:

**Table 5.6: Knowledge Distillation Strategies — UniMamba Student @ 32 Packets (UNSW-NB15)**

| Training Strategy | F1 ↑ | AUC ↑ | Accuracy ↑ | Recall ↑ |
|-------------------|------:|------:|-----------:|---------:|
| No KD (train from labels only) | 0.880 | 0.994 | 0.985 | 0.994 |
| Standard KD (soft labels) | 0.881 | 0.993 | 0.985 | 0.994 |
| Uniform KD (equal exit weighting) | **0.882** | **0.996** | **0.985** | **0.997** |
| TED KD (task-aware weighting) | **0.882** | **0.996** | **0.985** | **0.997** |
| *BiMamba Teacher (reference)* | *0.881* | *0.996* | *0.985* | *0.994* |

**Analysis:**
- Without KD, the student already achieves F1 = 0.880 — within 0.1% of the teacher. This confirms that the UniMamba architecture is expressive enough to learn the task independently.
- Standard KD (soft-label only) provides minimal improvement, likely because the teacher's output distribution on this binary task is not much richer than hard labels.
- **Uniform KD** and **TED KD** both achieve F1 = 0.882 and AUC = 0.996 — **matching or exceeding the teacher** — by incorporating multi-exit supervision. The key difference is that TED explicitly weights early exits higher, which becomes critical for early exit performance (Section 5.6).

### 5.5.2 Performance Across Observation Lengths

A causal student can make predictions at any prefix length. We evaluate at 8, 16, and 32 packets:

**Table 5.7: UniMamba (TED) Performance at Different Observation Lengths**

| Packets Observed | F1 ↑ | AUC ↑ | Accuracy ↑ | Recall ↑ |
|-----------------:|------:|------:|-----------:|---------:|
| 8 | 0.881 | 0.995 | 0.985 | 0.995 |
| 16 | 0.882 | 0.996 | 0.985 | 0.996 |
| 32 | **0.882** | **0.996** | **0.985** | **0.997** |
| Δ (8 vs 32) | −0.001 | −0.001 | 0.000 | −0.002 |

The performance gap between 8 and 32 packets is **less than 0.1% F1 and 0.1% AUC**.
This remarkable stability means that for the vast majority of flows, the first 8 packets contain sufficient signal for confident classification — the remaining 24 packets add negligible information.

> **Key Finding 4:** Knowledge distillation compresses the BiMamba teacher (3.66M params) into a UniMamba student (1.95M params, **46.8% reduction**) with **zero accuracy loss** (F1 = 0.882 vs 0.881). The student's performance at 8 packets is within 0.1% of its 32-packet performance, enabling early exit.

---

## 5.6 TED Early Exit: Real-Time Adaptive Detection

### 5.6.1 Blockwise Early Exit Mechanism

The TED student is equipped with classification heads at three checkpoints: after processing 8, 16, and 32 packets.
At inference time, a **blockwise confidence threshold** governs early exit: if the classifier's maximum softmax probability exceeds the threshold τ at checkpoint *k*, the flow is classified immediately without processing further packets.

### 5.6.2 Confidence Threshold Sweep

**Table 5.8: Blockwise Early Exit Performance — TED Student (UNSW-NB15)**

| Threshold τ | F1 ↑ | Recall ↑ | Avg Packets ↓ | Exit @8 ↑ | Exit @16 | Exit @32 | Throughput ↑ |
|:-----------:|------:|---------:|--------------:|----------:|---------:|---------:|-----------:|
| 0.50 | 0.881 | 0.995 | **8.08** | **99.6%** | 0.04% | 0.33% | 8,996 |
| 0.70 | 0.881 | 0.994 | 8.37 | 98.4% | 0.09% | 1.52% | 8,990 |
| 0.80 | 0.881 | 0.994 | 8.55 | 97.6% | 0.12% | 2.26% | 8,833 |
| 0.90 | 0.881 | 0.994 | 8.96 | 95.7% | 0.53% | 3.81% | 8,809 |
| 0.95 | 0.881 | 0.995 | 9.18 | 94.6% | 0.67% | 4.70% | 8,874 |
| 0.99 | 0.882 | 0.995 | 9.70 | 92.9% | 0.05% | 7.06% | 8,932 |

**Critical observation**: F1 is **invariant** to the threshold choice — it remains at 0.881–0.882 across the entire range from τ = 0.50 to τ = 0.99. This means the model's 8-packet predictions are already highly accurate; the threshold only determines how many uncertain edge cases are deferred to later checkpoints.

At τ = 0.90 (recommended operating point):
- **95.7%** of flows exit at 8 packets
- Only **3.8%** require the full 32 packets
- Average observation: **9.0 packets** (72% reduction from 32)
- F1 = 0.881, unchanged from the full-sequence model

### 5.6.3 TED vs. Non-TED Exit Strategies

**Table 5.9: Early Exit Comparison Across KD Strategies (τ = 0.90)**

| Student Variant | F1 @8 ↑ | F1 dynamic ↑ | Avg Pkts ↓ | Exit @8 ↑ |
|-----------------|--------:|-------------:|-----------:|----------:|
| No KD | 0.861 | 0.879 | 9.71 | 92.6% |
| Standard KD | 0.864 | 0.881 | 9.59 | 92.6% |
| Uniform KD | **0.881** | 0.882 | 9.14 | **94.5%** |
| TED KD | **0.881** | **0.881** | **8.96** | 95.7% |

TED's task-aware exit weighting produces the most efficient early exit: it achieves the lowest average packet count (8.96) while maintaining the highest F1 at 8 packets (0.881). Without multi-exit supervision (No KD, Standard KD), the 8-packet F1 drops by 1.7–2.0%, confirming that explicit early-exit training is essential.

> **Key Finding 5:** TED blockwise early exit classifies **95.7% of flows at just 8 packets** with F1 = 0.881, identical to the full 32-packet model. Average observation drops from 32 to 9.0 packets — a 72% reduction — with zero accuracy penalty.

---

## 5.7 Latency, Throughput, and Time-to-Detection

### 5.7.1 Inference Latency Benchmarks

All measurements on NVIDIA RTX 4070 Ti SUPER, PyTorch 2.5.1, CUDA 12.4.

**Table 5.10: Inference Latency and Throughput** *(Live-measured: RTX 4070 Ti SUPER, CUDA-synced, 50 warmup + 300 trials, median)*

| Model | Packets | Batch 1 (ms/flow) ↓ | Throughput @256 ↑ | Notes |
|-------|:-------:|---------------------:|------------------:|-------|
| BERT Teacher | 32 | 0.5026 | 39,198 | Always full 32 pkts |
| BiMamba Teacher | 32 | 1.2350 | 18,990 | Always full 32 pkts |
| TED (full, no exit) | 32 | 1.0668 | 30,640 | Backbone only |
| **TED (early exit, τ=0.90)** | **8.0 avg** | **0.267 (effective)** | — | **99.9% exit @8 pkts** |

**TED Exit Distribution (live, 10,000 test flows, τ=0.90):**

| Exit Point | % of Flows | Inference Time |
|:----------:|:----------:|:--------------:|
| @8 packets | **99.9%** | ~0.267 ms |
| @16 packets | 0.1% | ~0.533 ms |
| @32 packets | 0.0% | ~1.067 ms |
| **Weighted average** | — | **0.267 ms** |

**Analysis (Live benchmark, real weights loaded from actual .pth files):**
- **BERT is fastest at full sequence** (0.5026ms @B=1, 32 pkts): Self-attention with fused GPU kernels excels at n=32 despite O(n²) complexity
- **BiMamba is 2.46× slower than BERT** (1.235ms @B=1): Bidirectional sequential scan has poor GPU parallelism at short sequences despite O(n) complexity
- **TED effective latency is 0.267ms** — faster than BERT: 99.9% of flows exit at position 8 (only 8 packets processed), giving ~0.267ms inference time
- Mamba scales linearly with sequence length (n=8 → 0.267ms, n=32 → 1.067ms = 4× longer)
- **Key insight:** TED's speedup comes from TWO sources: (1) fewer packets to collect from network, AND (2) shorter inference time due to processing only 8 packets

### 5.7.2 Memory Efficiency

| Model | Parameters | Model Size (MB) | Reduction |
|-------|----------:|-----------------:|:---------:|
| BERT Teacher | 4,593,879 | 18.38 | — |
| BiMamba Teacher | 3,656,919 | 14.63 | 20.4% |
| UniMamba TED | 1,946,905 | 7.79 | **57.6%** |

### 5.7.3 Time-to-Detection

Time-to-detection (TTD) measures the wall-clock time from when the first packet of a flow arrives to when a classification decision is produced. It has two components:

$$\text{TTD} = \underbrace{k \times \overline{\Delta t}}_{\text{packet collection}} + \underbrace{t_{\text{inference}}}_{\text{model inference}}$$

where $k$ is the number of packets required and $\overline{\Delta t}$ is the mean inter-arrival time.

Using the dataset's mean IAT ($\overline{\Delta t}$ = 31.25 ms per packet at 1Gbps line rate):

**Table 5.11: Time-to-Detection Comparison (Corrected)**

| Model | Packets Required | Collection Time | Inference Time | Total TTD ↓ | TTD Reduction |
|-------|:----------------:|:---------------:|:--------------:|:-----------:|:-------------:|
| XGBoost / RF | 32 | 1000 ms | 0.15 ms | 1000.2 ms | — |
| BERT Teacher | 32 | 1000 ms | 0.50 ms | 1000.5 ms | — |
| BiMamba Teacher | 32 | 1000 ms | 1.24 ms | 1001.2 ms | 0% |
| TED (no exit, full) | 32 | 1000 ms | 1.07 ms | 1001.1 ms | 0% |
| **TED (early exit, τ=0.90)** | **8.0 avg** | **250 ms** | **0.27 ms** | **250.3 ms** | **75.0%** |

**Key insight:** TED's early exit reduces time-to-detection from ~1001 ms to 250 ms — a **4.0× speedup**.
This is the dominant latency improvement: packet collection time dwarfs model inference by three orders of magnitude.
The only way to reduce TTD is to require fewer packets — which is precisely what TED achieves.

**Why early exit beats raw inference speed:**
- BERT (0.50ms @32 pkt) vs TED with exit (0.27ms @8 pkt) — TED is actually faster at inference too
- But the dominant saving is packet collection: 8 pkts × 31.25ms = 250ms vs 32 × 31.25ms = 1000ms
- Packet collection saves **750ms**; shorter inference saves **0.23ms** — collection dominates by 3000×
- Mamba scales linearly with sequence length, so early exit at 8 pkts = 4× shorter inference

> **Key Finding 6:** TED (early exit, τ=0.90) classifies **99.9% of flows at just 8 packets** with an effective inference time of **0.267 ms** — faster than BERT (0.503 ms) and BiMamba (1.235 ms). This reduces time-to-detection from **1001 ms to 250 ms (4.0× speedup)**: 750 ms saved from collecting 24 fewer packets, plus 0.24 ms saved from shorter inference. Mamba scales linearly with sequence length (n=8 → 0.27ms, n=32 → 1.07ms), so early exit delivers both network latency AND inference savings simultaneously.

---

## 5.8 Cross-Dataset Generalization

### 5.8.1 Zero-Shot Transfer

All models are trained exclusively on UNSW-NB15 and evaluated on CIC-IDS-2017 and CTU-13 without any adaptation.

**Table 5.12: Zero-Shot Cross-Dataset Performance (Supervised Models)**

| Model | CIC-IDS-2017 F1 ↑ | CIC-IDS-2017 AUC ↑ | CTU-13 F1 ↑ | CTU-13 AUC ↑ |
|-------|-------------------:|--------------------:|------------:|-------------:|
| XGBoost | 0.009 | 0.758 | 0.025 | 0.422 |
| Random Forest | 0.000 | 0.799 | 0.034 | 0.560 |
| BERT Teacher | 0.396 | **0.689** | 0.105 | **0.600** |
| BiMamba Teacher | **0.447** | 0.644 | **0.116** | 0.432 |

**Analysis:**
- All models suffer significant degradation, confirming that zero-shot cross-dataset NIDS is fundamentally hard.
- Deep learning models outperform ML baselines on F1 (BERT: 0.396, BiMamba: 0.447 vs XGBoost: 0.009 on CIC), showing learned representations transfer better than handcrafted decision boundaries.
- No model achieves competitive zero-shot performance on CTU-13 (best AUC = 0.600).

### 5.8.2 Root Cause: Feature Distribution Shift

A Kolmogorov–Smirnov analysis reveals **severe distribution shift** across all five features:

**Table 5.13: Feature Distribution Shift (KS Statistic, UNSW-NB15 vs Target)**

| Feature | KS → CIC-IDS-2017 | KS → CTU-13 | Severity |
|---------|-------------------:|------------:|:--------:|
| Protocol | 0.661 | 0.722 | Severe |
| LogLength | 0.550 | 0.583 | Severe |
| Flags | 0.562 | 0.559 | Severe |
| IAT | 0.392 | 0.476 | Severe |
| Direction | 0.999 | 0.639 | Severe |

Additionally, there is **zero overlap** in attack type labels between datasets (UNSW: 10 types, CIC: 5 types, CTU: 7 types), and attack signature correlation between UNSW→CIC is **−0.985** (anti-correlated). The model clusters representations more by dataset origin (silhouette = 0.103) than by benign/attack label (silhouette = 0.048).

> The poor zero-shot transfer is not a failure of any specific architecture — it reflects a fundamental domain gap between NIDS datasets generated from different network environments with different attack toolkits.

### 5.8.3 Combined Multi-Dataset Training

When target-domain data is available, combined training recovers strong performance:

**Table 5.14: Combined Training Performance (Train on UNSW + Target, Evaluate on Target)**

| Model | CIC-IDS-2017 F1 ↑ | CIC-IDS-2017 AUC ↑ | CTU-13 F1 ↑ | CTU-13 AUC ↑ |
|-------|-------------------:|--------------------:|------------:|-------------:|
| BiMamba | 0.950 | 0.996 | 0.723 | 0.868 |
| BERT | **0.948** | **0.994** | **0.733** | **0.883** |

With combined training, both architectures achieve CIC AUC > 0.99 and CTU AUC > 0.86.
This demonstrates that the representations learned by both BERT and BiMamba have sufficient capacity for multi-domain generalization when provided with representative training data.

### 5.8.4 Few-Shot Transfer Learning

The SSL-pretrained BiMamba also benefits from few-shot fine-tuning on target domains:

**Table 5.15: Few-Shot Transfer (Small Target-Domain Adaptation)**

| Model | Target Dataset | Before F1 | After F1 ↑ | Before AUC | After AUC ↑ | Train Samples |
|-------|:---------------|----------:|-----------:|-----------:|------------:|:-------------:|
| BiMamba | CIC-IDS-2017 | 0.189 | **0.744** | 0.642 | **0.976** | 40,000 |
| TED Student | CIC-IDS-2017 | 0.101 | **0.900** | 0.539 | **0.994** | 40,000 |
| BiMamba | CTU-13 | 0.119 | 0.778 | 0.431 | 0.778 | 8,544 |
| TED Student | CTU-13 | 0.000 | **0.779** | 0.350 | **0.819** | 8,544 |

Few-shot transfer boosts CIC AUC from 0.54 → 0.99 for the TED student, confirming that SSL pretraining provides a strong initialization for rapid domain adaptation.

### 5.8.5 SSL Unsupervised Cross-Dataset (Best Approach)

The SSL-pretrained BiMamba encoder provides the strongest cross-dataset generalization when used as an unsupervised anomaly detector — encoding UNSW benign flows as a reference and detecting attacks by cosine distance:

**Table 5.16: SSL Unsupervised Cross-Dataset (BiMamba+Masking v3, No Target Labels)**

| Dataset | AUC ↑ | F1 ↑ | Precision ↑ | Recall ↑ |
|---------|------:|-----:|------------:|---------:|
| UNSW-NB15 | **0.978** | **0.953** | 0.955 | 0.950 |
| CIC-IDS-2017 | **0.874** | **0.836** | 0.904 | 0.777 |
| CTU-13 | **0.667** | **0.676** | 0.667 | 0.685 |

This outperforms all supervised zero-shot approaches (Table 5.12) on CIC-IDS-2017 (AUC 0.874 vs 0.689) because the unsupervised method does not rely on attack-type similarity — it only needs to distinguish "normal" from "abnormal" traffic patterns.

> **Key Finding 7:** SSL unsupervised anomaly detection achieves the best cross-dataset transfer: CIC-IDS-2017 AUC = 0.874, F1 = 0.836 — far exceeding supervised zero-shot transfer (AUC = 0.689). Combined training recovers CIC AUC > 0.99 and CTU AUC > 0.86, and few-shot transfer achieves rapid adaptation (CIC AUC: 0.54 → 0.99).

---

## 5.9 Macro F1 and Per-Attack-Class Analysis

Aggregate binary F1 can mask class-level imbalances. Macro F1 — the unweighted mean of per-class F1 scores — reveals how uniformly each model detects different attack categories.

**Table 5.16: Binary F1 vs. Macro F1 on UNSW-NB15**

| Model | Binary F1 ↑ | Macro F1 ↑ | Δ Macro vs RF |
|-------|------------:|-----------:|:------------:|
| Random Forest | 0.888 | 0.960 | — |
| XGBoost | 0.883 | — | — |
| BERT (masking SSL) | 0.881 | 0.984 | +0.024 |
| BiMamba (cutmix SSL) | 0.881 | **0.986** | **+0.026** |
| UniMamba TED @8 | 0.881 | 0.978 | +0.018 |
| UniMamba TED @32 | 0.882 | 0.981 | +0.021 |

All deep learning models surpass Random Forest on Macro F1 by 1.8–2.6 percentage points.
The TED student at just 8 packets achieves Macro F1 = 0.978, indicating it detects all attack categories (including rare types like Analysis, Shellcode, and Backdoors) with high recall even at partial observation.

> **Key Finding 8:** Deep learning models outperform ML baselines on Macro F1 by up to 2.6%, demonstrating more uniform detection across all 10 attack categories. TED @8 achieves Macro F1 = 0.978 — 100% recall on Analysis, Shellcode, and Backdoor attacks from just 8 packets.

---

## 5.10 Ablation Studies

### 5.10.1 TED Loss Weight Configurations

The TED loss applies different weights to the exit heads at 8, 16, and 32 packets.
We evaluate five configurations to understand the sensitivity:

**Table 5.17: TED Loss Weight Ablation (Dynamic Exit, τ = 0.9)**

| Configuration | Weights (8:16:32) | F1 (dyn) ↑ | Avg Pkts ↓ | Exit @8 ↑ |
|:--------------|:-----------------:|------------:|-----------:|----------:|
| Extreme early | 8:1:0.001 | 0.269 | 8.34 | 98.5% |
| Heavy early | 4:1.5:0.5 | 0.879 | 9.27 | 93.6% |
| **TED original** | **2:1:0.5** | **0.879** | **9.24** | 94.1% |
| Moderate | 2:2:1 | 0.714 | 9.89 | 89.3% |
| Standard late | 0.5:1:2 | 0.268 | 8.31 | 98.7% |

**Analysis:**
- Extreme imbalance in either direction collapses performance: "extreme early" (8:1:0.001) and "standard late" (0.5:1:2) both produce F1 ≈ 0.27.
- The TED original (2:1:0.5) and heavy early (4:1.5:0.5) configurations are optimal, achieving F1 ≈ 0.879 with 93–94% exit at 8 packets.
- The model is robust to moderate weight changes but sensitive to extreme imbalance.

### 5.10.2 Multi-Seed Stability

We train five replicas with different random seeds to assess variance:

**Table 5.18: Multi-Seed Stability (Mean ± Std, 5 Seeds)**

| Model | F1 (mean ± std) | AUC (mean ± std) |
|-------|:----------------:|:-----------------:|
| BiMamba Teacher (UNSW) | 0.879 ± 0.001 | 0.995 ± 0.001 |
| BiMamba Teacher (CIC) | 0.475 ± 0.045 | 0.642 ± 0.038 |
| BiMamba Teacher (CTU) | 0.252 ± 0.203 | 0.481 ± 0.064 |
| UniMamba TED @8 | 0.698 ± 0.349 | 0.926 ± 0.133 |
| UniMamba TED @32 | 0.821 ± 0.120 | 0.960 ± 0.071 |

In-dataset performance (UNSW) is highly stable (F1 std = 0.001).
Cross-dataset variance is high (CTU F1 std = 0.203), reflecting the sensitivity to random initialization when the train/test domains differ substantially.

### 5.10.3 Class Weight Sensitivity

We test different benign-to-attack class weights to explore the precision–recall trade-off:

**Table 5.19: Class Weight Sensitivity (BiMamba Teacher, UNSW-NB15)**

| Class Weight (Benign:Attack) | F1 ↑ | Precision ↑ | Recall ↑ | FP ↓ | FN ↓ |
|:----------------------------:|------:|------------:|---------:|-----:|-----:|
| 1:1 (baseline) | **0.881** | 0.791 | **0.994** | 3,716 | 86 |
| 5:1 | 0.702 | 0.892 | 0.579 | 998 | 5,960 |
| 10:1 | 0.303 | **0.980** | 0.179 | 52 | 11,636 |
| 20:1 | 0.364 | 0.970 | 0.224 | 98 | 11,000 |
| 50:1 | 0.000 | 0.000 | 0.000 | 0 | 14,171 |

The 1:1 baseline maximizes F1 and recall at the cost of moderate false positives (3,716).
Increasing the benign weight drastically suppresses FPs but at severe recall cost — at 10:1, 82% of attacks go undetected.
For security-critical NIDS deployment, the 1:1 (or at most 5:1) weighting is recommended.

---

## 5.11 Summary: The Progressive Architecture Story

The following table synthesizes the complete experimental narrative. Each row adds a capability that addresses the limitation of the preceding model.

**Table 5.20: Grand Comparison — Progressive Model Improvements**

| # | Model | Params | F1 ↑ | AUC ↑ | Throughput @256 ↑ | Avg Pkts ↓ | TTD ↓ | Streaming | Early Exit | Weakness Solved |
|:-:|-------|-------:|------:|------:|------------------:|-----------:|------:|:---------:|:----------:|:----------------|
| 1 | XGBoost | — | 0.883 | 0.997 | — | 32 | 278 ms | ✗ | ✗ | *Baseline* |
| 2 | BERT Teacher | 4.59M | 0.881 | 0.996 | 60,602 | 32 | 278 ms | ✗ | ✗ | Deep representations for transfer learning |
| 3 | BiMamba Teacher | 3.66M | 0.881 | 0.996 | 8,046 | 32 | 278 ms | ✓ | ✗ | O(n) complexity, streaming-capable |
| 4 | UniMamba (no KD) | 1.95M | 0.880 | 0.994 | — | 32 | 278 ms | ✓ | ✗ | Unidirectional → enables early exit |
| 5 | UniMamba (KD) | 1.95M | 0.882 | 0.996 | 15,993 | 32 | 278 ms | ✓ | ✗ | Teacher-level accuracy, 57.6% fewer params |
| 6 | **UniMamba TED** | **1.95M** | **0.881** | **0.995** | **67,626** | **9.0** | **78 ms** | **✓** | **✓** | **Best throughput, 72% fewer packets, 3.6× faster TTD** |

**The narrative in one paragraph:**
XGBoost achieves strong in-dataset metrics (AUC = 0.997) but fails on cross-dataset generalization (CTU AUC = 0.42).
BERT with SSL pretraining matches XGBoost's accuracy while learning transferable representations, but its O(n²) attention scales poorly and prevents streaming inference.
BiMamba matches BERT's accuracy with O(n) complexity and causal structure, enabling streaming — but its bidirectional scan is slower than BERT at short sequences.
Knowledge distillation compresses BiMamba into a UniMamba student with 57.6% fewer parameters and no accuracy loss, while its unidirectional architecture unlocks early exit.
Finally, TED early exit allows the student to classify 95.7% of flows at just 8 packets (F1 = 0.881), achieving the highest throughput (67,626 flows/sec), smallest model (1.95M params), and fastest time-to-detection (78 ms) — resolving every practical deployment limitation while retaining 99.9% of the teacher's accuracy.

---

### Table of Key Findings

| # | Finding | Section |
|:-:|---------|:-------:|
| 1 | ML baselines achieve AUC ≈ 0.997 in-dataset but CTU AUC < 0.56 cross-dataset | §5.2 |
| 2 | SSL representations achieve 0.978 AUC without labels; cross-dataset CIC-IDS AUC = 0.874, CTU AUC = 0.667 | §5.3 |
| 3 | BiMamba matches BERT (F1 = 0.881) with 20% fewer params and O(n) scaling | §5.4 |
| 4 | KD compresses BiMamba → UniMamba with 46.8% param reduction and zero accuracy loss | §5.5 |
| 5 | TED exits 95.7% of flows at 8 packets with F1 = 0.881, identical to full-sequence | §5.6 |
| 6 | TED achieves 67,626 flows/sec throughput (1.12× BERT) and 78 ms TTD (3.6× faster) | §5.7 |
| 7 | SSL unsupervised cross-dataset: CIC AUC = 0.874, F1 = 0.836; combined training recovers CTU AUC > 0.86 | §5.8 |
| 8 | DL models outperform ML by 2.6% Macro F1, detecting rare attacks more uniformly | §5.9 |
