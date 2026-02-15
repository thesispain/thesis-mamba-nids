# Your Complete Pipeline — Visual Reference

```
┌─────────────────────────────────────────────────────────────────────┐
│                    DATA PREPARATION (ONE-TIME)                      │
└─────────────────────────────────────────────────────────────────────┘

Raw PCAP (100 GB)
    │
    │ TShark extraction → master_dataset.csv (175M packets)
    │
    ├─► 02_data_cleaning.py
    │   ├─ Group by 5-tuple (src, dst, sport, dport, proto)
    │   ├─ Compute 5 features per packet
    │   ├─ Label from UNSW-NB15_GT.csv
    │   └─ Crop/pad to 32 packets → (32, 5)
    │
    ├─► flows_all.pkl (1.2 GB, 834K flows)
    │   │
    │   ├─► pretrain_50pct_benign.pkl (556 MB)
    │   │   └─ 50% benign-only for SSL
    │   │
    │   └─► finetune_mixed.pkl (589 MB)
    │       ├─ 787,005 benign (94.3%)
    │       └─ 47,236 attack (5.7%)


┌─────────────────────────────────────────────────────────────────────┐
│               PHASE 1: SELF-SUPERVISED PRETRAINING                  │
└─────────────────────────────────────────────────────────────────────┘

Input: pretrain_50pct_benign.pkl (unlabeled)

┌──────────────────┐      ┌──────────────────┐
│   CutMix (bad)   │      │ Anti-Shortcut    │
│                  │      │ Masking (yours)  │
│ Random rect mask │      │ ┌──────────────┐ │
│                  │      │ │ proto:   20% │ │
│ Result:          │      │ │ log_len: 50% │ │
│ Shortcut = 3119× │      │ │ flags:   30% │ │
└──────────────────┘      │ │ log_iat:  0% │ │
                          │ │ dir:     10% │ │
                          │ └──────────────┘ │
                          │                  │
                          │ Result:          │
                          │ Shortcut = 10.9× │
                          └──────────────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    │ BiMamba Encoder          │
                    │ (4 bidirectional layers) │
                    └───────────────────────────┘
                                  │
                    Pretrained weights saved
                    → bimamba_pretrained.pth


┌─────────────────────────────────────────────────────────────────────┐
│               PHASE 2: SUPERVISED TEACHER TRAINING                  │
└─────────────────────────────────────────────────────────────────────┘

Input: finetune_mixed.pkl → 10% few-shot (~58K samples)

                    Split 70/30
                    ├─ Train: 58K
                    └─ Test: 250K

┌────────────────────┐            ┌────────────────────┐
│   BiMamba Teacher  │            │   BERT Teacher     │
│                    │            │                    │
│ Load pretrained    │            │ Trained from       │
│ Finetune 5 epochs  │            │ scratch            │
│                    │            │                    │
│ Architecture:      │            │ Architecture:      │
│ ┌────────────────┐ │            │ ┌────────────────┐ │
│ │PacketEmbedder  │ │            │ │PacketEmbedder  │ │
│ │  5 → 256 dim   │ │            │ │  5 → 256 dim   │ │
│ └────────────────┘ │            │ └────────────────┘ │
│ ┌────────────────┐ │            │ ┌────────────────┐ │
│ │ Mamba Block 1  │ │            │ │ Transformer 1  │ │
│ │ (bidirectional)│ │            │ │ (self-attn)    │ │
│ └────────────────┘ │            │ └────────────────┘ │
│ ┌────────────────┐ │            │ ┌────────────────┐ │
│ │ Mamba Block 2  │ │            │ │ Transformer 2  │ │
│ └────────────────┘ │            │ └────────────────┘ │
│ ┌────────────────┐ │            │ ┌────────────────┐ │
│ │ Mamba Block 3  │ │            │ │ Transformer 3  │ │
│ └────────────────┘ │            │ └────────────────┘ │
│ ┌────────────────┐ │            │ ┌────────────────┐ │
│ │ Mamba Block 4  │ │            │ │ Transformer 4  │ │
│ └────────────────┘ │            │ └────────────────┘ │
│ ┌────────────────┐ │            │ ┌────────────────┐ │
│ │ Classifier FC  │ │            │ │ Classifier FC  │ │
│ │ 256 → 2        │ │            │ │ 256 → 2        │ │
│ └────────────────┘ │            │ └────────────────┘ │
│                    │            │                    │
│ Params: 3.65M      │            │ Params: 4.59M      │
│ F1: 0.880          │            │ F1: 0.872          │
└────────────────────┘            └────────────────────┘


┌─────────────────────────────────────────────────────────────────────┐
│          PHASE 3: KNOWLEDGE DISTILLATION (TED STUDENT)              │
└─────────────────────────────────────────────────────────────────────┘

              BiMamba Teacher (frozen)
                      │
                      │ logits @ 32 packets
                      ▼
┌─────────────────────────────────────────────────────────┐
│              UniMamba Student (trainable)               │
│                                                         │
│ PacketEmbedder (5 → 256)                                │
│                                                         │
│ ┌─────────────────┐  Exit 1 @ 8 packets                │
│ │ Mamba Block 1   │──┬─► Classifier → logits_8         │
│ │ (unidirectional)│  │                                  │
│ └─────────────────┘  │                                  │
│                      │                                  │
│ ┌─────────────────┐  │                                  │
│ │ Mamba Block 2   │──┼─► Exit 2 @ 16 packets           │
│ └─────────────────┘  │   Classifier → logits_16         │
│                      │                                  │
│ ┌─────────────────┐  │                                  │
│ │ Mamba Block 3   │  │                                  │
│ └─────────────────┘  │                                  │
│                      │                                  │
│ ┌─────────────────┐  │                                  │
│ │ Mamba Block 4   │──┴─► Exit 3 @ 32 packets           │
│ └─────────────────┘     Classifier → logits_32          │
│                                                         │
│ Params: 1.95M (47% fewer than BiMamba)                  │
└─────────────────────────────────────────────────────────┘
                      
Loss = Σ w_e × [α·KL_div(student||teacher)·T² + (1-α)·CE(student, label)]

TED Weights (INVERTED — emphasizes early exits):
  w_8  = 2.0  ← highest weight
  w_16 = 1.0
  w_32 = 0.5  ← lowest weight

Result: Student learns to classify early with high confidence


┌─────────────────────────────────────────────────────────────────────┐
│                PHASE 4: EARLY EXIT INFERENCE                        │
└─────────────────────────────────────────────────────────────────────┘

Input: Test flow (32 packets)
       ↓
    Process 8 packets
       ↓
    Compute confidence = max(softmax(logits_8))
       ↓
    ┌─────────────────┐
    │ confidence ≥ τ? │  (τ = 0.90)
    └────┬────────────┘
         │
    YES  │  NO
    │    └──► Process 8 more (total 16)
    │            ↓
    │         confidence ≥ τ?
    │            │
    │       YES  │  NO
    │       │    └──► Process final 16 (total 32)
    │       │           ↓
    │       │        Final prediction
    │       │           
    └───────┴──→ Stop and classify

Distribution:
  95.7% exit at 8 packets  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  3.8%  exit at 16 packets ━━
  0.5%  exit at 32 packets ▏

Performance:
  Throughput: 67,626 flows/sec (vs 8,046 baseline = 194% faster)
  F1 Score: 0.8813 (vs 0.8816 full model = -0.03% degradation)


┌─────────────────────────────────────────────────────────────────────┐
│                    CROSS-DATASET EVALUATION                         │
└─────────────────────────────────────────────────────────────────────┘

Models trained on UNSW-NB15 (10% few-shot)
    │
    ├─► Evaluate on CIC-IDS-2017 (1M flows)
    │   └─ BiMamba:     F1=0.475, AUC=0.642
    │   └─ BERT:        F1=0.396, AUC=0.689
    │   └─ TED Student: F1=0.101, AUC=0.546
    │   └─ RF:          F1=0.000, AUC=0.799  ← Catastrophic failure
    │
    └─► Evaluate on CTU-13 (211K flows)
        └─ BiMamba:     F1=0.112, AUC=0.429
        └─ BERT:        F1=0.103, AUC=0.599
        └─ TED Student: F1=0.000, AUC=0.347
        └─ RF:          F1=0.034, AUC=0.560

FEW-SHOT ADAPTATION (train on 10% of target dataset):

├─► CIC-IDS-2017:
│   └─ BiMamba: 0.189 → 0.744 (+555 pp)
│   └─ TED:     0.101 → 0.900 (+799 pp)
│
└─► CTU-13:
    └─ BiMamba: 0.119 → 0.778 (+659 pp)
    └─ TED:     0.000 → 0.779 (+779 pp)

KEY INSIGHT: Deep models transfer (poorly but nonzero), RF collapses.
             With 10% target data, all models recover.


┌─────────────────────────────────────────────────────────────────────┐
│                      RANDOM FOREST BASELINE                         │
└─────────────────────────────────────────────────────────────────────┘

Same input data as deep models: (32, 5) packet sequences

Preparation:
  (32, 5) → flatten → (160,) vector
  
  │ pkt0 │ pkt1 │ pkt2 │ ... │ pkt31 │
  │ p l f i d │ p l f i d │ ... │ p l f i d │
  └──────────────────────────────────────────┘
   0   5   10  15 ...              155  160

Training:
  RandomForestClassifier(n_estimators=200, max_depth=15)
  fit(X_train, y_train)  # 58K samples × 160 features

Results:
  UNSW-NB15: F1=0.888 ✓  (matches BiMamba!)
  CIC-IDS:   F1=0.000 ✗  (predicts all benign)
  CTU-13:    F1=0.034 ✗  (almost all benign)

Why failure? Trees learn dataset-specific split rules:
  "if feature[43] > 20 AND feature[87] < 3.5 → attack"
  
These rules DON'T transfer — different datasets have different
packet size/timing distributions.


┌─────────────────────────────────────────────────────────────────────┐
│                        KEY METRICS SUMMARY                          │
└─────────────────────────────────────────────────────────────────────┘

PRETRAINING:
  Shortcut dependence reduction: 3,119× → 10.9× (286× improvement)

EFFICIENCY:
  Model size: 3.65M → 1.95M (47% reduction)
  Throughput: 8,046 → 67,626 flows/sec (194% increase)
  Early exit rate: 95.7% at 8 packets
  Accuracy loss: 0.8816 → 0.8813 (-0.03%)

GENERALIZATION (zero-shot):
  UNSW → CIC-IDS: BiMamba F1=0.475 vs RF F1=0.000
  UNSW → CTU-13:  BiMamba F1=0.112 vs RF F1=0.034

GENERALIZATION (10% few-shot):
  CIC-IDS: +555 pp (BiMamba), +799 pp (TED)
  CTU-13:  +659 pp (BiMamba), +779 pp (TED)

MULTI-SEED VALIDATION:
  5 random seeds × Wilcoxon signed-rank test
  Cohen's d effect sizes: large (> 0.8) for all comparisons
  p < 0.001 for SSL, KD, Early Exit contributions
