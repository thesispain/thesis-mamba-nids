# BERT Supervised Zero-Shot Fine-Tuning Plan
**Goal**: Improve BERT's cross-dataset generalization (fix CIC AUC collapse from 0.627 → 0.85+)  
**Created**: February 24, 2026  
**Status**: PLAN (Not yet implemented)

---

## 🚨 Problem Statement

### Current Performance (SSL k-NN Baseline):
| Model | UNSW AUC | CIC AUC (zero-shot) | Status |
|-------|----------|---------------------|--------|
| **BERT @32** | 0.9780 | **0.6274** | ❌ SEVERE OVERFIT |
| **UniMamba @8** | 0.9827 | **0.9199** | ✅ Generalizes well |

**Critical Issue**: BERT overfits to UNSW training distribution and **collapses on CIC-IDS-2017** (+29 percentage point gap!)

**Hypothesis**: BERT's attention mechanism memorizes UNSW-specific patterns. Need supervised fine-tuning with **strong regularization** to improve generalization.

---

## 📐 Architecture

### Model Configuration:
```
Input: [batch_size, 32 packets, 5 features]
  ↓
PacketEmbedder(d=256, de=32)
  ↓
BERT Encoder (4 layers, 4 heads, d=256) ← SSL pretrained, initially frozen
  ↓ [CLS token representation]
  ↓
Classifier Head:
  - Linear(256 → 128) + ReLU + Dropout(0.3)
  - Linear(128 → 1) + Sigmoid
  ↓
Output: Binary prediction (benign=0, attack=1)
```

**Total Parameters**:
- BERT backbone: 3.2M (pretrained SSL)
- Classifier head: ~33K (randomly initialized)

---

## 🎯 Training Strategy

### Phase A: Warm-up (5 epochs)
**Purpose**: Adapt classifier without destroying SSL representations

```python
Stage: "classifier_only"
Frozen: BERT layers 0-2 (bottom 3 layers)
Trainable: BERT layer 3 (top layer) + classifier head
Learning rate: 1e-4
Batch size: 128
```

**Why freeze bottom layers?**  
SSL learned robust low-level packet features (protocol, flags). Only adapt high-level attack patterns.

### Phase B: Full Fine-Tune (10 epochs max)
**Purpose**: Adapt entire model, but STOP EARLY to prevent overfit

```python
Stage: "full_finetune"
Frozen: None
Trainable: All layers
Learning rate: 5e-5 (50% reduction!)
Early stopping: patience=3 (stop if val_loss doesn't improve)
Target val AUC: 0.95 (NOT 0.99! Intentionally underfit!)
```

---

## 🛡️ Anti-Overfitting Techniques

### 1. **Dropout** (Force Redundancy)
```python
nn.Dropout(p=0.3)  # Applied in classifier head
```
Prevents memorization by randomly disabling 30% of neurons during training.

### 2. **Weight Decay** (L2 Regularization)
```python
optimizer = torch.optim.AdamW(params, lr=5e-5, weight_decay=1e-2)
```
Penalizes large weights → forces simpler decision boundaries.

### 3. **Label Smoothing**
```python
# Instead of hard labels [0, 1]
# Use soft labels [0.05, 0.95]
loss = BCEWithLogitsLoss(reduction='none')
smoothed_labels = labels * 0.9 + 0.05
```
Prevents overconfident predictions on training data.

### 4. **Gradient Clipping**
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```
Stabilizes training, prevents sudden weight updates.

### 5. **Mixup Augmentation**
```python
# Mix two flows: λ*flow_A + (1-λ)*flow_B
# Mix labels:    λ*label_A + (1-λ)*label_B
lam = np.random.beta(0.2, 0.2)  # α=0.2
```
Forces model to learn smooth interpolations → better generalization.

### 6. **Early Stopping**
```python
best_val_auc = 0
patience = 3
for epoch in range(15):
    if val_auc > best_val_auc:
        best_val_auc = val_auc
        save_checkpoint()
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= 3:
            break  # STOP EARLY!
```

**Target**: Stop training at **UNSW val AUC = 0.95** instead of pushing to 0.99.

---

## 📊 Evaluation Protocol

### Training/Validation:
- **Dataset**: UNSW-NB15 only
- **Split**: 80% train (80K flows) / 20% val (20K flows)
- **Balance**: 50/50 benign/attack in both splits
- **Metric**: Monitor **validation AUC** (stop at 0.95)

### Zero-Shot Test (Critical!):
- **Dataset**: CIC-IDS-2017 (1.08M flows, NEVER seen in training!)
- **Evaluation**: Binary classification (benign vs attack)
- **Metrics**: AUC, F1, Precision, Recall

### Per-Attack Type Analysis:
Test on individual attack families in CIC:
```
- DDoS
- DoS (GoldenEye, Hulk, Slowloris, etc.)
- PortScan
- FTP-Patator
- SSH-Patator
- Web attacks
- Infiltration
- Bot
```

**Goal**: Show BERT generalizes to **unseen attack types** from different dataset distribution.

---

## 🧪 Checkpoint Strategy

Save 3 versions and compare:

| Checkpoint | Description | Expected UNSW AUC | Expected CIC AUC |
|------------|-------------|-------------------|------------------|
| **epoch_5** | Warm-up complete | 0.92 | 0.75 |
| **best_val** | Early stopped | 0.95 | **0.85** ← target |
| **epoch_10** | Full training | 0.99 | 0.68 (overfit!) |

**Prediction**: `best_val` checkpoint will have **lower UNSW AUC** but **higher CIC AUC** than SSL baseline!

---

## 📈 Success Criteria

### Minimum Viable:
- ✅ CIC AUC improves from **0.627 → 0.75+** (usable in production)
- ✅ UNSW AUC stays **> 0.93** (acceptable degradation)

### Excellent Result:
- ✅ CIC AUC reaches **0.85+** (close to UniMamba's 0.92)
- ✅ UNSW AUC stays **> 0.95**

### Best Case:
- ✅ CIC AUC **> 0.90** (matches UniMamba!)
- ✅ Maintains BERT's **1.3x throughput advantage**
- ✅ Thesis argument: "BERT + proper regularization = fast AND accurate!"

---

## 🔬 Experimental Variants

If initial approach doesn't reach CIC AUC > 0.85, try:

### Variant A: Domain-Adversarial Training
```python
# Add adversarial loss: force encoder to NOT predict dataset source
domain_loss = -BCELoss(domain_pred, dataset_label)  # Gradient reversal
total_loss = task_loss + 0.1 * domain_loss
```
Forces BERT to learn **dataset-agnostic** features.

### Variant B: Multi-Dataset Training
```python
# Train on BOTH UNSW + CIC (but still test zero-shot on CTU-13)
train_data = unsw_train + cic_train  # Mixed
test_data = ctu13  # Completely unseen
```
Tests if BERT needs diverse training to generalize.

### Variant C: Contrastive Loss Only
```python
# Skip classification head, use contrastive learning
loss = SupConLoss(features, labels)  # Supervised contrastive
# Then use k-NN for zero-shot test (like current SSL baseline)
```
Keeps representation-learning approach but with supervision.

---

## ⚡ Implementation Files

### Scripts to Create:
1. **`train_bert_supervised_zeroshot.py`**  
   - Main training loop (Phase A → Phase B)
   - Checkpointing at epochs 5, best_val, 10
   - Anti-overfitting techniques applied

2. **`eval_bert_zeroshot_checkpoints.py`**  
   - Load all 3 checkpoints
   - Test on CIC-IDS-2017 (zero-shot)
   - Per-attack type breakdown

3. **`compare_bert_tuned_vs_unimamba.py`**  
   - Side-by-side: SSL BERT vs Fine-tuned BERT vs UniMamba
   - Metrics: AUC (UNSW + CIC), Latency, Throughput, TTD
   - Final verdict table

---

## 📦 Expected Outputs

### Training Artifacts:
```
weights/supervised_zeroshot/
├── bert_epoch5_warmup.pth          # After Phase A
├── bert_best_val.pth                # Early stopped (main result!)
├── bert_epoch10_full.pth            # Overfit comparison
└── training_curves.json             # Loss/AUC over epochs
```

### Result Files:
```
results/
├── bert_zeroshot_eval.json          # Per-checkpoint CIC performance
├── bert_per_attack_cic.txt          # Attack-type breakdown
└── bert_tuned_vs_unimamba.json      # Final comparison
```

---

## 🎓 Thesis Defense Strategy

### If BERT Fine-Tuned Matches UniMamba Accuracy:

**Argument Structure**:
```
1. Baseline Comparison:
   - BERT SSL: Fast (42K fps) but overfit (CIC AUC=0.63)
   - UniMamba: Moderate speed (32K fps), good generalization (CIC AUC=0.92)

2. BERT Improvement:
   - Fine-tuned BERT: Fast (42K fps) AND generalizes (CIC AUC=0.85)
   - Shows supervised learning fixes Transformer overfitting

3. UniMamba Unique Advantage:
   - EVEN IF BERT matches accuracy, UniMamba has:
     ✅ 4.4x faster TTD (70ms vs 310ms) — cannot be replicated!
     ✅ 1.8x smaller memory (6.9MB vs 12.2MB)
     ✅ True early exit (BERT needs full 32 packets)

4. Defense Position:
   "UniMamba's contribution is NOT just accuracy—it's the ONLY architecture
    enabling true causal early exit. This 240ms TTD improvement is critical
    for time-sensitive IDS scenarios where attack propagation is exponential."
```

### Anticipated Questions:

**Q1**: "If fine-tuned BERT is as accurate, why use UniMamba?"  
**A1**: "BERT requires 32 packets (310ms). UniMamba decides at 8 packets (70ms). For fast attacks like SYN flood, this 240ms can mean blocking at 1000 flows vs 10,000 flows—difference between minor nuisance and network outage."

**Q2**: "Is 25% throughput loss worth it?"  
**A2**: "Yes. Even UniMamba's 32K fps exceeds typical network loads (most enterprise networks < 10K fps). The 4.4x TTD improvement is MORE valuable than excess throughput."

**Q3**: "Can BERT do early exit too?"  
**A3**: "No. Transformers use bidirectional attention—CLS token at position 0 depends on ALL future packets. Mamba's causal SSM only depends on past, enabling true streaming inference."

---

## 📅 Timeline

| Step | Task | Duration | Output |
|------|------|----------|--------|
| 1 | Prepare UNSW labeled splits (train/val) | 5 min | `unsw_supervised_splits.pkl` |
| 2 | Train Phase A (warm-up, 5 epochs) | 15 min | `bert_epoch5_warmup.pth` |
| 3 | Train Phase B (early stop ~7 epochs) | 20 min | `bert_best_val.pth` |
| 4 | Train Phase B (full, 10 epochs) | 30 min | `bert_epoch10_full.pth` |
| 5 | Eval all checkpoints on CIC (zero-shot) | 10 min | `bert_zeroshot_eval.json` |
| 6 | Per-attack breakdown (11 attack types) | 5 min | `bert_per_attack_cic.txt` |
| 7 | Final comparison benchmark | 5 min | `bert_tuned_vs_unimamba.json` |
| **Total** | | **~90 minutes** | |

---

## 🔧 Implementation Pseudocode

### `train_bert_supervised_zeroshot.py`

```python
# ── PHASE A: WARM-UP ──────────────────────────────────────
bert = BERTEncoder(d=256, n_heads=4, n_layers=4)
bert.load_state_dict(torch.load('ssl_bert_paper.pth'))

classifier = nn.Sequential(
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(128, 1)
)

# Freeze bottom layers
for param in bert.layers[:3].parameters():
    param.requires_grad = False

optimizer = torch.optim.AdamW([
    {'params': bert.layers[3].parameters(), 'lr': 1e-4},
    {'params': classifier.parameters(), 'lr': 1e-4}
], weight_decay=1e-2)

for epoch in range(5):
    # Training loop with mixup
    for x, y in train_loader:
        # Mixup
        lam = np.random.beta(0.2, 0.2)
        idx = torch.randperm(x.size(0))
        x_mix = lam * x + (1 - lam) * x[idx]
        y_mix = lam * y + (1 - lam) * y[idx]
        
        # Forward
        rep = bert(x_mix)
        logits = classifier(rep)
        
        # Label smoothing
        y_smooth = y_mix * 0.9 + 0.05
        loss = F.binary_cross_entropy_with_logits(logits, y_smooth)
        
        # Backward (with gradient clipping)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

# Save warm-up checkpoint
torch.save(bert.state_dict(), 'bert_epoch5_warmup.pth')

# ── PHASE B: FULL FINE-TUNE WITH EARLY STOPPING ──────────
# Unfreeze all layers
for param in bert.parameters():
    param.requires_grad = True

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-2)

best_val_auc = 0
patience_counter = 0

for epoch in range(6, 16):  # Max 10 more epochs
    train_loss = train_epoch(bert, classifier, train_loader, optimizer)
    val_auc = evaluate(bert, classifier, val_loader)
    
    print(f"Epoch {epoch}: Val AUC = {val_auc:.4f}")
    
    # Early stopping logic
    if val_auc > best_val_auc:
        best_val_auc = val_auc
        torch.save(bert.state_dict(), 'bert_best_val.pth')
        patience_counter = 0
    else:
        patience_counter += 1
    
    # STOP if not improving AND val_auc >= 0.95
    if patience_counter >= 3 and val_auc >= 0.95:
        print(f"Early stopping at epoch {epoch} (Val AUC = {val_auc:.4f})")
        break

# Save final checkpoint (likely overfit)
torch.save(bert.state_dict(), 'bert_epoch10_full.pth')
```

---

## 📊 Evaluation Protocol

### 1. Load Checkpoints
```python
checkpoints = [
    ('SSL Baseline', 'ssl_bert_paper.pth', 'SSL k-NN only'),
    ('Epoch 5', 'bert_epoch5_warmup.pth', 'Warm-up complete'),
    ('Best Val', 'bert_best_val.pth', 'Early stopped'),
    ('Epoch 10', 'bert_epoch10_full.pth', 'Full training')
]
```

### 2. Zero-Shot Test on CIC-IDS-2017
```python
for name, ckpt_path, desc in checkpoints:
    bert.load_state_dict(torch.load(ckpt_path))
    
    # Test on CIC (NEVER seen during training!)
    test_auc = evaluate_binary(bert, classifier, cic_test_loader)
    
    print(f"{name:15} CIC AUC = {test_auc:.4f}")
```

### 3. Per-Attack Breakdown
```python
attack_types = [
    'DDoS', 'DoS GoldenEye', 'DoS Hulk', 'DoS Slowloris', 'DoS slowhttptest',
    'FTP-Patator', 'SSH-Patator', 'PortScan', 'Bot', 'Web Attack', 'Infiltration'
]

for attack in attack_types:
    subset = [d for d in cic_test if d['attack_type'] == attack]
    metrics = evaluate_per_attack(bert, classifier, subset)
    print(f"{attack:20} AUC={metrics['auc']:.4f} F1={metrics['f1']:.4f}")
```

---

## 📈 Expected Results

### Hypothesis 1: Early Stopping Improves Generalization
```
Checkpoint       | UNSW Val AUC | CIC Test AUC | Generalization Gap
-----------------+--------------+--------------+--------------------
SSL Baseline     | 0.978        | 0.627        | 0.351 (BAD!)
Epoch 5 (warmup) | 0.920        | 0.750        | 0.170
Best Val (stop)  | 0.950        | 0.850        | 0.100 ← Target!
Epoch 10 (full)  | 0.995        | 0.680        | 0.315 (overfit!)
```

**Pattern**: Higher UNSW AUC ≠ Better generalization! Peak at early stopping.

### Hypothesis 2: BERT Can Match UniMamba with Proper Training
```
Model                | CIC AUC | Latency (B=32) | Throughput | TTD
---------------------+---------+----------------+------------+------
BERT (SSL only)      | 0.627   | 0.024 ms/flow  | 42K fps    | 310ms
BERT (fine-tuned)    | 0.850   | 0.024 ms/flow  | 42K fps    | 310ms ← Target
UniMamba @8          | 0.920   | 0.032 ms/flow  | 32K fps    | 70ms
```

**If achieved**: BERT becomes competitive on accuracy while keeping speed.  
**But**: UniMamba still wins on TTD (4.4x) and memory (1.8x).

---

## 🎯 Thesis Argument Refinement

### Scenario A: BERT Fine-Tuned Matches UniMamba (CIC AUC ≈ 0.85-0.90)

**Narrative**:
```
"This work demonstrates that while Transformer-based BERT can achieve 
competitive accuracy (CIC AUC=0.85) and higher throughput (42K fps vs 32K fps) 
through supervised fine-tuning, it fundamentally cannot perform early exit 
due to bidirectional attention requirements.

UniMamba's causal state-space design enables true packet-level early exit, 
reducing time-to-detect from 310ms to 70ms (4.4× improvement). In IDS 
deployment, this 240ms advantage determines whether attacks are blocked at 
initial reconnaissance (100 flows) or after full-scale launch (10,000+ flows).

While BERT optimizes throughput, UniMamba optimizes RESPONSE TIME—the 
metric that matters for preventing damage."
```

### Scenario B: UniMamba Still Outperforms (CIC AUC gap > 0.05)

**Narrative**:
```
"Supervised fine-tuning improves BERT's cross-dataset generalization 
(CIC AUC: 0.627→0.80), but UniMamba's causal architecture fundamentally 
enables better zero-shot transfer (CIC AUC=0.92). Combined with 4.4× 
faster time-to-detect and 1.8× smaller memory footprint, UniMamba 
provides a superior accuracy-efficiency Pareto front for IDS deployment."
```

---

## 🔄 Fallback Plans

### If CIC AUC doesn't improve beyond 0.70:
1. **Check data leakage**: Ensure CIC test has NO overlap with UNSW train
2. **Try different SSL checkpoint**: Use `ssl_bert_anti.pth` instead
3. **Increase regularization**: dropout=0.5, weight_decay=1e-1
4. **Domain adaptation**: Add source domain discrimination

### If training diverges:
1. **Reduce learning rate**: 1e-5 instead of 5e-5
2. **Smaller batch size**: 64 instead of 128
3. **Gradient accumulation**: Effective batch 256 with accumulation

---

## 💾 File Structure

```
thesis_final/final jupyter files/
├── train_bert_supervised_zeroshot.py    ← Main training script
├── eval_bert_zeroshot_checkpoints.py    ← Checkpoint evaluation
├── compare_bert_tuned_vs_unimamba.py    ← Final comparison
├── weights/
│   └── supervised_zeroshot/
│       ├── bert_epoch5_warmup.pth       ← Phase A output
│       ├── bert_best_val.pth            ← MAIN RESULT
│       └── bert_epoch10_full.pth        ← Overfit comparison
└── results/
    ├── bert_zeroshot_eval.json          ← Zero-shot CIC results
    ├── bert_per_attack_cic.txt          ← Per-attack breakdown
    └── bert_tuned_vs_unimamba_final.json ← Complete comparison
```

---

## ✅ Next Steps

1. **Prepare UNSW splits** (balanced train/val)
2. **Implement training script** with all anti-overfitting techniques
3. **Run training** (~90 min total)
4. **Evaluate zero-shot** on CIC-IDS-2017
5. **Compare** with UniMamba @8 baseline
6. **Update thesis** with final accuracy vs speed trade-off analysis

---

## 📌 Key Insight

**Current Status**: BERT is "fast but dumb" (generalizes poorly to CIC)  
**Goal**: Make BERT "fast AND smart" (match UniMamba accuracy while keeping throughput)  
**Method**: Supervised learning + aggressive regularization + early stopping  
**Expected**: CIC AUC improves from **0.627 → 0.85+**, closing the gap with UniMamba (0.92)

**But Even If Successful**: UniMamba retains **4.4x TTD advantage** (70ms vs 310ms) due to architectural early exit capability—this is the CORE thesis contribution!

---

**Status**: ⏳ READY TO IMPLEMENT  
**Author**: Thesis Verification Protocol  
**Last Updated**: February 24, 2026
