# Architecture Audit & Cleanup Log

**Date:** 2025-02-21  
**Purpose:** Document all models, weights, and architectures before total workspace cleanup and fresh retraining.

---

## WHY WE'RE STARTING OVER

1. **At least 4 different BERT architectures** used across scripts — weights loaded with `strict=False` silently hiding mismatches
2. **SSL "BiMamba" pretrained model was actually a BiLSTM** (857K params vs 3.65M for real BiMamba)
3. **Student KD AUC=0.89 on CIC is suspicious** — teacher only gets 0.8346, needs fresh verification
4. **Weight files are a mess** — 100+ `.pth` files scattered everywhere, no clear lineage

---

## MODEL ARCHITECTURES FOUND (4 DIFFERENT BERTs!)

### BERT-A: `01_train_ssl_bert.py` (SSL Pretraining)
```
PacketEmbedder: emb_proto(256,16), emb_flags(64,16), emb_dir(2,4), proj_len/iat(1→16)
  → fusion(68 → 256)
BertEncoder: d_model=256, nhead=8, num_layers=2, ff=1024
  → proj_head: Linear(256→256) → ReLU → Linear(256→128)
  → Uses h[:,0,:] (first packet, NO CLS token)
```

### BERT-B: `04_train_bert_cutmix.py` (CutMix SSL)
```
PacketEmbedder: emb_proto(256,32), emb_flags(64,32), emb_dir(2,8), proj_len/iat(1→32)
  → fusion(136 → 256), LayerNorm
LearnablePositionalEncoding: pe_emb = nn.Embedding(max_len, d_model)
BertEncoder: d_model=256, nhead=8, num_layers=4, ff=512
  → proj_head: Linear(256→256) → ReLU → Linear(256→128)
  → Uses mean pooling
Classifier head: Linear(128→64) → ReLU → Dropout(0.2) → Linear(64→2)
```

### BERT-C: `07_train_bert_recreate.py` (Recreated BERT)
```
PacketEmbedding: emb_proto(256,32), emb_flags(256,32), emb_dir(4,16), proj_len/iat(1→32)
  → feature_mixer(144 → 256)
  → pos_emb = nn.Parameter(zeros(1, MAX_LEN+1, 256))
  → cls_token = nn.Parameter(zeros(1, 1, 256))  ← PREPENDED CLS TOKEN
PacketTransformer: d_model=256, nhead=4, num_layers=4, ff=1024, activation='gelu'
  → projection_head: Linear(256→256) → ReLU → Linear(256→128)
  → Uses encoded[:,0,:] (CLS token output)
PacketBERTClassifier head: Linear(256→256) → ReLU → Linear(256→2)
```

### BERT-D: `FULL_PIPELINE_SSL_PRETRAINING.py` in `final jupyter files/`
```
PacketEmbedder: emb_proto(256,32), emb_flags(64,32), emb_dir(2,8), proj_len/iat(1→32)
  → fusion(136 → 256), LayerNorm
  (Same embedder as BERT-B, but different encoder)
```

### KEY DIFFERENCES BETWEEN BERT VARIANTS:
| Feature | BERT-A | BERT-B | BERT-C |
|---------|--------|--------|--------|
| emb_dir | (2, 4) | (2, 8) | (4, 16) |
| emb_flags | (64, 16) | (64, 32) | (256, 32) |
| Fusion dim | 68→256 | 136→256 | 144→256 |
| CLS token | No | No | **Yes (nn.Parameter)** |
| Positional enc | None | LearnedPE(Embedding) | nn.Parameter |
| Layers/Heads | 2L/8H | 4L/8H | 4L/4H |
| FF dim | 1024 | 512 | 1024 |
| Activation | relu | relu | gelu |
| Pooling | h[:,0,:] | mean | CLS token |
| Classifier head | — | 128→64→2 | 256→256→2 |

**These architectures are INCOMPATIBLE. Loading weights with `strict=False` silently drops mismatched keys.**

---

## BiMamba ARCHITECTURES

### BiMamba in `05_distill_from_cutmix.py`:
```
BiMambaEncoder: d_model=256, n_layers=4
  PacketEmbedder: fusion(136 → 256)
  layers (forward): 4x Mamba(d_model=256, d_state=16, d_conv=4, expand=2)
  layers_rev (backward): 4x Mamba (same config)
  Fusion: norm(fwd + flip(bwd(flip(feat))) + feat)  ← RESIDUAL + BIDIRECTIONAL
  proj_head: Linear(256→256) → ReLU → Linear(256→128)
Classifier head: Linear(128→64) → ReLU → Dropout(0.2) → Linear(64→2)
```

### BiMamba in `FULL_PIPELINE_SSL_PRETRAINING.py` (first definition, line 62):
```
BiMambaEncoder: d_model=256, n_layers=4, d_state=32
  mamba_fwd: Sequential of 4x Mamba(d_model=256, d_state=32)
  mamba_bwd: Sequential of 4x Mamba(d_model=256, d_state=32)
  projection: Linear(d_model*2, 128)  ← CONCATENATES fwd+bwd (different from above!)
```

### BiMamba in `FULL_PIPELINE_SSL_PRETRAINING.py` (second definition, line 275):
```
BiMambaEncoder: d_model=256, n_layers=4
  layers: 4x Mamba(d_model=256, d_state=16, d_conv=4, expand=2)
  layers_rev: 4x Mamba (same)
  proj_head: Linear(256→256) → ReLU → Linear(256→128)
  recon_head: Linear(256→5)  ← reconstruction head for SSL
```

---

## STUDENT ARCHITECTURE

### BlockwiseStudent in `05_distill_from_cutmix.py`:
```
d_model=256, n_layers=4, exit_points=[8, 16, 32]
PacketEmbedder: fusion(136→256)
layers: 4x forward-only Mamba(d_model=256, d_state=16, d_conv=4, expand=2)
norm: LayerNorm(256)
exit_classifiers: ModuleDict at positions {8, 16, 32}
  each: Linear(258→64) → ReLU → Dropout → Linear(64→2)
  (258 = 256 + 2 from confidence)
confidence_heads: Linear(258→64) → ReLU → Linear(64→1) → Sigmoid
KD settings: temperature=4, alpha=0.5
```

---

## XGBoost (VERIFIED CORRECT)
```
Input: flows with features shape (32, 5) → FLATTENED to 160 columns
Params: max_depth=6, eta=0.1, 100 boosting rounds
         scale_pos_weight = neg/pos ratio
         objective: binary:logistic
Trained on: UNSW-NB15
Result: CIC-IDS-2017 cross-dataset AUC = 0.8776 ✅ VERIFIED
```

---

## SSL "BiMamba" WAS ACTUALLY BiLSTM

**Critical finding:** `bimamba_ssl_pretrained.pth` (3.3M) contains BiLSTM weights, NOT BiMamba:
- BiLSTM has 857K parameters (with LSTM layers), BiMamba has 3.65M parameters (with Mamba SSM layers)
- The file was produced by an earlier version of the SSL notebook that used `nn.LSTM`
- This means ALL SSL-pretrained BiMamba results were actually BiLSTM results
- BiLSTM SSL unsupervised anomaly detection: AUC=0.887 on CIC (KNN on embeddings)
- BiLSTM SSL unsupervised anomaly detection: AUC=0.948 on UNSW (in-domain)

---

## VERIFIED RESULTS (to compare against fresh retraining)

| Model | UNSW AUC | CIC AUC | Method |
|-------|----------|---------|--------|
| XGBoost (tabular) | ~0.99 | 0.8776 | Supervised, flatten 32×5→160 |
| BiMamba Teacher (supervised) | 0.9978 | 0.8346 | Supervised classification |
| BiLSTM SSL (unsupervised) | 0.948 | 0.887 | KNN anomaly on SSL embeddings |
| Student KD | ? | 0.8955?? | **SUSPICIOUS — needs fresh verification** |

---

## ALL WEIGHT FILES DELETED (inventory)

### Root level (9 files, ~72M):
- `bimamba_pretrained_paper.pth` (14M)
- `calibrated_blockwise_mamba.pth` (5.6M)
- `corrected_model_10pct.pth` (6.8M)
- `improved_combined_model.pth` (7.1M)
- `lepn_model.pth` (5.9M)
- `model_generalized.pth` (5.3M)
- `model_robust.pth` (15M)
- `streaming_lepn_model.pth` (6.3M)
- `thesis_final_model.pth` (5.8M)

### thesis_final/weights/teachers/ (11 files):
- `teacher_bert_finetuned.pth` (18M)
- `teacher_bert_cutmix.pth` (14M)
- `teacher_bert_cls.pth` (13M)
- `teacher_bert_recreate.pth` (13M)
- `teacher_bimamba_retrained.pth` (15M)
- `teacher_bimamba_scratch.pth` (14M)
- `teacher_bimamba_cutmix.pth` (14M)
- `teacher_bimamba_cutmix_fulldata.pth` (14M)
- `teacher_unimamba_light.pth` (3.9M)
- Plus student copies in teachers folder

### thesis_final/weights/ssl/ (14 files):
- `bimamba_ssl_pretrained.pth` (3.3M) — **ACTUALLY BiLSTM!**
- `bert_standard_ssl.pth` (18M)
- Multiple v2/v4/v5 variants and partials

### thesis_final/weights/students/ + students_cutmix/ + students_bert/ (13 files):
- `student_standard_kd.pth`, `student_ted.pth`, `student_uniform_kd.pth`, `student_no_kd.pth` (7.5M each)
- CutMix variants, BERT-distilled variants

### thesis_final/checkpoints/ (~30 files, 510M):
- Multiple seed checkpoints (s42, s123, s456, s789, s1024)
- v2 variants

### thesis_final/weights_backup_feb18/ (12 files, 149M):
- Backup copies of teacher/student weights

### Organized_Final/ (18 files scattered in subfolders):
- Phase 1, 2 variants, student variants
- `mamba_scratch_best.pth`, `teacher_fewshot.pth`, etc.

### thesis_final/weights/training_pipeline/ (2 files):
- `teacher_bimamba.pth` (6.8M)
- `bert_ssl.pth` (6.6M)

**TOTAL: ~1.9GB of weight files deleted**

---

## ALL TRAINING SCRIPTS DELETED (47 files in code/)

Key scripts preserved in documentation above:
- `01_train_ssl_bert.py` — BERT-A architecture (68→256, 2L)
- `04_train_bert_cutmix.py` — BERT-B architecture (136→256, 4L, LearnedPE)
- `05_distill_from_cutmix.py` — Student + BiMamba architectures
- `06_distill_from_scratch.py` — Alternative student training
- `07_train_bert_recreate.py` — BERT-C architecture (144→256, CLS token, 4L)
- `08_distill_from_bert_recreate.py` — Student from BERT-C
- `xgboost_cross_dataset.py` — XGBoost baseline

---

## DATASETS KEPT

| File | Size | Content |
|------|------|---------|
| `Organized_Final/data/unswnb15_full/flows_all.pkl` | 1.2G | Full UNSW-NB15 (1.62M flows) |
| `Organized_Final/data/unswnb15_full/pretrain_50pct_benign.pkl` | 556M | 787K benign for SSL |
| `Organized_Final/data/unswnb15_full/finetune_mixed.pkl` | 589M | 834K mixed for supervised |
| `thesis_final/data/cicids2017_flows.pkl` | 715M | CIC-IDS-2017 (1.08M flows) |
| `thesis_final/data/ctu13_flows.pkl` | 141M | CTU-13 |
| `thesis_final/data/cicdos2019_flows.pkl` | 448K | CIC-DoS-2019 |

**Flow format:** Each flow = `{'features': ndarray(32,5), 'label': int}`  
**5 features per packet:** protocol, length, flags, IAT, direction

---

## WHAT TO DO FOR FRESH RETRAINING

1. **Pick ONE consistent architecture** for BERT and BiMamba
2. **Use `strict=True`** for all weight loading — no silent mismatches
3. **Train actual BiMamba SSL** — not BiLSTM
4. **Verify Student KD results** from scratch — don't trust previous 0.89 CIC AUC
5. **All work in `thesis_final/final jupyter files/`**

---

## CLEANUP SUMMARY

**KEPT:**
- `thesis_final/final jupyter files/` — working folder
- `thesis_final/data/` — CIC + CTU datasets
- `Organized_Final/data/unswnb15_full/` — UNSW datasets
- `trash/` — per user request
- `mamba_env/` — Python environment
- `.git/` — version control

**DELETED:**
- 100+ `.pth` weight files (~1.9GB)
- 47 Python scripts in `thesis_final/code/`
- 13 Jupyter notebooks in `thesis_final/`
- 11 Python scripts in `thesis_final/` root
- 7 markdown files in `thesis_final/`
- All root-level scripts, notebooks, logs, PDFs, CSVs
- `thesis_final/weights/`, `checkpoints/`, `weights_backup_feb18/`, `weights_cicids/`
- `thesis_final/plots/`, `plots_cicids/`, `results/`, `results_chapter/`, `results_cicids/`, `results_ssl_pipeline/`
- `thesis_final/defense_slides/`, `docs/`, `figures/`, `logs/`, `scripts/`
- `Organized_Final/` (except `data/unswnb15_full/`)
- `photos/`, `mcpserver/`, `cicids_labels/`, `__pycache__/`, `weights/`
- Root: `Final_Master_Dataset.csv` (7.3G), `GeneratedLabelledFlows.zip` (271M)
- Root: `Organized_Final/data/*.csv` (28G of raw CSVs)
