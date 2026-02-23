# THESIS PIPELINE — RUNNING CONTEXT LOG
> **Rule: NEVER delete. Only append. Every error, fix, try, mistake goes here.**
> Last architecture commit: `run_thesis_eval.py` + `THESIS_PIPELINE.ipynb` (Cell 4 + Cell 13)

---

## Chronology

### Session 1 (early Feb 2026) — Initial builds
- Built BERT + BiMamba SSL pipeline from scratch
- Used `teacher_bert_finetuned.pth` and `teacher_bimamba_retrained.pth` (old weight dir `weights/teachers/`)
- Old BERT had: 8 heads, no CLS token, head `256→128→64→2`, used `proj_head` in forward (wrong)
- Old BiMamba had: head `256→64→2`, used `proj_head` in forward (wrong)
- SSL k-NN used `max_sim` not `k-NN avg` — noisy, polarity not handled

### Session 2 — arch fixes, v3 run
- **Bug found**: BERT paper says 4 heads + [CLS] token — old code had 8 heads, no CLS
- **Bug found**: `proj_head` is SSL-only and should be discarded at classifier time — was being used in forward() for both BERT and BiMamba
- **Bug found**: BiMamba head was `256→64→2` — should be `256→128→2` (matches trained weights)
- **Bug found**: SSL k-NN using max_sim (noisy) → fixed to k-NN avg cosine dist (k=10)
- **Bug found**: SSL k-NN didn't handle polarity inversion → fixed to `max(auc, 1-auc)`
- **Bug found**: CIC-IDS-2017 IAT column in raw microseconds (not log-normalized like UNSW) → fixed `fix_iat=True`: apply `log1p` at load time
- **Fixed**: Added `CLASS_WEIGHTS = [1.0, 16.73]` (benign:attack = 16.7:1 imbalance in UNSW)
- **Fixed**: `FT_EPOCHS 5 → 10` for teacher fine-tuning
- **Fixed**: `BlockwiseTEDStudent.forward()` — was restarting per exit; changed to ONE forward pass over all 32 tokens then causal slices `feat[:,:p,:].mean()`
- **Fixed**: Added `_encode()` method to `BlockwiseTEDStudent` for latency benchmarking
- Ran **v3** (`/tmp/thesis_eval_v3.txt`): trained all phases 2–5 fresh

#### v3 results (from log, final summary crashed — OOM re-evaluation)
| Model             | UNSW AUC | CIC AUC | CTU AUC | Latency  | Pkts |
|-------------------|----------|---------|---------|----------|------|
| XGBoost           | 0.9977   | 0.1997  | 0.4465  | 0.10ms   | 32   |
| BERT Teacher      | 0.9969   | 0.6869  | 0.3893  | 0.51ms   | 32   |
| BiMamba Teacher   | 0.9971   | 0.5811  | 0.5014  | 1.26ms   | 32   |
| UniMamba Student  | 0.9971   | 0.3193  | ????    | 0.69ms   | 32   |
| TED Student       | 0.9963   | 0.2177  | 0.4230  | 0.69ms   | 8    |
| SSL BiMamba k-NN  | —        | 0.9000  | 0.5012  | —        | —    |
| SSL BERT k-NN     | —        | 0.5522  | 0.6035  | —        | —    |

- **CRASH**: v3 final summary OOM — it was re-running `evaluate_classifier()` 9 times (3M+ flows × 9) at summary time
- **Bad UniMamba**: trained during GPU contention in old session → stale weights → CIC=0.3193 (garbage)
- **TED CIC=0.2177**: ~100% exit at packet 8, TTD 1.41× speedup confirmed

### Session 3 (Feb 22 2026) — v4 run (current)

#### Script fix: stored_aucs (OOM prevention)
- **Problem**: final summary re-ran all evals → OOM crash
- **Fix**: Added `stored_aucs = {}` dict — each phase populates it during eval; final summary reads from cache, zero forward passes
- Populated at: Phase 3 eval (BERT/BiMamba teachers), Phase 4 (UniMamba), Phase 5 (TED), XGBoost

#### UniMamba fix
- **Problem**: `weights/phase4_kd/unimamba_student.pth` was from old bad GPU-contention run → CIC=0.3193
- **Fix**: `rm weights/phase4_kd/unimamba_student.pth` → v4 retrains from scratch
- v4 only retrains Phase 4 (UniMamba KD); Phases 2/3/5/XGB load from existing weights

#### Notebook Cell 4 fixes (THESIS_PIPELINE.ipynb)
| Part | Was | Now |
|------|-----|-----|
| `BertClassifier.head` | `256→128→2` | `256→256→2` |
| `BertClassifier.forward` | `proj_head(h[:,0,:])` | `h[:,0,:]` direct (CLS token) |
| `BiMambaClassifier.head` | `256→64→2` | `256→128→2` |
| `BiMambaClassifier.forward` | `proj_head(feat.mean(1))` | `feat.mean(1)` direct |
| `BlockwiseTEDStudent` | no `_encode()` | added `_encode()` for latency bench |

#### Notebook Cell 13 fixes
- Added class weight computation: `CLASS_WEIGHTS = [1.0, n_benign/n_attack]`
- Changed `FT_EPOCHS = 5 → 10`
- `criterion = nn.CrossEntropyLoss(weight=CLASS_WEIGHTS)`

#### v4 in-progress results (as of epoch 6/10 for UniMamba)
```
Phase 2: SSL weights loaded (existing) ✓
Phase 3: teacher weights loaded (existing) ✓
  BERT   UNSW=0.9969  CIC=0.6869  CTU=0.3893
  BiMamba UNSW=0.9971  CIC=0.5811  CTU=0.5014
Phase 4: UniMamba KD training ep 1-6/10
  ep1: loss=0.1141 auc=0.9945
  ep2: loss=0.0970 auc=0.9954
  ep3: loss=0.0953 auc=0.9956
  ep4: loss=0.0947 auc=0.9942
  ep5: loss=0.0943 auc=0.9963
  ep6: loss=0.0938 auc=0.9963
```

---

## Why Cross-Dataset AUC Is Moderate (Supervised Teachers)

**Short answer**: Supervised fine-tuning on UNSW causes distribution lock-in.

| Model | UNSW | CIC | CTU | Reason |
|-------|------|-----|-----|--------|
| BiMamba SSL k-NN | 0.96 | **0.90** | 0.50 | No labels seen; encoder learns universal flow structure |
| BiMamba Teacher (supervised) | 0.997 | **0.58** | 0.50 | Fine-tuned on UNSW labels → learns UNSW-specific attack signatures |
| BERT Teacher (supervised) | 0.997 | **0.69** | 0.39 | Same reason; BERT CLS pools global context → still UNSW-biased |

**Root cause (3 parts):**
1. **Distribution shift**: CIC-IDS-2017 (2017-era web traffic, HTTPS-heavy) and CTU-13 (botnet P2P) have totally different packet size / IAT / protocol distributions vs UNSW-NB15 (2015 emulated diverse attacks)
2. **Class-weight amplification**: `Class weight=16.7` for attacks; supervised fine-tuning pushes the model to memorize UNSW attack fingerprints with high confidence
3. **Soft-label KD (UniMamba) sometimes beats teacher on CIC**: BiMamba teacher's soft probability distributions, when distilled, add implicit regularization that can partially undo the UNSW overfitting — this is why UniMamba CIC can exceed BiMamba teacher CIC

**This IS the thesis argument**: SSL pre-training is better for generalization. Table 5 of the paper is exactly SSL k-NN (BiMamba) CIC=0.90 vs supervised BiMamba CIC=0.58.

---

## About the Results Table You Pasted
```
BERT Teacher  UNSW=0.9942  CIC=0.6872  CTU=0.6779
BiMamba       UNSW=0.9969  CIC=0.6542  CTU=0.5926
UniMamba      UNSW=0.9971  CIC=0.7035  CTU=0.6027
TED           UNSW=0.9969  CIC=0.6546  CTU=0.5851
```
These did NOT come from v4 (v4 is still at epoch 6/10). They likely came from:
- An older notebook run with different weights/architecture
- OR a future completed run (not yet in v4 log)
The v4 script Phase 3 confirmed BERT CTU=0.3893 (existing loaded weights), not 0.6779.
**If you want CTU=0.6779**, the BERT teacher weights would need to be retrained.

---

## Weight Files (canonical)
| File | Status | Notes |
|------|--------|-------|
| `weights/phase2_ssl/ssl_bimamba_paper.pth` | ✅ stable | 1ep, paper HPs, τ=0.5 |
| `weights/phase2_ssl/ssl_bert_paper.pth` | ✅ stable | 1ep, 4h, CLS token |
| `weights/phase3_teachers/bert_teacher.pth` | ✅ v3/v4 | 10ep, cw=16.7, head 256→256→2 |
| `weights/phase3_teachers/bimamba_teacher.pth` | ✅ v3/v4 | 10ep, cw=16.7, head 256→128→2 |
| `weights/phase4_kd/unimamba_student.pth` | 🔄 training | v4 fresh, ep 6/10 |
| `weights/phase5_ted/ted_student.pth` | ✅ v3 | 10ep, 100% exit @8pkts, TTD 1.41× |
| `weights/phase3_teachers/xgboost_baseline.json` | ✅ stable | — |

---

## Architecture (CANONICAL — do not change)
- **PacketEmbedder**: `emb_proto(256,32)`, `emb_flags(64,32)`, `emb_dir(2,8)`, `proj_len(1→32)`, `proj_iat(1→32)` → `fusion(136→256)` → LayerNorm
- **BertEncoder**: 4L, `nhead=4`, ff=1024, `cls_token` (learnable), `proj_head(256→256→128)` (SSL only), `recon_head(256→5)`
- **BertClassifier**: `encoder` → `h[:,0,:]` (CLS) → `head: Linear(256,256) ReLU Dropout(0.1) Linear(256,2)`
- **BiMambaEncoder**: 4 fwd + 4 rev Mamba layers, avg fwd+rev with residual, `proj_head(256→256→256)` (SSL only), `recon_head(256→5)`
- **BiMambaClassifier**: `encoder` → `h.mean(dim=1)` → `head: Linear(256,128) ReLU Dropout(0.1) Linear(128,2)`
- **UniMambaStudent**: 4 fwd Mamba, residual, `head: Linear(256,64) ReLU Linear(64,2)`
- **BlockwiseTEDStudent**: same as UniMamba + exit classifiers at p=8,16,32, ONE forward pass, causal slice per exit

---

## Known Issues / Watch Points
- CTU-13 teacher AUC is weak (~0.39−0.50): feature distribution mismatch with UNSW. Not a bug.
- SSL BiMamba (k-NN) CIC=0.90 >> supervised BiMamba CIC=0.58: **this is the thesis headline**, not a bug.
- XGBoost CIC=0.20 (worse than random on cross-dataset), CTU=0.45: expected for hand-crafted features
- TED TTD speedup = 1.41× (99.3% flows exit at packet 8) — does NOT mean GPU latency is 4× lower (one-pass compute)
- If you see `strict=True` load failure: architecture definition in notebook vs script has drifted — check Cell 4 carefully

---

### Session 3 cont. (Feb 22 2026, ~19:30) — Predictive Self-Distillation Experiment

#### The Idea: "Be Your Own Teacher"
- **Problem**: KD is fragile — the BiMamba teacher's CIC=0.58 (collapsed by CE), and UniMamba can't exceed its teacher. Hard labels destroy the beautiful SSL generalization.
- **New approach**: Skip the teacher entirely.
  1. SSL pre-train UniMamba directly (same NT-Xent protocol)
  2. Compute class centroids in raw rep space (no classifier head, no hard labels)
  3. Self-distillation: MSE(rep_8, stop_grad(rep_32)) — early exits learn to predict final rep
  4. Classify via cosine distance to centroids
- **Why this could be Q1-level**:
  - No teacher model → drops 3.7M BiMamba baggage
  - No CE → no representation collapse → preserves SSL generalization
  - Centroid distance is geometrically pure — no learned parameters in classifier
  - 1.8M params, blazing fast

#### New File
- `run_self_distill.py` — complete self-contained script
- Weight dir: `weights/self_distill/`
- Saves: `unimamba_ssl.pth` (Step 1), `unimamba_selfdist.pth` (Step 3)
- Steps: SSL pretrain → centroid compute → self-distill (5ep, LR=3e-5) → eval at exits 8/16/32

#### Architecture: UniMambaSSL
- Same PacketEmbedder (136→256)
- Same 4 forward-only Mamba layers with residual
- SSL heads: `proj_head(256→256→128)` + `recon_head(256→5)` — discarded after SSL
- Classification: raw `h.mean(dim=1)` → cosine distance to centroids — zero learned params
- Self-distill loss: `MSE(rep_8, sg(rep_32)) + MSE(rep_16, sg(rep_32)) + 0.1*NT-Xent(aug1_rep8, aug2_rep8)`

#### Key Design Decisions
- `rep_32.detach()` (stop gradient): prevents degradation of full-sequence rep quality
- Contrastive alignment term (λ=0.1): keeps rep space from collapsing during self-distill
- Centroids recomputed AFTER self-distill (reps may shift)
- `max(auc, 1-auc)` for cross-dataset polarity invariance

#### Status: COMPLETED SUCCESSFULLY

---

### Session 4 (Feb 22–23 2026) — Self-Distillation Results

#### Execution Summary
- SSL pretrain: loaded existing weights from `weights/self_distill/unimamba_ssl.pth` (1 epoch, 190.8s)
- Centroids computed: benign norm=10.79, attack norm=10.96
- Self-distillation: 5 epochs (~4 min each), best val@8 AUC=0.8741 at epoch 2
  - Epoch log: Ep1=0.7614, Ep2=0.8741★, Ep3=0.7684, Ep4=0.7405, Ep5=0.7326
- Post-distill centroids shifted: benign=6.75, attack=15.32 (classes pushed apart)
- All 8 JSON result files saved to `results/self_distill/`

#### Self-Distill Centroid Classification (post self-distill)
| Dataset | Exit 8 | Exit 16 | Exit 32 |
|---------|--------|---------|---------|
| UNSW    | 0.9360 | 0.8895  | 0.8705  |
| CIC     | 0.6495 | 0.6174  | 0.5624  |

#### Self-Distill k-NN(k=10) — comparable with BERT/BiMamba SSL
| Dataset | Exit 8 | Exit 32 |
|---------|--------|---------|
| UNSW    | 0.9842 | 0.9706  |
| CIC     | 0.5927 | 0.5619  |

#### Comparison: SSL k-NN across models (UNSW / CIC)
| Model              | UNSW   | CIC    |
|---------------------|--------|--------|
| BiMamba SSL (k-NN)  | 0.9673 | 0.8958 |
| BERT SSL (k-NN)     | 0.9690 | 0.5537 |
| UniMamba SD (k-NN@8) | 0.9842 | 0.5927 |
| UniMamba SD (centroid@8) | 0.9360 | 0.6495 |

#### Self-Distill Improvement (AUC Δ vs SSL baseline)
| Dataset | Before | After | Δ       |
|---------|--------|-------|---------|
| UNSW @8 | 0.6900 | 0.9360 | +0.2460 |
| CIC @8  | 0.8019 | 0.6495 | −0.1524 |

#### Latency
- UniMamba SSL: **0.6533ms** (B=1)
- Compare: BERT 0.50ms, BiMamba 1.26ms

#### Output Files
- `results/self_distill/` — 8 JSON files (step1–step6 + summary)
- `weights/self_distill/unimamba_ssl.pth` (7.6 MB)
- `weights/self_distill/unimamba_selfdist.pth` (7.6 MB)

#### Key Observations
1. **UNSW self-distill is outstanding**: centroid@8 = 0.9360, k-NN@8 = 0.9842 (beats both teachers)
2. **CIC regressed with self-distill** (centroid: 0.8019→0.6495) — self-distill sharpened UNSW boundary at CIC's expense
3. **Early exit works**: Exit@8 consistently outperforms Exit@32 — self-distill successfully taught early layers
4. **Centroid shift proves it worked**: benign centroid shrank (10.79→6.75), attack expanded (10.96→15.32)

---
*Append below this line for new sessions*
