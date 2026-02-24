# 🎯 PLAN: Synthesizing Dataset for Post-Scan Attack Detection

## 📊 **PROBLEM STATEMENT**

Based on your comprehensive metrics analysis:

### Current Performance Issues:
1. **F1 Score Capped**: 
   - CIC-IDS DoS attacks: F1 = 0.01-0.12 (very low!)
   - High Recall (>97%) but **extremely low Precision** (<7%)
   - Reason: massive false positive rate

2. **PortScan Becomes "Benign-looking"**:
   - PortScan AUC = 0.96 (good) but...
   - Modern low-rate scans mimic benign browsing
   - Temporal features lost in static flow representation

3. **Class Imbalance**:
   - CIC-IDS: 81% benign, only 14% PortScan, <3% DoS
   - UNSW: 94% benign, 6% attack
   - Model biased toward majority class

---

## 🔬 **SOLUTION: Synthetic Data Augmentation Strategy**

### **Phase 1: Attack Signature Extraction & Profiling** (Week 1-2)

#### Step 1.1: Profile Existing Attack Patterns
```python
# Extract statistical signatures per attack type
for attack_type in ['PortScan', 'DDoS', 'DoS GoldenEye', 'DoS Hulk']:
    - Compute packet-level distributions:
      * Protocol distribution (TCP/UDP/ICMP)
      * Packet size (mean, std, quantiles)
      * IAT (inter-arrival time): mean, variance, burstiness
      * Flags: SYN/ACK/FIN/RST ratios
      * Direction: request/response balance
    
    - Flow-level statistics:
      * Flow duration
      * Packets per second
      * Bytes per second
      * Bidirectionality score
```

**Deliverable**: Attack signature profiles (JSON format)

---

#### Step 1.2: Identify Hardest-to-Detect Attacks
```python
# From your metrics, focus on:
LOW_PRECISION_ATTACKS = {
    'DDoS': {'precision': 0.067, 'recall': 0.977},
    'DoS GoldenEye': {'precision': 0.009, 'recall': 0.983},
    'DoS Hulk': {'precision': 0.008, 'recall': 0.986}
}

# These have HIGH RECALL but LOW PRECISION
# → They're being detected, but with tons of false alarms
# → Need more discriminative features
```

**Action**: Create augmented training samples that **amplify discriminative features**

---

### **Phase 2: Synthetic Data Generation** (Week 3-4)

#### Strategy A: **Feature-Space Data Augmentation**

Use **SMOTE** (Synthetic Minority Over-sampling Technique) for minority attacks:

```python
from imblearn.over_sampling import SMOTE, ADASYN

# Apply SMOTE to balance dataset
X_train, y_train = load_training_data()
smote = SMOTE(sampling_strategy='auto', k_neighbors=5)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Result: synthetic flows for minority classes (DoS, Bot, Infiltration)
```

**Pros**: Fast, proven technique for imbalance
**Cons**: May create unrealistic feature combinations

---

#### Strategy B: **Conditional GAN (cGAN) for Realistic Attack Synthesis**

Train a conditional GAN to generate attack flows:

```python
class AttackFlowGenerator(nn.Module):
    """
    Generator: z (noise) + attack_label → synthetic flow
    Discriminator: flow → real/fake + attack_label
    """
    def __init__(self):
        self.generator = nn.Sequential(
            nn.Linear(128 + 10, 256),  # 128 noise + 10 attack classes
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 32 * 5)  # 32 packets × 5 features
        )
        
    def forward(self, noise, attack_label):
        z = torch.cat([noise, attack_label], dim=1)
        return self.generator(z).view(-1, 32, 5)

# Train on real attack flows
# Generate synthetic DoS, PortScan variants
```

**Pros**: More realistic, captures attack distribution
**Cons**: Requires training, may mode collapse

---

#### Strategy C: **Mutation-Based Data Augmentation**

Perturb real attack flows to create variants:

```python
def mutate_attack_flow(flow, mutation_type):
    """
    Create attack variants by controlled perturbation
    """
    if mutation_type == 'temporal_stretch':
        # Increase IAT to mimic slow-rate attack
        flow[:, 3] *= np.random.uniform(1.5, 3.0)
    
    elif mutation_type == 'packet_drop':
        # Simulate packet loss (evading detection)
        drop_mask = np.random.rand(32) > 0.2
        flow[drop_mask] = 0  # zero-pad dropped packets
    
    elif mutation_type == 'protocol_switch':
        # Change TCP → UDP (protocol polymorphism)
        flow[:, 0] = 17  # UDP protocol number
    
    elif mutation_type == 'size_jitter':
        # Add noise to packet sizes
        flow[:, 1] += np.random.randint(-50, 50, size=32)
        flow[:, 1] = np.clip(flow[:, 1], 40, 1500)
    
    return flow

# Generate 10x variants per attack sample
for attack_flow in attack_dataset:
    for _ in range(10):
        mutated = mutate_attack_flow(attack_flow, random.choice(MUTATIONS))
        synthetic_dataset.append(mutated)
```

**Pros**: Interpretable, preserves attack semantics
**Cons**: Still based on real data, limited diversity

---

### **Phase 3: Targeted PortScan Synthesis** (Week 5)

Since PortScan is **easy to detect** (AUC=0.96) but can become "benign-looking":

#### Synthesize **Stealthy PortScan Variants**:

1. **Slow VerticalScan**:
   - Target: Single IP, multiple ports
   - IAT: 5-60 seconds (mimic human browsing)
   - Packet sizes: randomized (avoid uniform distribution)

2. **HorizontalScan with Decoys**:
   - Target: Multiple IPs, single port
   - Mix with legitimate DNS/HTTP requests
   - Randomize scanning order (avoid sequential)

3. **Distributed Scan**:
   - Spread scan across multiple source IPs
   - Coordinate timing to avoid temporal correlation

```python
def generate_stealthy_portscan(n_samples=10000):
    scans = []
    for _ in range(n_samples):
        scan_type = random.choice(['slow_vertical', 'horizontal', 'distributed'])
        
        if scan_type == 'slow_vertical':
            # Single target, many ports, long IAT
            flow = create_base_flow(
                protocol='TCP',
                dst_ip='<target>',
                dst_ports=random.sample(range(1, 65535), 32),
                iat_mean=30.0,  # 30 sec between probes
                packet_sizes=[52, 60, 54, 52]  # SYN packet sizes
            )
        
        elif scan_type == 'horizontal':
            # Many targets, single port
            flow = create_base_flow(
                protocol='TCP',
                dst_ips=generate_random_ips(32),
                dst_port=80,  # common port
                iat_mean=2.0,
                packet_sizes=[60, 40]  # SYN-ACK
            )
        
        # Add benign-like features
        flow = add_legitimate_cover_traffic(flow)
        scans.append(flow)
    
    return scans
```

---

### **Phase 4: Training & Validation** (Week 6-7)

#### Step 4.1: Data Split Strategy

```plaintext
Original Dataset:
  - CIC-IDS: 1,084,972 flows
  - 81% benign, 14% PortScan, 2% DDoS, 3% other

Augmented Dataset:
  - Keep 100% original benign (no augmentation)
  - Oversample minority attacks:
    * DDoS: 24,599 → 100,000 (4x via SMOTE + mutation)
    * DoS: 14,000 → 60,000 (4x)
    * PortScan: 158,239 → 200,000 (1.3x stealthy variants)
    * Bot/Infiltration: 100x (critical but rare)
  
  Final: 50/50 benign/attack (balanced)
```

#### Step 4.2: Training Protocol

1. **Baseline**: Train on original imbalanced data
   - Result: High recall, low precision (your current state)

2. **Augmented Training**: Train on balanced synthetic data
   - Use class weights: `weight = 1.0 / class_frequency`
   - Apply focal loss to handle remaining imbalance:
     ```python
     focal_loss = -alpha * (1 - p)^gamma * log(p)
     ```

3. **Validation**: Test on **ONLY REAL DATA** (never synthetic)
   - CIC-IDS test set (original, no augmentation)
   - Cross-dataset: CTU-13, UNSW-NB15

#### Step 4.3: Evaluation Metrics

Track improvement in **precision** for low-performing attacks:

| Attack | Original F1 | Target F1 | Strategy |
|--------|-------------|-----------|----------|
| DDoS | 0.13 | **0.60+** | SMOTE + mutation |
| DoS GoldenEye | 0.02 | **0.40+** | GAN synthesis |
| DoS Hulk | 0.02 | **0.40+** | GAN synthesis |
| Bot | 0.00 | **0.30+** | 100x oversampling |

**Success Criteria**:
- F1 score improvement of ≥3x for minority attacks
- Maintain AUC ≥ 0.85 overall
- Cross-dataset generalization (CTU-13, UNSW) preserved

---

### **Phase 5: Post-Scan Attack Simulation** (Week 8)

Since you mentioned **post-scan attacks** becoming hard to detect:

#### Create **Attack Sequence Datasets**:

Real attacks follow: `Recon → Exploit → C2 → Exfiltration`

Current datasets only capture **single-stage flows**.

**Solution**: Synthesize **multi-stage attack sequences**:

```python
class MultiStageAttackGenerator:
    def generate_attack_campaign(self):
        stages = []
        
        # Stage 1: Stealth PortScan (slow, 1-2 hours)
        recon_flows = generate_stealthy_portscan(duration=7200)
        stages.append(('recon', recon_flows))
        
        # Stage 2: Exploitation (single flow, looks benign)
        exploit_flow = generate_http_exploit(
            url='/admin/login',
            payload='<script>...</script>',
            response_code=200  # success
        )
        stages.append(('exploit', [exploit_flow]))
        
        # Stage 3: Command & Control (periodic beacons)
        c2_flows = generate_c2_beacons(
            interval=300,  # 5 min
            duration=86400,  # 24 hours
            protocol='HTTPS'  # encrypted
        )
        stages.append(('c2', c2_flows))
        
        # Stage 4: Data Exfiltration (large upload)
        exfil_flow = generate_exfiltration(
            data_size_mb=500,
            rate_limit_mbps=0.5,  # slow to avoid detection
            destination='cloud-storage.com'
        )
        stages.append(('exfiltration', [exfil_flow]))
        
        return stages
```

**Why This Helps**:
- Current models see **isolated flows** → miss temporal context
- Multi-stage attacks look benign in isolation
- Training on sequences improves **graph-based** or **RNN** models

---

## 📊 **EXPECTED RESULTS**

### Before Augmentation (Current):
| Metric | CIC-IDS | CTU-13 | UNSW |
|--------|---------|--------|------|
| AUC | 0.92 | 0.53 | 0.98 |
| F1 (DDoS) | **0.13** | - | - |
| F1 (DoS) | **0.02** | - | - |
| Precision (overall) | 0.53 | 0.66 | 0.58 |

### After Augmentation (Target):
| Metric | CIC-IDS | CTU-13 | UNSW |
|--------|---------|--------|------|
| AUC | 0.92 | 0.53 | 0.98 |
| F1 (DDoS) | **0.60+** | - | - |
| F1 (DoS) | **0.40+** | - | - |
| Precision (overall) | **0.75+** | **0.70+** | **0.70+** |

---

## 🛠️ **IMPLEMENTATION ROADMAP**

### Week 1-2: Profiling & Baseline
- [ ] Run comprehensive_metrics_report.py ✅ DONE
- [ ] Profile attack signatures (protocol, size, IAT distributions)
- [ ] Identify hardest-to-detect attacks

### Week 3-4: Synthesis
- [ ] Implement SMOTE baseline
- [ ] Implement mutation-based augmentation
- [ ] (Optional) Train conditional GAN if time permits

### Week 5: Stealthy PortScan
- [ ] Generate slow-scan variants
- [ ] Generate horizontal/distributed scan variants

### Week 6-7: Training
- [ ] Train on augmented data with class weights
- [ ] Validate on real test data only
- [ ] Measure precision improvement

### Week 8: Multi-Stage (Thesis Extension)
- [ ] Create attack sequence datasets
- [ ] Explore temporal models (LSTM/Transformer)
- [ ] Defense day: present as "future work"

---

## 🎓 **THESIS INTEGRATION**

### Chapter: "Addressing Class Imbalance via Synthetic Data"

1. **Motivation**: 
   - F1 capped due to extreme imbalance (2% DDoS)
   - Low precision = high false positive rate → unusable in practice

2. **Methodology**:
   - SMOTE for minority attacks
   - Mutation-based augmentation for realism
   - Stealthy PortScan synthesis for adversarial robustness

3. **Results**:
   - Table: F1 improvement per attack type
   - Figure: Precision-Recall curves (before/after)
   - Cross-dataset validation (CTU-13, UNSW)

4. **Discussion**:
   - Trade-off: Precision ↑, Recall ↓ (acceptable)
   - Synthetic data risk: model may overfit to generated patterns
   - Mitigation: Validate on **only real data**

5. **Future Work**:
   - Multi-stage attack detection (graph neural networks)
   - Online learning to adapt to new attack variants

---

## 📚 **REFERENCES**

1. **SMOTE**: Chawla et al., "SMOTE: Synthetic Minority Over-sampling Technique" (2002)
2. **GAN for IDS**: Sharafaldin et al., "A Detailed Analysis of the CICIDS2017 Data Set" (2018)
3. **Stealthy Scans**: Zalewski, "Silence on the Wire" (2005)
4. **Class Imbalance in IDS**: Dal Pozzolo et al., "Calibrating Probability with Undersampling" (2015)

---

## ✅ **DELIVERABLES**

1. **Code**:
   - `synthesize_attacks.py` (SMOTE + mutation)
   - `generate_stealthy_scans.py`
   - `train_on_augmented_data.py`

2. **Data**:
   - `cic_ids_augmented_balanced.pkl` (2x size of original)
   - `stealthy_portscans.pkl` (100K samples)

3. **Results**:
   - `augmentation_results.json` (per-attack F1 improvement)
   - Figures: Precision-Recall curves, Confusion matrices

4. **Thesis Chapter**:
   - 8-10 pages on data augmentation methodology

---

## 🚀 **START HERE**

Run this script to profile existing attacks:

```python
python3 thesis_final/final jupyter files/profile_attack_signatures.py
```

(I can create this script for you - just say the word!)

---

**Questions?** Let me know which phase you want to start with!
