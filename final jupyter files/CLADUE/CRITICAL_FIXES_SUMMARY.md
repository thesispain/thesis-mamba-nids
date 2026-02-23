# 🚨 NOTEBOOK CRITICAL FIXES - QUICK SUMMARY

## Top 6 Critical Bugs (Fix TODAY!)

### 1. ❌ XGBoost Missing - NO FEATURE EXTRACTION!

**Problem:** Your data is raw packets (32×5), XGBoost needs tabular features (1×49)

**Your code does NOT convert packets → features!**

**Fix:**
```python
def extract_statistical_features(flow_data):
    """Convert (32,5) packets → (49,) features"""
    features = []
    for sample in flow_data:
        packets = sample['features']
        
        # Basic stats
        n_packets = packets.shape[0]
        lengths = packets[:, 1]
        iats = packets[:, 3]
        
        # Build feature vector
        feat = [
            n_packets,
            lengths.mean(), lengths.std(), lengths.max(),
            iats.mean(), iats.std(), iats.max(),
            # ... add more to reach 49 features
        ]
        features.append(feat)
    
    return np.array(features)

# Then:
X_train = extract_statistical_features(unsw_data)
xgb = XGBClassifier()
xgb.fit(X_train, y_train)
```

---

### 2. ❌ load_model_safe() Defined TWICE!

**Problem:** Cell 6 has good version, Cell 8 overwrites it with bad version!

**Cell 6:** 60 lines, maps "encoder.head" → "head", handles all key mismatches ✓
**Cell 8:** 15 lines, simple loading, NO key mapping ❌

**Result:** Models load with wrong weights!

**Fix:** **DELETE the entire load_model_safe() function from Cell 8!**

---

### 3. ❌ Using CPU Not GPU!

**Problem:**
```python
DEVICE = torch.device('cpu')  # Cell 2
```

**Result:** 
- All latency measurements are on CPU (10-100× slower than GPU)
- Your TTD claims are wrong
- torch.cuda.synchronize() crashes

**Fix:**
```python
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if DEVICE.type == 'cpu':
    print("⚠️ WARNING: GPU not available! Latency measurements invalid!")
```

---

### 4. ❌ Wrong Test Set!

**Problem:**
```python
UNSW_TEST_PKL = "finetune_mixed.pkl"  # This is NOT a test set!
```

**Result:** Testing on training data = cheating!

**Fix:**
```python
UNSW_TEST_PKL = "unsw_test.pkl"  # Use actual test split
# OR:
UNSW_TEST_PKL = "test_flows.pkl"
```

---

### 5. ❌ TTD Calculation Uses Guesses!

**Problem:**
```python
calculate_ttd(packets_needed=8, gpu_latency_ms=lat_ted/4)  # Why /4???
```

**Result:** You're guessing 8-packet latency = full_latency / 4!

**Fix:**
```python
# Measure separately at each exit point
x_8pkt = torch.randn(1, 8, 5).to(DEVICE)
lat_8 = measure_latency(lambda x: ted.forward_at_exit(x, 8)[0], x_8pkt)

x_32pkt = torch.randn(1, 32, 5).to(DEVICE)  
lat_32 = measure_latency(lambda x: ted.forward_at_exit(x, 32)[0], x_32pkt)

ttd_8 = (8-1)*31.25 + lat_8    # Real measurement!
ttd_32 = (32-1)*31.25 + lat_32
```

---

### 6. ❌ Latency Function Won't Run on CPU!

**Problem:**
```python
def measure_lat(...):
    torch.cuda.synchronize()  # Crashes if DEVICE='cpu'!
    t0 = time.time()
    for _ in range(100): model(x)
    torch.cuda.synchronize()  # Crashes if DEVICE='cpu'!
    return (time.time() - t0) * 10
```

**Fix:**
```python
def measure_lat(model, input_shape=(1, 32, 5)):
    x = torch.randn(input_shape).to(DEVICE)
    
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = model(x)
    
    # Sync only if GPU
    if DEVICE.type == 'cuda':
        torch.cuda.synchronize()
    
    # Measure
    t0 = time.time()
    for _ in range(100):
        with torch.no_grad():
            _ = model(x)
    
    if DEVICE.type == 'cuda':
        torch.cuda.synchronize()
    
    return (time.time() - t0) * 10  # ms per sample
```

---

## Quick Fix Priority:

### Fix RIGHT NOW:
1. Delete `load_model_safe()` from Cell 8
2. Change `DEVICE = torch.device('cpu')` to `'cuda'`
3. Fix `UNSW_TEST_PKL = "finetune_mixed.pkl"` to `"unsw_test.pkl"`

### Fix TODAY:
4. Add XGBoost feature extraction function
5. Add XGBoost training & evaluation code
6. Fix latency measurement (handle CPU case)

### Fix THIS WEEK:
7. Measure TTD at each exit (don't guess /4)
8. Add Student No-KD in-domain test
9. Add exit distribution analysis

---

## The XGBoost Question:

**Q:** "For XGBoost, don't we need tabular data but did my code do it?"

**A:** **NO! Your code does NOT do it!**

**Your data:**
```python
# Raw packets (32 × 5)
[proto, length, flags, iat, direction]  # Packet 1
[proto, length, flags, iat, direction]  # Packet 2
...
```

**XGBoost needs:**
```python
# Statistical features (1 × 49)
[n_packets, mean_length, std_length, max_length, ...]
```

**You must write `extract_statistical_features()` to convert!**

This is why XGBoost is missing - you need the feature extraction code first!

---

## Bottom Line:

**6 critical bugs, all fixable!**

**Priority 1:** Delete Cell 8 `load_model_safe()` (5 seconds)
**Priority 2:** Switch to GPU device (10 seconds)  
**Priority 3:** Add XGBoost feature extraction (30 minutes)
**Priority 4:** Fix TTD measurement (15 minutes)
**Priority 5:** Fix latency measurement (10 minutes)
**Priority 6:** Use correct test set (5 seconds)

**Total time: ~1 hour of fixes → then you're ready!** ✅
