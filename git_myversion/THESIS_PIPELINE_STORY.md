# Thesis Narrative Pipeline: The "Story", The Fix, and The Speed

> **User Question:** "How are you training so fast? Did you use less data?"
> **Answer:** **YES.** The current run used **1% of the training data** (approx 10k-20k flows) as a "Proof of Concept" to verify the code works.
>
> **For Final Thesis Results:** We must switch `DATA_PCT = 1.0` (100%), which will take hours to train.

---

## 1. The Narrative (The "Story")

We successfully implemented the following pipeline to prove the Thesis arguments:

| Step | Model | Role | Result (on 1% Data) | Narrative Point |
| :--- | :--- | :--- | :--- | :--- |
| **1** | **BERT** | Baseline 1 | **F1 0.98** | "High Accuracy, but Slow (O(n²))" |
| **2** | **UniMamba** | Baseline 2 | **F1 0.97** | "Fast but usually weaker (though strong here)" |
| **3** | **BiMamba** | **The Teacher** | **F1 0.97** | "The Oracle. Matches BERT accuracy, better speed." |
| **4** | **Student (KD)** | Standard Student | **F1 0.98** | "Matches Teacher accuracy, high throughput." |
| **5** | **Student (TED)** | **Early Exit** | **F1 0.96** | "Algorithm exits at **Packet 8** (94% of time). Fastest Latency." |

---

## 2. The "Oracle Failure" Explained

**Initial Confusion:**
*   You asked why the "Oracle" (BiMamba Teacher) failed initially (F1 0.02).
*   **Reason:** The specific file `teacher_bimamba_masking.pth` was **corrupted/empty**. It was NOT the good SSL model.

**The Fix:**
*   We modified the script to **delete** the bad teacher file.
*   We loaded the **Good SSL Brain** (`bimamba_masking_v2.pth`).
*   We fine-tuned it on the task.
*   **Result:** Teacher F1 recovered to **0.97**.

---

## 3. "Why was it so fast?" (The 1% Factor)

To debug the pipeline and fix the Teacher issue quickly, I set the data limit to **1%** in `run_unified_pipeline.py`.

```python
# run_unified_pipeline.py
def main():
    # Allow user to specify data percentage (default 1%)
    DATA_PCT = 0.01 
```

### Impact of 1% vs 100%
*   **1% Data:** Training takes ~2 minutes per model. Results are "indicative" but not final.
*   **100% Data:** Training will take ~1-2 hours per model. Results will be robust.

**Recommendation:**
Now that we verified the code works and the Teacher is fixed, we should run with `DATA_PCT = 1.0` (or at least 10-20%) for the final charts.

---

## 4. Next Steps

1.  **Modify Script:** Change `DATA_PCT` to `1.0`.
2.  **Run Overnight:** Execute `python scripts/run_unified_pipeline.py`.
3.  **Plot Results:** Use the final metrics for the Thesis.
