# RAG Evaluation Datasets - Status Report

**Last Updated:** January 27, 2026  
**Environment:** Modal (Python 3.10, datasets==1.18.0)

---

## Quick Summary

| Category | Count | Datasets |
|----------|-------|----------|
| ✅ **Working** | 5/7 | NQ, TriviaQA, WebQuestions, CuratedTrec, FEVER |
| ⚠️ **Manual Download** | 1/7 | MS-MARCO |
| ❌ **Unavailable** | 1/7 | SearchQA |

**Core RAG Benchmarks**: All 4 working ✓ (NQ, TriviaQA, WebQ, CuratedTrec)

---

## Detailed Status

### ✅ Working Datasets (5/7)

#### 1. Natural Questions (NQ)
- **Source**: DPR repository (Facebook Research)
- **Test samples**: 3,610
- **Dev samples**: 8,757
- **Status**: Fully automated download ✓

#### 2. TriviaQA  
- **Source**: DPR repository (Facebook Research)
- **Test samples**: 11,313
- **Dev samples**: 18,669
- **Status**: Fully automated download ✓

#### 3. WebQuestions
- **Source**: DPR repository (Facebook Research)
- **Test samples**: 2,032
- **Dev samples**: N/A (403 Forbidden - not critical)
- **Status**: Working (test set only) ✓

#### 4. CuratedTrec
- **Source**: DPR repository (Facebook Research)
- **Test samples**: 694
- **Dev samples**: N/A (403 Forbidden - not critical)
- **Status**: Working (test set only) ✓

#### 5. FEVER (Fact Extraction and VERification)
- **Source**: HuggingFace `fever/fever` dataset
- **Test samples**: 37,566
- **Status**: Fully automated download ✓

---

### ⚠️ Manual Download Required (1/7)

#### 6. MS-MARCO

**Why It Doesn't Work Automatically:**

1. **Microsoft Blob Storage Error (Primary Issue)**
   ```
   Couldn't reach https://msmarco.blob.core.windows.net/msmsarcov1/train_v1.1.json.gz 
   Error 409: Conflict
   ```
   - Microsoft's blob storage returns HTTP 409 errors
   - This is an external infrastructure issue beyond our control

2. **FlashRAG Incompatibility (Modal Environment)**
   ```
   Invalid pattern: '**' can only be an entire path component
   ```
   - FlashRAG uses glob patterns (`**`) incompatible with datasets==1.18.0
   - Works fine locally with newer datasets versions (e.g., 3.2.0)
   - Modal environment requires datasets==1.18.0 for other dataset compatibility

**Local Testing Results:**
- ✅ Verified working with `load_dataset('RUC-NLPIR/FlashRAG_datasets', 'msmarco-qa')`
- Successfully loaded 808,731 train samples + 101,093 dev samples
- **Only works with datasets >= 2.0** (not available in Modal's Python 3.10 + datasets 1.18.0)

**Manual Workaround:**

1. **Download from Kaggle**:
   - URL: https://www.kaggle.com/datasets/parthplc/ms-marco-dataset/data
   - Files needed: `ms-marco-train.csv`, `ms-marco-valid.csv`

2. **Upload to Modal Volume**:
   ```bash
   modal volume put rag-data ms-marco-train.csv eval_datasets/ms-marco-train.csv
   modal volume put rag-data ms-marco-valid.csv eval_datasets/ms-marco-valid.csv
   ```

3. **Re-run Preparation**:
   ```bash
   modal run modal_rag_eval.py::prepare_datasets
   ```
   The code will automatically detect and use the Kaggle CSV files.

**Impact:** Low - MS-MARCO is optional for RAG evaluation (core benchmarks all work)

---

### ❌ Unavailable (1/7)

#### 7. SearchQA

**Issue:** Data corruption in HuggingFace repository

**Error:**
```
json.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0)
```

**What Happens:**
1. Dataset files download successfully ✓ (3.15 GB total)
   - train: 2.23 GB
   - test: 622 MB
   - val: 314 MB
2. JSON parsing fails during dataset script execution ✗
3. Error occurs in dataset's own loading script (line 141): `data = json.load(f)`

**Investigation:**
- ✗ `kyunghyuncho/search_qa` - JSON corruption in downloaded files
- ✗ Direct download fallback - Files return 404 Not Found
- ✗ No alternative working sources found

**Root Cause:** The JSON files in the HuggingFace repository are corrupted or malformed. This is not fixable from our code - it's a bug in the upstream dataset repository.

**Impact:** Low - SearchQA is a supplementary dataset, not part of RAG paper's core benchmarks

---

## Impact Assessment

### For RAG Paper Reproduction

**Primary Benchmarks (from original paper):**
- ✅ Natural Questions (NQ) - WORKING
- ✅ TriviaQA - WORKING  
- ✅ WebQuestions - WORKING
- ✅ CuratedTrec - WORKING

**Result:** **All core benchmarks fully functional** ✓

### Additional Datasets

**Supplementary:**
- ✅ FEVER - WORKING (bonus dataset for fact verification)
- ⚠️ MS-MARCO - MANUAL DOWNLOAD (optional generation task)
- ❌ SearchQA - UNAVAILABLE (non-critical Jeopardy QA)

### Recommendation

**Option 1 (Recommended):** Run with 5 working datasets
- Covers all core RAG benchmarks from the paper
- Includes FEVER as a bonus fact-checking benchmark
- No manual intervention needed

**Option 2:** Add MS-MARCO for completeness
- Follow manual download steps above
- Adds generation/span-extraction task to evaluation
- Requires ~15 min of manual setup

---

## Technical Details

### Environment Constraints

**Modal Environment:**
- Python 3.10
- datasets==1.18.0 (required for SearchQA and other dataset scripts)
- pyarrow<15.0.0
- numpy<2.0.0

**Local Environment (for comparison):**
- Python 3.12
- datasets>=2.0 (FlashRAG compatible)
- All constraints relaxed

### Why datasets==1.18.0?

Some dataset loading scripts (like SearchQA's) use deprecated features only available in datasets==1.18.0:
- Dataset script execution (removed in datasets 2.0+)
- Legacy dataset loading APIs

This creates a version conflict:
- ✅ SearchQA needs datasets==1.18.0 (but has data corruption anyway)
- ✗ FlashRAG/MS-MARCO needs datasets>=2.0 (glob pattern support)

**Trade-off:** Pinning to 1.18.0 was initially for SearchQA compatibility, but SearchQA is broken regardless. Could consider upgrading to datasets>=2.0 and skipping SearchQA entirely (since it's already unusable).

---

## Files Modified

1. `prepare_eval_datasets.py`:
   - MS-MARCO: Tries ms_marco v1.1, falls back to Kaggle CSV detection
   - SearchQA: Attempts load with error handling, fails gracefully
   - All datasets: Proper error messages and skip logic

2. `modal_rag_eval.py`:
   - Added streaming output (subprocess.Popen)
   - Added log capture to persistent volume
   - Real-time progress visibility

3. `DATASET_STATUS.md` (this file):
   - Comprehensive documentation of all findings

---

## Next Steps

1. **Run evaluation** with 5 working datasets
2. **Optionally** add MS-MARCO via manual download if needed
3. **Monitor** Microsoft blob storage - may be fixed in future
4. **Check** SearchQA repository for updates (unlikely - appears abandoned)

---

## Quick Start Commands

**Test dataset preparation:**
```bash
modal run modal_rag_eval.py::prepare_datasets --max-samples 5
```

**Run full evaluation (test mode):**
```bash
modal run modal_rag_eval.py --test-mode
```

**Check logs:**
```bash
modal volume get rag-data results/dataset_preparation.log ./dataset_prep.log
cat dataset_prep.log
```
