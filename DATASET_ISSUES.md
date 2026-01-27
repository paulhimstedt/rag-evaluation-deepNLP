# Dataset Download Issues & Solutions

This document details the dataset download issues encountered and potential solutions.

## Test Mode Clarification

When you run `modal run modal_rag_eval.py --test-mode`, the script:

1. **Downloads & prepares** ALL 7 datasets (if not already cached)
2. **Evaluates** only 1 combination: Natural Questions (NQ) × RAG-Sequence model
3. **Purpose**: Quick validation that the pipeline works before committing to the full 21 evaluations
4. **Time**: ~2 hours (includes model download on first run)
5. **Cost**: ~$3-5

The test mode doesn't skip dataset preparation - it only limits the evaluation phase to a single dataset/model combination for faster feedback.

## Dataset Status Summary

| Dataset | Status | Test Samples | Sources | Notes |
|---------|--------|--------------|---------|-------|
| Natural Questions (NQ) | ✅ Working | 3,610 | DPR | None |
| TriviaQA | ✅ Working | 11,313 | DPR | None |
| WebQuestions | ✅ Fixed | 2,032 | DPR, FlashRAG, stanfordnlp | Multiple fallback sources |
| CuratedTrec | ✅ Fixed | 694 | DPR, FlashRAG | FlashRAG as fallback |
| MS-MARCO | ⚠️ Manual | - | Kaggle, HuggingFace | Requires manual Kaggle download |
| SearchQA | ✅ Fixed | 43,228 | kyunghyuncho/search_qa | Config "train_test_val" |
| FEVER | ✅ Fixed | ~10k | fever/fever | Use verification_mode="no_checks" |

## Successful Datasets (No Action Needed)

### 1. Natural Questions (NQ)
- **Source**: DPR (Facebook AI Research)
- **URL**: `https://dl.fbaipublicfiles.com/dpr/data/retriever/nq-{split}.qa.csv`
- **Status**: ✅ Working perfectly
- **Test samples**: 3,610
- **Dev samples**: 8,757

### 2. TriviaQA
- **Source**: DPR (Facebook AI Research)
- **URL**: `https://dl.fbaipublicfiles.com/dpr/data/retriever/trivia-{split}.qa.csv.gz`
- **Status**: ✅ Working perfectly
- **Test samples**: 11,313
- **Dev samples**: 8,837

## Fixed Datasets (New Alternative Sources)

### 3. WebQuestions
- **Status**: ✅ FIXED with multiple fallback sources
- **Primary**: DPR test set ✅
- **Fallback 1**: `RUC-NLPIR/FlashRAG_datasets` with config "webquestions"
- **Fallback 2**: `stanfordnlp/web_questions`
- **Test samples**: 2,032

**Implementation**:
```python
# Now tries DPR first, then FlashRAG, then stanfordnlp
dataset = load_dataset("RUC-NLPIR/FlashRAG_datasets", "webquestions")
# or
dataset = load_dataset("stanfordnlp/web_questions")
```

### 4. CuratedTrec
- **Status**: ✅ FIXED with FlashRAG fallback
- **Primary**: DPR test set ✅
- **Fallback**: `RUC-NLPIR/FlashRAG_datasets` with config "curatedtrec"
- **Test samples**: 694

**Implementation**:
```python
# Now tries DPR first, then FlashRAG
dataset = load_dataset("RUC-NLPIR/FlashRAG_datasets", "curatedtrec")
```

### 5. SearchQA
- **Status**: ✅ FIXED
- **Source**: `kyunghyuncho/search_qa` with config "train_test_val"
- **Test samples**: 43,228

**Implementation**:
```python
dataset = load_dataset("kyunghyuncho/search_qa", "train_test_val")
test_data = dataset['test']
```

### 6. FEVER
- **Status**: ✅ FIXED
- **Source**: `fever/fever` with verification disabled
- **Test samples**: ~10,000

**Implementation**:
```python
# Try fever/fever first, fallback to fever
dataset = load_dataset("fever/fever", "v1.0", verification_mode="no_checks")
# or
dataset = load_dataset("fever", "v1.0", verification_mode="no_checks")
```

## Partially Working Datasets (Manual Download Option)
### 7. MS-MARCO
- **Status**: ⚠️ Requires manual download from Kaggle
- **Error**: `Invalid pattern: '**' can only be an entire path component`
- **Kaggle Source**: https://www.kaggle.com/datasets/parthplc/ms-marco-dataset/data
  - Download `train.csv` and `valid.csv`
  - Place in `eval_datasets/` folder as `ms-marco-train.csv` and `ms-marco-valid.csv`

**Problem**: The error suggests an issue with how the `datasets` library version handles glob patterns. This is likely a compatibility issue between `datasets==1.18.0` and the MS-MARCO dataset structure.

**Solutions implemented**:

1. **Automatic Kaggle detection**: The code now checks for Kaggle CSV files first
2. **HuggingFace fallback**: Tries different versions (v2.1, v1.1) with `trust_remote_code=True`

**Manual steps**:
```bash
# Download from Kaggle (requires Kaggle account)
# Place files in: eval_datasets/ms-marco-train.csv and eval_datasets/ms-marco-valid.csv
```

**Code now handles both**:
```python
# Checks for Kaggle files first, then tries HuggingFace
if kaggle_csv_exists:
    load_from_csv()
else:
    for version in ["v2.1", "v1.1"]:
        dataset = load_dataset("microsoft/ms_marco", version, trust_remote_code=True)
```

### 8. HF_TOKEN Configuration

**Issue**: HF_TOKEN was not being set, which could cause issues with gated or private datasets.

**Solution**: The code now checks for and preserves HF_TOKEN if set in environment.

**To set HF_TOKEN in Modal**:
```bash
# Set Modal secret
modal secret create huggingface HF_TOKEN=your_token_here
```

**In code**:
```python
# Now checks for HF_TOKEN
if 'HF_TOKEN' in os.environ:
    print("✓ HF_TOKEN found, will use for authentication")
```

## Recommended Actions

### Current Status (After Fixes)

**All 7 datasets now have working solutions!**

1. **Fully Automatic**: NQ, TriviaQA, WebQuestions, CuratedTrec, SearchQA, FEVER (6 datasets)
2. **Manual Option**: MS-MARCO (Kaggle download, but HuggingFace fallback also available)

**To run with all datasets:**
```bash
# All datasets should work now
modal run modal_rag_eval.py
```

**If MS-MARCO fails**, download from Kaggle:
1. Go to: https://www.kaggle.com/datasets/parthplc/ms-marco-dataset/data
2. Download `valid.csv`
3. Place in `eval_datasets/ms-marco-valid.csv`
4. Re-run

### Alternative: Focus on Core Datasets

For immediate RAG paper reproduction, **focus on the 6 fully automatic datasets**:
- NQ, TriviaQA (main results)
- WebQuestions, CuratedTrec (supplementary)
- SearchQA, FEVER (additional experiments)

This gives: 6 datasets × 3 models = **18 evaluations** covering all major results from the RAG paper.

## Testing Individual Datasets Locally

You can test dataset preparation locally before running on Modal:

```bash
# Test a specific dataset
python test_local_prep.py nq        # ✅ Should work
python test_local_prep.py triviaqa  # ✅ Should work
python test_local_prep.py searchqa  # ✅ Should work now
python test_local_prep.py fever     # ❌ Will show the error
```

## Impact on Paper Reproduction

The RAG paper's main results (Table 1) focus on:
1. **Natural Questions** - ✅ Working
2. **TriviaQA** - ✅ Working  
3. **WebQuestions** - ⚠️ Partial (test only)
4. **CuratedTrec** - ⚠️ Partial (test only)

These 4 datasets are sufficient to reproduce the core results from the paper. The other datasets (MS-MARCO, SearchQA, FEVER) are supplementary or used for different experiments.

## References

- **DPR Repository**: https://github.com/facebookresearch/DPR (archived)
- **RAG Paper**: https://arxiv.org/abs/2005.11401
- **DPR Paper**: https://arxiv.org/abs/2004.04906
- **SearchQA**: https://huggingface.co/datasets/kyunghyuncho/search_qa
- **FEVER**: https://fever.ai/
- **MS-MARCO**: https://microsoft.github.io/msmarco/

## Questions?

If you encounter other dataset issues:
1. Check the error message carefully
2. Try the alternative sources listed above
3. Search HuggingFace datasets for alternatives
4. Check if the dataset has been renamed or moved
5. Consider if the dataset is essential for your reproduction goals
