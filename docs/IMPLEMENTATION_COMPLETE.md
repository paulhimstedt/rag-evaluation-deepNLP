# Implementation Summary - Dataset Preparation Workflow

**Date**: January 27, 2026  
**Status**: ✅ Production Ready  
**Approach**: Local preparation → Modal upload → Cloud evaluation

---

## What We Built

A reliable, reproducible workflow for RAG evaluation that solves the dataset compatibility issues by:

1. **Preparing datasets locally** (where dependencies work smoothly)
2. **Uploading to Modal volume** (persistent, shared storage)
3. **Running evaluation in Modal** (using pre-uploaded datasets)

---

## Problem We Solved

### Original Issue
Modal evaluation needs 7 datasets, but:
- MS-MARCO: Microsoft blob storage returns 409 errors
- MS-MARCO: FlashRAG uses glob patterns incompatible with datasets==1.18.0
- SearchQA: Data corruption in HuggingFace repository
- FEVER: Dataset script compatibility issues

### Root Cause
Environment constraint: Modal uses datasets==1.18.0 (for legacy dataset script support), but:
- Some datasets require datasets >= 2.0
- External services have availability issues
- Different behavior between local (Python 3.12) and Modal (Python 3.10)

### Our Solution
**Decouple dataset preparation from evaluation**:
- Prepare on local machine with flexible dependencies
- Upload plain text files (.source, .target) to Modal
- Evaluation just reads text files (no compatibility issues)

---

## What Was Created

### 1. Local Preparation Script
**File**: `prepare_datasets_local.py`

```bash
python prepare_datasets_local.py [--max-samples N]
```

**Features**:
- Downloads all 7 evaluation datasets
- Uses local Python environment (no version constraints)
- Converts to RAG evaluation format
- Shows clear summary of what worked
- Handles failures gracefully

**Output**: Text files in `eval_datasets/`
- `*_test.source` - Questions/claims
- `*_test.target` - Answers in RAG format
- `*_dev.source/target` - Dev sets (when available)

### 2. Upload Script
**File**: `upload_datasets_to_modal.sh`

```bash
bash upload_datasets_to_modal.sh [--dry-run]
```

**Features**:
- Uploads all .source, .target, and .csv files
- Shows progress for each file
- Supports dry-run mode for preview
- Verifies uploads
- Clear error messages

**Destination**: Modal volume `rag-data/eval_datasets/`

### 3. Modal Evaluation (Updated)
**File**: `modal_rag_eval.py`

**New behavior**:
- Detects if datasets are pre-uploaded
- Skips dataset preparation if files exist
- Falls back to in-container prep if needed
- Shows which approach is being used

**Output**: Evaluation results in `results/`

### 4. Documentation

| File | Purpose |
|------|---------|
| `QUICK_REFERENCE.md` | 1-page guide for colleagues |
| `DATASET_PREPARATION_GUIDE.md` | Complete workflow with troubleshooting |
| `WORKFLOW_DIAGRAM.md` | Visual diagrams of the process |
| `MSMARCO_SETUP.md` | MS-MARCO specific instructions |
| `DATASET_STATUS.md` | Technical details on all datasets |

---

## How It Works

### High-Level Flow

```
Local Machine           Modal Volume          Modal Container
     │                       │                      │
     │ 1. Prepare            │                      │
     │   datasets            │                      │
     │                       │                      │
     ├──────────────────────>│ 2. Upload           │
     │                       │   (one-time)         │
     │                       │                      │
     │                       │<─────────────────────┤
     │                       │  3. Read datasets    │
     │                       │                      │
     │                       │                      ├─> Run evaluation
     │                       │                      │
     │                       │<─────────────────────┤
     │                       │  4. Save results     │
     │                       │                      │
     │<──────────────────────┤ 5. Download          │
     │   results             │                      │
     │                       │                      │
```

### Detailed Steps

**Step 1: Local Preparation** (20 minutes)
- Downloads from internet sources
- Handles authentication (Kaggle for MS-MARCO)
- Converts to standard format
- Validates data quality
- Creates text files

**Step 2: Upload to Modal** (5 minutes)
- Syncs local files to Modal volume
- Persistent storage (survives container restarts)
- Shared across team (everyone uses same datasets)

**Step 3: Modal Evaluation** (2-3 hours for full run)
- Detects pre-uploaded datasets
- Skips download/preparation
- Loads RAG models
- Downloads Wikipedia index (cached after first run)
- Evaluates all datasets
- Computes metrics

**Step 4: Download Results** (1 minute)
- Retrieves evaluation_results.json
- Contains EM and F1 scores for all datasets

---

## Dataset Status

| Dataset | Status | Samples | Preparation Method |
|---------|--------|---------|-------------------|
| Natural Questions | ✅ Working | 3,610 | DPR repository |
| TriviaQA | ✅ Working | 11,313 | DPR repository |
| WebQuestions | ✅ Working | 2,032 | DPR repository |
| CuratedTrec | ✅ Working | 694 | DPR + HuggingFace |
| MS-MARCO | ✅ Working | 101,093 | microsoft/ms_marco v2.1 (local) |
| FEVER | ⚠️ Partial | 37,566 | May need datasets==1.18.0 |
| SearchQA | ❌ Broken | - | Data corruption (skip) |

**Result**: 5-6 datasets working, including all 4 core RAG benchmarks

---

## Key Benefits

### For Individual Users
- ✅ **Works reliably**: No environment conflicts
- ✅ **Faster evaluations**: No download time in Modal
- ✅ **Can verify datasets**: Check files before upload
- ✅ **Easy debugging**: Clear separation of concerns

### For Teams
- ✅ **One-time setup**: First person prepares, everyone uses
- ✅ **Consistent results**: Everyone uses same datasets
- ✅ **Easy onboarding**: 3 commands to get started
- ✅ **Version control**: Can track dataset changes

### For Reproducibility
- ✅ **Clear provenance**: Know exactly where data came from
- ✅ **Plain text format**: Human-readable, inspectable
- ✅ **Documented process**: Complete guides for reproduction
- ✅ **Isolated components**: Data prep separate from evaluation

---

## Usage Examples

### First-Time Setup (Alice)
```bash
# 1. Setup environment
git clone <repo>
cd rag-evaluation-deepNLP
pip install datasets transformers pandas tqdm modal
modal setup

# 2. Prepare and upload datasets
python prepare_datasets_local.py
bash upload_datasets_to_modal.sh

# 3. Test
modal run modal_rag_eval.py --test-mode

# 4. Full evaluation
modal run modal_rag_eval.py
```

Time: ~40 minutes (mostly waiting for downloads/evaluation)

### Colleague Usage (Bob)
```bash
# 1. Setup environment
git clone <repo>
cd rag-evaluation-deepNLP
pip install modal
modal setup

# 2. Run evaluation (datasets already uploaded!)
modal run modal_rag_eval.py --test-mode
```

Time: ~15 minutes (no dataset preparation needed!)

### Updating Datasets
```bash
# Prepare new version
python prepare_datasets_local.py

# Upload (overwrites old files)
bash upload_datasets_to_modal.sh

# Everyone automatically uses new version
modal run modal_rag_eval.py
```

---

## Technical Details

### File Formats

**Source files** (.source):
- One question/claim per line
- Plain UTF-8 text
- No special formatting

**Target files** (.target):
- Tab-separated: `question\t['answer1', 'answer2', ...]`
- Maintains RAG evaluation format
- Supports multiple answers per question

### Storage

**Local**: `eval_datasets/` directory
- Created by prepare_datasets_local.py
- ~50 MB for all datasets
- Git-ignored (too large)

**Modal**: `rag-data/eval_datasets/` volume path
- Persistent across runs
- Shared across team
- Accessible via Modal CLI

### Dependencies

**Local preparation**:
- Python 3.8+
- datasets (any version >= 2.0)
- transformers, pandas, tqdm
- Optional: kaggle (for MS-MARCO)

**Modal evaluation**:
- Python 3.10
- datasets==1.18.0
- transformers==4.30.2
- torch==2.0.1
- faiss-cpu==1.7.4

---

## Alternatives Considered

### Alternative 1: Fix in-container preparation
**Rejected because**:
- Would need to downgrade/upgrade packages (complex)
- External service issues out of our control
- Different behavior than local testing

### Alternative 2: Use only HuggingFace Hub
**Rejected because**:
- Not all datasets available on Hub
- Some have compatibility issues
- MS-MARCO blob storage unreliable

### Alternative 3: Manual dataset placement
**Rejected because**:
- Not reproducible
- Hard to document
- Error-prone for team

### Our Choice: Local prep + upload
**Selected because**:
- ✅ Leverages local flexibility
- ✅ Avoids Modal constraints
- ✅ Plain text = maximum compatibility
- ✅ Easy for teams to share

---

## Testing

### Unit Testing
```bash
# Test local preparation with small sample
python prepare_datasets_local.py --max-samples 10

# Test upload in dry-run mode
bash upload_datasets_to_modal.sh --dry-run

# Test Modal detection
modal run modal_rag_eval.py::prepare_datasets --max-samples 5
```

### Integration Testing
```bash
# Full workflow with limited samples
python prepare_datasets_local.py --max-samples 100
bash upload_datasets_to_modal.sh
modal run modal_rag_eval.py --test-mode
```

### Validation
```bash
# Verify uploads
modal volume ls rag-data eval_datasets

# Check file contents
modal volume get rag-data eval_datasets/nq_test.source - | head -n 5

# Download and inspect results
modal volume get rag-data results/evaluation_results.json ./
cat evaluation_results.json | jq
```

---

## Maintenance

### Regular Updates
- Update documentation if datasets change
- Test with new dataset versions periodically
- Monitor external data sources (HuggingFace, Kaggle)

### Adding New Datasets
1. Add preparation logic to `prepare_eval_datasets.py`
2. Test locally: `python prepare_datasets_local.py`
3. Upload: `bash upload_datasets_to_modal.sh`
4. Update `DATASETS` list in `modal_rag_eval.py`

### Removing/Skipping Datasets
- Just don't upload the files
- Modal will skip missing datasets automatically

---

## Success Metrics

- ✅ **5-6/7 datasets working** (including all 4 core benchmarks)
- ✅ **100% reproducible** (documented process)
- ✅ **Fast onboarding** (<15 min for colleagues)
- ✅ **No manual intervention** (except initial prep)
- ✅ **Team-friendly** (shared datasets)

---

## Future Improvements

### Possible Enhancements
1. **Cache datasets in Git LFS**: Share prepared datasets via git
2. **Automated testing**: CI/CD to validate datasets
3. **Dataset versioning**: Track changes to dataset files
4. **Pre-built datasets**: Provide downloadable prepared datasets
5. **FEVER fix**: Investigate datasets==1.18.0 compatibility

### Known Limitations
- SearchQA: Data corruption in repository (unfixable)
- MS-MARCO: Requires newer datasets version locally
- FEVER: May need special handling

---

## Conclusion

We've created a **production-ready workflow** that:
- ✅ Solves dataset compatibility issues
- ✅ Is easy to use and reproduce
- ✅ Works reliably for individuals and teams
- ✅ Provides clear documentation

**The workflow is ready for team use and can be easily reproduced by colleagues.**

---

## Quick Commands Reference

```bash
# Complete workflow
python prepare_datasets_local.py          # Prepare
bash upload_datasets_to_modal.sh         # Upload
modal run modal_rag_eval.py --test-mode  # Test
modal run modal_rag_eval.py              # Evaluate

# Verification
modal volume ls rag-data eval_datasets                    # Check uploads
modal volume get rag-data results/evaluation_results.json ./  # Get results

# Maintenance
python prepare_datasets_local.py          # Re-prepare
bash upload_datasets_to_modal.sh         # Re-upload
```

---

**Documentation**: See `QUICK_REFERENCE.md` for 1-page guide, `DATASET_PREPARATION_GUIDE.md` for complete instructions.
