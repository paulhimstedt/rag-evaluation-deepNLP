# Dataset Preparation Guide

**Complete workflow for preparing and uploading RAG evaluation datasets**

This guide shows the **recommended approach**: prepare datasets locally (where all dependencies work smoothly) and upload the prepared files to Modal for evaluation.

---

## Why This Approach?

**Problem**: Modal uses `datasets==1.18.0` for compatibility, but some datasets require newer versions or have external service issues (Microsoft blob 409 errors, glob pattern incompatibilities).

**Solution**: Prepare datasets locally with newer dependencies, then upload the final `.source` and `.target` files to Modal. These are plain text files that work everywhere.

**Benefits**:
- ✅ Works consistently across all environments
- ✅ No dependency conflicts in Modal
- ✅ Faster Modal evaluation (no download time)
- ✅ Easy to reproduce for colleagues
- ✅ Can manually fix/augment datasets if needed

---

## Prerequisites

### Local Environment
```bash
# Python 3.8+ with standard packages
pip install datasets transformers pandas tqdm

# Optional: For MS-MARCO auto-download
pip install kaggle
# Then set up ~/.kaggle/kaggle.json (see MSMARCO_SETUP.md)
```

### Modal Environment
```bash
# Install Modal CLI
pip install modal

# Authenticate (one-time)
modal setup
```

---

## Step-by-Step Workflow

### Step 1: Prepare Datasets Locally

Run the local preparation script:

```bash
# Full preparation (all datasets, full size)
python prepare_datasets_local.py

# Or test with limited samples
python prepare_datasets_local.py --max-samples 100
```

**What happens**:
- Downloads all 7 evaluation datasets
- Converts to RAG evaluation format (.source and .target files)
- Saves to `eval_datasets/` directory
- Shows summary of what worked

**Expected output**:
```
================================================================================
RAG Evaluation - Local Dataset Preparation
================================================================================
Output directory: /path/to/eval_datasets
================================================================================

=== Preparing Natural Questions (NQ) ===
✓ Created eval_datasets/nq_test.source (3610 samples)

=== Preparing TriviaQA ===
✓ Created eval_datasets/triviaqa_test.source (11313 samples)

... (more datasets)

================================================================================
Dataset Preparation Complete!
================================================================================
✓ nq                  :   3610 samples
✓ triviaqa            :  11313 samples
✓ webquestions        :   2032 samples
✓ curatedtrec         :    694 samples
✓ msmarco             : 101093 samples  (if Kaggle configured)
✗ searchqa            :      0 samples  (known data corruption)
✓ fever               :  37566 samples
================================================================================
Total: 6/7 datasets working, 156308 samples

Next steps:
1. Review the prepared datasets in eval_datasets
2. Upload to Modal: bash upload_datasets_to_modal.sh
3. Run evaluation: modal run modal_rag_eval.py --test-mode
```

**Troubleshooting**:
- **MS-MARCO fails**: See [MSMARCO_SETUP.md](MSMARCO_SETUP.md) for Kaggle setup, or download manually
- **Other dataset fails**: Check internet connection, HuggingFace Hub status
- **Import errors**: Install missing packages with pip

### Step 2: Review Prepared Datasets

Check what was created:

```bash
ls -lh eval_datasets/

# Should see files like:
# nq_test.source          - Questions, one per line
# nq_test.target          - Answers in RAG format
# nq_dev.source          - Dev set (if available)
# nq_dev.target
# triviaqa_test.source
# triviaqa_test.target
# ...
```

**File format**:
- `.source` files: One question/claim per line (UTF-8 text)
- `.target` files: Tab-separated `question\t['answer1', 'answer2']` format

Example:
```bash
# View first few questions
head -n 3 eval_datasets/nq_test.source

# View corresponding answers
head -n 3 eval_datasets/nq_test.target
```

### Step 3: Upload to Modal

Use the upload script to sync to Modal volume:

```bash
# Dry run (see what would be uploaded)
bash upload_datasets_to_modal.sh --dry-run

# Actual upload
bash upload_datasets_to_modal.sh
```

**What happens**:
- Finds all `.source`, `.target`, and `.csv` files in `eval_datasets/`
- Uploads each file to Modal volume `rag-data`
- Places them in `eval_datasets/` directory on volume
- Shows progress and verification

**Expected output**:
```
========================================================================
Upload Datasets to Modal Volume
========================================================================
Volume: rag-data
Local:  eval_datasets/
Remote: eval_datasets/
========================================================================

Found 12 .source files and 12 .target files

Uploading dataset files...

Uploading nq_test.source (234K)...
✓ Uploaded nq_test.source

Uploading nq_test.target (456K)...
✓ Uploaded nq_test.target

... (more files)

========================================================================
Upload Complete!
========================================================================

Next Steps:
1. Verify datasets on Modal volume:
   modal volume ls rag-data eval_datasets

2. Run test evaluation (5 samples):
   modal run modal_rag_eval.py --test-mode

3. Run full evaluation:
   modal run modal_rag_eval.py
========================================================================
```

### Step 4: Verify Upload

Check that files are on Modal volume:

```bash
# List uploaded datasets
modal volume ls rag-data eval_datasets

# Check specific file
modal volume get rag-data eval_datasets/nq_test.source - | head -n 5
```

### Step 5: Run Evaluation

Now run the RAG evaluation in Modal:

```bash
# Test run with 5 samples per dataset
modal run modal_rag_eval.py --test-mode

# Full evaluation
modal run modal_rag_eval.py
```

**What happens**:
- Modal detects pre-uploaded datasets
- Skips dataset preparation (uses uploaded files)
- Runs RAG evaluation on all datasets
- Saves results to volume

**Expected output**:
```
================================================================================
Preparing evaluation datasets
================================================================================

✓ Found 12 pre-uploaded dataset files:
  • curatedtrec_test.source (45.2 KB)
  • fever_test.source (2134.7 KB)
  • nq_test.source (234.1 KB)
  • triviaqa_test.source (678.3 KB)
  ...

⚠  Skipping dataset preparation - using pre-uploaded files
   (This is the recommended workflow for reproducibility)

... (evaluation proceeds)
```

---

## Updating Datasets

### Re-prepare and Upload

If you need to update datasets (e.g., fix issues, add more samples):

```bash
# 1. Re-prepare locally
python prepare_datasets_local.py

# 2. Re-upload to Modal
bash upload_datasets_to_modal.sh

# 3. Verify and run
modal volume ls rag-data eval_datasets
modal run modal_rag_eval.py --test-mode
```

### Remove and Re-download

To force re-download from scratch:

```bash
# Remove local files
rm -rf eval_datasets/

# Remove from Modal
modal volume rm rag-data eval_datasets/*.source
modal volume rm rag-data eval_datasets/*.target

# Re-prepare
python prepare_datasets_local.py
bash upload_datasets_to_modal.sh
```

---

## Team Collaboration

### Sharing Prepared Datasets

**Option 1: Upload to Modal (Recommended)**
```bash
# Person A prepares and uploads
python prepare_datasets_local.py
bash upload_datasets_to_modal.sh

# Person B just uses them
modal run modal_rag_eval.py
```

Everyone with access to the Modal workspace can use the same uploaded datasets.

**Option 2: Share Local Files**
```bash
# Create archive
tar -czf eval_datasets.tar.gz eval_datasets/

# Share via Git LFS, cloud storage, etc.
# Colleagues download and extract
tar -xzf eval_datasets.tar.gz

# Then upload to their Modal
bash upload_datasets_to_modal.sh
```

### Reproducibility Checklist

For colleagues to reproduce your setup:

- [ ] Clone repository
- [ ] Install dependencies: `pip install -r requirements.txt`
- [ ] Install Modal: `pip install modal` + `modal setup`
- [ ] Download datasets: `python prepare_datasets_local.py`
- [ ] Upload to Modal: `bash upload_datasets_to_modal.sh`
- [ ] Run evaluation: `modal run modal_rag_eval.py --test-mode`

---

## Troubleshooting

### "modal volume put failed"

**Problem**: Modal volume doesn't exist
```bash
# Create volume
modal volume create rag-data
```

**Problem**: Authentication expired
```bash
# Re-authenticate
modal setup
```

### "No .source files found"

**Problem**: Dataset preparation failed
```bash
# Check what failed
python prepare_datasets_local.py

# Look for error messages in output
# Common issues:
# - MS-MARCO needs Kaggle setup (see MSMARCO_SETUP.md)
# - SearchQA has data corruption (skip it)
# - Network issues (check internet)
```

### "Dataset not found in Modal"

**Problem**: Files not uploaded or wrong path
```bash
# Check what's on volume
modal volume ls rag-data eval_datasets

# If empty, upload again
bash upload_datasets_to_modal.sh

# If files are there but Modal doesn't see them,
# might be a path issue - check EVAL_DATASETS_DIR in modal_rag_eval.py
```

### Modal still tries to download datasets

**Problem**: Modal not detecting uploaded files
```bash
# Check if files exist on volume
modal volume ls rag-data eval_datasets

# If files are there, check modal_rag_eval.py
# Should see: "Found X pre-uploaded dataset files"

# If not, files might be in wrong location
# Correct location: eval_datasets/*.source (not in subdirectories)
```

---

## Advanced Usage

### Custom Dataset Modifications

You can manually edit prepared datasets:

```bash
# Edit locally
vim eval_datasets/nq_test.source

# Re-upload just that file
modal volume put rag-data eval_datasets/nq_test.source eval_datasets/nq_test.source
modal volume put rag-data eval_datasets/nq_test.target eval_datasets/nq_test.target
```

### Subset Evaluation

To evaluate on just specific datasets:

```bash
# Prepare only what you need locally
python prepare_datasets_local.py

# Upload only specific files
modal volume put rag-data eval_datasets/nq_test.source eval_datasets/nq_test.source
modal volume put rag-data eval_datasets/nq_test.target eval_datasets/nq_test.target

# Modify modal_rag_eval.py DATASETS list to include only 'nq'
```

### Downloading Results

After evaluation completes:

```bash
# Download results
modal volume get rag-data results/evaluation_results.json ./

# Download logs
modal volume get rag-data results/evaluation.log ./

# Download all results
modal volume get rag-data results/ ./results/
```

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│  Local Machine                                              │
│                                                             │
│  1. prepare_datasets_local.py                               │
│     • Downloads from HuggingFace/Kaggle                     │
│     • Converts to .source/.target format                    │
│     • Uses newer dependencies (datasets>=2.0)               │
│     • Saves to eval_datasets/                               │
│                                                             │
│  2. upload_datasets_to_modal.sh                             │
│     • Uploads .source/.target files                         │
│     • Plain text → works everywhere                         │
│     • One-time setup per dataset                            │
└─────────────────────────────────────────────────────────────┘
                           ↓ Upload
┌─────────────────────────────────────────────────────────────┐
│  Modal Volume: rag-data                                     │
│                                                             │
│  eval_datasets/                                             │
│  ├── nq_test.source                                         │
│  ├── nq_test.target                                         │
│  ├── triviaqa_test.source                                   │
│  ├── triviaqa_test.target                                   │
│  └── ... (other datasets)                                   │
└─────────────────────────────────────────────────────────────┘
                           ↓ Read
┌─────────────────────────────────────────────────────────────┐
│  Modal Container                                            │
│                                                             │
│  modal_rag_eval.py                                          │
│  • Detects pre-uploaded files                               │
│  • Skips dataset preparation                                │
│  • Loads .source/.target directly                           │
│  • Runs RAG evaluation                                      │
│  • Saves results to volume                                  │
└─────────────────────────────────────────────────────────────┘
```

**Key insight**: By preparing locally and uploading text files, we avoid all environment compatibility issues while maintaining full reproducibility.

---

## Summary

**Recommended workflow**:
1. ✅ `python prepare_datasets_local.py` - Prepare locally
2. ✅ `bash upload_datasets_to_modal.sh` - Upload to Modal
3. ✅ `modal run modal_rag_eval.py` - Run evaluation

**Benefits**:
- No dependency conflicts
- Faster evaluation (no download time)
- Easy to reproduce
- Can manually fix datasets
- Works for entire team

**Time investment**:
- First time: ~30 min (download + upload)
- After that: Datasets are ready for everyone
- Each evaluation: Just run it (no dataset prep)
