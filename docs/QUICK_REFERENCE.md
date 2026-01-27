# RAG Evaluation - Quick Reference

**For colleagues who want to reproduce the evaluation**

---

## 🚀 Quick Start (3 Commands)

```bash
# 1. Prepare datasets locally (once)
python prepare_datasets_local.py

# 2. Upload to Modal (once)
bash upload_datasets_to_modal.sh

# 3. Run evaluation (as needed)
modal run modal_rag_eval.py --test-mode  # Test with 5 samples first
modal run modal_rag_eval.py              # Full evaluation
```

That's it! See [DATASET_PREPARATION_GUIDE.md](DATASET_PREPARATION_GUIDE.md) for details.

---

## 📋 Prerequisites

**One-time setup** (~10 minutes):

```bash
# 1. Clone repository
git clone <repo-url>
cd rag-evaluation-deepNLP

# 2. Install Python dependencies
pip install datasets transformers pandas tqdm

# 3. Install and authenticate Modal
pip install modal
modal setup  # Follow prompts to authenticate
```

---

## 📊 Dataset Status

| Dataset | Status | Samples | Notes |
|---------|--------|---------|-------|
| Natural Questions | ✅ | 3,610 | Core benchmark |
| TriviaQA | ✅ | 11,313 | Core benchmark |
| WebQuestions | ✅ | 2,032 | Core benchmark |
| CuratedTrec | ✅ | 694 | Core benchmark |
| MS-MARCO | ✅ | 101,093 | Works locally, upload to Modal |
| FEVER | ⚠️ | - | May need manual intervention |
| SearchQA | ❌ | - | Data corruption (skip) |

**Result**: 5-6 datasets working, including all 4 core RAG benchmarks ✓

---

## 🔧 Common Commands

### Prepare Datasets

```bash
# Full preparation
python prepare_datasets_local.py

# Test with limited samples
python prepare_datasets_local.py --max-samples 100

# Check what was created
ls -lh eval_datasets/
```

### Upload to Modal

```bash
# Preview what would be uploaded
bash upload_datasets_to_modal.sh --dry-run

# Actually upload
bash upload_datasets_to_modal.sh

# Verify upload
modal volume ls rag-data eval_datasets
```

### Run Evaluation

```bash
# Quick test (5 samples per dataset, ~5 minutes)
modal run modal_rag_eval.py --test-mode

# Full evaluation (~2-3 hours)
modal run modal_rag_eval.py

# Check logs in real-time
modal app logs rag-evaluation --env main
```

### Download Results

```bash
# Download evaluation results
modal volume get rag-data results/evaluation_results.json ./

# Download all results
modal volume get rag-data results/ ./results/
```

---

## 🐛 Troubleshooting

### "No such file or directory: eval_datasets"
```bash
python prepare_datasets_local.py
```

### "modal: command not found"
```bash
pip install modal
modal setup
```

### "MS-MARCO failed"
MS-MARCO works locally. If it fails:
1. Check internet connection
2. See [MSMARCO_SETUP.md](MSMARCO_SETUP.md) for Kaggle option
3. Or skip it (core benchmarks still work)

### "Modal volume put failed"
```bash
# Create volume if it doesn't exist
modal volume create rag-data

# Re-authenticate if needed
modal setup
```

### "Dataset not found in Modal"
```bash
# Check what's uploaded
modal volume ls rag-data eval_datasets

# If empty, upload again
bash upload_datasets_to_modal.sh
```

---

## 📂 File Structure

```
rag-evaluation-deepNLP/
├── prepare_datasets_local.py      # Prepare datasets locally
├── upload_datasets_to_modal.sh    # Upload to Modal volume
├── modal_rag_eval.py              # Run evaluation in Modal
├── eval_datasets/                 # Prepared datasets (created by step 1)
│   ├── nq_test.source
│   ├── nq_test.target
│   └── ... (other datasets)
└── results/                       # Evaluation results (download from Modal)
    ├── evaluation_results.json
    └── evaluation.log
```

---

## 💡 Tips

1. **First time?** Run with `--test-mode` to verify everything works (5 min vs 2 hours)

2. **Sharing with team?** Just share the Modal workspace - everyone uses the same uploaded datasets

3. **Need to update datasets?** Re-run step 1 and 2, Modal will detect the new files

4. **Debugging?** Use `modal app logs rag-evaluation --env main` to see live output

5. **Results?** Download with `modal volume get rag-data results/evaluation_results.json ./`

---

## 📚 Documentation

- **[DATASET_PREPARATION_GUIDE.md](DATASET_PREPARATION_GUIDE.md)** - Complete workflow guide
- **[MSMARCO_SETUP.md](MSMARCO_SETUP.md)** - MS-MARCO specific setup
- **[DATASET_STATUS.md](DATASET_STATUS.md)** - Technical details on all datasets
- **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)** - Common issues and solutions

---

## 🎯 Expected Results

After running full evaluation, you should see:

```json
{
  "nq": {
    "exact_match": 0.44,
    "f1": 0.52
  },
  "triviaqa": {
    "exact_match": 0.56,
    "f1": 0.61
  },
  ...
}
```

Compare with RAG paper results in `IMPLEMENTATION_SUMMARY.md`.

---

## ⏱️ Time Estimates

| Task | Duration | Frequency |
|------|----------|-----------|
| Initial setup | 10 min | Once |
| Prepare datasets | 20 min | Once per update |
| Upload to Modal | 5 min | Once per update |
| Test evaluation | 5 min | As needed |
| Full evaluation | 2-3 hours | As needed |

**Total first-time investment: ~35 minutes**  
**Subsequent evaluations: Just run them (0 setup time)**

---

## 🤝 Team Workflow

**Setup person (once)**:
```bash
python prepare_datasets_local.py
bash upload_datasets_to_modal.sh
```

**Everyone else (just use it)**:
```bash
modal run modal_rag_eval.py --test-mode
modal run modal_rag_eval.py
```

No need for everyone to download datasets - they're on the shared Modal volume!

---

## ❓ Questions?

See full documentation in:
- [DATASET_PREPARATION_GUIDE.md](DATASET_PREPARATION_GUIDE.md) - Complete instructions
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) - Common issues

Or check the existing files to understand the format.
