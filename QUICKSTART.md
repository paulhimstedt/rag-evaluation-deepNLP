# Quick Start Guide - RAG Evaluation on Modal

This guide will help you quickly set up and run RAG evaluations on Modal.

## Prerequisites

1. **Python 3.8+** installed
2. **Modal account** (sign up at https://modal.com)
3. **Modal CLI** installed

## Step 1: Install Modal

```bash
pip install modal
```

## Step 2: Authenticate with Modal

```bash
modal token new
```

This will open a browser window for authentication.

## Step 3: Test Dataset Preparation Locally (Optional but Recommended)

Test downloading a single dataset first:

```bash
python test_local_prep.py nq
```

This will:
- Download the Natural Questions dataset
- Convert it to evaluation format
- Verify the files were created correctly
- Show a sample question and answer

Expected output:
```
Testing nq dataset preparation
...
✓ nq preparation successful: 3610 samples

Verifying nq files:
  ✓ Source file: ./eval_datasets_test/nq_test.source (3610 lines)
  ✓ Target file: ./eval_datasets_test/nq_test.target (3610 lines)

  Sample from nq:
    Q: who sings does he love me with reba...
    A: who sings does he love me with reba→['Linda Davis']
```

## Step 4: Run Full Modal Evaluation

### Option A: Complete Evaluation (All Datasets × All Models)

```bash
modal run modal_rag_eval.py
```

**Time:** ~12-15 hours
**Cost:** ~$20-40
**Evaluations:** 21 (7 datasets × 3 models)

This will:
1. Download wiki_dpr index (~2 hours, one-time)
2. Prepare all datasets (~30 minutes)
3. Run all evaluations (~10-12 hours)
4. Generate comparison table

**Note:** First run includes model downloads (~2GB per model, 3 models total). Subsequent runs skip model downloads.

### Option B: Test Mode (Single Evaluation)

```bash
modal run modal_rag_eval.py --test-mode
```

**Time:** ~2-3 hours (includes model download on first run)
**Cost:** ~$3-5
**Evaluations:** 1 (NQ × RAG-Sequence only)

Use this to verify everything works before running the full evaluation.

**Note:** The first run will take longer as it downloads the RAG model (~2GB). Subsequent runs will be faster since models are cached.

## Step 5: View Results

Results are saved to the Modal volume and also available locally:

```bash
# View comparison table
cat results/comparison_table.txt

# View raw results JSON
cat results/evaluation_results.json
```

Example output:
```
====================================================================================================
RAG Evaluation Results - Comparison with Paper
====================================================================================================
Dataset         Model           EM (Ours)    EM (Paper)   Diff       Status
----------------------------------------------------------------------------------------------------
nq              rag_sequence    44.30        44.5         -0.2       ✓ Close
nq              rag_token       44.10        44.1         +0.0       ✓ Close
triviaqa        rag_sequence    56.50        56.8         -0.3       ✓ Close
...
====================================================================================================
```

## Troubleshooting

### Issue: Dataset download fails

**Solution:** See [DATASET_ISSUES.md](DATASET_ISSUES.md) for comprehensive documentation of dataset issues and solutions.

Quick summary of dataset status:
- ✅ **Working**: NQ, TriviaQA
- ⚠️ **Partial**: WebQuestions (test only), CuratedTrec (test only)  
- ❌ **Failed**: MS-MARCO, FEVER
- ✅ **Fixed**: SearchQA (code updated)

You can still run evaluations with the 4 working datasets. To test locally:
```bash
python test_local_prep.py [dataset_name]
```

### Issue: Modal authentication fails

**Solution:**
```bash
modal token new --force
```

### Issue: GPU out of memory

**Solution:** Edit `modal_rag_eval.py` and reduce batch size:
```python
'--eval_batch_size', '4',  # Changed from 8
```

### Issue: Index download times out

**Solution:** Increase timeout in `setup_wiki_index()`:
```python
timeout=10800,  # 3 hours instead of 2
```

### Issue: HFValidationError with wiki_dpr index

If you see an error like:
```
HFValidationError: Repo id must be in the form 'repo_name' or 'namespace/repo_name': 
'https://storage.googleapis.com/huggingface-nlp/datasets/wiki_dpr/'
```

**Solution:** This is already fixed in the code. The script explicitly specifies `--index_name compressed` when calling eval_rag.py. Make sure you have the latest version of modal_rag_eval.py.

### Issue: Evaluation timeout (FunctionTimeoutError)

If you see:
```
FunctionTimeoutError: Task's current input hit its timeout of 3600s
```

**Solution:** This is already fixed. The timeout is now set to 7200s (2 hours) per evaluation. The first run takes longer due to model downloads. If you still encounter timeouts with slow connections, you can increase it further in [modal_rag_eval.py](modal_rag_eval.py#L265):
```python
timeout=10800,  # 3 hours if needed
```

## Advanced Usage

### Run Specific Components

**Setup wiki_dpr index only:**
```bash
modal run modal_rag_eval.py --setup-only
```

**Prepare datasets only:**
```bash
modal run modal_rag_eval.py --datasets-only
```

### List Available Datasets

```bash
modal run modal_rag_eval.py::list_datasets
```

### Clean Up Results

```bash
modal run modal_rag_eval.py::cleanup_results
```

### Monitor Progress

Check Modal dashboard: https://modal.com/apps

Or view logs:
```bash
modal app logs rag-evaluation
```

## File Structure

After running, you'll have:

```
rag/
├── modal_rag_eval.py          # Main Modal script
├── prepare_eval_datasets.py   # Dataset preparation
├── eval_rag.py                # Evaluation script (existing)
├── utils_rag.py               # Utilities (existing)
├── requirements_modal.txt     # Dependencies
├── test_local_prep.py         # Local testing script
├── eval_datasets/             # Prepared datasets (on Modal volume)
└── results/                   # Evaluation results
    ├── evaluation_results.json
    ├── comparison_table.txt
    └── *_preds.txt            # Predictions per evaluation
```

## Cost Estimation

Approximate costs on Modal:

| Operation | Time | Cost |
|-----------|------|------|
| Index download (one-time) | 2 hours | ~$2 |
| Dataset prep | 30 min | ~$1 |
| Model download (one-time, per model) | 15-30 min | Included |
| Single evaluation (T4) | 45-60 min | ~$1-2 |
| Full evaluation (21 evals) | 12-15 hours | ~$25-40 |
| **Total first run** | **~18 hours** | **~$30-45** |
| **Subsequent runs** | **~15 hours** | **~$25-40** |

**Note:** First run includes downloading 3 RAG models (~6GB total). Subsequent runs are faster.

## Expected Results

You should see Exact Match (EM) scores within ±2-3 points of paper results:

| Dataset | Model | Paper EM | Expected Range |
|---------|-------|----------|----------------|
| NQ | RAG-Seq | 44.5 | 42-47 |
| NQ | RAG-Token | 44.1 | 42-46 |
| TriviaQA | RAG-Seq | 56.8 | 54-59 |
| TriviaQA | RAG-Token | 55.2 | 53-58 |
| WebQuestions | RAG-Seq | 45.2 | 43-48 |
| WebQuestions | RAG-Token | 45.5 | 43-48 |
| CuratedTrec | RAG-Seq | 52.2 | 50-55 |
| CuratedTrec | RAG-Token | 50.0 | 48-53 |

## Next Steps

After successful evaluation:

1. **Analyze Results:** Compare with paper tables
2. **Investigate Differences:** If EM differs by >3 points
3. **Try Variations:** Test different n_docs values (10, 50)
4. **Extend:** Add custom datasets or models

## Getting Help

- **Modal Documentation:** https://modal.com/docs
- **RAG Paper:** https://arxiv.org/abs/2005.11401
- **HuggingFace RAG:** https://huggingface.co/docs/transformers/model_doc/rag

## FAQ

**Q: Can I use my own FAISS index?**
A: Yes, modify `run_evaluation()` to add `--index_path /path/to/index`

**Q: Can I evaluate on dev sets instead of test sets?**
A: Yes, modify dataset paths to use `*_dev.source` instead of `*_test.source`

**Q: Can I use a different GPU?**
A: Yes, edit `modal_rag_eval.py` and change `gpu="T4"` to `gpu="A10G"` or `gpu="A100"`

**Q: How do I add a custom dataset?**
A: Add a preparation function in `prepare_eval_datasets.py` and add the dataset name to the `DATASETS` list in `modal_rag_eval.py`

**Q: Can I run this without Modal?**
A: Yes, but you'll need to manually download the wiki_dpr index and set up the environment. The Modal script can be adapted for local execution.
