# RAG Paper Reproduction on Modal

This directory contains the implementation for reproducing RAG paper evaluation results on Modal infrastructure.

## Overview

This implementation evaluates RAG models on all 7 datasets from the paper:
- **Open-Domain QA (from DPR):** Natural Questions (NQ), TriviaQA, WebQuestions, CuratedTrec
- **Generation Tasks:** MS-MARCO NLG, SearchQA (Jeopardy), FEVER

Compares three models:
- `facebook/rag-sequence-nq` - RAG with sequence-level marginalization
- `facebook/rag-token-nq` - RAG with token-level marginalization
- `facebook/bart-large` - BART baseline (no retrieval)

## Files

### New Files Created

1. **`modal_rag_eval.py`** - Main Modal orchestration script
   - Sets up Modal infrastructure (volume, image, stub)
   - Downloads wiki_dpr compressed index (~80-90GB, one-time)
   - Runs 21 evaluations (7 datasets × 3 models)
   - Generates comparison table with paper results

2. **`prepare_eval_datasets.py`** - Dataset preparation script
   - Downloads datasets from DPR and HuggingFace
   - Converts to `.source`/`.target` format expected by `eval_rag.py`
   - Handles all 7 datasets with error handling

3. **`requirements_modal.txt`** - Modal-specific dependencies
   - Pins `datasets==1.18.0` as requested
   - Compatible versions of transformers, faiss, torch, etc.

### Existing Files Used

4. **`eval_rag.py`** - Evaluation script (no modifications needed ✓)
   - Already supports prebuilt index via default parameters
   - When `--index_name` and `--index_path` are not specified, uses cached wiki_dpr

5. **`utils_rag.py`** - Utility functions for metrics (EM, F1)

## Storage Budget (250GB Total)

| Component | Size | Notes |
|-----------|------|-------|
| wiki_dpr compressed index | ~80-90GB | Passages + embeddings + FAISS HNSW |
| RAG model checkpoints | ~4GB | facebook/rag-sequence-nq, rag-token-nq |
| BART baseline | ~2GB | facebook/bart-large |
| Evaluation datasets | ~5-10GB | All 7 datasets combined |
| PyTorch + dependencies | ~10GB | In Modal image |
| Working space | ~50GB | Generation outputs, temp files |
| **Total Estimated** | **~150GB** | **Well under 250GB limit ✓** |

## Usage

### Prerequisites

1. Install Modal:
   ```bash
   pip install modal
   ```

2. Authenticate with Modal:
   ```bash
   modal token new
   ```

### Running Full Evaluation

Run complete evaluation pipeline (all 7 datasets × 3 models):

```bash
modal run modal_rag_eval.py
```

This will:
1. Setup wiki_dpr index (one-time, ~2 hours)
2. Prepare all evaluation datasets (~30 minutes)
3. Run 21 evaluations (~6-12 hours total)
4. Generate comparison table with paper results

### Running Partial Evaluations

**Setup only** (download wiki_dpr index):
```bash
modal run modal_rag_eval.py --setup-only
```

**Prepare datasets only**:
```bash
modal run modal_rag_eval.py --datasets-only
```

**Test mode** (single evaluation for testing):
```bash
modal run modal_rag_eval.py --test-mode
```

### Utility Functions

**List available datasets**:
```bash
modal run modal_rag_eval.py::list_datasets
```

**Clean up results**:
```bash
modal run modal_rag_eval.py::cleanup_results
```

## Dataset Details

### DPR Datasets (CSV format)

Downloaded from Facebook Research DPR repository:

1. **Natural Questions (NQ)**
   - Test: 3,610 samples
   - Format: question, answers (JSON array)
   - URL: `https://dl.fbaipublicfiles.com/dpr/data/retriever/nq-test.qa.csv`

2. **TriviaQA**
   - Test: ~11,000 samples
   - Format: question, answers (gzipped CSV)
   - URL: `https://dl.fbaipublicfiles.com/dpr/data/retriever/trivia-test.qa.csv.gz`

3. **WebQuestions**
   - Test: 2,033 samples
   - Format: question, answers
   - URL: `https://dl.fbaipublicfiles.com/dpr/data/retriever/webquestions-test.qa.csv`

4. **CuratedTrec**
   - Test: 635 samples
   - Format: question, answers
   - URL: `https://dl.fbaipublicfiles.com/dpr/data/retriever/curatedtrec-test.qa.csv`

### HuggingFace Datasets

Loaded via HuggingFace datasets library:

5. **MS-MARCO NLG v2.1**
   - Abstractive question answering
   - Source: `microsoft/ms_marco` (v2.1 config)
   - Alternative: `din0s/msmarco-nlgen`

6. **SearchQA (Jeopardy)**
   - Question generation from Jeopardy answers
   - Source: `lucadiliello/searchqa`
   - Alternative: `kyunghyuncho/search_qa`

7. **FEVER**
   - Fact verification (3-way: supports/refutes/not enough info)
   - Source: `fever/fever` (v1.0)
   - Creates both 3-way and 2-way evaluation sets

### Output Format

All datasets converted to:
- `{dataset}_test.source` - One question/claim per line
- `{dataset}_test.target` - Tab-separated: `question\t['answer1', 'answer2']`

## Implementation Details

### FAISS Index Strategy

✅ **Solution:** Use prebuilt `wiki_dpr` compressed index from HuggingFace

- **What it contains:**
  - ~21 million Wikipedia passages (100-word chunks)
  - 768-dimensional DPR embeddings
  - FAISS HNSW index for efficient retrieval

- **Why compressed variant:**
  - Full `exact` index: ~150GB
  - `compressed` variant: ~80-90GB (fits in 250GB budget)
  - Uses HNSW (Hierarchical Navigable Small World) approximation
  - Negligible accuracy loss for evaluation purposes

- **Download approach:**
  ```python
  from datasets import load_dataset
  dataset = load_dataset("facebook/wiki_dpr", "psgs_w100.nq.compressed")
  ```

- **Key insight:** No manual FAISS index building required! HuggingFace provides prebuilt index that RagRetriever automatically uses when cached.

### Modal Configuration

**GPU Selection:**
- Using `T4` GPU for cost-effectiveness
- Alternative: `A10G` for faster evaluation (higher cost)
- BART baseline could run on CPU-only

**Volume Paths:**
- `/data/huggingface_cache` - HF models and datasets cache
- `/data/eval_datasets` - Prepared evaluation datasets
- `/data/results` - Evaluation outputs and predictions

**Environment Variables:**
```python
HF_HOME = '/data/huggingface_cache'
HF_DATASETS_CACHE = '/data/huggingface_cache/datasets'
TRANSFORMERS_CACHE = '/data/huggingface_cache/transformers'
```

### Evaluation Parameters

Following paper settings:
- **n_docs:** 5 (number of retrieved documents)
- **num_beams:** 4 (beam search width)
- **max_length:** 50 (max answer length)
- **batch_size:** 8 (per GPU)
- **eval_mode:** 'e2e' (end-to-end EM/F1 evaluation)

## Expected Results

From RAG paper Table 1 (test set Exact Match scores):

| Dataset | RAG-Sequence | RAG-Token | Notes |
|---------|--------------|-----------|-------|
| Natural Questions | 44.5 | 44.1 | Open-domain QA |
| TriviaQA | 56.8 | 55.2 | Trivia questions |
| WebQuestions | 45.2 | 45.5 | Web-based QA |
| CuratedTrec | 52.2 | 50.0 | TREC questions |

**Note:** Minor differences expected due to:
- Library version differences (datasets 1.18.0 vs paper's version)
- Potential data preprocessing differences
- Different random seeds for generation
- Compressed vs exact FAISS index

Differences within ±2-3 EM points are normal and acceptable.

## Output Files

After evaluation completes:

1. **`results/evaluation_results.json`**
   - Raw results for all 21 evaluations
   - Contains metrics, status, sample counts

2. **`results/comparison_table.txt`**
   - Formatted comparison table
   - Shows our results vs paper results
   - Highlights significant differences

3. **`results/{dataset}_{model}_preds.txt`**
   - Generated predictions for each evaluation
   - One answer per line

## Troubleshooting

### Common Issues

**1. Index download timeout:**
- Increase timeout in `setup_wiki_index()` function
- Default: 7200s (2 hours), may need more on slow connections

**2. Dataset download failures:**
- Check HuggingFace Hub status
- Try alternative dataset sources (see `prepare_eval_datasets.py`)
- Retry with `--datasets-only` flag

**3. GPU out of memory:**
- Reduce `eval_batch_size` in `run_evaluation()`
- Use smaller GPU or add more memory
- Default: 8, try 4 or 2

**4. Metrics not parsing:**
- Check `eval_rag.py` output format
- Update regex in `parse_metrics_from_output()`

### Debugging

Enable verbose logging in `run_evaluation()`:
```python
cmd.extend(['--print_predictions', '--print_docs'])
```

Check Modal logs:
```bash
modal app logs rag-evaluation
```

## Development Notes

### Design Decisions

1. **No eval_rag.py modifications needed:**
   - Defaults (`index_name=None`, `index_path=None`) work perfectly
   - RagRetriever automatically uses cached wiki_dpr index

2. **Separate preparation script:**
   - Modular design for easier debugging
   - Can prepare datasets locally before Modal run
   - Independent error handling per dataset

3. **Modal volume for persistence:**
   - Index and datasets persist across runs
   - Avoids re-downloading large files
   - Enables incremental evaluation

4. **QA mode for gold data:**
   - Using `--gold_data_mode qa` with tab-separated format
   - Supports multiple correct answers per question
   - Format: `question\t['answer1', 'answer2']`

### Future Enhancements

Possible improvements:
- [ ] Add retrieval-only evaluation mode (precision@k)
- [ ] Test with different n_docs values (10, 50)
- [ ] Evaluate on dev sets in addition to test sets
- [ ] Add support for custom FAISS index
- [ ] Implement error recovery and resume capability
- [ ] Add visualization of results (plots, charts)

## References

- **RAG Paper:** [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401)
- **DPR Repository:** https://github.com/facebookresearch/DPR
- **HuggingFace RAG:** https://huggingface.co/docs/transformers/model_doc/rag
- **wiki_dpr Dataset:** https://huggingface.co/datasets/facebook/wiki_dpr

## Timeline Estimate

Based on Modal infrastructure:

1. **First-time setup:** ~2-3 hours
   - Wiki_dpr index download: ~1.5-2 hours
   - Dataset preparation: ~30 minutes
   - Environment setup: ~10 minutes

2. **Evaluation runs:** ~6-12 hours
   - Per evaluation: ~20-40 minutes (varies by dataset size)
   - 21 evaluations total (7 datasets × 3 models)
   - Can run in parallel with multiple Modal workers

3. **Total end-to-end:** ~12-15 hours
   - Mostly automated, minimal supervision needed

**Cost estimate:** ~$20-40 on Modal (depending on GPU choice and run time)

## License

This implementation follows the license of the original RAG repository (Apache 2.0).
