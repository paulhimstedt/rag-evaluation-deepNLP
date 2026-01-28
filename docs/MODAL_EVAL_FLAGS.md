# Modal Evaluation CLI Flags

This doc explains the extra CLI flags available in `modal_rag_eval.py` for tighter control and faster test runs.

## Available flags

### Dataset and model selection
- `--datasets nq,triviaqa,webquestions,curatedtrec,msmarco,searchqa,fever_3way`
  - Comma-separated list of datasets to evaluate.
  - Only datasets present on the Modal volume (`eval_datasets/*_test.source`) are used.
- `--models rag_sequence,rag_token,bart`
  - Comma-separated list of models to evaluate.

### Speed and runtime control
- `--n-docs 5`
  - Number of documents retrieved per query.
  - Lower values are faster but reduce quality.
- `--eval-batch-size 8`
  - Batch size for evaluation.
  - Larger can be faster but may hit GPU memory limits.
- `--results-file evaluation_results.json`
  - Output filename for evaluation results JSON.
  - Use unique names for parallel runs to avoid overwriting.
  - Comparison tables are written to `./results/comparison_table_<results-file-stem>.txt`.

### Existing flags (unchanged behavior)
- `--setup-only` : download the wiki_dpr index and exit.
- `--datasets-only` : prepare datasets and exit.
- `--test-mode` : single evaluation (NQ + rag_sequence only). Ignores `--datasets` and `--models`.
- `--max-samples` : only used for in-container dataset preparation. Ignored when you pre-upload files.

## Examples

### Full evaluation (default)
```bash
modal run modal_rag_eval.py
```

### Full evaluation, only one dataset
```bash
modal run modal_rag_eval.py --datasets nq
```

### One dataset + one model
```bash
modal run modal_rag_eval.py --datasets nq --models rag_sequence
```

### Unique results file (parallel-safe)
```bash
modal run modal_rag_eval.py --datasets nq --models rag_sequence --results-file evaluation_results_nq.json
```

### Faster full test on one dataset (max speed)
1) Cache index once:
```bash
modal run modal_rag_eval.py --setup-only
```

2) Cache passages once:
```bash
modal run modal_rag_eval.py::setup_wiki_passages
```

3) Run a fast, full pipeline test on one dataset:
```bash
modal run modal_rag_eval.py --datasets nq --models rag_sequence --n-docs 1 --eval-batch-size 16
```

If you see GPU OOM or instability, lower the batch size:
```bash
modal run modal_rag_eval.py --datasets nq --models rag_sequence --n-docs 1 --eval-batch-size 8
```

### Cache status (wiki_dpr index + passages)
```bash
modal run modal_rag_eval.py::cache_status
```

### Setup passages (wiki_dpr full passages dataset)
```bash
modal run modal_rag_eval.py::setup_wiki_passages
```

### Aggregate results from parallel runs
```bash
modal run modal_rag_eval.py::aggregate_results
```

By default it aggregates all `results/evaluation_results*.json` files on the Modal volume
and writes `results/comparison_table_aggregate.txt`.

## Notes
- `--datasets` filters against the datasets present on the Modal volume. Unknown or missing datasets are ignored with a warning.
- `--test-mode` is a special case for a single quick run and overrides dataset/model selection.
- For the fastest validation of the full loop, use a single dataset + single model + low `--n-docs`.
- This Modal image enables `hf_transfer` for faster Hugging Face Hub downloads.
- `HF_XET_HIGH_PERFORMANCE=1` is set at runtime to enable newer high-performance transfer mode when available.
- Passages + index are cached under `datasets/facebook___wiki_dpr/psgs_w100.nq.compressed` (fast path) and reused during evaluation.
