# Experiments: How to Run

This page lists all runnable experiments in this repo and the exact commands.
Assumes you have already prepared and uploaded datasets to Modal unless noted.

## 0) One-time setup (recommended)

```bash
# Prepare datasets locally
python prepare_datasets_local.py

# Upload to Modal volume
bash upload_datasets_to_modal.sh
```

## 1) Core Open-Domain QA (Table 1 baseline)

Full evaluation (all datasets x all models):

```bash
modal run modal_rag_eval.py
```

Smoke test (single NQ + RAG-Sequence):

```bash
modal run modal_rag_eval.py --test-mode
```

Single dataset/model:

```bash
modal run modal_rag_eval.py --datasets nq --models rag_sequence --results-file evaluation_results_nq.json
```

## 2) NQ subset-size experiment (size vs. quality)

```bash
modal run modal_rag_eval.py::nq_subset_experiment --subset-sizes "50,100,500,1000,3610" --models "rag_sequence,rag_token" --results-file nq_subset_results.json
```

If datasets are already uploaded:

```bash
modal run modal_rag_eval.py::nq_subset_experiment --skip-setup
```

## 3) Retrieval-only evaluation (Precision/Recall@K)

Requires the optional `nq_retrieval` dataset prepared by `prepare_eval_datasets.py`.

```bash
modal run modal_rag_eval.py --datasets nq_retrieval --models rag_sequence --eval-mode retrieval --n-docs 5 --results-file evaluation_results_nq_retrieval.json
```

## 4) MS-MARCO generation evaluation

MS-MARCO is optional and needs extra setup. See `docs/MSMARCO_SETUP.md`.
Once prepared, run:

```bash
modal run modal_rag_eval.py --datasets msmarco --models rag_sequence,rag_token,bart --results-file evaluation_results_msmarco.json
```

## 5) Table 3 example generations

```bash
modal run modal_rag_eval.py::table3_examples
```

Skip setup if caches already exist:

```bash
modal run modal_rag_eval.py::table3_examples --skip-setup
```

## 6) Download results from Modal

```bash
modal volume get rag-data results/evaluation_results.json ./
```

You can also download any custom results file you used in `--results-file`.
