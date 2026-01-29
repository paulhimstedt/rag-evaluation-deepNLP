# Evaluation Recovery Plan (Time-Constrained)

This document describes how to salvage partial results, stop a long-running Modal job, and run
time-bounded evaluations that still produce defensible tables.

## 0) Stop the current sequential run

If a long Modal run is still executing, stop it to avoid burning time/budget:

1. Modal UI: stop the running app in the dashboard **or**
2. CLI:
   ```bash
   modal app list
   modal app stop <app_id>
   ```

## 1) Pull intermediate artifacts from Modal

The main results JSON is only written at the end of a run, but **per-eval predictions** are written
as soon as each dataset/model finishes.

```bash
modal volume ls rag-data results
modal volume get rag-data results ./results_from_modal
```

Look for files like:
```
results/nq_rag_sequence_preds.txt
results/triviaqa_rag_token_preds.txt
...
```

## 2) Recompute metrics locally from predictions

`eval_rag.py` can compute EM/F1 from predictions without re-running models (if `--predictions_path`
exists). Example:

```bash
python eval_rag.py \
  --model_name_or_path facebook/rag-sequence-nq \
  --model_type rag_sequence \
  --evaluation_set eval_datasets/nq_test.source \
  --gold_data_path eval_datasets/nq_test.target \
  --gold_data_mode qa \
  --predictions_path results_from_modal/nq_rag_sequence_preds.txt \
  --eval_mode e2e
```

Repeat for each completed dataset/model. This gives you defensible EM/F1 for Table 1 datasets.

## 3) Run time-bounded evaluations for large datasets

The evaluation script now supports `--max_eval_samples` (limits evaluation size without changing
the dataset files). Use this to get **partial but consistent** numbers for MSMARCO, SearchQA,
FEVER.

Examples:

```bash
modal run modal_rag_eval.py \
  --datasets msmarco \
  --models rag_sequence,rag_token,bart \
  --max-eval-samples 5000 \
  --results-file evaluation_results_msmarco_5k.json

modal run modal_rag_eval.py \
  --datasets searchqa \
  --models rag_sequence,rag_token,bart \
  --max-eval-samples 10000 \
  --results-file evaluation_results_searchqa_10k.json

modal run modal_rag_eval.py \
  --datasets fever_3way \
  --models rag_sequence,rag_token,bart \
  --max-eval-samples 10000 \
  --results-file evaluation_results_fever_10k.json
```

These runs complete much faster and still generate comparable internal metrics. Note: they are **not**
paper-comparable due to dataset truncation and metric mismatches.

## 4) Aggregate results into a single JSON (optional)

If you run per-dataset jobs, aggregate on the Modal volume:

```bash
modal run modal_rag_eval.py::aggregate_results --results-prefix evaluation_results
```

Then download:

```bash
modal volume get rag-data results ./results_from_modal
```

## 5) Table feasibility and what you can report

### Table 1 (Open-Domain QA EM)
✅ **Feasible**: NQ, TriviaQA, WebQuestions, CuratedTrec  
Uses EM from `eval_rag.py` (directly comparable to the paper’s EM table for RAG-Token/Seq).

### Table 2 (Generation + classification)
⚠ **Partially feasible**:
- MSMARCO / SearchQA in the paper use BLEU/ROUGE/Q-BLEU. Our current pipeline reports EM/F1 only.
- FEVER 3-way can be treated as label accuracy (EM), but still not directly comparable to the paper’s
  reported accuracy if you truncate.

You can still report **internal EM/F1** as “non‑paper‑comparable metrics” for partial findings.

### Table 5 (Distinct-3 ratio)
❌ **Not available** in current pipeline. We do not compute distinct-3 or Q‑BLEU.
Would require a new evaluation script that computes these metrics from `*_preds.txt`.

### Table 6 (Ablations: BM25 / Frozen)
❌ **Not available** in current pipeline. Those variants are not implemented in this repo.

## 6) BART baseline fix

`eval_rag.py` now handles BART correctly (no retriever usage) using a standard tokenizer. This
unblocks BART baselines for the QA-style datasets.
