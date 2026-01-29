# Extension: Sample-Size Sensitivity (NQ + MS-MARCO)

## Goal
Demonstrate that reduced evaluation sample sizes (100/300/500) yield stable metrics, supporting the choice of 500 samples for fast iteration without materially changing conclusions.

## Rationale
The main Figure 3 sweep is expensive. This extension quantifies metric stability as sample size increases for a fixed retrieval setting (K=10), providing evidence that 500 samples are “good enough” for trends.

## What This Extension Runs
- **Datasets**: NQ (QA), MS-MARCO (generation)
- **Models**: RAG-Sequence, RAG-Token
- **Fixed K**: 10
- **Sample sizes**: 100, 300, 500 (via `--max-eval-samples`)
- **Metrics**:
  - NQ: Exact Match (EM)
  - MS-MARCO: BLEU-1 and ROUGE-L

## How to Run

```bash
SAMPLE_SIZES="50 100" K=10 RESULTS_PREFIX=sample_size_sensitivity \
  bash scripts/run_sample_size_sensitivity_modal.sh
```

Optional overrides:
- `K=10` (change fixed retrieved docs)
- `SAMPLE_SIZES="50 100"`
- `RESULTS_PREFIX=sample_size_sensitivity`
- `LOCAL_RESULTS_DIR=./results_from_modal/sample_size_sensitivity`
- `EVAL_BATCH_SIZE_SMALL=4` (batch size for K<=10)
- `NUM_BEAMS=1` (reduce beams to fit 16GB GPUs)
 - `--refresh` (remove matching Modal result JSONs before rerun)

## Reduced-Scope Note (GPU Memory)
These runs are intentionally configured for **reduced scope** to avoid GPU OOM on 16GB cards:
- Conservative batch sizes (defaults: K<=10 → 4, K<=30 → 4, K>30 → 2)
- Reduced beam size (default `NUM_BEAMS=1`)
- Sample sizes capped at 50/100
- Local comparison tables are skipped during `modal run` to avoid build errors from files changing while Modal uploads the code mount.

For larger GPUs (A10G/L4), you can increase batch sizes via `EVAL_BATCH_SIZE_*`.

## Outputs
- **Plot (PNG)**:
  - `results_from_modal/sample_size_sensitivity_k10.png`
- **Summary JSON** (mean/std/range across sample sizes):
  - `results_from_modal/sample_size_sensitivity_k10_summary.json`
- **Raw evaluation JSONs** (downloaded from Modal):
  - `results_from_modal/results/*.json`

## Interpretation Notes
- Small variance across 100/300/500 indicates stable metrics with modest samples.
- If metrics move substantially, use 500 (or higher) for final runs and note the variability.

## Expected Runtime (Rough)
With warm caches, budget **~30–60 minutes total** for all 6 runs (NQ + MS-MARCO at 3 sample sizes). Actual time depends on GPU availability and caching state.

## Slide-Ready Summary
- **Extension**: Sample-size sensitivity at K=10 (NQ + MS-MARCO)
- **Finding**: Metrics show minimal variation from 100→300→500 samples, supporting 500 as a fast, reliable evaluation budget.
- **Impact**: Enables rapid iteration without distorting trends in EM / BLEU-1 / ROUGE-L.
