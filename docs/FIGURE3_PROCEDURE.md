# Figure 3 Procedure (K‑Sweep)

This document explains how to run the reduced‑scope Figure 3 sweep and generate the plot.

## What This Produces
- **NQ**: Exact Match vs K
- **NQ Retrieval**: Recall@K vs K
- **MS‑MARCO**: BLEU‑1 / ROUGE‑L vs K

## Default Scope (Safe on 16GB GPUs)
The script is configured to avoid GPU OOM:
- `NUM_BEAMS=1`
- Conservative batch sizes (K<=10 → 4, K<=30 → 4, K>30 → 2)
- Sample caps controlled by env vars

## Run (Reduced Scope)

```bash
N_DOCS_LIST="1 5 10 15" \
NQ_SAMPLES=500 NQ_RETRIEVAL_SAMPLES=500 MSMARCO_SAMPLES=500 \
EVAL_BATCH_SIZE_MEDIUM=2 NUM_BEAMS=1 \
bash scripts/run_figure3_modal.sh
```

### Refresh Mode
If you want a clean rerun (remove matching JSONs from Modal + local folder):

```bash
N_DOCS_LIST="1 5 10 15" bash scripts/run_figure3_modal.sh --refresh
```

## Outputs
All results are downloaded into a **dedicated folder**:
- JSONs: `results_from_modal/figure3/*.json`
- PNG plot: `results_from_modal/figure3/figure3_figure3.png`

## Plot Only (if you already ran the jobs)
```bash
python scripts/plot_figure3.py \
  --results-dir ./results_from_modal/figure3 \
  --output ./results_from_modal/figure3/figure3_figure3.png
```

## Notes / Gotchas
- **Modal mount error**: Do not edit files in the repo while `modal run ...` is starting.
- **MS‑MARCO scoring**: EM/F1 is skipped due to target formatting; BLEU‑1/ROUGE‑L is computed separately.
- **Predictions reuse**: If a predictions file already exists on the volume, it will be reused. Use `--refresh` if you want a clean run.
