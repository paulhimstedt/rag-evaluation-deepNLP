# RAG Evaluation Reproduction (DeepNLP)

This repository focuses on **reproducing evaluation results** from the RAG paper (Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks) using the Hugging Face RAG research project as the baseline. The goal is to align our evaluation metrics with the paper’s reported numbers by standardizing dataset preparation, evaluation, and result comparison.

**Reference (original project & paper context):**
- https://github.com/huggingface/transformers-research-projects/tree/main/rag

---

## What’s in this repo

- Dataset preparation for RAG benchmarks
- Modal-based evaluation workflow (repeatable + shareable)
- Result comparison tables against the paper’s reported metrics

If you’re looking for the original Transformers/RAG README, it’s preserved at:
- `docs/README_TRANSFORMERS_ORIGINAL.md`

---

## Setup

### 1) Python environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install modal
```

### 2) Hugging Face token (for dataset access)

Set `HF_TOKEN` locally and in Modal:

```bash
# Local (pick one)
export HF_TOKEN=hf_yourTokenHere
# or follow docs/ENV_SETUP.md to use .env

# Modal secret (required for Modal runs)
modal secret create huggingface HF_TOKEN=hf_yourTokenHere
```

### 3) Authenticate Modal

```bash
modal setup
```

---

## Quick start (recommended workflow)

```bash
# 1. Prepare datasets locally
python prepare_datasets_local.py

# 2. Upload to Modal volume
bash upload_datasets_to_modal.sh

# 3. Run evaluation
modal run modal_rag_eval.py --test-mode  # quick smoke test
modal run modal_rag_eval.py              # full evaluation
```

To download results:

```bash
modal volume get rag-data results/evaluation_results.json ./
```

---

## Documentation

Start here:
- `docs/QUICK_REFERENCE.md` (one‑page workflow)
- `docs/DATASET_PREPARATION_GUIDE.md` (full setup + troubleshooting)
- `docs/WORKFLOW_DIAGRAM.md` (visual overview)
- `docs/EXPERIMENTS.md` (all experiments + exact commands)

Other useful references:
- `docs/DATASET_STATUS.md`
- `docs/TROUBLESHOOTING.md`
- `docs/MSMARCO_SETUP.md`

---

## Notes

- This repo is focused on **evaluation reproduction**, not training.
- Some datasets have known issues or partial support; see `docs/DATASET_STATUS.md`.

