#!/usr/bin/env bash
set -euo pipefail

# Figure 3 sweep runner for Modal.
# Customize via env vars:
#   N_DOCS_LIST="5 10 20 30 40 50"
#   NQ_SAMPLES=500
#   NQ_RETRIEVAL_SAMPLES=500
#   MSMARCO_SAMPLES=500
#   RESULTS_PREFIX="figure3"

N_DOCS_LIST="${N_DOCS_LIST:-"5 10 20 30 40 50"}"
NQ_SAMPLES="${NQ_SAMPLES:-500}"
NQ_RETRIEVAL_SAMPLES="${NQ_RETRIEVAL_SAMPLES:-500}"
MSMARCO_SAMPLES="${MSMARCO_SAMPLES:-500}"
RESULTS_PREFIX="${RESULTS_PREFIX:-figure3}"

echo "N_DOCS_LIST: ${N_DOCS_LIST}"
echo "NQ_SAMPLES: ${NQ_SAMPLES}"
echo "NQ_RETRIEVAL_SAMPLES: ${NQ_RETRIEVAL_SAMPLES}"
echo "MSMARCO_SAMPLES: ${MSMARCO_SAMPLES}"
echo "RESULTS_PREFIX: ${RESULTS_PREFIX}"

select_eval_batch_size() {
  local k="$1"
  if [ "$k" -le 10 ]; then
    echo "${EVAL_BATCH_SIZE_SMALL:-8}"
  elif [ "$k" -le 30 ]; then
    echo "${EVAL_BATCH_SIZE_MEDIUM:-6}"
  else
    echo "${EVAL_BATCH_SIZE_LARGE:-4}"
  fi
}

for k in ${N_DOCS_LIST}; do
  eval_bs="$(select_eval_batch_size "$k")"
  modal run modal_rag_eval.py::main \
    --datasets nq \
    --models rag_sequence,rag_token \
    --n-docs "${k}" \
    --eval-batch-size "${eval_bs}" \
    --max-eval-samples "${NQ_SAMPLES}" \
    --results-file "${RESULTS_PREFIX}_nq_k${k}.json"
done

for k in ${N_DOCS_LIST}; do
  eval_bs="$(select_eval_batch_size "$k")"
  modal run modal_rag_eval.py::main \
    --datasets nq_retrieval \
    --models rag_sequence,rag_token \
    --n-docs "${k}" \
    --eval-mode retrieval \
    --eval-batch-size "${eval_bs}" \
    --max-eval-samples "${NQ_RETRIEVAL_SAMPLES}" \
    --results-file "${RESULTS_PREFIX}_nq_retrieval_k${k}.json"
done

for k in ${N_DOCS_LIST}; do
  eval_bs="$(select_eval_batch_size "$k")"
  modal run modal_rag_eval.py::main \
    --datasets msmarco \
    --models rag_sequence,rag_token \
    --n-docs "${k}" \
    --eval-batch-size "${eval_bs}" \
    --max-eval-samples "${MSMARCO_SAMPLES}" \
    --results-file "${RESULTS_PREFIX}_msmarco_k${k}.json"
done

mkdir -p results_from_modal
modal volume get rag-data results ./results_from_modal

python scripts/plot_figure3.py \
  --results-dir ./results_from_modal \
  --output "./results_from_modal/${RESULTS_PREFIX}_figure3.png"
