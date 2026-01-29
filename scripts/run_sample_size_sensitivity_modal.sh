#!/usr/bin/env bash
set -euo pipefail

# Sample-size sensitivity runner for Modal (K fixed).
# Customize via env vars:
#   SAMPLE_SIZES="100 300 500"
#   K=10
#   RESULTS_PREFIX="sample_size_sensitivity"

SAMPLE_SIZES="${SAMPLE_SIZES:-"100 300 500"}"
K="${K:-10}"
RESULTS_PREFIX="${RESULTS_PREFIX:-sample_size_sensitivity}"

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

echo "SAMPLE_SIZES: ${SAMPLE_SIZES}"
echo "K: ${K}"
echo "RESULTS_PREFIX: ${RESULTS_PREFIX}"

eval_bs="$(select_eval_batch_size "$K")"

for s in ${SAMPLE_SIZES}; do
  modal run modal_rag_eval.py::main \
    --datasets nq \
    --models rag_sequence,rag_token \
    --n-docs "${K}" \
    --eval-batch-size "${eval_bs}" \
    --max-eval-samples "${s}" \
    --results-file "${RESULTS_PREFIX}_nq_k${K}_s${s}.json"

done

for s in ${SAMPLE_SIZES}; do
  modal run modal_rag_eval.py::main \
    --datasets msmarco \
    --models rag_sequence,rag_token \
    --n-docs "${K}" \
    --eval-batch-size "${eval_bs}" \
    --max-eval-samples "${s}" \
    --results-file "${RESULTS_PREFIX}_msmarco_k${K}_s${s}.json"

done

mkdir -p results_from_modal
modal volume get rag-data results ./results_from_modal

python scripts/plot_sample_size_sensitivity.py \
  --results-dir ./results_from_modal \
  --k "${K}" \
  --sample-sizes "$(echo "${SAMPLE_SIZES}" | tr ' ' ',')" \
  --output "./results_from_modal/${RESULTS_PREFIX}_k${K}.png" \
  --summary-output "./results_from_modal/${RESULTS_PREFIX}_k${K}_summary.json"
