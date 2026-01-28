#!/usr/bin/env bash
set -euo pipefail

# Full evaluation: one Modal run per dataset (parallel by default).
# Uses unique --results-file per dataset to avoid overwriting.

DATASETS=(nq triviaqa webquestions curatedtrec msmarco searchqa fever_3way)
MODELS="rag_sequence,rag_token,bart"
N_DOCS="${N_DOCS:-5}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
EVAL_MODE="${EVAL_MODE:-e2e}"
RESULTS_PREFIX="${RESULTS_PREFIX:-evaluation_results}"
CHECK_CACHE="${CHECK_CACHE:-1}"

if [[ "${CHECK_CACHE}" == "1" ]]; then
  modal run modal_rag_eval.py::cache_status || true
fi

echo "Starting full evaluation with:"
echo "  datasets: ${DATASETS[*]}"
echo "  models: ${MODELS}"
echo "  n_docs: ${N_DOCS}"
echo "  eval_batch_size: ${EVAL_BATCH_SIZE}"
echo "  eval_mode: ${EVAL_MODE}"
echo "  results prefix: ${RESULTS_PREFIX}"
echo

pids=()
for ds in "${DATASETS[@]}"; do
  results_file="${RESULTS_PREFIX}_${ds}.json"
  echo "Launching ${ds} -> ${results_file}"
  modal run modal_rag_eval.py \
    --datasets "${ds}" \
    --models "${MODELS}" \
    --n-docs "${N_DOCS}" \
    --eval-batch-size "${EVAL_BATCH_SIZE}" \
    --eval-mode "${EVAL_MODE}" \
    --results-file "${results_file}" &
  pids+=("$!")
done

echo
echo "Waiting for ${#pids[@]} runs..."
wait
echo "All runs completed."
