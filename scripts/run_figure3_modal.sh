#!/usr/bin/env bash
set -euo pipefail

# Figure 3 sweep runner for Modal.
# Customize via env vars:
#   N_DOCS_LIST="5 10 20 30 40 50"
#   NQ_SAMPLES=500
#   NQ_RETRIEVAL_SAMPLES=500
#   MSMARCO_SAMPLES=500
#   RESULTS_PREFIX="figure3"
#   LOCAL_RESULTS_DIR="./results_from_modal/figure3"
#   NUM_BEAMS=1

N_DOCS_LIST="${N_DOCS_LIST:-"5 10 20 30 40 50"}"
NQ_SAMPLES="${NQ_SAMPLES:-500}"
NQ_RETRIEVAL_SAMPLES="${NQ_RETRIEVAL_SAMPLES:-500}"
MSMARCO_SAMPLES="${MSMARCO_SAMPLES:-500}"
RESULTS_PREFIX="${RESULTS_PREFIX:-figure3}"
NUM_BEAMS="${NUM_BEAMS:-1}"
LOCAL_RESULTS_DIR="${LOCAL_RESULTS_DIR:-./results_from_modal/figure3}"

REFRESH=0
for arg in "$@"; do
  case "$arg" in
    --refresh) REFRESH=1 ;;
    *)
      echo "Unknown argument: $arg"
      exit 1
      ;;
  esac
done

echo "N_DOCS_LIST: ${N_DOCS_LIST}"
echo "NQ_SAMPLES: ${NQ_SAMPLES}"
echo "NQ_RETRIEVAL_SAMPLES: ${NQ_RETRIEVAL_SAMPLES}"
echo "MSMARCO_SAMPLES: ${MSMARCO_SAMPLES}"
echo "RESULTS_PREFIX: ${RESULTS_PREFIX}"
echo "NUM_BEAMS: ${NUM_BEAMS}"
echo "LOCAL_RESULTS_DIR: ${LOCAL_RESULTS_DIR}"
echo "REFRESH: ${REFRESH}"

select_eval_batch_size() {
  local k="$1"
  if [ "$k" -le 10 ]; then
    echo "${EVAL_BATCH_SIZE_SMALL:-4}"
  elif [ "$k" -le 30 ]; then
    echo "${EVAL_BATCH_SIZE_MEDIUM:-4}"
  else
    echo "${EVAL_BATCH_SIZE_LARGE:-2}"
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
    --num-beams "${NUM_BEAMS}" \
    --skip-local-results \
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
    --num-beams "${NUM_BEAMS}" \
    --skip-local-results \
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
    --num-beams "${NUM_BEAMS}" \
    --skip-local-results \
    --results-file "${RESULTS_PREFIX}_msmarco_k${k}.json"
done

remote_has_file() {
  local fname="$1"
  modal volume ls rag-data results | grep -q "results/${fname}"
}

download_result() {
  local fname="$1"
  if remote_has_file "$fname"; then
    modal volume get rag-data "results/${fname}" "${LOCAL_RESULTS_DIR}/${fname}" --force
  else
    echo "⚠ Missing on volume: results/${fname}"
  fi
}

if [ "$REFRESH" -eq 1 ]; then
  echo "Refreshing remote results for ${RESULTS_PREFIX}..."
  for k in ${N_DOCS_LIST}; do
    modal volume rm rag-data "results/${RESULTS_PREFIX}_nq_k${k}.json" && \
      echo "✓ removed results/${RESULTS_PREFIX}_nq_k${k}.json" || \
      echo "· missing results/${RESULTS_PREFIX}_nq_k${k}.json"
    modal volume rm rag-data "results/${RESULTS_PREFIX}_nq_retrieval_k${k}.json" && \
      echo "✓ removed results/${RESULTS_PREFIX}_nq_retrieval_k${k}.json" || \
      echo "· missing results/${RESULTS_PREFIX}_nq_retrieval_k${k}.json"
    modal volume rm rag-data "results/${RESULTS_PREFIX}_msmarco_k${k}.json" && \
      echo "✓ removed results/${RESULTS_PREFIX}_msmarco_k${k}.json" || \
      echo "· missing results/${RESULTS_PREFIX}_msmarco_k${k}.json"
  done
  for k in ${N_DOCS_LIST}; do
    for model in rag_sequence rag_token; do
      modal volume rm rag-data "results/nq_${model}_k${k}_preds.txt" && \
        echo "✓ removed results/nq_${model}_k${k}_preds.txt" || \
        echo "· missing results/nq_${model}_k${k}_preds.txt"
      modal volume rm rag-data "results/nq_retrieval_${model}_k${k}_preds.txt" && \
        echo "✓ removed results/nq_retrieval_${model}_k${k}_preds.txt" || \
        echo "· missing results/nq_retrieval_${model}_k${k}_preds.txt"
      modal volume rm rag-data "results/msmarco_${model}_k${k}_preds.txt" && \
        echo "✓ removed results/msmarco_${model}_k${k}_preds.txt" || \
        echo "· missing results/msmarco_${model}_k${k}_preds.txt"
      modal volume rm rag-data "results/msmarco_${model}_k${k}_gen_metrics.json" && \
        echo "✓ removed results/msmarco_${model}_k${k}_gen_metrics.json" || \
        echo "· missing results/msmarco_${model}_k${k}_gen_metrics.json"
    done
  done
  rm -rf "${LOCAL_RESULTS_DIR}"
fi

mkdir -p "${LOCAL_RESULTS_DIR}"
for k in ${N_DOCS_LIST}; do
  download_result "${RESULTS_PREFIX}_nq_k${k}.json"
  download_result "${RESULTS_PREFIX}_nq_retrieval_k${k}.json"
  download_result "${RESULTS_PREFIX}_msmarco_k${k}.json"
done

python scripts/plot_figure3.py \
  --results-dir "${LOCAL_RESULTS_DIR}" \
  --output "${LOCAL_RESULTS_DIR}/${RESULTS_PREFIX}_figure3.png"
