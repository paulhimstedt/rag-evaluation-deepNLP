#!/usr/bin/env bash
set -euo pipefail

# Sample-size sensitivity runner for Modal (K fixed).
# Customize via env vars:
#   SAMPLE_SIZES="100 300 500"
#   K=10
#   RESULTS_PREFIX="sample_size_sensitivity"
#   LOCAL_RESULTS_DIR="./results_from_modal/sample_size_sensitivity"
#   NUM_BEAMS=1

SAMPLE_SIZES="${SAMPLE_SIZES:-"50 100"}"
K="${K:-10}"
RESULTS_PREFIX="${RESULTS_PREFIX:-sample_size_sensitivity}"
NUM_BEAMS="${NUM_BEAMS:-1}"
LOCAL_RESULTS_DIR="${LOCAL_RESULTS_DIR:-./results_from_modal/sample_size_sensitivity}"

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

echo "SAMPLE_SIZES: ${SAMPLE_SIZES}"
echo "K: ${K}"
echo "RESULTS_PREFIX: ${RESULTS_PREFIX}"
echo "NUM_BEAMS: ${NUM_BEAMS}"
echo "LOCAL_RESULTS_DIR: ${LOCAL_RESULTS_DIR}"
echo "REFRESH: ${REFRESH}"

eval_bs="$(select_eval_batch_size "$K")"

for s in ${SAMPLE_SIZES}; do
  modal run modal_rag_eval.py::main \
    --datasets nq \
    --models rag_sequence,rag_token \
    --n-docs "${K}" \
    --eval-batch-size "${eval_bs}" \
    --max-eval-samples "${s}" \
    --num-beams "${NUM_BEAMS}" \
    --skip-local-results \
    --results-file "${RESULTS_PREFIX}_nq_k${K}_s${s}.json"

done

for s in ${SAMPLE_SIZES}; do
  modal run modal_rag_eval.py::main \
    --datasets msmarco \
    --models rag_sequence,rag_token \
    --n-docs "${K}" \
    --eval-batch-size "${eval_bs}" \
    --max-eval-samples "${s}" \
    --num-beams "${NUM_BEAMS}" \
    --skip-local-results \
    --results-file "${RESULTS_PREFIX}_msmarco_k${K}_s${s}.json"

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
  for s in ${SAMPLE_SIZES}; do
    modal volume rm rag-data "results/${RESULTS_PREFIX}_nq_k${K}_s${s}.json" && \
      echo "✓ removed results/${RESULTS_PREFIX}_nq_k${K}_s${s}.json" || \
      echo "· missing results/${RESULTS_PREFIX}_nq_k${K}_s${s}.json"
    modal volume rm rag-data "results/${RESULTS_PREFIX}_msmarco_k${K}_s${s}.json" && \
      echo "✓ removed results/${RESULTS_PREFIX}_msmarco_k${K}_s${s}.json" || \
      echo "· missing results/${RESULTS_PREFIX}_msmarco_k${K}_s${s}.json"
  done
  for model in rag_sequence rag_token; do
    # Remove stale predictions (with and without sample suffix).
    modal volume rm rag-data "results/nq_${model}_k${K}_preds.txt" && \
      echo "✓ removed results/nq_${model}_k${K}_preds.txt" || \
      echo "· missing results/nq_${model}_k${K}_preds.txt"
    modal volume rm rag-data "results/nq_${model}_k${K}_s*_preds.txt" && \
      echo "✓ removed results/nq_${model}_k${K}_s*_preds.txt" || \
      echo "· missing results/nq_${model}_k${K}_s*_preds.txt"
    modal volume rm rag-data "results/msmarco_${model}_k${K}_preds.txt" && \
      echo "✓ removed results/msmarco_${model}_k${K}_preds.txt" || \
      echo "· missing results/msmarco_${model}_k${K}_preds.txt"
    modal volume rm rag-data "results/msmarco_${model}_k${K}_s*_preds.txt" && \
      echo "✓ removed results/msmarco_${model}_k${K}_s*_preds.txt" || \
      echo "· missing results/msmarco_${model}_k${K}_s*_preds.txt"
    modal volume rm rag-data "results/msmarco_${model}_k${K}_gen_metrics.json" && \
      echo "✓ removed results/msmarco_${model}_k${K}_gen_metrics.json" || \
      echo "· missing results/msmarco_${model}_k${K}_gen_metrics.json"
  done
  rm -rf "${LOCAL_RESULTS_DIR}"
fi

mkdir -p "${LOCAL_RESULTS_DIR}"
for s in ${SAMPLE_SIZES}; do
  download_result "${RESULTS_PREFIX}_nq_k${K}_s${s}.json"
  download_result "${RESULTS_PREFIX}_msmarco_k${K}_s${s}.json"
done

python scripts/plot_sample_size_sensitivity.py \
  --results-dir "${LOCAL_RESULTS_DIR}" \
  --k "${K}" \
  --sample-sizes "$(echo "${SAMPLE_SIZES}" | tr ' ' ',')" \
  --output "${LOCAL_RESULTS_DIR}/${RESULTS_PREFIX}_k${K}.png" \
  --summary-output "${LOCAL_RESULTS_DIR}/${RESULTS_PREFIX}_k${K}_summary.json"
