#!/bin/bash
#
# Run Table 3 example generations on Modal using small subsets.
#
# Usage:
#   bash scripts/run_table3_examples_modal.sh [--skip-setup] [--n-docs N] [--batch-size N] [--results-file FILE]
#
# Notes:
# - Assumes eval datasets are already uploaded to the Modal volume.
# - If not, run: bash upload_datasets_to_modal.sh

set -euo pipefail

if ! command -v modal >/dev/null 2>&1; then
  echo "❌ Modal CLI not found. Install with: pip install modal"
  exit 1
fi

ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-setup)
      ARGS+=("--skip-setup")
      shift
      ;;
    --n-docs)
      ARGS+=("--n-docs" "$2")
      shift 2
      ;;
    --batch-size)
      ARGS+=("--eval-batch-size" "$2")
      shift 2
      ;;
    --results-file)
      ARGS+=("--results-file" "$2")
      shift 2
      ;;
    --no-recalculate)
      ARGS+=("--recalculate" "false")
      shift
      ;;
    -h|--help)
      echo "Usage: bash scripts/run_table3_examples_modal.sh [--skip-setup] [--n-docs N] [--batch-size N] [--results-file FILE]"
      echo ""
      echo "Options:"
      echo "  --skip-setup        Skip wiki_dpr setup (use only if cached on Modal)."
      echo "  --n-docs N          Number of docs to retrieve per query (default: 5)."
      echo "  --batch-size N      Eval batch size (default: 4)."
      echo "  --results-file FILE Output JSON filename (default: table3_examples.json)."
      echo "  --no-recalculate    Reuse existing predictions if present."
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

echo "Running: modal run modal_rag_eval.py::table3_examples ${ARGS[*]:-}"
modal run modal_rag_eval.py::table3_examples "${ARGS[@]}"
