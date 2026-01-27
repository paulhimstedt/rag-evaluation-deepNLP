"""
Modal deployment script for RAG evaluation.
Reproduces RAG paper results on all 7 evaluation datasets.

Usage:
    modal run modal_rag_eval.py

This will:
1. Download wiki_dpr compressed index (~80-90GB, one-time)
2. Prepare all 7 evaluation datasets
3. Run 21 evaluations (7 datasets × 3 models)
4. Generate comparison table vs paper results
"""

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

import modal

# ============================================================================
# Modal Infrastructure Setup
# ============================================================================

# Create Modal app and persistent volume
app = modal.App("rag-evaluation")
volume = modal.Volume.from_name("rag-data", create_if_missing=True)

# Define container image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git")  # Needed for GitPython
    .pip_install(
        # Core dependencies with strict version control for compatibility
        "numpy>=1.19.0,<2.0.0",  # Pin to 1.x for pyarrow/datasets compatibility
        "pyarrow>=6.0.0,<15.0.0",  # Compatible with datasets 1.18.0 (needs PyExtensionType, removed in 21.0.0)
        "datasets==1.18.0",  # Pin as requested
        "transformers==4.30.2",  # Specific version known to work with datasets 1.18.0 and torch 2.0
        "torch==2.0.1",  # Pin specific version for stability
        "faiss-cpu==1.7.4",  # Pin for reproducibility
        "pandas>=1.3.0,<2.0.0",  # 1.x series for compatibility
        "tqdm>=4.60.0",
        "psutil>=5.8.0",
        "GitPython>=3.1.0",
        # Note: pytorch-lightning not needed for evaluation, only for training
    )
    # Add local Python files to the image
    .add_local_dir(
        ".",
        remote_path="/workspace",
        # Only include Python and text files, exclude common patterns
        ignore=lambda p: (
            p.name.startswith('.') or
            p.name in ['__pycache__', '.git', 'results', 'eval_datasets', 'eval_datasets_test'] or
            not (p.suffix in ['.py', '.txt'] or p.is_dir())
        )
    )
)

# ============================================================================
# Constants
# ============================================================================

# Volume paths
VOLUME_PATH = "/data"
HF_CACHE_DIR = f"{VOLUME_PATH}/huggingface_cache"
EVAL_DATASETS_DIR = f"{VOLUME_PATH}/eval_datasets"
RESULTS_DIR = f"{VOLUME_PATH}/results"
CODE_DIR = "/workspace"

# Dataset names
DATASETS = ['nq', 'triviaqa', 'webquestions', 'curatedtrec', 'msmarco', 'searchqa', 'fever_3way']

# Model configurations
MODELS = {
    'rag_sequence': 'facebook/rag-sequence-nq',
    'rag_token': 'facebook/rag-token-nq',
    'bart': 'facebook/bart-large',
}

# Paper results for comparison (Table 1 from RAG paper)
PAPER_RESULTS = {
    'nq': {'rag_sequence': 44.5, 'rag_token': 44.1},
    'triviaqa': {'rag_sequence': 56.8, 'rag_token': 55.2},
    'webquestions': {'rag_sequence': 45.2, 'rag_token': 45.5},
    'curatedtrec': {'rag_sequence': 52.2, 'rag_token': 50.0},
    # Generation tasks don't have exact EM scores in paper
    'msmarco': {},
    'searchqa': {},
    'fever_3way': {},
}

# ============================================================================
# Helper Functions
# ============================================================================


def parse_metrics_from_output(output: str) -> Dict[str, float]:
    """Parse EM and F1 metrics from eval_rag.py output."""
    metrics = {}

    # Look for patterns like "F1: 52.10" and "EM: 44.30"
    # Try multiple patterns to be more robust
    f1_patterns = [
        r'F1:\s+([\d.]+)',
        r'f1:\s+([\d.]+)',
        r'F1\s*=\s*([\d.]+)',
        r'"f1":\s*([\d.]+)',
    ]
    em_patterns = [
        r'EM:\s+([\d.]+)',
        r'em:\s+([\d.]+)',
        r'EM\s*=\s*([\d.]+)',
        r'"em":\s*([\d.]+)',
        r'exact_match:\s+([\d.]+)',
    ]

    # Try each pattern
    for pattern in f1_patterns:
        f1_match = re.search(pattern, output, re.IGNORECASE)
        if f1_match:
            metrics['f1'] = float(f1_match.group(1))
            break

    for pattern in em_patterns:
        em_match = re.search(pattern, output, re.IGNORECASE)
        if em_match:
            metrics['em'] = float(em_match.group(1))
            break

    return metrics


def format_comparison_table(results: List[Dict]) -> str:
    """Format results as a comparison table with paper results."""
    lines = []
    lines.append("=" * 100)
    lines.append("RAG Evaluation Results - Comparison with Paper")
    lines.append("=" * 100)
    lines.append(f"{'Dataset':<15} {'Model':<15} {'EM (Ours)':<12} {'EM (Paper)':<12} {'Diff':<10} {'Status':<10}")
    lines.append("-" * 100)

    for result in sorted(results, key=lambda x: (x['dataset'], x['model'])):
        dataset = result['dataset']
        model = result['model']
        our_em = result['metrics'].get('em', 0.0)
        paper_em = PAPER_RESULTS.get(dataset, {}).get(model, None)

        if paper_em is not None:
            diff = our_em - paper_em
            diff_str = f"{diff:+.1f}"
            status = "✓ Close" if abs(diff) < 3.0 else "⚠ Check"
        else:
            diff_str = "N/A"
            status = "No baseline"

        paper_em_str = f"{paper_em:.1f}" if paper_em is not None else "N/A"

        lines.append(
            f"{dataset:<15} {model:<15} {our_em:<12.2f} {paper_em_str:<12} {diff_str:<10} {status:<10}"
        )

    lines.append("=" * 100)
    return "\n".join(lines)


# ============================================================================
# Modal Functions
# ============================================================================


@app.function(
    image=image,
    volumes={VOLUME_PATH: volume},
    timeout=7200,  # 2 hours for initial index download
    cpu=8.0,
    memory=16384,  # 16GB RAM
)
def setup_wiki_index():
    """
    Download compressed wiki_dpr index to Modal volume.
    This is a one-time operation (~80-90GB).
    """
    print("=" * 80)
    print("Setting up wiki_dpr compressed index")
    print("=" * 80)

    # Set HuggingFace cache to volume
    os.environ['HF_HOME'] = HF_CACHE_DIR
    os.environ['HF_DATASETS_CACHE'] = f"{HF_CACHE_DIR}/datasets"
    os.environ['TRANSFORMERS_CACHE'] = f"{HF_CACHE_DIR}/transformers"
    
    # Set HF_TOKEN if available (for gated/private datasets)
    # Note: Set this in Modal secrets if needed for private datasets
    if 'HF_TOKEN' in os.environ:
        print("✓ HF_TOKEN found, will use for authentication")

    # Create directories
    os.makedirs(HF_CACHE_DIR, exist_ok=True)

    # Import here to avoid issues with image building
    from datasets import load_dataset

    # Check if already downloaded
    index_marker = Path(HF_CACHE_DIR) / "wiki_dpr_compressed.marker"
    if index_marker.exists():
        print("✓ wiki_dpr index already cached")
        return

    print("Downloading wiki_dpr compressed index (this will take a while)...")
    print("Expected size: ~80-90GB")

    try:
        # Download compressed index
        dataset = load_dataset(
            "facebook/wiki_dpr",
            "psgs_w100.nq.compressed",
            split="train",
            cache_dir=f"{HF_CACHE_DIR}/datasets"
        )

        print(f"✓ Index loaded successfully: {len(dataset)} passages")

        # Create marker file
        index_marker.write_text("Index downloaded successfully")

        # Commit to persist on volume
        volume.commit()

        print("✓ wiki_dpr index setup complete")

    except Exception as e:
        print(f"✗ Error downloading index: {e}")
        raise

    return {"status": "success", "num_passages": len(dataset)}


@app.function(
    image=image,
    volumes={VOLUME_PATH: volume},
    timeout=3600,
    cpu=4.0,
)
def prepare_datasets():
    """
    Prepare all 7 evaluation datasets.
    Downloads from DPR and HuggingFace, converts to eval format.
    """
    print("=" * 80)
    print("Preparing evaluation datasets")
    print("=" * 80)
    
    # Set HF_TOKEN if available
    if 'HF_TOKEN' in os.environ:
        print("✓ Using HF_TOKEN for dataset access")

    # Add code directory to path
    import sys
    sys.path.insert(0, CODE_DIR)

    # Import and run dataset preparation
    from prepare_eval_datasets import DatasetPreparer

    preparer = DatasetPreparer(output_dir=EVAL_DATASETS_DIR)
    results = preparer.prepare_all()

    # Commit to volume
    volume.commit()

    return results


@app.function(
    image=image,
    gpu="T4",  # T4 GPU for cost-effective evaluation
    volumes={VOLUME_PATH: volume},
    timeout=7200,  # 2 hours per evaluation (includes model download + index init + inference)
    cpu=4.0,
    memory=16384,
)
def run_evaluation(
    dataset_name: str,
    model_type: str,
    eval_mode: str = "e2e",
    n_docs: int = 5,
) -> Dict:
    """
    Run evaluation for one dataset-model combination.

    Args:
        dataset_name: Name of dataset (e.g., 'nq', 'triviaqa')
        model_type: Model type ('rag_sequence', 'rag_token', 'bart')
        eval_mode: Evaluation mode ('e2e' or 'retrieval')
        n_docs: Number of documents to retrieve

    Returns:
        Dict with evaluation results
    """
    print("=" * 80)
    print(f"Evaluating {model_type} on {dataset_name}")
    print("=" * 80)

    
    # Preserve HF_TOKEN if set
    # Note: Set HF_TOKEN in Modal secrets if you need access to gated datasets
    # Set HuggingFace cache to volume
    os.environ['HF_HOME'] = HF_CACHE_DIR
    os.environ['HF_DATASETS_CACHE'] = f"{HF_CACHE_DIR}/datasets"
    os.environ['TRANSFORMERS_CACHE'] = f"{HF_CACHE_DIR}/transformers"

    # Add CODE_DIR to PYTHONPATH for imports
    os.environ['PYTHONPATH'] = CODE_DIR + ':' + os.environ.get('PYTHONPATH', '')

    # Get model name
    model_name = MODELS[model_type]

    # Construct paths
    source_path = f"{EVAL_DATASETS_DIR}/{dataset_name}_test.source"
    target_path = f"{EVAL_DATASETS_DIR}/{dataset_name}_test.target"
    preds_path = f"{RESULTS_DIR}/{dataset_name}_{model_type}_preds.txt"

    # Create results directory
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Check if dataset files exist
    if not os.path.exists(source_path):
        return {
            'dataset': dataset_name,
            'model': model_type,
            'status': 'error',
            'error': f'Dataset source file not found: {source_path}'
        }

    if not os.path.exists(target_path):
        return {
            'dataset': dataset_name,
            'model': model_type,
            'status': 'error',
            'error': f'Dataset target file not found: {target_path}'
        }

    # Construct eval_rag.py command
    cmd = [
        'python', f'{CODE_DIR}/eval_rag.py',
        '--model_name_or_path', model_name,
        '--model_type', model_type,
        '--evaluation_set', source_path,
        '--gold_data_path', target_path,
        '--gold_data_mode', 'qa',  # Use qa mode for tab-separated format
        '--predictions_path', preds_path,
        '--eval_mode', eval_mode,
        '--n_docs', str(n_docs),
        '--eval_batch_size', '8',
        '--print_predictions',
    ]

    # For RAG models, add index configuration
    # Explicitly specify compressed index to avoid URL validation errors
    if model_type in ['rag_sequence', 'rag_token']:
        cmd.extend(['--index_name', 'compressed'])

    print(f"Running command: {' '.join(cmd)}")

    try:
        # Run evaluation
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=CODE_DIR,
            env=os.environ.copy()
        )

        # Parse metrics from output
        metrics = parse_metrics_from_output(result.stdout)

        # Debug: Print last part of stdout/stderr if metrics are empty
        if not metrics:
            print(f"\n⚠ Warning: No metrics found in output")
            print(f"Last 500 chars of stdout:\n{result.stdout[-500:]}")
            print(f"Last 500 chars of stderr:\n{result.stderr[-500:] if result.stderr else 'None'}")
            print(f"Return code: {result.returncode}")

        # Count samples
        num_samples = 0
        if os.path.exists(source_path):
            with open(source_path, 'r') as f:
                num_samples = sum(1 for _ in f)

        print(f"✓ Evaluation complete")
        print(f"  Samples: {num_samples}")
        print(f"  Metrics: {metrics}")

        # Commit results to volume
        volume.commit()

        return {
            'dataset': dataset_name,
            'model': model_type,
            'eval_mode': eval_mode,
            'n_docs': n_docs,
            'num_samples': num_samples,
            'metrics': metrics,
            'status': 'success',
            'stdout': result.stdout[-2000:],  # Last 2000 chars
            'stderr': result.stderr[-1000:] if result.stderr else '',
        }

    except Exception as e:
        print(f"✗ Evaluation failed: {e}")
        return {
            'dataset': dataset_name,
            'model': model_type,
            'status': 'error',
            'error': str(e),
        }


@app.function(
    image=image,
    volumes={VOLUME_PATH: volume},
    timeout=300,
)
def save_results(results: List[Dict], output_file: str = "evaluation_results.json"):
    """Save evaluation results to JSON file."""
    results_path = f"{RESULTS_DIR}/{output_file}"
    os.makedirs(RESULTS_DIR, exist_ok=True)

    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    volume.commit()

    print(f"✓ Results saved to {results_path}")
    return results_path


# ============================================================================
# Main Entrypoint
# ============================================================================


@app.local_entrypoint()
def main(
    setup_only: bool = False,
    datasets_only: bool = False,
    test_mode: bool = False,
):
    """
    Main entrypoint for RAG evaluation on Modal.

    Args:
        setup_only: Only setup wiki_dpr index and exit
        datasets_only: Only prepare datasets and exit
        test_mode: Run only a single evaluation for testing (NQ dataset × RAG-Sequence model)
                   This is useful for:
                   - Verifying the setup works before running the full evaluation
                   - Testing changes without waiting for all 21 evaluations
                   - Estimating time/cost before committing to the full run
                   Expected time: ~2 hours, Cost: ~$3-5
    """
    print("\n" + "=" * 80)
    print("RAG Paper Reproduction - Modal Evaluation")
    print("=" * 80 + "\n")

    # Copy code files to container
    # This happens automatically via Modal's volume mounting

    # Step 1: Setup wiki_dpr index (one-time operation)
    print("\n[1/4] Setting up wiki_dpr index...")
    setup_result = setup_wiki_index.remote()
    print(f"✓ Setup complete: {setup_result}")

    if setup_only:
        print("\nSetup complete! Exiting as requested.")
        return

    # Step 2: Prepare evaluation datasets
    print("\n[2/4] Preparing evaluation datasets...")
    dataset_results = prepare_datasets.remote()
    print(f"✓ Datasets prepared: {dataset_results}")

    if datasets_only:
        print("\nDataset preparation complete! Exiting as requested.")
        return

    # Step 3: Run evaluations
    print("\n[3/4] Running evaluations...")

    if test_mode:
        print("⚠ Test mode: Running single evaluation only (NQ × RAG-Sequence)")
        print("  This is a quick validation run to ensure the pipeline works.")
        print("  Expected: ~2 hours, ~$3-5")
        datasets_to_eval = ['nq']
        models_to_eval = ['rag_sequence']
    else:
        # Filter to only datasets that were successfully prepared
        datasets_to_eval = [d for d in DATASETS if dataset_results.get(d, 0) > 0]
        if not datasets_to_eval:
            print("✗ No datasets were successfully prepared!")
            return []
        print(f"Evaluating on {len(datasets_to_eval)} datasets: {', '.join(datasets_to_eval)}")
        models_to_eval = list(MODELS.keys())

    # Run all evaluations
    results = []
    total = len(datasets_to_eval) * len(models_to_eval)
    current = 0

    for dataset in datasets_to_eval:
        for model in models_to_eval:
            current += 1
            print(f"\n[{current}/{total}] Evaluating {model} on {dataset}...")

            result = run_evaluation.remote(
                dataset_name=dataset,
                model_type=model,
                eval_mode='e2e',
                n_docs=5,
            )

            results.append(result)

            # Print immediate feedback
            if result['status'] == 'success':
                metrics = result['metrics']
                print(f"  ✓ EM: {metrics.get('em', 0):.2f}, F1: {metrics.get('f1', 0):.2f}")
            else:
                print(f"  ✗ Error: {result.get('error', 'Unknown error')}")

    # Step 4: Save results and generate comparison
    print("\n[4/4] Generating results...")

    # Save raw results
    save_results.remote(results, "evaluation_results.json")

    # Generate comparison table
    comparison = format_comparison_table(results)
    print("\n" + comparison)

    # Save comparison table
    comparison_path = f"{RESULTS_DIR}/comparison_table.txt"
    with open(comparison_path.replace(RESULTS_DIR, "./results"), 'w') as f:
        f.write(comparison)

    print(f"\n✓ All evaluations complete!")
    print(f"  Results saved to: ./results/evaluation_results.json")
    print(f"  Comparison table: ./results/comparison_table.txt")

    # Calculate success rate
    successful = sum(1 for r in results if r['status'] == 'success')
    print(f"\n  Success rate: {successful}/{len(results)} ({100*successful/len(results):.1f}%)")

    return results


# ============================================================================
# Additional Utility Functions
# ============================================================================


@app.function(
    image=image,
    volumes={VOLUME_PATH: volume},
)
def list_datasets():
    """List all prepared datasets in the volume."""
    import glob

    dataset_files = glob.glob(f"{EVAL_DATASETS_DIR}/*.source")
    datasets = [Path(f).stem.replace('_test', '') for f in dataset_files]

    print("Available datasets:")
    for ds in sorted(datasets):
        source = f"{EVAL_DATASETS_DIR}/{ds}_test.source"
        if os.path.exists(source):
            with open(source) as f:
                count = sum(1 for _ in f)
            print(f"  - {ds}: {count} samples")

    return datasets


@app.function(
    image=image,
    volumes={VOLUME_PATH: volume},
)
def cleanup_results():
    """Clean up results directory."""
    import shutil

    if os.path.exists(RESULTS_DIR):
        shutil.rmtree(RESULTS_DIR)
        os.makedirs(RESULTS_DIR, exist_ok=True)
        volume.commit()
        print("✓ Results directory cleaned")


# ============================================================================
# Usage Examples
# ============================================================================

"""
# Run full evaluation pipeline:
modal run modal_rag_eval.py

# Only setup wiki_dpr index:
modal run modal_rag_eval.py --setup-only

# Only prepare datasets:
modal run modal_rag_eval.py --datasets-only

# Test mode (single evaluation):
modal run modal_rag_eval.py --test-mode

# List available datasets:
modal run modal_rag_eval.py::list_datasets

# Clean up results:
modal run modal_rag_eval.py::cleanup_results
"""
