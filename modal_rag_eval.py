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
from typing import Dict, List

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
        "numpy>=1.19.0,<1.24.0",  # Avoid np.object removal in numpy>=1.24 (datasets 1.18.0 uses it)
        "pyarrow>=6.0.0,<15.0.0",  # Compatible with datasets 1.18.0 (needs PyExtensionType, removed in 21.0.0)
        "datasets==1.18.0",  # Pin as requested
        "transformers==4.30.2",  # Specific version known to work with datasets 1.18.0 and torch 2.0
        "torch==2.0.1",  # Pin specific version for stability
        "faiss-cpu==1.7.4",  # Pin for reproducibility
        "hf_transfer>=0.1.6",  # Optional: faster HF Hub downloads when enabled
        "pandas>=1.3.0,<2.0.0",  # 1.x series for compatibility
        "tqdm>=4.60.0",
        "psutil>=5.8.0",
        "GitPython>=3.1.0",
        "kaggle>=1.5.12",  # For automatic MS-MARCO download
        "sacrebleu>=2.3.0,<3.0.0",  # BLEU-1 for MS-MARCO plotting
        "rouge-score>=0.1.2,<0.2.0",  # ROUGE-L for MS-MARCO plotting
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
# Optional datasets (prepared but not included in default full runs)
OPTIONAL_DATASETS = ['nq_retrieval']

# Model configurations
MODELS = {
    'rag_sequence': 'facebook/rag-sequence-nq',
    'rag_token': 'facebook/rag-token-nq',
    'bart': 'facebook/bart-large',
}

TABLE3_EXAMPLES = {
    "msmarco": [
        "define middle ear",
        "what currency needed in scotland",
    ],
    "jeopardy_qg": [
        "Washington",
        "The Divine Comedy",
    ],
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
    precision_patterns = [
        r'Precision@(\d+):\s*([\d.]+)',
        r'precision@(\d+):\s*([\d.]+)',
    ]
    recall_patterns = [
        r'Recall@(\d+):\s*([\d.]+)',
        r'recall@(\d+):\s*([\d.]+)',
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

    for pattern in precision_patterns:
        precision_match = re.search(pattern, output, re.IGNORECASE)
        if precision_match:
            metrics['precision_at_k'] = float(precision_match.group(2))
            metrics['k'] = int(precision_match.group(1))
            break

    for pattern in recall_patterns:
        recall_match = re.search(pattern, output, re.IGNORECASE)
        if recall_match:
            metrics['recall_at_k'] = float(recall_match.group(2))
            if 'k' not in metrics:
                metrics['k'] = int(recall_match.group(1))
            break

    return metrics


def parse_csv_arg(value: str) -> List[str]:
    if not value:
        return []
    return [v.strip() for v in value.split(',') if v.strip()]


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
    timeout=43200,  # 12 hours for initial index download
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
    os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = "1"
    os.environ['HF_XET_HIGH_PERFORMANCE'] = "1"

    try:
        import huggingface_hub
        print(f"huggingface_hub version: {huggingface_hub.__version__}")
    except Exception as e:
        print(f"⚠ Unable to read huggingface_hub version: {e}")
    
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
    timeout=43200,  # 12 hours for passages download
    cpu=8.0,
    memory=16384,
)
def setup_wiki_passages():
    """
    Download wiki_dpr passages (psgs_w100.nq.no_index) to Modal volume.
    This is a one-time operation (~139GB total download+generated size).
    """
    print("=" * 80)
    print("Setting up wiki_dpr passages (psgs_w100.nq.compressed)")
    print("=" * 80)

    # Set HuggingFace cache to volume
    os.environ['HF_HOME'] = HF_CACHE_DIR
    os.environ['HF_DATASETS_CACHE'] = f"{HF_CACHE_DIR}/datasets"
    os.environ['TRANSFORMERS_CACHE'] = f"{HF_CACHE_DIR}/transformers"
    os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = "1"
    os.environ['HF_XET_HIGH_PERFORMANCE'] = "1"

    try:
        import huggingface_hub
        print(f"huggingface_hub version: {huggingface_hub.__version__}")
    except Exception as e:
        print(f"⚠ Unable to read huggingface_hub version: {e}")

    # Create directories
    os.makedirs(HF_CACHE_DIR, exist_ok=True)

    # Check marker + expected cache path (fast path: facebook___wiki_dpr)
    passages_marker = Path(HF_CACHE_DIR) / "wiki_dpr_passages.marker"
    expected_cache = (
        Path(HF_CACHE_DIR)
        / "datasets"
        / "facebook___wiki_dpr"
        / "psgs_w100.nq.compressed"
    )
    if passages_marker.exists() and expected_cache.exists():
        print("✓ wiki_dpr passages already cached")
        return {"status": "cached"}
    if passages_marker.exists() and not expected_cache.exists():
        print("⚠ Passages marker exists but expected cache not found. Proceeding to download.")

    print("Downloading wiki_dpr passages (this will take a while)...")
    print("Expected size: ~139GB (download+generated)")

    try:
        # Download passages dataset
        from datasets import load_dataset

        dataset = load_dataset(
            "facebook/wiki_dpr",
            "psgs_w100.nq.compressed",
            split="train",
            cache_dir=f"{HF_CACHE_DIR}/datasets",
        )

        print(f"✓ Passages loaded successfully: {len(dataset)} passages")

        passages_marker.write_text("Passages downloaded successfully")
        volume.commit()

        print("✓ wiki_dpr passages setup complete")
        return {"status": "success", "num_passages": len(dataset)}

    except Exception as e:
        print(f"✗ Error downloading passages: {e}")
        raise


@app.function(
    image=image,
    volumes={VOLUME_PATH: volume},
    timeout=3600,
    cpu=4.0,
)
def prepare_datasets(max_samples: int = 0):
    """
    Prepare all 7 evaluation datasets.
    Downloads from DPR and HuggingFace, converts to eval format.
    """
    print("=" * 80)
    print("Preparing evaluation datasets")
    print("=" * 80)
    
    # Check if datasets are already uploaded
    existing_datasets = []
    if os.path.exists(EVAL_DATASETS_DIR):
        source_files = [f for f in os.listdir(EVAL_DATASETS_DIR) if f.endswith('_test.source')]
        if source_files:
            existing_datasets = [f.replace('_test.source', '') for f in source_files]
            print(f"\n✓ Found {len(source_files)} pre-uploaded test dataset files:")
            for f in sorted(source_files):
                size = os.path.getsize(os.path.join(EVAL_DATASETS_DIR, f)) / 1024
                print(f"  • {f} ({size:.1f} KB)")
            print("\n⚠  Skipping dataset preparation - using pre-uploaded files")
            print("   (This is the recommended workflow for reproducibility)")
            print("\nTo re-download datasets, either:")
            print("  1. Remove files: modal volume rm rag-data eval_datasets/*.source")
            print("  2. Or prepare locally and re-upload (see docs/DATASET_PREPARATION_GUIDE.md)")
            print()
            
            # Create results with sample counts (only if target exists)
            results = {}
            for f in source_files:
                dataset_name = f.replace('_test.source', '')
                source_path = os.path.join(EVAL_DATASETS_DIR, f)
                target_path = os.path.join(EVAL_DATASETS_DIR, f"{dataset_name}_test.target")
                if not os.path.exists(target_path):
                    print(f"  ⚠ Missing target file for {dataset_name}: {target_path}")
                    results[dataset_name] = 0
                    continue
                with open(source_path, 'r') as fh:
                    results[dataset_name] = sum(1 for _ in fh)

            # Generate optional datasets if missing (e.g., nq_retrieval)
            missing_optional = []
            for ds in OPTIONAL_DATASETS:
                src = os.path.join(EVAL_DATASETS_DIR, f"{ds}_test.source")
                tgt = os.path.join(EVAL_DATASETS_DIR, f"{ds}_test.target")
                if not (os.path.exists(src) and os.path.exists(tgt)):
                    missing_optional.append(ds)

            if missing_optional:
                print(f"\n📥 Preparing missing optional datasets: {', '.join(missing_optional)}")
                import sys
                if CODE_DIR not in sys.path:
                    sys.path.insert(0, CODE_DIR)
                from prepare_eval_datasets import DatasetPreparer
                preparer = DatasetPreparer(output_dir=EVAL_DATASETS_DIR, max_samples=max_samples)
                if "nq_retrieval" in missing_optional:
                    try:
                        results["nq_retrieval"] = preparer.prepare_nq_retrieval()
                    except Exception as e:
                        print(f"✗ NQ retrieval preparation failed: {e}")
                        results["nq_retrieval"] = 0

            results_file = f"{RESULTS_DIR}/dataset_preparation_results.json"
            os.makedirs(RESULTS_DIR, exist_ok=True)
            with open(results_file, 'w') as f:
                json.dump({
                    "status": "pre-uploaded",
                    "datasets": results,
                    "note": "Datasets were uploaded via upload_datasets_to_modal.sh"
                }, f, indent=2)

            volume.commit()
            return results
    
    # No pre-uploaded datasets found - proceed with preparation
    print("\n📥 No pre-uploaded datasets found, attempting in-container preparation...")
    print("   Note: Some datasets may fail due to environment constraints")
    print("   Recommended: Use prepare_datasets_local.py + upload_datasets_to_modal.sh\n")
    
    # Set HF_TOKEN if available
    if 'HF_TOKEN' in os.environ:
        print("✓ Using HF_TOKEN for dataset access")

    # Add code directory to path
    import sys
    sys.path.insert(0, CODE_DIR)

    # Capture detailed logs to file
    log_file = f"{RESULTS_DIR}/dataset_preparation.log"
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Import and run dataset preparation with output capture
    from prepare_eval_datasets import DatasetPreparer
    import io
    import contextlib

    # Capture both stdout and stderr
    log_capture = io.StringIO()
    with contextlib.redirect_stdout(log_capture), contextlib.redirect_stderr(log_capture):
        preparer = DatasetPreparer(output_dir=EVAL_DATASETS_DIR, max_samples=max_samples)
        results = preparer.prepare_all()
    
    # Write logs to file
    log_content = log_capture.getvalue()
    with open(log_file, 'w') as f:
        f.write(log_content)
    
    # Also print to console
    print(log_content)
    
    # Save results summary
    results_file = f"{RESULTS_DIR}/dataset_preparation_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    # Commit to volume
    volume.commit()

    return results


@app.function(
    image=image,
    gpu="T4",  # T4 GPU for cost-effective evaluation
    volumes={VOLUME_PATH: volume},
    timeout=43200,  # 12 hours per evaluation (includes model download + index init + inference)
    cpu=4.0,
    memory=16384,
)
def run_evaluation(
    dataset_name: str,
    model_type: str,
    eval_mode: str = "e2e",
    n_docs: int = 5,
    eval_batch_size: int = 8,
    max_eval_samples: int = 0,
    num_beams: int = 0,
) -> Dict:
    """
    Run evaluation for one dataset-model combination.

    Args:
        dataset_name: Name of dataset (e.g., 'nq', 'triviaqa')
        model_type: Model type ('rag_sequence', 'rag_token', 'bart')
        eval_mode: Evaluation mode ('e2e' or 'retrieval')
        n_docs: Number of documents to retrieve
        num_beams: Beam size override (0 means default from eval_rag.py)

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
    os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = "1"
    os.environ['HF_XET_HIGH_PERFORMANCE'] = "1"

    # Add CODE_DIR to PYTHONPATH for imports
    os.environ['PYTHONPATH'] = CODE_DIR + ':' + os.environ.get('PYTHONPATH', '')

    # Get model name
    model_name = MODELS[model_type]

    # Construct paths
    source_path = f"{EVAL_DATASETS_DIR}/{dataset_name}_test.source"
    target_path = f"{EVAL_DATASETS_DIR}/{dataset_name}_test.target"
    preds_path = f"{RESULTS_DIR}/{dataset_name}_{model_type}_k{n_docs}_preds.txt"

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
    recalculate = False
    if os.path.exists(preds_path) and os.path.getsize(preds_path) == 0:
        recalculate = True
        print(f"⚠ Found empty predictions file, forcing recompute: {preds_path}")

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
        '--eval_batch_size', str(eval_batch_size),
        '--print_predictions',
    ]
    if num_beams and num_beams > 0:
        cmd.extend(['--num_beams', str(num_beams)])
    if eval_mode == "retrieval":
        cmd.extend(['--k', str(n_docs)])
    if max_eval_samples and max_eval_samples > 0:
        cmd.extend(['--max_eval_samples', str(max_eval_samples)])
    if recalculate:
        cmd.append('--recalculate')

    # For RAG models, add index configuration
    # Explicitly specify compressed index to avoid URL validation errors
    if model_type in ['rag_sequence', 'rag_token']:
        cmd.extend(['--index_name', 'compressed'])
        cmd.extend([
            '--passages_dataset', 'facebook/wiki_dpr',
            '--passages_config', 'psgs_w100.nq.compressed',
        ])

    print(f"Running command: {' '.join(cmd)}")

    try:
        # Run evaluation with streaming output
        # This allows us to see progress as the model downloads and runs
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Merge stderr into stdout
            text=True,
            cwd=CODE_DIR,
            env=os.environ.copy(),
            bufsize=1,  # Line buffered
            universal_newlines=True
        )

        # Stream output in real-time and collect for parsing
        stdout_lines = []
        print("\n" + "=" * 80)
        print("Evaluation Output:")
        print("=" * 80)
        for line in process.stdout:
            print(line, end='')  # Print without extra newline
            stdout_lines.append(line)
        
        # Wait for process to complete
        return_code = process.wait()
        stdout_text = ''.join(stdout_lines)
        print("=" * 80 + "\n")

        # Parse metrics from output
        metrics = parse_metrics_from_output(stdout_text)

        # Optional generation metrics for MS-MARCO (BLEU-1, ROUGE-L)
        if dataset_name == "msmarco" and eval_mode == "e2e":
            gen_metrics_path = f"{RESULTS_DIR}/{dataset_name}_{model_type}_k{n_docs}_gen_metrics.json"
            gen_cmd = [
                'python', f'{CODE_DIR}/scripts/eval_gen_metrics.py',
                '--predictions_path', preds_path,
                '--gold_data_path', target_path,
                '--gold_data_mode', 'qa',
                '--output_json', gen_metrics_path,
            ]
            try:
                subprocess.run(gen_cmd, check=True, cwd=CODE_DIR, env=os.environ.copy())
                if os.path.exists(gen_metrics_path):
                    with open(gen_metrics_path, 'r') as f:
                        gen_metrics = json.load(f)
                    metrics.update(gen_metrics)
            except Exception as e:
                print(f"⚠ Generation metrics failed: {e}")

        # Debug: Print warning if metrics are empty
        if not metrics:
            print(f"\n⚠ Warning: No metrics found in output")
            print(f"Last 500 chars of output:\n{stdout_text[-500:] if stdout_text else 'None'}")
            print(f"Return code: {return_code}")

        # Count samples
        num_samples = 0
        if os.path.exists(source_path):
            with open(source_path, 'r') as f:
                num_samples = sum(1 for _ in f)

        effective_samples = num_samples
        if max_eval_samples and max_eval_samples > 0:
            effective_samples = min(num_samples, max_eval_samples)

        print(f"✓ Evaluation complete")
        print(f"  Samples: {effective_samples}")
        print(f"  Metrics: {metrics}")

        # Commit results to volume
        volume.commit()

        return {
            'dataset': dataset_name,
            'model': model_type,
            'eval_mode': eval_mode,
            'n_docs': n_docs,
            'num_samples': effective_samples,
            'max_eval_samples': max_eval_samples,
            'metrics': metrics,
            'status': 'success',
            'stdout': stdout_text[-2000:],  # Last 2000 chars
            'return_code': return_code,
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
    timeout=600,
    cpu=1.0,
    memory=4096,
)
def prepare_table3_subsets():
    """Create minimal Table 3 datasets on the Modal volume."""
    def build_subset(dataset_name: str, questions: List[str]) -> Dict:
        source_path = Path(EVAL_DATASETS_DIR) / f"{dataset_name}_test.source"
        target_path = Path(EVAL_DATASETS_DIR) / f"{dataset_name}_test.target"
        if not source_path.exists() or not target_path.exists():
            return {
                "status": "error",
                "error": f"Missing dataset files for {dataset_name}",
                "source_path": str(source_path),
                "target_path": str(target_path),
            }

        question_set = set(questions)
        indices = {}
        duplicates = {}
        with open(source_path, "r") as f_src:
            for idx, line in enumerate(f_src):
                text = line.strip()
                if text in question_set:
                    duplicates[text] = duplicates.get(text, 0) + 1
                    if text not in indices:
                        indices[text] = idx

        missing = [q for q in questions if q not in indices]
        if missing:
            return {
                "status": "error",
                "error": f"Missing questions in {dataset_name}: {missing}",
                "source_path": str(source_path),
                "target_path": str(target_path),
            }

        wanted_indices = set(indices.values())
        target_lines = {}
        with open(target_path, "r") as f_tgt:
            for idx, line in enumerate(f_tgt):
                if idx in wanted_indices:
                    target_lines[idx] = line.strip()
                    if len(target_lines) == len(wanted_indices):
                        break

        subset_name = f"{dataset_name}_table3"
        subset_source = Path(EVAL_DATASETS_DIR) / f"{subset_name}_test.source"
        subset_target = Path(EVAL_DATASETS_DIR) / f"{subset_name}_test.target"

        with open(subset_source, "w") as f_src, open(subset_target, "w") as f_tgt:
            for q in questions:
                idx = indices[q]
                f_src.write(q + "\n")
                target_line = target_lines.get(idx)
                if not target_line:
                    target_line = f"{q}\t['']"
                f_tgt.write(target_line + "\n")

        dupes = {k: v for k, v in duplicates.items() if v > 1}
        return {
            "status": "ok",
            "dataset_name": subset_name,
            "source_path": str(subset_source),
            "target_path": str(subset_target),
            "gold_data_mode": "qa",
            "inputs": questions,
            "duplicates": dupes,
        }

    results = {}
    errors = []

    msmarco_result = build_subset("msmarco", TABLE3_EXAMPLES["msmarco"])
    results["msmarco"] = msmarco_result
    if msmarco_result.get("status") != "ok":
        errors.append(msmarco_result.get("error", "Unknown MSMARCO error"))

    # Jeopardy question generation examples (input is the answer)
    jeopardy_inputs = TABLE3_EXAMPLES["jeopardy_qg"]
    jeopardy_name = "jeopardy_qg_table3"
    jeopardy_source = Path(EVAL_DATASETS_DIR) / f"{jeopardy_name}_test.source"
    jeopardy_target = Path(EVAL_DATASETS_DIR) / f"{jeopardy_name}_test.target"

    with open(jeopardy_source, "w") as f_src, open(jeopardy_target, "w") as f_tgt:
        for text in jeopardy_inputs:
            f_src.write(text + "\n")
            f_tgt.write("N/A\n")

    results["jeopardy_qg"] = {
        "status": "ok",
        "dataset_name": jeopardy_name,
        "source_path": str(jeopardy_source),
        "target_path": str(jeopardy_target),
        "gold_data_mode": "ans",
        "inputs": jeopardy_inputs,
        "note": "Placeholder references (metrics are not meaningful).",
    }

    meta_path = Path(RESULTS_DIR) / "table3_subset_meta.json"
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(meta_path, "w") as f_meta:
        json.dump({"datasets": results, "errors": errors}, f_meta, indent=2)

    volume.commit()

    status = "ok" if not errors else "error"
    return {"status": status, "errors": errors, "datasets": results, "meta_path": str(meta_path)}


@app.function(
    image=image,
    gpu="T4",
    volumes={VOLUME_PATH: volume},
    timeout=43200,
    cpu=4.0,
    memory=16384,
)
def run_table3_eval(
    dataset_key: str,
    dataset_name: str,
    source_path: str,
    target_path: str,
    gold_data_mode: str,
    model_type: str,
    n_docs: int = 5,
    eval_batch_size: int = 4,
    recalculate: bool = True,
) -> Dict:
    """Run evaluation on Table 3 subset and return predictions."""
    print("=" * 80)
    print(f"Table 3 eval: {model_type} on {dataset_key}")
    print("=" * 80)

    # Set HuggingFace cache to volume
    os.environ['HF_HOME'] = HF_CACHE_DIR
    os.environ['HF_DATASETS_CACHE'] = f"{HF_CACHE_DIR}/datasets"
    os.environ['TRANSFORMERS_CACHE'] = f"{HF_CACHE_DIR}/transformers"
    os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = "1"
    os.environ['HF_XET_HIGH_PERFORMANCE'] = "1"

    # Add CODE_DIR to PYTHONPATH for imports
    os.environ['PYTHONPATH'] = CODE_DIR + ':' + os.environ.get('PYTHONPATH', '')

    if not os.path.exists(source_path):
        return {
            "dataset": dataset_key,
            "model": model_type,
            "status": "error",
            "error": f"Source file not found: {source_path}",
        }
    if not os.path.exists(target_path):
        return {
            "dataset": dataset_key,
            "model": model_type,
            "status": "error",
            "error": f"Target file not found: {target_path}",
        }

    model_name = MODELS[model_type]
    preds_path = f"{RESULTS_DIR}/{dataset_name}_{model_type}_k{n_docs}_preds.txt"
    os.makedirs(RESULTS_DIR, exist_ok=True)

    cmd = [
        'python', f'{CODE_DIR}/eval_rag.py',
        '--model_name_or_path', model_name,
        '--model_type', model_type,
        '--evaluation_set', source_path,
        '--gold_data_path', target_path,
        '--gold_data_mode', gold_data_mode,
        '--predictions_path', preds_path,
        '--eval_mode', 'e2e',
        '--n_docs', str(n_docs),
        '--eval_batch_size', str(eval_batch_size),
        '--print_predictions',
    ]
    if recalculate:
        cmd.append('--recalculate')
    if model_type in ['rag_sequence', 'rag_token']:
        cmd.extend(['--index_name', 'compressed'])
        cmd.extend([
            '--passages_dataset', 'facebook/wiki_dpr',
            '--passages_config', 'psgs_w100.nq.compressed',
        ])

    print(f"Running command: {' '.join(cmd)}")

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=CODE_DIR,
            env=os.environ.copy(),
            bufsize=1,
            universal_newlines=True,
        )

        stdout_lines = []
        print("\n" + "=" * 80)
        print("Evaluation Output:")
        print("=" * 80)
        for line in process.stdout:
            print(line, end="")
            stdout_lines.append(line)

        return_code = process.wait()
        stdout_text = ''.join(stdout_lines)
        print("=" * 80 + "\n")

        metrics = parse_metrics_from_output(stdout_text)

        inputs = []
        predictions = []
        if os.path.exists(source_path):
            with open(source_path, "r") as f_src:
                inputs = [line.rstrip("\n") for line in f_src]
        if os.path.exists(preds_path):
            with open(preds_path, "r") as f_pred:
                predictions = [line.rstrip("\n") for line in f_pred]

        volume.commit()

        return {
            "dataset": dataset_key,
            "dataset_name": dataset_name,
            "model": model_type,
            "gold_data_mode": gold_data_mode,
            "n_docs": n_docs,
            "eval_batch_size": eval_batch_size,
            "metrics": metrics,
            "inputs": inputs,
            "predictions": predictions,
            "status": "success",
            "stdout": stdout_text[-2000:],
            "return_code": return_code,
        }
    except Exception as e:
        print(f"✗ Table 3 eval failed: {e}")
        return {
            "dataset": dataset_key,
            "dataset_name": dataset_name,
            "model": model_type,
            "status": "error",
            "error": str(e),
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
    max_samples: int = 0,
    max_eval_samples: int = 0,
    datasets: str = "",
    models: str = "",
    n_docs: int = 5,
    eval_batch_size: int = 8,
    eval_mode: str = "e2e",
    results_file: str = "evaluation_results.json",
    num_beams: int = 0,
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
        max_samples: Max samples per dataset (0 means no limit)
        max_eval_samples: Limit number of evaluation samples per dataset (0 means no limit)
        datasets: Comma-separated list of datasets to evaluate (e.g., "nq,triviaqa")
        models: Comma-separated list of models to evaluate (rag_sequence, rag_token, bart)
        n_docs: Number of documents to retrieve per query
        eval_batch_size: Batch size for evaluation
        eval_mode: Evaluation mode ("e2e" or "retrieval")
        results_file: Output filename for evaluation results JSON
        num_beams: Beam size override (0 means default from eval_rag.py)
    """
    print("\n" + "=" * 80)
    print("RAG Paper Reproduction - Modal Evaluation")
    print("=" * 80 + "\n")

    # Copy code files to container
    # This happens automatically via Modal's volume mounting

    # Step 1: Setup wiki_dpr index (one-time operation)
    print("\n[1/5] Setting up wiki_dpr index...")
    setup_result = setup_wiki_index.remote()
    print(f"✓ Setup complete: {setup_result}")

    if setup_only:
        print("\nSetup complete! Exiting as requested.")
        return

    # Step 2: Prepare evaluation datasets
    print("\n[2/5] Preparing evaluation datasets...")
    dataset_results = prepare_datasets.remote(max_samples=max_samples)
    print(f"✓ Datasets prepared: {dataset_results}")

    if datasets_only:
        print("\nDataset preparation complete! Exiting as requested.")
        return

    # Step 3: Setup wiki_dpr passages (one-time operation)
    print("\n[3/5] Setting up wiki_dpr passages...")
    passages_result = setup_wiki_passages.remote()
    print(f"✓ Passages setup complete: {passages_result}")

    # Step 4: Run evaluations
    print("\n[4/5] Running evaluations...")

    if eval_mode not in ["e2e", "retrieval"]:
        print(f"✗ Invalid eval_mode: {eval_mode} (expected 'e2e' or 'retrieval')")
        return []

    if test_mode:
        print("⚠ Test mode: Running single evaluation only (NQ × RAG-Sequence)")
        print("  This is a quick validation run to ensure the pipeline works.")
        print("  Expected: ~2 hours, ~$3-5")
        if datasets or models:
            print("  Note: --datasets/--models ignored in test mode")
        datasets_to_eval = ['nq']
        models_to_eval = ['rag_sequence']
    else:
        available_datasets = list_datasets.remote()
        if not available_datasets:
            print("✗ No datasets found on Modal volume!")
            return []

        base_datasets = [d for d in DATASETS if d in available_datasets]
        optional_datasets = [d for d in OPTIONAL_DATASETS if d in available_datasets]
        if datasets:
            requested = parse_csv_arg(datasets)
            invalid = [d for d in requested if d not in base_datasets and d not in optional_datasets]
            if invalid:
                print(f"⚠ Ignoring unknown/unavailable datasets: {', '.join(invalid)}")
            datasets_to_eval = [d for d in requested if d in base_datasets or d in optional_datasets]
        else:
            datasets_to_eval = base_datasets

        if not datasets_to_eval:
            print("✗ No valid datasets selected!")
            return []

        available_models = list(MODELS.keys())
        if models:
            requested_models = parse_csv_arg(models)
            invalid_models = [m for m in requested_models if m not in available_models]
            if invalid_models:
                print(f"⚠ Ignoring unknown models: {', '.join(invalid_models)}")
            models_to_eval = [m for m in requested_models if m in available_models]
        else:
            models_to_eval = available_models

        if not models_to_eval:
            print("✗ No valid models selected!")
            return []

        print(f"Evaluating on {len(datasets_to_eval)} datasets: {', '.join(datasets_to_eval)}")
        print(f"Models: {', '.join(models_to_eval)}")
        print(f"n_docs={n_docs}, eval_batch_size={eval_batch_size}, eval_mode={eval_mode}, num_beams={num_beams or 'default'}")

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
                eval_mode=eval_mode,
                n_docs=n_docs,
                eval_batch_size=eval_batch_size,
                max_eval_samples=max_eval_samples,
                num_beams=num_beams,
            )

            results.append(result)

            # Print immediate feedback
            if result['status'] == 'success':
                metrics = result['metrics']
                if result.get('eval_mode') == 'retrieval':
                    print(
                        f"  ✓ Precision@K: {metrics.get('precision_at_k', 0):.2f}, "
                        f"Recall@K: {metrics.get('recall_at_k', 0):.2f}"
                    )
                else:
                    print(f"  ✓ EM: {metrics.get('em', 0):.2f}, F1: {metrics.get('f1', 0):.2f}")
            else:
                print(f"  ✗ Error: {result.get('error', 'Unknown error')}")

    # Step 5: Save results and generate comparison
    print("\n[5/5] Generating results...")

    # Save raw results
    save_results.remote(results, results_file)

    # Generate comparison table
    comparison = format_comparison_table(results)
    print("\n" + comparison)

    # Save comparison table (unique per results file)
    comparison_name = Path(results_file).name.replace(".json", "")
    local_results_dir = Path("./results")
    local_results_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = local_results_dir / f"comparison_table_{comparison_name}.txt"
    with open(comparison_path, 'w') as f:
        f.write(comparison)

    print(f"\n✓ All evaluations complete!")
    print(f"  Results saved to: ./results/{results_file}")
    print(f"  Comparison table: {comparison_path}")

    # Calculate success rate
    successful = sum(1 for r in results if r['status'] == 'success')
    print(f"\n  Success rate: {successful}/{len(results)} ({100*successful/len(results):.1f}%)")

    return results


# ============================================================================
# Table 3 Reproduction Entry Point
# ============================================================================


@app.local_entrypoint()
def table3_examples(
    skip_setup: bool = False,
    n_docs: int = 5,
    eval_batch_size: int = 4,
    recalculate: bool = True,
    results_file: str = "table3_examples.json",
):
    """
    Reproduce Table 3 example generations on Modal.

    Args:
        skip_setup: Skip wiki_dpr setup (use only if cache is already available).
        n_docs: Number of documents to retrieve per query.
        eval_batch_size: Batch size for evaluation.
        recalculate: Force regeneration even if predictions exist.
        results_file: Output filename for Table 3 results JSON.
    """
    print("\n" + "=" * 80)
    print("RAG Paper Reproduction - Table 3 Examples")
    print("=" * 80 + "\n")

    if not skip_setup:
        print("[1/3] Setting up wiki_dpr index...")
        setup_wiki_index.remote()
        print("[2/3] Setting up wiki_dpr passages...")
        setup_wiki_passages.remote()
    else:
        print("Skipping wiki_dpr setup as requested.")

    print("[3/3] Preparing Table 3 subset datasets...")
    subset_info = prepare_table3_subsets.remote()
    if subset_info.get("status") != "ok":
        print("✗ Failed to prepare Table 3 subsets:")
        for err in subset_info.get("errors", []):
            print(f"  - {err}")
        return []

    datasets = subset_info.get("datasets", {})
    models = ["bart", "rag_token", "rag_sequence"]
    model_labels = {"bart": "BART", "rag_token": "RAG-T", "rag_sequence": "RAG-S"}

    results = []
    for dataset_key in ["msmarco", "jeopardy_qg"]:
        ds = datasets.get(dataset_key)
        if not ds or ds.get("status") != "ok":
            print(f"⚠ Skipping {dataset_key}: dataset not ready")
            continue

        for model in models:
            result = run_table3_eval.remote(
                dataset_key=dataset_key,
                dataset_name=ds["dataset_name"],
                source_path=ds["source_path"],
                target_path=ds["target_path"],
                gold_data_mode=ds["gold_data_mode"],
                model_type=model,
                n_docs=n_docs,
                eval_batch_size=eval_batch_size,
                recalculate=recalculate,
            )
            results.append(result)

    # Print compact summary table
    print("\n" + "=" * 80)
    print("Table 3 Example Generations")
    print("=" * 80)

    pred_map = {(r.get("dataset"), r.get("model")): r.get("predictions", []) for r in results}
    for dataset_key in ["msmarco", "jeopardy_qg"]:
        ds = datasets.get(dataset_key, {})
        inputs = ds.get("inputs", [])
        if not inputs:
            continue
        print(f"\nTask: {dataset_key}")
        if dataset_key == "jeopardy_qg":
            print("Note: Metrics are not meaningful for question generation examples.")
        for idx, text in enumerate(inputs):
            print(f"Input: {text}")
            for model in models:
                preds = pred_map.get((dataset_key, model), [])
                generation = preds[idx] if idx < len(preds) else "<missing>"
                print(f"  {model_labels[model]}: {generation}")

    save_results.remote(results, results_file)
    print(f"\n✓ Table 3 results saved to: {RESULTS_DIR}/{results_file}")
    print(f"  Subset metadata: {subset_info.get('meta_path')}")

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

    dataset_files = glob.glob(f"{EVAL_DATASETS_DIR}/*_test.source")
    datasets = [Path(f).stem.replace('_test', '') for f in dataset_files]

    print("Available datasets:")
    for ds in sorted(datasets):
        source = f"{EVAL_DATASETS_DIR}/{ds}_test.source"
        if os.path.exists(source):
            with open(source) as f:
                num_samples = sum(1 for _ in f)
            print(f"  {ds}: {num_samples} samples")

    return datasets


@app.function(
    image=image,
    volumes={VOLUME_PATH: volume},
)
def get_dataset_logs():
    """Retrieve dataset preparation logs from the volume."""
    log_file = f"{RESULTS_DIR}/dataset_preparation.log"
    results_file = f"{RESULTS_DIR}/dataset_preparation_results.json"
    
    logs = {}
    
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            logs['full_log'] = f.read()
    else:
        logs['full_log'] = "Log file not found"
    
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            logs['results'] = json.load(f)
    else:
        logs['results'] = {}
    
    return logs


@app.function(
    image=image,
    volumes={VOLUME_PATH: volume},
)
def aggregate_results(results_prefix: str = "evaluation_results"):
    """Aggregate multiple results JSONs on the volume and generate a combined comparison table."""
    import glob

    pattern = f"{RESULTS_DIR}/{results_prefix}*.json"
    result_files = sorted(glob.glob(pattern))
    if not result_files:
        msg = f"No results files found for pattern: {pattern}"
        print(msg)
        return {"status": "empty", "message": msg, "files": []}

    results = []
    for path in result_files:
        try:
            with open(path, "r") as f:
                results.extend(json.load(f))
        except Exception as e:
            print(f"⚠ Skipping {path}: {e}")

    comparison = format_comparison_table(results)
    output_path = f"{RESULTS_DIR}/comparison_table_aggregate.txt"
    with open(output_path, "w") as f:
        f.write(comparison)

    volume.commit()

    print(f"✓ Aggregated {len(result_files)} files")
    print(f"✓ Comparison table saved to {output_path}")

    return {"status": "ok", "files": result_files, "output": output_path}


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


@app.function(
    image=image,
    volumes={VOLUME_PATH: volume},
)
def cache_status():
    """Report cache status and sizes for wiki_dpr datasets on the volume."""
    import glob
    import subprocess

    def du(path: str):
        if not os.path.exists(path):
            return None
        try:
            out = subprocess.check_output(["du", "-sh", path], text=True).strip()
            return out.split()[0] if out else None
        except Exception:
            return None

    marker = f"{HF_CACHE_DIR}/wiki_dpr_compressed.marker"
    passages_marker = f"{HF_CACHE_DIR}/wiki_dpr_passages.marker"
    dataset_roots = [
        f"{HF_CACHE_DIR}/datasets/wiki_dpr",
        f"{HF_CACHE_DIR}/datasets/facebook___wiki_dpr",
    ]
    no_index = []
    compressed = []
    for root in dataset_roots:
        no_index.extend(glob.glob(f"{root}/psgs_w100.nq.no_index*"))
        compressed.extend(glob.glob(f"{root}/psgs_w100.nq.compressed*"))
    no_index = sorted(set(no_index))
    compressed = sorted(set(compressed))

    status = {
        "hf_cache_dir": {"path": HF_CACHE_DIR, "exists": os.path.exists(HF_CACHE_DIR), "size": du(HF_CACHE_DIR)},
        "marker": {"path": marker, "exists": os.path.exists(marker)},
        "passages_marker": {"path": passages_marker, "exists": os.path.exists(passages_marker)},
        "dataset_roots": [
            {"path": root, "exists": os.path.exists(root), "size": du(root)}
            for root in dataset_roots
        ],
        "no_index": [{"path": p, "size": du(p)} for p in no_index],
        "compressed": [{"path": p, "size": du(p)} for p in compressed],
    }

    print("Cache status:")
    print(f"  HF cache: {status['hf_cache_dir']}")
    print(f"  Marker: {status['marker']}")
    print(f"  Passages marker: {status['passages_marker']}")
    print("  Dataset roots:")
    for entry in status["dataset_roots"]:
        print(f"    - {entry}")
    print(f"  no_index entries: {len(status['no_index'])}")
    for entry in status["no_index"]:
        print(f"    - {entry}")
    print(f"  compressed entries: {len(status['compressed'])}")
    for entry in status["compressed"]:
        print(f"    - {entry}")

    return status


# ============================================================================
# Usage Examples
# ============================================================================

"""
# Run full evaluation pipeline:
modal run modal_rag_eval.py

# Only setup wiki_dpr index:
modal run modal_rag_eval.py --setup-only

# Only setup wiki_dpr passages:
modal run modal_rag_eval.py::setup_wiki_passages

# Only prepare datasets:
modal run modal_rag_eval.py --datasets-only

# Test mode (single evaluation):
modal run modal_rag_eval.py --test-mode

# Limit dataset size for faster runs:
modal run modal_rag_eval.py --max-samples 200

# Run a single dataset/model with faster settings:
modal run modal_rag_eval.py --datasets nq --models rag_sequence --n-docs 1 --eval-batch-size 8

# Run with a unique results file (parallel-safe):
modal run modal_rag_eval.py --datasets nq --models rag_sequence --results-file evaluation_results_nq.json

# List available datasets:
modal run modal_rag_eval.py::list_datasets

# Clean up results:
modal run modal_rag_eval.py::cleanup_results

# Check cache status:
modal run modal_rag_eval.py::cache_status

# Aggregate results from multiple runs:
modal run modal_rag_eval.py::aggregate_results

# Reproduce Table 3 example generations:
modal run modal_rag_eval.py::table3_examples

# Reproduce Table 3 (skip wiki_dpr setup):
modal run modal_rag_eval.py::table3_examples --skip-setup
"""
