#!/usr/bin/env python3
"""
Local dataset preparation script for RAG evaluation.

This script prepares all evaluation datasets locally (where we have newer
dependencies and fewer constraints) and saves them to eval_datasets/.
These prepared files can then be uploaded to Modal for evaluation.

Usage:
    python prepare_datasets_local.py [--max-samples N]

After running, use upload_datasets_to_modal.sh to sync to Modal.
"""

import argparse
import sys
from pathlib import Path

from prepare_eval_datasets import DatasetPreparer


def main():
    parser = argparse.ArgumentParser(description='Prepare RAG evaluation datasets locally')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Limit number of samples per dataset (for testing)')
    parser.add_argument('--output-dir', type=str, default='eval_datasets',
                        help='Output directory for prepared datasets')
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 80)
    print("RAG Evaluation - Local Dataset Preparation")
    print("=" * 80)
    print(f"Output directory: {output_dir.absolute()}")
    if args.max_samples:
        print(f"Max samples: {args.max_samples} (testing mode)")
    print("=" * 80)
    print()
    
    preparer = DatasetPreparer(output_dir=output_dir, max_samples=args.max_samples)
    results = preparer.prepare_all()
    
    print()
    print("=" * 80)
    print("Dataset Preparation Complete!")
    print("=" * 80)
    
    # Summary
    total_samples = 0
    working_datasets = 0
    for dataset, count in results.items():
        status = "✓" if count > 0 else "✗"
        print(f"{status} {dataset:20s}: {count:6d} samples")
        if count > 0:
            total_samples += count
            working_datasets += 1
    
    print("=" * 80)
    print(f"Total: {working_datasets}/7 datasets working, {total_samples} samples")
    print()
    
    # Show prepared files
    prepared_files = sorted(output_dir.glob("*.source"))
    if prepared_files:
        print("Prepared files:")
        for f in prepared_files:
            target = f.with_suffix('.target')
            size_src = f.stat().st_size / 1024
            size_tgt = target.stat().st_size / 1024 if target.exists() else 0
            print(f"  {f.name:30s} ({size_src:7.1f} KB)")
            if target.exists():
                print(f"  {target.name:30s} ({size_tgt:7.1f} KB)")
        
        print()
        print("=" * 80)
        print("Next steps:")
        print("=" * 80)
        print("1. Review the prepared datasets in", output_dir.absolute())
        print("2. Upload to Modal:")
        print("   bash upload_datasets_to_modal.sh")
        print("3. Run evaluation:")
        print("   modal run modal_rag_eval.py --test-mode")
        print("")
        print("📚 Documentation: See docs/DATASET_PREPARATION_GUIDE.md for details")
        print()
    else:
        print("⚠  No datasets were successfully prepared!")
        sys.exit(1)


if __name__ == '__main__':
    main()
