#!/usr/bin/env python3
"""
Local test script for dataset preparation.
Tests downloading and preparing a single dataset before running on Modal.

Usage:
    python test_local_prep.py [dataset_name]

Examples:
    python test_local_prep.py nq           # Test Natural Questions only
    python test_local_prep.py triviaqa     # Test TriviaQA only
    python test_local_prep.py              # Test all datasets
"""

import argparse
import sys
from pathlib import Path

from prepare_eval_datasets import DatasetPreparer


def test_single_dataset(preparer, dataset_name):
    """Test preparation of a single dataset."""
    print(f"\n{'='*80}")
    print(f"Testing {dataset_name} dataset preparation")
    print(f"{'='*80}\n")

    try:
        if dataset_name == 'nq':
            result = preparer.prepare_nq()
        elif dataset_name == 'triviaqa':
            result = preparer.prepare_triviaqa()
        elif dataset_name == 'webquestions':
            result = preparer.prepare_webquestions()
        elif dataset_name == 'curatedtrec':
            result = preparer.prepare_curatedtrec()
        elif dataset_name == 'msmarco':
            result = preparer.prepare_msmarco()
        elif dataset_name == 'searchqa':
            result = preparer.prepare_searchqa()
        elif dataset_name == 'fever':
            result = preparer.prepare_fever()
        else:
            print(f"Unknown dataset: {dataset_name}")
            return False

        if result > 0:
            print(f"\n✓ {dataset_name} preparation successful: {result} samples")
            return True
        else:
            print(f"\n✗ {dataset_name} preparation failed")
            return False

    except Exception as e:
        print(f"\n✗ Error preparing {dataset_name}: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_dataset_files(output_dir, dataset_name, split='test'):
    source_path = Path(output_dir) / f"{dataset_name}_{split}.source"
    target_path = Path(output_dir) / f"{dataset_name}_{split}.target"

    if not source_path.exists():
        print(f"  ✗ Missing source file: {source_path}")
        return False

    if not target_path.exists():
        print(f"  ✗ Missing target file: {target_path}")
        return False

    # Count lines (FIX: specify encoding)
    with open(source_path, encoding="utf-8") as f:
        source_lines = sum(1 for _ in f)
    with open(target_path, encoding="utf-8") as f:
        target_lines = sum(1 for _ in f)

    print(f"  ✓ Source file: {source_path} ({source_lines} lines)")
    print(f"  ✓ Target file: {target_path} ({target_lines} lines)")

    if source_lines != target_lines:
        print(f"  ⚠ Warning: Line count mismatch! Source: {source_lines}, Target: {target_lines}")

    # Show sample
    print(f"\n  Sample from {dataset_name}:")
    with open(source_path, encoding="utf-8") as f:
        sample_source = f.readline().strip()
    with open(target_path, encoding="utf-8") as f:
        sample_target = f.readline().strip()

    print(f"    Q: {sample_source[:100]}...")
    print(f"    A: {sample_target[:100]}...")

    return True



def main():
    parser = argparse.ArgumentParser(
        description="Test dataset preparation locally"
    )
    parser.add_argument(
        'dataset',
        nargs='?',
        choices=['nq', 'triviaqa', 'webquestions', 'curatedtrec',
                 'msmarco', 'searchqa', 'fever', 'all'],
        default='all',
        help='Dataset to test (default: all)'
    )
    parser.add_argument(
        '--output-dir',
        default='./eval_datasets_test',
        help='Output directory for test datasets'
    )

    args = parser.parse_args()

    # Create preparer
    preparer = DatasetPreparer(output_dir=args.output_dir)

    # Test datasets
    if args.dataset == 'all':
        datasets = ['nq', 'triviaqa', 'webquestions', 'curatedtrec',
                   'msmarco', 'searchqa', 'fever']
    else:
        datasets = [args.dataset]

    results = {}
    for dataset in datasets:
        success = test_single_dataset(preparer, dataset)
        results[dataset] = success

        if success:
            # Verify files were created
            print(f"\nVerifying {dataset} files:")
            verify_dataset_files(args.output_dir, dataset)

    # Print summary
    print(f"\n{'='*80}")
    print("Test Summary")
    print(f"{'='*80}")

    for dataset, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{dataset:20s}: {status}")

    total = len(results)
    passed = sum(1 for s in results.values() if s)
    print(f"\nTotal: {passed}/{total} datasets prepared successfully")

    if passed < total:
        print("\n⚠ Some datasets failed to prepare. Check errors above.")
        sys.exit(1)
    else:
        print("\n✓ All datasets prepared successfully!")
        print(f"\nDatasets saved to: {args.output_dir}")
        print("\nYou can now run the full Modal evaluation:")
        print("  modal run modal_rag_eval.py")
        sys.exit(0)


if __name__ == "__main__":
    main()
