#!/usr/bin/env python3
"""
Retrieve and display dataset preparation logs from Modal.
"""
import modal

# Get the app
app = modal.App.lookup("rag-evaluation", create_if_missing=False)

# Get the logs function
get_dataset_logs = modal.Function.lookup("rag-evaluation", "get_dataset_logs")

print("=" * 80)
print("Retrieving Dataset Preparation Logs from Modal")
print("=" * 80)

try:
    logs = get_dataset_logs.remote()
    
    print("\n" + "=" * 80)
    print("DATASET PREPARATION RESULTS")
    print("=" * 80)
    
    if logs.get('results'):
        results = logs['results']
        print("\nDataset Status:")
        for dataset, count in results.items():
            status = f"✓ {count} samples" if count > 0 else "✗ Failed"
            print(f"  {dataset:20s}: {status}")
    
    print("\n" + "=" * 80)
    print("FULL PREPARATION LOG")
    print("=" * 80)
    print()
    
    if logs.get('full_log'):
        # Split log by dataset sections for easier reading
        log_content = logs['full_log']
        
        # Print MS-MARCO section
        if "=== Preparing MS-MARCO" in log_content:
            print("\n" + "=" * 80)
            print("MS-MARCO PREPARATION")
            print("=" * 80)
            ms_start = log_content.find("=== Preparing MS-MARCO")
            ms_end = log_content.find("=== Preparing SearchQA")
            if ms_end == -1:
                ms_end = log_content.find("=== Preparing FEVER")
            if ms_end != -1:
                print(log_content[ms_start:ms_end])
            else:
                print(log_content[ms_start:])
        
        # Print SearchQA section
        if "=== Preparing SearchQA" in log_content:
            print("\n" + "=" * 80)
            print("SEARCHQA PREPARATION")
            print("=" * 80)
            sq_start = log_content.find("=== Preparing SearchQA")
            sq_end = log_content.find("=== Preparing FEVER")
            if sq_end == -1:
                sq_end = log_content.find("Dataset Preparation Summary")
            if sq_end != -1:
                print(log_content[sq_start:sq_end])
            else:
                print(log_content[sq_start:])
    else:
        print("No logs available")
    
    print("\n" + "=" * 80)

except Exception as e:
    print(f"\n✗ Error retrieving logs: {e}")
    print("\nMake sure you've run the Modal evaluation at least once:")
    print("  modal run modal_rag_eval.py --datasets-only")
