"""Test SearchQA dataset loading and structure with datasets==1.18.0"""
from datasets import load_dataset

print("Testing SearchQA dataset loading...")
print("=" * 70)

try:
    print("\nLoading kyunghyuncho/search_qa with 'train_test_val' config...")
    dataset = load_dataset("kyunghyuncho/search_qa", "train_test_val")
    
    print(f"✓ Successfully loaded!")
    print(f"\nAvailable splits: {list(dataset.keys())}")
    
    # Check test split
    test_data = dataset["test"]
    print(f"\nTEST Split:")
    print(f"  Sample count: {len(test_data)}")
    print(f"  Fields: {list(test_data[0].keys())}")
    
    # Show first sample
    print(f"\n  First sample:")
    sample = test_data[0]
    for key, value in sample.items():
        if isinstance(value, str):
            display = value[:100] + "..." if len(value) > 100 else value
            print(f"    {key}: {display}")
        elif isinstance(value, list):
            print(f"    {key}: [list with {len(value)} items]")
            if len(value) > 0 and isinstance(value[0], str):
                print(f"      First item: {value[0][:80]}...")
        else:
            print(f"    {key}: {value}")
    
    # Verify Q&A extraction
    print("\n  Extracting Q&A for evaluation:")
    question = sample.get('question', '')
    answer = sample.get('answer', '')
    
    print(f"    Question: {question[:100] if question else 'NOT FOUND'}...")
    print(f"    Answer: {answer[:100] if answer else 'NOT FOUND'}...")
    
    if question and answer:
        print("\n✓ SearchQA dataset verified - ready for evaluation!")
    else:
        print("\n⚠ Warning: Question or answer field missing!")
    
    print("=" * 70)
    
except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    print("\nFull traceback:")
    traceback.print_exc()
