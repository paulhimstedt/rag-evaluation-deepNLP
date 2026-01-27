# Troubleshooting Guide - RAG Evaluation on Modal

This guide covers potential issues you may encounter during evaluation and their solutions.

## Issues Already Fixed

### ✅ Modal API Changes (Fixed)
- **Issue**: `modal.Stub` not found
- **Solution**: Updated to `modal.App` (Modal 1.0 API)
- **Status**: Fixed in code

### ✅ Python Version Compatibility (Fixed)
- **Issue**: Python 3.9 not supported
- **Solution**: Updated to Python 3.10
- **Status**: Fixed in code

### ✅ PyArrow Compatibility (Fixed)
- **Issue**: `PyExtensionType` not found in pyarrow 21.0+
- **Solution**: Pinned `pyarrow>=6.0.0,<15.0.0`
- **Status**: Fixed in code

### ✅ NumPy Version Incompatibility (Fixed)
- **Issue**: NumPy 2.x incompatible with older pyarrow/datasets
- **Solution**: Pinned `numpy>=1.19.0,<2.0.0`
- **Status**: Fixed in code

### ✅ pytorch-lightning Build Error (Fixed)
- **Issue**: Old pytorch-lightning incompatible with Python 3.10+
- **Solution**: Removed (not needed for evaluation)
- **Status**: Fixed in code

## Potential Future Issues

### 1. Dataset Download Issues

#### Issue: wiki_dpr Index Download Timeout
**Symptom:**
```
TimeoutError: Function timed out after 7200 seconds
```

**Solutions:**
```python
# Option A: Increase timeout in modal_rag_eval.py
@app.function(
    timeout=10800,  # 3 hours instead of 2
)

# Option B: Use smaller index (if available)
dataset = load_dataset("facebook/wiki_dpr", "psgs_w100.nq.exact")
```

#### Issue: HuggingFace Hub Rate Limiting
**Symptom:**
```
HTTPError: 429 Too Many Requests
```

**Solution:**
```python
# Set HF token if you have one
os.environ['HF_TOKEN'] = 'your_token_here'
```

#### Issue: Dataset Download Failure (Network)
**Symptom:**
```
ConnectionError: Failed to download dataset
```

**Solution:**
```bash
# Retry with exponential backoff
modal run modal_rag_eval.py --setup-only  # Try again
```

### 2. Memory Issues

#### Issue: GPU Out of Memory (OOM)
**Symptom:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**
```python
# Option A: Reduce batch size in run_evaluation()
'--eval_batch_size', '4',  # or even '2'

# Option B: Use larger GPU
@app.function(
    gpu="A10G",  # or "A100"
)

# Option C: Use CPU for BART baseline
if model_type == 'bart':
    # Don't use GPU for BART
    pass
```

#### Issue: Container Memory Exceeded
**Symptom:**
```
MemoryError: Container killed due to memory limit
```

**Solutions:**
```python
# Increase memory allocation
@app.function(
    memory=32768,  # 32GB instead of 16GB
)
```

### 3. Volume and Storage Issues

#### Issue: Volume Full
**Symptom:**
```
OSError: [Errno 28] No space left on device
```

**Solutions:**
```bash
# Check volume usage
modal volume ls rag-data

# Clean up old results
modal run modal_rag_eval.py::cleanup_results

# Increase volume size (if needed)
# Modal volumes are auto-expanding, but you can check limits
```

#### Issue: Volume Commit Failed
**Symptom:**
```
VolumeCommitError: Failed to commit volume changes
```

**Solution:**
```python
# Add retry logic
try:
    volume.commit()
except Exception as e:
    print(f"Commit failed: {e}, retrying...")
    time.sleep(5)
    volume.commit()
```

### 4. Model Loading Issues

#### Issue: Model Not Found
**Symptom:**
```
OSError: facebook/rag-sequence-nq does not appear to be a valid model
```

**Solutions:**
```python
# Option A: Verify model exists
# Check https://huggingface.co/facebook/rag-sequence-nq

# Option B: Use alternative model name
model_name = "facebook/rag-sequence-base"  # Try base instead of nq
```

#### Issue: Incompatible Model Version
**Symptom:**
```
ValueError: Model was trained with transformers v4.x, but you have v4.y
```

**Solution:**
```python
# Pin exact transformers version that matches model
.pip_install("transformers==4.30.2")
```

### 5. FAISS Index Issues

#### Issue: Index Corruption
**Symptom:**
```
RuntimeError: Failed to load FAISS index
```

**Solutions:**
```bash
# Delete and re-download index
modal run modal_rag_eval.py --setup-only  # Force re-download

# Or manually clear cache
modal volume rm rag-data:/huggingface_cache/datasets/facebook___wiki_dpr
```

#### Issue: Index Format Mismatch
**Symptom:**
```
ValueError: Index type not recognized
```

**Solution:**
```python
# Explicitly specify index type
model_kwargs["index_name"] = "compressed"  # or "exact" or "legacy"
```

### 6. Evaluation Script Issues

#### Issue: eval_rag.py Import Error
**Symptom:**
```
ModuleNotFoundError: No module named 'utils_rag'
```

**Solution:**
```python
# Ensure CODE_DIR is in Python path
import sys
sys.path.insert(0, "/workspace")
os.chdir("/workspace")  # Also change working directory
```

#### Issue: Dataset Format Mismatch
**Symptom:**
```
ValueError: Expected tab-separated format
```

**Solution:**
```python
# Check dataset format
with open(source_path) as f:
    print(f.readline())  # Should be plain text
with open(target_path) as f:
    print(f.readline())  # Should be "question\t['answer1']"

# Fix in prepare_eval_datasets.py if needed
```

### 7. GPU Availability Issues

#### Issue: No GPUs Available
**Symptom:**
```
ResourceExhausted: No T4 GPUs available
```

**Solutions:**
```python
# Option A: Wait and retry
# Modal will queue your job

# Option B: Use different GPU
@app.function(
    gpu="A10G",  # Try different tier
)

# Option C: Run CPU-only (slower)
@app.function(
    cpu=8.0,
)
# And modify eval_rag.py to use CPU
args.device = torch.device("cpu")
```

### 8. Permission and Path Issues

#### Issue: Permission Denied
**Symptom:**
```
PermissionError: [Errno 13] Permission denied: '/data/...'
```

**Solution:**
```python
# Ensure directories are created with proper permissions
os.makedirs(RESULTS_DIR, exist_ok=True, mode=0o755)
```

#### Issue: File Not Found
**Symptom:**
```
FileNotFoundError: [Errno 2] No such file or directory: '/workspace/eval_rag.py'
```

**Solution:**
```python
# Verify files were mounted
import os
print(os.listdir("/workspace"))

# Check .add_local_dir() includes the file
# Make sure file doesn't match ignore patterns
```

### 9. Networking and Timeouts

#### Issue: Connection Timeout
**Symptom:**
```
requests.exceptions.ConnectTimeout: Connection timed out
```

**Solution:**
```python
# Increase timeouts in download functions
urllib.request.urlretrieve(url, path, timeout=300)  # 5 minutes

# Or add retry logic with exponential backoff
for attempt in range(3):
    try:
        download_file(url)
        break
    except TimeoutError:
        time.sleep(2 ** attempt)
```

### 10. Results Parsing Issues

#### Issue: Metrics Not Parsed
**Symptom:**
```
KeyError: 'em' not in metrics
```

**Solution:**
```python
# Check eval_rag.py output format
print(result.stdout)  # Debug output

# Update regex in parse_metrics_from_output()
em_match = re.search(r'EM:\s+([\d.]+)', output, re.IGNORECASE)
```

## Debugging Strategies

### Enable Verbose Logging

```python
# In modal_rag_eval.py, add to run_evaluation()
cmd.extend(['--print_predictions', '--print_docs'])

# Check Modal logs
modal app logs rag-evaluation
```

### Test Components Individually

```bash
# Test dataset preparation locally
python test_local_prep.py nq

# Test single dataset on Modal
modal run modal_rag_eval.py --test-mode

# Test setup only
modal run modal_rag_eval.py --setup-only
```

### Check Volume Contents

```python
@app.function(volumes={"/data": volume})
def debug_volume():
    import os
    print("Volume contents:")
    for root, dirs, files in os.walk("/data"):
        print(f"{root}:")
        for d in dirs[:5]:  # First 5 dirs
            print(f"  DIR: {d}")
        for f in files[:5]:  # First 5 files
            print(f"  FILE: {f}")
        if len(files) > 5:
            print(f"  ... and {len(files)-5} more files")
        break  # Only show top level

modal run modal_rag_eval.py::debug_volume
```

### Monitor Resource Usage

```python
# Add to evaluation function
import psutil
print(f"CPU: {psutil.cpu_percent()}%")
print(f"Memory: {psutil.virtual_memory().percent}%")
print(f"Disk: {psutil.disk_usage('/data').percent}%")
```

## Emergency Fixes

### Force Clean Start

```bash
# Delete volume and start fresh
modal volume rm rag-data
modal run modal_rag_eval.py --setup-only
```

### Skip Problematic Dataset

```python
# In modal_rag_eval.py main(), modify datasets list:
datasets_to_eval = ['nq', 'triviaqa']  # Skip others temporarily
```

### Use Fallback Models

```python
# If facebook models fail, try alternatives
MODELS = {
    'rag_sequence': 'facebook/rag-sequence-base',  # Use base instead of nq
    'rag_token': 'facebook/rag-token-base',
    'bart': 'facebook/bart-base',  # Use smaller model
}
```

## Getting Help

### Check Modal Status

- Modal Dashboard: https://modal.com/apps
- Modal Status: https://status.modal.com

### Useful Commands

```bash
# View running apps
modal app list

# View logs
modal app logs rag-evaluation

# Stop running app
modal app stop rag-evaluation

# Check volume
modal volume ls rag-data
```

### Information to Include When Reporting Issues

1. **Full error traceback**
2. **Modal run command used**
3. **modal_rag_eval.py version** (first 10 lines)
4. **Python/pip versions** in local environment
5. **Modal app logs** (last 100 lines)
6. **Volume size** (`modal volume ls rag-data`)

## Common Error Patterns

### Pattern: Import Errors
```
ModuleNotFoundError: No module named 'X'
```
**Likely Cause**: Dependency not installed or version mismatch
**Fix**: Add to `.pip_install()` in modal_rag_eval.py

### Pattern: CUDA Errors
```
RuntimeError: CUDA error: ...
```
**Likely Cause**: GPU memory issue or driver incompatibility
**Fix**: Reduce batch size or use different GPU tier

### Pattern: Timeout Errors
```
TimeoutError: ... timed out after X seconds
```
**Likely Cause**: Operation taking longer than expected
**Fix**: Increase timeout parameter in function decorator

### Pattern: Volume Errors
```
VolumeError: ...
```
**Likely Cause**: Volume not committed or path issues
**Fix**: Ensure `volume.commit()` is called after writes

## Performance Optimization

### Speed Up Evaluations

```python
# Use A10G instead of T4 (2-3x faster, slightly more expensive)
@app.function(gpu="A10G")

# Increase batch size if memory allows
'--eval_batch_size', '16',  # From 8

# Run multiple evaluations in parallel
# (Advanced: requires modifying main() to use asyncio)
```

### Reduce Costs

```python
# Use T4 GPU (slower but cheaper)
@app.function(gpu="T4")

# Reduce batch size to use smaller GPU
'--eval_batch_size', '4',

# Skip some datasets for initial testing
datasets_to_eval = ['nq']  # Just one dataset
```

## Version Compatibility Matrix

| Component | Version | Compatible With |
|-----------|---------|-----------------|
| Python | 3.10 | Modal 2025.06+ |
| numpy | 1.19-1.26 | pyarrow <15.0 |
| pyarrow | 6.0-14.0 | datasets 1.18.0 |
| datasets | 1.18.0 | transformers 4.x |
| transformers | 4.30.2 | torch 2.0.x |
| torch | 2.0.1 | Python 3.10 |
| faiss-cpu | 1.7.4 | numpy <2.0 |

## Still Having Issues?

If you're stuck after trying these solutions:

1. **Simplify**: Try running just one component (setup, datasets, single eval)
2. **Update**: Check if there's a newer version of the code
3. **Local Test**: Try `test_local_prep.py` to isolate the issue
4. **Ask**: Open an issue with full details (see "Information to Include" above)

## Success Indicators

You'll know everything is working when you see:

```
✓ Setup complete: {'status': 'success', 'num_passages': 21015324}
✓ Datasets prepared: {'nq': 3610, 'triviaqa': 11313, ...}
✓ Evaluation complete
  Samples: 3610
  Metrics: {'em': 44.3, 'f1': 52.1}
```

Good luck with your evaluation! 🚀
