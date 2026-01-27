# MS-MARCO Dataset Setup

MS-MARCO requires special handling due to external service issues. This guide provides options for getting MS-MARCO working.

## Quick Status

- ❌ **Automatic HuggingFace download**: Broken (Microsoft blob 409 errors, glob pattern issues)
- ✅ **Kaggle auto-download**: Works if you have credentials configured
- ✅ **Manual download**: Always works

## Option 1: Automatic Kaggle Download (Recommended)

### Local Setup

1. **Install Kaggle package**:
   ```bash
   pip install kaggle
   ```

2. **Get Kaggle API credentials**:
   - Go to https://www.kaggle.com/settings/account
   - Scroll to "API" section
   - Click "Create New Token"
   - This downloads `kaggle.json`

3. **Install credentials**:
   ```bash
   mkdir -p ~/.kaggle
   mv ~/Downloads/kaggle.json ~/.kaggle/
   chmod 600 ~/.kaggle/kaggle.json
   ```

4. **Run dataset preparation**:
   ```bash
   python prepare_eval_datasets.py
   # Or with Modal:
   modal run modal_rag_eval.py::prepare_datasets
   ```

MS-MARCO will auto-download from Kaggle!

### Modal Setup (For Remote Execution)

To enable Kaggle auto-download in Modal, you need to pass credentials as a Modal Secret:

1. **Create Modal secret with Kaggle credentials**:
   ```bash
   modal secret create kaggle-credentials \
     KAGGLE_USERNAME="your-kaggle-username" \
     KAGGLE_KEY="your-kaggle-key"
   ```
   
   Get these values from your `~/.kaggle/kaggle.json`:
   ```json
   {"username": "your-kaggle-username", "key": "your-kaggle-key"}
   ```

2. **Update modal_rag_eval.py** to use the secret (add to function decorator):
   ```python
   @app.function(
       image=image,
       volumes={VOLUME_PATH: volume},
       secrets=[modal.Secret.from_name("kaggle-credentials")],  # Add this line
       timeout=3600,
   )
   def prepare_datasets(max_samples: int = None):
       # ... rest of function
   ```

3. **Run dataset preparation**:
   ```bash
   modal run modal_rag_eval.py::prepare_datasets
   ```

Now MS-MARCO will auto-download in Modal too!

## Option 2: Manual Download (Always Works)

If you don't want to set up Kaggle API credentials:

### Steps

1. **Download from Kaggle** (requires Kaggle account):
   - Visit: https://www.kaggle.com/datasets/parthplc/ms-marco-dataset/data
   - Click "Download" button
   - Extract the ZIP file

2. **For Local Use**:
   ```bash
   cp ~/Downloads/ms-marco-train.csv ./eval_datasets/
   cp ~/Downloads/ms-marco-valid.csv ./eval_datasets/
   python prepare_eval_datasets.py
   ```

3. **For Modal Use**:
   ```bash
   modal volume put rag-data ms-marco-train.csv eval_datasets/ms-marco-train.csv
   modal volume put rag-data ms-marco-valid.csv eval_datasets/ms-marco-valid.csv
   modal run modal_rag_eval.py::prepare_datasets
   ```

The script will automatically detect and use these CSV files.

## Verification

After setup, verify MS-MARCO is working:

```bash
# Local test
python -c "from prepare_eval_datasets import DatasetPreparer; p = DatasetPreparer('eval_datasets', max_samples=5); print(f'Result: {p.prepare_msmarco()} samples')"

# Modal test
modal run modal_rag_eval.py::prepare_datasets --max-samples 5
```

Look for: `✓ Created eval_datasets/msmarco_test.source (X samples)`

## Troubleshooting

### "Kaggle package not installed"
```bash
pip install kaggle
```

### "Could not find kaggle.json"
Follow the credential setup steps above. Make sure `kaggle.json` is in `~/.kaggle/` (Linux/Mac) or `C:\Users\<username>\.kaggle\` (Windows).

### "PermissionError: [Errno 13] Permission denied: kaggle.json"
```bash
chmod 600 ~/.kaggle/kaggle.json
```

### Still not working?
Fall back to Option 2 (manual download) - it always works!

## Why is MS-MARCO Problematic?

1. **Microsoft's blob storage** returns HTTP 409 errors (external issue)
2. **Newer HuggingFace versions** use glob patterns incompatible with datasets==1.18.0
3. **FlashRAG** uses glob patterns that break in datasets==1.18.0

These are all external issues. The Kaggle dataset is a reliable mirror that works consistently.

## Dataset Info

- **Source**: Kaggle mirror of MS-MARCO QA dataset
- **Size**: ~180MB compressed
- **Train samples**: ~808K
- **Validation samples**: ~101K
- **Format**: CSV with query/passage columns
- **Use case**: Document passage retrieval + answer generation

## Do I Need MS-MARCO?

**No** - MS-MARCO is optional for RAG evaluation.

The RAG paper's **core benchmarks** all work automatically:
- ✅ Natural Questions (NQ)
- ✅ TriviaQA
- ✅ WebQuestions
- ✅ CuratedTrec

MS-MARCO adds passage generation evaluation but isn't required for reproducing the paper's main results.
