# Using .env file for HuggingFace Token

The `.env` file has been created for you to store your HuggingFace token.

## Setup Instructions

1. **Get your HuggingFace token**:
   - Go to: https://huggingface.co/settings/tokens
   - Create a new token (read access is sufficient)
   - Copy the token

2. **Edit the .env file**:
   ```bash
   # Open the .env file and replace the placeholder
   nano .env
   # or
   code .env
   ```

3. **Replace the placeholder**:
   ```bash
   # Change this:
   HF_TOKEN=your_token_here
   
   # To this (with your actual token):
   HF_TOKEN=hf_yourActualTokenHere
   ```

4. **For Modal deployment**:
   
   The token in `.env` is for **local testing only**. For Modal, you need to set it as a secret:
   
   ```bash
   modal secret create huggingface HF_TOKEN=hf_yourActualTokenHere
   ```

## Usage

### Local Testing
```bash
# The token will be automatically loaded from .env
python test_local_prep.py webquestions
```

### Modal Deployment
```bash
# Modal will use the secret you set
modal run modal_rag_eval.py --test-mode
```

## Important Notes

- ✅ `.env` is in `.gitignore` - your token won't be committed to git
- ✅ `.env.example` is the template (safe to commit)
- ✅ Never share your HF_TOKEN publicly
- ✅ The token provides read access to HuggingFace datasets

## Verification

To check if your token is set correctly:

```bash
# Local
python -c "from dotenv import load_dotenv; import os; load_dotenv(); print('Token set!' if os.getenv('HF_TOKEN') else 'Token not found')"

# Modal
modal secret list
```
