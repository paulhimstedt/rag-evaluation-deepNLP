# Table 3 Reproduction Findings

Date: 2026-01-29

## Summary
When reproducing Table 3 example generations on Modal, outputs are very short
and often echo the input. This is expected with the current checkpoint choices
and decoding settings.

## Observed Behavior
- MSMARCO examples often return single words or short noun phrases.
- Jeopardy question-generation examples frequently echo the input answer
  string or produce a short related phrase.

## Root Causes
1) Task/model mismatch
   - Table 3 uses task-specific fine-tuned checkpoints in the RAG paper.
   - Current runs use `facebook/rag-token-nq`, `facebook/rag-sequence-nq`,
     and `facebook/bart-large`, which are tuned for NQ or general seq2seq,
     not MSMARCO NLG or Jeopardy question generation.
   - This biases the models toward short answer-style outputs.

2) Decoding settings allow very short outputs
   - `eval_rag.py` uses `min_length=1` and no length penalties, so the decoder
     can terminate early once it emits a short answer.

3) Jeopardy QG inputs are not prompted
   - For QG, inputs are the answer strings only, with no explicit "generate a
     question" prompt. Targets are placeholders, so metrics are not meaningful.

## Attempted Switch to Task-Specific Checkpoints
Goal: Use task-specific fine-tuned checkpoints for MSMARCO NLG and Jeopardy QG.

Result: As of 2026-01-29, no official, publicly discoverable Hugging Face RAG
checkpoints for MSMARCO NLG or Jeopardy QG were found. The official Meta/Facebook
RAG releases on HF appear limited to the NQ variants
(rag-token-nq / rag-sequence-nq).

## Recommendations
- If you have task-specific model IDs (local or Hugging Face), provide them and
  we can wire them into the Table 3 run.
- Otherwise, consider:
  - Fine-tuning RAG on MSMARCO NLG and Jeopardy QG.
  - Adjusting decoding settings (min_length, length_penalty, max_length) for
    longer generations.
  - Adding explicit QG prompting (e.g., "Answer: X. Question:").
