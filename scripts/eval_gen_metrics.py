#!/usr/bin/env python3
"""
Compute BLEU-1 and ROUGE-L for generation datasets (e.g., MS-MARCO).
Reads predictions and gold targets, aligns by line, and outputs metrics.
"""

import argparse
import ast
import json
from typing import List

import pandas as pd
from rouge_score import rouge_scorer
from sacrebleu.metrics import BLEU


def load_predictions(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines()]


def load_references(path: str, mode: str, limit: int) -> List[str]:
    refs: List[str] = []
    if mode == "qa":
        df = pd.read_csv(path, sep="\t", header=None, nrows=limit)
        for answer_list in df[1]:
            try:
                answers = ast.literal_eval(answer_list)
            except Exception:
                answers = []
            ref = ""
            if isinstance(answers, list):
                for a in answers:
                    if isinstance(a, str) and a.strip():
                        ref = a.strip()
                        break
            elif isinstance(answers, str):
                ref = answers.strip()
            refs.append(ref)
    else:
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if limit and i >= limit:
                    break
                refs.append(line.strip())
    return refs


def compute_bleu1(preds: List[str], refs: List[str]) -> float:
    if not preds:
        return 0.0
    bleu = BLEU(max_ngram_order=1)
    return float(bleu.corpus_score(preds, [refs]).score)


def compute_rouge_l(preds: List[str], refs: List[str]) -> float:
    if not preds:
        return 0.0
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    total = 0.0
    for pred, ref in zip(preds, refs):
        score = scorer.score(ref, pred)["rougeL"].fmeasure
        total += score
    return 100.0 * total / len(preds)


def main():
    parser = argparse.ArgumentParser(description="Compute BLEU-1 and ROUGE-L for generation tasks")
    parser.add_argument("--predictions_path", required=True, type=str)
    parser.add_argument("--gold_data_path", required=True, type=str)
    parser.add_argument("--gold_data_mode", default="qa", choices=["qa", "ans"], type=str)
    parser.add_argument("--output_json", default="", type=str)
    args = parser.parse_args()

    preds = load_predictions(args.predictions_path)
    refs = load_references(args.gold_data_path, args.gold_data_mode, limit=len(preds))

    n = min(len(preds), len(refs))
    preds = preds[:n]
    refs = refs[:n]

    bleu1 = compute_bleu1(preds, refs)
    rouge_l = compute_rouge_l(preds, refs)

    print(f"Samples: {n}")
    print(f"BLEU-1: {bleu1:.2f}")
    print(f"ROUGE-L: {rouge_l:.2f}")

    if args.output_json:
        payload = {
            "bleu1": bleu1,
            "rougeL": rouge_l,
            "num_samples": n,
        }
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)


if __name__ == "__main__":
    main()
