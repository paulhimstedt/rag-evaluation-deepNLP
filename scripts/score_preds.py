#!/usr/bin/env python3
"""
Score EM/F1 from prediction files without importing transformers.

Default mode: scan a directory for *_preds.txt files and compute EM/F1 using
gold targets in eval_datasets/.
"""

import argparse
import ast
import csv
import glob
import os
import re
import string
from collections import Counter


def normalize_answer(text: str) -> str:
    def remove_articles(t):
        return re.sub(r"\b(a|an|the)\b", " ", t)

    def white_space_fix(t):
        return " ".join(t.split())

    def remove_punc(t):
        exclude = set(string.punctuation)
        return "".join(ch for ch in t if ch not in exclude)

    def lower(t):
        return t.lower()

    return white_space_fix(remove_articles(remove_punc(lower(text))))


def exact_match(prediction: str, ground_truth: str) -> bool:
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()
    if not pred_tokens and not gt_tokens:
        return 1.0
    if not pred_tokens or not gt_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def metric_max(metric_fn, prediction, ground_truths):
    return max(metric_fn(prediction, gt) for gt in ground_truths)


def load_gold_answers(gold_path, limit):
    golds = []
    with open(gold_path, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if limit is not None and len(golds) >= limit:
                break
            if len(row) < 2:
                continue
            try:
                answers = ast.literal_eval(row[1])
                if isinstance(answers, str):
                    answers = [answers]
            except Exception:
                answers = [row[1]]
            golds.append(answers)
    return golds


def score_preds(preds_path, gold_path):
    preds = [line.rstrip("\n") for line in open(preds_path, encoding="utf-8")]
    if not preds:
        return 0, 0.0, 0.0
    golds = load_gold_answers(gold_path, limit=len(preds))

    total = 0
    em = 0.0
    f1 = 0.0
    for pred, gts in zip(preds, golds):
        em += metric_max(exact_match, pred, gts)
        f1 += metric_max(f1_score, pred, gts)
        total += 1
    return total, 100.0 * em / total, 100.0 * f1 / total


def parse_dataset_model(filename, models):
    for model in models:
        suffix = f"_{model}_preds.txt"
        if filename.endswith(suffix):
            dataset = filename[: -len(suffix)]
            return dataset, model
    return None, None


def main():
    parser = argparse.ArgumentParser(description="Score EM/F1 from prediction files.")
    parser.add_argument("--preds", type=str, help="Path to a single *_preds.txt file")
    parser.add_argument("--gold", type=str, help="Path to corresponding *_test.target file")
    parser.add_argument("--preds-dir", type=str, default="results_from_modal")
    parser.add_argument("--gold-dir", type=str, default="eval_datasets")
    parser.add_argument("--pattern", type=str, default="*_preds.txt")
    parser.add_argument("--output-csv", type=str, default="")
    args = parser.parse_args()

    models = ["rag_sequence", "rag_token", "bart"]

    results = []
    if args.preds:
        if not args.gold:
            raise SystemExit("--gold is required when --preds is provided")
        total, em, f1 = score_preds(args.preds, args.gold)
        results.append({
            "dataset": os.path.basename(args.gold).replace("_test.target", ""),
            "model": "unknown",
            "samples": total,
            "em": em,
            "f1": f1,
            "preds": args.preds,
        })
    else:
        pattern = os.path.join(args.preds_dir, args.pattern)
        for path in sorted(glob.glob(pattern)):
            filename = os.path.basename(path)
            dataset, model = parse_dataset_model(filename, models)
            if not dataset:
                continue
            gold_path = os.path.join(args.gold_dir, f"{dataset}_test.target")
            if not os.path.exists(gold_path):
                print(f"Skipping {filename}: missing gold {gold_path}")
                continue
            total, em, f1 = score_preds(path, gold_path)
            results.append({
                "dataset": dataset,
                "model": model,
                "samples": total,
                "em": em,
                "f1": f1,
                "preds": path,
            })

    if not results:
        print("No results found.")
        return

    print(f"{'dataset':15s} {'model':12s} {'samples':8s} {'EM':7s} {'F1':7s}")
    for r in results:
        print(f"{r['dataset']:15s} {r['model']:12s} {r['samples']:<8d} {r['em']:7.2f} {r['f1']:7.2f}")

    if args.output_csv:
        with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["dataset", "model", "samples", "em", "f1", "preds"])
            writer.writeheader()
            writer.writerows(results)
        print(f"Wrote {len(results)} rows to {args.output_csv}")


if __name__ == "__main__":
    main()
