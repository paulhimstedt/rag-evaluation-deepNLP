#!/usr/bin/env python3
"""
Plot sample-size sensitivity for NQ EM and MS-MARCO BLEU/ROUGE.
"""

import argparse
import glob
import json
import math
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


def load_results(paths: List[str]) -> List[Dict]:
    results: List[Dict] = []
    for path in paths:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                results.extend(data)
        except Exception:
            continue
    return results


def collect_by_samples(
    results: List[Dict],
    dataset: str,
    model: str,
    metric: str,
    n_docs: int,
    eval_mode: str,
    sample_sizes: List[int],
) -> List[Tuple[int, float]]:
    points: Dict[int, float] = {}
    for r in results:
        if r.get("dataset") != dataset or r.get("model") != model:
            continue
        if eval_mode and r.get("eval_mode") != eval_mode:
            continue
        if n_docs and int(r.get("n_docs", -1)) != n_docs:
            continue
        metrics = r.get("metrics", {})
        if metric not in metrics:
            continue
        samples = r.get("num_samples") or r.get("max_eval_samples")
        if samples is None:
            continue
        samples = int(samples)
        if sample_sizes and samples not in sample_sizes:
            continue
        points[samples] = float(metrics[metric])

    ordered = sorted(points.items(), key=lambda x: x[0])
    return ordered


def summarize(points: List[Tuple[int, float]]) -> Dict:
    if not points:
        return {}
    vals = [p[1] for p in points]
    mean = sum(vals) / len(vals)
    variance = sum((v - mean) ** 2 for v in vals) / len(vals)
    std = math.sqrt(variance)
    return {
        "mean": mean,
        "std": std,
        "min": min(vals),
        "max": max(vals),
        "range": max(vals) - min(vals),
    }


def plot_lines(ax, points: List[Tuple[int, float]], label: str, style: str):
    if not points:
        return
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    ax.plot(xs, ys, style, label=label, linewidth=2)


def main():
    parser = argparse.ArgumentParser(description="Plot sample-size sensitivity")
    parser.add_argument("--results-dir", default="./results_from_modal", type=str)
    parser.add_argument("--results-files", default="", type=str)
    parser.add_argument("--output", default="sample_size_sensitivity.png", type=str)
    parser.add_argument("--summary-output", default="", type=str)
    parser.add_argument("--k", default=10, type=int)
    parser.add_argument("--sample-sizes", default="100,300,500", type=str)
    args = parser.parse_args()

    sample_sizes = [int(s.strip()) for s in args.sample_sizes.split(",") if s.strip()]

    paths: List[str] = []
    if args.results_files:
        paths = [p.strip() for p in args.results_files.split(",") if p.strip()]
    else:
        paths = glob.glob(f"{args.results_dir}/*.json")
        if not paths:
            paths = glob.glob(f"{args.results_dir}/results/*.json")

    results = load_results(paths)
    if not results:
        raise SystemExit("No evaluation results found. Provide --results-files or --results-dir.")

    summary: Dict[str, Dict] = {}

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    if sample_sizes:
        fig.suptitle(f"Sample sizes: {', '.join(str(s) for s in sample_sizes)}", fontsize=10)

    # NQ EM
    ax = axes[0]
    label_map = {"rag_token": "RAG-Token", "rag_sequence": "RAG-Seq"}
    for model, style in [("rag_token", "o-"), ("rag_sequence", "s-")]:
        pts = collect_by_samples(results, "nq", model, "em", args.k, "e2e", sample_sizes)
        plot_lines(ax, pts, label_map.get(model, model), style)
        summary_key = f"nq:{model}:em"
        summary[summary_key] = summarize(pts)
    ax.set_title("NQ Exact Match vs Sample Size")
    ax.set_xlabel("Samples")
    ax.set_ylabel("EM")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # MS-MARCO BLEU/ROUGE
    ax = axes[1]
    for model, style in [("rag_token", "o"), ("rag_sequence", "s")]:
        pts_rouge = collect_by_samples(results, "msmarco", model, "rougeL", args.k, "e2e", sample_sizes)
        pts_bleu = collect_by_samples(results, "msmarco", model, "bleu1", args.k, "e2e", sample_sizes)
        label = label_map.get(model, model)
        plot_lines(ax, pts_rouge, f"{label} ROUGE-L", f"{style}-")
        plot_lines(ax, pts_bleu, f"{label} BLEU-1", f"{style}--")
        summary[f"msmarco:{model}:rougeL"] = summarize(pts_rouge)
        summary[f"msmarco:{model}:bleu1"] = summarize(pts_bleu)
    ax.set_title("MS-MARCO BLEU-1 / ROUGE-L vs Sample Size")
    ax.set_xlabel("Samples")
    ax.set_ylabel("Score")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    fig.savefig(args.output, dpi=200)
    print(f"Saved figure to {args.output}")

    if args.summary_output:
        payload = {
            "sample_sizes": sample_sizes,
            "metrics": summary,
        }
        with open(args.summary_output, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Saved summary to {args.summary_output}")
    else:
        print("Summary:")
        if sample_sizes:
            print(f"  sample_sizes: {sample_sizes}")
        for key, stats in summary.items():
            if not stats:
                continue
            print(
                f"  {key}: mean={stats['mean']:.2f}, std={stats['std']:.2f}, "
                f"range={stats['range']:.2f}"
            )


if __name__ == "__main__":
    main()
