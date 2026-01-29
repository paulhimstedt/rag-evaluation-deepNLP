#!/usr/bin/env python3
"""
Plot Figure 3-style panels from evaluation results JSON files.
"""

import argparse
import glob
import json
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


def collect_metric(results: List[Dict], dataset: str, model: str, metric: str) -> List[Tuple[int, float]]:
    points: List[Tuple[int, float]] = []
    for r in results:
        if r.get("dataset") != dataset or r.get("model") != model:
            continue
        metrics = r.get("metrics", {})
        if metric not in metrics:
            continue
        n_docs = r.get("n_docs")
        if n_docs is None:
            continue
        points.append((int(n_docs), float(metrics[metric])))
    points.sort(key=lambda x: x[0])
    return points


def plot_lines(ax, points: List[Tuple[int, float]], label: str, style: str):
    if not points:
        return
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    ax.plot(xs, ys, style, label=label, linewidth=2)


def main():
    parser = argparse.ArgumentParser(description="Plot Figure 3 from evaluation results JSONs")
    parser.add_argument("--results-dir", default="./results_from_modal", type=str)
    parser.add_argument("--results-files", default="", type=str)
    parser.add_argument("--output", default="figure3.png", type=str)
    args = parser.parse_args()

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

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)

    # Left: NQ EM vs K
    ax = axes[0]
    plot_lines(ax, collect_metric(results, "nq", "rag_token", "em"), "RAG-Token", "o-")
    plot_lines(ax, collect_metric(results, "nq", "rag_sequence", "em"), "RAG-Seq", "s-")
    ax.set_title("NQ Exact Match")
    ax.set_xlabel("K Retrieved Docs")
    ax.set_ylabel("EM")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Center: NQ Retrieval Recall@K
    ax = axes[1]
    plot_lines(ax, collect_metric(results, "nq_retrieval", "rag_token", "recall_at_k"), "RAG-Token", "o-")
    plot_lines(ax, collect_metric(results, "nq_retrieval", "rag_sequence", "recall_at_k"), "RAG-Seq", "s-")
    ax.set_title("NQ Answer Recall @ K")
    ax.set_xlabel("K Retrieved Docs")
    ax.set_ylabel("Recall@K")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Right: MS-MARCO BLEU-1 / ROUGE-L
    ax = axes[2]
    plot_lines(ax, collect_metric(results, "msmarco", "rag_token", "rougeL"), "RAG-Tok R-L", "o-")
    plot_lines(ax, collect_metric(results, "msmarco", "rag_token", "bleu1"), "RAG-Tok B-1", "o--")
    plot_lines(ax, collect_metric(results, "msmarco", "rag_sequence", "rougeL"), "RAG-Seq R-L", "s-")
    plot_lines(ax, collect_metric(results, "msmarco", "rag_sequence", "bleu1"), "RAG-Seq B-1", "s--")
    ax.set_title("MS-MARCO BLEU-1 / ROUGE-L")
    ax.set_xlabel("K Retrieved Docs")
    ax.set_ylabel("Score")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.savefig(args.output, dpi=200)
    print(f"Saved figure to {args.output}")


if __name__ == "__main__":
    main()
