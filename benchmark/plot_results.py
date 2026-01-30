#!/usr/bin/env python3
"""
Plotting script for minisgl benchmark results.

Reads CSV files from benchmark output directories and generates comparison plots.

Usage:
    # Single policy results
    python plot_results.py results/fcfs/

    # Compare multiple policies
    python plot_results.py results/fcfs/ results/srpt/ results/sjf/

    # Custom output directory
    python plot_results.py results/fcfs/ results/srpt/ --output plots/
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def read_ttft_csv(path: Path) -> List[float]:
    """Read TTFT values from CSV file."""
    ttfts = []
    with open(path / "ttft.csv", "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ttfts.append(float(row["ttft_ms"]))
    return ttfts


def read_tpot_csv(path: Path) -> Tuple[List[float], Dict[int, List[float]]]:
    """Read TPOT values from CSV file.

    Returns:
        all_tpots: Flat list of all TPOT values
        per_request_tpots: Dict mapping request_id to list of TPOT values
    """
    all_tpots = []
    per_request_tpots: Dict[int, List[float]] = {}
    with open(path / "tpot.csv", "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            req_id = int(row["request_id"])
            tpot = float(row["tpot_ms"])
            all_tpots.append(tpot)
            if req_id not in per_request_tpots:
                per_request_tpots[req_id] = []
            per_request_tpots[req_id].append(tpot)
    return all_tpots, per_request_tpots


def read_e2e_csv(path: Path) -> Tuple[List[float], List[int], List[int]]:
    """Read E2E latency values from CSV file.

    Returns:
        e2e_times: List of E2E latencies in seconds
        input_lens: List of input lengths
        output_lens: List of output lengths
    """
    e2e_times = []
    input_lens = []
    output_lens = []
    with open(path / "e2e.csv", "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            e2e_times.append(float(row["e2e_s"]))
            input_lens.append(int(row["input_len"]))
            output_lens.append(int(row["output_len"]))
    return e2e_times, input_lens, output_lens


def read_summary_csv(path: Path) -> Dict[str, Dict[str, float]]:
    """Read summary statistics from CSV file."""
    summary = {}
    with open(path / "summary.csv", "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            metric = row["metric"]
            summary[metric] = {
                "avg": float(row["avg"]) if row["avg"] else 0,
                "p50": float(row["p50"]) if row["p50"] else 0,
                "p90": float(row["p90"]) if row["p90"] else 0,
                "p99": float(row["p99"]) if row["p99"] else 0,
                "max": float(row["max"]) if row["max"] else 0,
            }
    return summary


def compute_percentiles(data: List[float]) -> Dict[str, float]:
    """Compute percentile statistics for a list of values."""
    if not data:
        return {"avg": 0, "p50": 0, "p90": 0, "p99": 0, "max": 0}
    sorted_data = sorted(data)
    n = len(sorted_data)
    return {
        "avg": sum(data) / n,
        "p50": sorted_data[int(n * 0.5)],
        "p90": sorted_data[int(n * 0.9)],
        "p99": sorted_data[int(n * 0.99)],
        "max": max(data),
    }


def plot_latency_cdf(
    data_dict: Dict[str, List[float]],
    title: str,
    xlabel: str,
    output_path: Path,
) -> None:
    """Plot CDF of latency values for multiple policies."""
    plt.figure(figsize=(10, 6))

    for label, data in data_dict.items():
        sorted_data = np.sort(data)
        cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        plt.plot(sorted_data, cdf, label=label, linewidth=2)

    plt.xlabel(xlabel)
    plt.ylabel("CDF")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_percentile_bars(
    summaries: Dict[str, Dict[str, Dict[str, float]]],
    metric: str,
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    """Plot bar chart comparing percentiles across policies."""
    policies = list(summaries.keys())
    percentiles = ["avg", "p50", "p90", "p99"]

    x = np.arange(len(percentiles))
    width = 0.8 / len(policies)

    plt.figure(figsize=(10, 6))

    for i, policy in enumerate(policies):
        values = [summaries[policy][metric][p] for p in percentiles]
        offset = (i - len(policies) / 2 + 0.5) * width
        plt.bar(x + offset, values, width, label=policy)

    plt.xlabel("Percentile")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(x, ["Avg", "P50", "P90", "P99"])
    plt.legend()
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_throughput_comparison(
    summaries: Dict[str, Dict[str, Dict[str, float]]],
    output_path: Path,
) -> None:
    """Plot throughput comparison across policies."""
    policies = list(summaries.keys())

    token_throughputs = [summaries[p]["throughput_token_s"]["avg"] for p in policies]
    req_throughputs = [summaries[p]["throughput_req_s"]["avg"] for p in policies]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Token throughput
    bars1 = ax1.bar(policies, token_throughputs, color="steelblue")
    ax1.set_ylabel("Tokens/second")
    ax1.set_title("Token Throughput")
    ax1.tick_params(axis="x", rotation=45)
    for bar, val in zip(bars1, token_throughputs):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                 f"{val:.0f}", ha="center", va="bottom")

    # Request throughput
    bars2 = ax2.bar(policies, req_throughputs, color="coral")
    ax2.set_ylabel("Requests/second")
    ax2.set_title("Request Throughput")
    ax2.tick_params(axis="x", rotation=45)
    for bar, val in zip(bars2, req_throughputs):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                 f"{val:.2f}", ha="center", va="bottom")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_latency_by_output_len(
    data_dict: Dict[str, Tuple[List[float], List[int]]],
    title: str,
    output_path: Path,
    num_bins: int = 10,
) -> None:
    """Plot E2E latency vs output length for multiple policies."""
    plt.figure(figsize=(10, 6))

    for label, (e2e_times, output_lens) in data_dict.items():
        # Bin by output length
        min_len, max_len = min(output_lens), max(output_lens)
        bins = np.linspace(min_len, max_len, num_bins + 1)
        bin_centers = (bins[:-1] + bins[1:]) / 2

        binned_latencies = [[] for _ in range(num_bins)]
        for e2e, out_len in zip(e2e_times, output_lens):
            bin_idx = min(int((out_len - min_len) / (max_len - min_len + 1) * num_bins), num_bins - 1)
            binned_latencies[bin_idx].append(e2e)

        avg_latencies = [np.mean(b) if b else 0 for b in binned_latencies]
        plt.plot(bin_centers, avg_latencies, "o-", label=label, linewidth=2, markersize=6)

    plt.xlabel("Output Length (tokens)")
    plt.ylabel("Average E2E Latency (s)")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_fairness_analysis(
    data_dict: Dict[str, Tuple[List[float], List[int]]],
    output_path: Path,
) -> None:
    """Plot fairness analysis: how latency varies with request size."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Short vs Long requests comparison
    for label, (e2e_times, output_lens) in data_dict.items():
        median_len = np.median(output_lens)
        short_latencies = [e for e, o in zip(e2e_times, output_lens) if o <= median_len]
        long_latencies = [e for e, o in zip(e2e_times, output_lens) if o > median_len]

        short_stats = compute_percentiles(short_latencies)
        long_stats = compute_percentiles(long_latencies)

        x = np.array([0, 1])
        width = 0.15
        offset = list(data_dict.keys()).index(label) * width - len(data_dict) * width / 2 + width / 2

        # P90 latency for short vs long
        axes[0].bar(x + offset, [short_stats["p90"], long_stats["p90"]], width, label=label)

    axes[0].set_xticks([0, 1])
    axes[0].set_xticklabels(["Short Requests\n(< median)", "Long Requests\n(> median)"])
    axes[0].set_ylabel("P90 E2E Latency (s)")
    axes[0].set_title("Fairness: P90 Latency by Request Size")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis="y")

    # Latency ratio (normalized by output length)
    for label, (e2e_times, output_lens) in data_dict.items():
        # Latency per output token
        latency_per_token = [e / o if o > 0 else 0 for e, o in zip(e2e_times, output_lens)]
        sorted_data = np.sort(latency_per_token)
        cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        axes[1].plot(sorted_data * 1000, cdf, label=label, linewidth=2)

    axes[1].set_xlabel("Latency per Output Token (ms)")
    axes[1].set_ylabel("CDF")
    axes[1].set_title("Normalized Latency Distribution")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_summary_table(
    summaries: Dict[str, Dict[str, Dict[str, float]]],
    output_path: Path,
) -> None:
    """Create a summary table image."""
    policies = list(summaries.keys())

    # Prepare data
    rows = []
    row_labels = []

    metrics = [
        ("ttft_ms", "TTFT (ms)"),
        ("tpot_ms", "TPOT (ms)"),
        ("e2e_s", "E2E (s)"),
        ("throughput_token_s", "Tokens/s"),
        ("throughput_req_s", "Req/s"),
    ]

    for metric_key, metric_label in metrics:
        row_labels.append(metric_label)
        row = []
        for policy in policies:
            if metric_key in summaries[policy]:
                val = summaries[policy][metric_key]["avg"]
                if "ms" in metric_label:
                    row.append(f"{val:.2f}")
                elif "s" in metric_label and "token" not in metric_key.lower():
                    row.append(f"{val:.3f}")
                else:
                    row.append(f"{val:.1f}")
            else:
                row.append("-")
        rows.append(row)

    # Create table
    fig, ax = plt.subplots(figsize=(max(8, len(policies) * 2), len(metrics) * 0.8 + 1))
    ax.axis("off")

    table = ax.table(
        cellText=rows,
        rowLabels=row_labels,
        colLabels=policies,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    # Style header
    for i in range(len(policies)):
        table[(0, i)].set_facecolor("#4472C4")
        table[(0, i)].set_text_props(color="white", weight="bold")

    plt.title("Benchmark Summary", fontsize=14, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Plot benchmark results from CSV files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "input_dirs",
        nargs="+",
        type=Path,
        help="Input directories containing benchmark CSV files",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Output directory for plots (default: first input directory)",
    )
    parser.add_argument(
        "--labels", "-l",
        nargs="+",
        help="Labels for each input directory (default: directory names)",
    )
    args = parser.parse_args()

    # Determine output directory
    if args.output is None:
        args.output = args.input_dirs[0]
    args.output.mkdir(parents=True, exist_ok=True)

    # Determine labels
    if args.labels:
        labels = args.labels
    else:
        labels = [d.name for d in args.input_dirs]

    if len(labels) != len(args.input_dirs):
        raise ValueError("Number of labels must match number of input directories")

    # Load data from all directories
    all_ttft: Dict[str, List[float]] = {}
    all_tpot: Dict[str, List[float]] = {}
    all_e2e: Dict[str, Tuple[List[float], List[int]]] = {}
    all_summaries: Dict[str, Dict[str, Dict[str, float]]] = {}

    for label, input_dir in zip(labels, args.input_dirs):
        if not input_dir.exists():
            print(f"Warning: Directory {input_dir} does not exist, skipping")
            continue

        try:
            all_ttft[label] = read_ttft_csv(input_dir)
            tpots, _ = read_tpot_csv(input_dir)
            all_tpot[label] = tpots
            e2e_times, _, output_lens = read_e2e_csv(input_dir)
            all_e2e[label] = (e2e_times, output_lens)
            all_summaries[label] = read_summary_csv(input_dir)
            print(f"Loaded data from {input_dir} ({len(all_ttft[label])} requests)")
        except Exception as e:
            print(f"Error loading data from {input_dir}: {e}")
            continue

    if not all_ttft:
        print("No valid data found, exiting")
        return

    print(f"\nGenerating plots in {args.output}...")

    # Generate plots
    # 1. TTFT CDF
    plot_latency_cdf(
        all_ttft,
        "Time To First Token (TTFT) Distribution",
        "TTFT (ms)",
        args.output / "ttft_cdf.png",
    )
    print("  - ttft_cdf.png")

    # 2. TPOT CDF
    plot_latency_cdf(
        all_tpot,
        "Time Per Output Token (TPOT) Distribution",
        "TPOT (ms)",
        args.output / "tpot_cdf.png",
    )
    print("  - tpot_cdf.png")

    # 3. E2E CDF
    e2e_only = {k: v[0] for k, v in all_e2e.items()}
    plot_latency_cdf(
        e2e_only,
        "End-to-End Latency Distribution",
        "E2E Latency (s)",
        args.output / "e2e_cdf.png",
    )
    print("  - e2e_cdf.png")

    # 4. Percentile bar charts
    plot_percentile_bars(
        all_summaries,
        "ttft_ms",
        "TTFT Percentile Comparison",
        "TTFT (ms)",
        args.output / "ttft_percentiles.png",
    )
    print("  - ttft_percentiles.png")

    plot_percentile_bars(
        all_summaries,
        "tpot_ms",
        "TPOT Percentile Comparison",
        "TPOT (ms)",
        args.output / "tpot_percentiles.png",
    )
    print("  - tpot_percentiles.png")

    plot_percentile_bars(
        all_summaries,
        "e2e_s",
        "E2E Latency Percentile Comparison",
        "E2E Latency (s)",
        args.output / "e2e_percentiles.png",
    )
    print("  - e2e_percentiles.png")

    # 5. Throughput comparison
    plot_throughput_comparison(all_summaries, args.output / "throughput.png")
    print("  - throughput.png")

    # 6. Latency by output length
    plot_latency_by_output_len(
        all_e2e,
        "E2E Latency vs Output Length",
        args.output / "latency_by_output_len.png",
    )
    print("  - latency_by_output_len.png")

    # 7. Fairness analysis
    plot_fairness_analysis(all_e2e, args.output / "fairness.png")
    print("  - fairness.png")

    # 8. Summary table
    plot_summary_table(all_summaries, args.output / "summary_table.png")
    print("  - summary_table.png")

    print(f"\nDone! All plots saved to {args.output}")


if __name__ == "__main__":
    main()
