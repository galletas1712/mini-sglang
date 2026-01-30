#!/usr/bin/env python
"""
Benchmark sweep over different max_running_req values.

Runs the Qwen benchmark with max_running_req = 1, 2, 4, 8, 16, 32, 64, 128, 256
and compares results.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

# Add python directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

from minisgl.benchmark.client import (
    benchmark_trace,
    get_model_name,
    process_benchmark_results,
    read_qwen_trace,
    scale_traces,
)
from minisgl.utils import init_logger
from openai import AsyncOpenAI as OpenAI
from transformers import AutoTokenizer

logger = init_logger(__name__)

URL = "https://raw.githubusercontent.com/alibaba-edu/qwen-bailian-usagetraces-anon/refs/heads/main/qwen_traceA_blksz_16.jsonl"


def download_qwen_trace(url: str) -> str:
    dir = Path(__file__).parent / "online"
    file_path = dir / "qwen_trace.jsonl"
    if not file_path.exists():
        import urllib.request
        logger.info(f"Downloading trace from {url} to {file_path}...")
        urllib.request.urlretrieve(url, file_path)
        logger.info("Download completed.")
    return str(file_path)


def start_server(model_path: str, max_running_req: int, port: int = 1919, cuda_device: str | None = None) -> subprocess.Popen:
    """Start the minisgl server with given max_running_req."""
    cmd = [
        sys.executable, "-m", "minisgl.server",
        "--model-path", model_path,
        "--max-running-requests", str(max_running_req),
        "--port", str(port),
        "--cuda-graph-max-bs", str(min(max_running_req, 128)),
    ]
    logger.info(f"Starting server with max_running_req={max_running_req}...")
    logger.info(f"Command: {' '.join(cmd)}")

    env = os.environ.copy()
    if cuda_device is not None:
        env["CUDA_VISIBLE_DEVICES"] = cuda_device
        logger.info(f"Using CUDA_VISIBLE_DEVICES={cuda_device}")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        preexec_fn=os.setsid,
    )
    return proc


def wait_for_server(port: int, timeout: float = 300) -> bool:
    """Wait for server to be ready."""
    import socket
    start = time.time()
    while time.time() - start < timeout:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                logger.info("Server is ready!")
                return True
        except (socket.timeout, ConnectionRefusedError, OSError):
            time.sleep(1)
    return False


def stop_server(proc: subprocess.Popen) -> None:
    """Stop the server process group."""
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=10)
    except Exception as e:
        logger.warning(f"Error stopping server: {e}")
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            pass


async def run_benchmark(port: int, num_requests: int, scale: float, tokenizer_path: str) -> dict:
    """Run the benchmark and return results."""
    async with OpenAI(base_url=f"http://127.0.0.1:{port}/v1", api_key="") as client:
        MODEL = await get_model_name(client)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        traces = read_qwen_trace(download_qwen_trace(URL), tokenizer, n=num_requests, dummy=True)
        traces = scale_traces(traces, scale)
        results = await benchmark_trace(client, traces, MODEL)
        return results


def summarize_results(results: list) -> dict:
    """Compute summary statistics from benchmark results (RawResult objects)."""
    # RawResult has: input_len, output_len, message, tics
    # tics[0] = start time, tics[1] = first token time, tics[-1] = end time
    ttfts = []
    tpots = []
    latencies = []
    total_output_tokens = 0

    start_times = []
    end_times = []

    for r in results:
        if len(r.tics) >= 2:
            ttft = r.tics[1] - r.tics[0]  # time to first token
            ttfts.append(ttft)
            start_times.append(r.tics[0])
            end_times.append(r.tics[-1])
            latency = r.tics[-1] - r.tics[0]
            latencies.append(latency)
            total_output_tokens += r.output_len

            if r.output_len > 1 and len(r.tics) > 2:
                # time per output token (excluding first token)
                tpot = (r.tics[-1] - r.tics[1]) / (r.output_len - 1)
                tpots.append(tpot)

    def percentile(data, p):
        if not data:
            return 0.0
        sorted_data = sorted(data)
        idx = int(len(sorted_data) * p / 100)
        return sorted_data[min(idx, len(sorted_data) - 1)]

    def mean(data):
        return sum(data) / len(data) if data else 0.0

    total_time = max(end_times) - min(start_times) if end_times else 1.0

    return {
        "num_requests": len(results),
        "throughput_req_per_s": len(results) / total_time if total_time > 0 else 0,
        "throughput_tok_per_s": total_output_tokens / total_time if total_time > 0 else 0,
        "ttft_mean": mean(ttfts),
        "ttft_p50": percentile(ttfts, 50),
        "ttft_p99": percentile(ttfts, 99),
        "tpot_mean": mean(tpots),
        "tpot_p50": percentile(tpots, 50),
        "tpot_p99": percentile(tpots, 99),
        "latency_mean": mean(latencies),
        "latency_p50": percentile(latencies, 50),
        "latency_p99": percentile(latencies, 99),
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark sweep over max_running_req values")
    parser.add_argument(
        "--model-path",
        type=str,
        default="Qwen/Qwen3-0.6B",
        help="Model path (default: Qwen/Qwen3-0.6B)",
    )
    parser.add_argument(
        "--num-requests", "-n",
        type=int,
        default=200,
        help="Number of requests per benchmark (default: 200)",
    )
    parser.add_argument(
        "--scale", "-s",
        type=float,
        default=0.1,
        help="Trace time scale (default: 0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=1919,
        help="Server port (default: 1919)",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="benchmark/results_max_running_req",
        help="Output directory for results",
    )
    parser.add_argument(
        "--max-running-req-values",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 16, 32, 64, 128, 256],
        help="max_running_req values to test (default: powers of 2 from 1 to 256)",
    )
    parser.add_argument(
        "--cuda-device",
        type=str,
        default=None,
        help="CUDA device to use (e.g., '0' or '1')",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_summaries = []

    for max_running_req in args.max_running_req_values:
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing max_running_req = {max_running_req}")
        logger.info(f"{'='*60}")

        # Start server
        proc = start_server(args.model_path, max_running_req, args.port, args.cuda_device)

        try:
            # Wait for server to be ready
            if not wait_for_server(args.port):
                logger.error(f"Server failed to start for max_running_req={max_running_req}")
                continue

            # Give it a bit more time to fully initialize
            time.sleep(5)

            # Run benchmark
            results = asyncio.run(run_benchmark(
                args.port, args.num_requests, args.scale, args.model_path
            ))

            # Process results
            summary = summarize_results(results)
            summary["max_running_req"] = max_running_req
            all_summaries.append(summary)

            # Save individual results
            result_dir = output_dir / f"max_running_req_{max_running_req}"
            process_benchmark_results(results, output_dir=result_dir)

            logger.info(f"Results for max_running_req={max_running_req}:")
            logger.info(f"  Throughput: {summary['throughput_req_per_s']:.2f} req/s, {summary['throughput_tok_per_s']:.2f} tok/s")
            logger.info(f"  TTFT: mean={summary['ttft_mean']*1000:.1f}ms, p50={summary['ttft_p50']*1000:.1f}ms, p99={summary['ttft_p99']*1000:.1f}ms")
            logger.info(f"  TPOT: mean={summary['tpot_mean']*1000:.1f}ms, p50={summary['tpot_p50']*1000:.1f}ms, p99={summary['tpot_p99']*1000:.1f}ms")

        finally:
            # Stop server
            logger.info("Stopping server...")
            stop_server(proc)
            time.sleep(2)  # Give time for cleanup

    # Print comparison table
    logger.info(f"\n{'='*80}")
    logger.info("COMPARISON TABLE")
    logger.info(f"{'='*80}")
    logger.info(f"{'max_running_req':>15} {'throughput':>12} {'TTFT_p50':>10} {'TTFT_p99':>10} {'TPOT_p50':>10} {'TPOT_p99':>10}")
    logger.info(f"{'':-^15} {'':-^12} {'':-^10} {'':-^10} {'':-^10} {'':-^10}")

    for s in all_summaries:
        logger.info(
            f"{s['max_running_req']:>15} "
            f"{s['throughput_tok_per_s']:>12.1f} "
            f"{s['ttft_p50']*1000:>10.1f} "
            f"{s['ttft_p99']*1000:>10.1f} "
            f"{s['tpot_p50']*1000:>10.1f} "
            f"{s['tpot_p99']*1000:>10.1f}"
        )

    # Save summary CSV
    import csv
    summary_file = output_dir / "summary.csv"
    if all_summaries:
        with open(summary_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_summaries[0].keys())
            writer.writeheader()
            writer.writerows(all_summaries)
        logger.info(f"\nSummary saved to {summary_file}")


if __name__ == "__main__":
    main()
