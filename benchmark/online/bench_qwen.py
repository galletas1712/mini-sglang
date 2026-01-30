from __future__ import annotations

import argparse
import asyncio
import os
import random
from pathlib import Path

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
    dir = Path(os.path.dirname(__file__))
    # download the file if not exists
    file_path = dir / "qwen_trace.jsonl"
    if not file_path.exists():
        import urllib.request

        logger.info(f"Downloading trace from {url} to {file_path}...")
        urllib.request.urlretrieve(url, file_path)
        logger.info("Download completed.")
    return str(file_path)


async def main(args: argparse.Namespace):
    random.seed(42)  # reproducibility
    async with OpenAI(base_url=f"http://127.0.0.1:{args.port}/v1", api_key="") as client:
        MODEL = await get_model_name(client)
        tokenizer = AutoTokenizer.from_pretrained(MODEL)
        TRACES = read_qwen_trace(download_qwen_trace(URL), tokenizer, n=args.num_requests, dummy=True)
        logger.info(f"Start benchmarking with {args.num_requests} requests using model {MODEL}...")
        for scale in args.scales:
            traces = scale_traces(TRACES, scale)
            results = await benchmark_trace(client, traces, MODEL)

            # Determine output directory for this scale
            output_dir = None
            if args.output_dir:
                output_dir = Path(args.output_dir) / f"scale_{scale}"

            process_benchmark_results(results, output_dir=output_dir)
        logger.info("Benchmarking completed.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark with Qwen trace")
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=1919,
        help="Server port (default: 1919)",
    )
    parser.add_argument(
        "--num-requests", "-n",
        type=int,
        default=1000,
        help="Number of requests to benchmark (default: 1000)",
    )
    parser.add_argument(
        "--scales", "-s",
        type=float,
        nargs="+",
        default=[0.1],
        help="Trace time scales to test (default: 0.1)",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=None,
        help="Directory to save benchmark results as CSV files",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args))
