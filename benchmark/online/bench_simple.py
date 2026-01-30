import argparse
import asyncio
import random
import sys
from pathlib import Path
from typing import List

from minisgl.benchmark.client import (
    benchmark_one,
    benchmark_one_batch,
    generate_prompt,
    get_model_name,
    process_benchmark_results,
)
from minisgl.utils import init_logger
from openai import AsyncOpenAI as OpenAI
from transformers import AutoTokenizer

logger = init_logger(__name__)


async def main(args: argparse.Namespace):
    try:
        random.seed(42)  # reproducibility

        async def generate_task(max_bs: int) -> List[str]:
            """Generate a list of tasks with random lengths."""
            result = []
            for _ in range(max_bs):
                length = random.randint(1, args.max_input)
                message = generate_prompt(tokenizer, length)
                result.append(message)
                await asyncio.sleep(0)
            return result

        # Create the async client
        async with OpenAI(base_url=f"http://127.0.0.1:{args.port}/v1", api_key="") as client:
            MODEL = await get_model_name(client)
            tokenizer = AutoTokenizer.from_pretrained(MODEL)

            logger.info(f"Loaded tokenizer from {MODEL}")
            logger.info("Testing connection to server...")

            # Test connection with a simple request first
            try:
                gen_task = asyncio.create_task(generate_task(args.batch_size))
                test_msg = generate_prompt(tokenizer, 100)
                test_result = await benchmark_one(client, test_msg, 2, MODEL, pbar=False)
                if len(test_result.tics) <= 2:
                    logger.info("Server connection test failed")
                    return
                logger.info("Server connection successful")
            except Exception as e:
                logger.warning("Server connection failed")
                logger.warning(f"Make sure the server is running on http://127.0.0.1:{args.port}")
                raise e from e

            msgs = await gen_task
            output_lengths = [random.randint(args.min_output, args.max_output) for _ in range(args.batch_size)]
            logger.info(f"Generated {len(msgs)} test messages")

            logger.info("Running benchmark...")
            try:
                results = await benchmark_one_batch(
                    client, msgs, output_lengths, MODEL
                )
                output_dir = Path(args.output_dir) if args.output_dir else None
                process_benchmark_results(results, output_dir=output_dir)
            except Exception as e:
                logger.info(f"Error with batch size {args.batch_size}: {e}")
            logger.info("Benchmark completed.")

    except Exception as e:
        print(f"Error in main: {e}")
        sys.exit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple batch benchmark")
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=1919,
        help="Server port (default: 1919)",
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=64,
        help="Batch size (default: 64)",
    )
    parser.add_argument(
        "--max-input",
        type=int,
        default=8192,
        help="Maximum input length in tokens (default: 8192)",
    )
    parser.add_argument(
        "--min-output",
        type=int,
        default=16,
        help="Minimum output length in tokens (default: 16)",
    )
    parser.add_argument(
        "--max-output",
        type=int,
        default=1024,
        help="Maximum output length in tokens (default: 1024)",
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
