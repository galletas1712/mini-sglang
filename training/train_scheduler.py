#!/usr/bin/env python3
"""
Training script for the Mini-SGLang RL scheduler policy.

Learner process that:
1. Consumes trajectory files from the actor (Mini-SGLang with RLScheduler)
2. Computes PPO/BC updates
3. Saves checkpoints for hot-reload

Usage:
    python training/train_scheduler.py \
        --trajectory-dir /tmp/trajectories \
        --checkpoint-dir checkpoints/ \
        --checkpoint-name policy_latest.pt
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import torch
import torch.optim as optim

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

from minisgl.scheduler.rl_policy import PolicyConfig, SchedulerPolicy
from minisgl.scheduler.rl_trajectory import iter_trajectory_files, TrajectoryBatch
from ppo_utils import (
    PPOConfig,
    ProcessedBatch,
    process_trajectory_batch,
    compute_ppo_loss,
    compute_bc_loss,
    compute_combined_loss,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train_scheduler")


class TrainingPhase:
    """Training phase enum."""
    BC_ONLY = "bc"
    BC_PPO = "bc+ppo"
    PPO_ONLY = "ppo"


def save_checkpoint(
    policy: SchedulerPolicy,
    optimizer: optim.Optimizer,
    step: int,
    metrics: dict,
    checkpoint_dir: Path,
    checkpoint_name: str,
    schema_version: str = "1.0",
) -> Path:
    """Save policy checkpoint with metadata."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / checkpoint_name

    checkpoint = {
        "model_state_dict": policy.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "metrics": metrics,
        "schema_version": schema_version,
        "policy_config": {
            "d_global": policy.config.d_global,
            "d_request": policy.config.d_request,
            "d_hidden": policy.config.d_hidden,
            "max_k": policy.config.max_k,
        },
        "feature_config": {
            "d_global": 6,
            "d_request": 10,
            "global_feature_names": [
                "kv_cache_usage", "num_pending_norm", "num_running_norm",
                "available_slots_norm", "prefill_budget_norm", "decode_inflight_norm",
            ],
            "request_feature_names": [
                "is_pending", "is_chunked", "input_len_norm", "cached_len_norm",
                "extend_len_norm", "output_len_norm", "remain_len_norm",
                "cache_hit_ratio", "age_log1p", "progress",
            ],
        },
    }

    # Write to temp file then rename (atomic)
    tmp_path = path.with_suffix(".tmp")
    torch.save(checkpoint, tmp_path)
    tmp_path.rename(path)

    logger.info(f"Saved checkpoint to {path} (step={step})")
    return path


def load_checkpoint(
    path: Path,
    policy: SchedulerPolicy,
    optimizer: optim.Optimizer | None = None,
) -> int:
    """Load checkpoint. Returns the step number."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    if "model_state_dict" in checkpoint:
        policy.load_state_dict(checkpoint["model_state_dict"])
        if optimizer and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return checkpoint.get("step", 0)
    else:
        # Raw state dict
        policy.load_state_dict(checkpoint)
        return 0


def collect_trajectories(
    trajectory_dir: Path,
    processed_files: set[str],
    min_steps: int = 100,
    max_steps: int = 10000,
) -> tuple[list, set[str]]:
    """
    Collect unprocessed trajectory steps.

    Returns (steps, updated_processed_files).
    """
    steps = []
    new_processed = set()

    for batch in iter_trajectory_files(trajectory_dir):
        batch_key = str(batch.metadata.get("flush_time", time.time()))
        if batch_key in processed_files:
            continue

        steps.extend(batch.steps)
        new_processed.add(batch_key)

        if len(steps) >= max_steps:
            break

    if len(steps) < min_steps:
        return [], processed_files

    return steps, processed_files | new_processed


def train_step(
    policy: SchedulerPolicy,
    optimizer: optim.Optimizer,
    steps: list,
    phase: str,
    ppo_config: PPOConfig,
    bc_weight: float,
    device: torch.device,
) -> dict:
    """Run a single training step."""
    # Process trajectory into batched format
    batch = process_trajectory_batch(
        steps,
        gamma=ppo_config.gamma,
        lam=ppo_config.lam,
        normalize_advantages=True,
    )

    # Compute loss based on phase
    if phase == TrainingPhase.BC_ONLY:
        loss, metrics = compute_bc_loss(policy, batch, device)
    elif phase == TrainingPhase.PPO_ONLY:
        loss, metrics = compute_ppo_loss(policy, batch, ppo_config, device)
    else:  # BC + PPO
        loss, metrics = compute_combined_loss(
            policy, batch, ppo_config, bc_weight, device
        )

    # Backward pass
    optimizer.zero_grad()
    loss.backward()

    # Gradient clipping
    grad_norm = torch.nn.utils.clip_grad_norm_(
        policy.parameters(), ppo_config.max_grad_norm
    )
    metrics["grad_norm"] = grad_norm.item()

    optimizer.step()

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train Mini-SGLang RL scheduler policy")
    parser.add_argument(
        "--trajectory-dir", type=str, required=True,
        help="Directory containing trajectory files"
    )
    parser.add_argument(
        "--checkpoint-dir", type=str, default="checkpoints",
        help="Directory for saving checkpoints"
    )
    parser.add_argument(
        "--checkpoint-name", type=str, default="policy_latest.pt",
        help="Name of checkpoint file (for hot-reload)"
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Device to train on (cpu or cuda)"
    )
    parser.add_argument(
        "--lr", type=float, default=3e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--training-phases", type=str, default="bc,bc+ppo,ppo",
        help="Comma-separated training phases"
    )
    parser.add_argument(
        "--phase-steps", type=str, default="1000,2000,inf",
        help="Steps per phase (comma-separated, 'inf' for no limit)"
    )
    parser.add_argument(
        "--bc-weight", type=float, default=0.5,
        help="BC weight for bc+ppo phase"
    )
    parser.add_argument(
        "--batch-min-steps", type=int, default=100,
        help="Minimum trajectory steps per training batch"
    )
    parser.add_argument(
        "--batch-max-steps", type=int, default=2000,
        help="Maximum trajectory steps per training batch"
    )
    parser.add_argument(
        "--checkpoint-interval", type=int, default=100,
        help="Steps between checkpoints"
    )
    parser.add_argument(
        "--log-interval", type=int, default=10,
        help="Steps between logging"
    )
    parser.add_argument(
        "--poll-interval", type=float, default=5.0,
        help="Seconds to wait when no new trajectories"
    )
    parser.add_argument(
        "--max-k", type=int, default=32,
        help="Maximum batch size (policy output dimension)"
    )
    args = parser.parse_args()

    # Setup
    device = torch.device(args.device)
    trajectory_dir = Path(args.trajectory_dir)
    checkpoint_dir = Path(args.checkpoint_dir)

    # Parse training phases
    phases = args.training_phases.split(",")
    phase_steps = []
    for s in args.phase_steps.split(","):
        phase_steps.append(float("inf") if s == "inf" else int(s))

    logger.info(f"Training phases: {list(zip(phases, phase_steps))}")

    # Initialize policy (G=6 global features, F=10 request features)
    # max_k should match the candidate_k used during trajectory collection
    config = PolicyConfig(d_global=6, d_request=10, d_hidden=128, max_k=args.max_k)
    policy = SchedulerPolicy(config).to(device)

    # Initialize optimizer
    optimizer = optim.Adam(policy.parameters(), lr=args.lr)

    # PPO config
    ppo_config = PPOConfig()

    # Resume from checkpoint if specified
    step = 0
    if args.resume:
        step = load_checkpoint(Path(args.resume), policy, optimizer)
        logger.info(f"Resumed from {args.resume} at step {step}")

    # Track processed trajectory files
    processed_files: set[str] = set()

    # Determine starting phase
    current_phase_idx = 0
    steps_in_phase = 0
    cumulative_steps = 0
    for i, max_steps in enumerate(phase_steps):
        cumulative_steps += max_steps
        if step < cumulative_steps:
            current_phase_idx = i
            steps_in_phase = step - (cumulative_steps - max_steps)
            break
    else:
        current_phase_idx = len(phases) - 1
        steps_in_phase = 0

    logger.info(
        f"Starting training: phase={phases[current_phase_idx]}, "
        f"step={step}, device={device}"
    )

    # Training loop
    try:
        while True:
            current_phase = phases[current_phase_idx]
            max_phase_steps = phase_steps[current_phase_idx]

            # Collect trajectories
            steps_data, processed_files = collect_trajectories(
                trajectory_dir,
                processed_files,
                min_steps=args.batch_min_steps,
                max_steps=args.batch_max_steps,
            )

            if not steps_data:
                logger.debug(
                    f"No new trajectories, waiting {args.poll_interval}s..."
                )
                time.sleep(args.poll_interval)
                continue

            # Train step
            policy.train()
            metrics = train_step(
                policy,
                optimizer,
                steps_data,
                current_phase,
                ppo_config,
                args.bc_weight,
                device,
            )

            step += 1
            steps_in_phase += 1

            # Logging
            if step % args.log_interval == 0:
                metrics_str = " ".join(
                    f"{k}={v:.4f}" for k, v in metrics.items()
                )
                logger.info(
                    f"[{current_phase}] step={step} batch_size={len(steps_data)} "
                    f"{metrics_str}"
                )

            # Checkpointing
            if step % args.checkpoint_interval == 0:
                save_checkpoint(
                    policy, optimizer, step, metrics,
                    checkpoint_dir, args.checkpoint_name,
                )

            # Phase transition
            if steps_in_phase >= max_phase_steps:
                current_phase_idx += 1
                steps_in_phase = 0

                if current_phase_idx >= len(phases):
                    logger.info("All training phases complete!")
                    break

                logger.info(
                    f"Transitioning to phase: {phases[current_phase_idx]}"
                )

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")

    # Final checkpoint
    save_checkpoint(
        policy, optimizer, step, {},
        checkpoint_dir, "policy_final.pt",
    )
    logger.info(f"Training complete. Final step: {step}")


if __name__ == "__main__":
    main()
