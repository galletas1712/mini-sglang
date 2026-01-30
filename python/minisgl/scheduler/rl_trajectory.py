"""
Trajectory collection and storage for RL scheduler training.

Handles buffering, serialization, and file management for trajectory data.
"""
from __future__ import annotations

import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import torch


@dataclass
class TrajectoryStep:
    """Single step of trajectory data."""
    global_features: torch.Tensor       # (G,)
    request_features: torch.Tensor      # (K, F)
    valid_mask: torch.Tensor            # (K,)
    num_pending: int
    num_running: int
    action: int                         # Selected batch size N
    log_prob: float                     # Log probability of action
    value: float                        # Value estimate
    reward: float = 0.0                 # Reward (set later)
    done: bool = False                  # Episode boundary
    timestamp: float = 0.0              # Step timestamp
    teacher_action: int | None = None   # Teacher batch size (greedy FCFS)
    actual_scheduled: int | None = None # Actual number scheduled (for over-selection penalty)


@dataclass
class TrajectoryBatch:
    """Serializable batch of trajectory steps."""
    steps: list[TrajectoryStep]
    metadata: dict = field(default_factory=dict)

    def save(self, directory: Path) -> Path:
        """Save batch to a pickle file."""
        directory.mkdir(parents=True, exist_ok=True)
        filename = f"traj_{int(time.time() * 1000)}.pkl"
        path = directory / filename
        # Write to temp file then rename (atomic)
        tmp_path = path.with_suffix(".tmp")
        with open(tmp_path, 'wb') as f:
            pickle.dump(self, f)
        tmp_path.rename(path)
        return path

    @classmethod
    def load(cls, path: Path) -> "TrajectoryBatch":
        """Load batch from pickle file."""
        with open(path, 'rb') as f:
            return pickle.load(f)


class TrajectoryBuffer:
    """
    Buffer for collecting and managing trajectory data.

    Automatically flushes to disk when buffer is full.
    """

    def __init__(
        self,
        directory: Path | str,
        buffer_size: int = 1000,
        flush_on_done: bool = True,
    ):
        self.directory = Path(directory)
        self.buffer_size = buffer_size
        self.flush_on_done = flush_on_done
        self._steps: list[TrajectoryStep] = []
        self._metadata: dict = {"start_time": time.time()}

    def add(self, step: TrajectoryStep) -> None:
        """Add a step to the buffer."""
        self._steps.append(step)

        # Flush if buffer is full or episode ended
        if len(self._steps) >= self.buffer_size:
            self.flush()
        elif self.flush_on_done and step.done:
            self.flush()

    def flush(self) -> Path | None:
        """Flush buffer to disk."""
        if not self._steps:
            return None

        self._metadata["flush_time"] = time.time()
        self._metadata["num_steps"] = len(self._steps)

        batch = TrajectoryBatch(
            steps=self._steps,
            metadata=self._metadata.copy(),
        )
        path = batch.save(self.directory)

        # Reset buffer
        self._steps = []
        self._metadata = {"start_time": time.time()}

        return path

    def __len__(self) -> int:
        return len(self._steps)


def iter_trajectory_files(
    directory: Path | str,
    delete_after_read: bool = False,
) -> Iterator[TrajectoryBatch]:
    """
    Iterate over trajectory files in a directory.

    Args:
        directory: Directory containing trajectory files
        delete_after_read: If True, delete files after reading

    Yields:
        TrajectoryBatch objects
    """
    directory = Path(directory)
    if not directory.exists():
        return

    # Sort by modification time (oldest first)
    files = sorted(
        directory.glob("traj_*.pkl"),
        key=lambda p: p.stat().st_mtime,
    )

    for path in files:
        try:
            batch = TrajectoryBatch.load(path)
            yield batch
            if delete_after_read:
                path.unlink()
        except Exception:
            # Skip corrupted files
            continue


@dataclass
class RewardConfig:
    """Configuration for reward computation."""
    throughput_weight: float = 0.1       # Per-completion bonus
    cache_hit_weight: float = 0.05       # Prefix cache utilization
    starvation_weight: float = 0.01      # Old request penalty
    age_threshold: float = 30.0          # Seconds before starvation penalty
    over_selection_weight: float = 0.1   # Penalty for selecting N > actual scheduled


def compute_step_reward(
    n_in_system: int,
    dt: float,
    newly_finished: int = 0,
    cache_hit_ratio: float = 0.0,
    max_request_age: float = 0.0,
    config: RewardConfig | None = None,
) -> float:
    """
    Compute per-step reward.

    Primary signal is Little's Law: minimize requests in system over time.

    Args:
        n_in_system: Number of requests in system (pending + running)
        dt: Time delta since last step
        newly_finished: Number of requests that finished this step
        cache_hit_ratio: Average cache hit ratio for scheduled requests
        max_request_age: Age of oldest request in seconds
        config: Reward configuration

    Returns:
        Reward value
    """
    config = config or RewardConfig()

    # Core reward: area under queue length (negative because we minimize)
    sojourn_penalty = -n_in_system * dt

    # Completion bonus
    completion_bonus = config.throughput_weight * newly_finished

    # Cache efficiency bonus
    cache_bonus = config.cache_hit_weight * cache_hit_ratio

    # Starvation penalty for old requests
    if max_request_age > config.age_threshold:
        starvation_penalty = -config.starvation_weight * (
            max_request_age - config.age_threshold
        ) ** 2
    else:
        starvation_penalty = 0.0

    return sojourn_penalty + completion_bonus + cache_bonus + starvation_penalty
