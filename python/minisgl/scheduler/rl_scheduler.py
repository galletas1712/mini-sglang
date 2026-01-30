"""
RL-based scheduler for Mini-SGLang.

Overrides the base Scheduler to inject learned batch size selection.
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from minisgl.utils import init_logger

from .config import SchedulerConfig
from .rl_action import BatchSizeDistribution
from .rl_observation import ObservationBuilder, RLObservation
from .rl_policy import PolicyConfig, PolicyOutput, SchedulerPolicy
from .rl_trajectory import (
    RewardConfig,
    TrajectoryBuffer,
    TrajectoryStep,
    compute_step_reward,
)
from .scheduler import ForwardInput, Scheduler

if TYPE_CHECKING:
    pass


logger = init_logger(__name__)


class RLScheduler(Scheduler):
    """
    Deep RL-based scheduler for Mini-SGLang.

    Key design decisions:
    1. Overrides `_schedule_next_batch()` to inject RL policy
    2. Policy selects batch size N (how many requests to process)
    3. Requests are always processed in FCFS order
    4. Falls back to greedy (max N) on policy errors
    """

    def __init__(
        self,
        config: SchedulerConfig,
        rl_config: dict[str, Any] | None = None,
    ):
        super().__init__(config)

        rl_config = rl_config or {}

        # RL configuration (also read from environment)
        self.rl_policy_path = rl_config.get(
            'policy_path',
            os.environ.get('RL_SCHEDULER_POLICY_PATH'),
        )
        self.rl_use_greedy = rl_config.get(
            'use_greedy',
            os.environ.get('RL_SCHEDULER_USE_GREEDY', 'true').lower() == 'true',
        )
        self.rl_fallback_on_error = rl_config.get(
            'fallback_on_error',
            os.environ.get('RL_SCHEDULER_FALLBACK', 'true').lower() == 'true',
        )
        self.rl_candidate_k = rl_config.get(
            'candidate_k',
            int(os.environ.get('RL_SCHEDULER_CANDIDATE_K', str(config.max_running_req))),
        )
        self.rl_training_mode = rl_config.get(
            'training_mode',
            os.environ.get('RL_SCHEDULER_TRAINING_MODE', 'inference'),
        )
        self.rl_trajectory_dir = rl_config.get(
            'trajectory_dir',
            os.environ.get('RL_SCHEDULER_TRAJECTORY_DIR'),
        )

        # Create observation builder
        self._obs_builder = ObservationBuilder(
            max_running_req=config.max_running_req,
            max_seq_len=self.engine.max_seq_len,
            num_pages=self.engine.num_pages,
            max_prefill_budget=config.max_extend_tokens,
            candidate_k=self.rl_candidate_k,
        )

        # Load policy (only on rank 0 since only rank 0 runs inference)
        self.rl_policy: SchedulerPolicy | None = None
        self._policy_checkpoint_mtime: float = 0.0
        if self.rl_policy_path and self.tp_info.is_primary():
            try:
                self.rl_policy = self._load_policy(self.rl_policy_path)
                if self.rl_policy:
                    self.rl_policy.eval()
                    logger.info(f"RLScheduler: Loaded policy from {self.rl_policy_path}")
            except Exception as e:
                logger.warning(f"RLScheduler: Failed to load policy: {e}")

        # Metrics
        self._rl_inference_count = 0
        self._rl_fallback_count = 0
        self._rl_overhead_ms = 0.0
        self._rl_last_metrics_log = time.time()

        # Trajectory collection (for training)
        self._trajectory_buffer: TrajectoryBuffer | None = None
        self._prev_step_data: dict | None = None
        self._reward_config = RewardConfig()

        if self.rl_trajectory_dir and self.rl_training_mode != 'inference':
            self._trajectory_buffer = TrajectoryBuffer(
                directory=Path(self.rl_trajectory_dir),
                buffer_size=1000,
            )
            logger.info(f"RLScheduler: Trajectory collection enabled -> {self.rl_trajectory_dir}")

    def _load_policy(self, path: str | None) -> SchedulerPolicy | None:
        """Load policy from checkpoint."""
        if path is None:
            return None

        path_obj = Path(path)
        if not path_obj.exists():
            logger.warning(f"Policy checkpoint not found: {path}")
            return None

        # Track modification time for hot-reload
        self._policy_checkpoint_mtime = path_obj.stat().st_mtime

        config = PolicyConfig(d_global=6, d_request=10, d_hidden=128, max_k=self.rl_candidate_k)
        policy = SchedulerPolicy(config)

        checkpoint = torch.load(path, map_location="cpu", weights_only=False)

        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            policy.load_state_dict(checkpoint['model_state_dict'])
        else:
            policy.load_state_dict(checkpoint)

        return policy

    def _maybe_reload_policy(self) -> None:
        """Hot-reload policy if checkpoint has been updated (only on rank 0)."""
        if not self.tp_info.is_primary():
            return

        if self.rl_policy_path is None:
            return

        path = Path(self.rl_policy_path)
        if not path.exists():
            return

        current_mtime = path.stat().st_mtime
        if current_mtime > self._policy_checkpoint_mtime:
            try:
                new_policy = self._load_policy(self.rl_policy_path)
                if new_policy:
                    self.rl_policy = new_policy
                    self.rl_policy.eval()
                    logger.info(f"RLScheduler: Hot-reloaded policy from {self.rl_policy_path}")
            except Exception as e:
                logger.warning(f"RLScheduler: Failed to hot-reload policy: {e}")

    def _schedule_next_batch(self, curr_step_max_running_req: int | None = None) -> ForwardInput | None:
        """
        Override parent's scheduling method with RL policy.

        Strategy:
        1. Build observation
        2. Query policy for batch size N
        3. Pass curr_step_max_running_req=N to parent's scheduling
        """
        # Check for hot-reload periodically
        if self._rl_inference_count % 100 == 0:
            self._maybe_reload_policy()

        # For TP > 1, rank 0 decides whether to use policy and broadcasts
        # This prevents deadlocks when policy loading differs across ranks
        use_policy = self.rl_policy is not None
        if self.tp_info.size > 1:
            use_policy_tensor = torch.tensor([1 if use_policy else 0], dtype=torch.int64)
            torch.distributed.broadcast(
                use_policy_tensor, src=0, group=self.engine.tp_cpu_group
            )
            use_policy = use_policy_tensor[0].item() == 1

        if not use_policy:
            return super()._schedule_next_batch(curr_step_max_running_req)

        if not self.rl_fallback_on_error:
            return self._schedule_with_policy()

        try:
            return self._schedule_with_policy()
        except Exception as e:
            self._rl_fallback_count += 1
            logger.warning(f"RLScheduler: Policy error ({e}). Falling back to greedy.")
            return super()._schedule_next_batch(curr_step_max_running_req)

    def _schedule_with_policy(self) -> ForwardInput | None:
        """Schedule using the learned policy.

        Only rank 0 runs policy inference to avoid redundant computation.
        The batch size is broadcast to other ranks for consistency.

        All ranks build observations to ensure they follow the same code path
        and make consistent scheduling decisions.
        """
        t0 = time.perf_counter()

        # All ranks build observation to ensure consistent code paths
        obs = self._obs_builder.build(self)
        batch_size: int = 0
        log_prob: torch.Tensor | None = None
        value: torch.Tensor | None = None

        # Skip if no pending requests
        if obs.num_pending == 0:
            return super()._schedule_next_batch()

        # Only rank 0 runs policy inference
        if self.tp_info.is_primary():
            # Get policy output
            with torch.no_grad():
                output = self.rl_policy(
                    obs.global_features,
                    obs.request_features,
                    obs.valid_mask,
                )

            # Check for NaN/Inf
            if torch.isnan(output.batch_size_logits).any() or torch.isinf(output.batch_size_logits).any():
                raise ValueError("Policy produced NaN/Inf logits")

            # Create batch size distribution
            # Cap by: candidate_k, available table slots, and max_running_req
            available_slots = self.table_manager.available_size
            effective_max = min(
                obs.num_pending,
                self.rl_candidate_k,
                available_slots,
            )
            dist = BatchSizeDistribution(
                output.batch_size_logits,
                num_pending=effective_max,
            )

            if self.rl_use_greedy or self.rl_training_mode == 'inference':
                batch_size = dist.greedy()
                log_prob = None
            else:
                batch_size, log_prob = dist.sample()
                value = output.value

        # Sync batch size across TP ranks (rank 0 broadcasts, others receive)
        if self.tp_info.size > 1:
            batch_size = self._sync_batch_size_across_ranks(batch_size)

        # Track RL overhead
        t1 = time.perf_counter()
        self._rl_inference_count += 1
        self._rl_overhead_ms += (t1 - t0) * 1000

        # Collect trajectory data (for training) - only on rank 0
        if self.tp_info.is_primary() and self._trajectory_buffer is not None and log_prob is not None:
            self._collect_trajectory_step(obs, batch_size, log_prob, value)

        # Log metrics periodically
        if time.time() - self._rl_last_metrics_log > 60:
            self._log_metrics()

        # Schedule with the policy-selected batch size
        result = super()._schedule_next_batch(curr_step_max_running_req=batch_size)

        # Record actual number scheduled for over-selection penalty
        if self.tp_info.is_primary() and self._prev_step_data is not None:
            if result is not None:
                # Count prefill requests (not decode)
                actual_scheduled = sum(
                    1 for r in result.batch.reqs
                    if hasattr(r, 'cached_len')  # Prefill requests have cached_len
                )
            else:
                actual_scheduled = 0
            self._prev_step_data['actual_scheduled'] = actual_scheduled

        return result

    def _sync_batch_size_across_ranks(self, batch_size: int) -> int:
        """
        Broadcast batch size from rank 0 to ensure consistency across TP ranks.

        This is critical because each rank builds observations independently,
        and timing differences can cause different policy outputs. Without
        synchronization, ranks would have different batch sizes, causing
        mismatched batches in tensor-parallel forward passes.
        """
        batch_size_tensor = torch.tensor([batch_size], dtype=torch.int64)

        # Broadcast from rank 0
        torch.distributed.broadcast(
            batch_size_tensor,
            src=0,
            group=self.engine.tp_cpu_group,
        )

        return int(batch_size_tensor[0].item())

    def _collect_trajectory_step(
        self,
        obs: RLObservation,
        batch_size: int,
        log_prob: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        """Collect trajectory step for training."""
        now = time.time()

        # Compute reward for previous step (delayed by one step)
        if self._prev_step_data is not None:
            reward = self._compute_reward(self._prev_step_data, obs)
            step = TrajectoryStep(
                global_features=self._prev_step_data['global_features'],
                request_features=self._prev_step_data['request_features'],
                valid_mask=self._prev_step_data['valid_mask'],
                num_pending=self._prev_step_data['num_pending'],
                num_running=self._prev_step_data['num_running'],
                action=self._prev_step_data['action'],
                log_prob=self._prev_step_data['log_prob'],
                value=self._prev_step_data['value'],
                reward=reward,
                done=False,
                timestamp=self._prev_step_data['timestamp'],
                teacher_action=self._prev_step_data.get('teacher_action'),
                actual_scheduled=self._prev_step_data.get('actual_scheduled'),
            )
            self._trajectory_buffer.add(step)

        # Store current step (reward filled next iteration)
        # Teacher action for batch size is: take as many as possible (greedy FCFS)
        # Capped by available slots to ensure curr_step_max_running_req <= max_running_req
        available_slots = self.table_manager.available_size
        effective_max = min(obs.num_pending, self.rl_candidate_k, available_slots)
        self._prev_step_data = {
            'global_features': obs.global_features.clone(),
            'request_features': obs.request_features.clone(),
            'valid_mask': obs.valid_mask.clone(),
            'num_pending': obs.num_pending,
            'num_running': obs.num_running,
            'action': batch_size,
            'log_prob': log_prob.item(),
            'value': value.item(),
            'timestamp': now,
            # Teacher action is greedy: take all available (capped by slots)
            'teacher_action': effective_max,
            # actual_scheduled will be set after we see the result
            'actual_scheduled': None,
        }

    def _compute_reward(self, prev_step: dict, current_obs: RLObservation) -> float:
        """Compute reward for the previous step."""
        n_prev = prev_step['num_pending'] + prev_step['num_running']
        n_curr = current_obs.num_pending + current_obs.num_running

        dt = time.time() - prev_step['timestamp']

        # Finished requests
        finished = max(0, n_prev - n_curr)

        # Penalty for over-selection: if policy chose N but only M < N could be scheduled
        # This enforces curr_step_max_running_req <= actually achievable
        over_selection_penalty = 0.0
        action = prev_step['action']
        actual = prev_step.get('actual_scheduled')
        if actual is not None and action > actual:
            # Penalty proportional to the gap (squared for stronger signal)
            gap = action - actual
            over_selection_penalty = -self._reward_config.over_selection_weight * (gap ** 2)

        base_reward = compute_step_reward(
            n_in_system=n_prev,
            dt=dt,
            newly_finished=finished,
            config=self._reward_config,
        )

        return base_reward + over_selection_penalty

    def _log_metrics(self) -> None:
        """Log RL scheduler metrics."""
        count = max(self._rl_inference_count, 1)
        avg_overhead = self._rl_overhead_ms / count

        logger.info(
            f"RLScheduler metrics: "
            f"inferences={self._rl_inference_count}, "
            f"fallbacks={self._rl_fallback_count}, "
            f"avg_overhead_ms={avg_overhead:.2f}"
        )

        self._rl_last_metrics_log = time.time()

    def get_rl_metrics(self) -> dict[str, float]:
        """Return RL scheduler metrics."""
        count = max(self._rl_inference_count, 1)
        return {
            "rl_inference_count": self._rl_inference_count,
            "rl_fallback_count": self._rl_fallback_count,
            "rl_avg_overhead_ms": self._rl_overhead_ms / count,
            "rl_trajectory_buffer_size": len(self._trajectory_buffer) if self._trajectory_buffer else 0,
        }

    def shutdown(self) -> None:
        """Shutdown scheduler and flush trajectory buffer."""
        # Add final trajectory step with terminal reward before flushing
        if self._trajectory_buffer is not None and self._prev_step_data is not None:
            # Create terminal step with zero reward (no next state to compute reward from)
            step = TrajectoryStep(
                global_features=self._prev_step_data['global_features'],
                request_features=self._prev_step_data['request_features'],
                valid_mask=self._prev_step_data['valid_mask'],
                num_pending=self._prev_step_data['num_pending'],
                num_running=self._prev_step_data['num_running'],
                action=self._prev_step_data['action'],
                log_prob=self._prev_step_data['log_prob'],
                value=self._prev_step_data['value'],
                reward=0.0,  # Terminal step, no reward signal
                done=True,  # Mark as episode boundary
                timestamp=self._prev_step_data['timestamp'],
                teacher_action=self._prev_step_data.get('teacher_action'),
                actual_scheduled=self._prev_step_data.get('actual_scheduled'),
            )
            self._trajectory_buffer.add(step)
            self._prev_step_data = None

        # Flush remaining trajectory data
        if self._trajectory_buffer is not None:
            self._trajectory_buffer.flush()
            logger.info("RLScheduler: Flushed trajectory buffer on shutdown")

        super().shutdown()
