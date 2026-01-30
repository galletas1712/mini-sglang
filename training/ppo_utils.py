"""
PPO training utilities for Mini-SGLang RL scheduler.

Includes GAE computation, PPO loss, and behavior cloning loss.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    pass

# Add parent for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

from minisgl.scheduler.rl_action import BatchSizeDistribution
from minisgl.scheduler.rl_policy import PolicyOutput, SchedulerPolicy
from minisgl.scheduler.rl_trajectory import TrajectoryStep


@dataclass
class PPOConfig:
    """PPO hyperparameters."""
    gamma: float = 0.99                 # Discount factor
    lam: float = 0.95                   # GAE lambda
    clip_epsilon: float = 0.2           # PPO clipping parameter
    value_coef: float = 0.5             # Value loss coefficient
    entropy_coef: float = 0.01          # Entropy bonus coefficient
    max_grad_norm: float = 0.5          # Gradient clipping
    ppo_epochs: int = 4                 # PPO update epochs
    minibatch_size: int = 64            # Minibatch size


@dataclass
class ProcessedBatch:
    """Processed trajectory data for training."""
    global_features: torch.Tensor       # (B, G)
    request_features: torch.Tensor      # (B, N, F)
    valid_masks: torch.Tensor           # (B, N)
    num_pendings: torch.Tensor          # (B,)
    actions: list[int]                  # B x batch_size
    old_log_probs: torch.Tensor         # (B,)
    old_values: torch.Tensor            # (B,)
    rewards: torch.Tensor               # (B,)
    advantages: torch.Tensor            # (B,)
    returns: torch.Tensor               # (B,)
    teacher_actions: list[int | None]   # B x batch_size or None


def compute_gae(
    rewards: list[float],
    values: list[float],
    dones: list[bool],
    gamma: float = 0.99,
    lam: float = 0.95,
) -> tuple[list[float], list[float]]:
    """
    Compute Generalized Advantage Estimation (GAE).

    Returns (advantages, returns).
    """
    advantages = []
    returns = []

    gae = 0.0
    next_value = 0.0

    # Reverse iteration
    for i in reversed(range(len(rewards))):
        if dones[i]:
            next_value = 0.0
            gae = 0.0

        delta = rewards[i] + gamma * next_value - values[i]
        gae = delta + gamma * lam * gae
        advantages.insert(0, gae)
        returns.insert(0, gae + values[i])
        next_value = values[i]

    return advantages, returns


def process_trajectory_batch(
    steps: list[TrajectoryStep],
    gamma: float = 0.99,
    lam: float = 0.95,
    normalize_advantages: bool = True,
) -> ProcessedBatch:
    """
    Process trajectory steps into batched tensors for training.
    """
    if not steps:
        raise ValueError("Empty trajectory batch")

    # Extract data
    rewards = [s.reward for s in steps]
    values = [s.value for s in steps]
    dones = [s.done for s in steps]

    # Compute GAE
    advantages, returns = compute_gae(rewards, values, dones, gamma, lam)

    # Stack features
    global_features = torch.stack([s.global_features for s in steps])
    request_features = torch.stack([s.request_features for s in steps])
    valid_masks = torch.stack([s.valid_mask for s in steps])
    num_pendings = torch.tensor([s.num_pending for s in steps])

    actions = [s.action for s in steps]
    teacher_actions = [s.teacher_action for s in steps]

    old_log_probs = torch.tensor([s.log_prob for s in steps])
    old_values = torch.tensor(values)
    rewards_tensor = torch.tensor(rewards)
    advantages_tensor = torch.tensor(advantages, dtype=torch.float32)
    returns_tensor = torch.tensor(returns, dtype=torch.float32)

    # Normalize advantages
    if normalize_advantages and len(advantages_tensor) > 1:
        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (
            advantages_tensor.std() + 1e-8
        )

    return ProcessedBatch(
        global_features=global_features,
        request_features=request_features,
        valid_masks=valid_masks,
        num_pendings=num_pendings,
        actions=actions,
        old_log_probs=old_log_probs,
        old_values=old_values,
        rewards=rewards_tensor,
        advantages=advantages_tensor,
        returns=returns_tensor,
        teacher_actions=teacher_actions,
    )


def compute_ppo_loss(
    policy: SchedulerPolicy,
    batch: ProcessedBatch,
    config: PPOConfig,
    device: torch.device,
) -> tuple[torch.Tensor, dict]:
    """
    Compute PPO clipped objective for batch size policy.

    Returns (loss, metrics).
    """
    # Move to device
    global_features = batch.global_features.to(device)
    request_features = batch.request_features.to(device)
    valid_masks = batch.valid_masks.to(device)
    old_log_probs = batch.old_log_probs.to(device)
    advantages = batch.advantages.to(device)
    returns = batch.returns.to(device)

    B = global_features.shape[0]

    # Forward pass
    output = policy(global_features, request_features, valid_masks)

    # Compute new log probs and entropy
    policy_losses = []
    entropies = []

    for i in range(B):
        num_pending = batch.num_pendings[i].item()
        if num_pending == 0:
            continue

        # Create batch size distribution
        dist = BatchSizeDistribution(
            output.batch_size_logits[i],
            num_pending=num_pending,
        )

        action = batch.actions[i]  # int: batch size
        new_log_prob = dist.log_prob_of(action)
        old_log_prob = old_log_probs[i]
        advantage = advantages[i]

        # PPO ratio
        ratio = torch.exp(new_log_prob - old_log_prob)
        clipped_ratio = torch.clamp(ratio, 1 - config.clip_epsilon, 1 + config.clip_epsilon)

        # Clipped objective (negative because we minimize)
        policy_loss = -torch.min(ratio * advantage, clipped_ratio * advantage)
        policy_losses.append(policy_loss)

        # Entropy
        entropies.append(dist.entropy())

    if not policy_losses:
        return torch.tensor(0.0, device=device, requires_grad=True), {"loss": 0.0}

    # Aggregate losses
    policy_loss = torch.stack(policy_losses).mean()
    value_loss = F.mse_loss(output.value, returns)
    entropy = torch.stack(entropies).mean()

    # Total loss
    loss = policy_loss + config.value_coef * value_loss - config.entropy_coef * entropy

    metrics = {
        "loss": loss.item(),
        "policy_loss": policy_loss.item(),
        "value_loss": value_loss.item(),
        "entropy": entropy.item(),
    }

    return loss, metrics


def compute_bc_loss(
    policy: SchedulerPolicy,
    batch: ProcessedBatch,
    device: torch.device,
) -> tuple[torch.Tensor, dict]:
    """
    Compute behavior cloning loss (NLL of teacher batch size).

    Returns (loss, metrics).
    """
    # Move to device
    global_features = batch.global_features.to(device)
    request_features = batch.request_features.to(device)
    valid_masks = batch.valid_masks.to(device)

    B = global_features.shape[0]

    # Forward pass
    output = policy(global_features, request_features, valid_masks)

    # Compute BC loss for each sample
    bc_losses = []

    for i in range(B):
        num_pending = batch.num_pendings[i].item()
        teacher_action = batch.teacher_actions[i]  # int: teacher batch size

        if num_pending == 0 or teacher_action is None:
            continue

        # Create batch size distribution
        dist = BatchSizeDistribution(
            output.batch_size_logits[i],
            num_pending=num_pending,
        )

        # Negative log likelihood of teacher action
        log_prob = dist.log_prob_of(teacher_action)
        bc_losses.append(-log_prob)

    if not bc_losses:
        return torch.tensor(0.0, device=device, requires_grad=True), {"bc_loss": 0.0}

    bc_loss = torch.stack(bc_losses).mean()

    metrics = {
        "loss": bc_loss.item(),
        "bc_loss": bc_loss.item(),
    }

    return bc_loss, metrics


def compute_combined_loss(
    policy: SchedulerPolicy,
    batch: ProcessedBatch,
    config: PPOConfig,
    bc_weight: float,
    device: torch.device,
) -> tuple[torch.Tensor, dict]:
    """
    Compute combined BC + PPO loss.

    Returns (loss, metrics).
    """
    ppo_loss, ppo_metrics = compute_ppo_loss(policy, batch, config, device)
    bc_loss, bc_metrics = compute_bc_loss(policy, batch, device)

    # Weighted combination
    loss = (1 - bc_weight) * ppo_loss + bc_weight * bc_loss

    metrics = {
        "loss": loss.item(),
        "ppo_loss": ppo_metrics.get("policy_loss", 0.0),
        "bc_loss": bc_metrics.get("bc_loss", 0.0),
        "value_loss": ppo_metrics.get("value_loss", 0.0),
        "entropy": ppo_metrics.get("entropy", 0.0),
    }

    return loss, metrics
