"""
Action distribution for RL scheduler.

Categorical distribution over batch sizes.
"""
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.distributions import Categorical


@dataclass
class RLAction:
    """Sampled action from the policy."""
    batch_size: int          # Number of requests to process (N)
    log_prob: torch.Tensor   # Log probability of this action


class BatchSizeDistribution:
    """
    Distribution over batch sizes via categorical softmax.

    Samples/scores N ∈ {0, 1, ..., K} where K = min(num_pending, max_k).
    """

    def __init__(self, logits: torch.Tensor, num_pending: int):
        """
        Args:
            logits: (K+1,) raw scores for batch sizes 0..K
            num_pending: actual pending count (can't choose N > num_pending)
        """
        self.logits = logits
        self.num_pending = num_pending
        self.max_k = logits.shape[-1] - 1  # logits has K+1 elements

    def sample(self) -> tuple[int, torch.Tensor]:
        """Sample a batch size. Returns (batch_size, log_prob)."""
        # Mask out invalid choices (N > num_pending)
        valid_logits = self.logits.clone()
        valid_logits[self.num_pending + 1:] = float('-inf')

        dist = Categorical(logits=valid_logits)
        n = dist.sample()
        return n.item(), dist.log_prob(n)

    def log_prob_of(self, batch_size: int) -> torch.Tensor:
        """Compute log-probability of a given batch size.

        Args:
            batch_size: The batch size to compute probability for.

        Returns:
            Log probability of the batch size.

        Raises:
            ValueError: If batch_size is out of valid range.
        """
        if batch_size < 0 or batch_size > self.num_pending:
            raise ValueError(
                f"batch_size {batch_size} is out of valid range [0, {self.num_pending}]"
            )

        valid_logits = self.logits.clone()
        valid_logits[self.num_pending + 1:] = float('-inf')

        log_probs = F.log_softmax(valid_logits, dim=-1)
        return log_probs[batch_size]

    def greedy(self) -> int:
        """Greedy decoding: return the highest-scoring valid batch size."""
        valid_logits = self.logits.clone()
        valid_logits[self.num_pending + 1:] = float('-inf')
        return valid_logits.argmax().item()

    def entropy(self) -> torch.Tensor:
        """Compute entropy of the distribution."""
        valid_logits = self.logits.clone()
        valid_logits[self.num_pending + 1:] = float('-inf')

        probs = F.softmax(valid_logits, dim=-1)
        log_probs = F.log_softmax(valid_logits, dim=-1)
        # Avoid log(0) issues by only summing over valid entries
        entropy = -(probs[:self.num_pending + 1] * log_probs[:self.num_pending + 1]).sum()
        return entropy.clamp(min=0.0)
