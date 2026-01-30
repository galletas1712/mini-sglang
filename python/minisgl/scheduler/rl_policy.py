"""
Policy network for RL scheduler.

DeepSets architecture for batch size selection.
"""
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class PolicyConfig:
    """Policy network configuration."""
    d_global: int = 6           # Global feature dimension
    d_request: int = 10         # Per-request feature dimension
    d_hidden: int = 128         # Hidden dimension
    max_k: int = 32             # Maximum batch size (output 0..max_k)


@dataclass
class PolicyOutput:
    """Output from the policy network."""
    batch_size_logits: torch.Tensor  # (K+1,) logits for choosing N ∈ {0..K}
    value: torch.Tensor              # Scalar or (B,) state value estimate


class SchedulerPolicy(nn.Module):
    """
    DeepSets-style policy network for Mini-SGLang scheduling.

    Permutation equivariant over requests.
    Outputs batch size logits for categorical sampling.
    """

    def __init__(self, config: PolicyConfig):
        super().__init__()
        self.config = config
        d = config.d_hidden

        # Per-request encoder
        self.request_encoder = nn.Sequential(
            nn.Linear(config.d_request, d),
            nn.LayerNorm(d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.LayerNorm(d),
            nn.ReLU(),
        )

        # Global feature encoder
        self.global_encoder = nn.Sequential(
            nn.Linear(config.d_global, d),
            nn.LayerNorm(d),
            nn.ReLU(),
        )

        # Aggregation network
        self.aggregator = nn.Sequential(
            nn.Linear(d, d),
            nn.ReLU(),
        )

        # Batch size head: outputs logits for 0, 1, ..., max_k
        self.batch_size_head = nn.Sequential(
            nn.Linear(d * 2, d),
            nn.ReLU(),
            nn.Linear(d, config.max_k + 1),
        )

        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(d * 2, d),
            nn.ReLU(),
            nn.Linear(d, 1),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with orthogonal initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        global_features: torch.Tensor,
        request_features: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> PolicyOutput:
        """
        Forward pass.

        Args:
            global_features: (G,) or (B, G)
            request_features: (N, F) or (B, N, F)
            valid_mask: (N,) or (B, N)
        """
        batched = request_features.dim() == 3
        if not batched:
            global_features = global_features.unsqueeze(0)
            request_features = request_features.unsqueeze(0)
            valid_mask = valid_mask.unsqueeze(0)

        B, N, F = request_features.shape
        d = self.config.d_hidden

        # Encode requests: (B, N, d)
        h = self.request_encoder(request_features)

        # Encode global: (B, d)
        g = self.global_encoder(global_features)

        # Masked mean pooling: (B, d)
        mask_expanded = valid_mask.unsqueeze(-1).float()
        h_masked = h * mask_expanded
        h_sum = h_masked.sum(dim=1)
        h_count = valid_mask.sum(dim=1, keepdim=True).clamp(min=1).float()
        h_agg = h_sum / h_count

        # Aggregated context: (B, d)
        context = self.aggregator(h_agg) + g

        # Combined representation for heads: (B, 2d)
        combined = torch.cat([h_agg, g], dim=-1)

        # Batch size logits: (B, K+1)
        batch_size_logits = self.batch_size_head(combined)

        # State value: (B,)
        value = self.value_head(combined).squeeze(-1)

        if not batched:
            batch_size_logits = batch_size_logits.squeeze(0)
            value = value.squeeze(0)

        return PolicyOutput(batch_size_logits=batch_size_logits, value=value)
