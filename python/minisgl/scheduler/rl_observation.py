"""
Observation builder for RL scheduler.

Constructs normalized features from scheduler state for the policy network.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .scheduler import Scheduler


@dataclass
class RLObservation:
    """Observation for the RL policy."""
    global_features: torch.Tensor      # (G,), float32
    request_features: torch.Tensor     # (K, F), float32 - first K pending requests
    valid_mask: torch.Tensor           # (K,), bool
    request_ids: list[int | None]      # (K,), uid or None
    num_valid: int
    num_pending: int
    num_running: int


class ObservationBuilder:
    """
    Pre-allocates torch tensors for observation building.

    Create once in RLScheduler.__init__, reuse each step.
    Only observes the first K pending requests (FCFS order).
    """

    NUM_GLOBAL_FEATURES = 6
    NUM_REQUEST_FEATURES = 10

    def __init__(
        self,
        max_running_req: int,
        max_seq_len: int,
        num_pages: int,
        max_prefill_budget: int,
        candidate_k: int,
    ):
        self.max_running_req = max_running_req
        self.max_seq_len = max_seq_len
        self.num_pages = num_pages
        self.max_prefill_budget = max_prefill_budget
        self.candidate_k = candidate_k

        # Pre-allocate tensors (only for K candidates now)
        self._global_features = torch.zeros(
            self.NUM_GLOBAL_FEATURES, dtype=torch.float32
        )
        self._request_features = torch.zeros(
            (candidate_k, self.NUM_REQUEST_FEATURES), dtype=torch.float32
        )
        self._valid_mask = torch.zeros(candidate_k, dtype=torch.bool)
        self._request_ids: list[int | None] = [None] * candidate_k

    def build(self, scheduler: "Scheduler") -> RLObservation:
        """Build observation from scheduler state."""
        # Reset tensors
        self._request_features.zero_()
        self._valid_mask.zero_()
        for i in range(self.candidate_k):
            self._request_ids[i] = None

        prefill_mgr = scheduler.prefill_manager
        decode_mgr = scheduler.decode_manager
        cache_mgr = scheduler.cache_manager
        table_mgr = scheduler.table_manager

        num_pending = len(prefill_mgr.pending_list)
        num_running = len(decode_mgr.running_reqs)

        # Global features (G = 6)
        # 0: kv_cache_usage
        kv_used = cache_mgr.num_pages - cache_mgr.available_size
        self._global_features[0] = kv_used / max(cache_mgr.num_pages, 1)
        # 1: num_pending_norm
        self._global_features[1] = min(num_pending, self.max_running_req) / self.max_running_req
        # 2: num_running_norm
        self._global_features[2] = num_running / self.max_running_req
        # 3: available_slots_norm
        self._global_features[3] = table_mgr.available_size / self.max_running_req
        # 4: prefill_budget_norm
        self._global_features[4] = scheduler.prefill_budget / self.max_prefill_budget
        # 5: decode_inflight_norm
        self._global_features[5] = decode_mgr.inflight_tokens / max(self.max_seq_len, 1)

        # Only look at first K candidates (FCFS order)
        candidates = list(prefill_mgr.pending_list[:self.candidate_k])

        now = time.time()
        idx = 0

        # Build features for first K pending requests only
        for pending_req in candidates:
            input_len = pending_req.input_len
            cached_len = pending_req.cached_len
            output_len = pending_req.output_len

            # 0: is_pending (always 1.0 for pending requests)
            self._request_features[idx, 0] = 1.0
            # 1: is_chunked
            self._request_features[idx, 1] = float(pending_req.chunked_req is not None)
            # 2: input_len_norm
            self._request_features[idx, 2] = input_len / self.max_seq_len
            # 3: cached_len_norm
            self._request_features[idx, 3] = cached_len / self.max_seq_len
            # 4: extend_len_norm (tokens to compute)
            self._request_features[idx, 4] = (input_len - cached_len) / self.max_seq_len
            # 5: output_len_norm
            self._request_features[idx, 5] = output_len / self.max_seq_len
            # 6: remain_len_norm (for pending, same as output_len)
            self._request_features[idx, 6] = output_len / self.max_seq_len
            # 7: cache_hit_ratio
            self._request_features[idx, 7] = cached_len / max(input_len, 1)
            # 8: age_log1p
            self._request_features[idx, 8] = math.log1p(max(0.0, now - pending_req.arrival_time))
            # 9: progress (0 for pending)
            self._request_features[idx, 9] = 0.0

            self._valid_mask[idx] = True
            self._request_ids[idx] = pending_req.uid
            idx += 1

        return RLObservation(
            global_features=self._global_features,
            request_features=self._request_features,
            valid_mask=self._valid_mask,
            request_ids=self._request_ids.copy(),  # Copy to avoid mutation
            num_valid=idx,
            num_pending=num_pending,  # Total pending, not just observed
            num_running=num_running,
        )
