"""CGM-aware augmentations used during JEPA pretraining.

The paper describes augmentations that simulate value perturbations,
heterogeneous sampling rates, and sensor dropouts, encouraging daily
representations that are robust to sensor artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class CGMAugmentConfig:
    value_jitter_std: float = 0.015  # in normalized units (~2.7 mg/dL)
    value_scale_range: tuple[float, float] = (0.97, 1.03)  # calibration drift
    subsample_prob: float = 0.3  # simulate a 15-min sensor from a 5-min one
    subsample_keep_every: int = 3
    dropout_prob: float = 0.3  # add a synthetic contiguous sensor gap
    dropout_max_len: int = 24  # grid steps (= 2 hours)


class CGMAugment:
    def __init__(self, cfg: CGMAugmentConfig | None = None):
        self.cfg = cfg or CGMAugmentConfig()

    @torch.no_grad()
    def __call__(
        self, values: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Augment a batch of (B, L) daily grids and their masks."""
        cfg = self.cfg
        B, L = values.shape
        values = values.clone()
        mask = mask.clone()

        scale = torch.empty(B, 1, device=values.device).uniform_(*cfg.value_scale_range)
        jitter = torch.randn_like(values) * cfg.value_jitter_std
        values = (values * scale + jitter) * mask

        do_sub = torch.rand(B, device=values.device) < cfg.subsample_prob
        if do_sub.any():
            keep = torch.zeros(L, dtype=torch.bool, device=values.device)
            keep[:: cfg.subsample_keep_every] = True
            sub_mask = mask[do_sub] * keep.float()
            mask[do_sub] = sub_mask
            values[do_sub] = values[do_sub] * sub_mask

        do_gap = torch.rand(B, device=values.device) < cfg.dropout_prob
        for b in torch.nonzero(do_gap).flatten().tolist():
            gap_len = int(torch.randint(4, cfg.dropout_max_len + 1, (1,)).item())
            start = int(torch.randint(0, L - gap_len, (1,)).item())
            mask[b, start : start + gap_len] = 0.0
            values[b, start : start + gap_len] = 0.0

        return values, mask
