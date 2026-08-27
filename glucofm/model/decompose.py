"""Dual-stream state/event decomposition of a daily glucose grid.

A Gaussian low-pass filter with a *learnable* bandwidth splits the signal
into a slow physiological "state" stream (low-frequency glycemic baseline)
and a transient "event" stream (short-term deviations: meals, exercise,
sensor artifacts). Per the paper, the bandwidth sigma is learnable within
2-12 grid steps (roughly 10-60 minutes at 5-minute resolution) and
initialized at 6.0.

Smoothing is mask-aware: unobserved cells contribute nothing, and the
normalizer is the smoothed mask, so gaps do not drag the state toward zero.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

SIGMA_MIN = 2.0
SIGMA_MAX = 12.0
SIGMA_INIT = 6.0

# Kernel support radius in grid steps; 3 sigma_max = 36 covers the widest kernel.
_KERNEL_RADIUS = 36


def _inverse_sigmoid(y: float) -> float:
    return math.log(y / (1.0 - y))


class DualStreamDecomposition(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        init_frac = (SIGMA_INIT - SIGMA_MIN) / (SIGMA_MAX - SIGMA_MIN)
        self._sigma_raw = nn.Parameter(torch.tensor(_inverse_sigmoid(init_frac)))
        offsets = torch.arange(-_KERNEL_RADIUS, _KERNEL_RADIUS + 1, dtype=torch.float32)
        self.register_buffer("_offsets", offsets)

    @property
    def sigma(self) -> torch.Tensor:
        """Current bandwidth in grid steps, constrained to [SIGMA_MIN, SIGMA_MAX]."""
        return SIGMA_MIN + (SIGMA_MAX - SIGMA_MIN) * torch.sigmoid(self._sigma_raw)

    def _kernel(self) -> torch.Tensor:
        k = torch.exp(-0.5 * (self._offsets / self.sigma) ** 2)
        return (k / k.sum()).view(1, 1, -1)

    def forward(
        self, values: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Split (B, L) values/mask into (state, event), both (B, L).

        state + event == values at observed cells; both are 0 where unobserved.
        """
        kernel = self._kernel()
        v = (values * mask).unsqueeze(1)
        m = mask.unsqueeze(1)
        num = F.conv1d(v, kernel, padding=_KERNEL_RADIUS)
        den = F.conv1d(m, kernel, padding=_KERNEL_RADIUS)
        state = (num / den.clamp(min=1e-6)).squeeze(1) * mask
        event = (values - state) * mask
        return state, event
