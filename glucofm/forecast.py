"""Short-horizon glucose forecasting on top of the pretrained encoder.

Task: given a day's CGM readings up to time t, predict the next 30 minutes
(6 grid steps) as deltas from the last observed value. The frozen GlucoFM
encoder embeds the partial day (its pretraining saw sensor dropout and
masked patches, so a day truncated at t is in-distribution); a small MLP
head maps [pooled embedding, recent trend, time-of-day] to the 6 deltas.

Evaluation is subject-disjoint (GlucoFM-Bench ships a participant-level
train/test split) against the two canonical short-horizon baselines:
last-value persistence and linear-trend extrapolation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from glucofm.data.grid import GLUCOSE_MAX, GLUCOSE_MIN, GRID_LEN
from glucofm.model.glucofm import GlucoFM

HORIZON = 6  # 6 x 5 min = 30 minutes
HISTORY_STEPS = 36  # require 3 hours of context
_HALF_RANGE = (GLUCOSE_MAX - GLUCOSE_MIN) / 2.0  # normalized units -> mg/dL
_SLOPE_STEPS = 3  # slope over the last 15 minutes
_RECENT_STEPS = 12  # raw last-hour trace fed directly to the head
AUX_DIM = 4 + _RECENT_STEPS


@dataclass
class ForecastSamples:
    day_idx: np.ndarray  # (S,)
    t_idx: np.ndarray  # (S,) grid position of the last observed reading


def sample_forecast_points(
    mask: np.ndarray,
    n_per_day: int = 6,
    seed: int = 0,
    min_history_obs: int = 24,
) -> ForecastSamples:
    """Pick (day, t) pairs with observed history, slope window, and future."""
    rng = np.random.default_rng(seed)
    day_idx, t_idx = [], []
    for d in range(mask.shape[0]):
        m = mask[d]
        candidates = []
        for t in range(HISTORY_STEPS, GRID_LEN - HORIZON):
            if (
                m[t - _SLOPE_STEPS : t + 1].all()
                and m[t + 1 : t + 1 + HORIZON].all()
                and m[t - HISTORY_STEPS : t].sum() >= min_history_obs
            ):
                candidates.append(t)
        if candidates:
            take = rng.choice(len(candidates), min(n_per_day, len(candidates)), replace=False)
            for i in take:
                day_idx.append(d)
                t_idx.append(candidates[i])
    return ForecastSamples(
        np.asarray(day_idx, dtype=np.int64), np.asarray(t_idx, dtype=np.int64)
    )


@torch.no_grad()
def forecast_features(
    model: GlucoFM,
    values: np.ndarray,
    mask: np.ndarray,
    samples: ForecastSamples,
    batch: int = 256,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build (X, y, slope) arrays for the head.

    X: (S, fused_dim + AUX_DIM) = [pooled embedding of the day truncated at
       t; last value, 15-min slope per step, sin/cos time-of-day at t; the
       last hour's readings as mask-weighted deltas from the last value]
    y: (S, HORIZON) future deltas from the last value (normalized units)
    slope: (S,) kept separately for the linear-trend baseline.
    """
    model.eval()
    S = len(samples.day_idx)
    grid_pos = np.arange(GRID_LEN)[None, :]

    xs, ys, slopes = [], [], []
    for i in range(0, S, batch):
        d = samples.day_idx[i : i + batch]
        t = samples.t_idx[i : i + batch]
        v = values[d].copy()
        m = mask[d].copy()
        future = grid_pos > t[:, None]
        v[future] = 0.0
        m[future] = 0.0

        pooled = model(torch.from_numpy(v).float(), torch.from_numpy(m).float()).numpy()

        rows = np.arange(len(d))
        v_full = values[d]
        v_t = v_full[rows, t]
        slope = (v_t - v_full[rows, t - _SLOPE_STEPS]) / _SLOPE_STEPS
        phase = 2 * np.pi * t / GRID_LEN
        aux = np.stack([v_t, slope, np.sin(phase), np.cos(phase)], axis=1)
        recent = np.stack(
            [
                (v_full[rows, t - k] - v_t) * mask[d][rows, t - k]
                for k in range(_RECENT_STEPS - 1, -1, -1)
            ],
            axis=1,
        )
        xs.append(np.concatenate([pooled, aux, recent], axis=1))

        y = np.stack(
            [v_full[rows, t + k] - v_t for k in range(1, HORIZON + 1)], axis=1
        )
        ys.append(y)
        slopes.append(slope)

    return (
        np.concatenate(xs).astype(np.float32),
        np.concatenate(ys).astype(np.float32),
        np.concatenate(slopes).astype(np.float32),
    )


class ForecastHead(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, HORIZON)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_head(
    x: np.ndarray,
    y: np.ndarray,
    epochs: int = 40,
    batch_size: int = 512,
    lr: float = 1e-3,
    seed: int = 0,
) -> ForecastHead:
    torch.manual_seed(seed)
    head = ForecastHead(x.shape[1])
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    xt = torch.from_numpy(x)
    yt = torch.from_numpy(y)
    n = len(xt)
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        order = rng.permutation(n)
        for i in range(0, n, batch_size):
            idx = order[i : i + batch_size]
            loss = F.smooth_l1_loss(head(xt[idx]), yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    return head


def evaluate_forecast(
    head: ForecastHead, x: np.ndarray, y: np.ndarray, slope: np.ndarray
) -> dict[str, float]:
    """RMSE/MAE at the 30-minute horizon in mg/dL, vs both baselines."""
    with torch.no_grad():
        pred = head(torch.from_numpy(x)).numpy()

    k = np.arange(1, HORIZON + 1)[None, :]
    baselines = {
        "model": pred,
        "persistence": np.zeros_like(y),
        "linear_trend": slope[:, None] * k,
    }
    out: dict[str, float] = {"n_samples": float(len(y))}
    for name, p in baselines.items():
        err = (p - y) * _HALF_RANGE  # mg/dL
        out[f"{name}_rmse_30min"] = float(np.sqrt(np.mean(err[:, -1] ** 2)))
        out[f"{name}_mae_30min"] = float(np.mean(np.abs(err[:, -1])))
        out[f"{name}_rmse_all"] = float(np.sqrt(np.mean(err**2)))
    return out
