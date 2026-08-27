"""Alignment of irregular CGM recordings to a fixed 24-hour grid.

GlucoFM aligns each daily recording to a chronological grid at
``GRID_MINUTES`` = 5-minute resolution, giving ``GRID_LEN`` = 288 positions
per day, and keeps a binary observation mask marking which grid cells were
actually observed (sensors report at 5- or 15-minute cadence and drop out).
"""

from __future__ import annotations

import numpy as np

GRID_MINUTES = 5
GRID_LEN = 24 * 60 // GRID_MINUTES  # 288

# Physiological range reported by CGM sensors (mg/dL); values are clipped to
# this range before normalization.
GLUCOSE_MIN = 40.0
GLUCOSE_MAX = 400.0


def align_to_grid(
    timestamps_min: np.ndarray, values: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Align one day of irregular CGM samples to the 288-cell 5-minute grid.

    Args:
        timestamps_min: minutes since local midnight, shape (n,), any order.
        values: glucose readings in mg/dL, shape (n,).

    Returns:
        (grid, mask): both shape (288,). ``grid`` holds the last observed
        reading per cell (0.0 where unobserved); ``mask`` is 1.0 where the
        cell has at least one reading.
    """
    timestamps_min = np.asarray(timestamps_min, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if timestamps_min.shape != values.shape:
        raise ValueError("timestamps and values must have the same shape")

    grid = np.zeros(GRID_LEN, dtype=np.float32)
    mask = np.zeros(GRID_LEN, dtype=np.float32)

    in_day = (timestamps_min >= 0) & (timestamps_min < 24 * 60)
    idx = (timestamps_min[in_day] // GRID_MINUTES).astype(np.int64)
    vals = np.clip(values[in_day], GLUCOSE_MIN, GLUCOSE_MAX)

    order = np.argsort(timestamps_min[in_day], kind="stable")
    grid[idx[order]] = vals[order]  # later samples win within a cell
    mask[idx[order]] = 1.0
    return grid, mask


def normalize(grid: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Map observed glucose to roughly [-1, 1]; unobserved cells stay 0."""
    center = (GLUCOSE_MAX + GLUCOSE_MIN) / 2.0
    half_range = (GLUCOSE_MAX - GLUCOSE_MIN) / 2.0
    return ((grid - center) / half_range) * mask


def denormalize(x: np.ndarray) -> np.ndarray:
    center = (GLUCOSE_MAX + GLUCOSE_MIN) / 2.0
    half_range = (GLUCOSE_MAX - GLUCOSE_MIN) / 2.0
    return x * half_range + center
