"""Hand-crafted clinical CGM feature baseline.

The standard day-level glycemia metrics a clinician or classical ML
pipeline would compute: mean/variability, time-in-range bands, extremes,
rate-of-change, and time-of-day means. Probing these with the exact same
protocol as the encoder embeddings answers the question the paper's
baselines answer: does the foundation model learn anything a feature
engineer would not get for free?
"""

from __future__ import annotations

import numpy as np

from glucofm.data.grid import GRID_LEN, denormalize

FEATURE_NAMES = [
    "mean", "std", "cv", "min", "max",
    "tir_70_180", "tbr_70", "tbr_54", "tar_180", "tar_250",
    "mean_abs_delta", "obs_frac",
    "night_mean", "morning_mean", "afternoon_mean", "evening_mean",
]


def day_features(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """(N, 288) normalized grids -> (N, 16) clinical features in mg/dL units."""
    g = denormalize(values.astype(np.float64))
    g[mask == 0] = np.nan

    with np.errstate(invalid="ignore"):
        mean = np.nanmean(g, axis=1)
        std = np.nanstd(g, axis=1)
        cv = std / np.maximum(mean, 1e-6)
        gmin = np.nanmin(g, axis=1)
        gmax = np.nanmax(g, axis=1)
        tir = np.nanmean((g >= 70) & (g <= 180), axis=1)
        tbr70 = np.nanmean(g < 70, axis=1)
        tbr54 = np.nanmean(g < 54, axis=1)
        tar180 = np.nanmean(g > 180, axis=1)
        tar250 = np.nanmean(g > 250, axis=1)
        delta = np.abs(np.diff(g, axis=1))
        mean_abs_delta = np.nanmean(delta, axis=1)
        obs_frac = mask.mean(axis=1)
        q = GRID_LEN // 4
        night = np.nanmean(g[:, :q], axis=1)
        morning = np.nanmean(g[:, q : 2 * q], axis=1)
        afternoon = np.nanmean(g[:, 2 * q : 3 * q], axis=1)
        evening = np.nanmean(g[:, 3 * q :], axis=1)

    feats = np.stack(
        [mean, std, cv, gmin, gmax, tir, tbr70, tbr54, tar180, tar250,
         mean_abs_delta, obs_frac, night, morning, afternoon, evening],
        axis=1,
    )
    # A day with an empty time-of-day quarter yields NaNs: fill with the
    # column mean so the probe stays defined.
    col_mean = np.nanmean(feats, axis=0)
    nan_pos = np.isnan(feats)
    feats[nan_pos] = np.take(col_mean, np.nonzero(nan_pos)[1])
    return feats.astype(np.float32)
