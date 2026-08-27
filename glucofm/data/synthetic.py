"""Physiologically-motivated synthetic CGM cohort generator.

Used for pretraining smoke runs and tests when no real CGM cohort is
reachable (network-restricted environments). Days are simulated with a
circadian baseline (dawn phenomenon), meal-driven postprandial excursions,
AR(1) sensor noise, and sensor dropout gaps — then aligned to the 288-cell
grid like any real recording.

Each synthetic subject carries a binary ``dysglycemia`` label (elevated
baseline, larger/longer excursions, higher variability) so downstream
linear-probe evaluation has a ground-truth signal to recover.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from glucofm.data.grid import GRID_LEN, GRID_MINUTES, align_to_grid, normalize


@dataclass
class SyntheticCGMConfig:
    n_subjects: int = 40
    days_per_subject: int = 8
    dysglycemia_fraction: float = 0.5
    sampling_minutes: int = 5  # native sensor cadence
    dropout_prob: float = 0.3  # chance a day contains a sensor gap
    max_gap_minutes: int = 120
    seed: int = 0


def _simulate_day(rng: np.random.Generator, dysglycemic: bool, subject_offset: float) -> tuple[np.ndarray, np.ndarray]:
    """Simulate one day of glucose at 1-minute resolution, return (t_min, mg/dL)."""
    t = np.arange(0, 24 * 60, dtype=np.float64)

    baseline = (125.0 if dysglycemic else 90.0) + subject_offset
    # Dawn phenomenon: mild early-morning rise peaking ~6-8am.
    circadian = (12.0 if dysglycemic else 6.0) * np.exp(
        -0.5 * ((t - 7 * 60) / 90.0) ** 2
    )

    glucose = baseline + circadian

    n_meals = rng.integers(3, 6)
    meal_times = np.sort(rng.uniform(6 * 60, 22 * 60, size=n_meals))
    for mt in meal_times:
        peak = rng.uniform(60, 140) if dysglycemic else rng.uniform(25, 70)
        rise_tau = rng.uniform(20, 40)
        fall_tau = rng.uniform(70, 140) if dysglycemic else rng.uniform(40, 80)
        dt = t - mt
        shape = np.where(
            dt < 0,
            0.0,
            (1 - np.exp(-np.maximum(dt, 0) / rise_tau)) * np.exp(-np.maximum(dt, 0) / fall_tau),
        )
        glucose += peak * shape / max(shape.max(), 1e-6)

    # AR(1) physiological + sensor noise.
    noise_scale = 6.0 if dysglycemic else 3.5
    eps = rng.normal(0, 1, size=t.size)
    ar = np.zeros_like(eps)
    for i in range(1, t.size):
        ar[i] = 0.9 * ar[i - 1] + eps[i]
    glucose += noise_scale * ar / ar.std().clip(min=1e-6)

    return t, glucose


def generate_cohort(cfg: SyntheticCGMConfig) -> dict[str, np.ndarray]:
    """Generate a cohort of daily CGM grids.

    Returns dict with:
        values:  (N, 288) normalized glucose, zeros where unobserved
        mask:    (N, 288) observation mask
        subject: (N,) subject index per day
        label:   (n_subjects,) binary dysglycemia label per subject
    """
    rng = np.random.default_rng(cfg.seed)
    labels = (rng.uniform(size=cfg.n_subjects) < cfg.dysglycemia_fraction).astype(np.int64)

    all_values, all_masks, all_subjects = [], [], []
    for s in range(cfg.n_subjects):
        subject_offset = rng.normal(0, 8.0)
        for _ in range(cfg.days_per_subject):
            t, g = _simulate_day(rng, bool(labels[s]), subject_offset)

            # Subsample to the sensor cadence (heterogeneous rates supported).
            step = max(cfg.sampling_minutes, 1)
            t_s, g_s = t[::step], g[::step]

            # Sensor dropout: remove a contiguous gap.
            if rng.uniform() < cfg.dropout_prob:
                gap_len = rng.uniform(20, cfg.max_gap_minutes)
                gap_start = rng.uniform(0, 24 * 60 - gap_len)
                keep = (t_s < gap_start) | (t_s > gap_start + gap_len)
                t_s, g_s = t_s[keep], g_s[keep]

            grid, mask = align_to_grid(t_s, g_s)
            all_values.append(normalize(grid, mask))
            all_masks.append(mask)
            all_subjects.append(s)

    return {
        "values": np.stack(all_values).astype(np.float32),
        "mask": np.stack(all_masks).astype(np.float32),
        "subject": np.asarray(all_subjects, dtype=np.int64),
        "label": labels,
    }
