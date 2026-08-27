import numpy as np
import pytest
import torch

from glucofm.data.augment import CGMAugment
from glucofm.data.grid import GRID_LEN, align_to_grid, denormalize, normalize
from glucofm.data.synthetic import SyntheticCGMConfig, generate_cohort


def test_grid_alignment_places_readings():
    t = np.array([0.0, 5.0, 17.0, 1435.0])
    v = np.array([100.0, 120.0, 140.0, 90.0])
    grid, mask = align_to_grid(t, v)
    assert grid.shape == (GRID_LEN,) and mask.shape == (GRID_LEN,)
    assert mask.sum() == 4
    assert grid[0] == 100.0 and grid[1] == 120.0
    assert grid[3] == 140.0  # 17 min -> cell 3
    assert grid[287] == 90.0
    assert grid[mask == 0].sum() == 0


def test_grid_alignment_last_reading_wins_and_clips():
    t = np.array([2.0, 4.0])
    v = np.array([500.0, 30.0])  # same cell, out-of-range values
    grid, mask = align_to_grid(t, v)
    assert mask.sum() == 1
    assert grid[0] == 40.0  # last reading, clipped to sensor floor


def test_normalize_roundtrip():
    t = np.arange(0, 1440, 5.0)
    v = np.linspace(50, 350, t.size)
    grid, mask = align_to_grid(t, v)
    norm = normalize(grid, mask)
    assert np.abs(norm).max() <= 1.0
    assert np.allclose(denormalize(norm)[mask == 1], grid[mask == 1], atol=1e-3)


def test_synthetic_cohort_shapes_and_labels():
    cfg = SyntheticCGMConfig(n_subjects=6, days_per_subject=3, seed=1)
    c = generate_cohort(cfg)
    assert c["values"].shape == (18, GRID_LEN)
    assert c["mask"].shape == (18, GRID_LEN)
    assert c["subject"].shape == (18,)
    assert c["label"].shape == (6,)
    assert set(np.unique(c["subject"])) == set(range(6))
    assert c["mask"].mean() > 0.5  # mostly observed
    assert np.all(c["values"][c["mask"] == 0] == 0)


def test_synthetic_dysglycemia_is_higher_glucose():
    cfg = SyntheticCGMConfig(n_subjects=30, days_per_subject=2, seed=2)
    c = generate_cohort(cfg)
    means = []
    for s in range(30):
        sel = c["subject"] == s
        v, m = c["values"][sel], c["mask"][sel]
        means.append(v.sum() / m.sum())
    means = np.array(means)
    assert means[c["label"] == 1].mean() > means[c["label"] == 0].mean()


def test_augment_preserves_shapes_and_mask_consistency():
    torch.manual_seed(0)
    cfg = SyntheticCGMConfig(n_subjects=4, days_per_subject=2, seed=3)
    c = generate_cohort(cfg)
    v = torch.from_numpy(c["values"])
    m = torch.from_numpy(c["mask"])
    av, am = CGMAugment()(v, m)
    assert av.shape == v.shape and am.shape == m.shape
    # augmentation can only remove observations, never invent them
    assert torch.all(am <= m)
    # values remain zero where unobserved
    assert torch.all(av[am == 0] == 0)
