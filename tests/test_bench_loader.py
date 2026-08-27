import numpy as np
import pandas as pd
import pytest

from glucofm.data.glucofm_bench import grid_align_frame
from glucofm.data.grid import GRID_LEN


def _fake_frame():
    rows = []
    rng = np.random.default_rng(0)
    for i, (name, base) in enumerate([("ShanghaiT1DM", 150.0), ("Hall2018", 95.0)]):
        # 3 days of 5-minute readings starting at an arbitrary epoch offset
        t0 = (1_600_000_000 + i * 10_000_000) // 86400 * 86400  # midnight-aligned
        ts = t0 + np.arange(0, 3 * 86400, 300, dtype=np.float64)
        bg = base + 20 * np.sin(2 * np.pi * ts / 86400) + rng.normal(0, 3, ts.size)
        rows.append({"dataset": name, "subject_id": f"s{i}", "timestamp": ts.tolist(), "BGvalue": bg.tolist()})
    return pd.DataFrame(rows)


def test_grid_align_frame_shapes_and_subjects():
    c = grid_align_frame(_fake_frame())
    assert c["values"].shape[1] == GRID_LEN
    assert c["values"].shape[0] == c["mask"].shape[0] == c["subject"].shape[0]
    # 3 full days per participant expected (partial boundary days may drop)
    assert len(np.unique(c["subject"])) == 2
    assert c["dataset_name"].tolist() == ["ShanghaiT1DM", "Hall2018"]
    assert (c["values"].shape[0]) >= 4
    assert np.all(c["values"][c["mask"] == 0] == 0)
    assert c["mask"].mean() > 0.9  # dense 5-min sampling


def test_grid_align_frame_skips_sparse_series():
    df = _fake_frame()
    df.at[1, "timestamp"] = df.at[1, "timestamp"][:10]
    df.at[1, "BGvalue"] = df.at[1, "BGvalue"][:10]
    c = grid_align_frame(df)
    assert c["dataset_name"].tolist() == ["ShanghaiT1DM"]
