"""Loader for the GlucoFM-Bench aggregated CGM benchmark on Hugging Face.

https://huggingface.co/datasets/glucofmbench/GlucoFM-Bench (CC BY-NC-SA 4.0)
aggregates 12 open-access CGM cohorts (BIG IDEAs, D1NAMO, HUPA-UCM,
Colas2019, ShanghaiT1DM/T2DM, Bris-T1D, T1DM-UOM, CGMacros, AZT1D,
Hall2018) at a harmonized 5-minute resolution. One row per participant:

    dataset: str, subject_id: str, timestamp: list[float] (seconds),
    BGvalue: list[float] (mg/dL)

Requires network access to huggingface.co (this loader is a no-go in
egress-restricted sandboxes — use the Shanghai loader or the synthetic
generator there).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from glucofm.data.grid import align_to_grid, normalize

_REPO = "glucofmbench/GlucoFM-Bench"
_MIN_READINGS_PER_DAY = 48


def load_glucofm_bench(
    split: str = "train", cache_dir: str | Path | None = None
) -> dict[str, np.ndarray]:
    """Download (if needed) and grid-align GlucoFM-Bench daily series.

    Returns values (N, 288), mask (N, 288), subject (N,), and dataset_name
    (n_subjects,) — the source cohort per subject, usable as a coarse
    downstream label (e.g. T1DM vs T2DM cohorts).
    """
    import pandas as pd
    from huggingface_hub import hf_hub_download

    parquet = hf_hub_download(
        _REPO,
        f"data/{split}-00000-of-00001.parquet",
        repo_type="dataset",
        cache_dir=str(cache_dir) if cache_dir else None,
    )
    return grid_align_frame(pd.read_parquet(parquet))


def grid_align_frame(df) -> dict[str, np.ndarray]:
    """Grid-align a GlucoFM-Bench-schema dataframe (one row per participant
    with ``dataset``/``subject_id``/``timestamp``/``BGvalue`` columns)."""
    all_values, all_masks, all_subject_idx, ds_names = [], [], [], []
    subject_counter = 0
    for row in df.itertuples(index=False):
        ts = np.asarray(row.timestamp, dtype=np.float64)
        bg = np.asarray(row.BGvalue, dtype=np.float64)
        ok = np.isfinite(ts) & np.isfinite(bg)
        ts, bg = ts[ok], bg[ok]
        if ts.size == 0:
            continue

        day_index = np.floor(ts / 86400.0).astype(np.int64)
        minutes = (ts % 86400.0) / 60.0
        used = False
        for d in np.unique(day_index):
            sel = day_index == d
            if sel.sum() < _MIN_READINGS_PER_DAY:
                continue
            grid, mask = align_to_grid(minutes[sel], bg[sel])
            all_values.append(normalize(grid, mask))
            all_masks.append(mask)
            all_subject_idx.append(subject_counter)
            used = True
        if used:
            ds_names.append(str(row.dataset))
            subject_counter += 1

    return {
        "values": np.stack(all_values).astype(np.float32),
        "mask": np.stack(all_masks).astype(np.float32),
        "subject": np.asarray(all_subject_idx, dtype=np.int64),
        "dataset_name": np.asarray(ds_names),
    }
