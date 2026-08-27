"""Loader for the ShanghaiT1DM / ShanghaiT2DM CGM dataset.

Real-world CGM cohort (Zhao et al., "Chinese diabetes datasets for
data-driven machine learning", Scientific Data 2023; CC BY 4.0,
https://doi.org/10.6084/m9.figshare.c.6310860): 12 T1DM and 100 T2DM
patients with 3-14 days of CGM sampled every 15 minutes, one Excel table
per recording period named ``<patient>_<period>_<date>.xls[x]``.

Each calendar day of each recording is aligned to the 288-cell 5-minute
grid (15-minute sensors simply occupy every third cell, which the
observation mask records). Subject labels: 1 = T1DM, 0 = T2DM — a real
downstream diabetes-type prediction task for linear probing.

Requires: pandas + openpyxl + xlrd (``pip install glucofm[data] xlrd``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from glucofm.data.grid import align_to_grid, normalize

_MIN_READINGS_PER_DAY = 48  # at 15-min cadence: at least 12 hours observed


def _cgm_column(columns) -> str | None:
    """Find the CGM column; a few tables name it 'CGM ' instead of
    'CGM (mg / dl)' (values are mg/dL throughout the dataset)."""
    for c in columns:
        if str(c).strip().lower().startswith("cgm"):
            return c
    return None


def load_shanghai_cohort(
    root: str | Path, min_days_per_subject: int = 1
) -> dict[str, np.ndarray]:
    """Load Shanghai_T1DM/ and Shanghai_T2DM/ Excel tables under ``root``.

    Returns the same dict layout as ``glucofm.data.synthetic.generate_cohort``:
    values (N, 288), mask (N, 288), subject (N,), label (n_subjects,) with
    label 1 for T1DM and 0 for T2DM. Subject indices are dense (0..S-1).
    """
    import pandas as pd

    root = Path(root)
    days_by_subject: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    label_by_subject: dict[str, int] = {}

    for folder, lab in (("Shanghai_T1DM", 1), ("Shanghai_T2DM", 0)):
        d = root / folder
        if not d.is_dir():
            raise FileNotFoundError(f"missing dataset folder: {d}")
        for path in sorted(d.iterdir()):
            if path.suffix.lower() not in (".xls", ".xlsx"):
                continue
            patient_id = f"{folder}:{path.name.split('_')[0]}"
            try:
                df = pd.read_excel(path)
            except Exception as exc:  # malformed table: skip, don't abort the cohort
                print(f"warning: skipping {path.name}: {exc}")
                continue
            cgm_col = _cgm_column(df.columns)
            if cgm_col is None or "Date" not in df.columns:
                print(f"warning: skipping {path.name}: no CGM/Date column")
                continue
            df = df.dropna(subset=["Date", cgm_col])
            ts = pd.to_datetime(df["Date"])
            values = pd.to_numeric(df[cgm_col], errors="coerce").to_numpy(dtype=np.float64)
            keep = np.isfinite(values)
            ts, values = ts[keep], values[keep]
            minutes = (
                ts.dt.hour.to_numpy() * 60
                + ts.dt.minute.to_numpy()
                + ts.dt.second.to_numpy() / 60.0
            )
            day = ts.dt.normalize().to_numpy()
            for d in np.unique(day):
                sel = day == d
                if sel.sum() < _MIN_READINGS_PER_DAY:
                    continue
                grid, mask = align_to_grid(minutes[sel], values[sel])
                days_by_subject.setdefault(patient_id, []).append(
                    (normalize(grid, mask), mask)
                )
                label_by_subject[patient_id] = lab

    subjects = sorted(
        s for s, days in days_by_subject.items() if len(days) >= min_days_per_subject
    )
    all_values, all_masks, all_subject_idx, labels = [], [], [], []
    for i, s in enumerate(subjects):
        labels.append(label_by_subject[s])
        for v, m in days_by_subject[s]:
            all_values.append(v)
            all_masks.append(m)
            all_subject_idx.append(i)

    return {
        "values": np.stack(all_values).astype(np.float32),
        "mask": np.stack(all_masks).astype(np.float32),
        "subject": np.asarray(all_subject_idx, dtype=np.int64),
        "label": np.asarray(labels, dtype=np.int64),
    }
