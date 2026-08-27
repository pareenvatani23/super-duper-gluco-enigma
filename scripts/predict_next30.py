#!/usr/bin/env python3
"""Predict the next 30 minutes of glucose from a personal CGM export.

Supports Glooko CGM exports (``cgm_data_1.csv``: a name/date-range header
line, then Timestamp + "CGM Glucose Value (mmol/l)" columns) and plain
CSVs with timestamp/value columns in mg/dL.

NOT a medical device. Do not use for insulin dosing or treatment decisions.

Example:
  python scripts/predict_next30.py --csv /path/to/cgm_data_1.csv \
      --encoder checkpoints/glucofm_bench.pt --head checkpoints/forecast_head.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from glucofm.data.grid import GRID_LEN, align_to_grid, denormalize, normalize
from glucofm.forecast import (
    _HALF_RANGE,
    _RECENT_STEPS,
    _SLOPE_STEPS,
    HORIZON,
    ForecastHead,
)
from glucofm.model.glucofm import GlucoFM, GlucoFMConfig

MMOL_TO_MGDL = 18.016


def load_glooko_cgm(path: str):
    import pandas as pd

    df = pd.read_csv(path, skiprows=1)
    ts = pd.to_datetime(df["Timestamp"])
    col = next(c for c in df.columns if "Glucose" in c)
    v = df[col].astype(float).to_numpy()
    if "mmol" in col.lower():
        v = v * MMOL_TO_MGDL
    return ts, v


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--encoder", default="checkpoints/glucofm_bench.pt")
    ap.add_argument("--head", default="checkpoints/forecast_head.pt")
    ap.add_argument("--results", default=None, help="optional forecast.json with test RMSE for error bars")
    args = ap.parse_args()

    ts, mgdl = load_glooko_cgm(args.csv)
    order = np.argsort(ts.to_numpy())
    ts, mgdl = ts.iloc[order], mgdl[order]

    last = ts.iloc[-1]
    today = ts.dt.normalize() == last.normalize()
    minutes = (ts.dt.hour * 60 + ts.dt.minute + ts.dt.second / 60).to_numpy()[today]
    grid, mask = align_to_grid(minutes, mgdl[today.to_numpy()])
    values = normalize(grid, mask)

    t = int(np.nonzero(mask)[0].max())
    if not mask[t - _SLOPE_STEPS : t + 1].all():
        raise SystemExit("need an unbroken last 15 minutes of readings to forecast")

    ck = torch.load(args.encoder, map_location="cpu", weights_only=False)
    model = GlucoFM(GlucoFMConfig(**ck["config"]))
    model.load_state_dict(ck["model"])
    model.eval()
    hd = torch.load(args.head, map_location="cpu", weights_only=False)
    head = ForecastHead(hd["in_dim"])
    head.load_state_dict(hd["head"])
    head.eval()

    with torch.no_grad():
        pooled = model(
            torch.from_numpy(values[None]).float(), torch.from_numpy(mask[None]).float()
        ).numpy()[0]
    v_t = values[t]
    slope = (values[t] - values[t - _SLOPE_STEPS]) / _SLOPE_STEPS
    phase = 2 * np.pi * t / GRID_LEN
    recent = np.array(
        [(values[t - k] - v_t) * mask[t - k] for k in range(_RECENT_STEPS - 1, -1, -1)]
    )
    x = np.concatenate(
        [pooled, [v_t, slope, np.sin(phase), np.cos(phase)], recent]
    ).astype(np.float32)
    with torch.no_grad():
        deltas = head(torch.from_numpy(x[None])).numpy()[0]

    now_mgdl = float(denormalize(np.asarray(v_t)))
    rmse = None
    if args.results and Path(args.results).exists():
        rmse = json.loads(Path(args.results).read_text()).get("model_rmse_30min")

    print(f"last reading: {last} -> {now_mgdl:.0f} mg/dL ({now_mgdl/MMOL_TO_MGDL:.1f} mmol/L)")
    for k in range(HORIZON):
        pred = now_mgdl + deltas[k] * _HALF_RANGE
        line = f"  +{(k+1)*5:>2d} min: {pred:.0f} mg/dL ({pred/MMOL_TO_MGDL:.1f} mmol/L)"
        if rmse is not None and k == HORIZON - 1:
            line += f"  [typical error at 30 min: +/-{rmse:.0f} mg/dL]"
        print(line)
    print("note: research prototype - not for treatment decisions")


if __name__ == "__main__":
    main()
