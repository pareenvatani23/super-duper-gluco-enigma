#!/usr/bin/env python3
"""Train and evaluate the 30-minute forecasting head.

Uses a frozen pretrained encoder; trains the head on GlucoFM-Bench's train
split and evaluates on its participant-disjoint test split (or synthetic
data for smoke runs).

Example:
  python scripts/forecast_train.py --encoder checkpoints/glucofm_bench.pt \
      --data bench --out checkpoints/forecast_head.pt --results results/forecast.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from glucofm.forecast import (
    evaluate_forecast,
    forecast_features,
    sample_forecast_points,
    train_head,
)
from glucofm.model.glucofm import GlucoFM, GlucoFMConfig


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--data", choices=["bench", "synthetic"], default="bench")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--samples-per-day", type=int, default=12)
    ap.add_argument("--out", default="checkpoints/forecast_head.pt")
    ap.add_argument("--results", default="results/forecast.json")
    args = ap.parse_args()

    ck = torch.load(args.encoder, map_location="cpu", weights_only=False)
    model = GlucoFM(GlucoFMConfig(**ck["config"]))
    model.load_state_dict(ck["model"])
    model.eval()

    if args.data == "bench":
        from glucofm.data.glucofm_bench import load_glucofm_bench

        train_c = load_glucofm_bench("train")
        test_c = load_glucofm_bench("test")
    else:
        from glucofm.data.synthetic import SyntheticCGMConfig, generate_cohort

        train_c = generate_cohort(SyntheticCGMConfig(n_subjects=30, days_per_subject=6, seed=0))
        test_c = generate_cohort(SyntheticCGMConfig(n_subjects=10, days_per_subject=6, seed=1))

    tr_s = sample_forecast_points(train_c["mask"], n_per_day=args.samples_per_day, seed=0)
    te_s = sample_forecast_points(test_c["mask"], n_per_day=args.samples_per_day, seed=1)
    print(f"forecast samples: train {len(tr_s.day_idx)}, test {len(te_s.day_idx)}")

    x_tr, y_tr, _ = forecast_features(model, train_c["values"], train_c["mask"], tr_s)
    x_te, y_te, slope_te = forecast_features(model, test_c["values"], test_c["mask"], te_s)

    head = train_head(x_tr, y_tr, epochs=args.epochs)
    results = evaluate_forecast(head, x_te, y_te, slope_te)
    for k, v in results.items():
        print(f"{k}: {v:.2f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"head": head.state_dict(), "in_dim": x_tr.shape[1]}, out)
    rp = Path(args.results)
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text(json.dumps(results, indent=2))
    print(f"saved {out} and {rp}")


if __name__ == "__main__":
    main()
