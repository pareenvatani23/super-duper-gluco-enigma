#!/usr/bin/env python3
"""Evaluate a pretrained GlucoFM checkpoint on downstream probes.

Example:
  python scripts/evaluate.py --checkpoint checkpoints/glucofm.pt \
      --data shanghai --data-root /path/to/shanghai --results results/eval.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from glucofm.eval.hypo import next_day_hypo_auc
from glucofm.eval.probe import linear_probe_auc
from glucofm.model.glucofm import GlucoFM, GlucoFMConfig


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data", choices=["shanghai", "synthetic"], default="shanghai")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--results", default="results/eval.json")
    args = ap.parse_args()

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = GlucoFM(GlucoFMConfig(**ck["config"]))
    model.load_state_dict(ck["model"])
    model.eval()

    if args.data == "shanghai":
        from glucofm.data.shanghai import load_shanghai_cohort

        cohort = load_shanghai_cohort(args.data_root)
    else:
        from glucofm.data.synthetic import SyntheticCGMConfig, generate_cohort

        cohort = generate_cohort(SyntheticCGMConfig(n_subjects=60, days_per_subject=8))

    results: dict = {}

    aucs = [
        linear_probe_auc(
            model, cohort["values"], cohort["mask"], cohort["subject"], cohort["label"], seed=s
        )
        for s in range(5)
    ]
    results["subject_label_probe"] = {
        "auc_mean": float(np.mean(aucs)),
        "auc_std": float(np.std(aucs)),
    }
    print(f"subject-label probe AUC: {np.mean(aucs):.3f} +/- {np.std(aucs):.3f}")

    hypo = [
        next_day_hypo_auc(model, cohort["values"], cohort["mask"], cohort["subject"], seed=s)
        for s in range(5)
    ]
    results["next_day_hypo"] = {
        "auc_mean": float(np.mean([h["auc"] for h in hypo])),
        "auc_std": float(np.std([h["auc"] for h in hypo])),
        "persistence_auc_mean": float(np.mean([h["persistence_auc"] for h in hypo])),
        "positive_rate": hypo[0]["positive_rate"],
    }
    print(
        f"next-day hypoglycemia probe AUC: {results['next_day_hypo']['auc_mean']:.3f} "
        f"+/- {results['next_day_hypo']['auc_std']:.3f} "
        f"(persistence baseline {results['next_day_hypo']['persistence_auc_mean']:.3f}, "
        f"positive rate {results['next_day_hypo']['positive_rate']:.2f})"
    )

    out = Path(args.results)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
