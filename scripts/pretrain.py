#!/usr/bin/env python3
"""Pretrain GlucoFM and run subject-disjoint linear-probe evaluation.

Data sources (--data):
  shanghai   ShanghaiT1DM/T2DM Excel cohort (pass --data-root); real data,
             downstream task = T1DM vs T2DM.
  bench      GlucoFM-Bench from Hugging Face (needs network); downstream
             task = T1DM-cohort vs T2DM-cohort membership.
  synthetic  Built-in physiological simulator; downstream task =
             dysglycemia label.

Example:
  python scripts/pretrain.py --data shanghai --data-root /path/to/shanghai \
      --epochs 40 --out checkpoints/glucofm.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from glucofm.eval.probe import linear_probe_auc
from glucofm.model.glucofm import GlucoFM
from glucofm.pretrain.trainer import TrainConfig, pretrain


def load_data(args) -> dict[str, np.ndarray]:
    if args.data == "shanghai":
        from glucofm.data.shanghai import load_shanghai_cohort

        if not args.data_root:
            raise SystemExit("--data-root is required for --data shanghai")
        return load_shanghai_cohort(args.data_root)
    if args.data == "bench":
        from glucofm.data.glucofm_bench import load_glucofm_bench

        c = load_glucofm_bench("train")
        names = c["dataset_name"]
        is_t1 = np.char.find(np.char.lower(names.astype(str)), "t1") >= 0
        is_t2 = np.char.find(np.char.lower(names.astype(str)), "t2") >= 0
        c["label"] = np.where(is_t1, 1, np.where(is_t2, 0, -1))
        return c
    from glucofm.data.synthetic import SyntheticCGMConfig, generate_cohort

    return generate_cohort(SyntheticCGMConfig(n_subjects=60, days_per_subject=8))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", choices=["shanghai", "bench", "synthetic"], default="synthetic")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="checkpoints/glucofm.pt")
    ap.add_argument("--results", default="results/pretrain.json")
    args = ap.parse_args()

    cohort = load_data(args)
    n_days, _ = cohort["values"].shape
    hours = float(cohort["mask"].sum()) * 5 / 60
    print(f"cohort: {n_days} days, {len(np.unique(cohort['subject']))} subjects, "
          f"{hours:.0f} observed CGM hours")

    model = GlucoFM()
    print(f"encoder parameters: {model.num_parameters():,}")

    history = pretrain(
        model,
        cohort["values"],
        cohort["mask"],
        TrainConfig(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, seed=args.seed),
        checkpoint_path=args.out,
    )

    results = {
        "days": n_days,
        "subjects": int(len(np.unique(cohort["subject"]))),
        "observed_hours": hours,
        "encoder_parameters": model.num_parameters(),
        "final_loss": float(np.mean(history["loss"][-10:])),
        "initial_loss": float(np.mean(history["loss"][:10])),
        "learned_sigma": float(model.decompose.sigma.detach()),
    }

    if "label" in cohort:
        keep = cohort["label"] != -1
        if keep.all() or keep.sum() >= 4:
            sub_keep = np.isin(cohort["subject"], np.nonzero(keep)[0])
            aucs = [
                linear_probe_auc(
                    model,
                    cohort["values"][sub_keep],
                    cohort["mask"][sub_keep],
                    cohort["subject"][sub_keep],
                    cohort["label"],
                    seed=s,
                )
                for s in range(5)
            ]
            results["probe_auc_mean"] = float(np.mean(aucs))
            results["probe_auc_std"] = float(np.std(aucs))
            print(f"linear probe ROC-AUC over 5 splits: "
                  f"{results['probe_auc_mean']:.3f} +/- {results['probe_auc_std']:.3f}")

    out = Path(args.results)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
