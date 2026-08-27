# GlucoFM — open reimplementation

An independent, from-scratch reimplementation of **GlucoFM: A Dual-Stream
Foundation Model for Continuous Glucose Monitoring**
([arXiv:2605.30865](https://arxiv.org/abs/2605.30865), Google Research & UNSW
Sydney, announced August 2026). The official code and weights have not been
released; this package rebuilds the method from the paper's public
description. Nothing here derives from Google source code, and this is a
research prototype — not a medical device.

## What it implements

- **24-hour grid alignment** — irregular CGM recordings mapped to a fixed
  5-minute grid (288 cells/day) with observation masks
  (`glucofm/data/grid.py`).
- **Dual-stream decomposition** — a mask-aware Gaussian filter with a
  *learnable* bandwidth (σ ∈ [2, 12] grid steps ≈ 10–60 min, init 6.0) splits
  each day into a slow "state" stream and a transient "event" stream
  (`glucofm/model/decompose.py`).
- **~0.72M-parameter encoder** — per-stream transformers over 24 one-hour
  patches with circular time-of-day features, fused into 128-dim tokens and
  encoded by a fusion transformer (`glucofm/model/glucofm.py`; the default
  config lands at 716,545 parameters, matching the paper's reported 0.72M).
- **JEPA pretraining** — masked contextual latent prediction (mask ratio
  U(0.5, 0.6), learnable mask tokens, EMA target encoder) plus next-patch
  temporal-dynamics prediction, with CGM-aware augmentations (value jitter,
  heterogeneous sampling rates, sensor dropout) (`glucofm/pretrain/`).
- **Subject-disjoint linear probing** — the paper's downstream evaluation
  protocol (`glucofm/eval/probe.py`).

## Data sources

| Source | Flag | Notes |
| --- | --- | --- |
| ShanghaiT1DM/T2DM | `--data shanghai --data-root <dir>` | Real cohort, 112 patients, 15-min CGM (Zhao et al. 2023, CC BY 4.0). Downstream task: T1DM vs T2DM. |
| GlucoFM-Bench | `--data bench` | 12 aggregated open cohorts from [Hugging Face](https://huggingface.co/datasets/glucofmbench/GlucoFM-Bench); needs network access. |
| Synthetic | `--data synthetic` | Built-in physiological simulator (meals, dawn phenomenon, AR(1) noise, dropout); used by tests/CI. |

## Usage

```bash
pip install -e .[dev]           # torch, numpy, pytest
pip install -e .[data] xlrd     # extra: pandas/pyarrow/hf-hub/Excel readers

pytest tests/ -v                # verify the implementation

python scripts/pretrain.py --data shanghai \
    --data-root /path/to/shanghai_dataset \
    --epochs 30 --out checkpoints/glucofm.pt --results results/shanghai.json
```

The script pretrains the encoder, saves a checkpoint, and reports the
subject-disjoint linear-probe ROC-AUC over five stratified splits, writing a
JSON summary to `--results`.

## Results (this repo's runs)

Pretrained on the real Shanghai cohort (1,315 days / 112 subjects /
~10,240 CGM hours, CPU): JEPA loss 2.23 → 0.75, learned σ ≈ 6.3 steps,
embedding effective rank 60.6/128; frozen-probe AUCs: same-day
hypoglycemia 0.79, next-day hypoglycemia 0.67 (persistence baseline
0.72), T1DM-vs-T2DM 0.56 ± 0.13. See `docs/RESULTS.md` for the full
verification table, the latent-collapse fix, and limitations; `results/`
holds machine-written run summaries.
