# Results and verification log

Two pretraining runs were performed, both CPU-only:

1. **ShanghaiT1DM/T2DM** (local, Zhao et al. 2023, CC BY 4.0): 1,315
   patient-days from 112 subjects (~10,240 observed CGM hours).
2. **GlucoFM-Bench** (on a GitHub Actions runner, since Hugging Face is
   unreachable from the dev sandbox): the aggregated 12-cohort benchmark —
   **8,370 patient-days from 529 subjects (~191,450 observed CGM hours)**,
   which actually exceeds the paper's 109,066-hour pretraining corpus
   (the bench data is interpolation-densified to 5-minute cadence, so its
   observed-hour count runs high).

## Headline: scaling the data works

| Probe (subject-disjoint, frozen encoder) | Shanghai (10k h) | GlucoFM-Bench (191k h) |
| --- | --- | --- |
| T1DM vs T2DM cohort | 0.56 ± 0.13 | **0.90 ± 0.01** |
| Next-day hypoglycemia | 0.67 ± 0.04 | **0.71 ± 0.03** |
| Next-day persistence baseline | 0.72 (not beaten) | 0.68 (**beaten**) |

On the larger corpus the learned bandwidth settled at σ ≈ 9.6 grid steps
(~48 min) versus 6.3 on Shanghai — with more heterogeneous cohorts the
model prefers a wider state/event split. Final JEPA loss: 2.37 → 0.58.
The bench checkpoint is committed at `checkpoints/glucofm_bench.pt`
(also archived as a run artifact on the training workflow run).

## Does the implementation match the paper?

| Paper spec | This repo | Status |
| --- | --- | --- |
| 24-hour grid, 5-min cells (288 positions), observation masks | `glucofm/data/grid.py` | ✅ tested |
| Learnable Gaussian bandwidth σ ∈ [2, 12] steps, init 6.0 | σ constrained via sigmoid; init verified | ✅ tested |
| 24 one-hour patches, 128-dim fused tokens, circular time-of-day features | `glucofm/model/glucofm.py` | ✅ tested |
| ~0.72M-parameter encoder | 716,545 parameters | ✅ tested |
| Mask ratio sampled from U(0.5, 0.6) | verified in tests | ✅ tested |
| EMA target encoder, latent (non-reconstruction) objectives | smooth-L1 on layer-normed latents | ✅ tested |
| Masked contextual prediction + temporal dynamics prediction | both objectives implemented | ✅ tested |
| CGM-aware augmentations (value perturbation, sampling rates, dropout) | `glucofm/data/augment.py` | ✅ tested |
| Subject-disjoint linear probing | stratified subject splits | ✅ tested |

The full pytest suite (23 tests) covers all of the above plus training
behavior: loss decrease, gradient flow, EMA update direction, no gradient
into the target encoder, σ staying in bounds, and a latent-collapse
regression test.

## Pretraining on real data (60 epochs)

- JEPA total loss: **2.23 → 0.75**; the masked-context term settles around
  0.33 rather than collapsing toward zero (see below).
- Learned bandwidth σ: 6.00 → **6.29** grid steps (~31 minutes) — it moves,
  stays in bounds, and settles near its physiologically sensible init.
- Embedding health: pooled per-dimension std **0.96**, covariance effective
  rank **60.6 / 128**.

## Downstream probes (frozen encoder, subject-disjoint, 5 splits)

| Task | AUC | Reference point |
| --- | --- | --- |
| Same-day hypoglycemia decodability | **0.79** | 0.5 chance |
| Next-day hypoglycemia | **0.67 ± 0.04** | persistence baseline 0.72 |
| T1DM vs T2DM (subject level) | **0.56 ± 0.13** | 0.5 chance; only 12 T1DM subjects |
| Dysglycemia (synthetic cohort, CI test) | **≥ 0.80** (test-enforced) | 0.5 chance |

## The main engineering finding: latent collapse at small scale

The first full pretraining run converged nicely (loss −80%) but produced
embeddings with an **effective rank of 1.8 out of 128** — the classic JEPA
failure mode where masked patches are predictable from time-of-day position
alone, so the encoder never needs to encode the day's content. Notably the
encoder is position-dominated *already at initialization*, so the objective
simply never had a reason to leave that basin. Every downstream probe was
near chance.

Fix (documented deviation from the paper, which does not describe its
anti-collapse mechanism beyond the EMA target encoder): a VICReg-style
regularizer — a variance hinge computed **across days at each patch
position** plus covariance decorrelation of the pooled embedding — applied
to a clean (un-masked) online forward, since the masked context branch's
variance is capped by construction (half its tokens are the same learnable
mask token). The temporal-dynamics objective also moved to the clean
forward. Effect: effective rank 1.8 → 60.6, same-day hypo decodability
0.65 → 0.79, next-day hypo 0.56 → 0.67.

## Honest limitations

- With ~1/10 of the paper's pretraining data and CPU-scale training, the
  frozen embeddings do not yet beat the persistence baseline (0.72) on
  next-day hypoglycemia; combining embeddings with persistence does not
  help either (0.68). More data (e.g. the full GlucoFM-Bench aggregation)
  is the first thing to try.
- T1DM vs T2DM probing is noisy with only 12 positive subjects; the
  ±0.13 spread across splits reflects that, not model instability.
- Hyperparameters the paper does not disclose (optimizer schedule, exact
  head shapes, augmentation magnitudes, loss weights) were chosen by
  standard practice and are marked in the code.
- This is a research prototype for retrospective analysis, not a medical
  device.

## Reproduce

```bash
pytest tests/ -v
python scripts/pretrain.py --data shanghai --data-root <shanghai_dir> \
    --epochs 60 --out checkpoints/glucofm.pt --results results/shanghai.json
python scripts/evaluate.py --checkpoint checkpoints/glucofm.pt \
    --data shanghai --data-root <shanghai_dir> --results results/eval.json
```

Machine-written summaries of the runs behind this document are in
`results/`.
