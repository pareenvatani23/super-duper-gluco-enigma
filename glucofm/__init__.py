"""Independent open reimplementation of GlucoFM (arXiv:2605.30865).

GlucoFM is a dual-stream foundation model for continuous glucose monitoring
(CGM) introduced by Google Research and UNSW Sydney. This package rebuilds the
method from the paper's public description:

- irregular CGM recordings aligned to a fixed 24-hour grid at 5-minute
  resolution (288 positions) with observation masks,
- decomposition into a slow "state" stream and a transient "event" stream via
  a Gaussian filter with a learnable bandwidth,
- tokenization into 24 one-hour patches fused into 128-dimensional tokens with
  circular time-of-day features,
- JEPA-style pretraining: masked contextual latent prediction against an EMA
  target encoder plus next-patch temporal-dynamics prediction, with CGM-aware
  augmentations.

The official code has not been released; nothing here is derived from Google
source code.
"""

from glucofm.model.glucofm import GlucoFM, GlucoFMConfig

__all__ = ["GlucoFM", "GlucoFMConfig"]
