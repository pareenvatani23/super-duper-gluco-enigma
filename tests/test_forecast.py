import numpy as np
import torch

from glucofm.data.synthetic import SyntheticCGMConfig, generate_cohort
from glucofm.forecast import (
    AUX_DIM,
    HORIZON,
    evaluate_forecast,
    forecast_features,
    sample_forecast_points,
    train_head,
)
from glucofm.model.glucofm import GlucoFM, GlucoFMConfig


def _small_model():
    return GlucoFM(
        GlucoFMConfig(
            stream_dim=32, stream_layers=1, stream_ff=64,
            fused_dim=64, fusion_layers=1, fusion_ff=128,
        )
    )


def test_sample_points_respect_constraints():
    c = generate_cohort(SyntheticCGMConfig(n_subjects=6, days_per_subject=2, seed=0))
    s = sample_forecast_points(c["mask"], n_per_day=4, seed=0)
    assert len(s.day_idx) > 0
    for d, t in zip(s.day_idx[:50], s.t_idx[:50]):
        assert c["mask"][d, t - 3 : t + 1].all()
        assert c["mask"][d, t + 1 : t + 1 + HORIZON].all()


def test_features_and_targets_shapes():
    c = generate_cohort(SyntheticCGMConfig(n_subjects=4, days_per_subject=2, seed=1))
    model = _small_model()
    s = sample_forecast_points(c["mask"], n_per_day=3, seed=0)
    x, y, slope = forecast_features(model, c["values"], c["mask"], s)
    assert x.shape == (len(s.day_idx), model.cfg.fused_dim + AUX_DIM)
    assert y.shape == (len(s.day_idx), HORIZON)
    assert slope.shape == (len(s.day_idx),)
    assert np.isfinite(x).all() and np.isfinite(y).all()


def test_trained_head_beats_persistence_on_synthetic():
    torch.manual_seed(0)
    tr = generate_cohort(SyntheticCGMConfig(n_subjects=20, days_per_subject=4, seed=2))
    te = generate_cohort(SyntheticCGMConfig(n_subjects=8, days_per_subject=4, seed=3))
    model = _small_model()
    s_tr = sample_forecast_points(tr["mask"], n_per_day=6, seed=0)
    s_te = sample_forecast_points(te["mask"], n_per_day=6, seed=1)
    x_tr, y_tr, _ = forecast_features(model, tr["values"], tr["mask"], s_tr)
    x_te, y_te, slope_te = forecast_features(model, te["values"], te["mask"], s_te)
    head = train_head(x_tr, y_tr, epochs=30)
    res = evaluate_forecast(head, x_te, y_te, slope_te)
    assert res["model_rmse_30min"] < res["persistence_rmse_30min"]
