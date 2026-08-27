import numpy as np

from glucofm.data.synthetic import SyntheticCGMConfig, generate_cohort
from glucofm.eval.features import FEATURE_NAMES, day_features
from glucofm.eval.probe import matrix_probe_auc


def test_day_features_finite_and_shaped():
    c = generate_cohort(SyntheticCGMConfig(n_subjects=8, days_per_subject=3, seed=0))
    f = day_features(c["values"], c["mask"])
    assert f.shape == (c["values"].shape[0], len(FEATURE_NAMES))
    assert np.isfinite(f).all()
    # mean glucose lands in a plausible mg/dL band
    assert 60 < f[:, 0].mean() < 250


def test_day_features_work_on_sparse_15min_sensor():
    c = generate_cohort(
        SyntheticCGMConfig(n_subjects=6, days_per_subject=2, sampling_minutes=15, seed=1)
    )
    f = day_features(c["values"], c["mask"])
    assert np.isfinite(f).all()
    assert f[:, FEATURE_NAMES.index("mean_abs_delta")].std() > 0


def test_feature_probe_recovers_synthetic_label():
    c = generate_cohort(SyntheticCGMConfig(n_subjects=24, days_per_subject=4, seed=2))
    f = day_features(c["values"], c["mask"])
    aucs = [matrix_probe_auc(f, c["subject"], c["label"], seed=s) for s in range(3)]
    assert np.mean(aucs) >= 0.8
