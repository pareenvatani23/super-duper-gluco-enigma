import numpy as np

from glucofm.data.synthetic import SyntheticCGMConfig, generate_cohort
from glucofm.eval.probe import linear_probe_auc, roc_auc
from glucofm.model.glucofm import GlucoFM, GlucoFMConfig
from glucofm.pretrain.trainer import TrainConfig, pretrain


def test_roc_auc_known_values():
    assert roc_auc(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert roc_auc(np.array([1, 1, 0, 0]), np.array([0.1, 0.2, 0.8, 0.9])) == 0.0
    assert abs(roc_auc(np.array([0, 1, 0, 1]), np.array([0.5, 0.5, 0.5, 0.5])) - 0.5) < 1e-9


def test_pretrained_embeddings_separate_dysglycemia():
    """End-to-end expectation check: after JEPA pretraining, a frozen linear
    probe must recover the dysglycemia label on held-out subjects far above
    chance. This is the paper's linear-probing protocol in miniature."""
    c = generate_cohort(
        SyntheticCGMConfig(n_subjects=24, days_per_subject=4, seed=7)
    )
    model = GlucoFM(
        GlucoFMConfig(
            stream_dim=32, stream_layers=1, stream_ff=64,
            fused_dim=64, fusion_layers=1, fusion_ff=128,
        )
    )
    pretrain(
        model,
        c["values"],
        c["mask"],
        TrainConfig(epochs=10, batch_size=16, lr=2e-3, log_every=1000),
    )
    auc = linear_probe_auc(
        model, c["values"], c["mask"], c["subject"], c["label"], seed=0
    )
    assert auc >= 0.8, f"probe AUC {auc:.3f} below expectation"
