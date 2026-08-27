import numpy as np
import torch

from glucofm.data.synthetic import SyntheticCGMConfig, generate_cohort
from glucofm.model.glucofm import GlucoFM, GlucoFMConfig
from glucofm.pretrain.jepa import JEPAConfig, JEPAPretrainer
from glucofm.pretrain.trainer import TrainConfig, pretrain


def _small_model() -> GlucoFM:
    return GlucoFM(
        GlucoFMConfig(
            stream_dim=32, stream_layers=1, stream_ff=64,
            fused_dim=64, fusion_layers=1, fusion_ff=128,
        )
    )


def _batch(n_subjects=6, days=2, seed=0):
    c = generate_cohort(SyntheticCGMConfig(n_subjects=n_subjects, days_per_subject=days, seed=seed))
    return torch.from_numpy(c["values"]), torch.from_numpy(c["mask"])


def test_mask_ratio_in_paper_range():
    torch.manual_seed(0)
    p = JEPAPretrainer(_small_model())
    pm = p._sample_patch_mask(64, torch.device("cpu"))
    ratios = pm.float().mean(dim=1)
    # per-sample ratios drawn from U(0.5, 0.6); 24 patches quantize to ~1/24
    assert ratios.min() >= 0.5 - 1 / 24
    assert ratios.max() <= 0.6 + 1 / 24


def test_losses_are_finite_and_backprop():
    torch.manual_seed(0)
    v, m = _batch()
    p = JEPAPretrainer(_small_model())
    out = p(v, m)
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    grads = [q.grad for q in p.online.parameters() if q.grad is not None]
    assert len(grads) > 0
    assert all(torch.isfinite(g).all() for g in grads)


def test_target_encoder_gets_no_gradient():
    torch.manual_seed(0)
    v, m = _batch()
    p = JEPAPretrainer(_small_model())
    p(v, m)["loss"].backward()
    assert all(q.grad is None for q in p.target.parameters())


def test_ema_update_moves_target_toward_online():
    torch.manual_seed(0)
    p = JEPAPretrainer(_small_model(), JEPAConfig(ema_momentum=0.9))
    with torch.no_grad():
        for q in p.online.parameters():
            q.add_(1.0)
    before = [q.clone() for q in p.target.parameters()]
    p.update_target()
    moved = [
        (a - b).abs().max() for a, b in zip(p.target.parameters(), before)
    ]
    assert max(float(x) for x in moved) > 0


def test_pretraining_reduces_loss():
    """The core 'is it learning' check: JEPA loss must drop substantially."""
    c = generate_cohort(SyntheticCGMConfig(n_subjects=12, days_per_subject=4, seed=5))
    model = _small_model()
    hist = pretrain(
        model,
        c["values"],
        c["mask"],
        TrainConfig(epochs=12, batch_size=16, lr=2e-3, log_every=1000),
    )
    first = np.mean(hist["loss"][:3])
    last = np.mean(hist["loss"][-3:])
    assert last < 0.7 * first, f"loss did not decrease enough: {first:.4f} -> {last:.4f}"


def test_learned_sigma_stays_in_bounds_after_training():
    c = generate_cohort(SyntheticCGMConfig(n_subjects=6, days_per_subject=2, seed=6))
    model = _small_model()
    pretrain(model, c["values"], c["mask"], TrainConfig(epochs=2, batch_size=8, log_every=1000))
    s = float(model.decompose.sigma)
    assert 2.0 <= s <= 12.0
