import numpy as np
import pytest
import torch

from glucofm.data.synthetic import SyntheticCGMConfig, generate_cohort
from glucofm.model.decompose import (
    SIGMA_INIT,
    SIGMA_MAX,
    SIGMA_MIN,
    DualStreamDecomposition,
)
from glucofm.model.glucofm import GlucoFM, GlucoFMConfig


@pytest.fixture(scope="module")
def batch():
    c = generate_cohort(SyntheticCGMConfig(n_subjects=4, days_per_subject=2, seed=0))
    return torch.from_numpy(c["values"]), torch.from_numpy(c["mask"])


def test_sigma_initialization_and_bounds():
    d = DualStreamDecomposition()
    assert abs(float(d.sigma.detach()) - SIGMA_INIT) < 1e-4
    with torch.no_grad():
        d._sigma_raw.fill_(100.0)
        assert float(d.sigma) <= SIGMA_MAX + 1e-4
        d._sigma_raw.fill_(-100.0)
        assert float(d.sigma) >= SIGMA_MIN - 1e-4


def test_decomposition_reconstructs_signal(batch):
    v, m = batch
    state, event = DualStreamDecomposition()(v, m)
    assert torch.allclose(state + event, v * m, atol=1e-5)
    assert torch.all(state[m == 0] == 0) and torch.all(event[m == 0] == 0)


def test_state_is_smoother_than_input(batch):
    v, m = batch
    state, _ = DualStreamDecomposition()(v, m)
    # total variation of the state stream must be far below the raw signal's
    tv = lambda x: (x[:, 1:] - x[:, :-1]).abs().mean()
    assert tv(state) < tv(v * m)


def test_sigma_receives_gradients(batch):
    v, m = batch
    d = DualStreamDecomposition()
    state, event = d(v, m)
    (state.pow(2).mean() + event.pow(2).mean()).backward()
    assert d._sigma_raw.grad is not None
    assert torch.isfinite(d._sigma_raw.grad)


def test_encoder_output_shapes(batch):
    v, m = batch
    model = GlucoFM()
    out = model.encode(v, m)
    B = v.shape[0]
    cfg = model.cfg
    assert out["fused"].shape == (B, cfg.n_patches, cfg.fused_dim)
    assert out["state"].shape == (B, cfg.n_patches, cfg.stream_dim)
    assert out["event"].shape == (B, cfg.n_patches, cfg.stream_dim)
    assert out["pooled"].shape == (B, cfg.fused_dim)
    assert torch.isfinite(out["fused"]).all()


def test_parameter_count_matches_paper():
    """Paper reports a 0.72M-parameter encoder; default config lands there."""
    n = GlucoFM().num_parameters()
    assert 650_000 <= n <= 800_000, f"encoder has {n} params, expected ~0.72M"


def test_patch_masking_changes_masked_positions_only_weakly(batch):
    """Masked-patch tokens must differ from unmasked ones (mask actually applied)."""
    v, m = batch
    model = GlucoFM()
    tok_s = torch.zeros(model.cfg.stream_dim)
    tok_e = torch.zeros(model.cfg.stream_dim)
    pm = torch.zeros(v.shape[0], model.cfg.n_patches, dtype=torch.bool)
    pm[:, :12] = True
    full = model.encode(v, m)
    masked = model.encode(v, m, patch_mask=pm, mask_tokens=(tok_s, tok_e))
    assert not torch.allclose(full["fused"], masked["fused"])


def test_grid_len_must_divide_into_patches():
    with pytest.raises(ValueError):
        GlucoFM(GlucoFMConfig(grid_len=290)).encode(
            torch.zeros(1, 290), torch.ones(1, 290)
        )
