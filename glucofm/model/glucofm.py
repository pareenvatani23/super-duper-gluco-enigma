"""GlucoFM encoder: dual-stream state/event transformer with token fusion.

Pipeline (per the paper's description):

  (B, 288) values + mask
    -> Gaussian dual-stream decomposition (learnable bandwidth)
    -> 24 one-hour patches per stream (12 grid steps each)
    -> per-stream patch embeddings + circular time-of-day features
    -> per-stream transformer encoders ("state" and "event")
    -> fusion into 128-dim tokens -> fusion transformer
    -> (B, 24, 128) fused daily representation

The default configuration lands at ~0.72M encoder parameters, matching the
model size reported for GlucoFM.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn

from glucofm.data.grid import GRID_LEN
from glucofm.model.decompose import DualStreamDecomposition


@dataclass
class GlucoFMConfig:
    grid_len: int = GRID_LEN  # 288
    n_patches: int = 24  # one-hour patches
    stream_dim: int = 64
    stream_layers: int = 2
    stream_ff: int = 256
    fused_dim: int = 128
    fusion_layers: int = 3
    fusion_ff: int = 384
    n_heads: int = 4
    dropout: float = 0.0

    @property
    def patch_len(self) -> int:
        if self.grid_len % self.n_patches:
            raise ValueError("grid_len must divide evenly into patches")
        return self.grid_len // self.n_patches  # 12


def _encoder(dim: int, layers: int, ff: int, heads: int, dropout: float) -> nn.TransformerEncoder:
    layer = nn.TransformerEncoderLayer(
        d_model=dim,
        nhead=heads,
        dim_feedforward=ff,
        dropout=dropout,
        activation="gelu",
        batch_first=True,
        norm_first=True,
    )
    return nn.TransformerEncoder(
        layer, num_layers=layers, norm=nn.LayerNorm(dim), enable_nested_tensor=False
    )


class _StreamBranch(nn.Module):
    """Patch-embed one stream and encode it with a small transformer."""

    def __init__(self, cfg: GlucoFMConfig):
        super().__init__()
        # Per-patch features: patch values + patch observation mask.
        self.embed = nn.Linear(2 * cfg.patch_len, cfg.stream_dim)
        self.time_proj = nn.Linear(2, cfg.stream_dim, bias=False)
        self.encoder = _encoder(
            cfg.stream_dim, cfg.stream_layers, cfg.stream_ff, cfg.n_heads, cfg.dropout
        )

    def embed_tokens(
        self, stream: torch.Tensor, mask: torch.Tensor, time_feats: torch.Tensor
    ) -> torch.Tensor:
        """(B, L) stream + mask -> (B, P, stream_dim) pre-encoder tokens."""
        B, L = stream.shape
        P = time_feats.shape[0]
        patches = torch.cat(
            [stream.view(B, P, L // P), mask.view(B, P, L // P)], dim=-1
        )
        return self.embed(patches) + self.time_proj(time_feats).unsqueeze(0)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.encoder(tokens)


class GlucoFM(nn.Module):
    def __init__(self, cfg: GlucoFMConfig | None = None):
        super().__init__()
        self.cfg = cfg or GlucoFMConfig()
        cfg = self.cfg

        self.decompose = DualStreamDecomposition()
        self.state_branch = _StreamBranch(cfg)
        self.event_branch = _StreamBranch(cfg)
        self.fuse = nn.Linear(2 * cfg.stream_dim, cfg.fused_dim)
        self.fusion_encoder = _encoder(
            cfg.fused_dim, cfg.fusion_layers, cfg.fusion_ff, cfg.n_heads, cfg.dropout
        )

        # Circular time-of-day features per one-hour patch: (sin, cos) of the
        # patch-center phase.
        phase = 2 * math.pi * (torch.arange(cfg.n_patches) + 0.5) / cfg.n_patches
        self.register_buffer(
            "time_feats", torch.stack([phase.sin(), phase.cos()], dim=-1)
        )

    def encode(
        self,
        values: torch.Tensor,
        mask: torch.Tensor,
        patch_mask: torch.Tensor | None = None,
        mask_tokens: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Encode a batch of daily grids.

        Args:
            values: (B, 288) normalized glucose (0 where unobserved).
            mask: (B, 288) observation mask.
            patch_mask: optional (B, 24) bool — True marks patches to be
                *hidden* from the encoder (JEPA context branch). Hidden
                patches are replaced by the learnable ``mask_tokens``
                (state, event) supplied by the pretrainer.
            mask_tokens: pair of (stream_dim,) learnable tokens; required
                when patch_mask is given.

        Returns dict with:
            fused:  (B, 24, fused_dim) fused tokens
            state:  (B, 24, stream_dim) encoded state-stream tokens
            event:  (B, 24, stream_dim) encoded event-stream tokens
            pooled: (B, fused_dim) masked mean over fused tokens
        """
        state, event = self.decompose(values, mask)
        s_tok = self.state_branch.embed_tokens(state, mask, self.time_feats)
        e_tok = self.event_branch.embed_tokens(event, mask, self.time_feats)

        if patch_mask is not None:
            if mask_tokens is None:
                raise ValueError("mask_tokens required when patch_mask is set")
            pm = patch_mask.unsqueeze(-1)
            s_tok = torch.where(pm, mask_tokens[0].expand_as(s_tok), s_tok)
            e_tok = torch.where(pm, mask_tokens[1].expand_as(e_tok), e_tok)

        s_enc = self.state_branch(s_tok)
        e_enc = self.event_branch(e_tok)
        fused = self.fusion_encoder(self.fuse(torch.cat([s_enc, e_enc], dim=-1)))

        # Pool over patches that contain at least one observation.
        P = self.cfg.n_patches
        patch_obs = mask.view(mask.shape[0], P, -1).amax(dim=-1)  # (B, P)
        w = patch_obs / patch_obs.sum(dim=1, keepdim=True).clamp(min=1e-6)
        pooled = (fused * w.unsqueeze(-1)).sum(dim=1)

        return {"fused": fused, "state": s_enc, "event": e_enc, "pooled": pooled}

    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Convenience: return the pooled daily embedding (B, fused_dim)."""
        return self.encode(values, mask)["pooled"]

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
