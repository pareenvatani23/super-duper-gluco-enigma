"""JEPA-style pretraining for GlucoFM.

Two complementary latent objectives (no raw-value reconstruction):

1. Masked contextual latent prediction — 50-60% of the 24 one-hour patches
   in the online (context) branch are replaced with learnable mask tokens;
   a predictor maps the context branch's fused tokens at masked positions
   to the latents produced by an EMA *target* encoder that sees the full
   (un-masked, un-augmented) day.

2. Temporal dynamics prediction — lightweight transition heads predict the
   target encoder's next-patch state and event representations from the
   online branch's current-patch stream representations.

Targets are layer-normalized and taken under stop-gradient; the target
encoder is an exponential moving average of the online encoder.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from glucofm.data.augment import CGMAugment
from glucofm.model.glucofm import GlucoFM


@dataclass
class JEPAConfig:
    mask_ratio_min: float = 0.5
    mask_ratio_max: float = 0.6
    ema_momentum: float = 0.996
    dynamics_weight: float = 0.5
    predictor_hidden: int = 128
    transition_hidden: int = 64


def _mlp(d_in: int, d_hidden: int, d_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(d_in, d_hidden), nn.GELU(), nn.Linear(d_hidden, d_out)
    )


class JEPAPretrainer(nn.Module):
    def __init__(self, model: GlucoFM, cfg: JEPAConfig | None = None):
        super().__init__()
        self.cfg = cfg or JEPAConfig()
        self.online = model
        self.target = copy.deepcopy(model)
        for p in self.target.parameters():
            p.requires_grad_(False)

        d_stream = model.cfg.stream_dim
        d_fused = model.cfg.fused_dim
        self.mask_token_state = nn.Parameter(torch.zeros(d_stream))
        self.mask_token_event = nn.Parameter(torch.zeros(d_stream))
        nn.init.normal_(self.mask_token_state, std=0.02)
        nn.init.normal_(self.mask_token_event, std=0.02)

        self.predictor = _mlp(d_fused, self.cfg.predictor_hidden, d_fused)
        self.state_transition = _mlp(d_stream, self.cfg.transition_hidden, d_stream)
        self.event_transition = _mlp(d_stream, self.cfg.transition_hidden, d_stream)

        self.augment = CGMAugment()

    @torch.no_grad()
    def update_target(self) -> None:
        m = self.cfg.ema_momentum
        for po, pt in zip(self.online.parameters(), self.target.parameters()):
            pt.mul_(m).add_(po.detach(), alpha=1.0 - m)
        for bo, bt in zip(self.online.buffers(), self.target.buffers()):
            bt.copy_(bo)

    def _sample_patch_mask(self, B: int, device: torch.device) -> torch.Tensor:
        """(B, 24) bool mask; per-sample ratio drawn from U(min, max)."""
        P = self.online.cfg.n_patches
        ratios = torch.empty(B, device=device).uniform_(
            self.cfg.mask_ratio_min, self.cfg.mask_ratio_max
        )
        n_masked = (ratios * P).round().long().clamp(1, P - 1)
        scores = torch.rand(B, P, device=device)
        order = scores.argsort(dim=1)
        patch_mask = torch.zeros(B, P, dtype=torch.bool, device=device)
        for b in range(B):
            patch_mask[b, order[b, : n_masked[b]]] = True
        return patch_mask

    def forward(
        self, values: torch.Tensor, mask: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """One pretraining step's losses for a batch of (B, 288) days."""
        B = values.shape[0]
        device = values.device

        aug_values, aug_mask = self.augment(values, mask)
        patch_mask = self._sample_patch_mask(B, device)

        online_out = self.online.encode(
            aug_values,
            aug_mask,
            patch_mask=patch_mask,
            mask_tokens=(self.mask_token_state, self.mask_token_event),
        )
        with torch.no_grad():
            target_out = self.target.encode(values, mask)
            tgt_fused = F.layer_norm(
                target_out["fused"], target_out["fused"].shape[-1:]
            )
            tgt_state = F.layer_norm(
                target_out["state"], target_out["state"].shape[-1:]
            )
            tgt_event = F.layer_norm(
                target_out["event"], target_out["event"].shape[-1:]
            )

        # Objective 1: masked contextual latent prediction.
        pred = self.predictor(online_out["fused"])
        ctx_loss = F.smooth_l1_loss(pred[patch_mask], tgt_fused[patch_mask])

        # Objective 2: next-patch temporal dynamics over both streams.
        pred_state_next = self.state_transition(online_out["state"][:, :-1])
        pred_event_next = self.event_transition(online_out["event"][:, :-1])
        dyn_loss = F.smooth_l1_loss(
            pred_state_next, tgt_state[:, 1:]
        ) + F.smooth_l1_loss(pred_event_next, tgt_event[:, 1:])

        loss = ctx_loss + self.cfg.dynamics_weight * dyn_loss
        return {"loss": loss, "context_loss": ctx_loss, "dynamics_loss": dyn_loss}
