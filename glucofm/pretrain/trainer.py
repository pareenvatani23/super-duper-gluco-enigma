"""Simple full-batch-shuffled pretraining loop with checkpointing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from glucofm.model.glucofm import GlucoFM
from glucofm.pretrain.jepa import JEPAConfig, JEPAPretrainer


@dataclass
class TrainConfig:
    epochs: int = 20
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 0.01
    warmup_frac: float = 0.1
    seed: int = 0
    log_every: int = 20


def pretrain(
    model: GlucoFM,
    values: np.ndarray,
    mask: np.ndarray,
    cfg: TrainConfig | None = None,
    jepa_cfg: JEPAConfig | None = None,
    checkpoint_path: str | Path | None = None,
) -> dict[str, list[float]]:
    """Pretrain ``model`` in place on (N, 288) values/mask arrays.

    Returns a history dict with per-step total/context/dynamics losses.
    """
    cfg = cfg or TrainConfig()
    torch.manual_seed(cfg.seed)

    pretrainer = JEPAPretrainer(model, jepa_cfg)
    opt = torch.optim.AdamW(
        [p for p in pretrainer.parameters() if p.requires_grad],
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    v = torch.from_numpy(values).float()
    m = torch.from_numpy(mask).float()
    n = v.shape[0]
    steps_per_epoch = max(1, (n + cfg.batch_size - 1) // cfg.batch_size)
    total_steps = cfg.epochs * steps_per_epoch
    warmup = max(1, int(cfg.warmup_frac * total_steps))

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        t = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + np.cos(np.pi * t))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    history: dict[str, list[float]] = {"loss": [], "context_loss": [], "dynamics_loss": []}
    rng = np.random.default_rng(cfg.seed)
    step = 0
    pretrainer.train()
    for epoch in range(cfg.epochs):
        order = rng.permutation(n)
        for i in range(0, n, cfg.batch_size):
            idx = order[i : i + cfg.batch_size]
            losses = pretrainer(v[idx], m[idx])
            opt.zero_grad(set_to_none=True)
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(pretrainer.parameters(), 1.0)
            opt.step()
            sched.step()
            pretrainer.update_target()

            for k in history:
                history[k].append(float(losses[k].detach()))
            if step % cfg.log_every == 0:
                print(
                    f"epoch {epoch} step {step}: loss={history['loss'][-1]:.4f} "
                    f"ctx={history['context_loss'][-1]:.4f} "
                    f"dyn={history['dynamics_loss'][-1]:.4f} "
                    f"sigma={float(model.decompose.sigma.detach()):.2f}"
                )
            step += 1

    if checkpoint_path is not None:
        path = Path(checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"model": model.state_dict(), "config": model.cfg.__dict__}, path
        )
        print(f"saved checkpoint to {path}")
    return history
