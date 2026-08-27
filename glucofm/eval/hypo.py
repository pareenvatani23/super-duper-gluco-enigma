"""Next-day hypoglycemia prediction from frozen daily embeddings.

A clinically-posed downstream task in the spirit of the paper's metabolic
prediction suite: embed day *t* with the frozen encoder and predict whether
the subject's next recorded day contains any observed reading below
70 mg/dL. Evaluation is subject-disjoint. A "persistence" baseline (predict
tomorrow's label with today's) is reported for context.

Assumes each subject's days appear contiguously and in chronological order
in the cohort arrays, which the Shanghai and GlucoFM-Bench loaders provide.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from glucofm.data.grid import GLUCOSE_MAX, GLUCOSE_MIN
from glucofm.eval.probe import roc_auc
from glucofm.model.glucofm import GlucoFM

HYPO_MGDL = 70.0


def day_hypo_labels(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """(N,) 1 if a day contains any observed reading below 70 mg/dL."""
    center = (GLUCOSE_MAX + GLUCOSE_MIN) / 2.0
    half = (GLUCOSE_MAX - GLUCOSE_MIN) / 2.0
    thresh = (HYPO_MGDL - center) / half
    return (((values < thresh) & (mask > 0)).any(axis=1)).astype(np.int64)


def next_day_hypo_auc(
    model: GlucoFM,
    values: np.ndarray,
    mask: np.ndarray,
    subject: np.ndarray,
    train_frac: float = 0.6,
    seed: int = 0,
    epochs: int = 300,
    lr: float = 0.05,
) -> dict[str, float]:
    """Probe day-t embeddings for day-t+1 hypoglycemia; subject-disjoint AUC."""
    model.eval()
    with torch.no_grad():
        v = torch.from_numpy(values).float()
        m = torch.from_numpy(mask).float()
        embs = []
        for i in range(0, v.shape[0], 256):
            embs.append(model(v[i : i + 256], m[i : i + 256]))
        day_emb = torch.cat(embs).numpy()

    hypo = day_hypo_labels(values, mask)

    # Build (day t -> label of day t+1) pairs within each subject.
    pair_x, pair_y, pair_subj, pair_persist = [], [], [], []
    for s in np.unique(subject):
        idx = np.nonzero(subject == s)[0]
        for a, b in zip(idx[:-1], idx[1:]):
            pair_x.append(day_emb[a])
            pair_y.append(hypo[b])
            pair_subj.append(s)
            pair_persist.append(hypo[a])
    x = np.stack(pair_x)
    y = np.asarray(pair_y)
    subj = np.asarray(pair_subj)
    persist = np.asarray(pair_persist, dtype=np.float64)

    # Stratify subjects by whether they ever have a positive next-day label.
    rng = np.random.default_rng(seed)
    subj_ids = np.unique(subj)
    subj_has_pos = np.array([y[subj == s].max() for s in subj_ids])
    train_subj: list[int] = []
    for cls in np.unique(subj_has_pos):
        cls_ids = rng.permutation(subj_ids[subj_has_pos == cls])
        n_train = min(len(cls_ids) - 1, max(1, int(train_frac * len(cls_ids))))
        train_subj.extend(cls_ids[:n_train].tolist())
    train_sel = np.isin(subj, train_subj)
    test_sel = ~train_sel
    if y[test_sel].min() == y[test_sel].max():  # degenerate split: reshuffle
        return next_day_hypo_auc(model, values, mask, subject, train_frac, seed + 100, epochs, lr)

    mean = x[train_sel].mean(axis=0)
    std = x[train_sel].std(axis=0) + 1e-6
    xt = torch.from_numpy((x - mean) / std).float()
    yt = torch.from_numpy(y).float()

    torch.manual_seed(seed)
    w = torch.zeros(xt.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=lr)
    pos_weight = torch.tensor(
        float((y[train_sel] == 0).sum() / max(1, (y[train_sel] == 1).sum()))
    )
    for _ in range(epochs):
        logits = xt[train_sel] @ w + b
        loss = F.binary_cross_entropy_with_logits(
            logits, yt[train_sel], pos_weight=pos_weight
        ) + 1e-3 * w.pow(2).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        scores = (xt[test_sel] @ w + b).numpy()
    return {
        "auc": roc_auc(y[test_sel], scores),
        "persistence_auc": roc_auc(y[test_sel], persist[test_sel]),
        "n_pairs_test": int(test_sel.sum()),
        "positive_rate": float(y.mean()),
    }
