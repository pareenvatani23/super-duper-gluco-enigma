"""Subject-disjoint linear probing, the paper's downstream evaluation protocol.

The encoder is frozen; daily embeddings are averaged per subject and a
logistic-regression probe is trained on training subjects and evaluated on
held-out subjects (subject-disjoint split), reporting ROC-AUC.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from glucofm.model.glucofm import GlucoFM


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Rank-based ROC-AUC (Mann-Whitney), no sklearn dependency."""
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=np.float64)
    pos, neg = scores[labels == 1], scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        raise ValueError("need both classes for AUC")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(order.size, dtype=np.float64)
    ranks[order] = np.arange(1, order.size + 1)
    # average ranks over ties
    combined = np.concatenate([pos, neg])
    for v in np.unique(combined):
        tie = combined == v
        if tie.sum() > 1:
            ranks[tie] = ranks[tie].mean()
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


@torch.no_grad()
def subject_embeddings(
    model: GlucoFM, values: np.ndarray, mask: np.ndarray, subject: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Frozen pooled embeddings averaged per subject.

    Returns (subject_ids, embeddings) with matching first dimension.
    """
    model.eval()
    v = torch.from_numpy(values).float()
    m = torch.from_numpy(mask).float()
    embs = []
    for i in range(0, v.shape[0], 256):
        embs.append(model(v[i : i + 256], m[i : i + 256]))
    day_emb = torch.cat(embs).numpy()

    subject_ids = np.unique(subject)
    per_subject = np.stack([day_emb[subject == s].mean(axis=0) for s in subject_ids])
    return subject_ids, per_subject


def matrix_probe_auc(
    day_matrix: np.ndarray,
    subject: np.ndarray,
    label: np.ndarray,
    train_frac: float = 0.6,
    seed: int = 0,
    epochs: int = 300,
    lr: float = 0.05,
) -> float:
    """Subject-disjoint logistic probe over any (N_days, F) feature matrix.

    Day features are averaged per subject; used both for encoder embeddings
    (via ``linear_probe_auc``) and for hand-crafted feature baselines.
    """
    subject_ids = np.unique(subject)
    embs = np.stack([day_matrix[subject == s].mean(axis=0) for s in subject_ids])
    y = label[subject_ids]

    # Stratified subject-disjoint split so both classes appear on each side
    # even for imbalanced cohorts (e.g. 12 T1DM vs 98 T2DM).
    rng = np.random.default_rng(seed)
    train_parts, test_parts = [], []
    for cls in np.unique(y):
        cls_idx = rng.permutation(np.nonzero(y == cls)[0])
        n_train_cls = min(len(cls_idx) - 1, max(1, int(train_frac * len(cls_idx))))
        train_parts.append(cls_idx[:n_train_cls])
        test_parts.append(cls_idx[n_train_cls:])
    train_idx = np.concatenate(train_parts)
    test_idx = np.concatenate(test_parts)

    mean = embs[train_idx].mean(axis=0)
    std = embs[train_idx].std(axis=0) + 1e-6
    x = torch.from_numpy((embs - mean) / std).float()
    t = torch.from_numpy(y).float()

    torch.manual_seed(seed)
    w = torch.zeros(x.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=lr)
    for _ in range(epochs):
        logits = x[train_idx] @ w + b
        loss = F.binary_cross_entropy_with_logits(logits, t[train_idx])
        loss = loss + 1e-3 * w.pow(2).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        scores = (x[test_idx] @ w + b).numpy()
    return roc_auc(y[test_idx], scores)


def linear_probe_auc(
    model: GlucoFM,
    values: np.ndarray,
    mask: np.ndarray,
    subject: np.ndarray,
    label: np.ndarray,
    train_frac: float = 0.6,
    seed: int = 0,
    epochs: int = 300,
    lr: float = 0.05,
) -> float:
    """Train a logistic probe on frozen embeddings; subject-disjoint ROC-AUC."""
    model.eval()
    v = torch.from_numpy(values).float()
    m = torch.from_numpy(mask).float()
    with torch.no_grad():
        day_emb = torch.cat(
            [model(v[i : i + 256], m[i : i + 256]) for i in range(0, v.shape[0], 256)]
        ).numpy()
    return matrix_probe_auc(day_emb, subject, label, train_frac, seed, epochs, lr)
