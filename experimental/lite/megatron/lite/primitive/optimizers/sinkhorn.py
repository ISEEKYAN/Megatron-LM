# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Fresh-N Algorithm-1 Sinkhorn direction for logical parameter matrices.

The caller owns sharding/collectives: this primitive deliberately accepts one
complete logical matrix, so it cannot accidentally compute the mask or column
norms from rank-local fragments.  Distributed adapters must gather the logical
matrix before calling it and shard the returned direction afterwards.
"""

from __future__ import annotations

import math

import torch

K = 11
TAU = 1e-3
EPS = 1e-20
GAMMA = 0.18


def sinkhorn_direction(nesterov: torch.Tensor) -> torch.Tensor:
    """Return ``sqrt(n) * U`` from Algorithm 1, restarting from current ``N``.

    This is intentionally stateless: carrying a prior normalized ``U`` would
    be a warm-start and changes finite-K behavior when a mask changes.
    """
    if nesterov.ndim != 2 or not nesterov.is_floating_point():
        raise ValueError('Sinkhorn requires a floating logical matrix [m, n]')
    if not nesterov.numel() or not torch.isfinite(nesterov).all():
        raise ValueError('Sinkhorn requires a nonempty finite logical matrix')
    n = nesterov.float()
    row_norm = torch.linalg.vector_norm(n, dim=1)
    mask = row_norm <= TAU * row_norm.mean()
    update = n.clone()
    update[mask] = 0
    for iteration in range(K):
        axis = 1 if iteration % 2 == 0 else 0
        update = update / (torch.linalg.vector_norm(update, dim=axis, keepdim=True) + EPS)
    return update * math.sqrt(n.shape[1])


def algorithm1_update(
    weight: torch.Tensor,
    momentum: torch.Tensor,
    gradient: torch.Tensor,
    *,
    lr: float,
    beta: float = 0.95,
    multiplier: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One exact local Algorithm-1 update, returning ``(W_next, M, N)``.

    ``lr * multiplier`` is supplied by the group router; Engram uses a 5x
    multiplier.  There is no decay and no normalized-state cache.
    """
    if any(x.shape != weight.shape or x.dtype != torch.float32 for x in (momentum, gradient)):
        raise ValueError('Algorithm 1 requires matching FP32 momentum and gradient')
    if weight.ndim != 2 or not all(math.isfinite(x) for x in (lr, beta, multiplier)):
        raise ValueError('Invalid Algorithm 1 inputs')
    if not 0 <= beta < 1 or lr < 0:
        raise ValueError('Invalid Algorithm 1 hyperparameters')
    m = beta * momentum + (1 - beta) * gradient
    n = beta * m + (1 - beta) * gradient
    direction = sinkhorn_direction(n)
    return weight - (GAMMA * lr * multiplier) * direction, m, n
