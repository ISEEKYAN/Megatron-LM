# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Fresh-N Algorithm-1 Sinkhorn direction for logical parameter matrices."""

from __future__ import annotations

import math

import torch
import torch.distributed as dist

from .headwise_muon import StagedMatrixOptimizer

K = 11
TAU = 1e-3
EPS = 1e-20
GAMMA = 0.18


def _reduce(tensor, group, op=dist.ReduceOp.SUM):
    if group is not None:
        dist.all_reduce(tensor, op=op, group=group)
    return tensor


def _any(flag, device, row_group, column_group):
    value = torch.tensor(int(flag), device=device, dtype=torch.int32)
    _reduce(value, row_group, dist.ReduceOp.MAX)
    _reduce(value, column_group, dist.ReduceOp.MAX)
    return bool(value.item())


def sinkhorn_direction(nesterov, *, row_group=None, column_group=None, trace=None):
    """Fresh current-N direction; groups partition rows and columns respectively."""
    if nesterov.ndim != 2 or nesterov.dtype != torch.float32:
        raise ValueError('Sinkhorn requires an FP32 matrix [m, n]')
    rows = _reduce(torch.tensor(nesterov.shape[0], device=nesterov.device), row_group)
    columns = _reduce(
        torch.tensor(nesterov.shape[1], device=nesterov.device), column_group
    )
    if rows.item() <= 0 or columns.item() <= 0:
        raise ValueError('Sinkhorn requires a nonempty logical matrix')
    if _any(
        not torch.isfinite(nesterov).all(), nesterov.device, row_group, column_group
    ):
        raise ValueError('Sinkhorn requires a finite logical matrix')

    def norm(matrix, axis):
        # Scale before squaring so finite large/tiny FP32 N retains its norm.
        group = column_group if axis == 1 else row_group
        shape = list(matrix.shape)
        shape[axis] = 1
        maximum = (
            matrix.abs().amax(dim=axis, keepdim=True)
            if matrix.shape[axis]
            else matrix.new_zeros(shape)
        )
        _reduce(maximum, group, dist.ReduceOp.MAX)
        scale = torch.where(maximum == 0, torch.ones_like(maximum), maximum)
        squares = (matrix / scale).square().sum(dim=axis, keepdim=True)
        _reduce(squares, column_group if axis == 1 else row_group)
        return squares.sqrt() * maximum

    rho = norm(nesterov, 1)
    mean = _reduce(rho.sum(), row_group) / rows
    update = nesterov.clone()
    update.masked_fill_(rho <= TAU * mean, 0)
    for iteration in range(K):
        axis = 1 if iteration % 2 == 0 else 0
        update = update / (norm(update, axis) + EPS)
        if trace is not None:
            trace(iteration + 1, update.detach().clone())
    return update * math.sqrt(columns.item())


def algorithm1_update(
    weight: torch.Tensor,
    momentum: torch.Tensor,
    gradient: torch.Tensor,
    *,
    lr: float,
    beta: float = 0.95,
    multiplier: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One exact local Algorithm-1 update, returning ``(W_next, M, N)``."""
    if any(
        x.shape != weight.shape or x.dtype != torch.float32
        for x in (momentum, gradient)
    ):
        raise ValueError('Algorithm 1 requires matching FP32 momentum and gradient')
    if weight.ndim != 2 or not all(math.isfinite(x) for x in (lr, beta, multiplier)):
        raise ValueError('Invalid Algorithm 1 inputs')
    if not 0 <= beta < 1 or lr < 0:
        raise ValueError('Invalid Algorithm 1 hyperparameters')
    m = beta * momentum + (1 - beta) * gradient
    n = beta * m + (1 - beta) * gradient
    direction = sinkhorn_direction(n)
    return weight - (GAMMA * lr * multiplier) * direction, m, n


def _span(rows, parts, rank):
    size, extra = divmod(rows, parts)
    begin = rank * size + min(rank, extra)
    return begin, begin + size + (rank < extra)


class Sinkhorn(StagedMatrixOptimizer):
    _label, _momentum_key = 'Sinkhorn', 'momentum'

    """FP32 Algorithm 1 with replica-sharded momentum and staged publication."""

    def __init__(
        self, params, *, lr, row_group=None, column_group=None, replica_group=None
    ):
        super().__init__(params, dict(lr=lr, multiplier=1.0))
        self.row_group = row_group
        self.column_group = column_group
        self.replica_group = replica_group
        self.replica_size = (
            1 if replica_group is None else dist.get_world_size(replica_group)
        )
        self.replica_rank = 0 if replica_group is None else dist.get_rank(replica_group)
        self._prepared = None
        if self.replica_size > 1 and row_group is None:
            raise ValueError('Replica-owned rows require a logical row group')
        parameters = [p for group in self.param_groups for p in group['params']]
        if len({id(p) for p in parameters}) != len(parameters):
            raise ValueError('Each Sinkhorn parameter owner must appear once')
        for group in self.param_groups:
            for parameter in group['params']:
                if (
                    parameter.ndim != 2
                    or parameter.dtype != torch.float32
                    or not parameter.requires_grad
                ):
                    raise ValueError('Sinkhorn owns trainable FP32 matrix masters only')
                if (
                    any(g is not None for g in (row_group, column_group, replica_group))
                    and not parameter.is_cuda
                ):
                    raise ValueError(
                        'Distributed Sinkhorn requires resident CUDA state'
                    )

    def _gradient_shard(self, gradient):
        if self.replica_group is None:
            return gradient
        rows, columns = gradient.shape
        width = max(1, (rows + self.replica_size - 1) // self.replica_size)
        packed = gradient.new_zeros(self.replica_size, width, columns)
        for rank in range(self.replica_size):
            start, end = _span(rows, self.replica_size, rank)
            packed[rank, : end - start].copy_(gradient[start:end])
        result = gradient.new_empty(width, columns)
        dist.reduce_scatter_tensor(
            result, packed.flatten(0, 1), group=self.replica_group
        )
        start, end = _span(rows, self.replica_size, self.replica_rank)
        return result[: end - start]

    def _replicate(self, shard, rows):
        if self.replica_group is None:
            return shard
        width = max(1, (rows + self.replica_size - 1) // self.replica_size)
        padded = shard.new_zeros(width, shard.shape[1])
        padded[: shard.shape[0]].copy_(shard)
        gathered = [torch.empty_like(padded) for _ in range(self.replica_size)]
        dist.all_gather(gathered, padded, group=self.replica_group)
        pieces = []
        for rank, value in enumerate(gathered):
            start, end = _span(rows, self.replica_size, rank)
            pieces.append(value[: end - start])
        return torch.cat(pieces)

    @torch.no_grad()
    def prepare_step(self):
        """Stage W/M without changing live weights or state; return false on inf."""
        if self._prepared is not None:
            raise RuntimeError('A Sinkhorn step is already prepared')
        inputs = []
        for group in self.param_groups:
            lr, multiplier = group['lr'], group['multiplier']
            if not all(math.isfinite(x) and x >= 0 for x in (lr, multiplier)):
                raise ValueError('Invalid Sinkhorn learning rate or multiplier')
            for parameter in group['params']:
                gradient = getattr(parameter, 'main_grad', parameter.grad)
                if gradient is None:
                    gradient = torch.zeros_like(parameter)
                invalid = (
                    gradient.dtype != torch.float32
                    or gradient.shape != parameter.shape
                    or gradient.device != parameter.device
                )
                if _any(invalid, parameter.device, self.row_group, self.column_group):
                    raise ValueError('Sinkhorn requires matching native FP32 gradients')
                if _any(
                    not torch.isfinite(gradient).all(),
                    parameter.device,
                    self.row_group,
                    self.column_group,
                ):
                    return False
                inputs.append((parameter, gradient, lr, multiplier))
        prepared = []
        for parameter, gradient, lr, multiplier in inputs:
            start, end = _span(parameter.shape[0], self.replica_size, self.replica_rank)
            gradient = self._gradient_shard(gradient)
            previous = self.state.get(parameter, {}).get('momentum')
            if previous is None:
                previous = torch.zeros_like(gradient)
            momentum = 0.95 * previous + 0.05 * gradient
            nesterov = 0.95 * momentum + 0.05 * gradient
            if _any(
                not torch.isfinite(nesterov).all(),
                parameter.device,
                self.row_group,
                self.column_group,
            ):
                return False
            direction = sinkhorn_direction(
                nesterov, row_group=self.row_group, column_group=self.column_group
            )
            candidate = parameter[start:end] - (GAMMA * lr * multiplier) * direction
            if _any(
                not torch.isfinite(candidate).all(),
                parameter.device,
                self.row_group,
                self.column_group,
            ):
                return False
            prepared.append(
                (parameter, self._replicate(candidate, parameter.shape[0]), momentum)
            )
        self._prepared = prepared
        return True

    def agree_skip(self, skip):
        """OR a publication/skip decision across the logical matrix grid."""
        parameter = self.param_groups[0]['params'][0]
        return _any(skip, parameter.device, self.row_group, self.column_group)

    def state_dict(self):
        result = super().state_dict()
        ranks = tuple(
            None if group is None else tuple(dist.get_process_group_ranks(group))
            for group in (self.row_group, self.column_group, self.replica_group)
        )
        result['sinkhorn_layout'] = [
            (tuple(p.shape), self.replica_size, self.replica_rank, ranks)
            for group in self.param_groups
            for p in group['params']
        ]
        return result

    def load_state_dict(self, state_dict):
        self._idle('restore')
        current = self.state_dict()['sinkhorn_layout']
        if state_dict.get('sinkhorn_layout') != current:
            raise ValueError('Sinkhorn checkpoint layout differs; reshard explicitly')
        saved_ids = [
            pid for group in state_dict['param_groups'] for pid in group['params']
        ]
        parameters = [p for group in self.param_groups for p in group['params']]
        if len(saved_ids) != len(parameters):
            raise ValueError('Sinkhorn checkpoint parameter count differs')
        for pid, parameter in zip(saved_ids, parameters):
            start, end = _span(parameter.shape[0], self.replica_size, self.replica_rank)
            self._validate_momentum(
                state_dict['state'].get(pid), (end - start, parameter.shape[1])
            )
        super().load_state_dict(
            {k: v for k, v in state_dict.items() if k != 'sinkhorn_layout'}
        )
