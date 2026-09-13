# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Explicit tensor shards and full-matrix optimizer publication windows.

Forward/backward weights stay local. Matrix optimizers retain full momentum
and temporarily gather masters/gradients; nonlinear matrix updates must never
silently become updates on independent local matrices.
"""

from contextlib import contextmanager
from dataclasses import dataclass

import torch
import torch.distributed as dist
from megatron.lite.primitive.ckpt.hf_weights import allgather_concat


@dataclass(frozen=True)
class TensorShard:
    shape: tuple[int, ...]
    dim: int
    rank: int
    size: int

    def __post_init__(self):
        if (
            any(type(value) is not int for value in (self.dim, self.rank, self.size))
            or self.size < 1
            or not self.shape
            or any(type(value) is not int or value < 1 for value in self.shape)
        ):
            raise ValueError('Tensor shard requires positive integral dimensions')
        if not 0 <= self.rank < self.size or not 0 <= self.dim < len(self.shape):
            raise ValueError('Invalid tensor shard rank or axis')
        if self.shape[self.dim] % self.size:
            raise ValueError('Tensor shard axis must be divisible by TP size')

    @property
    def local_shape(self):
        shape = list(self.shape)
        shape[self.dim] //= self.size
        return tuple(shape)

    def slice(self, full):
        if tuple(full.shape) != self.shape:
            raise ValueError(
                'Tensor shard requires the full logical shape; double slicing is invalid'
            )
        width = self.shape[self.dim] // self.size
        return full.narrow(self.dim, self.rank * width, width).contiguous()


def logical_shape(parameter):
    layout = getattr(parameter, 'tp_shard', None)
    return tuple(parameter.shape) if layout is None else layout.shape


def gather_parameter(parameter, group, value=None):
    value = parameter.detach() if value is None else value
    layout = getattr(parameter, 'tp_shard', None)
    if layout is None:
        return value
    if group is None:
        raise ValueError('Sharded parameter requires an explicit TP group')
    if tuple(value.shape) != layout.local_shape:
        raise ValueError('Tensor shard storage disagrees with its declared local shape')
    return allgather_concat(value, layout.size, group, layout.dim)


def slice_parameter(parameter, full):
    layout = getattr(parameter, 'tp_shard', None)
    return full if layout is None else layout.slice(full)


def agree_finite(valid, group, device):
    flag = torch.tensor(int(valid), device=device)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=group)
    return bool(flag.item())


@torch.no_grad()
def broadcast_module(module, group):
    """Synchronize initial owners, including FP8 byte storage, before sharding."""
    source = dist.get_global_rank(group, 0)
    for tensor in (*module.parameters(), *module.buffers()):
        value = tensor.detach().contiguous().reshape(-1).view(torch.uint8)
        dist.broadcast(value, source, group=group)
        tensor.copy_(value.view(tensor.dtype).reshape(tensor.shape))


@contextmanager
def full_matrix_parameters(parameters, group, *, publish=False):
    """Keep owner identities; expose logical matrices only inside this window.

    This is an optimizer/checkpoint boundary, never a forward weight gather.
    Gradients and original shard storage are restored even if staging fails.
    All participants must enter in identical owner order.
    """
    saved = []
    try:
        for p in parameters:
            if getattr(p, 'tp_shard', None) is None:
                continue
            data, grad, main = p.data, p.grad, getattr(p, 'main_grad', None)
            full = gather_parameter(p, group)
            active = torch.tensor(
                int(main is not None or grad is not None), device=p.device
            )
            dist.all_reduce(active, group=group)
            gradient = main if main is not None else grad
            if active.item():
                gradient = gather_parameter(
                    p, group, torch.zeros_like(p) if gradient is None else gradient
                )
            saved.append((p, data, grad, main))
            p.grad = p.main_grad = None
            p.data = full
            p.grad = p.main_grad = gradient
        yield
        if publish:
            with torch.no_grad():
                for p, data, _, _ in saved:
                    data.copy_(slice_parameter(p, p.data))
    finally:
        for p, data, grad, main in saved:
            p.grad = p.main_grad = None
            p.data = data
            p.grad, p.main_grad = grad, main
