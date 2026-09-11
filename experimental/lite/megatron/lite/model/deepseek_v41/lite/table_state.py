# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Transactional resident Engram state; the Sinkhorn transform is supplied later."""

import math

import torch
from megatron.lite.primitive.quantization.block_fp8 import quantize_block_fp8


class EngramTableState:
    """Own full local-shard FP32 gradients/momentum and publication lifetime.

    FP32 master is the port representation. No visited-row filtering is allowed:
    even a row with zero current gradient participates in momentum and updates.
    Replica reduction/state sharding and global skip agreement belong to the
    distributed optimizer integration. This object does not implement Sinkhorn.
    """

    def __init__(self, table):
        self.table = table
        self.version = 0
        self.last_step = -1
        self.active_step = None
        self._ready = False
        master = table.master
        if master is not None and master.dtype != torch.float32:
            raise ValueError('Engram state requires a native FP32 master')
        self.main_grad = None if master is None else torch.zeros_like(master)
        self.momentum = None if master is None else torch.zeros_like(master)

    def begin(self, step):
        if self.active_step is not None:
            raise RuntimeError('A table step is already active')
        if type(step) is not int or step <= self.last_step:
            raise RuntimeError('step tag must increase monotonically')
        self.active_step, self._ready = step, False
        if self.main_grad is not None:
            self.main_grad.zero_()

    def accept_gradient(self, step, gradient):
        if step != self.active_step or self._ready:
            raise RuntimeError('Stale or duplicate table gradient return')
        if self.main_grad is not None:
            if (
                gradient is None
                or gradient.dtype != torch.float32
                or gradient.shape != self.main_grad.shape
                or gradient.device != self.main_grad.device
            ):
                raise ValueError('Require full local-shard FP32 gradient')
            self.main_grad.copy_(gradient.detach())
        elif gradient is not None:
            raise ValueError('Frozen table cannot receive gradients')
        self._ready = True

    def _finish(self):
        self.last_step = self.active_step
        self.active_step, self._ready = None, False
        if self.main_grad is not None:
            self.main_grad.zero_()

    @torch.no_grad()
    def step(self, update_rule, *, lr, beta=0.95, skip=False):
        """Prepare full M/N, then transactionally publish an explicit update.

        update_rule(N) must return a dense direction for every local row. Tests
        use the identity rule to isolate Nesterov/state semantics. E2 supplies
        the actual logical-matrix Sinkhorn transform and LR factors separately.
        All candidate tensors are prepared before touching persistent state.
        """
        if self.active_step is None or not self._ready:
            raise RuntimeError('Table gradients are pending or no step is active')
        if (
            not math.isfinite(lr)
            or lr < 0
            or not math.isfinite(beta)
            or not 0 <= beta < 1
        ):
            raise ValueError('Invalid learning rate or momentum coefficient')
        if skip or self.table.master is None:
            self._finish()
            return False
        if not torch.isfinite(self.main_grad).all():
            self._finish()
            return False
        momentum = beta * self.momentum + (1 - beta) * self.main_grad
        nesterov = beta * momentum + (1 - beta) * self.main_grad
        direction = update_rule(nesterov)
        if (
            not isinstance(direction, torch.Tensor)
            or direction.shape != self.main_grad.shape
            or direction.dtype != torch.float32
            or direction.device != self.main_grad.device
        ):
            raise ValueError('Update rule must return the full local FP32 matrix')
        candidate = self.table.master.detach() - lr * direction
        if not torch.isfinite(candidate).all() or not torch.isfinite(momentum).all():
            self._finish()
            return False
        values, scales = quantize_block_fp8(candidate, (1, 32), scale_format='e8m0')
        # No live prefetch views are allowed at this boundary. The parameter
        # identity is stable for optimizer bindings; publication buffers replace
        # the pair only after both have been computed successfully.
        self.table.master.copy_(candidate)
        self.momentum.copy_(momentum)
        self.table.weight, self.table.scale = values, scales
        self.version += 1
        self._finish()
        return True

    def state_dict(self):
        if self.active_step is not None:
            raise RuntimeError('Cannot checkpoint an active table step')
        state = {
            'format_version': 1,
            'version': self.version,
            'last_step': self.last_step,
            'trainable': self.table.master is not None,
            'weight': self.table.weight.detach().clone(),
            'scale': self.table.scale.detach().clone(),
        }
        if self.table.master is not None:
            state.update(
                master=self.table.master.detach().clone(),
                momentum=self.momentum.clone(),
            )
        return state

    @torch.no_grad()
    def load_state_dict(self, state):
        if self.active_step is not None:
            raise RuntimeError('Cannot restore an active table step')
        expected = {
            'format_version',
            'version',
            'last_step',
            'trainable',
            'weight',
            'scale',
        }
        if self.table.master is not None:
            expected |= {'master', 'momentum'}
        if (
            set(state) != expected
            or state['format_version'] != 1
            or type(state['trainable']) is not bool
            or state['trainable'] != (self.table.master is not None)
            or type(state['version']) is not int
            or state['version'] < 0
            or type(state['last_step']) is not int
            or state['last_step'] < -1
        ):
            raise ValueError('Incompatible Engram state metadata')
        destinations = {'weight': self.table.weight, 'scale': self.table.scale}
        if self.table.master is not None:
            destinations.update(master=self.table.master, momentum=self.momentum)
        prepared = {}
        for key, destination in destinations.items():
            source = state[key]
            if (
                not isinstance(source, torch.Tensor)
                or source.dtype != destination.dtype
                or source.shape != destination.shape
            ):
                raise ValueError(f'Invalid state tensor: {key}')
            if not torch.isfinite(source.float()).all():
                raise ValueError(f'Nonfinite state tensor: {key}')
            prepared[key] = source.detach().to(destination.device).clone()
        if self.table.master is not None:
            self.table.master.copy_(prepared['master'])
            self.momentum.copy_(prepared['momentum'])
            self.main_grad.zero_()
        self.table.weight, self.table.scale = prepared['weight'], prepared['scale']
        self.version, self.last_step = state['version'], state['last_step']
