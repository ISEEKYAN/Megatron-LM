# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Muon on explicitly declared logical matrices, with FP32 owner state."""

import math
from copy import deepcopy
from functools import partial

import torch
from megatron.lite.primitive.quantization.block_fp8 import quantize_block_fp8


def _matrix_shape(shape):
    return (
        isinstance(shape, (tuple, list))
        and len(shape) in (2, 3)
        and all(type(d) is int and d > 0 for d in shape)
    )


class StagedMatrixOptimizer(torch.optim.Optimizer):
    """Common candidate publication; subclasses supply direction and layout checks."""

    def _idle(self, action):
        if self._prepared is not None:
            raise RuntimeError(f'Cannot {action} a prepared {self._label} step')

    def candidates(self):
        if self._prepared is None:
            raise RuntimeError(f'No prepared {self._label} step')
        return tuple((p, value) for p, value, _ in self._prepared)

    @torch.no_grad()
    def commit_step(self):
        for (p, value), (_, _, momentum) in zip(self.candidates(), self._prepared):
            p.copy_(value)
            self.state[p][self._momentum_key] = momentum
        self.discard_step()

    def discard_step(self):
        self._prepared = None

    def step(self, closure=None):
        if closure is not None:
            raise ValueError(f'{self._label} requires explicit accumulated gradients')
        if not self.prepare_step():
            return False
        self.commit_step()
        return True

    def state_dict(self):
        self._idle('checkpoint')
        return super().state_dict()

    def _validate_momentum(self, saved, shape):
        if saved is None:
            return
        momentum = saved.get(self._momentum_key)
        if (
            set(saved) != {self._momentum_key}
            or not isinstance(momentum, torch.Tensor)
            or momentum.dtype != torch.float32
            or momentum.shape != shape
            or not torch.isfinite(momentum).all()
        ):
            raise ValueError(
                f'{self._label} checkpoint requires matching finite FP32 momentum'
            )


class HeadwiseMuon(StagedMatrixOptimizer):
    _label, _momentum_key = 'Muon', 'momentum_buffer'

    def __init__(
        self,
        params,
        *,
        lr,
        ns_steps,
        coefficient_type,
        weight_decay=0.1,
        momentum=0.95,
        update_rms=0.18,
    ):
        from emerging_optimizers.orthogonalized_optimizers.muon_utils import (
            newton_schulz,
        )

        if type(ns_steps) is not int or ns_steps < 1:
            raise ValueError('ns_steps must be a positive integer')
        # Validate the explicitly selected backend API/config before allocating state.
        newton_schulz(torch.zeros(1, 1), ns_steps, coefficient_type=coefficient_type)
        self._orthogonalize = newton_schulz
        self._prepared = None
        super().__init__(
            params,
            dict(
                lr=lr,
                weight_decay=weight_decay,
                momentum=momentum,
                update_rms=update_rms,
                ns_steps=ns_steps,
                coefficient_type=coefficient_type,
            ),
        )
        self._validate_groups()

    def _validate_groups(self):
        seen = set()
        for group in self.param_groups:
            shape = group.get('matrix_shape')
            if not _matrix_shape(shape):
                raise ValueError(
                    'An explicit positive logical matrix shape is required'
                )
            partitions = group.get('matrix_partitions')
            if partitions is not None and (
                not isinstance(partitions, (list, tuple))
                or not partitions
                or not all(_matrix_shape(part) for part in partitions)
                or sum(math.prod(part) for part in partitions) != math.prod(shape)
            ):
                raise ValueError(
                    'Logical partitions must cover the physical matrix exactly'
                )
            for key in ('lr', 'weight_decay', 'momentum', 'update_rms'):
                if not math.isfinite(group[key]) or group[key] < 0:
                    raise ValueError(f'Invalid Muon {key}')
            if group['momentum'] >= 1:
                raise ValueError('Muon momentum must be less than one')
            for p in group['params']:
                if id(p) in seen:
                    raise ValueError('Duplicate Muon parameter owner')
                seen.add(id(p))
                if (
                    p.numel() != math.prod(shape)
                    or not p.is_contiguous()
                    or p.dtype != torch.float32
                    or not p.requires_grad
                ):
                    raise ValueError(
                        'Muon requires a contiguous FP32 master matching its logical shape'
                    )

    @torch.no_grad()
    def prepare_step(self):
        if self._prepared is not None:
            raise RuntimeError('A Muon step is already prepared')
        self._validate_groups()
        prepared = []
        for group in self.param_groups:
            for p in group['params']:
                grad = getattr(p, 'main_grad', p.grad)
                if grad is None:
                    continue
                if (
                    grad.dtype != torch.float32
                    or grad.shape != p.shape
                    or grad.is_sparse
                ):
                    raise ValueError(
                        'Muon requires matching dense native FP32 gradients'
                    )
                if not torch.isfinite(grad).all():
                    return False
                previous = self.state.get(p, {}).get(
                    'momentum_buffer', torch.zeros_like(p)
                )
                beta = group['momentum']
                momentum = beta * previous + (1 - beta) * grad
                nesterov = beta * momentum + (1 - beta) * grad
                from emerging_optimizers.utils import fp32_matmul_precision

                shapes = group.get('matrix_partitions') or (group['matrix_shape'],)
                chunks = nesterov.flatten().split(
                    [math.prod(shape) for shape in shapes]
                )
                logical = [chunk.reshape(shape) for chunk, shape in zip(chunks, shapes)]
                matrices = [
                    matrix
                    for part in logical
                    for matrix in (part.unbind(0) if part.ndim == 3 else (part,))
                ]
                directions = []
                with torch.autocast(
                    device_type=p.device.type, enabled=False
                ), fp32_matmul_precision('highest'):
                    for matrix in matrices:
                        update = self._orthogonalize(
                            matrix,
                            group['ns_steps'],
                            coefficient_type=group['coefficient_type'],
                        )
                        rms = update.square().mean().sqrt()
                        directions.append(
                            update * (group['update_rms'] / rms.clamp_min(1e-30))
                        )
                update = torch.cat([direction.flatten() for direction in directions])
                candidate = p * (1 - group['lr'] * group['weight_decay'])
                candidate = candidate - group['lr'] * update.reshape_as(p)
                if (
                    not torch.isfinite(candidate).all()
                    or not torch.isfinite(momentum).all()
                ):
                    return False
                prepared.append((p, candidate, momentum))
        self._prepared = prepared
        return True

    def load_state_dict(self, state_dict):
        self._idle('restore')
        saved = state_dict['param_groups']
        if len(saved) != len(self.param_groups) or any(
            tuple(a['matrix_shape']) != tuple(b['matrix_shape'])
            or a.get('matrix_partitions') != b.get('matrix_partitions')
            for a, b in zip(saved, self.param_groups)
        ):
            raise ValueError('Muon logical layout changed; reshard explicitly')
        for old, current in zip(saved, self.param_groups):
            if len(old['params']) != len(current['params']) or any(
                old[key] != current[key]
                for key in ('ns_steps', 'coefficient_type', 'momentum', 'update_rms')
            ):
                raise ValueError('Muon backend recipe or owner count changed')
            for pid, parameter in zip(old['params'], current['params']):
                self._validate_momentum(state_dict['state'].get(pid), parameter.shape)
        super().load_state_dict(state_dict)
        self._validate_groups()


class MixedOptimizer:
    """Atomic local publication across matrix optimizers, AdamW and FP8 storage."""

    def __init__(self, groups, config, tables=()):
        from .sinkhorn import Sinkhorn

        if not groups or any(
            p.dtype != torch.float32 for g in groups for p in g['params']
        ):
            raise ValueError('V4.1 optimizer requires native FP32 parameter masters')
        for group in groups:
            for p in group['params']:
                if not hasattr(p, '_v41_main_grad_hook'):
                    p.main_grad = p.grad
                    p._v41_main_grad_hook = p.register_post_accumulate_grad_hook(
                        _publish_main_grad
                    )
        self.config = config
        self.optimizers = []
        backends = {
            'muon': partial(
                HeadwiseMuon,
                ns_steps=config.ns_steps,
                coefficient_type=config.coefficient_type,
            ),
            'sinkhorn': Sinkhorn,
            'adamw': partial(
                torch.optim.AdamW, betas=(0.9, 0.95), eps=1e-20, foreach=False
            ),
        }
        for algorithm, factory in backends.items():
            selected = [g for g in groups if g['algorithm'] == algorithm]
            if selected:
                self.optimizers.append(factory(selected, lr=config.lr))
        self.tables = list(tables)

    def _validate_trainability(self):
        """The assembling model supplies phase and trainability validation."""

    @property
    def param_groups(self):
        return [g for o in self.optimizers for g in o.param_groups]

    def zero_grad(self, set_to_none=True):
        for backend in self.optimizers:
            backend.zero_grad(set_to_none=set_to_none)
        for group in self.param_groups:
            for p in group['params']:
                p.main_grad = p.grad

    @torch.no_grad()
    def step(self):
        self._validate_trainability()
        parameters = [p for g in self.param_groups for p in g['params']]
        gradients = [
            p.main_grad if getattr(p, 'main_grad', None) is not None else p.grad
            for p in parameters
        ]
        active = [g for g in gradients if g is not None]
        if any(g.dtype != torch.float32 or g.is_sparse for g in active):
            raise ValueError('Expected native dense FP32 main_grad')
        norm = (
            torch.stack([g.double().square().sum() for g in active]).sum().sqrt()
            if active
            else torch.tensor(0.0)
        )
        if not torch.isfinite(norm):
            return False, float(norm), None
        coefficient = (
            min(1.0, self.config.clip_grad / (float(norm) + 1e-6))
            if self.config.clip_grad
            else 1.0
        )
        # Reversible gradient views permit retry on failed publication.
        original = [(p, p.grad, getattr(p, 'main_grad', None)) for p in parameters]
        staged_adam, candidates = [], {}
        try:
            for p, g in zip(parameters, gradients):
                p.grad = None if g is None else g * coefficient
                p.main_grad = p.grad
            for backend in self.optimizers:
                if isinstance(backend, torch.optim.AdamW):
                    candidate = deepcopy(backend)
                    for old, new in zip(backend.param_groups, candidate.param_groups):
                        for p, q in zip(old['params'], new['params']):
                            q.grad = p.grad
                    candidate.step()
                    for old, new in zip(backend.param_groups, candidate.param_groups):
                        for p, q in zip(old['params'], new['params']):
                            candidates[id(p)] = q
                    if any(
                        not torch.isfinite(v).all()
                        for s in candidate.state.values()
                        for v in s.values()
                        if isinstance(v, torch.Tensor)
                    ):
                        return False, float(norm), None
                    staged_adam.append((backend, candidate))
                else:
                    if not backend.prepare_step():
                        return False, float(norm), None
                    candidates.update(
                        (id(p), value) for p, value in backend.candidates()
                    )
            if any(not torch.isfinite(p).all() for p in candidates.values()):
                return False, float(norm), None
            storage = []
            for table in self.tables:
                weight, scale = quantize_block_fp8(
                    candidates[id(table.master)], (1, 32), scale_format='e8m0'
                )
                if (
                    not torch.isfinite(weight.float()).all()
                    or not torch.isfinite(scale.float()).all()
                ):
                    return False, float(norm), None
                storage.append((table, weight, scale))
            for backend in self.optimizers:
                if not isinstance(backend, torch.optim.AdamW):
                    backend.commit_step()
            for backend, candidate in staged_adam:
                for old, new in zip(backend.param_groups, candidate.param_groups):
                    for p, q in zip(old['params'], new['params']):
                        p.copy_(q)
                backend.load_state_dict(candidate.state_dict())
            for table, weight, scale in storage:
                table.weight.copy_(weight)
                table.scale.copy_(scale)
            return True, float(norm), None
        finally:
            for backend in self.optimizers:
                if not isinstance(backend, torch.optim.AdamW):
                    backend.discard_step()
            for p, grad, main in original:
                p.grad, p.main_grad = grad, main

    def state_dict(self):
        self._validate_trainability()
        return dict(
            owners=[g['owner_key'] for g in self.param_groups],
            clip_grad=self.config.clip_grad,
            optimizers=[o.state_dict() for o in self.optimizers],
        )

    def load_state_dict(self, state):
        self._validate_trainability()
        if state.get('clip_grad') != self.config.clip_grad:
            raise ValueError('Optimizer clipping contract differs')
        if state.get('owners') != [g['owner_key'] for g in self.param_groups] or len(
            state['optimizers']
        ) != len(self.optimizers):
            raise ValueError('Optimizer owner layout differs')
        for backend, saved in zip(self.optimizers, state['optimizers']):
            backend.load_state_dict(saved)


def _publish_main_grad(parameter):
    if parameter.grad.dtype != torch.float32:
        raise RuntimeError('V4.1 gradient producer did not return native FP32')
    parameter.main_grad = parameter.grad
