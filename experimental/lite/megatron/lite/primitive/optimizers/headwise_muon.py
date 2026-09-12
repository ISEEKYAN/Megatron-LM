# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Muon on explicitly declared logical matrices, with FP32 owner state."""

import math

import torch


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
            if (
                not isinstance(shape, (tuple, list))
                or len(shape) not in (2, 3)
                or any(type(d) is not int or d < 1 for d in shape)
            ):
                raise ValueError(
                    'An explicit positive logical matrix shape is required'
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

                logical = nesterov.reshape(group['matrix_shape'])
                matrices = logical.unbind(0) if logical.ndim == 3 else (logical,)
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
                update = torch.stack(directions) if logical.ndim == 3 else directions[0]
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
