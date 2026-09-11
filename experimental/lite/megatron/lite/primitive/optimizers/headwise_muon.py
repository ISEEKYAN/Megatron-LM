# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Muon on explicitly declared logical matrices, with FP32 owner state.

The caller owns layout interpretation. ``matrix_shape`` is (rows, columns) or
(heads, rows, columns), independent of parameter names and physical flattening.
Newton-Schulz is supplied by NVIDIA emerging_optimizers, never reimplemented.
This local primitive requires reassembled matrices; distributed lowering is a
separate contract. prepare/commit permits an atomic mixed-optimizer coordinator.
"""

import math

import torch


class HeadwiseMuon(torch.optim.Optimizer):
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

    def candidates(self):
        if self._prepared is None:
            raise RuntimeError('No prepared Muon step')
        return tuple((p, value) for p, value, _ in self._prepared)

    @torch.no_grad()
    def commit_step(self):
        if self._prepared is None:
            raise RuntimeError('No prepared Muon step')
        for p, value, momentum in self._prepared:
            p.copy_(value)
            self.state[p]['momentum_buffer'] = momentum
        self._prepared = None

    def discard_step(self):
        self._prepared = None

    def step(self, closure=None):
        if closure is not None:
            raise ValueError('Muon requires explicit accumulated gradients')
        if not self.prepare_step():
            return False
        self.commit_step()
        return True

    def state_dict(self):
        if self._prepared is not None:
            raise RuntimeError('Cannot checkpoint a prepared Muon step')
        return super().state_dict()

    def load_state_dict(self, state_dict):
        if self._prepared is not None:
            raise RuntimeError('Cannot restore a prepared Muon step')
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
                state = state_dict['state'].get(pid)
                if state is None:
                    continue
                momentum = state.get('momentum_buffer')
                if (
                    set(state) != {'momentum_buffer'}
                    or not isinstance(momentum, torch.Tensor)
                    or momentum.dtype != torch.float32
                    or momentum.shape != parameter.shape
                    or not torch.isfinite(momentum).all()
                ):
                    raise ValueError(
                        'Muon checkpoint requires matching finite FP32 momentum'
                    )
        super().load_state_dict(state_dict)
        self._validate_groups()
