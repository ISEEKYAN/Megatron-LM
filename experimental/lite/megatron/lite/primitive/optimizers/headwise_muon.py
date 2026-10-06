# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Muon on explicitly declared logical matrices, with FP32 owner state."""

import math
from copy import deepcopy

import torch
from megatron.lite.primitive.quantization.block_fp8 import quantize_block_fp8


def _matrix_shape(shape):
    return (
        isinstance(shape, (tuple, list))
        and len(shape) in (2, 3)
        and all(type(d) is int and d > 0 for d in shape)
    )


def _parameters(optimizer):
    return [p for group in optimizer.param_groups for p in group['params']]


def _pairs(left, right):
    return zip(_parameters(left), _parameters(right))


class HeadwiseMuon(torch.optim.Optimizer):
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
        torch.optim.Optimizer.__init__(
            self,
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

    from .staged_update import (
        _idle,
        _validate_momentum,
        candidates,
        commit_step,
        discard_step,
        state_dict,
        step,
    )


class MixedOptimizer:
    """Local correctness coordinator; every backend publishes on one commit.

    AdamW staging uses the real torch backend on private candidate parameters.
    It deliberately costs extra local storage. Distributed staging/sharding is
    owned by the parallel integration rather than silently approximated here.
    """

    # The model's group_builder owns LR ratios and fixed weight decay.
    owns_param_group_policy = True

    def __init__(
        self,
        model,
        config,
        *,
        group_builder,
        owners,
        stats_factory,
        dp_group=None,
        ps=None,
        rebuild=None,
    ):
        if not math.isfinite(config.clip_grad) or config.clip_grad < 0:
            raise ValueError('Invalid gradient clipping threshold')
        groups = group_builder()
        self._rebuild = rebuild
        self.model = model
        self.dp_group = dp_group
        self.ps = ps
        self.expert_parameters, self.routers, tables, self.row_group = owners()
        self.expert_ids = {id(p) for p in self.expert_parameters}
        self._stats_factory = stats_factory
        self._modality_loads = [None] * len(self.routers)
        self._base___init__(groups, config, tables)
        self.row_parameters = (
            [table.master for table in tables] if self.row_group is not None else []
        )
        self.row_ids = {id(p) for p in self.row_parameters}
        if self.row_parameters:
            from megatron.lite.primitive.optimizers.sinkhorn import Sinkhorn

            # Reuse the distributed logical-row algorithm; local Sinkhorn would
            # change rho_mean, column norms and hence every shard's update.
            backend = next(o for o in self.optimizers if isinstance(o, Sinkhorn))
            sharded = [
                g for g in backend.param_groups if id(g['params'][0]) in self.row_ids
            ]
            backend.param_groups = [
                g
                for g in backend.param_groups
                if id(g['params'][0]) not in self.row_ids
            ]
            self.optimizers.append(
                Sinkhorn(sharded, lr=config.lr, row_group=self.row_group)
            )

    @torch.no_grad()
    def finalize_grads(self):
        if self.ps.ep_size > 1:
            self.finalize_expert_grads()
        # Lookup backward sums requests from all data/context ranks. Dense DDP
        # averages the same objective; apply that normalization exactly once.
        for p in self.row_parameters:
            grad = p.main_grad if p.main_grad is not None else p.grad
            if grad is not None:
                grad.div_(self.ps.dp_cp_size)
                p.grad = p.main_grad = grad

    @torch.no_grad()
    def finalize_expert_grads(self):
        """Expert gradients already sum source tokens within EP dispatch.

        Sum only matching expert replicas, then divide by the dense DP size
        used to scale the loss. Never average different experts together.
        """
        import torch.distributed as dist

        ps = self.ps
        for p in self.expert_parameters:
            grad = p.main_grad if p.main_grad is not None else p.grad
            active = torch.tensor(int(grad is not None), device=p.device)
            dist.all_reduce(active, group=ps.ep_dp_group)
            if not active.item():
                continue
            if grad is None:
                grad = torch.zeros_like(p)
            dist.all_reduce(grad, group=ps.ep_dp_group)
            grad.div_(ps.dp_size)
            p.grad = p.main_grad = grad

    def _grad_norm(self, parameters, gradients):
        if self.ps is None or (self.ps.ep_size == 1 and not self.row_parameters):
            return self._base__grad_norm(parameters, gradients)
        # Dense gradients are replicated. Count each dense owner once and
        # sum the disjoint expert shards across EP, not expert-DP replicas.
        dense = torch.zeros((), dtype=torch.float64, device=parameters[0].device)
        expert = torch.zeros_like(dense)
        row = torch.zeros_like(dense)
        for p, grad in zip(parameters, gradients, strict=True):
            if grad is not None:
                target = (
                    row
                    if id(p) in self.row_ids
                    else expert if id(p) in self.expert_ids else dense
                )
                target.add_(grad.double().square().sum())
        if self.ps.ep_size > 1:
            torch.distributed.all_reduce(expert, group=self.ps.ep_group)
        if self.row_parameters:
            torch.distributed.all_reduce(row, group=self.row_group)
        return (dense + expert + row).sqrt()

    def _all_finite(self, valid):
        if self.dp_group is None and (
            self.ps is None or (self.ps.ep_size == 1 and self.row_group is None)
        ):
            return valid
        flag = torch.tensor(int(valid), device=next(self.model.parameters()).device)
        torch.distributed.all_reduce(
            flag, op=torch.distributed.ReduceOp.MIN, group=self.dp_group
        )
        return bool(flag.item())

    def accumulate_modality_loads(self, loads):
        """One forward snapshot: global layer order, then packed sample order."""
        for index, (previous, entries) in enumerate(
            zip(self._modality_loads, loads, strict=True)
        ):
            for stats in entries:
                previous = self._stats_factory(
                    (
                        stats.counts.detach().clone()
                        if previous is None
                        else previous.counts + stats.counts
                    ),
                    (
                        stats.total_tokens.detach().clone()
                        if previous is None
                        else previous.total_tokens + stats.total_tokens
                    ),
                )
            self._modality_loads[index] = previous

    def zero_grad(self, set_to_none=True):
        self._base_zero_grad(set_to_none=set_to_none)
        self._modality_loads = [None] * len(self.routers)

    def step(self):
        result = self._base_step()
        if result[0]:
            for router, stats in zip(self.routers, self._modality_loads, strict=True):
                if stats is not None:
                    if self.dp_group is not None:
                        torch.distributed.all_reduce(stats.counts, group=self.dp_group)
                        torch.distributed.all_reduce(
                            stats.total_tokens, group=self.dp_group
                        )
                    router.update_bias(stats)
        # Successful publication and overflow skips both finish this window.
        # Exceptions from the transactional backend retain statistics for retry.
        self._modality_loads = [None] * len(self.routers)
        return result

    def reconfigure_vision(self, mask):
        """Change a completed training stage, retaining common owners' momentum.

        Frozen owners release their state. Newly trainable owners start with
        empty state. This is an explicit post-training transition, not an
        automatic pretraining unfreeze or LR schedule.
        """
        from megatron.lite.primitive.modules.vision_training import VisionTrainability

        if not isinstance(mask, VisionTrainability):
            raise TypeError('Expected explicit visual trainability mask')
        if self.dp_group is not None:
            raise NotImplementedError(
                'Rebuild the DP bundle when changing vision trainability'
            )
        self._validate_trainability()
        if any(
            p.grad is not None or getattr(p, 'main_grad', None) is not None
            for p in self.model.parameters()
        ):
            raise RuntimeError('Zero gradients before changing the training stage')
        previous = self.model.vision_trainability
        try:
            mask.apply(self.model)
            candidate = self._rebuild()
        except Exception:
            previous.apply(self.model)
            raise
        old_states = {
            (id(p), type(backend)): state
            for backend in self.optimizers
            for p, state in backend.state.items()
        }
        for backend in candidate.optimizers:
            for group in backend.param_groups:
                for p in group['params']:
                    old = old_states.get((id(p), type(backend)))
                    if old is not None:
                        backend.state[p] = deepcopy(old)
        self.optimizers, self.tables = candidate.optimizers, candidate.tables

    def _validate_trainability(self):
        schedule = getattr(self.model, 'vision_schedule', None)
        if schedule is not None and schedule.stage != 'idle':
            raise RuntimeError('Optimizer requires completed vision backward')
        expected = {id(p) for p in self.model.parameters() if p.requires_grad}
        actual = {id(p) for g in self.param_groups for p in g['params']}
        if actual != expected:
            raise ValueError(
                'Trainability changed; rebuild optimizer groups before training'
            )

    def _base___init__(self, groups, config, tables=()):
        from .sinkhorn import Sinkhorn

        if not groups or any(
            p.dtype != torch.float32 for g in groups for p in g['params']
        ):
            raise ValueError('Mixed optimizer requires native FP32 parameter masters')
        for group in groups:
            for p in group['params']:
                if not hasattr(p, '_native_main_grad_hook'):
                    p.main_grad = p.grad
                    p._native_main_grad_hook = p.register_post_accumulate_grad_hook(
                        _publish_main_grad
                    )
        self.config = config
        self.optimizers = []
        for algorithm, optimizer in (
            ('muon', HeadwiseMuon),
            ('sinkhorn', Sinkhorn),
            ('adamw', torch.optim.AdamW),
        ):
            selected = [g for g in groups if g['algorithm'] == algorithm]
            if selected:
                kwargs = {}
                if algorithm == 'muon':
                    kwargs = dict(
                        ns_steps=config.ns_steps,
                        coefficient_type=config.coefficient_type,
                    )
                elif algorithm == 'adamw':
                    kwargs = dict(betas=(0.9, 0.95), eps=1e-20, foreach=False)
                self.optimizers.append(optimizer(selected, lr=config.lr, **kwargs))
        self.tables = list(tables)

    @property
    def param_groups(self):
        return [g for o in self.optimizers for g in o.param_groups]

    def _base_zero_grad(self, set_to_none=True):
        for backend in self.optimizers:
            backend.zero_grad(set_to_none=set_to_none)
        for p in _parameters(self):
            p.main_grad = p.grad

    def _base__grad_norm(self, parameters, gradients):
        active = [g for g in gradients if g is not None]
        return (
            torch.stack([g.double().square().sum() for g in active]).sum().sqrt()
            if active
            else torch.tensor(0.0)
        )

    def _finite(self, tensors):
        return self._all_finite(
            not any(not torch.isfinite(tensor).all() for tensor in tensors)
        )

    @torch.no_grad()
    def _base_step(self):
        self._validate_trainability()
        parameters = _parameters(self)
        gradients = [
            p.main_grad if getattr(p, 'main_grad', None) is not None else p.grad
            for p in parameters
        ]
        active = [g for g in gradients if g is not None]
        if any(g.dtype != torch.float32 or g.is_sparse for g in active):
            raise ValueError('Expected native dense FP32 main_grad')
        norm = self._grad_norm(parameters, gradients)
        if not self._all_finite(bool(torch.isfinite(norm))):
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
                    for p, q in _pairs(backend, candidate):
                        q.grad = p.grad
                    candidate.step()
                    candidates.update((id(p), q) for p, q in _pairs(backend, candidate))
                    if not self._finite(
                        v
                        for s in candidate.state.values()
                        for v in s.values()
                        if isinstance(v, torch.Tensor)
                    ):
                        return False, float(norm), None
                    staged_adam.append((backend, candidate))
                else:
                    if not self._all_finite(backend.prepare_step()):
                        return False, float(norm), None
                    candidates.update(
                        (id(p), value) for p, value in backend.candidates()
                    )
            if not self._finite(candidates.values()):
                return False, float(norm), None
            storage = []
            for table in self.tables:
                weight, scale = quantize_block_fp8(
                    candidates[id(table.master)], (1, 32), scale_format='e8m0'
                )
                if not self._finite(value.float() for value in (weight, scale)):
                    return False, float(norm), None
                storage.append((table, weight, scale))
            for backend in self.optimizers:
                if not isinstance(backend, torch.optim.AdamW):
                    backend.commit_step()
            for backend, candidate in staged_adam:
                for p, q in _pairs(backend, candidate):
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
        raise RuntimeError('Native gradient producer did not return native FP32')
    parameter.main_grad = parameter.grad
