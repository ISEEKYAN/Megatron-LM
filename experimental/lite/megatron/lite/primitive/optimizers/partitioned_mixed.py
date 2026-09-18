# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Transactional mixed optimization with explicit partitioned-owner policies."""

import math
from copy import deepcopy

import torch

from .headwise_muon import MixedOptimizer


class PartitionedMixedOptimizer(MixedOptimizer):
    """Local correctness coordinator; every backend publishes on one commit.

    AdamW staging uses the real torch backend on private candidate parameters.
    It deliberately costs extra local storage. Distributed staging/sharding is
    owned by the parallel integration rather than silently approximated here.
    """

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
        self._group_builder, self._owners = group_builder, owners
        self.model = model
        self.dp_group = dp_group
        self.ps = ps
        self.expert_parameters, self.routers, tables, self.row_group = owners()
        self.expert_ids = {id(p) for p in self.expert_parameters}
        self._stats_factory = stats_factory
        self._modality_loads = [None] * len(self.routers)
        super().__init__(groups, config, tables)
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
            return super()._grad_norm(parameters, gradients)
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
        if self.ps is None or (self.ps.ep_size == 1 and self.row_group is None):
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
        super().zero_grad(set_to_none=set_to_none)
        self._modality_loads = [None] * len(self.routers)

    def step(self):
        result = super().step()
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
            candidate = (
                self._rebuild()
                if self._rebuild is not None
                else type(self)(
                    self.model,
                    self.config,
                    dp_group=self.dp_group,
                    ps=self.ps,
                    group_builder=self._group_builder,
                    owners=self._owners,
                    stats_factory=self._stats_factory,
                )
            )
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
