# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Object-based V4.1 routing and single-rank transactional optimizer assembly.

Release keys are audit labels, never dispatch inputs. Unknown active objects
fail closed. Vision execution/its explicit trainability mask belong to the
multimodal assembly; archival vision/MTP owners allocate no optimizer state.
"""

import math
from copy import deepcopy
from dataclasses import dataclass

import torch
import torch.distributed as dist
from megatron.lite.primitive.optimizers.headwise_muon import HeadwiseMuon
from megatron.lite.primitive.optimizers.sinkhorn import Sinkhorn
from megatron.lite.primitive.quantization.block_fp8 import quantize_block_fp8


@dataclass(frozen=True)
class OptimizerConfig:
    lr: float
    ns_steps: int
    coefficient_type: str
    clip_grad: float = 1.0


def parameter_groups(model, *, lr):
    if not math.isfinite(lr) or lr < 0:
        raise ValueError('Invalid base learning rate')
    model.validate_parameter_bindings()
    bindings = {id(b.tensor): b for b in model.parameter_bindings()}
    registered = list(model.named_parameters(remove_duplicate=False))
    if len(registered) != len({id(p) for _, p in registered}):
        raise ValueError('Unexpected parameter alias in module tree')
    groups, seen = [], set()

    def add(p, algorithm, *, role, shape=None, multiplier=1, decay=0.1, heads=None):
        if id(p) in seen:
            raise ValueError('Duplicate optimizer owner')
        seen.add(id(p))
        b = bindings.get(id(p))
        if b is None or b.role != role:
            raise ValueError('Unknown or mismatched parameter owner role')
        if b.head_count != heads:
            raise ValueError('Unresolved or incorrect logical head count')
        if not p.requires_grad:
            return
        if algorithm in ('muon', 'sinkhorn') and p.ndim != 2:
            raise ValueError(
                'Matrix optimizer requires the declared two-dimensional owner'
            )
        if shape is not None and math.prod(shape) != p.numel():
            raise ValueError('Logical matrix shape disagrees with actual owner')
        if (
            shape is not None
            and len(shape) == 3
            and tuple(p.shape) != (shape[0] * shape[1], shape[2])
        ):
            raise ValueError('Head layout disagrees with physical matrix axes')
        groups.append(
            dict(
                params=[p],
                algorithm=algorithm,
                owner_key=b.release_key,
                matrix_shape=tuple(p.shape) if shape is None else tuple(shape),
                lr=lr * multiplier,
                weight_decay=decay,
            )
        )

    def norm(module, role='norm'):
        add(module.weight, 'adamw', role=role)

    if model.embed is not None:
        add(model.embed.weight, 'sinkhorn', role='embedding', decay=0)
    if model.head is not None:
        add(model.head.weight, 'sinkhorn', role='head', decay=0)
        norm(model.norm)
    for block in model.layers:
        if block is None:
            continue
        a, c = block.attn, block.attn.config
        for module, role in (
            (a.wq_a, 'wq_a'),
            (a.wkv, 'wkv'),
            (a.wo_a, 'wo_a'),
            (a.wo_b, 'wo_b'),
        ):
            add(module.weight, 'muon', role=role)
        add(
            a.wq_b.weight,
            'muon',
            role='wq_b',
            shape=(c.heads, c.head_dim, c.q_rank),
            heads=c.heads,
        )
        norm(a.q_norm)
        norm(a.kv_norm)
        add(a.attn_sink, 'adamw', role='attention_sink', decay=0)
        if a.compressor is not None:
            add(a.compressor.wkv.weight, 'muon', role='compressor')
            norm(a.compressor.norm, 'compressor')
            if hasattr(a.compressor, 'wgate'):
                add(a.compressor.wgate.weight, 'muon', role='compressor')
        if a.indexer is not None:
            for p in a.indexer.parameters():
                if p.requires_grad:
                    raise ValueError('Post-training indexer must remain frozen')
                if id(p) not in bindings or bindings[id(p)].role != 'indexer':
                    raise ValueError('Unknown indexer parameter')
                seen.add(id(p))
        norm(block.attn_norm)
        norm(block.ffn_norm)
        for mixes in (block.attn_mixes, block.ffn_mixes):
            add(mixes.fn, 'muon', role='hyper_connection')
            add(mixes.base, 'adamw', role='hyper_connection', decay=0)
            add(mixes.scale, 'adamw', role='hyper_connection', decay=0)
        add(block.ffn.gate.router.gate.weight, 'muon', role='router')
        for expert in block.ffn.experts:
            for module in (expert.w1, expert.w2, expert.w3):
                add(module.weight, 'muon', role='expert')
        shared = block.ffn.shared_experts
        if shared is not None:
            for module in (shared.w1, shared.w2, shared.w3):
                add(module.weight, 'muon', role='shared_expert')
        if block.engram is not None:
            e = block.engram
            if e.embed.master is not None:
                add(
                    e.embed.master,
                    'sinkhorn',
                    role='engram_table',
                    multiplier=5,
                    decay=0,
                )
            add(e.wkv.weight, 'muon', role='engram_projection', multiplier=5)
            for p in (e.q_weight, e.k_weight):
                add(p, 'adamw', role='engram_norm', multiplier=5)
    # F3a owners stay live for archive/export but text-only training freezes them.
    for module, role in ((model.vision, 'vision'), (model.aligner, 'aligner')):
        if module is not None:
            for p in module.parameters():
                if p.requires_grad:
                    raise ValueError(
                        'Trainable visual owners require multimodal optimizer routing'
                    )
                add(p, 'adamw', role=role)
    for attribute in ('image_start', 'image_end', 'image_newline'):
        p = getattr(model, attribute, None)
        if p is not None:
            if p.requires_grad:
                raise ValueError(
                    'Trainable image delimiters require multimodal optimizer routing'
                )
            add(p, 'adamw', role='image_delimiter')
    if seen != set(bindings):
        raise ValueError('Unknown parameter owner; no catch-all optimizer route')
    return groups


class V41Optimizer:
    """Local correctness coordinator; every backend publishes on one commit.

    AdamW staging uses the real torch backend on private candidate parameters.
    It deliberately costs extra local storage. Distributed staging/sharding is
    owned by the parallel integration rather than silently approximated here.
    """

    def __init__(self, model, config, *, parallel_state=None):
        if not isinstance(config, OptimizerConfig):
            raise TypeError('V4.1 requires an explicit model OptimizerConfig')
        if not math.isfinite(config.clip_grad) or config.clip_grad < 0:
            raise ValueError('Invalid gradient clipping threshold')
        groups = parameter_groups(model, lr=config.lr)
        if not groups or any(
            p.dtype != torch.float32 for g in groups for p in g['params']
        ):
            raise ValueError('V4.1 optimizer requires native FP32 parameter masters')
        self.config = config
        self.parallel_state = parallel_state
        self.commit_group = None if parallel_state is None else parallel_state.pp_group
        self.optimizers = []
        for algorithm in ('muon', 'sinkhorn', 'adamw'):
            selected = [g for g in groups if g['algorithm'] == algorithm]
            if not selected:
                continue
            if algorithm == 'muon':
                backend = HeadwiseMuon(
                    selected,
                    lr=config.lr,
                    ns_steps=config.ns_steps,
                    coefficient_type=config.coefficient_type,
                )
            elif algorithm == 'sinkhorn':
                backend = Sinkhorn(selected, lr=config.lr)
            else:
                backend = torch.optim.AdamW(
                    selected, lr=config.lr, betas=(0.9, 0.95), eps=1e-20, foreach=False
                )
            self.optimizers.append(backend)
        self.tables = [
            b.engram.embed
            for b in model.layers
            if b is not None
            and b.engram is not None
            and b.engram.embed.master is not None
        ]

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
        parameters = [p for g in self.param_groups for p in g['params']]
        gradients = [
            p.main_grad if getattr(p, 'main_grad', None) is not None else p.grad
            for p in parameters
        ]
        active = [g for g in gradients if g is not None]
        if any(g.dtype != torch.float32 or g.is_sparse for g in active):
            raise ValueError('Expected native dense FP32 main_grad')
        norm_squared = (
            torch.stack([g.double().square().sum() for g in active]).sum()
            if active
            else torch.zeros((), dtype=torch.float64, device=parameters[0].device)
        )
        if self.commit_group is not None:
            dist.all_reduce(norm_squared, group=self.commit_group)
        norm = norm_squared.sqrt()
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
        ready = True
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
                        ready = False
                    staged_adam.append((backend, candidate))
                else:
                    if not backend.prepare_step():
                        ready = False
                    else:
                        candidates.update(
                            (id(p), value) for p, value in backend.candidates()
                        )
            ready = ready and all(torch.isfinite(p).all() for p in candidates.values())
            if not self._all_ready(ready, parameters[0].device):
                return False, float(norm), None
            storage = []
            for table in self.tables:
                weight, scale = quantize_block_fp8(
                    candidates.get(id(table.master), table.master),
                    (1, 32),
                    scale_format='e8m0',
                )
                if (
                    not torch.isfinite(weight.float()).all()
                    or not torch.isfinite(scale.float()).all()
                ):
                    ready = False
                storage.append((table, weight, scale))
            if not self._all_ready(ready, parameters[0].device):
                return False, float(norm), None
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

    def _all_ready(self, ready, device):
        flag = torch.tensor(int(bool(ready)), dtype=torch.int32, device=device)
        if self.commit_group is not None:
            dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=self.commit_group)
        return bool(flag.item())

    def state_dict(self):
        return dict(
            owners=[g['owner_key'] for g in self.param_groups],
            clip_grad=self.config.clip_grad,
            optimizers=[o.state_dict() for o in self.optimizers],
        )

    def load_state_dict(self, state):
        if state.get('clip_grad') != self.config.clip_grad:
            raise ValueError('Optimizer clipping contract differs')
        if state.get('owners') != [g['owner_key'] for g in self.param_groups] or len(
            state['optimizers']
        ) != len(self.optimizers):
            raise ValueError('Optimizer owner layout differs')
        for backend, saved in zip(self.optimizers, state['optimizers']):
            backend.load_state_dict(saved)
