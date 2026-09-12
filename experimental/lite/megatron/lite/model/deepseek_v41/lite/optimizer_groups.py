# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Object-based V4.1 routing and single-rank transactional optimizer assembly.

Release keys are audit labels, never dispatch inputs. Unknown active objects
fail closed. Vision execution/its explicit trainability mask belong to the
multimodal assembly; frozen visual and archival MTP owners allocate no state.
"""

import math
from copy import deepcopy
from dataclasses import dataclass
from operator import attrgetter

import torch
from megatron.lite.primitive.optimizers.headwise_muon import MixedOptimizer


@dataclass(frozen=True)
class VisionOptimizerConfig:
    """Caller-selected post-training LR/decay, not pretraining defaults.

    Image vectors retain the DS4 non-matrix AdamW representation. They are
    never reshaped into Muon/Sinkhorn matrices. All numeric policy is explicit.
    """

    encoder_lr_multiplier: float
    image_vector_lr_multiplier: float
    image_vector_weight_decay: float

    def __post_init__(self):
        if any(not math.isfinite(value) or value < 0 for value in vars(self).values()):
            raise ValueError('Visual optimizer policy requires finite nonnegative values')


@dataclass(frozen=True)
class OptimizerConfig:
    lr: float
    ns_steps: int
    coefficient_type: str
    clip_grad: float = 1.0
    vision_policy: VisionOptimizerConfig | None = None


# Role -> (matrix algorithm, matrix decay, vector decay, LR multiplier).
# Roles are validated against an independently enumerated object inventory below.
_RULES = {
    **dict.fromkeys(
        ('wq_a', 'wq_b', 'wkv', 'wo_a', 'wo_b', 'router', 'expert', 'shared_expert', 'compressor'),
        ('muon', 0.1, 0.1, 1),
    ),
    'embedding': ('sinkhorn', 0, 0, 1),
    'head': ('sinkhorn', 0, 0, 1),
    'norm': ('adamw', 0.1, 0.1, 1),
    'attention_sink': ('adamw', 0, 0, 1),
    'hyper_connection': ('muon', 0.1, 0, 1),
    'engram_table': ('sinkhorn', 0, 0, 5),
    'engram_projection': ('muon', 0.1, 0.1, 5),
    'engram_norm': ('adamw', 0.1, 0.1, 5),
    'vision': ('muon', 0.1, 0.1, 1),
    'aligner': ('muon', 0.1, 0.1, 1),
    'image_delimiter': ('adamw', 0, 0, 1),
}


def parameter_groups(model, *, lr, vision_policy=None):
    if not math.isfinite(lr) or lr < 0:
        raise ValueError('Invalid base learning rate')
    model.validate_parameter_bindings()
    bindings = {id(b.tensor): b for b in model.parameter_bindings()}
    registered = list(model.named_parameters(remove_duplicate=False))
    if len(registered) != len({id(p) for _, p in registered}):
        raise ValueError('Unexpected parameter alias in module tree')
    groups, seen = [], set()

    def add(p, role, *, shape=None, heads=None, partitions=None, vector=False, **policy):
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
        algorithm, matrix_decay, vector_decay, multiplier = _RULES[role]
        # Selection follows logical matrix rank, never a release-key prefix.
        if vector:
            algorithm = 'adamw'
        if algorithm in ('muon', 'sinkhorn') and p.ndim != 2:
            raise ValueError('Matrix optimizer requires the declared two-dimensional owner')
        if shape is not None and math.prod(shape) != p.numel():
            raise ValueError('Logical matrix shape disagrees with actual owner')
        if (
            shape is not None
            and len(shape) == 3
            and tuple(p.shape) != (shape[0] * shape[1], shape[2])
        ):
            raise ValueError('Head layout disagrees with physical matrix axes')
        decay = vector_decay if vector else matrix_decay
        groups.append(
            dict(
                params=[p],
                algorithm=algorithm,
                owner_key=b.release_key,
                matrix_shape=tuple(p.shape) if shape is None else tuple(shape),
                matrix_partitions=partitions,
                lr=lr * policy.get('multiplier', multiplier),
                weight_decay=policy.get('decay', decay),
            )
        )

    def route(owner, paths, role, **policy):
        for path in paths.split():
            add(attrgetter(path)(owner), role, **policy)

    for paths, role in (
        ('embed.weight', 'embedding'),
        ('head.weight', 'head'),
        ('norm.weight', 'norm'),
    ):
        route(model, paths, role)
    for block in model.layers:
        a, c = block.attn, block.attn.config
        for role in ('wq_a', 'wkv', 'wo_a', 'wo_b'):
            route(a, role + '.weight', role)
        route(a, 'wq_b.weight', 'wq_b', shape=(c.heads, c.head_dim, c.q_rank), heads=c.heads)
        for owner, paths, role in (
            (a, 'q_norm.weight kv_norm.weight', 'norm'),
            (a, 'attn_sink', 'attention_sink'),
        ):
            route(owner, paths, role)
        if a.compressor is not None:
            route(a.compressor, 'wkv.weight', 'compressor')
            route(a.compressor, 'norm.weight', 'compressor', vector=True)
            if hasattr(a.compressor, 'wgate'):
                route(a.compressor, 'wgate.weight', 'compressor')
        if a.indexer is not None:
            for p in a.indexer.parameters():
                if p.requires_grad:
                    raise ValueError('Post-training indexer must remain frozen')
                if id(p) not in bindings or bindings[id(p)].role != 'indexer':
                    raise ValueError('Unknown indexer parameter')
                seen.add(id(p))
        route(block, 'attn_norm.weight ffn_norm.weight', 'norm')
        for mixes in (block.attn_mixes, block.ffn_mixes):
            route(mixes, 'fn', 'hyper_connection')
            route(mixes, 'base scale', 'hyper_connection', vector=True)
        route(block.ffn, 'gate.router.gate.weight', 'router')
        for role, experts in (
            ('expert', block.ffn.experts),
            ('shared_expert', [block.ffn.shared_experts]),
        ):
            for expert in experts:
                if expert is not None:
                    route(expert, 'w1.weight w2.weight w3.weight', role)
        if block.engram is not None:
            e = block.engram
            if e.embed.master is not None:
                route(e, 'embed.master', 'engram_table')
            for paths, role in (
                ('wkv.weight', 'engram_projection'),
                ('q_weight k_weight', 'engram_norm'),
            ):
                route(e, paths, role)
    vision = model.vision
    if hasattr(vision, 'patch_embed'):
        encoder_active = any(
            p.requires_grad for m in (vision.patch_embed, vision.blocks) for p in m.parameters()
        )
        vectors = (model.image_start, model.image_end, model.image_newline)
        if (encoder_active or any(p.requires_grad for p in vectors)) and not isinstance(
            vision_policy, VisionOptimizerConfig
        ):
            raise ValueError(
                'Active encoder/image vectors require an explicit visual optimizer policy'
            )
        multiplier = vision_policy.encoder_lr_multiplier if encoder_active else 1

        def visual_linear(module, role, *, multiplier=1, partitions=None):
            shape = (module.out_features, module.in_features)
            if math.prod(shape) != module.weight.numel() or tuple(module.weight.shape) != shape:
                raise ValueError('Visual linear shape disagrees with physical owner')
            if module.bias is not None and tuple(module.bias.shape) != (shape[0],):
                raise ValueError('Visual bias shape disagrees with physical owner')
            if partitions is not None and (
                any(any(type(d) is not int or d < 1 for d in part) for part in partitions)
                or sum(math.prod(part) for part in partitions) != module.weight.numel()
                or any(part[-1] != shape[-1] for part in partitions)
            ):
                raise ValueError('Logical partitions disagree with visual matrix shape')
            add(module.weight, role, shape=shape, multiplier=multiplier, partitions=partitions)
            if module.bias is not None:
                add(module.bias, role, multiplier=multiplier, decay=0, vector=True)

        visual_linear(vision.patch_embed.proj, 'vision', multiplier=multiplier)
        for block in vision.blocks:
            a, dim = block.attn, block.attn.wqkv.in_features
            partitions = ((a.n_heads, a.head_dim, dim), (a.n_heads, a.head_dim, dim), (dim, dim))
            for module, parts in (
                (a.wqkv, partitions),
                (a.wo, None),
                (block.mlp.w1, None),
                (block.mlp.w2, None),
            ):
                visual_linear(module, 'vision', multiplier=multiplier, partitions=parts)
            for module in (block.norm1, block.norm2):
                add(module.weight, 'vision', multiplier=multiplier, vector=True)
        route(vision, 'norm.weight', 'vision', vector=True)
        for module in (model.aligner.w1, model.aligner.w2):
            visual_linear(module, 'aligner')
        for vector in vectors:
            policy = (
                dict(
                    multiplier=vision_policy.image_vector_lr_multiplier,
                    decay=vision_policy.image_vector_weight_decay,
                )
                if vector.requires_grad
                else {}
            )
            add(vector, 'image_delimiter', **policy)
    if seen != set(bindings):
        raise ValueError('Unknown parameter owner; no catch-all optimizer route')
    return groups


class V41Optimizer(MixedOptimizer):
    """Local correctness coordinator; every backend publishes on one commit.

    AdamW staging uses the real torch backend on private candidate parameters.
    It deliberately costs extra local storage. Distributed staging/sharding is
    owned by the parallel integration rather than silently approximated here.
    """

    def __init__(self, model, config):
        if not isinstance(config, OptimizerConfig):
            raise TypeError('V4.1 requires an explicit model OptimizerConfig')
        if not math.isfinite(config.clip_grad) or config.clip_grad < 0:
            raise ValueError('Invalid gradient clipping threshold')
        groups = parameter_groups(model, lr=config.lr, vision_policy=config.vision_policy)
        self.model = model
        tables = [
            b.engram.embed
            for b in model.layers
            if b.engram is not None and b.engram.embed.master is not None
        ]
        super().__init__(groups, config, tables)
        from .training import RoutingStep

        if model.routing_step is None:
            model.routing_step = RoutingStep()

    def zero_grad(self, set_to_none=True):
        super().zero_grad(set_to_none=set_to_none)
        self.model.routing_step.clear()

    def step(self):
        result = super().step()
        try:
            if result[0]:
                self.model.routing_step.publish()
        finally:
            # Both committed and skipped steps consume their microbatch loads.
            self.model.routing_step.clear()
        return result

    def reconfigure_vision(self, mask):
        """Change a completed training stage, retaining common owners' momentum.

        Frozen owners release their state. Newly trainable owners start with
        empty state. This is an explicit post-training transition, not an
        automatic pretraining unfreeze or LR schedule.
        """
        from .training import VisionTrainability

        if not isinstance(mask, VisionTrainability):
            raise TypeError('Expected explicit visual trainability mask')
        self._validate_trainability()
        if any(
            p.grad is not None or getattr(p, 'main_grad', None) is not None
            for p in self.model.parameters()
        ):
            raise RuntimeError('Zero gradients before changing the training stage')
        previous = self.model.vision_trainability
        try:
            mask.apply(self.model)
            candidate = type(self)(self.model, self.config)
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
            raise ValueError('Trainability changed; rebuild optimizer groups before training')
