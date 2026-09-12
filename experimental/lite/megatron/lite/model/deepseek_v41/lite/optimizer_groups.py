# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Object-based V4.1 routing and single-rank transactional optimizer assembly.

Release keys are audit labels, never dispatch inputs. Unknown active objects
fail closed. Vision execution/its explicit trainability mask belong to the
multimodal assembly; frozen visual and archival MTP owners allocate no state.
"""

import math
from copy import deepcopy
from dataclasses import dataclass

import torch
from megatron.lite.primitive.optimizers.headwise_muon import HeadwiseMuon
from megatron.lite.primitive.optimizers.sinkhorn import Sinkhorn
from megatron.lite.primitive.quantization.block_fp8 import quantize_block_fp8


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
            raise ValueError(
                'Visual optimizer policy requires finite nonnegative values'
            )


@dataclass(frozen=True)
class OptimizerConfig:
    lr: float
    ns_steps: int
    coefficient_type: str
    clip_grad: float = 1.0
    vision_policy: VisionOptimizerConfig | None = None


def parameter_groups(model, *, lr, vision_policy=None):
    if not math.isfinite(lr) or lr < 0:
        raise ValueError('Invalid base learning rate')
    model.validate_parameter_bindings()
    bindings = {id(b.tensor): b for b in model.parameter_bindings()}
    registered = list(model.named_parameters(remove_duplicate=False))
    if len(registered) != len({id(p) for _, p in registered}):
        raise ValueError('Unexpected parameter alias in module tree')
    groups, seen = [], set()

    def add(
        p,
        algorithm,
        *,
        role,
        shape=None,
        multiplier=1,
        decay=0.1,
        heads=None,
        partitions=None
    ):
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
                matrix_partitions=partitions,
                lr=lr * multiplier,
                weight_decay=decay,
            )
        )

    def norm(module, role='norm'):
        add(module.weight, 'adamw', role=role)

    add(model.embed.weight, 'sinkhorn', role='embedding', decay=0)
    add(model.head.weight, 'sinkhorn', role='head', decay=0)
    norm(model.norm)
    for block in model.layers:
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
    # Enumerate live visual objects explicitly; frozen owners are still audited.
    vision = model.vision
    if hasattr(vision, 'patch_embed'):
        encoder_active = any(
            p.requires_grad
            for module in (vision.patch_embed, vision.blocks)
            for p in module.parameters()
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
            add(
                module.weight,
                'muon',
                role=role,
                shape=shape,
                multiplier=multiplier,
                partitions=partitions,
            )
            if module.bias is not None:
                add(module.bias, 'adamw', role=role, multiplier=multiplier, decay=0)

        visual_linear(vision.patch_embed.proj, 'vision', multiplier=multiplier)
        for block in vision.blocks:
            a = block.attn
            dim = a.wqkv.in_features
            partitions = (
                (a.n_heads, a.head_dim, dim),
                (a.n_heads, a.head_dim, dim),
                (dim, dim),
            )
            visual_linear(
                a.wqkv, 'vision', multiplier=multiplier, partitions=partitions
            )
            visual_linear(a.wo, 'vision', multiplier=multiplier)
            # Inherit the existing fused physical gate/up matrix, without a new split.
            for module in (block.mlp.w1, block.mlp.w2):
                visual_linear(module, 'vision', multiplier=multiplier)
            for module in (block.norm1, block.norm2):
                add(module.weight, 'adamw', role='vision', multiplier=multiplier)
        norm(vision.norm, 'vision')
        for module in (model.aligner.w1, model.aligner.w2):
            visual_linear(module, 'aligner')
        for vector in vectors:
            add(
                vector,
                'adamw',
                role='image_delimiter',
                multiplier=(
                    vision_policy.image_vector_lr_multiplier
                    if vector.requires_grad
                    else 1
                ),
                decay=(
                    vision_policy.image_vector_weight_decay
                    if vector.requires_grad
                    else 0
                ),
            )
    if seen != set(bindings):
        raise ValueError('Unknown parameter owner; no catch-all optimizer route')
    return groups


class V41Optimizer:
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
        groups = parameter_groups(
            model, lr=config.lr, vision_policy=config.vision_policy
        )
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
        self.model = model
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
            if b.engram is not None and b.engram.embed.master is not None
        ]

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

    @property
    def param_groups(self):
        return [g for o in self.optimizers for g in o.param_groups]

    def zero_grad(self, set_to_none=True):
        for backend in self.optimizers:
            backend.zero_grad(set_to_none=set_to_none)
        for group in self.param_groups:
            for p in group['params']:
                p.main_grad = p.grad

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
