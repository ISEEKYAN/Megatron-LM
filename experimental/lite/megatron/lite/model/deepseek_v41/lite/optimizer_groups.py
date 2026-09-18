# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Object-based V4.1 routing and replicated transactional optimizer assembly.

Release keys are audit labels, never dispatch inputs. Unknown active objects
fail closed. Vision execution/its explicit trainability mask belong to the
multimodal assembly; frozen visual and archival MTP owners allocate no state.
"""


from megatron.lite.primitive.optimizers.parameter_groups import OwnedParameterGroups
from megatron.lite.primitive.optimizers.partitioned_mixed import (
    PartitionedMixedOptimizer,
)
from megatron.lite.primitive.optimizers.vision_config import (
    OptimizerConfig,
    VisionOptimizerConfig,
)

# Role -> (matrix algorithm, matrix decay, vector decay, LR multiplier).
# Roles are validated against an independently enumerated object inventory below.
_RULES = {
    **dict.fromkeys(
        'wq_a wq_b wkv wo_a wo_b router expert shared_expert compressor'.split(),
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
    builder = OwnedParameterGroups(model, _RULES, lr)
    add, route, visual_linear = builder.add, builder.route, builder.visual_linear
    bindings, seen = builder.bindings, builder.seen
    route(model, 'embed.weight', 'embedding')
    route(model, 'head.weight', 'head')
    route(model, 'norm.weight', 'norm')
    for block in model.layers:
        a, c = block.attn, block.attn.config
        for role in ('wq_a', 'wkv', 'wo_a', 'wo_b'):
            route(a, role + '.weight', role)
        route(
            a,
            'wq_b.weight',
            'wq_b',
            shape=(c.heads, c.head_dim, c.q_rank),
            heads=c.heads,
        )
        route(a, 'q_norm.weight kv_norm.weight', 'norm')
        route(a, 'attn_sink', 'attention_sink')
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
            route(e, 'wkv.weight', 'engram_projection')
            route(e, 'q_weight k_weight', 'engram_norm')
    vision = model.vision
    if hasattr(vision, 'patch_embed'):
        encoder_active = any(
            p.requires_grad
            for m in (vision.patch_embed, vision.blocks)
            for p in m.parameters()
        )
        vectors = (model.image_start, model.image_end, model.image_newline)
        if (encoder_active or any(p.requires_grad for p in vectors)) and not isinstance(
            vision_policy, VisionOptimizerConfig
        ):
            raise ValueError(
                'Active encoder/image vectors require an explicit visual optimizer policy'
            )
        multiplier = vision_policy.encoder_lr_multiplier if encoder_active else 1

        visual_linear(vision.patch_embed.proj, 'vision', multiplier=multiplier)
        for block in vision.blocks:
            a, dim = block.attn, block.attn.wqkv.in_features
            partitions = (
                (a.n_heads, a.head_dim, dim),
                (a.n_heads, a.head_dim, dim),
                (dim, dim),
            )
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
    return builder.finish()


def _optimizer_owners(model):
    experts = [
        b.tensor
        for b in model.parameter_bindings()
        if b.role == 'expert' and b.tensor.requires_grad
    ]
    routers = [block.ffn.gate for block in model.layers]
    tables = [
        b.engram.embed
        for b in model.layers
        if b.engram is not None and b.engram.embed.master is not None
    ]
    return experts, routers, tables, model.engram_group


class V41Optimizer(PartitionedMixedOptimizer):
    def __init__(self, model, config, *, dp_group=None, ps=None):
        if not isinstance(config, OptimizerConfig):
            raise TypeError('V4.1 requires an explicit model OptimizerConfig')
        from .moe import ModalityLoad

        super().__init__(
            model,
            config,
            dp_group=dp_group,
            ps=ps,
            group_builder=lambda: parameter_groups(
                model, lr=config.lr, vision_policy=config.vision_policy
            ),
            owners=lambda: _optimizer_owners(model),
            stats_factory=ModalityLoad,
            rebuild=lambda: type(self)(model, config, dp_group=dp_group),
        )
        self.engram_parameters, self.engram_ids = self.row_parameters, self.row_ids
