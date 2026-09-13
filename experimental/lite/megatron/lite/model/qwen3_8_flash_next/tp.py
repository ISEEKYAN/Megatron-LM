# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Projection-column TP with replicated decoder consumers and ETP=1 experts.

Gather each projection's complete output before its original consumer. Thus the
GDN six-section layout, QSA query/gate pairing and shared SwiGLU layout remain
unchanged. Only projection weights are sharded; HC/PLE, recurrence and routing
are replicated. The shared primitive sums input dgrad across TP exactly once.
"""
import re


def _column_classes():
    from megatron.lite.primitive.parallel.linear import (
        ColumnParallelLinear,
        VanillaColumnParallelLinear,
    )

    return ColumnParallelLinear, VanillaColumnParallelLinear


def projection_shard(name):
    return bool(
        re.fullmatch(
            r'(lm_head|layers\.\d+\.(linear_attn\.(in_proj|o_proj)|'
            r'self_attn\.[qkvo]_proj|mlp\.shared_expert\.(gate_up|down)))\.linear\.weight',
            name,
        )
    )


def serial_parameter_name(name):
    if name == 'lm_head.linear.weight' or re.fullmatch(
        r'layers\.\d+\.self_attn\.[qkvo]_proj\.linear\.weight', name
    ):
        return name.replace('.linear.weight', '.weight')
    return name


def parallelize_projections(model, ps):
    column, vanilla = _column_classes()
    targets = [(model, 'lm_head', False)]
    for layer in model.layers:
        if layer.linear_attn is not None:
            targets += [(layer.linear_attn, key, True) for key in ('in_proj', 'o_proj')]
        if layer.self_attn is not None:
            targets += [
                (layer.self_attn, key, False)
                for key in ('q_proj', 'k_proj', 'v_proj', 'o_proj')
            ]
        targets += [(layer.mlp.shared_expert, key, True) for key in ('gate_up', 'down')]
    for parent, key, use_te in targets:
        original = getattr(parent, key)
        weight = original.linear.weight if use_te else original.weight
        dout, din = weight.shape
        # A gathered column output preserves full-matrix consumer semantics.
        replacement = (column if use_te else vanilla)(din, dout, ps, gather_output=True)
        setattr(parent, key, replacement)
    for name, parameter in model.named_parameters():
        # Full-gradient replicas must be counted once, never SP-summed.
        parameter.tensor_model_parallel = projection_shard(name)
        parameter.sequence_parallel = False


def finalize_replicated_experts(chunks, finalize, tp_size):
    """EDP sums identical TP token replicas; average them once before stepping.

    MCore's default expert scale assumes SP partitions the tokens. Here only
    projections are partitioned, so replicated experts see every token on each
    TP rank. Use the public buffer scaling API after reduction has completed.
    """
    finalize()
    for chunk in chunks:
        for buffer in chunk.expert_parallel_buffers:
            buffer.scale_gradients(1.0 / tp_size)
