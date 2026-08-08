# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import inspect

import pytest

from megatron.lite.primitive.modules.moe_ep_chunk_overlap_policy import (
    ep_chunk_ranges,
    recompute_modules_for_ep_chunk_overlap,
    validate_ep_chunk_overlap_config,
)


def test_fixed_two_chunk_ranges_cover_each_token_exactly_once():
    ranges = ep_chunk_ranges(11)

    assert ranges == [(0, 6), (6, 11)]
    assert [token for start, end in ranges for token in range(start, end)] == list(
        range(11)
    )


@pytest.mark.parametrize("tokens", [0, 1])
def test_fixed_two_chunk_ranges_fail_loud_when_both_chunks_cannot_be_live(tokens):
    with pytest.raises(ValueError, match="at least two tokens"):
        ep_chunk_ranges(tokens)


def test_chunked_full_recompute_replaces_the_outer_moe_checkpoint():
    requested = ["core_attn", "moe", "mlp"]

    assert recompute_modules_for_ep_chunk_overlap(requested, enabled=False) == requested
    assert recompute_modules_for_ep_chunk_overlap(requested, enabled=True) == [
        "core_attn",
        "mlp",
    ]
    assert requested == ["core_attn", "moe", "mlp"]


def test_chunked_full_recompute_checkpoints_attention_without_nesting_moe():
    assert recompute_modules_for_ep_chunk_overlap(["full"], enabled=True) == ["attn"]


@pytest.mark.parametrize(
    "enabled,use_deepep,ep_size,topk",
    [(False, False, 1, 8), (False, True, 8, 8), (True, True, 8, 8)],
)
def test_qwen_model_contract_accepts_only_supported_combinations(
    enabled, use_deepep, ep_size, topk
):
    assert (
        validate_ep_chunk_overlap_config(
            enabled, use_deepep=use_deepep, ep_size=ep_size, topk=topk
        )
        is enabled
    )


@pytest.mark.parametrize(
    "enabled,use_deepep,ep_size",
    [(True, False, 8), (True, True, 1)],
)
def test_qwen_model_contract_fails_loud(enabled, use_deepep, ep_size):
    with pytest.raises(ValueError):
        validate_ep_chunk_overlap_config(
            enabled, use_deepep=use_deepep, ep_size=ep_size, topk=1
        )


def test_backward_op_orders_dgrad_before_delayed_wgrad(transformer_engine_import_stub):
    transformer_engine_import_stub()
    from megatron.lite.primitive.modules.moe_ep_chunk_overlap import EPChunkBackwardOp

    source = inspect.getsource(EPChunkBackwardOp._full_recompute_fused_backward_v6)

    assert source.index("pending_dispatch_bwd.append") < source.index(
        "flush_delayed_weight_grads"
    )
    assert source.index("flush_delayed_weight_grads") < source.index(
        "finish_deepep_dispatch_backward"
    )
    assert "torch.cuda.synchronize" not in source
