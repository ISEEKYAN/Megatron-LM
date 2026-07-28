# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import inspect

import pytest
import torch
from megatron.lite.primitive.modules.moe_ep_chunk_overlap_policy import (
    ep_chunk_ranges,
    parse_ep_chunk_spec,
    recompute_modules_for_ep_chunk_overlap,
    resolve_ep_chunk_overlap_chunks,
    validate_ep_chunk_overlap_config,
)


@pytest.mark.parametrize("value, expected", [(1, 1), (2, 2), ("2", 2)])
def test_chunk_spec_is_explicit_and_validated(value, expected):
    assert parse_ep_chunk_spec(value) == expected


@pytest.mark.parametrize("value", [0, -1, 3, 4, "auto", "zero", "1.5"])
def test_chunk_spec_rejects_invalid_values(value):
    with pytest.raises(ValueError):
        parse_ep_chunk_spec(value)


def test_chunk_ranges_cover_each_token_exactly_once():
    ranges = ep_chunk_ranges(11, 4)

    assert ranges == [(0, 3), (3, 6), (6, 9), (9, 11)]
    assert [token for start, end in ranges for token in range(start, end)] == list(
        range(11)
    )


def test_forward_and_backward_share_the_same_closed_chunk_count():
    for tokens in (8192, 16384, 32768):
        assert (
            resolve_ep_chunk_overlap_chunks(tokens, ep_size=8, hidden_size=4096, spec=2)
            == 2
        )


def test_chunked_full_recompute_replaces_the_outer_moe_checkpoint():
    requested = ["core_attn", "moe", "mlp"]

    assert recompute_modules_for_ep_chunk_overlap(requested, num_chunks=1) == requested
    assert recompute_modules_for_ep_chunk_overlap(requested, num_chunks=2) == [
        "core_attn",
        "mlp",
    ]
    assert requested == ["core_attn", "moe", "mlp"]


def test_chunked_full_recompute_checkpoints_attention_without_nesting_moe():
    assert recompute_modules_for_ep_chunk_overlap(["full"], num_chunks=2) == [
        "attn"
    ]


def test_chunked_backward_submits_dgrad_before_te_delayed_wgrad(
    transformer_engine_import_stub,
):
    transformer_engine_import_stub()
    from megatron.lite.primitive.modules.moe_ep_chunk_overlap import (
        EPChunkOverlapOperator,
    )

    source = inspect.getsource(EPChunkOverlapOperator._full_recompute_fused_backward_v6)

    assert source.count("torch.autograd.grad(") == 2
    assert source.index("expert_grads =") < source.index(
        "pending_dispatch_bwd.append"
    )
    assert source.index("pending_dispatch_bwd.append") < source.index(
        "pop_delayed_weight_grads"
    )
    assert "retain_graph=True" not in source


def test_chunked_ep_keeps_compute_on_the_autograd_caller_stream(
    transformer_engine_import_stub,
):
    transformer_engine_import_stub()
    from megatron.lite.primitive.modules.moe_ep_chunk_overlap import (
        EPChunkOverlapOperator,
    )

    source = inspect.getsource(EPChunkOverlapOperator._streams)

    assert "torch.cuda.current_stream(device)" in source


def test_chunked_ep_accumulates_parameter_grads_inside_backward(
    transformer_engine_import_stub,
):
    transformer_engine_import_stub()
    from megatron.lite.primitive.modules.moe_ep_chunk_overlap import (
        EPChunkOverlapOperator,
        _FullRecomputeFused,
    )

    forward_source = inspect.getsource(_FullRecomputeFused.forward)
    backward_source = inspect.getsource(_FullRecomputeFused.backward)
    operator_source = inspect.getsource(EPChunkOverlapOperator._forward_impl)

    assert "*params" not in forward_source
    assert "*params" not in operator_source
    assert "_accumulate_parameter_grads" in backward_source


@pytest.mark.parametrize(
    "chunks,use_deepep,ep_size", [(1, False, 1), (1, True, 8), (2, True, 8)]
)
def test_shared_model_contract_accepts_only_supported_combinations(
    chunks, use_deepep, ep_size
):
    assert (
        validate_ep_chunk_overlap_config(chunks, use_deepep=use_deepep, ep_size=ep_size)
        == chunks
    )


@pytest.mark.parametrize(
    "chunks,use_deepep,ep_size",
    [(0, True, 8), (3, True, 8), (2, False, 8), (2, True, 1)],
)
def test_shared_model_contract_fails_loud(chunks, use_deepep, ep_size):
    with pytest.raises(ValueError):
        validate_ep_chunk_overlap_config(chunks, use_deepep=use_deepep, ep_size=ep_size)


def test_production_chunked_ep_operator_is_importable(transformer_engine_import_stub):
    transformer_engine_import_stub()
    from megatron.lite.primitive.modules.moe_ep_chunk_overlap import (
        EPChunkOverlapOperator,
    )

    assert EPChunkOverlapOperator.__name__ == "EPChunkOverlapOperator"
    assert not issubclass(EPChunkOverlapOperator, torch.nn.Module)
    assert tuple(inspect.signature(EPChunkOverlapOperator).parameters) == (
        "router",
        "experts",
        "dispatcher",
        "forward_dispatchers",
        "router_forward",
    )


def test_operator_keeps_router_and_expert_checkpoint_names_model_owned(
    transformer_engine_import_stub,
):
    transformer_engine_import_stub()
    from megatron.lite.primitive.modules.moe_ep_chunk_overlap import (
        EPChunkOverlapOperator,
    )

    owner = torch.nn.Module()
    owner.router = torch.nn.Linear(2, 2, bias=False)
    owner.experts = torch.nn.Linear(2, 2, bias=False)
    operator = object.__new__(EPChunkOverlapOperator)
    operator.router = owner.router
    operator.experts = owner.experts
    owner.ep_chunk_overlap = operator

    assert tuple(owner.state_dict()) == ("router.weight", "experts.weight")
    assert "ep_chunk_overlap" not in owner._modules
