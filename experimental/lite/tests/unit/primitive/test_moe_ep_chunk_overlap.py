# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import inspect
from types import MethodType, SimpleNamespace

import pytest
import torch

from megatron.lite.primitive.modules.moe_ep_chunk_overlap_policy import (
    ep_chunk_ranges,
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


@pytest.mark.parametrize(
    "enabled,use_deepep,ep_size,topk",
    [(False, False, 1, 8), (False, True, 8, 8), (True, True, 8, 8)],
)
def test_qwen_model_contract_accepts_only_supported_combinations(
    enabled, use_deepep, ep_size, topk
):
    assert (
        validate_ep_chunk_overlap_config(
            enabled,
            use_deepep=use_deepep,
            ep_size=ep_size,
            topk=topk,
            max_token_rows_per_rank=4096 if enabled else None,
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


def test_three_ops_have_distinct_saved_context_and_recompute_contracts(
    transformer_engine_import_stub,
):
    transformer_engine_import_stub()
    from megatron.lite.primitive.modules.moe_ep_chunk_overlap import (
        EPChunkBackwardOp,
        EPChunkForwardOp,
        EPChunkFusedForwardBackwardOp,
    )

    forward_source = inspect.getsource(EPChunkForwardOp)
    backward_source = inspect.getsource(EPChunkBackwardOp)
    fused_source = inspect.getsource(EPChunkFusedForwardBackwardOp)
    primitive_source = inspect.getsource(
        inspect.getmodule(EPChunkFusedForwardBackwardOp)
    )

    assert "forward-only" not in forward_source
    assert "_SavedContextEPChunkFunction.apply" in forward_source
    assert "context" in inspect.signature(EPChunkBackwardOp.backward).parameters
    assert "_full_recompute" not in backward_source
    assert "backward_op" not in fused_source
    assert "_full_recompute_fused_backward" in fused_source
    assert "moe_act_recompute" not in primitive_source


def test_three_ops_direct_behavior_contracts(transformer_engine_import_stub):
    transformer_engine_import_stub()
    from megatron.lite.primitive.modules.moe_ep_chunk_overlap import (
        EPChunkBackwardOp,
        EPChunkForwardOp,
        EPChunkFusedForwardBackwardOp,
    )

    def workspace(op):
        dispatchers = [SimpleNamespace(use_deepep=True) for _ in range(2)]
        return SimpleNamespace(
            key=SimpleNamespace(op=op),
            dispatcher=lambda slot: dispatchers[slot],
            reset_tensors=lambda **_kwargs: None,
        )

    router = torch.nn.Identity()
    experts = torch.nn.Identity()
    backward = EPChunkBackwardOp(
        router=router, experts=experts, workspace=workspace("backward")
    )
    calls = {"forward": 0, "backward": 0, "fused": 0}
    marker = object()

    def saved_forward(_self, x, _ranges, input_shape, _dtype):
        calls["forward"] += 1
        return (x * 2).view(input_shape), marker

    def saved_backward(_self, context, grad):
        assert context is marker
        assert torch.is_grad_enabled() is True
        calls["backward"] += 1
        return grad * 2, [], []

    backward._saved_context_backward = MethodType(saved_backward, backward)
    forward = EPChunkForwardOp(
        router=router,
        experts=experts,
        workspace=workspace("forward"),
        backward_op=backward,
    )
    forward._forward_saved_context_async = MethodType(saved_forward, forward)
    forward._forward_output_async = MethodType(
        lambda _self, x, _ranges, input_shape, _dtype, **_kwargs: (x * 2).view(
            input_shape
        ),
        forward,
    )
    value = torch.randn(2, 4, requires_grad=True)
    output = forward(value)
    output.sum().backward()
    torch.testing.assert_close(output, value.detach() * 2)
    torch.testing.assert_close(value.grad, torch.full_like(value, 2))
    assert calls == {"forward": 1, "backward": 1, "fused": 0}
    with torch.no_grad():
        inference = forward(value.detach())
    torch.testing.assert_close(inference, value.detach() * 2)

    fused = EPChunkFusedForwardBackwardOp(
        router=router,
        experts=experts,
        workspace=workspace("fused_forward_backward"),
    )

    def fused_forward_backward(_self, _x, grad):
        calls["fused"] += 1
        return grad * 3, [], []

    fused._full_recompute_fused_backward = MethodType(fused_forward_backward, fused)
    grad_x, router_grads, expert_grads = fused.forward_backward(
        value.detach(), torch.ones_like(value)
    )
    torch.testing.assert_close(grad_x, torch.full_like(value, 3))
    assert router_grads == []
    assert expert_grads == []
    assert calls["fused"] == 1
