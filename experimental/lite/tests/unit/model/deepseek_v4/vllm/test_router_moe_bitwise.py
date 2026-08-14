from __future__ import annotations

import importlib.util
from unittest.mock import Mock

import pytest
import torch

from megatron.lite.primitive.kernels import vllm_ds4
from megatron.lite.primitive.kernels.vllm_ds4 import (
    DS4TopKAdapter,
    FusedExpertsAdapter,
    GateLinearAdapter,
    HashRouteAdapter,
    SharedExpertsAdapter,
)
from megatron.lite.model.deepseek_v4.vllm.primitive.router import fixed_route_vjp


def test_gate_linear_cpu_contract_preserves_official_tuple_output() -> None:
    hidden = torch.randn(3, 8)
    gate = Mock(return_value=(torch.randn(3, 16), {"aux": 1}))
    output = GateLinearAdapter()(gate, hidden)
    assert output is gate.return_value[0]
    gate.assert_called_once_with(hidden)


def test_hash_router_cpu_contract_calls_exact_custom_op(monkeypatch) -> None:
    op = Mock()
    monkeypatch.setattr(vllm_ds4, "_symbol", lambda namespace, name: op)
    logits = torch.randn(3, 8, dtype=torch.float32)
    tokens = torch.tensor([1, 2, 3], dtype=torch.int32)
    table = torch.arange(64, dtype=torch.int32).view(32, 2) % 8
    weights, ids = HashRouteAdapter()(
        logits,
        tokens,
        table,
        topk=2,
        renormalize=True,
        routed_scaling_factor=1.5,
    )
    assert weights.shape == ids.shape == (3, 2)
    assert op.call_args.args[3] is logits
    assert op.call_args.args[7] is tokens
    assert op.call_args.args[8] is table


def test_learned_router_cpu_contract_calls_official_dsv4_topk(monkeypatch) -> None:
    expected = (
        torch.ones(3, 6, dtype=torch.float32),
        torch.zeros(3, 6, dtype=torch.int32),
    )
    op = Mock(return_value=expected)
    monkeypatch.setattr(vllm_ds4, "_symbol", lambda namespace, name: op)
    logits = torch.randn(3, 256, dtype=torch.float32)
    correction_bias = torch.randn(256, dtype=torch.float32)
    actual = DS4TopKAdapter()(
        logits,
        correction_bias,
        indices_dtype=torch.int32,
        routed_scaling_factor=1.5,
    )
    assert all(a is e for a, e in zip(actual, expected, strict=True))
    op.assert_called_once_with(logits, correction_bias, torch.int32, 1.5)


def test_local_and_shared_expert_cpu_contracts(monkeypatch) -> None:
    x = torch.randn(2, 8)
    w1 = torch.randn(4, 16, 8)
    w2 = torch.randn(4, 8, 8)
    weights = torch.ones(2, 2)
    ids = torch.zeros(2, 2, dtype=torch.int32)
    official = Mock(return_value=x + 1)
    monkeypatch.setattr(vllm_ds4, "_symbol", lambda module, name: official)
    output = FusedExpertsAdapter()(
        x, w1, w2, weights, ids, activation="silu"
    )
    assert torch.equal(output, x + 1)
    official.assert_called_once()

    shared = Mock(return_value=(x + 2, None))
    shared_output = SharedExpertsAdapter()(shared, x)
    assert torch.equal(shared_output, x + 2)
    shared.assert_called_once_with(x)


def _router_ready() -> bool:
    return torch.cuda.is_available() and importlib.util.find_spec("vllm") is not None


@pytest.mark.gpus(1)
@pytest.mark.skipif(
    not _router_ready(),
    reason="requires CUDA and official vLLM DS4 hash-router compiled op",
)
def test_official_hash_router_is_bitwise_through_adapter() -> None:
    from vllm import _custom_ops

    torch.manual_seed(31)
    logits = torch.randn(4, 256, dtype=torch.float32, device="cuda")
    tokens = torch.arange(4, dtype=torch.int32, device="cuda")
    table = (
        torch.arange(64 * 2, dtype=torch.int32, device="cuda").view(64, 2) % 256
    )
    reference_weights = torch.empty(4, 2, dtype=torch.float32, device="cuda")
    reference_ids = torch.empty(4, 2, dtype=torch.int32, device="cuda")
    token_expert = torch.empty_like(reference_ids)
    _custom_ops.topk_hash_softplus_sqrt(
        reference_weights,
        reference_ids,
        token_expert,
        logits,
        True,
        1.5,
        None,
        tokens,
        table,
        None,
    )
    candidate_weights, candidate_ids = HashRouteAdapter()(
        logits,
        tokens,
        table,
        topk=2,
        renormalize=True,
        routed_scaling_factor=1.5,
    )
    torch.testing.assert_close(candidate_weights, reference_weights, rtol=0, atol=0)
    assert torch.equal(candidate_ids, reference_ids)


@pytest.mark.gpus(1)
@pytest.mark.skipif(
    not _router_ready(),
    reason="requires CUDA and official vLLM DS4 hash-router compiled op",
)
def test_real_shape_hash_router_vjp_survives_consumer_id_mutation() -> None:
    torch.manual_seed(41)
    tokens = 640
    experts = 256
    topk = 6
    logits = torch.randn(tokens, experts, dtype=torch.float32, device="cuda", requires_grad=True)
    token_ids = torch.randint(0, 129280, (tokens,), dtype=torch.int32, device="cuda")
    table = torch.randint(0, experts, (129280, topk), dtype=torch.int32, device="cuda")
    adapter = HashRouteAdapter()

    def visible(value):
        return adapter(
            value,
            token_ids,
            table,
            topk=topk,
            renormalize=True,
            routed_scaling_factor=1.5,
        )

    weights, returned_ids = fixed_route_vjp(
        visible,
        logits,
        renormalize=True,
        route_scale=1.5,
    )
    returned_ids.fill_(-1)
    weights.square().sum().backward()
    assert torch.isfinite(logits.grad).all()


@pytest.mark.gpus(1)
@pytest.mark.skipif(
    not _router_ready(),
    reason="requires CUDA and official vLLM DS4 learned-router implementation",
)
def test_official_learned_router_is_bitwise_through_adapter() -> None:
    from vllm.model_executor.layers.fused_moe.router.dsv4_topk import dsv4_topk

    torch.manual_seed(37)
    logits = torch.randn(4, 256, dtype=torch.float32, device="cuda")
    correction_bias = torch.randn(256, dtype=torch.float32, device="cuda")
    reference_weights, reference_ids = dsv4_topk(
        logits,
        correction_bias,
        torch.int32,
        1.5,
    )
    candidate_weights, candidate_ids = DS4TopKAdapter()(
        logits,
        correction_bias,
        indices_dtype=torch.int32,
        routed_scaling_factor=1.5,
    )
    torch.testing.assert_close(candidate_weights, reference_weights, rtol=0, atol=0)
    assert torch.equal(candidate_ids, reference_ids)
