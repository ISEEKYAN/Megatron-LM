from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


class _FakeHybridEPBuffer:
    runtime = object()
    num_of_hybrid_ep_ranks_per_nvlink_domain = 2

    def dispatch_with_permute(
        self,
        *,
        hidden,
        topk_idx=None,
        topk_weights=None,
        routing_map=None,
        probs=None,
        handle=None,
        **_kwargs,
    ):
        if handle is not None:
            token_rows = handle["token_rows"]
            dispatched = hidden.index_select(0, token_rows)
            return dispatched, None, None, None, handle

        positions = torch.nonzero(routing_map, as_tuple=False)
        order = torch.argsort(positions[:, 1], stable=True)
        positions = positions.index_select(0, order)
        token_rows = positions[:, 0]
        route_experts = positions[:, 1]
        dispatched = hidden.index_select(0, token_rows)
        counts = torch.bincount(
            route_experts, minlength=4
        )[:2].to(torch.int64)
        handle = {
            "token_rows": token_rows,
            "num_tokens": hidden.shape[0],
            "route_experts": route_experts,
        }
        dispatched_probs = probs[token_rows, route_experts]
        return dispatched, dispatched_probs, None, counts, handle

    def combine_with_unpermute(
        self, *, hidden, handle, probs=None, **_kwargs
    ):
        combined = hidden.new_zeros((handle["num_tokens"], hidden.shape[1]))
        combined.index_add_(0, handle["token_rows"], hidden)
        if probs is None:
            return combined, None
        combined_probs = probs.new_zeros((handle["num_tokens"], 4))
        token_rows = handle["token_rows"]
        combined_probs[token_rows, handle["route_experts"]] = probs
        return combined, combined_probs


def _install_hybridep(monkeypatch, hybridep_module):
    fake_buffer = _FakeHybridEPBuffer()
    monkeypatch.setattr(
        hybridep_module,
        "deep_ep",
        SimpleNamespace(HybridEPBuffer=object),
    )
    monkeypatch.setattr(
        hybridep_module.dist, "get_world_size", lambda *, group: 2
    )
    capacities = []

    def get_buffer(*args):
        capacities.append(args[-1])
        return fake_buffer

    monkeypatch.setattr(hybridep_module, "get_buffer", get_buffer)
    monkeypatch.setenv(
        "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN", "2"
    )
    return fake_buffer, capacities


def test_dispatcher_type_validation_and_deepep_fail_closed(
    monkeypatch, transformer_engine_import_stub
):
    transformer_engine_import_stub()
    from megatron.lite.primitive.modules import dispatcher as module

    for value in ("alltoall", "deepep", "hybridep"):
        assert module.validate_moe_token_dispatcher_type(value) == value
    with pytest.raises(ValueError, match="alltoall.*deepep.*hybridep"):
        module.validate_moe_token_dispatcher_type("auto")
    for value in (None, 0, -1, True, 1.5):
        with pytest.raises(ValueError, match="positive integer"):
            module.validate_hybridep_max_tokens_per_rank("hybridep", value)
    assert module.validate_hybridep_max_tokens_per_rank("hybridep", 8) == 8
    assert module.validate_hybridep_max_tokens_per_rank("alltoall", None) is None
    with pytest.raises(ValueError, match="positive integer"):
        module.TokenDispatcher(
            4,
            2,
            SimpleNamespace(ep_size=1),
            moe_token_dispatcher_type="hybridep",
        )

    monkeypatch.setattr(module, "deep_ep", None)
    with pytest.raises(RuntimeError, match="requires the DeepEP runtime"):
        module.TokenDispatcher(
            4,
            2,
            SimpleNamespace(ep_size=2, tp_ep_group=object()),
            moe_token_dispatcher_type="deepep",
        )


def test_all_regular_moe_protocols_use_dispatcher_type_api() -> None:
    lite_root = Path(__file__).resolve().parents[4]
    for model_name in (
        "deepseek_v4",
        "glm5",
        "kimi_k2",
        "qwen3_5",
        "qwen3_moe",
    ):
        source = (
            lite_root
            / "megatron"
            / "lite"
            / "model"
            / model_name
            / "lite"
            / "protocol.py"
        ).read_text()
        assert "moe_token_dispatcher_type" in source
        assert "hybridep_max_tokens_per_rank" in source
        assert "validate_hybridep_max_tokens_per_rank" in source
        assert "validate_moe_token_dispatcher_type" in source
        assert "use_deepep" not in source


def test_hybridep_topology_fails_closed(monkeypatch):
    from megatron.lite.primitive.modules import hybridep

    monkeypatch.delenv(
        "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN", raising=False
    )
    with pytest.raises(RuntimeError, match="requires explicit"):
        hybridep.validate_topology(object())

    monkeypatch.setenv(
        "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN", "3"
    )
    monkeypatch.setattr(
        hybridep.dist, "get_world_size", lambda *, group: 4
    )
    with pytest.raises(RuntimeError, match="not divisible"):
        hybridep.validate_topology(object())


def test_hybridep_buffer_uses_explicit_static_capacity(monkeypatch):
    from megatron.lite.primitive.modules import hybridep

    created = []

    class Buffer:
        runtime = object()
        num_of_hybrid_ep_ranks_per_nvlink_domain = 2

        def __init__(self, **kwargs):
            created.append(kwargs)

    monkeypatch.setattr(
        hybridep,
        "deep_ep",
        SimpleNamespace(HybridEPBuffer=Buffer),
    )
    monkeypatch.setattr(
        hybridep.dist, "get_world_size", lambda *, group: 2
    )
    monkeypatch.setattr(hybridep, "detect_accessible_ranks", lambda group: 2)
    monkeypatch.setenv(
        "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN", "2"
    )
    monkeypatch.setattr(hybridep, "_buffer", None)
    monkeypatch.setattr(hybridep, "_buffer_signature", None)

    group = object()
    hybridep.get_buffer(group, 256, 2, 8, 32)
    hybridep.get_buffer(group, 256, 2, 16, 32)

    assert len(created) == 1
    assert created[0]["max_num_of_tokens_per_rank"] == 32
    with pytest.raises(RuntimeError, match="below required"):
        hybridep.get_buffer(group, 256, 2, 33, 32)
    hybridep.get_buffer(group, 256, 2, 33, 64)
    assert len(created) == 2
    assert created[1]["max_num_of_tokens_per_rank"] == 64


def test_hybridep_native_topk_dispatch_combine_backward(
    monkeypatch, transformer_engine_import_stub
):
    transformer_engine_import_stub()
    from megatron.lite.primitive.modules import dispatcher as module
    from megatron.lite.primitive.modules import hybridep

    _, capacities = _install_hybridep(monkeypatch, hybridep)
    group = object()
    dispatcher = module.TokenDispatcher(
        4,
        2,
        SimpleNamespace(ep_size=2, ep_group=group),
        moe_token_dispatcher_type="hybridep",
        hybridep_max_tokens_per_rank=8,
    )
    hidden = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0]], requires_grad=True
    )
    scores = torch.tensor(
        [[0.25, 0.75], [0.4, 0.6]], requires_grad=True
    )
    indices = torch.tensor([[0, 1], [1, 0]])

    dispatched, counts, permuted_scores = dispatcher.dispatch(
        hidden, scores, indices
    )
    output = dispatcher.combine(
        dispatched * permuted_scores.unsqueeze(-1)
    )

    torch.testing.assert_close(counts, torch.tensor([2, 2]))
    assert capacities == [8]
    torch.testing.assert_close(output, hidden)
    assert dispatcher._hybridep_handle is None

    output.sum().backward()
    torch.testing.assert_close(hidden.grad, torch.ones_like(hidden))
    torch.testing.assert_close(
        scores.grad, torch.tensor([[3.0, 3.0], [7.0, 7.0]])
    )


def test_hybridep_handle_must_be_combined(
    monkeypatch, transformer_engine_import_stub
):
    transformer_engine_import_stub()
    from megatron.lite.primitive.modules import dispatcher as module
    from megatron.lite.primitive.modules import hybridep

    _install_hybridep(monkeypatch, hybridep)
    dispatcher = module.TokenDispatcher(
        4,
        2,
        SimpleNamespace(ep_size=2, ep_group=object()),
        moe_token_dispatcher_type="hybridep",
        hybridep_max_tokens_per_rank=8,
    )
    hidden = torch.ones(2, 2)
    scores = torch.full((2, 2), 0.5)
    indices = torch.tensor([[0, 1], [1, 0]])
    dispatcher.dispatch(hidden, scores, indices)

    with pytest.raises(RuntimeError, match="previous handle"):
        dispatcher.dispatch(hidden, scores, indices)
