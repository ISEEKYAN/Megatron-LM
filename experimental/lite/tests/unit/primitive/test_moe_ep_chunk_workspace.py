# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import inspect

import pytest
import torch


class _FakeEvent:
    def __init__(self, ready: bool = False):
        self.ready = ready

    def query(self) -> bool:
        return self.ready


class _FakeStream:
    def __init__(self):
        self.waited = []

    def wait_event(self, event) -> None:
        self.waited.append(event)


def _symbols(transformer_engine_import_stub):
    transformer_engine_import_stub()
    from megatron.lite.primitive.modules.moe_ep_chunk_overlap import (
        EP_CHUNK_COUNT,
        EPChunkBackwardOp,
        EPChunkForwardOp,
        EPChunkFusedForwardBackwardOp,
        EPChunkWorkspaceKey,
        EPChunkWorkspaceRegistry,
        _FusedEPChunkFunction,
    )

    return (
        EP_CHUNK_COUNT,
        EPChunkForwardOp,
        EPChunkBackwardOp,
        EPChunkFusedForwardBackwardOp,
        EPChunkWorkspaceKey,
        EPChunkWorkspaceRegistry,
        _FusedEPChunkFunction,
    )


def test_three_explicit_ops_have_fixed_two_chunk_contract(
    transformer_engine_import_stub,
):
    (
        chunk_count,
        forward_op,
        backward_op,
        fused_op,
        _key,
        _registry,
        _function,
    ) = _symbols(transformer_engine_import_stub)

    assert chunk_count == 2
    assert tuple(inspect.signature(forward_op.forward).parameters) == (
        "self",
        "x",
        "routing_input",
    )
    assert tuple(inspect.signature(backward_op.backward).parameters) == (
        "self",
        "x_saved",
        "grad_output",
        "routing_input",
    )
    assert tuple(inspect.signature(fused_op.forward).parameters) == (
        "self",
        "x",
        "routing_input",
    )


def test_three_ops_get_independent_cross_layer_workspaces(
    transformer_engine_import_stub,
):
    (
        _chunk_count,
        _forward_op,
        _backward_op,
        _fused_op,
        key_type,
        registry_type,
        _function,
    ) = _symbols(transformer_engine_import_stub)
    registry = registry_type()
    created = []

    def factory(slot: int):
        dispatcher = object()
        created.append((slot, dispatcher))
        return dispatcher

    common = dict(
        device_type="cpu",
        device_index=None,
        ep_group_id=7,
        dtype=torch.bfloat16,
        shape_profile=(128, 64, 8),
    )
    forward_key = key_type(op="forward", **common)
    backward_key = key_type(op="backward", **common)
    fused_key = key_type(op="fused_forward_backward", **common)

    forward_layer_0 = registry.get_or_create(forward_key, factory)
    forward_layer_47 = registry.get_or_create(forward_key, factory)
    backward = registry.get_or_create(backward_key, factory)
    fused = registry.get_or_create(fused_key, factory)

    assert forward_layer_0 is forward_layer_47
    assert forward_layer_0 is not backward
    assert backward is not fused
    assert len(created) == 6
    assert [slot for slot, _ in created] == [0, 1, 0, 1, 0, 1]
    assert "layer" not in key_type.__dataclass_fields__
    assert "chunk" not in key_type.__dataclass_fields__


def test_workspace_slots_are_stable_and_consumer_event_guarded(
    transformer_engine_import_stub,
):
    (
        _chunk_count,
        _forward_op,
        _backward_op,
        _fused_op,
        key_type,
        registry_type,
        _function,
    ) = _symbols(transformer_engine_import_stub)
    key = key_type(
        op="forward",
        device_type="cpu",
        device_index=None,
        ep_group_id=3,
        dtype=torch.float32,
        shape_profile=(8, 4, 2),
    )
    workspace = registry_type().get_or_create(key, lambda slot: f"dispatcher-{slot}")

    workspace.warmup_tensor("output", (8, 4), dtype=torch.float32, device="cpu")
    before = [workspace.tensor(slot, "output").data_ptr() for slot in range(2)]
    allocation_count = workspace.metrics()["allocations"]

    first = workspace.acquire(0)
    with pytest.raises(RuntimeError, match="already leased"):
        workspace.acquire(0)
    pending = _FakeEvent(ready=False)
    first.release(pending)

    stream = _FakeStream()
    second = workspace.acquire(0, stream=stream)
    assert stream.waited == [pending]
    second.release(_FakeEvent(ready=True))

    workspace.warmup_tensor("output", (8, 4), dtype=torch.float32, device="cpu")
    after = [workspace.tensor(slot, "output").data_ptr() for slot in range(2)]
    metrics = workspace.metrics()
    assert after == before
    assert metrics["allocations"] == allocation_count
    assert metrics["grows"] == 0
    assert metrics["fallbacks"] == 0


def test_workspace_shape_overflow_fails_loud_without_growth(
    transformer_engine_import_stub,
):
    (
        _chunk_count,
        _forward_op,
        _backward_op,
        _fused_op,
        key_type,
        registry_type,
        _function,
    ) = _symbols(transformer_engine_import_stub)
    key = key_type(
        op="backward",
        device_type="cpu",
        device_index=None,
        ep_group_id=5,
        dtype=torch.float32,
        shape_profile=(8, 4, 2),
    )
    workspace = registry_type().get_or_create(key, lambda slot: slot)
    workspace.warmup_tensor("grad_hidden", (8, 4), dtype=torch.float32, device="cpu")

    with pytest.raises(RuntimeError, match="exceeds the fixed workspace shape"):
        workspace.warmup_tensor(
            "grad_hidden", (9, 4), dtype=torch.float32, device="cpu"
        )

    assert workspace.metrics()["grows"] == 0
    assert workspace.metrics()["fallbacks"] == 0


def test_fused_autograd_keeps_input_alive_for_backward(
    transformer_engine_import_stub,
):
    (
        _chunk_count,
        _forward_op,
        _backward_op,
        _fused_op,
        _key,
        _registry,
        function,
    ) = _symbols(transformer_engine_import_stub)
    source = inspect.getsource(function)

    assert "ctx.save_for_backward(x_2d.detach(), saved_routing)" in source
    assert "ctx.backward_op = fused_op.backward_op" in source
    assert "ctx.saved_tensors" in source
    assert "synchronize" not in source
