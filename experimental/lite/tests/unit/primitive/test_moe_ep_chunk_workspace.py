# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import inspect
from types import SimpleNamespace

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
        EPChunkShapeProfile,
        EPChunkWorkspaceKey,
        EPChunkWorkspaceRegistry,
        _SavedContextEPChunkFunction,
    )

    return (
        EP_CHUNK_COUNT,
        EPChunkForwardOp,
        EPChunkBackwardOp,
        EPChunkFusedForwardBackwardOp,
        EPChunkShapeProfile,
        EPChunkWorkspaceKey,
        EPChunkWorkspaceRegistry,
        _SavedContextEPChunkFunction,
    )


def test_three_explicit_ops_have_fixed_two_chunk_contract(
    transformer_engine_import_stub,
):
    (
        chunk_count,
        forward_op,
        backward_op,
        fused_op,
        _profile,
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
        "context",
        "grad_output",
    )
    assert tuple(inspect.signature(fused_op.forward_backward).parameters) == (
        "self",
        "x_saved",
        "grad_output",
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
        profile_type,
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

    profile = profile_type(
        max_input_rows=128,
        hidden_size=64,
        topk=8,
        ep_size=8,
    )
    common = dict(
        device_type="cpu",
        device_index=None,
        ep_group_id=7,
        dtype=torch.bfloat16,
        shape_profile=profile,
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
    assert created == []
    forward_layer_0.materialize(device="cpu")
    backward.materialize(device="cpu")
    fused.materialize(device="cpu")
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
        profile_type,
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
        shape_profile=profile_type(
            max_input_rows=8,
            hidden_size=4,
            topk=2,
            ep_size=4,
        ),
    )
    workspace = registry_type().get_or_create(key, lambda slot: f"dispatcher-{slot}")

    workspace.warmup(device="cpu")
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
        profile_type,
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
        shape_profile=profile_type(
            max_input_rows=8,
            hidden_size=4,
            topk=2,
            ep_size=4,
        ),
    )
    workspace = registry_type().get_or_create(key, lambda slot: slot)
    workspace.materialize(device="cpu")
    workspace.warmup_tensor("grad_hidden", (8, 4), dtype=torch.float32, device="cpu")

    with pytest.raises(RuntimeError, match="exceeds the fixed workspace shape"):
        workspace.warmup_tensor(
            "grad_hidden", (9, 4), dtype=torch.float32, device="cpu"
        )

    assert workspace.metrics()["grows"] == 0
    assert workspace.metrics()["fallbacks"] == 0


def test_production_warmup_makes_backward_steady_state_allocation_free(
    transformer_engine_import_stub,
):
    (
        _chunk_count,
        _forward_op,
        _backward_op,
        _fused_op,
        profile_type,
        key_type,
        registry_type,
        _function,
    ) = _symbols(transformer_engine_import_stub)
    profile = profile_type(
        max_input_rows=8,
        hidden_size=4,
        topk=2,
        ep_size=4,
    )
    workspace = registry_type().get_or_create(
        key_type(
            op="backward",
            device_type="cpu",
            device_index=None,
            ep_group_id=11,
            dtype=torch.float32,
            shape_profile=profile,
        ),
        lambda slot: slot,
    )

    workspace.warmup(device="cpu")
    pointers = {
        (slot, name): workspace.tensor(slot, name).data_ptr()
        for slot in range(2)
        for name in ("grad_recv_hidden", "grad_recv_probs")
    }
    allocations = workspace.metrics()["allocations"]

    lease = workspace.acquire(0)
    lease.tensor("grad_recv_hidden", (9, 4), dtype=torch.float32, device="cpu")
    lease.tensor("grad_recv_probs", (9, 2), dtype=torch.float32, device="cpu")

    assert workspace.metrics()["allocations"] == allocations
    assert workspace.metrics()["runtime_allocations"] == 0
    evidence = workspace.evidence()
    assert evidence["runtime_allocations"] == 0
    assert evidence["data_ptrs"] == {
        f"{slot}:{name}": pointers[(slot, name)]
        for slot in range(2)
        for name in ("grad_recv_hidden", "grad_recv_probs")
    }
    assert {
        (slot, name): workspace.tensor(slot, name).data_ptr()
        for slot in range(2)
        for name in ("grad_recv_hidden", "grad_recv_probs")
    } == pointers


def test_profile_rejects_input_rows_beyond_qwen_capacity(
    transformer_engine_import_stub,
):
    (
        _chunk_count,
        _forward_op,
        _backward_op,
        _fused_op,
        profile_type,
        _key,
        _registry,
        _function,
    ) = _symbols(transformer_engine_import_stub)
    profile = profile_type(
        max_input_rows=8,
        hidden_size=4,
        topk=2,
        ep_size=4,
    )

    with pytest.raises(RuntimeError, match="exceeds fixed profile"):
        profile.validate_input_rows(9)


def test_workspace_reports_deepep_buffer_count_and_resident_bytes(
    transformer_engine_import_stub,
):
    (
        _chunk_count,
        _forward_op,
        _backward_op,
        _fused_op,
        profile_type,
        key_type,
        registry_type,
        _function,
    ) = _symbols(transformer_engine_import_stub)
    profile = profile_type(
        max_input_rows=8,
        hidden_size=4,
        topk=2,
        ep_size=2,
    )

    def dispatcher(_slot):
        return SimpleNamespace(
            buffer=object(),
            deepep_buffer_resident_bytes=1024,
        )

    workspace = registry_type().get_or_create(
        key_type(
            op="forward",
            device_type="cpu",
            device_index=None,
            ep_group_id=19,
            dtype=torch.float32,
            shape_profile=profile,
        ),
        dispatcher,
    )
    workspace.warmup(device="cpu")

    evidence = workspace.evidence()
    assert evidence["dispatcher_count"] == 2
    assert evidence["deepep_buffer_count"] == 2
    assert evidence["deepep_buffer_resident_bytes"] == 2048
    assert evidence["caller_owned_recv_proven"] is False


def test_workspace_is_lazy_and_registry_release_rebuilds_without_old_state(
    transformer_engine_import_stub,
):
    (
        _chunk_count,
        _forward_op,
        _backward_op,
        _fused_op,
        profile_type,
        key_type,
        registry_type,
        _function,
    ) = _symbols(transformer_engine_import_stub)
    key = key_type(
        op="backward",
        device_type="cpu",
        device_index=None,
        ep_group_id=23,
        dtype=torch.float32,
        shape_profile=profile_type(
            max_input_rows=8,
            hidden_size=4,
            topk=2,
            ep_size=2,
        ),
    )
    created = []

    def factory(slot):
        dispatcher = SimpleNamespace(
            slot=slot,
            use_deepep=True,
            buffer=object(),
            deepep_buffer_resident_bytes=256,
        )
        created.append(dispatcher)
        return dispatcher

    registry = registry_type()
    workspace = registry.get_or_create(key, factory)
    assert workspace.evidence() == {
        "allocations": 0,
        "runtime_allocations": 0,
        "waits": 0,
        "grows": 0,
        "fallbacks": 0,
        "data_ptrs": {},
        "dispatcher_count": 0,
        "deepep_buffer_count": 0,
        "deepep_buffer_resident_bytes": 0,
        "caller_owned_recv_proven": False,
        "materialized": False,
    }
    assert created == []

    workspace.materialize(device="cpu")
    first_dispatchers = tuple(created)
    first_tensors = [
        workspace.tensor(slot, name)
        for slot in range(2)
        for name in ("grad_recv_hidden", "grad_recv_probs")
    ]
    first_ptrs = {tensor.data_ptr() for tensor in first_tensors}
    assert workspace.evidence()["dispatcher_count"] == 2

    registry.release(key)
    assert workspace.evidence()["dispatcher_count"] == 0
    assert workspace.evidence()["deepep_buffer_resident_bytes"] == 0
    assert workspace.evidence()["data_ptrs"] == {}
    registry.release(key)

    workspace.materialize(device="cpu")
    assert registry.get_or_create(key, factory) is workspace
    assert {
        workspace.tensor(slot, name).data_ptr()
        for slot in range(2)
        for name in ("grad_recv_hidden", "grad_recv_probs")
    }.isdisjoint(first_ptrs)
    registry.release(key)

    rebuilt = registry.get_or_create(key, factory)
    assert rebuilt is not workspace
    rebuilt.materialize(device="cpu")
    assert all(
        rebuilt.dispatcher(slot) is not first_dispatchers[slot] for slot in range(2)
    )
    assert {
        rebuilt.tensor(slot, name).data_ptr()
        for slot in range(2)
        for name in ("grad_recv_hidden", "grad_recv_probs")
    }.isdisjoint(first_ptrs)


def test_workspace_release_rejects_active_lease_and_pending_event(
    transformer_engine_import_stub,
):
    (
        _chunk_count,
        _forward_op,
        _backward_op,
        _fused_op,
        profile_type,
        key_type,
        registry_type,
        _function,
    ) = _symbols(transformer_engine_import_stub)
    key = key_type(
        op="forward",
        device_type="cpu",
        device_index=None,
        ep_group_id=29,
        dtype=torch.float32,
        shape_profile=profile_type(
            max_input_rows=8,
            hidden_size=4,
            topk=2,
            ep_size=2,
        ),
    )
    registry = registry_type()
    workspace = registry.get_or_create(key, lambda slot: SimpleNamespace(slot=slot))
    lease = workspace.acquire(0)
    with pytest.raises(RuntimeError, match="slot 0 is leased"):
        registry.release(key)

    event = _FakeEvent(ready=False)
    lease.release(event)
    with pytest.raises(RuntimeError, match="pending consumer event"):
        registry.release(key)

    stream = _FakeStream()
    registry.release(key, stream=stream)
    assert stream.waited == [event]
    assert workspace.evidence()["dispatcher_count"] == 0


def test_workspace_release_never_uses_device_wide_synchronize(
    transformer_engine_import_stub,
):
    (
        _chunk_count,
        _forward_op,
        _backward_op,
        _fused_op,
        _profile,
        _key,
        registry_type,
        _function,
    ) = _symbols(transformer_engine_import_stub)

    assert "synchronize" not in inspect.getsource(registry_type.release)


@pytest.mark.parametrize("shape", [(2, 5, 4), (10, 4)], ids=["bshd", "thd-packed"])
def test_profile_checks_actual_flattened_rows_for_batched_and_packed_inputs(
    shape, transformer_engine_import_stub
):
    (
        _chunk_count,
        _forward_op,
        _backward_op,
        _fused_op,
        profile_type,
        _key,
        _registry,
        _function,
    ) = _symbols(transformer_engine_import_stub)
    profile = profile_type.for_fixed_two_chunk_ep(
        max_input_rows=9,
        hidden_size=4,
        topk=2,
        ep_size=8,
    )

    with pytest.raises(RuntimeError, match="input rows 10"):
        profile.validate_input(torch.empty(shape))


def test_qwen3_profile_freezes_deepep_worst_case_receive_capacity(
    transformer_engine_import_stub,
):
    (
        _chunk_count,
        _forward_op,
        _backward_op,
        _fused_op,
        profile_type,
        _key,
        _registry,
        _function,
    ) = _symbols(transformer_engine_import_stub)

    profile = profile_type.for_fixed_two_chunk_ep(
        max_input_rows=17,
        hidden_size=64,
        topk=8,
        ep_size=8,
    )

    assert profile.max_recv_rows == 9 * 8
    assert profile.topk == 8


def test_saved_context_autograd_keeps_forward_context_for_backward(
    transformer_engine_import_stub,
):
    (
        _chunk_count,
        _forward_op,
        _backward_op,
        _fused_op,
        _profile,
        _key,
        _registry,
        function,
    ) = _symbols(transformer_engine_import_stub)
    source = inspect.getsource(function)

    assert "ctx.saved_forward_context = saved_context" in source
    assert "ctx.backward_op = forward_op.backward_op" in source
    assert "ctx.backward_op.backward(" in source
    assert "synchronize" not in source


def test_saved_context_backward_uses_the_dispatcher_that_created_its_handle(
    transformer_engine_import_stub,
):
    transformer_engine_import_stub()
    from megatron.lite.primitive.modules.moe_ep_chunk_overlap import (
        _EPChunkOperationBase,
        _ForwardChunkContext,
    )

    fields = _ForwardChunkContext.__dataclass_fields__
    forward_source = inspect.getsource(
        _EPChunkOperationBase._forward_saved_context_async
    )
    backward_source = inspect.getsource(_EPChunkOperationBase._saved_context_backward)

    assert "dispatcher" in fields
    assert "dispatcher=dispatcher" in forward_source
    assert "dispatcher=saved.dispatcher" in backward_source
    assert "dispatcher=lease.dispatcher" not in backward_source


def test_saved_forward_context_does_not_pin_two_shared_slots_across_layers(
    transformer_engine_import_stub,
):
    transformer_engine_import_stub()
    from megatron.lite.primitive.modules.moe_ep_chunk_overlap import (
        _EPChunkOperationBase,
        _ForwardChunkContext,
    )

    fields = _ForwardChunkContext.__dataclass_fields__
    forward_source = inspect.getsource(
        _EPChunkOperationBase._forward_saved_context_async
    )

    assert "workspace_lease" not in fields
    assert "lease.release(consumed)" in forward_source
    assert 'handle=state["handle"]' in forward_source
