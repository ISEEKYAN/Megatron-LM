# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import inspect
from contextlib import contextmanager
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


def test_allocation_arena_reenters_one_pool_once_and_restores_depth_after_error(
    monkeypatch, transformer_engine_import_stub
):
    """Nested expert allocation scopes must not nest CUDA MemPool contexts."""
    transformer_engine_import_stub()
    import megatron.lite.primitive.modules.moe_ep_chunk_overlap as overlap

    entered = []

    @contextmanager
    def use_pool(pool, device=None):
        entered.append((pool, device))
        yield

    monkeypatch.setattr(overlap.torch.cuda, "use_mem_pool", use_pool)
    arena = overlap._EPChunkAllocationArena(
        allocation_pool=object(), device=torch.device("cuda", 0)
    )

    with arena.allocate():
        with arena.allocate():
            assert arena._allocation_depth == 2
    assert entered == [(arena.allocation_pool, torch.device("cuda", 0))]
    assert arena._allocation_depth == 0

    with pytest.raises(RuntimeError, match="boom"):
        with arena.allocate():
            with arena.allocate():
                raise RuntimeError("boom")
    assert arena._allocation_depth == 0
    assert entered == [(arena.allocation_pool, torch.device("cuda", 0))] * 2


def test_owned_expert_tensor_enters_the_dedicated_activation_pool(
    monkeypatch, transformer_engine_import_stub
):
    """Caller-owned activation allocations use the shared activation MemPool."""
    transformer_engine_import_stub()
    import megatron.lite.primitive.modules.moe_ep_chunk_overlap as overlap

    entered = []

    @contextmanager
    def use_device(_device):
        yield

    @contextmanager
    def use_pool(pool, device=None):
        entered.append((pool, device))
        yield

    monkeypatch.setattr(overlap.torch.cuda, "device", use_device)
    monkeypatch.setattr(overlap.torch.cuda, "MemPool", lambda **_kwargs: object())
    monkeypatch.setattr(overlap.torch.cuda, "use_mem_pool", use_pool)
    profile = overlap.EPChunkShapeProfile(
        max_input_rows=8, hidden_size=4, topk=2, ep_size=2
    )
    workspace = overlap.EPChunkWorkspaceRegistry().get_or_create(
        overlap.EPChunkWorkspaceKey(
            op="forward",
            device_type="cuda",
            device_index=0,
            ep_group_id=1,
            dtype=torch.bfloat16,
            shape_profile=profile,
        ),
        lambda slot: SimpleNamespace(slot=slot, use_deepep=True),
    )
    workspace.materialize()
    lease = workspace.acquire_expert_activation(
        stream=SimpleNamespace(device=torch.device("cuda", 0))
    )
    with lease.allocate():
        tensor = lease.tensor(
            "fc1_input", (3, 4), dtype=torch.bfloat16, device=torch.device("cpu")
        )

    assert tuple(tensor.shape) == (3, 4)
    assert len(entered) == 1
    assert entered[0][1] == torch.device("cuda", 0)
    assert workspace.evidence()["expert_activation_pool_count"] == 1
    lease.release(_FakeEvent(ready=True))


def test_explicit_frozen_activation_reservation_parks_without_tensor_ownership(
    monkeypatch, transformer_engine_import_stub
):
    """A frozen profile owns capacity in one pool, never persistent tensors."""
    transformer_engine_import_stub()
    import megatron.lite.primitive.modules.moe_ep_chunk_overlap as overlap

    pools = []

    @contextmanager
    def use_device(_device):
        yield

    @contextmanager
    def use_pool(_pool, device=None):
        assert device == torch.device("cuda", 0)
        yield

    monkeypatch.setattr(overlap.torch.cuda, "device", use_device)
    monkeypatch.setattr(
        overlap.torch.cuda, "MemPool", lambda **_kwargs: pools.append(object()) or pools[-1]
    )
    monkeypatch.setattr(overlap.torch.cuda, "use_mem_pool", use_pool)
    monkeypatch.setattr(overlap, "_EXPERT_ACTIVATION_SIZE_CLASS_BYTES", 1)
    real_empty = torch.empty
    pool_blocks = {}

    def pool_empty(shape, *args, **kwargs):
        """Model free-block reuse; production uses CUDA MemPool for this."""
        if isinstance(shape, tuple) and len(shape) == 1 and shape[0] > 1:
            key = (shape, tuple(sorted(kwargs.items())))
            return pool_blocks.setdefault(key, real_empty(shape, *args, **kwargs))
        return real_empty(shape, *args, **kwargs)

    monkeypatch.setattr(overlap.torch, "empty", pool_empty)
    profile = overlap.EPChunkShapeProfile(
        max_input_rows=8,
        hidden_size=4,
        topk=2,
        ep_size=2,
        expert_intermediate_size=3,
    )
    registry = overlap.EPChunkWorkspaceRegistry()

    def workspace(op):
        return registry.get_or_create(
            overlap.EPChunkWorkspaceKey(
                op=op,
                device_type="cuda",
                device_index=0,
                ep_group_id=991,
                dtype=torch.float32,
                shape_profile=profile,
            ),
            lambda slot: SimpleNamespace(slot=slot, use_deepep=True),
        )

    forward, fused = workspace("forward"), workspace("fused_forward_backward")
    stream = SimpleNamespace(device=torch.device("cuda", 0))
    forward.reserve_expert_activations(max_expert_rows=3, device=stream.device)
    coordinator = forward._expert_activation_owner.coordinator
    assert coordinator.frozen
    assert coordinator.arena.tensors == {}
    assert coordinator.arena.allocation_pool is pools[0]

    first = forward.acquire_expert_activation(stream=stream)
    fc1_input = first.tensor("fc1_input", (3, 4), dtype=torch.float32, device="cpu")
    fc1_output = first.tensor("fc1_output", (3, 6), dtype=torch.float32, device="cpu")
    fc2_output = first.tensor("fc2_output", (3, 4), dtype=torch.float32, device="cpu")
    assert set(coordinator.arena.tensors) == {"fc1_input", "fc1_output"}
    assert fc2_output.data_ptr() == fc1_input.data_ptr()
    first_signature = tuple(
        sorted((name, tensor.data_ptr()) for name, tensor in coordinator.arena.tensors.items())
    )
    first.release(_FakeEvent(ready=True))
    forward.reset_tensors()
    assert coordinator.arena.tensors == {}
    assert not hasattr(coordinator, "reserved_tensors")
    del fc1_input, fc1_output, fc2_output

    rehydrated = forward.acquire_expert_activation(stream=stream)
    rehydrated.tensor("fc1_input", (3, 4), dtype=torch.float32, device="cpu")
    rehydrated.tensor("fc1_output", (3, 6), dtype=torch.float32, device="cpu")
    rehydrated.tensor("fc2_output", (3, 4), dtype=torch.float32, device="cpu")
    assert tuple(
        sorted((name, tensor.data_ptr()) for name, tensor in coordinator.arena.tensors.items())
    ) == first_signature
    rehydrated.release(_FakeEvent(ready=True))
    forward.reset_tensors()

    second = fused.acquire_expert_activation(stream=stream)
    second.tensor("fc1_input", (3, 4), dtype=torch.float32, device="cpu")
    second.tensor("fc1_output", (3, 6), dtype=torch.float32, device="cpu")
    second.tensor("fc2_output", (3, 4), dtype=torch.float32, device="cpu")
    second.release(_FakeEvent(ready=True))
    assert fused.evidence()["expert_activation_pool_id"] == forward.evidence()["expert_activation_pool_id"]
    fused.reset_tensors()

    third = forward.acquire_expert_activation(stream=stream)
    third.tensor("fc1_input", (3, 4), dtype=torch.float32, device="cpu")
    third.tensor("fc1_output", (3, 6), dtype=torch.float32, device="cpu")
    third.tensor("fc2_output", (3, 4), dtype=torch.float32, device="cpu")
    with pytest.raises(RuntimeError, match="Frozen EP chunk expert activation"):
        third.tensor("fc1_input", (4, 4), dtype=torch.float32, device="cpu")
    third.release(_FakeEvent(ready=True))
    forward.reset_tensors()

    registry.release(forward.key)
    registry.release(fused.key)
    assert coordinator.arena.allocation_pool is None
    assert coordinator.capacity_bytes == {}
    assert coordinator.pointer_signatures_by_op == {}
    assert not coordinator.frozen


def test_activation_pool_parks_between_ops_at_observed_high_watermark(
    monkeypatch, transformer_engine_import_stub
):
    """Only explicit park hands one pool to the next OP without tensor residency."""
    transformer_engine_import_stub()
    import megatron.lite.primitive.modules.moe_ep_chunk_overlap as overlap

    pools = []

    @contextmanager
    def use_device(_device):
        yield

    @contextmanager
    def use_pool(_pool, device=None):
        assert device == torch.device("cuda", 0)
        yield

    monkeypatch.setattr(overlap.torch.cuda, "device", use_device)
    monkeypatch.setattr(
        overlap.torch.cuda,
        "MemPool",
        lambda **_kwargs: pools.append(object()) or pools[-1],
    )
    monkeypatch.setattr(overlap.torch.cuda, "use_mem_pool", use_pool)
    profile = overlap.EPChunkShapeProfile(
        max_input_rows=32, hidden_size=4, topk=2, ep_size=2
    )
    registry = overlap.EPChunkWorkspaceRegistry()

    def workspace(op):
        return registry.get_or_create(
            overlap.EPChunkWorkspaceKey(
                op=op,
                device_type="cuda",
                device_index=0,
                ep_group_id=11,
                dtype=torch.float32,
                shape_profile=profile,
            ),
            lambda slot: SimpleNamespace(slot=slot, use_deepep=True),
        )

    backward, fused = workspace("backward"), workspace("fused_forward_backward")
    stream = _FakeStream()
    stream.device = torch.device("cuda", 0)
    names = ("fc1_input", "fc1_output", "fc2_output")
    first = backward.acquire_expert_activation(stream=stream)
    for name in names:
        first.tensor(name, (16, 4), dtype=torch.float32, device="cpu")
    first.release(_FakeEvent(ready=True))
    # Per-chunk and per-layer leases retain their three storage references.
    repeated = backward.acquire_expert_activation(stream=stream)
    for name in names:
        repeated.tensor(name, (3, 4), dtype=torch.float32, device="cpu")
    pending = _FakeEvent(ready=False)
    repeated.release(pending)
    coordinator = backward._expert_activation_owner.coordinator
    capacity = coordinator.capacity_bytes["fc1_input"]
    pool = coordinator.arena.allocation_pool
    assert coordinator.arena.tensors
    assert coordinator.rehydrates == 0
    scratch_lease = backward.acquire(0, require_dispatcher=False)
    scratch_lease.tensor(
        "grad_expert_out", (3, 4), dtype=torch.float32, device="cpu"
    )
    scratch_lease.release(_FakeEvent(ready=True))
    backward.reset_tensors(stream=stream)
    assert stream.waited == [pending]
    assert coordinator.arena.tensors == {}
    assert coordinator.parks == 1
    assert backward.evidence()["data_ptrs"] == {}

    second = fused.acquire_expert_activation(stream=stream)
    smaller = {
        name: second.tensor(name, (3, 4), dtype=torch.float32, device="cpu")
        for name in names
    }
    assert stream.waited == [pending]
    assert all(tensor.numel() == 12 for tensor in smaller.values())
    assert coordinator.capacity_bytes["fc1_input"] == capacity
    assert coordinator.arena.allocation_pool is pool
    assert coordinator.rehydrates == 3
    second.release(_FakeEvent(ready=True))
    assert coordinator.arena.tensors
    assert pools == [pool]

    registry.release(backward.key)
    registry.release(fused.key)
    assert coordinator.arena.allocation_pool is None
    assert coordinator.capacity_bytes == {}
    assert coordinator.parks == coordinator.rehydrates == 0


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


@pytest.mark.parametrize("chunk_count", [2, 3, 4])
def test_logical_chunk_count_is_profile_contract_but_workspace_stays_two_slots(
    chunk_count, transformer_engine_import_stub
):
    transformer_engine_import_stub()
    from megatron.lite.primitive.modules.moe_ep_chunk_overlap import (
        EP_CHUNK_COUNT,
        EPChunkShapeProfile,
        EPChunkWorkspaceKey,
        EPChunkWorkspaceRegistry,
    )

    profile = EPChunkShapeProfile(
        max_input_rows=17, hidden_size=4, topk=2, ep_size=4, chunk_count=chunk_count
    )
    workspace = EPChunkWorkspaceRegistry().get_or_create(
        EPChunkWorkspaceKey(
            op="forward",
            device_type="cpu",
            device_index=None,
            ep_group_id=31,
            dtype=torch.float32,
            shape_profile=profile,
        ),
        lambda slot: f"dispatcher-{slot}",
    )
    workspace.materialize(device="cpu")

    assert profile.chunk_count == chunk_count
    assert EP_CHUNK_COUNT == 2
    assert workspace.evidence()["dispatcher_count"] == 2
    assert workspace.metrics()["fallbacks"] == 0


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

    profile = profile_type(max_input_rows=128, hidden_size=64, topk=8, ep_size=8)
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
        shape_profile=profile_type(max_input_rows=8, hidden_size=4, topk=2, ep_size=4),
    )
    workspace = registry_type().get_or_create(key, lambda slot: f"dispatcher-{slot}")

    first = workspace.acquire(0)
    before = first.tensor(
        "output", (8, 4), dtype=torch.float32, device="cpu"
    ).data_ptr()
    allocation_count = workspace.metrics()["allocations"]
    with pytest.raises(RuntimeError, match="already leased"):
        workspace.acquire(0)
    pending = _FakeEvent(ready=False)
    first.release(pending)

    stream = _FakeStream()
    second = workspace.acquire(0, stream=stream)
    assert stream.waited == [pending]
    after = second.tensor(
        "output", (8, 4), dtype=torch.float32, device="cpu"
    ).data_ptr()
    second.release(_FakeEvent(ready=True))

    metrics = workspace.metrics()
    assert after == before
    assert metrics["allocations"] == allocation_count
    assert metrics["grows"] == 0
    assert metrics["fallbacks"] == 0


def test_expert_scratch_can_exceed_recv_rows_but_not_expert_row_capacity(
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
        shape_profile=profile_type(max_input_rows=8, hidden_size=4, topk=2, ep_size=4),
    )
    workspace = registry_type().get_or_create(key, lambda slot: slot)
    workspace.materialize(device="cpu")
    assert workspace.evidence()["data_ptrs"] == {}
    lease = workspace.acquire(0)
    assert key.shape_profile.max_recv_rows == 16
    first = lease.tensor("grad_expert_out", (16, 4), dtype=torch.float32, device="cpu")
    grown = lease.tensor("grad_expert_out", (17, 4), dtype=torch.float32, device="cpu")

    assert grown.shape == (17, 4)
    assert grown.data_ptr() != first.data_ptr()
    assert workspace.metrics()["grows"] == 1

    with pytest.raises(RuntimeError, match="exceeds fixed profile capacity"):
        lease.tensor("grad_expert_out", (33, 4), dtype=torch.float32, device="cpu")

    assert workspace.metrics()["fallbacks"] == 0


@pytest.mark.parametrize("op", ["backward", "fused_forward_backward"])
def test_op_scratch_is_actual_shape_lazy_then_steady_state_stable(
    op, transformer_engine_import_stub
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
    profile = profile_type(max_input_rows=8, hidden_size=4, topk=2, ep_size=4)
    workspace = registry_type().get_or_create(
        key_type(
            op=op,
            device_type="cpu",
            device_index=None,
            ep_group_id=11,
            dtype=torch.float32,
            shape_profile=profile,
        ),
        lambda slot: slot,
    )

    workspace.materialize(device="cpu")
    assert workspace.metrics()["allocations"] == 0
    assert workspace.evidence()["data_ptrs"] == {}

    first_ptrs = {}
    for slot in range(2):
        lease = workspace.acquire(slot)
        first_ptrs[(slot, "grad_expert_out")] = lease.tensor(
            "grad_expert_out", (9, 4), dtype=torch.float32, device="cpu"
        ).data_ptr()
        lease.release(_FakeEvent(ready=True))

    first_metrics = workspace.metrics().copy()
    assert first_metrics["allocations"] == 2
    assert first_metrics["runtime_allocations"] == 2

    second_ptrs = {}
    for slot in range(2):
        lease = workspace.acquire(slot)
        second_ptrs[(slot, "grad_expert_out")] = lease.tensor(
            "grad_expert_out", (9, 4), dtype=torch.float32, device="cpu"
        ).data_ptr()
        lease.release(_FakeEvent(ready=True))

    assert second_ptrs == first_ptrs
    assert workspace.metrics() == first_metrics


def test_backward_scratch_only_lease_does_not_materialize_dispatchers_or_pools(
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
    created = []
    workspace = registry_type().get_or_create(
        key_type(
            op="backward",
            device_type="cpu",
            device_index=None,
            ep_group_id=17,
            dtype=torch.float32,
            shape_profile=profile_type(
                max_input_rows=8, hidden_size=4, topk=2, ep_size=4
            ),
        ),
        lambda slot: created.append(slot) or f"dispatcher-{slot}",
    )

    lease = workspace.acquire(0, require_dispatcher=False)
    scratch = lease.tensor("grad_expert_out", (9, 4), dtype=torch.float32, device="cpu")

    assert scratch.shape == (9, 4)
    assert created == []
    assert workspace.evidence()["dispatcher_count"] == 0
    assert workspace.evidence()["allocation_pool_count"] == 0
    with pytest.raises(RuntimeError, match="scratch-only lease has no dispatcher"):
        _ = lease.dispatcher
    lease.release(_FakeEvent(ready=True))


def test_dispatcher_backed_lease_still_materializes_two_dispatchers(
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
    created = []
    workspace = registry_type().get_or_create(
        key_type(
            op="fused_forward_backward",
            device_type="cpu",
            device_index=None,
            ep_group_id=19,
            dtype=torch.float32,
            shape_profile=profile_type(
                max_input_rows=8, hidden_size=4, topk=2, ep_size=4
            ),
        ),
        lambda slot: created.append(slot) or f"dispatcher-{slot}",
    )

    lease = workspace.acquire(0)

    assert lease.dispatcher == "dispatcher-0"
    assert created == [0, 1]
    assert workspace.evidence()["dispatcher_count"] == 2
    lease.release(_FakeEvent(ready=True))


def test_reset_tensors_waits_for_consumers_and_keeps_dispatchers(
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
    workspace = registry_type().get_or_create(
        key_type(
            op="backward",
            device_type="cpu",
            device_index=None,
            ep_group_id=23,
            dtype=torch.float32,
            shape_profile=profile_type(
                max_input_rows=8, hidden_size=4, topk=2, ep_size=4
            ),
        ),
        lambda slot: f"dispatcher-{slot}",
    )
    lease = workspace.acquire(0)
    tensor = lease.tensor("grad_expert_out", (9, 4), dtype=torch.float32, device="cpu")
    lease.release(_FakeEvent(ready=False))

    with pytest.raises(RuntimeError, match="pending consumer event"):
        workspace.reset_tensors()

    stream = _FakeStream()
    workspace.reset_tensors(stream=stream)

    evidence = workspace.evidence()
    assert len(stream.waited) == 1
    assert evidence["dispatcher_count"] == 2
    assert evidence["data_ptrs"] == {}
    assert evidence["tensor_details"] == {}
    assert tensor.shape == (9, 4)


@pytest.mark.parametrize("failure", ["leased_slot", "pending_slot"])
def test_reset_preflight_failure_preserves_activation_arena_atomically(
    failure, transformer_engine_import_stub
):
    """A rejected reset must not park activation state before slot preflight passes."""
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
    workspace = registry_type().get_or_create(
        key_type(
            op="backward",
            device_type="cpu",
            device_index=None,
            ep_group_id=230,
            dtype=torch.float32,
            shape_profile=profile_type(
                max_input_rows=8, hidden_size=4, topk=2, ep_size=2
            ),
        ),
        lambda slot: f"dispatcher-{slot}",
    )
    activation = workspace.acquire_expert_activation()
    activation.tensor("fc1_input", (4, 4), dtype=torch.float32, device="cpu")
    activation_event = _FakeEvent(ready=True)
    activation.release(activation_event)
    coordinator = workspace._expert_activation_owner.coordinator
    tensors_before = dict(coordinator.arena.tensors)
    counters_before = (coordinator.parks, coordinator.rehydrates)

    slot = workspace.acquire(0, require_dispatcher=False)
    if failure == "pending_slot":
        slot.release(_FakeEvent(ready=False))
        expected = "pending consumer event"
    else:
        expected = "slot 0 is leased"

    with pytest.raises(RuntimeError, match=expected):
        workspace.reset_tensors()

    assert coordinator.arena.tensors == tensors_before
    assert coordinator.consumer_event is activation_event
    assert (coordinator.parks, coordinator.rehydrates) == counters_before
    if failure == "leased_slot":
        slot.release(_FakeEvent(ready=True))


def test_workspace_evidence_reports_actual_scratch_shape_dtype_and_bytes(
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
    workspace = registry_type().get_or_create(
        key_type(
            op="backward",
            device_type="cpu",
            device_index=None,
            ep_group_id=29,
            dtype=torch.float32,
            shape_profile=profile_type(
                max_input_rows=8, hidden_size=4, topk=2, ep_size=4
            ),
        ),
        lambda slot: f"dispatcher-{slot}",
    )
    lease = workspace.acquire(0, require_dispatcher=False)
    lease.tensor("grad_expert_out", (9, 4), dtype=torch.float32, device="cpu")
    lease.release(_FakeEvent(ready=True))

    details = workspace.evidence()["tensor_details"]
    assert details == {
        "0:grad_expert_out": {
            "shape": (9, 4),
            "dtype": "torch.float32",
            "nbytes": 9 * 4 * 4,
        }
    }


def test_scratch_only_lease_allocates_from_its_own_backward_arena(
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
    workspace = registry_type().get_or_create(
        key_type(
            op="backward",
            device_type="cpu",
            device_index=None,
            ep_group_id=31,
            dtype=torch.float32,
            shape_profile=profile_type(
                max_input_rows=8, hidden_size=4, topk=2, ep_size=4
            ),
        ),
        lambda slot: f"unused-{slot}",
    )
    entered = []

    class FakeArena:
        @contextmanager
        def allocate(self):
            entered.append("enter")
            yield
            entered.append("exit")

    workspace._allocation_arenas[0] = FakeArena()
    lease = workspace.acquire(0, require_dispatcher=False)
    lease.tensor("grad_expert_out", (9, 4), dtype=torch.float32, device="cpu")

    assert entered == ["enter", "exit"]
    assert workspace.evidence()["dispatcher_count"] == 0
    assert lease.allocation_arena is workspace.allocation_arena(0)
    lease.release(_FakeEvent(ready=True))


def test_fused_dispatcher_lease_allocates_scratch_from_its_own_slot_arena(
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
    workspace = registry_type().get_or_create(
        key_type(
            op="fused_forward_backward",
            device_type="cpu",
            device_index=None,
            ep_group_id=41,
            dtype=torch.float32,
            shape_profile=profile_type(
                max_input_rows=8, hidden_size=4, topk=2, ep_size=4
            ),
        ),
        lambda slot: f"fused-{slot}",
    )
    entered = []

    class FakeArena:
        allocation_pool = None
        device = None

        @contextmanager
        def allocate(self):
            entered.append("enter")
            yield
            entered.append("exit")

    workspace._allocation_arenas[0] = FakeArena()
    lease = workspace.acquire(0)
    lease.tensor("grad_expert_out", (9, 4), dtype=torch.float32, device="cpu")

    assert entered == ["enter", "exit"]
    assert lease.allocation_arena is workspace.allocation_arena(0)
    lease.release(_FakeEvent(ready=True))


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
    profile = profile_type(max_input_rows=8, hidden_size=4, topk=2, ep_size=4)

    with pytest.raises(RuntimeError, match="exceeds two-slot profile"):
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
    profile = profile_type(max_input_rows=8, hidden_size=4, topk=2, ep_size=2)

    def dispatcher(_slot):
        return SimpleNamespace(buffer=object(), deepep_buffer_resident_bytes=1024)

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
    workspace.materialize(device="cpu")

    evidence = workspace.evidence()
    assert evidence["dispatcher_count"] == 2
    assert evidence["deepep_buffer_count"] == 2
    assert evidence["deepep_buffer_resident_bytes"] == 2048


def test_workspace_owns_two_comm_pools_and_one_activation_pool(
    monkeypatch, transformer_engine_import_stub
):
    transformer_engine_import_stub()
    import megatron.lite.primitive.modules.moe_ep_chunk_overlap as overlap

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
    created = []
    entered = []
    creation_devices = []

    class FakePool:
        pass

    @contextmanager
    def use_pool(pool, device=None):
        entered.append((pool, device))
        yield

    @contextmanager
    def use_device(device):
        creation_devices.append(device)
        yield

    def make_pool(*, allocator=None, use_on_oom=False, no_split=False):
        assert allocator is None
        assert use_on_oom is False
        assert no_split is False
        pool = FakePool()
        created.append(pool)
        return pool

    monkeypatch.setattr(overlap.torch.cuda, "MemPool", make_pool)
    monkeypatch.setattr(overlap.torch.cuda, "use_mem_pool", use_pool)
    monkeypatch.setattr(overlap.torch.cuda, "device", use_device)
    workspace = registry_type().get_or_create(
        key_type(
            op="forward",
            device_type="cuda",
            device_index=3,
            ep_group_id=31,
            dtype=torch.bfloat16,
            shape_profile=profile_type(
                max_input_rows=8, hidden_size=4, topk=2, ep_size=2
            ),
        ),
        lambda slot: SimpleNamespace(slot=slot, use_deepep=True),
    )
    workspace.materialize()
    assert len(created) == 2
    assert creation_devices == [torch.device("cuda", 3)]
    evidence = workspace.evidence()
    assert evidence["allocation_pool_count"] == 2
    assert set(evidence["allocation_pool_ids"].values()) == {
        id(created[0]),
        id(created[1]),
    }
    assert evidence["expert_activation_pool_count"] == 0
    assert evidence["expert_activation_pool_id"] is None
    assert evidence["recv_observer_enabled"] is False
    monkeypatch.setenv("MEGATRON_LITE_EP_CHUNK_SCRATCH_TRACE", "1")
    assert workspace.evidence()["recv_observer_enabled"] is True

    lease0 = workspace.acquire(0)
    assert workspace.evidence()["active_lease_count"] == 1
    with lease0.deepep_recv_allocation():
        pass
    with lease0.deepep_recv_allocation():
        pass
    assert entered == [(created[0], torch.device("cuda", 3))] * 2
    lease0.release(_FakeEvent(ready=True))
    assert workspace.evidence()["consumer_event_guard_count"] == 1

    activation = workspace.acquire_expert_activation(
        stream=SimpleNamespace(device=torch.device("cuda", 3))
    )
    activation.release(_FakeEvent(ready=True))
    assert workspace.evidence()["expert_activation_pool_count"] == 1
    assert workspace.evidence()["expert_activation_pool_id"] == id(created[2])

    lease1 = workspace.acquire(1)
    with lease1.deepep_recv_allocation():
        pass
    assert len(created) == 3
    lease1.release(_FakeEvent(ready=True))
    assert workspace.evidence()["allocation_pool_count"] == 2

    workspace.release()
    assert workspace.evidence()["allocation_pool_count"] == 0
    # Direct workspace close does not tear down the registry-shared pool.
    assert workspace.evidence()["expert_activation_pool_count"] == 1
    assert workspace._registry is not None
    workspace._registry.release(workspace.key)
    assert workspace.evidence()["expert_activation_pool_count"] == 0


def test_three_ops_own_distinct_expert_activation_arenas(
    monkeypatch, transformer_engine_import_stub
):
    transformer_engine_import_stub()
    import megatron.lite.primitive.modules.moe_ep_chunk_overlap as overlap

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

    @contextmanager
    def use_device(_device):
        yield

    monkeypatch.setattr(overlap.torch.cuda, "device", use_device)
    monkeypatch.setattr(overlap.torch.cuda, "MemPool", lambda **_kwargs: object())
    registry = registry_type()
    profile = profile_type(max_input_rows=8, hidden_size=4, topk=2, ep_size=2)
    workspaces = []
    for op in ("forward", "backward", "fused_forward_backward"):
        workspace = registry.get_or_create(
            key_type(
                op=op,
                device_type="cuda",
                device_index=3,
                ep_group_id=43,
                dtype=torch.bfloat16,
                shape_profile=profile,
            ),
            lambda slot: SimpleNamespace(slot=slot, use_deepep=True),
        )
        if op == "backward":
            workspace.prepare_scratch(device=torch.device("cuda", 3))
            assert workspace.evidence()["dispatcher_count"] == 0
            assert workspace.evidence()["allocation_pool_count"] == 0
            assert workspace.evidence()["expert_activation_pool_count"] == 0
            lease = workspace.acquire_expert_activation(
                stream=SimpleNamespace(device=torch.device("cuda", 3))
            )
            lease.release(_FakeEvent(ready=True))
            assert workspace.evidence()["dispatcher_count"] == 0
            assert workspace.evidence()["allocation_pool_count"] == 0
        else:
            workspace.materialize()
        workspaces.append(workspace)

    comm_ids = {
        pool_id
        for workspace in workspaces
        for pool_id in workspace.evidence()["allocation_pool_ids"].values()
    }
    assert len(
        {workspace.evidence()["expert_activation_pool_id"] for workspace in workspaces}
    ) == 1
    assert None not in {
        workspace.evidence()["expert_activation_pool_id"] for workspace in workspaces
    }
    assert [
        workspace.evidence()["allocation_pool_count"] for workspace in workspaces
    ] == [2, 0, 2]
    assert len(comm_ids) == 4
    assert None not in comm_ids
    assert len({id(workspace._expert_activation_owner) for workspace in workspaces}) == 3
    assert len(
        {id(workspace._expert_activation_owner.coordinator) for workspace in workspaces}
    ) == 1


def test_expert_activation_arena_waits_only_for_its_consumer_event(
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
    workspace = registry_type().get_or_create(
        key_type(
            op="fused_forward_backward",
            device_type="cpu",
            device_index=None,
            ep_group_id=41,
            dtype=torch.bfloat16,
            shape_profile=profile_type(
                max_input_rows=8, hidden_size=4, topk=2, ep_size=2
            ),
        ),
        lambda slot: SimpleNamespace(slot=slot, use_deepep=True),
    )
    stream = _FakeStream()
    first = workspace.acquire_expert_activation(stream=stream)
    with pytest.raises(RuntimeError, match="expert activation arena is already leased"):
        workspace.acquire_expert_activation(stream=stream)

    pending = _FakeEvent(ready=False)
    first.release(pending)
    second = workspace.acquire_expert_activation(stream=stream)

    assert stream.waited == [pending]
    assert workspace.evidence()["expert_activation_waits"] == 1
    assert workspace.metrics()["runtime_allocations"] == 0
    assert workspace.metrics()["grows"] == 0
    second.release(_FakeEvent(ready=True))


def test_expert_activation_lease_size_classes_adjacent_rows_without_growth(
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
    workspace = registry_type().get_or_create(
        key_type(
            op="fused_forward_backward",
            device_type="cpu",
            device_index=None,
            ep_group_id=43,
            dtype=torch.float32,
            shape_profile=profile_type(
                max_input_rows=8, hidden_size=4, topk=2, ep_size=2
            ),
        ),
        lambda slot: SimpleNamespace(slot=slot, use_deepep=True),
    )
    stream = _FakeStream()
    first = workspace.acquire_expert_activation(stream=stream)
    first_tensor = first.tensor("fc1_input", (5, 4), dtype=torch.float32, device="cpu")
    # A caller-owned autograd output can require gradients for one invocation;
    # the persistent base must stay non-grad when its next lease aliases it.
    first_tensor.requires_grad_(True)
    first_ptr = first_tensor.data_ptr()
    pending = _FakeEvent(ready=False)
    first.release(pending)

    second = workspace.acquire_expert_activation(stream=stream)
    second_tensor = second.tensor(
        "fc1_input", (3, 4), dtype=torch.float32, device="cpu"
    )

    assert stream.waited == [pending]
    assert second_tensor.data_ptr() == first_ptr
    assert not second_tensor.requires_grad
    evidence = workspace.evidence()["expert_activation_tensors"]["fc1_input"]
    assert evidence["shape"][0] >= 5
    assert (
        evidence["nbytes"] - first_tensor.numel() * first_tensor.element_size()
        < 8 * 1024 * 1024
    )
    assert evidence["data_ptr"] == first_ptr
    grow_guard = _FakeEvent(ready=False)
    second.release(grow_guard)

    third = workspace.acquire_expert_activation(stream=stream)
    grown_tensor = third.tensor("fc1_input", (7, 4), dtype=torch.float32, device="cpu")

    assert stream.waited == [pending, grow_guard]
    grown_evidence = workspace.evidence()["expert_activation_tensors"]["fc1_input"]
    assert grown_evidence["data_ptr"] == first_ptr
    assert grown_tensor.data_ptr() == first_ptr
    assert workspace._expert_activation_owner.grows == 0
    third.release(_FakeEvent(ready=True))


def test_multilayer_microbatch_probe_separates_owner_lifetime_from_slot_growth(
    monkeypatch, transformer_engine_import_stub
):
    """Trace 48 layer references and eight microbatches through real workspace APIs.

    The deliberately small class is a CPU-only boundary witness, not a claim
    about job15448830's CUDA routing shapes.  It exercises the workspace/owner
    lifecycle APIs.  Each logical EP op owns an independent
    event-guarded arena; the trace separately exposes exact-shape slot growth.
    """
    transformer_engine_import_stub()
    import megatron.lite.primitive.modules.moe_ep_chunk_overlap as overlap

    monkeypatch.setattr(overlap, "_EXPERT_ACTIVATION_SIZE_CLASS_BYTES", 64)
    profile = overlap.EPChunkShapeProfile(
        max_input_rows=64, hidden_size=4, topk=2, ep_size=2
    )
    registry = overlap.EPChunkWorkspaceRegistry()
    workspaces = {
        op: registry.get_or_create(
            overlap.EPChunkWorkspaceKey(
                op=op,
                device_type="cpu",
                device_index=None,
                ep_group_id=4808,
                dtype=torch.float32,
                shape_profile=profile,
            ),
            lambda slot: SimpleNamespace(slot=slot, use_deepep=True),
        )
        for op in ("forward", "backward", "fused_forward_backward")
    }
    layers = [workspaces for _ in range(48)]
    assert {layer["forward"].key for layer in layers} == {workspaces["forward"].key}
    assert {layer["backward"].key for layer in layers} == {workspaces["backward"].key}
    assert {layer["fused_forward_backward"].key for layer in layers} == {
        workspaces["fused_forward_backward"].key
    }
    assert len({id(workspace._expert_activation_owner) for workspace in workspaces.values()}) == 3
    assert len(
        {id(workspace._expert_activation_owner.coordinator) for workspace in workspaces.values()}
    ) == 1

    trace = []

    def request(workspace, *, microbatch, rows, slot):
        slot_lease = workspace.acquire(slot, require_dispatcher=False)
        scratch = slot_lease.tensor(
            "grad_expert_out", (rows, 4), dtype=torch.float32, device="cpu"
        )
        slot_lease.release(_FakeEvent(ready=True))
        activation_lease = workspace.acquire_expert_activation()
        activations = {
            name: activation_lease.tensor(
                name, (rows, 4), dtype=torch.float32, device="cpu"
            )
            for name in ("fc1_input", "fc1_output", "fc2_output")
        }
        activation_lease.release(_FakeEvent(ready=True))
        evidence = workspace.evidence()
        trace.append(
            {
                "op": workspace.key.op,
                "workspace_key": workspace.key,
                "activation_owner": id(workspace._expert_activation_owner),
                "microbatch": microbatch,
                "slot": slot,
                "requested_shape": tuple(scratch.shape),
                "rounded_capacity": {
                    name: overlap._expert_activation_capacity_bytes(
                        tensor.numel() * tensor.element_size()
                    )
                    for name, tensor in activations.items()
                },
                "scratch_ptr": scratch.data_ptr(),
                "activation_ptrs": {
                    name: tensor.data_ptr() for name, tensor in activations.items()
                },
                "grows": evidence["grows"],
                "activation_grows": evidence["expert_activation_grows"],
                "active_leases": evidence["active_lease_count"],
                "slot_release_event": True,
                "slot_event_guards": evidence["consumer_event_guard_count"],
                "activation_release_event": evidence[
                    "expert_activation_event_guarded"
                ],
            }
        )

    # Five and seven rows share the 128-byte bucket; nine crosses once into
    # 192 bytes.  Later microbatches repeat the warmed maximum.  The owner
    # trace therefore distinguishes a bucket transition from concurrent lease
    # pressure without claiming these CPU rows are the CUDA routing rows.
    for microbatch, rows in enumerate((5, 7, 9, 9, 9, 9, 9, 9)):
        for workspace in workspaces.values():
            for slot in range(2):
                request(workspace, microbatch=microbatch, rows=rows, slot=slot)

    assert len(trace) == 48
    assert {entry["op"] for entry in trace} == set(workspaces)
    assert {entry["workspace_key"] for entry in trace} == {
        workspace.key for workspace in workspaces.values()
    }
    assert len({entry["activation_owner"] for entry in trace}) == 3
    assert all(entry["active_leases"] == 0 for entry in trace)
    assert all(entry["slot_release_event"] for entry in trace)
    assert all(0 <= entry["slot_event_guards"] <= 2 for entry in trace)
    assert all(entry["activation_release_event"] for entry in trace)
    assert [
        set(entry["rounded_capacity"].values()) for entry in trace[:18:6]
    ] == [{128}, {128}, {192}]
    for workspace in workspaces.values():
        assert workspace.metrics()["grows"] == 4
        assert workspace._expert_activation_owner.coordinator.grows == 3
        for slot in range(2):
            ptrs = [
                entry["scratch_ptr"]
                for entry in trace
                if entry["op"] == workspace.key.op and entry["slot"] == slot
            ]
            assert len(set(ptrs)) == 3
        for name in ("fc1_input", "fc1_output", "fc2_output"):
            activation_ptrs = [
                entry["activation_ptrs"][name]
                for entry in trace
                if entry["op"] == workspace.key.op
            ]
            assert len(set(activation_ptrs[4:])) == 1
    # The forward-only FC2 output reuses FC1-input raw storage; backward and
    # fused retain their own lifetimes, so same-named logical tensors need not
    # share one pointer across OPs.
    forward_trace = [entry for entry in trace if entry["op"] == "forward"]
    assert all(
        entry["activation_ptrs"]["fc1_input"]
        == entry["activation_ptrs"]["fc2_output"]
        for entry in forward_trace
    )
    coordinator = workspaces["forward"]._expert_activation_owner.coordinator
    assert coordinator.max_requested_bytes == {
        "fc1_input": 144,
        "fc1_output": 144,
        "fc2_output": 144,
    }
    assert coordinator.capacity_bytes == {
        "fc1_input": 192,
        "fc1_output": 192,
        "fc2_output": 192,
    }

    forward_held = workspaces["forward"].acquire_expert_activation()
    with pytest.raises(RuntimeError, match="coordinator is already claimed"):
        workspaces["fused_forward_backward"].acquire_expert_activation()
    pending = _FakeEvent(ready=False)
    forward_held.release(pending)
    handoff_stream = _FakeStream()
    fused_held = workspaces["fused_forward_backward"].acquire_expert_activation(
        stream=handoff_stream
    )
    assert handoff_stream.waited == [pending]
    fused_held.release(_FakeEvent(ready=True))
    owners = tuple(workspace._expert_activation_owner for workspace in workspaces.values())
    for workspace in workspaces.values():
        registry.release(workspace.key)
    assert not registry._expert_activation_owners
    assert all(not owner.arena.tensors for owner in owners)


def test_each_op_reuses_its_own_large_activation_names_across_layers(
    transformer_engine_import_stub,
):
    """Adjacent routed row counts must not cold-grow five live activation names."""
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
    profile = profile_type(max_input_rows=131963, hidden_size=16, topk=2, ep_size=2)

    def make_workspace(op):
        return registry.get_or_create(
            key_type(
                op=op,
                device_type="cpu",
                device_index=None,
                ep_group_id=97,
                dtype=torch.float32,
                shape_profile=profile,
            ),
            lambda slot: SimpleNamespace(slot=slot, use_deepep=True),
        )

    forward, fused = make_workspace("forward"), make_workspace("fused_forward_backward")
    assert forward._expert_activation_owner is not fused._expert_activation_owner
    assert forward._expert_activation_owner.coordinator is fused._expert_activation_owner.coordinator
    names = ("fc1_input", "fc1_output", "fc2_output", "fc1_dgrad", "fc2_dgrad")
    first = forward.acquire_expert_activation()
    first_ptrs = {
        name: first.tensor(
            name, (131750, 16), dtype=torch.float32, device="cpu"
        ).data_ptr()
        for name in names
    }
    # No-grad forward recycles FC1 input for FC2 output, while preserving the
    # normal dgrad alias.  It therefore owns three physical slots.
    assert first_ptrs["fc1_input"] == first_ptrs["fc2_output"]
    assert first_ptrs["fc1_dgrad"] == first_ptrs["fc2_dgrad"]
    assert first_ptrs["fc2_output"] != first_ptrs["fc2_dgrad"]
    assert len(set(first_ptrs.values())) == 3
    owner = forward._expert_activation_owner
    assert set(owner.arena.tensors) == {
        "fc1_input",
        "fc1_output",
        "fc2_dgrad",
    }
    pending = _FakeEvent(ready=False)
    first.release(pending)
    stream = _FakeStream()
    second = fused.acquire_expert_activation(stream=stream)
    second_ptrs = {
        name: second.tensor(
            name, (131963, 16), dtype=torch.float32, device="cpu"
        ).data_ptr()
        for name in names
    }
    assert stream.waited == [pending]
    # Fused has an independent logical owner, but handoff preserves the three
    # physical base addresses after the forward lease's consumer event.
    assert second_ptrs["fc1_input"] == first_ptrs["fc1_input"]
    assert second_ptrs["fc1_output"] == first_ptrs["fc1_output"]
    assert second_ptrs["fc2_output"] != first_ptrs["fc2_output"]
    assert second_ptrs["fc2_dgrad"] == second_ptrs["fc2_output"]
    assert second_ptrs["fc1_dgrad"] == second_ptrs["fc2_output"]
    assert forward._expert_activation_owner.grows == 0
    assert fused._expert_activation_owner.grows == 0
    for tensor in fused._expert_activation_owner.arena.tensors.values():
        requested_bytes = 131963 * 16 * tensor.element_size()
        assert (
            tensor.numel() * tensor.element_size() - requested_bytes < 8 * 1024 * 1024
        )
    second.release(_FakeEvent(ready=True))

    isolated = registry.get_or_create(
        key_type(
            op="forward",
            device_type="cpu",
            device_index=None,
            ep_group_id=98,
            dtype=torch.float32,
            shape_profile=profile,
        ),
        lambda slot: SimpleNamespace(slot=slot, use_deepep=True),
    )
    assert isolated._expert_activation_owner is not forward._expert_activation_owner
    registry.release(forward.key, stream=stream)
    assert not fused._expert_activation_owner.arena.tensors
    assert not forward._expert_activation_owner.arena.tensors
    registry.release(fused.key)
    assert not fused._expert_activation_owner.arena.tensors


def test_per_op_phase_handoff_keeps_one_physical_arena_until_final_release(
    transformer_engine_import_stub,
):
    """Logical OP releases do not clear shared physical graph-address storage."""
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
    profile = profile_type(max_input_rows=16, hidden_size=4, topk=2, ep_size=2)

    def get(op):
        return registry.get_or_create(
            key_type(
                op=op,
                device_type="cpu",
                device_index=None,
                ep_group_id=4816,
                dtype=torch.float32,
                shape_profile=profile,
            ),
            lambda slot: SimpleNamespace(slot=slot, use_deepep=True),
        )

    forward = get("forward")
    fused = get("fused_forward_backward")
    forward_lease = forward.acquire_expert_activation()
    forward_tensor = forward_lease.tensor(
        "fc1_input", (5, 4), dtype=torch.float32, device="cpu"
    )
    pending = _FakeEvent(ready=False)
    forward_lease.release(pending)
    stream = _FakeStream()
    fused_lease = fused.acquire_expert_activation(stream=stream)
    fused_tensor = fused_lease.tensor(
        "fc1_input", (5, 4), dtype=torch.float32, device="cpu"
    )
    fused_lease.release(_FakeEvent(ready=True))

    assert stream.waited == [pending]
    assert forward_tensor.data_ptr() == fused_tensor.data_ptr()
    assert forward._expert_activation_owner.arena.tensors
    assert fused._expert_activation_owner.arena.tensors
    assert forward._expert_activation_owner is not fused._expert_activation_owner
    assert forward._expert_activation_owner.coordinator is fused._expert_activation_owner.coordinator
    coordinator = forward._expert_activation_owner.coordinator
    registry.release(forward.key, stream=stream)
    assert not coordinator.arena.tensors
    assert not fused._expert_activation_owner.arena.tensors
    next_forward_lease = forward.acquire_expert_activation()
    next_forward_tensor = next_forward_lease.tensor(
        "fc1_input", (5, 4), dtype=torch.float32, device="cpu"
    )
    next_forward_lease.release(_FakeEvent(ready=True))
    assert next_forward_tensor.data_ptr() != 0
    assert coordinator.rehydrates == 1
    registry.release(forward.key)
    registry.release(fused.key)
    assert not coordinator.arena.tensors
    assert not registry._expert_activation_arenas


def test_all_caller_owned_activations_bound_rows_and_bind_logical_trailing_shape(
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
    profile = profile_type(max_input_rows=4, hidden_size=2, topk=2, ep_size=2)
    registry = registry_type()
    workspace = registry.get_or_create(
        key_type(
            op="backward",
            device_type="cpu",
            device_index=None,
            ep_group_id=4818,
            dtype=torch.float32,
            shape_profile=profile,
        ),
        lambda slot: SimpleNamespace(slot=slot, use_deepep=True),
    )
    names = ("fc1_output", "fc2_output", "fc2_dgrad", "fc1_dgrad")
    lease = workspace.acquire_expert_activation()
    with pytest.raises(RuntimeError, match="Unknown EP chunk expert activation"):
        lease.tensor("unknown", (1, 5), dtype=torch.float32, device="cpu")
    with pytest.raises(RuntimeError, match="exceeds profile ceiling"):
        lease.tensor(
            "fc1_output",
            (profile.max_expert_rows, 5, 1),
            dtype=torch.float32,
            device="cpu",
        )
    with pytest.raises(RuntimeError, match="exceeds profile ceiling"):
        lease.tensor(
            "fc1_output",
            (profile.max_expert_rows, 5),
            dtype=torch.float64,
            device="cpu",
        )
    for name in names:
        lease.tensor(
            name,
            (profile.max_expert_rows, 5),
            dtype=torch.float32,
            device="cpu",
        )
    for name in names:
        with pytest.raises(RuntimeError, match="exceeds profile ceiling"):
            lease.tensor(
                name,
                (profile.max_expert_rows + 1, 5),
                dtype=torch.float32,
                device="cpu",
            )
    with pytest.raises(RuntimeError, match="exceeds profile ceiling"):
        lease.tensor(
            "fc1_input",
            (profile.max_expert_rows, profile.hidden_size + 1),
            dtype=torch.float32,
            device="cpu",
        )
    with pytest.raises(RuntimeError, match="exceeds profile ceiling"):
        lease.tensor(
            "fc1_input",
            (profile.max_expert_rows + 1, profile.hidden_size),
            dtype=torch.float32,
            device="cpu",
        )
    lease.release(_FakeEvent(ready=True))

    rebound = workspace.acquire_expert_activation()
    with pytest.raises(RuntimeError, match="trailing shape"):
        rebound.tensor(
            "fc1_output",
            (profile.max_expert_rows, 6),
            dtype=torch.float32,
            device="cpu",
        )
    rebound.release(_FakeEvent(ready=True))

    fused = registry.get_or_create(
        key_type(
            op="fused_forward_backward",
            device_type="cpu",
            device_index=None,
            ep_group_id=4819,
            dtype=torch.float32,
            shape_profile=profile,
        ),
        lambda slot: SimpleNamespace(slot=slot, use_deepep=True),
    )
    fused_lease = fused.acquire_expert_activation()
    fc2_output = fused_lease.tensor(
        "fc2_output", (profile.max_expert_rows, 6), dtype=torch.float32, device="cpu"
    )
    fc2_dgrad = fused_lease.tensor(
        "fc2_dgrad", (profile.max_expert_rows, 5), dtype=torch.float32, device="cpu"
    )
    fc1_dgrad = fused_lease.tensor(
        "fc1_dgrad", (profile.max_expert_rows, 4), dtype=torch.float32, device="cpu"
    )
    assert fc2_output.data_ptr() == fc2_dgrad.data_ptr() == fc1_dgrad.data_ptr()
    fused_lease.release(_FakeEvent(ready=True))

    coordinator = workspace._expert_activation_owner.coordinator
    registry.release(workspace.key)
    assert not coordinator.logical_trailing_shapes
    fused_coordinator = fused._expert_activation_owner.coordinator
    registry.release(fused.key)
    assert not fused_coordinator.logical_trailing_shapes


def test_expert_activation_growth_uses_requested_size_class_not_geometric_headroom(
    monkeypatch, transformer_engine_import_stub
):
    """A grown arena retains only the requested 8-MiB size class."""
    transformer_engine_import_stub()
    import megatron.lite.primitive.modules.moe_ep_chunk_overlap as overlap

    monkeypatch.setattr(overlap, "_EXPERT_ACTIVATION_SIZE_CLASS_BYTES", 64)
    registry = overlap.EPChunkWorkspaceRegistry()
    profile = overlap.EPChunkShapeProfile(
        max_input_rows=16, hidden_size=4, topk=2, ep_size=2
    )
    workspace = registry.get_or_create(
        overlap.EPChunkWorkspaceKey(
            op="fused_forward_backward",
            device_type="cpu",
            device_index=None,
            ep_group_id=4817,
            dtype=torch.float32,
            shape_profile=profile,
        ),
        lambda slot: SimpleNamespace(slot=slot, use_deepep=True),
    )
    first = workspace.acquire_expert_activation()
    first.tensor("fc1_input", (16, 4), dtype=torch.float32, device="cpu")
    first.release(_FakeEvent(ready=True))

    second = workspace.acquire_expert_activation()
    second.tensor("fc1_input", (17, 4), dtype=torch.float32, device="cpu")
    second.release(_FakeEvent(ready=True))

    evidence = workspace.evidence()["expert_activation_tensors"]["fc1_input"]
    assert workspace._expert_activation_owner.grows == 1
    assert evidence["nbytes"] == overlap._expert_activation_capacity_bytes(17 * 4 * 4)
    assert evidence["nbytes"] <= overlap._expert_activation_capacity_bytes(
        profile.max_expert_rows * profile.hidden_size * 4
    )


def test_delayed_fc2_wgrad_reads_output_before_safe_dgrad_slot_reuse(
    transformer_engine_import_stub,
):
    """FC2 output must survive until its delayed Wgrad callback consumes it."""
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
    workspace = registry_type().get_or_create(
        key_type(
            # Normal/deferred Wgrad must retain FC2 output separately.
            op="backward",
            device_type="cpu",
            device_index=None,
            ep_group_id=100,
            dtype=torch.float32,
            shape_profile=profile_type(
                max_input_rows=8, hidden_size=2, topk=2, ep_size=2
            ),
        ),
        lambda slot: SimpleNamespace(slot=slot, use_deepep=True),
    )
    lease = workspace.acquire_expert_activation()
    fc2_output = lease.tensor("fc2_output", (2, 2), dtype=torch.float32, device="cpu")
    fc2_dgrad = lease.tensor("fc2_dgrad", (2, 2), dtype=torch.float32, device="cpu")
    expected_fc2_output = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    fc2_output.copy_(expected_fc2_output)
    delayed_wgrad_read = torch.empty_like(fc2_output)

    # This is the production ordering bug: FC2 dgrad is written before the
    # delayed FC2 Wgrad callback reads its saved forward input.
    fc2_dgrad.fill_(17.0)
    delayed_wgrad_read.copy_(fc2_output)

    assert fc2_output.data_ptr() != fc2_dgrad.data_ptr()
    torch.testing.assert_close(delayed_wgrad_read, expected_fc2_output)
    lease.release(_FakeEvent(ready=True))


def test_forward_reuses_fc1_input_only_while_backward_and_fused_keep_their_contracts(
    transformer_engine_import_stub,
):
    """Only no-grad forward may recycle its dead FC1-input raw storage for FC2."""
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
    profile = profile_type(max_input_rows=8, hidden_size=2, topk=2, ep_size=2)

    def workspace(op):
        return registry.get_or_create(
            key_type(
                op=op,
                device_type="cpu",
                device_index=None,
                ep_group_id=102,
                dtype=torch.float32,
                shape_profile=profile,
            ),
            lambda slot: SimpleNamespace(slot=slot, use_deepep=True),
        )

    names = ("fc1_input", "fc1_output", "fc2_output", "fc2_dgrad", "fc1_dgrad")
    forward = workspace("forward")
    forward_lease = forward.acquire_expert_activation()
    forward_ptrs = {
        name: forward_lease.tensor(
            name, (2, 2), dtype=torch.float32, device="cpu"
        ).data_ptr()
        for name in names
    }
    assert forward_ptrs["fc1_input"] == forward_ptrs["fc2_output"]
    assert forward_ptrs["fc2_output"] != forward_ptrs["fc2_dgrad"]
    assert forward_ptrs["fc2_dgrad"] == forward_ptrs["fc1_dgrad"]
    forward_lease.release(_FakeEvent(ready=True))

    fused = workspace("fused_forward_backward")
    fused_lease = fused.acquire_expert_activation()
    fused_ptrs = {
        name: fused_lease.tensor(
            name, (2, 2), dtype=torch.float32, device="cpu"
        ).data_ptr()
        for name in names
    }
    assert fused_ptrs["fc2_output"] == fused_ptrs["fc2_dgrad"]
    assert fused_ptrs["fc2_dgrad"] == fused_ptrs["fc1_dgrad"]
    assert fused_ptrs["fc1_input"] != fused_ptrs["fc2_output"]
    assert len(set(fused_ptrs.values())) == 3
    fused_lease.release(_FakeEvent(ready=True))

    normal = workspace("backward")
    normal_lease = normal.acquire_expert_activation()
    normal_ptrs = {
        name: normal_lease.tensor(
            name, (2, 2), dtype=torch.float32, device="cpu"
        ).data_ptr()
        for name in names
    }
    assert normal_ptrs["fc2_output"] != normal_ptrs["fc2_dgrad"]
    assert normal_ptrs["fc2_dgrad"] == normal_ptrs["fc1_dgrad"]
    assert normal_ptrs["fc1_input"] != normal_ptrs["fc2_output"]
    normal_lease.release(_FakeEvent(ready=True))


def test_fc1_dgrad_reuses_fc2_dgrad_only_after_swiglu_consumes_it(
    transformer_engine_import_stub,
):
    """The shared dgrad slot is overwritten only after its SwiGLU consumer."""
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
    workspace = registry_type().get_or_create(
        key_type(
            op="fused_forward_backward",
            device_type="cpu",
            device_index=None,
            ep_group_id=101,
            dtype=torch.float32,
            shape_profile=profile_type(
                max_input_rows=8, hidden_size=2, topk=2, ep_size=2
            ),
        ),
        lambda slot: SimpleNamespace(slot=slot, use_deepep=True),
    )
    lease = workspace.acquire_expert_activation()
    fc2_dgrad = lease.tensor("fc2_dgrad", (2, 2), dtype=torch.float32, device="cpu")
    fc1_dgrad = lease.tensor("fc1_dgrad", (2, 2), dtype=torch.float32, device="cpu")
    expected_fc2_dgrad = torch.tensor([[2.0, 3.0], [5.0, 7.0]])
    fc2_dgrad.copy_(expected_fc2_dgrad)
    events = []
    consumed = torch.empty_like(fc2_dgrad)

    class SwiGLUConsumer(torch.autograd.Function):
        @staticmethod
        def forward(ctx, value):
            return value

        @staticmethod
        def backward(ctx, grad_output):
            events.append("swiglu consumes fc2_dgrad")
            consumed.copy_(grad_output)
            return grad_output

    input_value = torch.ones_like(fc2_dgrad, requires_grad=True)
    torch.autograd.grad(SwiGLUConsumer.apply(input_value), input_value, fc2_dgrad)
    assert events == ["swiglu consumes fc2_dgrad"]
    fc1_dgrad.fill_(19.0)
    events.append("fc1 writes shared dgrad")

    assert fc1_dgrad.data_ptr() == fc2_dgrad.data_ptr()
    assert events == ["swiglu consumes fc2_dgrad", "fc1 writes shared dgrad"]
    torch.testing.assert_close(consumed, expected_fc2_dgrad)
    lease.release(_FakeEvent(ready=True))


def test_failed_colored_alias_grow_does_not_bind_or_raise_high_watermark(
    transformer_engine_import_stub,
):
    """A failed new logical alias request must not become a sticky binding."""
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
    workspace = registry_type().get_or_create(
        key_type(
            op="fused_forward_backward",
            device_type="cpu",
            device_index=None,
            ep_group_id=99,
            dtype=torch.float32,
            shape_profile=profile_type(
                max_input_rows=8, hidden_size=4, topk=2, ep_size=2
            ),
        ),
        lambda slot: SimpleNamespace(slot=slot, use_deepep=True),
    )
    first_width = 2 * 1024 * 1024
    first = workspace.acquire_expert_activation()
    first_view = first.tensor(
        "fc2_dgrad", (1, first_width), dtype=torch.float32, device="cpu"
    )
    first_ptr = first_view.data_ptr()
    with pytest.raises(RuntimeError, match="during an active lease"):
        first.tensor(
            "fc1_dgrad", (1, first_width + 1), dtype=torch.float32, device="cpu"
        )
    assert first_view.data_ptr() == first_ptr
    coordinator = workspace._expert_activation_owner.coordinator
    assert "fc1_dgrad" not in coordinator.logical_trailing_shapes
    assert coordinator.max_requested_bytes["fc2_output"] == first_width * 4
    first.release(_FakeEvent(ready=True))

    # The failed width must not bind fc1_dgrad: the next lease can issue its
    # smaller, compatible logical view from the already allocated raw slot.
    second = workspace.acquire_expert_activation()
    second_view = second.tensor(
        "fc1_dgrad", (1, first_width), dtype=torch.float32, device="cpu"
    )
    assert second_view.data_ptr() == first_ptr
    second.release(_FakeEvent(ready=True))


@pytest.mark.parametrize(
    ("requested_bytes", "capacity_bytes"),
    [
        (532_676_608, 536_870_912),
        (530_579_456, 536_870_912),
        (534_773_760, 536_870_912),
    ],
)
def test_expert_activation_rounds_real_32k_requests_to_8mib_classes(
    requested_bytes, capacity_bytes, transformer_engine_import_stub
):
    """32K Qwen3 records remain in one 8 MiB class without cold growth."""
    transformer_engine_import_stub()
    from megatron.lite.primitive.modules.moe_ep_chunk_overlap import (
        _EXPERT_ACTIVATION_SIZE_CLASS_BYTES,
        _expert_activation_capacity_bytes,
    )

    assert _EXPERT_ACTIVATION_SIZE_CLASS_BYTES == 8 * 1024 * 1024
    assert _expert_activation_capacity_bytes(requested_bytes) == capacity_bytes
    # The next routed block may request the full 512 MiB; it must reuse this
    # class rather than cold-grow a second persistent buffer.
    assert _expert_activation_capacity_bytes(536_870_912) == capacity_bytes
    assert capacity_bytes - requested_bytes < _EXPERT_ACTIVATION_SIZE_CLASS_BYTES


def test_unbound_workspace_binds_to_first_runtime_stream_device(
    monkeypatch, transformer_engine_import_stub
):
    transformer_engine_import_stub()
    import megatron.lite.primitive.modules.moe_ep_chunk_overlap as overlap

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
    device_contexts = []

    @contextmanager
    def use_device(device):
        device_contexts.append(torch.device(device))
        yield

    @contextmanager
    def use_pool(_pool, device=None):
        yield

    monkeypatch.setattr(overlap.torch.cuda, "device", use_device)
    monkeypatch.setattr(overlap.torch.cuda, "MemPool", lambda **_kwargs: object())
    monkeypatch.setattr(overlap.torch.cuda, "use_mem_pool", use_pool)
    workspace = registry_type().get_or_create(
        key_type(
            op="forward",
            device_type="cuda",
            device_index=None,
            ep_group_id=37,
            dtype=torch.bfloat16,
            shape_profile=profile_type(
                max_input_rows=8, hidden_size=4, topk=2, ep_size=2
            ),
        ),
        lambda slot: SimpleNamespace(slot=slot, use_deepep=True),
    )
    stream = SimpleNamespace(device=torch.device("cuda", 5))

    lease = workspace.acquire(0, stream=stream)

    assert device_contexts == [torch.device("cuda", 5)]
    assert workspace.evidence()["materialized_device"] == "cuda:5"
    lease.release(_FakeEvent(ready=True))

    with pytest.raises(RuntimeError, match="already materialized on cuda:5"):
        workspace.acquire(1, stream=SimpleNamespace(device=torch.device("cuda", 6)))

    workspace.release()
    rebound = workspace.acquire(
        0, stream=SimpleNamespace(device=torch.device("cuda", 6))
    )
    assert device_contexts == [torch.device("cuda", 5), torch.device("cuda", 6)]
    assert workspace.evidence()["materialized_device"] == "cuda:6"
    rebound.release(_FakeEvent(ready=True))


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
        shape_profile=profile_type(max_input_rows=8, hidden_size=4, topk=2, ep_size=2),
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
        "tensor_details": {},
        "dispatcher_count": 0,
        "deepep_buffer_count": 0,
        "deepep_buffer_resident_bytes": 0,
        "allocation_pool_count": 0,
        "allocation_pool_ids": {},
        "expert_activation_pool_count": 0,
        "expert_activation_pool_id": None,
        "expert_activation_tensors": {},
        "expert_activation_waits": 0,
        "expert_activation_allocations": 0,
        "expert_activation_grows": 0,
        "expert_activation_parks": 0,
        "expert_activation_rehydrates": 0,
        "expert_activation_in_use": False,
        "expert_activation_event_guarded": False,
        "expert_activation_arena_id": id(
            workspace._expert_activation_owner.coordinator
        ),
        "expert_activation_arena_claimed_op": None,
        "expert_activation_max_requested_bytes": {},
        "expert_activation_capacity_bytes": {},
        "active_lease_count": 0,
        "consumer_event_guard_count": 0,
        "recv_observer_enabled": False,
        "materialized_device": None,
        "materialized": False,
    }
    assert created == []

    workspace.materialize(device="cpu")
    first_dispatchers = tuple(created)
    assert workspace.evidence()["data_ptrs"] == {}
    for slot in range(2):
        lease = workspace.acquire(slot)
        lease.tensor("grad_expert_out", (4, 4), dtype=torch.float32, device="cpu")
        lease.release(_FakeEvent(ready=True))
    activation = workspace.acquire_expert_activation()
    activation.tensor("fc1_input", (4, 4), dtype=torch.float32, device="cpu")
    activation.release(_FakeEvent(ready=True))
    assert workspace.evidence()["dispatcher_count"] == 2

    registry.release(key)
    assert workspace.evidence()["dispatcher_count"] == 0
    assert workspace.evidence()["deepep_buffer_resident_bytes"] == 0
    assert workspace.evidence()["data_ptrs"] == {}
    assert not workspace._expert_activation_owner.arena.tensors
    registry.release(key)

    workspace.materialize(device="cpu")
    assert registry.get_or_create(key, factory) is workspace
    assert workspace.evidence()["data_ptrs"] == {}
    for slot in range(2):
        lease = workspace.acquire(slot)
        lease.tensor("grad_expert_out", (4, 4), dtype=torch.float32, device="cpu")
        lease.release(_FakeEvent(ready=True))
    assert workspace.evidence()["data_ptrs"]
    activation = workspace.acquire_expert_activation()
    activation.tensor("fc1_input", (4, 4), dtype=torch.float32, device="cpu")
    activation.release(_FakeEvent(ready=True))
    registry.release(key)
    assert not workspace._expert_activation_owner.arena.tensors

    rebuilt = registry.get_or_create(key, factory)
    assert rebuilt is not workspace
    rebuilt.materialize(device="cpu")
    for slot in range(2):
        lease = rebuilt.acquire(slot)
        lease.tensor("grad_expert_out", (4, 4), dtype=torch.float32, device="cpu")
        lease.release(_FakeEvent(ready=True))
    assert all(
        rebuilt.dispatcher(slot) is not first_dispatchers[slot] for slot in range(2)
    )
    assert rebuilt.evidence()["data_ptrs"]
    registry.release(key)
    registry._expert_activation_owners[workspace._expert_activation_owner.key] = (
        object()
    )
    with pytest.raises(RuntimeError, match="replaced activation owner"):
        workspace.materialize(device="cpu")


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
        shape_profile=profile_type(max_input_rows=8, hidden_size=4, topk=2, ep_size=2),
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
    profile = profile_type.for_two_slot_chunked_ep(
        max_input_rows=9, hidden_size=4, topk=2, ep_size=8
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

    profile = profile_type.for_two_slot_chunked_ep(
        max_input_rows=17, hidden_size=64, topk=8, ep_size=8
    )

    assert profile.max_recv_rows == 9 * 8
    assert profile.max_expert_rows == 9 * 8 * 8
    assert profile.topk == 8


@pytest.mark.parametrize("rows", [0, 1, 72])
def test_shape_profile_accepts_recv_rows_at_or_below_capacity(
    rows, transformer_engine_import_stub
):
    profile = _symbols(transformer_engine_import_stub)[4](
        max_input_rows=17, hidden_size=4, topk=8, ep_size=8
    )

    profile.validate_recv_rows(rows)


def test_shape_profile_rejects_recv_rows_above_capacity(transformer_engine_import_stub):
    profile = _symbols(transformer_engine_import_stub)[4](
        max_input_rows=17, hidden_size=4, topk=8, ep_size=8
    )

    with pytest.raises(RuntimeError, match="recv rows 73.*capacity 72"):
        profile.validate_recv_rows(73)


@pytest.mark.parametrize("rows", [0, 73, 576])
def test_shape_profile_accepts_expert_rows_at_or_below_capacity(
    rows, transformer_engine_import_stub
):
    profile = _symbols(transformer_engine_import_stub)[4](
        max_input_rows=17, hidden_size=4, topk=8, ep_size=8
    )

    profile.validate_expert_rows(rows)


def test_shape_profile_rejects_expert_rows_above_capacity(
    transformer_engine_import_stub,
):
    profile = _symbols(transformer_engine_import_stub)[4](
        max_input_rows=17, hidden_size=4, topk=8, ep_size=8
    )

    with pytest.raises(RuntimeError, match="expert rows 577.*capacity 576"):
        profile.validate_expert_rows(577)


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
    assert "require_dispatcher=False" in backward_source


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

    assert "forward_workspace_lease" not in fields
    assert "lease.release(consumed)" in forward_source
    expert_finished = forward_source.index("expert_out = self.experts(")
    context_saved = forward_source.index("saved_chunks[chunk_idx] =")
    slot_released = forward_source.index("lease.release(consumed)")
    assert expert_finished < context_saved < slot_released
    for saved_tensor in ("recv_probs_base", "dispatched", "probs", "expert_out_edge"):
        assert saved_tensor in fields
    assert 'handle=state["handle"]' in forward_source
