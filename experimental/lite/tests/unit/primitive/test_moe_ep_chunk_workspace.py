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


def test_backward_scratch_grows_to_observed_shape_but_not_beyond_profile(
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
    assert workspace.evidence()["data_ptrs"] == {}
    lease = workspace.acquire(0)
    first = lease.tensor("grad_recv_hidden", (8, 4), dtype=torch.float32, device="cpu")
    grown = lease.tensor("grad_recv_hidden", (9, 4), dtype=torch.float32, device="cpu")

    assert grown.shape == (9, 4)
    assert grown.data_ptr() != first.data_ptr()
    assert workspace.metrics()["grows"] == 1

    with pytest.raises(RuntimeError, match="exceeds fixed profile capacity"):
        lease.tensor("grad_recv_hidden", (17, 4), dtype=torch.float32, device="cpu")

    assert workspace.metrics()["fallbacks"] == 0


def test_backward_scratch_is_actual_shape_lazy_then_steady_state_stable(
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

    workspace.materialize(device="cpu")
    assert workspace.metrics()["allocations"] == 0
    assert workspace.evidence()["data_ptrs"] == {}

    first_ptrs = {}
    for slot in range(2):
        lease = workspace.acquire(slot)
        first_ptrs[(slot, "grad_recv_hidden")] = lease.tensor(
            "grad_recv_hidden", (9, 4), dtype=torch.float32, device="cpu"
        ).data_ptr()
        first_ptrs[(slot, "grad_recv_probs")] = lease.tensor(
            "grad_recv_probs", (9, 2), dtype=torch.float32, device="cpu"
        ).data_ptr()
        lease.release(_FakeEvent(ready=True))

    first_metrics = workspace.metrics().copy()
    assert first_metrics["allocations"] == 4
    assert first_metrics["runtime_allocations"] == 4

    second_ptrs = {}
    for slot in range(2):
        lease = workspace.acquire(slot)
        second_ptrs[(slot, "grad_recv_hidden")] = lease.tensor(
            "grad_recv_hidden", (9, 4), dtype=torch.float32, device="cpu"
        ).data_ptr()
        second_ptrs[(slot, "grad_recv_probs")] = lease.tensor(
            "grad_recv_probs", (9, 2), dtype=torch.float32, device="cpu"
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
                max_input_rows=8,
                hidden_size=4,
                topk=2,
                ep_size=4,
            ),
        ),
        lambda slot: created.append(slot) or f"dispatcher-{slot}",
    )

    lease = workspace.acquire(0, require_dispatcher=False)
    scratch = lease.tensor(
        "grad_recv_hidden", (9, 4), dtype=torch.float32, device="cpu"
    )

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
                max_input_rows=8,
                hidden_size=4,
                topk=2,
                ep_size=4,
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
                max_input_rows=8,
                hidden_size=4,
                topk=2,
                ep_size=4,
            ),
        ),
        lambda slot: f"dispatcher-{slot}",
    )
    lease = workspace.acquire(0)
    tensor = lease.tensor("grad_recv_hidden", (9, 4), dtype=torch.float32, device="cpu")
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
                max_input_rows=8,
                hidden_size=4,
                topk=2,
                ep_size=4,
            ),
        ),
        lambda slot: f"dispatcher-{slot}",
    )
    lease = workspace.acquire(0, require_dispatcher=False)
    lease.tensor("grad_recv_hidden", (9, 4), dtype=torch.float32, device="cpu")
    lease.tensor("grad_recv_probs", (9, 2), dtype=torch.float32, device="cpu")
    lease.release(_FakeEvent(ready=True))

    details = workspace.evidence()["tensor_details"]
    assert details == {
        "0:grad_recv_hidden": {
            "shape": (9, 4),
            "dtype": "torch.float32",
            "nbytes": 9 * 4 * 4,
        },
        "0:grad_recv_probs": {
            "shape": (9, 2),
            "dtype": "torch.float32",
            "nbytes": 9 * 2 * 4,
        },
    }


def test_scratch_only_lease_allocates_from_borrowed_forward_arena(
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
                max_input_rows=8,
                hidden_size=4,
                topk=2,
                ep_size=4,
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

    lease = workspace.acquire(
        0,
        require_dispatcher=False,
        allocation_arena=FakeArena(),
    )
    lease.tensor("grad_recv_hidden", (9, 4), dtype=torch.float32, device="cpu")

    assert entered == ["enter", "exit"]
    assert workspace.evidence()["dispatcher_count"] == 0
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
                max_input_rows=8,
                hidden_size=4,
                topk=2,
                ep_size=4,
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
    lease.tensor("grad_recv_hidden", (9, 4), dtype=torch.float32, device="cpu")

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
    workspace.materialize(device="cpu")

    evidence = workspace.evidence()
    assert evidence["dispatcher_count"] == 2
    assert evidence["deepep_buffer_count"] == 2
    assert evidence["deepep_buffer_resident_bytes"] == 2048
    assert evidence["caller_owned_recv_proven"] is False


def test_workspace_lazily_owns_one_cuda_mem_pool_per_slot(
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
                max_input_rows=8,
                hidden_size=4,
                topk=2,
                ep_size=2,
            ),
        ),
        lambda slot: SimpleNamespace(slot=slot, use_deepep=True),
    )
    workspace.materialize()
    assert len(created) == 2
    assert creation_devices == [torch.device("cuda", 3)]
    assert workspace.evidence()["allocation_pool_count"] == 2

    lease0 = workspace.acquire(0)
    with lease0.deepep_recv_allocation():
        pass
    with lease0.deepep_recv_allocation():
        pass
    assert entered == [(created[0], torch.device("cuda", 3))] * 2
    lease0.release(_FakeEvent(ready=True))

    lease1 = workspace.acquire(1)
    with lease1.deepep_recv_allocation():
        pass
    assert len(created) == 2
    lease1.release(_FakeEvent(ready=True))
    assert workspace.evidence()["allocation_pool_count"] == 2

    workspace.release()
    assert workspace.evidence()["allocation_pool_count"] == 0


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
                max_input_rows=8,
                hidden_size=4,
                topk=2,
                ep_size=2,
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
        "tensor_details": {},
        "dispatcher_count": 0,
        "deepep_buffer_count": 0,
        "deepep_buffer_resident_bytes": 0,
        "allocation_pool_count": 0,
        "allocation_pool_scope": "deepep_dispatch_recv",
        "materialized_device": None,
        "caller_owned_recv_proven": False,
        "materialized": False,
    }
    assert created == []

    workspace.materialize(device="cpu")
    first_dispatchers = tuple(created)
    assert workspace.evidence()["data_ptrs"] == {}
    for slot in range(2):
        lease = workspace.acquire(slot)
        lease.tensor("grad_recv_hidden", (4, 4), dtype=torch.float32, device="cpu")
        lease.tensor("grad_recv_probs", (4, 2), dtype=torch.float32, device="cpu")
        lease.release(_FakeEvent(ready=True))
    first_ptrs = set(workspace.evidence()["data_ptrs"].values())
    assert workspace.evidence()["dispatcher_count"] == 2

    registry.release(key)
    assert workspace.evidence()["dispatcher_count"] == 0
    assert workspace.evidence()["deepep_buffer_resident_bytes"] == 0
    assert workspace.evidence()["data_ptrs"] == {}
    registry.release(key)

    workspace.materialize(device="cpu")
    assert registry.get_or_create(key, factory) is workspace
    assert workspace.evidence()["data_ptrs"] == {}
    for slot in range(2):
        lease = workspace.acquire(slot)
        lease.tensor("grad_recv_hidden", (4, 4), dtype=torch.float32, device="cpu")
        lease.tensor("grad_recv_probs", (4, 2), dtype=torch.float32, device="cpu")
        lease.release(_FakeEvent(ready=True))
    assert set(workspace.evidence()["data_ptrs"].values()).isdisjoint(first_ptrs)
    registry.release(key)

    rebuilt = registry.get_or_create(key, factory)
    assert rebuilt is not workspace
    rebuilt.materialize(device="cpu")
    for slot in range(2):
        lease = rebuilt.acquire(slot)
        lease.tensor("grad_recv_hidden", (4, 4), dtype=torch.float32, device="cpu")
        lease.tensor("grad_recv_probs", (4, 2), dtype=torch.float32, device="cpu")
        lease.release(_FakeEvent(ready=True))
    assert all(
        rebuilt.dispatcher(slot) is not first_dispatchers[slot] for slot in range(2)
    )
    assert set(rebuilt.evidence()["data_ptrs"].values()).isdisjoint(first_ptrs)


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

    assert "workspace_lease" not in fields
    assert "lease.release(consumed)" in forward_source
    assert 'handle=state["handle"]' in forward_source
