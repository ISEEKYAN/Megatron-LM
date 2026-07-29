# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import ast
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


def test_chunk_overlap_source_has_nsys_phase_ranges():
    source_path = (
        Path(__file__).parents[3]
        / "megatron"
        / "lite"
        / "primitive"
        / "modules"
        / "moe_ep_chunk_overlap.py"
    )
    source = source_path.read_text()

    for phase in (
        "forward.dispatch",
        "forward.expert",
        "forward.combine",
        "backward.dispatch",
        "backward.expert",
        "backward.combine",
        "backward.wgrad",
    ):
        assert phase in source


def test_experts_accept_te_dbias_placeholders_when_bias_is_disabled(
    monkeypatch, transformer_engine_import_stub
):
    transformer_engine_import_stub()
    from megatron.lite.primitive.modules import experts as experts_module

    class FakeStore:
        def __init__(self, linear):
            self.linear = linear
            self.pending = 1
            self.context = SimpleNamespace(empty=lambda: self.pending == 0)

        @staticmethod
        def delay_wgrad_compute():
            return True

        def pop(self):
            self.pending -= 1
            grads = [
                getattr(self.linear, f"weight{idx}").main_grad
                for idx in range(self.linear.num_gemms)
            ]
            return (None, [torch.empty(0), torch.empty(0)], None), [None, None, grads]

    class FakeGroupedLinear(torch.nn.Module):
        def __init__(self, num_gemms, *_args, bias, **_kwargs):
            super().__init__()
            self.num_gemms = num_gemms
            self.use_bias = bias
            for idx in range(num_gemms):
                param = torch.nn.Parameter(torch.zeros(2, 2))
                param.main_grad = torch.ones(2, 2)
                self.register_parameter(f"weight{idx}", param)
            self.wgrad_store = FakeStore(self)

    monkeypatch.setattr(
        experts_module.te, "GroupedLinear", FakeGroupedLinear, raising=False
    )
    config = SimpleNamespace(
        num_experts=2,
        hidden_size=2,
        moe_intermediate_size=2,
        swiglu_limit=0.0,
    )
    ps = SimpleNamespace(ep_size=1, etp_size=1, tp_size=1, etp_group=None)
    experts = experts_module.Experts(config, ps, delay_wgrad_compute=True)

    experts.flush_delayed_weight_grads(num_contexts=1)

    assert all(parameter.grad is None for parameter in experts.parameters())


def test_delayed_expert_wgrads_reuse_distopt_main_grad(
    monkeypatch, transformer_engine_import_stub
):
    transformer_engine_import_stub()
    from megatron.lite.primitive.modules import experts as experts_module

    grouped_linear_kwargs = []

    class FakeStore:
        context = SimpleNamespace(empty=lambda: False)

        @staticmethod
        def delay_wgrad_compute():
            return True

        @staticmethod
        def pop():
            return (None, [None, None], None), [None, None, []]

    class FakeGroupedLinear(torch.nn.Module):
        def __init__(self, num_gemms, *_args, bias, **kwargs):
            super().__init__()
            grouped_linear_kwargs.append(kwargs)
            self.use_bias = bias
            self.wgrad_store = FakeStore()
            for idx in range(num_gemms):
                param = torch.nn.Parameter(torch.zeros(2, 2))
                param.main_grad = torch.zeros_like(param)
                self.register_parameter(f"weight{idx}", param)

    monkeypatch.setattr(
        experts_module.te, "GroupedLinear", FakeGroupedLinear, raising=False
    )
    config = SimpleNamespace(
        num_experts=2,
        hidden_size=2,
        moe_intermediate_size=2,
        swiglu_limit=0.0,
    )
    ps = SimpleNamespace(ep_size=1, etp_size=1, tp_size=1, etp_group=None)
    experts_module.Experts(config, ps, delay_wgrad_compute=True)

    assert all(
        kwargs["fuse_wgrad_accumulation"] for kwargs in grouped_linear_kwargs
    )


def test_dispatch_local_backward_accumulates_duplicate_token_rows_and_weights(
    monkeypatch,
    transformer_engine_import_stub,
):
    transformer_engine_import_stub()
    from megatron.lite.primitive.modules.moe_ep_chunk_overlap import (
        _dispatch_local_backward,
    )
    from megatron.lite.primitive.utils.cuda_allocator import (
        pop_workspace_shape_metrics,
    )

    monkeypatch.setenv("MEGATRON_LITE_CUDA_WORKSPACE_SHAPE_METRICS", "1")
    chunk = SimpleNamespace(
        idx=0,
        row_id_map=torch.tensor([0, 0, 2]),
        prob_flat_indices=torch.tensor([1, 3, 5]),
        recv_hidden_shape=torch.Size((3, 2)),
        recv_hidden_dtype=torch.float32,
        recv_probs_shape=torch.Size((3, 2)),
        recv_probs_dtype=torch.float32,
        recv_capacity_rows=4,
    )
    grad_dispatched = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    grad_probs = torch.tensor([0.25, 0.5, 0.75])

    grad_hidden, grad_recv_probs, leases = _dispatch_local_backward(
        chunk, grad_dispatched, grad_probs
    )

    torch.testing.assert_close(
        grad_hidden, torch.tensor([[4.0, 6.0], [0.0, 0.0], [5.0, 6.0]])
    )
    torch.testing.assert_close(
        grad_recv_probs, torch.tensor([[0.0, 0.25], [0.0, 0.5], [0.0, 0.75]])
    )
    metrics = pop_workspace_shape_metrics()
    prefix = "perf/workspace_ep_chunk_backward_scratch_0"
    assert metrics[f"{prefix}_capacity_rows_max"] == 4
    assert metrics[f"{prefix}_hidden_bytes_max"] == 3 * 2 * 4
    assert metrics[f"{prefix}_probs_bytes_max"] == 3 * 2 * 4
    for lease in leases:
        lease.release()


def test_dispatch_local_backward_materializes_zero_probability_gradient(
    transformer_engine_import_stub,
):
    transformer_engine_import_stub()
    from megatron.lite.primitive.modules.moe_ep_chunk_overlap import (
        _dispatch_local_backward,
    )

    chunk = SimpleNamespace(
        idx=0,
        row_id_map=torch.empty(0, dtype=torch.long),
        prob_flat_indices=torch.empty(0, dtype=torch.long),
        recv_hidden_shape=torch.Size((0, 4)),
        recv_hidden_dtype=torch.bfloat16,
        recv_probs_shape=torch.Size((0, 2)),
        recv_probs_dtype=torch.float32,
        recv_capacity_rows=1,
    )

    grad_hidden, grad_probs, leases = _dispatch_local_backward(
        chunk, torch.empty(0, 4, dtype=torch.bfloat16), None
    )

    assert grad_hidden.shape == (0, 4)
    assert grad_probs.shape == (0, 2)
    for lease in leases:
        lease.release()


def test_dispatch_local_backward_falls_back_when_recv_rows_exceed_fixed_capacity(
    transformer_engine_import_stub,
):
    transformer_engine_import_stub()
    from megatron.lite.primitive.modules.moe_ep_chunk_overlap import (
        _dispatch_local_backward,
    )

    chunk = SimpleNamespace(
        idx=0,
        row_id_map=torch.tensor([0, 1, 2]),
        prob_flat_indices=torch.tensor([0, 2, 4]),
        recv_hidden_shape=torch.Size((3, 2)),
        recv_hidden_dtype=torch.float32,
        recv_probs_shape=torch.Size((3, 2)),
        recv_probs_dtype=torch.float32,
        recv_capacity_rows=2,
    )

    grad_hidden, grad_probs, leases = _dispatch_local_backward(
        chunk,
        torch.ones(3, 2),
        torch.ones(3),
    )

    assert grad_hidden.shape == (3, 2)
    assert grad_probs.shape == (3, 2)
    assert leases == []


def test_accumulate_reuses_first_chunk_gradient_storage(
    transformer_engine_import_stub,
):
    transformer_engine_import_stub()
    from megatron.lite.primitive.modules.moe_ep_chunk_overlap import _accumulate

    parameter = torch.nn.Parameter(torch.zeros(2, 2))
    first = torch.ones(2, 2)
    second = torch.full((2, 2), 2.0)
    accum = [None]

    _accumulate(accum, (parameter,), (first,))
    storage = accum[0].data_ptr()
    _accumulate(accum, (parameter,), (second,))

    assert accum[0].data_ptr() == storage
    torch.testing.assert_close(accum[0], torch.full((2, 2), 3.0))


def test_delayed_wgrad_excludes_expert_params_from_autograd_targets(
    transformer_engine_import_stub,
):
    transformer_engine_import_stub()
    from megatron.lite.primitive.modules.moe_ep_chunk_overlap import (
        _expert_grad_inputs,
    )

    dispatched = torch.ones(2, 2, requires_grad=True)
    probs = torch.ones(2, requires_grad=True)
    params = tuple(torch.nn.Parameter(torch.zeros(1)) for _ in range(2))
    inputs = _expert_grad_inputs(dispatched, probs)

    assert inputs == (dispatched, probs)
    assert all(all(param is not item for item in inputs) for param in params)


def test_forward_trace_launches_current_expert_before_blocking_next_dispatch(
    monkeypatch, transformer_engine_import_stub
):
    transformer_engine_import_stub()
    from megatron.lite.primitive.modules.moe_ep_chunk_overlap import (
        EPChunkOverlapOperator,
    )

    trace = []

    class FakeEvent:
        def record(self, stream):
            trace.append(f"event.record.{stream.name}")

    class FakeStream:
        def __init__(self, name):
            self.name = name

        def wait_event(self, _event):
            trace.append(f"{self.name}.wait")

    class FakeDispatcher:
        def __init__(self, chunk):
            self.chunk = chunk
            self.ep_size = 3
            self._local_tpe_list = [2]

        def submit_deepep_dispatch(
            self, hidden, _scores, _indices, *, num_worst_tokens=0
        ):
            assert num_worst_tokens == 0
            trace.append(f"dispatch.submit.{self.chunk}")
            return {
                "hidden": hidden,
                "recv_hidden": None,
                "recv_indices": None,
                "recv_probs": None,
                "recv_per_expert": None,
            }

        def finish_deepep_dispatch(self, state):
            trace.append(f"dispatch.finish.{self.chunk}")
            return state["hidden"], torch.tensor([2]), torch.ones(2)

        def prepare_deepep_combine(self, expert_output):
            trace.append(f"combine.prepare.{self.chunk}")
            return expert_output, f"handle-{self.chunk}"

        def submit_deepep_combine_prepared(self, output, _handle):
            trace.append(f"combine.submit.{self.chunk}")
            return {"output": output}

        def finish_deepep_combine(self, state):
            trace.append(f"combine.finish.{self.chunk}")
            return state["output"]

    class FakeExperts:
        def __call__(self, hidden, *_args, **_kwargs):
            chunk = int(hidden[0, 0].item() // 2)
            trace.append(f"expert.{chunk}")
            return hidden + 1

    compute = FakeStream("compute")
    comm = FakeStream("comm")
    caller = FakeStream("caller")
    monkeypatch.setattr(torch.cuda, "Event", FakeEvent)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda _device=None: caller)
    monkeypatch.setattr(torch.cuda, "stream", lambda _stream: nullcontext())

    operator = object.__new__(EPChunkOverlapOperator)
    operator.router = lambda hidden: (
        torch.ones(hidden.size(0), 1),
        torch.zeros(hidden.size(0), 1, dtype=torch.long),
    )
    operator.experts = FakeExperts()
    operator._streams = lambda _device: (compute, comm)
    dispatchers = [FakeDispatcher(idx) for idx in range(3)]
    operator._forward_dispatcher = lambda idx: dispatchers[idx]

    inputs = torch.arange(6, dtype=torch.float32).view(6, 1)
    output = operator._forward_output_async(
        inputs,
        [(0, 2), (2, 4), (4, 6)],
        inputs.shape,
        inputs.dtype,
        recv_capacity_rows=0,
        disable_expert_act_recompute=False,
    )

    torch.testing.assert_close(output, inputs + 1)
    operations = [item for item in trace if not item.startswith(("event.", "caller."))]
    assert operations == [
        "comm.wait",
        "dispatch.submit.0",
        "dispatch.finish.0",
        "expert.0",
        "combine.prepare.0",
        "comm.wait",
        "dispatch.submit.1",
        "comm.wait",
        "combine.submit.0",
        "dispatch.finish.1",
        "expert.1",
        "combine.prepare.1",
        "comm.wait",
        "dispatch.submit.2",
        "comm.wait",
        "combine.submit.1",
        "dispatch.finish.2",
        "expert.2",
        "combine.prepare.2",
        "comm.wait",
        "combine.submit.2",
        "combine.finish.0",
        "combine.finish.1",
        "combine.finish.2",
    ]


def test_recv_capacity_uses_ep_wide_max_for_uneven_thd(
    monkeypatch, transformer_engine_import_stub
):
    transformer_engine_import_stub()
    from megatron.lite.primitive.modules import moe_ep_chunk_overlap as overlap

    monkeypatch.delenv("MEGATRON_LITE_EP_CHUNK_WEIGHTS", raising=False)
    calls = []

    def fake_all_reduce(value, *, op, group):
        calls.append((int(value.item()), op, group))
        value.fill_(11)

    monkeypatch.setattr(overlap.dist, "all_reduce", fake_all_reduce)
    operator = object.__new__(overlap.EPChunkOverlapOperator)
    operator.dispatcher = SimpleNamespace(
        ep_size=3,
        ps=SimpleNamespace(tp_size=1, tp_ep_group="ep-group"),
    )

    capacity = operator._recv_capacity_rows(torch.empty(5, 2), chunks=2)

    # Global max rows=11 splits into [6, 5], so every receive rank is bounded
    # by six rows from each of the three EP senders. local_rows * EP (=15)
    # would be unsafe for this uneven THD batch.
    assert capacity == 18
    assert calls == [(5, overlap.dist.ReduceOp.MAX, "ep-group")]


def test_core_contains_token_wise_forward_and_backward_deepep_pipeline():
    source_path = (
        Path(__file__).parents[3]
        / "megatron"
        / "lite"
        / "primitive"
        / "modules"
        / "moe_ep_chunk_overlap.py"
    )
    tree = ast.parse(source_path.read_text())
    calls = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }

    assert {
        "submit_deepep_dispatch",
        "finish_deepep_dispatch",
        "submit_deepep_combine_prepared",
        "finish_deepep_combine",
        "submit_deepep_combine_backward",
        "finish_deepep_combine_backward",
        "submit_deepep_dispatch_backward",
        "finish_deepep_dispatch_backward",
    } <= calls


def test_workspace_shape_metrics_report_and_reset_jitter(monkeypatch):
    from megatron.lite.primitive.utils.cuda_allocator import (
        pop_workspace_shape_metrics,
        record_workspace_shape,
    )

    monkeypatch.setenv("MEGATRON_LITE_CUDA_WORKSPACE_SHAPE_METRICS", "1")
    record_workspace_shape(
        device_index=0,
        scope="ep_chunk_forward",
        slot=0,
        dimensions={"expert_rows": 100},
    )
    record_workspace_shape(
        device_index=0,
        scope="ep_chunk_forward",
        slot=0,
        dimensions={"expert_rows": 140},
    )

    metrics = pop_workspace_shape_metrics()

    prefix = "perf/workspace_ep_chunk_forward_0"
    assert metrics[f"{prefix}_calls"] == 2
    assert metrics[f"{prefix}_expert_rows_min"] == 100
    assert metrics[f"{prefix}_expert_rows_max"] == 140
    assert metrics[f"{prefix}_expert_rows_span"] == 40
    assert metrics[f"{prefix}_expert_rows_unique"] == 2
    assert pop_workspace_shape_metrics() == {}


def test_cuda_allocator_metrics_expose_fragmentation(monkeypatch):
    from megatron.lite.primitive.utils.cuda_allocator import (
        _reset_fixed_capacity_scratch_for_tests,
        cuda_allocator_metrics,
        lease_fixed_capacity_scratch,
    )

    _reset_fixed_capacity_scratch_for_tests()
    _scratch, lease = lease_fixed_capacity_scratch(
        scope="grad_recv_hidden",
        shape=(2, 4),
        capacity_rows=8,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    stats = {
        "active_bytes.all.current": 3 * 1024**3,
        "inactive_split_bytes.all.current": 2 * 1024**3,
        "inactive_split_bytes.all.peak": 5 * 1024**3,
        "segment.all.current": 7,
        "active.all.current": 11,
        "inactive_split.all.current": 13,
        "num_alloc_retries": 17,
        "num_ooms": 19,
    }
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda: 4 * 1024**3)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda: 10 * 1024**3)
    monkeypatch.setattr(torch.cuda, "memory_stats", lambda: stats)

    metrics = cuda_allocator_metrics()

    assert metrics["perf/cuda_memory_allocated_gb"] == 4
    assert metrics["perf/cuda_memory_reserved_gb"] == 10
    assert metrics["perf/cuda_reserved_minus_allocated_gb"] == 6
    assert metrics["perf/cuda_inactive_split_bytes_gb"] == 2
    assert metrics["perf/cuda_inactive_split_peak_gb"] == 5
    assert metrics["perf/cuda_segment_count"] == 7
    assert metrics["perf/cuda_active_block_count"] == 11
    assert metrics["perf/cuda_inactive_split_block_count"] == 13
    assert metrics["perf/cuda_num_alloc_retries"] == 17
    assert metrics["perf/cuda_num_ooms"] == 19
    assert metrics["perf/scratch_bytes"] == 8 * 4 * 4
    assert metrics["perf/scratch_grad_recv_hidden_bytes"] == 8 * 4 * 4
    lease.release()
    _reset_fixed_capacity_scratch_for_tests()


def test_fixed_capacity_scratch_reuses_rows_and_bounds_slots():
    from megatron.lite.primitive.utils.cuda_allocator import (
        _reset_fixed_capacity_scratch_for_tests,
        fixed_capacity_scratch_metrics,
        lease_fixed_capacity_scratch,
    )

    _reset_fixed_capacity_scratch_for_tests()
    first, first_lease = lease_fixed_capacity_scratch(
        scope="grad_recv_hidden",
        shape=(3, 4),
        capacity_rows=8,
        dtype=torch.float32,
        device=torch.device("cpu"),
        max_slots=2,
    )
    second, second_lease = lease_fixed_capacity_scratch(
        scope="grad_recv_hidden",
        shape=(5, 4),
        capacity_rows=8,
        dtype=torch.float32,
        device=torch.device("cpu"),
        max_slots=2,
    )
    assert first.shape == (3, 4)
    assert second.shape == (5, 4)
    assert first.untyped_storage().data_ptr() != second.untyped_storage().data_ptr()

    with pytest.raises(RuntimeError, match="all 2 slots are in use"):
        lease_fixed_capacity_scratch(
            scope="grad_recv_hidden",
            shape=(2, 4),
            capacity_rows=8,
            dtype=torch.float32,
            device=torch.device("cpu"),
            max_slots=2,
        )

    first_lease.release()
    reused, reused_lease = lease_fixed_capacity_scratch(
        scope="grad_recv_hidden",
        shape=(7, 4),
        capacity_rows=8,
        dtype=torch.float32,
        device=torch.device("cpu"),
        max_slots=2,
    )
    assert reused.untyped_storage().data_ptr() == first.untyped_storage().data_ptr()
    metrics = fixed_capacity_scratch_metrics()
    assert metrics["perf/scratch_bytes"] == 2 * 8 * 4 * 4
    assert metrics["perf/scratch_slots"] == 2
    assert metrics["perf/scratch_slots_in_use"] == 2
    assert metrics["perf/scratch_grad_recv_hidden_bytes"] == 2 * 8 * 4 * 4

    reused_lease.release()
    second_lease.release()
    _reset_fixed_capacity_scratch_for_tests()


def test_fixed_capacity_scratch_capacity_changes_keep_one_bounded_bank():
    from megatron.lite.primitive.utils.cuda_allocator import (
        _reset_fixed_capacity_scratch_for_tests,
        fixed_capacity_scratch_metrics,
        lease_fixed_capacity_scratch,
    )

    _reset_fixed_capacity_scratch_for_tests()
    _small, small_lease = lease_fixed_capacity_scratch(
        scope="grad_recv_hidden",
        shape=(4, 4),
        capacity_rows=8,
        dtype=torch.float32,
        device=torch.device("cpu"),
        max_slots=1,
    )
    small_lease.release()
    grown, grown_lease = lease_fixed_capacity_scratch(
        scope="grad_recv_hidden",
        shape=(12, 4),
        capacity_rows=16,
        dtype=torch.float32,
        device=torch.device("cpu"),
        max_slots=1,
    )
    grown_lease.release()
    reused, reused_lease = lease_fixed_capacity_scratch(
        scope="grad_recv_hidden",
        shape=(6, 4),
        capacity_rows=10,
        dtype=torch.float32,
        device=torch.device("cpu"),
        max_slots=1,
    )

    assert reused.untyped_storage().data_ptr() == grown.untyped_storage().data_ptr()
    metrics = fixed_capacity_scratch_metrics()
    assert metrics["perf/scratch_bytes"] == 16 * 4 * 4
    assert metrics["perf/scratch_slots"] == 1

    reused_lease.release()
    _reset_fixed_capacity_scratch_for_tests()


def test_fixed_capacity_scratch_release_records_the_consumer_stream(monkeypatch):
    from megatron.lite.primitive.utils.cuda_allocator import (
        _FixedScratchLease,
        _FixedScratchSlot,
    )

    class FakeTensor:
        is_cuda = True

    class FakeStream:
        cuda_stream = 1234

    class FakeEvent:
        def __init__(self):
            self.recorded_stream = None

        def record(self, stream):
            self.recorded_stream = stream

    event = FakeEvent()
    monkeypatch.setattr(torch.cuda, "Event", lambda: event)
    slot = _FixedScratchSlot(tensor=FakeTensor(), in_use=True)
    stream = FakeStream()

    _FixedScratchLease(slot, torch.device("cuda", 0)).release(stream=stream)

    assert slot.in_use is False
    assert slot.event is event
    assert event.recorded_stream is stream
    assert slot.stream_key == stream.cuda_stream


def test_deepep_state_tensors_record_the_consumer_stream(monkeypatch):
    import megatron.lite.primitive.modules.moe_ep_chunk_overlap as overlap

    class FakeTensor:
        is_cuda = True
        device = torch.device("cuda", 0)

        def __init__(self):
            self.recorded_stream = None

        def record_stream(self, stream):
            self.recorded_stream = stream

    stream = object()
    outer = FakeTensor()
    nested = FakeTensor()
    monkeypatch.setattr(
        overlap.torch, "is_tensor", lambda value: isinstance(value, FakeTensor)
    )
    monkeypatch.setattr(overlap.torch.cuda, "current_stream", lambda _device: stream)

    overlap._record_state_tensors_current_stream(
        {"recv_hidden": outer, "metadata": {"recv_probs": nested}, "handle": object()}
    )

    assert outer.recorded_stream is stream
    assert nested.recorded_stream is stream


def test_fixed_capacity_scratch_fails_loud_on_capacity_overflow():
    from megatron.lite.primitive.utils.cuda_allocator import (
        _reset_fixed_capacity_scratch_for_tests,
        lease_fixed_capacity_scratch,
    )

    _reset_fixed_capacity_scratch_for_tests()
    with pytest.raises(ValueError, match="rows 9 exceed fixed capacity 8"):
        lease_fixed_capacity_scratch(
            scope="grad_recv_hidden",
            shape=(9, 4),
            capacity_rows=8,
            dtype=torch.float32,
            device=torch.device("cpu"),
            max_slots=2,
        )
