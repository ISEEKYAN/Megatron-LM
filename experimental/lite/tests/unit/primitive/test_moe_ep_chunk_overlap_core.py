# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


SOURCE = (
    Path(__file__).parents[3]
    / "megatron/lite/primitive/modules/moe_ep_chunk_overlap.py"
)


def test_three_ops_expose_bounded_nvtx_phase_ranges():
    source = SOURCE.read_text()

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


def test_delayed_expert_wgrads_request_distopt_main_grad_reuse(
    monkeypatch, transformer_engine_import_stub
):
    transformer_engine_import_stub()
    from megatron.lite.primitive.modules import experts as experts_module

    grouped_linear_kwargs = []

    class FakeGroupedLinear(torch.nn.Module):
        def __init__(self, num_gemms, *_args, bias, **kwargs):
            super().__init__()
            grouped_linear_kwargs.append(kwargs)
            self.use_bias = bias
            for idx in range(num_gemms):
                self.register_parameter(
                    f"weight{idx}", torch.nn.Parameter(torch.zeros(2, 2))
                )

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

    assert all(kwargs["delay_wgrad_compute"] for kwargs in grouped_linear_kwargs)
    assert all(kwargs["fuse_wgrad_accumulation"] for kwargs in grouped_linear_kwargs)


def test_experts_accept_host_splits_without_a_device_count_tensor(
    monkeypatch, transformer_engine_import_stub
):
    transformer_engine_import_stub()
    from megatron.lite.primitive.modules import experts as experts_module

    seen_splits = []

    class FakeGroupedLinear(torch.nn.Module):
        def __init__(self, _num_gemms, _in_features, out_features, **_kwargs):
            super().__init__()
            self.out_features = out_features

        def forward(self, x, splits):
            seen_splits.append(splits)
            return x.new_ones((x.size(0), self.out_features))

    monkeypatch.setattr(
        experts_module.te, "GroupedLinear", FakeGroupedLinear, raising=False
    )
    monkeypatch.setattr(
        experts_module, "bias_swiglu_impl", lambda x, bias=None: x[:, :2]
    )
    config = SimpleNamespace(
        num_experts=2,
        hidden_size=2,
        moe_intermediate_size=2,
        swiglu_limit=0.0,
    )
    ps = SimpleNamespace(ep_size=1, etp_size=1, tp_size=1, etp_group=None)
    experts = experts_module.Experts(config, ps)

    out = experts(
        torch.ones(3, 2),
        None,
        None,
        tokens_per_expert_list=[1, 2],
    )

    assert out.shape == (3, 2)
    assert seen_splits == [[1, 2], [1, 2]]


def test_dispatch_local_backward_accumulates_duplicate_rows_in_workspace(
    transformer_engine_import_stub,
):
    transformer_engine_import_stub()
    from megatron.lite.primitive.modules.moe_ep_chunk_overlap import (
        EPChunkShapeProfile,
        EPChunkWorkspaceKey,
        EPChunkWorkspaceRegistry,
        _dispatch_local_backward,
    )

    key = EPChunkWorkspaceKey(
        op="backward",
        device_type="cpu",
        device_index=None,
        ep_group_id=1,
        dtype=torch.float32,
        shape_profile=EPChunkShapeProfile(
            max_input_rows=3,
            hidden_size=2,
            topk=2,
            ep_size=2,
        ),
    )
    workspace = EPChunkWorkspaceRegistry().get_or_create(key, lambda slot: slot)
    lease = workspace.acquire(0)
    dispatched_base = torch.full((4, 2), 17.0)
    recv_probs_base = torch.full((3, 2), 19.0)
    chunk = SimpleNamespace(
        idx=0,
        workspace_lease=lease,
        recv_probs_base=recv_probs_base,
        row_id_map=torch.tensor([0, 0, 2]),
        prob_flat_indices=torch.tensor([1, 3, 5]),
        recv_hidden_shape=torch.Size((3, 2)),
        recv_hidden_dtype=torch.float32,
        recv_probs_shape=torch.Size((3, 2)),
        recv_probs_dtype=torch.float32,
    )

    grad_hidden, grad_probs = _dispatch_local_backward(
        chunk,
        torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
        torch.tensor([0.25, 0.5, 0.75]),
        hidden_reuse_base=dispatched_base,
    )

    assert grad_hidden.data_ptr() == dispatched_base.data_ptr()
    assert grad_probs.data_ptr() == recv_probs_base.data_ptr()
    assert workspace.metrics()["runtime_allocations"] == 0

    torch.testing.assert_close(
        grad_hidden, torch.tensor([[4.0, 6.0], [0.0, 0.0], [5.0, 6.0]])
    )
    torch.testing.assert_close(
        grad_probs,
        torch.tensor([[0.0, 0.25], [0.0, 0.5], [0.0, 0.75]]),
    )


def test_dispatch_local_backward_clears_reused_scratch_before_sparse_writes(
    transformer_engine_import_stub,
):
    transformer_engine_import_stub()
    from megatron.lite.primitive.modules.moe_ep_chunk_overlap import (
        _dispatch_local_backward,
    )

    expert_out_base = torch.full((4, 2), 17.0)
    recv_probs_base = torch.full((3, 2), 19.0)

    chunk = SimpleNamespace(
        recv_probs_base=recv_probs_base,
        row_id_map=torch.tensor([0, 2]),
        prob_flat_indices=torch.tensor([1, 5]),
        recv_hidden_shape=torch.Size((3, 2)),
        recv_hidden_dtype=torch.float32,
        recv_probs_shape=torch.Size((3, 2)),
        recv_probs_dtype=torch.float32,
    )

    grad_hidden, grad_probs = _dispatch_local_backward(
        chunk,
        torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        torch.tensor([0.25, 0.75]),
        hidden_reuse_base=expert_out_base,
    )
    torch.testing.assert_close(
        grad_hidden, torch.tensor([[1.0, 2.0], [0.0, 0.0], [3.0, 4.0]])
    )
    torch.testing.assert_close(
        grad_probs,
        torch.tensor([[0.0, 0.25], [0.0, 0.0], [0.0, 0.75]]),
    )

    expert_out_base.fill_(23.0)
    recv_probs_base.fill_(29.0)
    _grad_hidden, grad_probs_none = _dispatch_local_backward(
        chunk,
        torch.tensor([[5.0, 6.0], [7.0, 8.0]]),
        None,
        hidden_reuse_base=expert_out_base,
    )
    torch.testing.assert_close(grad_probs_none, torch.zeros_like(grad_probs_none))


def test_dispatch_local_backward_fails_loud_for_invalid_hidden_reuse_storage(
    transformer_engine_import_stub,
):
    transformer_engine_import_stub()
    from megatron.lite.primitive.modules.moe_ep_chunk_overlap import (
        _dispatch_local_backward,
    )

    chunk = SimpleNamespace(
        recv_probs_base=torch.empty(3, 2),
        row_id_map=torch.tensor([0, 2]),
        prob_flat_indices=torch.tensor([1, 5]),
        recv_hidden_shape=torch.Size((3, 2)),
        recv_hidden_dtype=torch.float32,
        recv_probs_shape=torch.Size((3, 2)),
        recv_probs_dtype=torch.float32,
    )
    grad_dispatched = torch.ones(2, 2)
    invalid_bases = (
        torch.empty(2, 2),
        torch.empty(2, 4).t(),
        torch.empty(3, 2, dtype=torch.float64),
    )

    for hidden_reuse_base in invalid_bases:
        with pytest.raises(RuntimeError, match="hidden reuse storage"):
            _dispatch_local_backward(
                chunk,
                grad_dispatched,
                None,
                hidden_reuse_base=hidden_reuse_base,
            )


def test_accumulate_reuses_first_chunk_gradient_storage(
    transformer_engine_import_stub,
):
    transformer_engine_import_stub()
    from megatron.lite.primitive.modules.moe_ep_chunk_overlap import _accumulate

    parameter = torch.nn.Parameter(torch.zeros(2, 2))
    accum = [None]
    _accumulate(accum, (parameter,), (torch.ones(2, 2),))
    storage = accum[0].data_ptr()
    _accumulate(accum, (parameter,), (torch.full((2, 2), 2.0),))

    assert accum[0].data_ptr() == storage
    torch.testing.assert_close(accum[0], torch.full((2, 2), 3.0))


def test_core_contains_forward_backward_deepep_pipeline():
    tree = ast.parse(SOURCE.read_text())
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


def test_all_chunked_async_deepep_allocations_are_owned_by_comm_stream():
    tree = ast.parse(SOURCE.read_text())
    submit_names = {
        "submit_deepep_dispatch",
        "submit_deepep_combine_prepared",
        "submit_deepep_combine_backward",
        "submit_deepep_dispatch_backward",
    }
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in submit_names
    ]

    assert calls
    for call in calls:
        keywords = {keyword.arg: keyword.value for keyword in call.keywords}
        assert isinstance(keywords.get("allocate_on_comm_stream"), ast.Constant)
        assert keywords["allocate_on_comm_stream"].value is True
        if call.func.attr in {
            "submit_deepep_dispatch",
            "submit_deepep_combine_prepared",
        }:
            assert isinstance(keywords.get("async_finish"), ast.Constant)
            assert keywords["async_finish"].value is True


def test_only_deepep_recv_dispatch_uses_the_persistent_slot_mem_pool():
    tree = ast.parse(SOURCE.read_text())
    submit_names = {
        "submit_deepep_dispatch",
        "submit_deepep_combine_prepared",
        "submit_deepep_combine_backward",
        "submit_deepep_dispatch_backward",
    }
    submissions = []

    def visit(node, in_pool=False):
        if isinstance(node, ast.With):
            in_pool = in_pool or any(
                isinstance(item.context_expr, ast.Call)
                and isinstance(item.context_expr.func, ast.Attribute)
                and item.context_expr.func.attr == "deepep_recv_allocation"
                for item in node.items
            )
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in submit_names
        ):
            submissions.append((node.func.attr, in_pool))
        for child in ast.iter_child_nodes(node):
            visit(child, in_pool)

    visit(tree)
    assert submissions
    assert all(
        in_pool for name, in_pool in submissions if name == "submit_deepep_dispatch"
    )
    assert all(
        not in_pool for name, in_pool in submissions if name != "submit_deepep_dispatch"
    )


def test_production_path_has_no_global_cuda_sync_or_allocator_fallback():
    source = SOURCE.read_text()
    tree = ast.parse(source)
    cuda_calls = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Attribute)
        and isinstance(node.func.value.value, ast.Name)
        and node.func.value.value.id == "torch"
        and node.func.value.attr == "cuda"
    }

    assert "synchronize" not in cuda_calls
    assert "empty_cache" not in cuda_calls
    assert "oversize_fallback" not in source
    assert "in_use_fallback" not in source
    assert "slot_grow" not in source


def test_deepep_state_tensors_record_the_consumer_stream(
    monkeypatch, transformer_engine_import_stub
):
    transformer_engine_import_stub()
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


def test_chunked_forward_enqueues_experts_before_lifetime_bookkeeping():
    tree = ast.parse(SOURCE.read_text())
    forward_names = {"_forward_output_async", "_forward_saved_context_async"}

    for function in (
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name in forward_names
    ):
        expert_calls = [
            node
            for node in ast.walk(function)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "experts"
        ]
        record_calls = [
            node
            for node in ast.walk(function)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_record_state_tensors_current_stream"
        ]
        assert len(expert_calls) == len(record_calls) == 1
        assert expert_calls[0].lineno < record_calls[0].lineno


def test_recv_telemetry_is_cold_unless_explicitly_enabled(
    monkeypatch, transformer_engine_import_stub
):
    transformer_engine_import_stub()
    import megatron.lite.primitive.modules.moe_ep_chunk_overlap as overlap

    overlap._EP_CHUNK_RECV_ACTIVE.clear()
    overlap._EP_CHUNK_RECV_STATS.clear()
    monkeypatch.delenv("MEGATRON_LITE_EP_CHUNK_SCRATCH_TRACE", raising=False)

    overlap._record_ep_chunk_recv_tensors(
        action="acquire",
        phase="forward",
        workspace="forward",
        chunk_idx=0,
        recv_hidden=torch.zeros(2, 3),
        recv_probs=torch.zeros(2, 1),
    )

    assert overlap._EP_CHUNK_RECV_ACTIVE == {}
    assert overlap._EP_CHUNK_RECV_STATS == {}


def test_chunked_forward_submits_next_dispatch_before_expert_host_setup():
    tree = ast.parse(SOURCE.read_text())
    forward_names = {"_forward_output_async", "_forward_saved_context_async"}

    for function in (
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name in forward_names
    ):
        nested_names = {
            node.name for node in function.body if isinstance(node, ast.FunctionDef)
        }
        assert {"finish_dispatch", "run_expert"} <= nested_names
        loop = next(node for node in ast.walk(function) if isinstance(node, ast.For))
        calls = {
            node.func.id: node.lineno
            for node in ast.walk(loop)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        assert calls["finish_dispatch"] < calls["submit_dispatch"] < calls["run_expert"]


def test_saved_forward_routes_next_chunk_on_comm_stream():
    tree = ast.parse(SOURCE.read_text())
    function = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "_forward_saved_context_async"
    )
    submit = next(
        node
        for node in function.body
        if isinstance(node, ast.FunctionDef) and node.name == "submit_dispatch"
    )
    stream_contexts = [
        item.context_expr.args[0].id
        for node in ast.walk(submit)
        if isinstance(node, ast.With)
        for item in node.items
        if isinstance(item.context_expr, ast.Call)
        and isinstance(item.context_expr.func, ast.Attribute)
        and item.context_expr.func.attr == "stream"
        and isinstance(item.context_expr.args[0], ast.Name)
    ]

    assert stream_contexts == ["comm_stream"]


def test_saved_backward_chains_first_combine_to_caller_grad_readiness(
    transformer_engine_import_stub,
):
    transformer_engine_import_stub()
    from megatron.lite.primitive.modules.moe_ep_chunk_overlap import (
        _EPChunkOperationBase,
    )

    import inspect

    source = inspect.getsource(_EPChunkOperationBase._saved_context_backward)

    assert "caller_stream = torch.cuda.current_stream(grad_2d.device)" in source
    assert "grad_ready = torch.cuda.Event()" in source
    assert "grad_ready.record(caller_stream)" in source
    assert "last_deepep_event: Any | None = grad_ready" in source
    assert source.index("grad_ready.record(caller_stream)") < source.index(
        "submit_deepep_combine_backward"
    )
    assert "torch.cuda.synchronize" not in source


def test_saved_backward_uses_its_own_manual_unpermute_arena_after_recv_event(
    transformer_engine_import_stub,
):
    import inspect

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

    assert "allocation_arena" not in fields
    assert "forward_workspace_lease" not in fields
    assert "recv_consumed_event" in fields
    assert "allocation_arena=lease.allocation_arena" not in forward_source
    assert "recv_consumed_event.record(compute_stream)" in forward_source
    assert forward_source.index("recv_consumed_event.record(compute_stream)") < (
        forward_source.index("state.clear()")
    )
    assert "allocation_arena=saved.allocation_arena" not in backward_source
    assert "forward_workspace_lease=lease" not in forward_source
    assert "lease.release(consumed)" in forward_source
    assert "forward_workspace_lease" not in backward_source
    assert "compute_stream.wait_event(saved.recv_consumed_event)" in backward_source
    assert backward_source.index(
        "compute_stream.wait_event(saved.recv_consumed_event)"
    ) < backward_source.index("_dispatch_local_backward(")
    assert "cuda.synchronize" not in backward_source


def test_saved_context_wrapper_keeps_scratch_until_explicit_lifecycle_reset(
    transformer_engine_import_stub,
):
    import inspect

    transformer_engine_import_stub()
    from megatron.lite.primitive.modules.moe_ep_chunk_overlap import (
        _SavedContextEPChunkFunction,
    )

    source = inspect.getsource(_SavedContextEPChunkFunction.backward)

    assert "ctx.backward_op.backward(" in source
    assert "ctx.backward_op.workspace.reset_tensors(" not in source
    assert "cuda.synchronize" not in source


def test_saved_backward_reuses_manual_unpermute_storage_only_after_wgrad_flush(
    transformer_engine_import_stub,
):
    import inspect

    transformer_engine_import_stub()
    from megatron.lite.primitive.modules.moe_ep_chunk_overlap import (
        _EPChunkOperationBase,
    )

    source = inspect.getsource(_EPChunkOperationBase._saved_context_backward)
    grad_complete = source.index("expert_grads = torch.autograd.grad(")
    flushed = source.index("flush_delayed_weight_grads")
    flush_waited = source.index("compute_stream.wait_event(wgrad_done)")
    alias_taken = source.index("hidden_reuse_base = local_state.pop(", flush_waited)
    storage_overwritten = source.index("_dispatch_local_backward(", alias_taken)
    dispatch_submitted = source.index(
        "submit_deepep_dispatch_backward(", storage_overwritten
    )
    queued = source.index("pending_dispatch_bwd.append", grad_complete)

    assert grad_complete < queued < flushed < flush_waited < alias_taken
    assert alias_taken < storage_overwritten < dispatch_submitted
    assert "hidden_reuse_base = chunk.dispatched.detach()" not in source
    assert "Delayed grouped-linear wgrad retains its grad-output storage" in source
    for assignment in (
        "saved.dispatched = None",
        "saved.probs = None",
        "saved.expert_out = None",
        "saved.expert_out_edge = None",
        "chunk.dispatched = None",
        "chunk.probs = None",
        "chunk.expert_out = None",
        "chunk.expert_out_edge = None",
    ):
        assert assignment in source
    assert "saved.scores = None" not in source
    assert "saved.x = None" not in source
    assert "cuda.synchronize" not in source


def test_normal_and_fused_backward_reuse_manual_unpermute_workspace_storage(
    transformer_engine_import_stub,
):
    import inspect

    transformer_engine_import_stub()
    from megatron.lite.primitive.modules.moe_ep_chunk_overlap import (
        _BackwardChunk,
        _EPChunkOperationBase,
        _ForwardChunkContext,
        _dispatch_local_backward,
        _manual_unpermute_backward,
    )

    assert "recv_probs_base" in _BackwardChunk.__dataclass_fields__
    assert "recv_probs_base" in _ForwardChunkContext.__dataclass_fields__
    assert "recv_hidden_base" not in _BackwardChunk.__dataclass_fields__
    assert "recv_hidden_base" not in _ForwardChunkContext.__dataclass_fields__
    saved_source = inspect.getsource(_EPChunkOperationBase._forward_saved_context_async)
    normal_backward_source = inspect.getsource(
        _EPChunkOperationBase._saved_context_backward
    )
    fused_source = inspect.getsource(
        _EPChunkOperationBase._full_recompute_fused_backward_v6
    )
    helper_source = inspect.getsource(_dispatch_local_backward)
    unpermute_source = inspect.getsource(_manual_unpermute_backward)

    for source in (saved_source, fused_source):
        assert 'recv_hidden_base=state["recv_hidden"]' not in source
        assert 'recv_probs_base=state["recv_probs"]' in source
    for source in (normal_backward_source, fused_source):
        assert "hidden_reuse_base = chunk.dispatched.detach()" not in source
    assert "hidden_reuse_base = local_state.pop(" in normal_backward_source
    assert (
        '"grad_expert_out"'
        in normal_backward_source[normal_backward_source.index("hidden_reuse_base =") :]
    )
    assert "out=chunk.expert_out.detach()" in fused_source
    assert (
        'hidden_reuse_base = local_state.pop("grad_expert_out").detach()'
        in fused_source
    )
    assert "chunk.workspace_lease.tensor(" in unpermute_source
    assert '"grad_expert_out"' in unpermute_source
    assert "torch.index_select(" in unpermute_source
    assert "out=grad_expert_out" in unpermute_source
    assert "chunk.workspace_lease.tensor(" not in helper_source
    assert "hidden_reuse_base.detach()" in helper_source
    assert ".view(-1)[:required_hidden_numel]" in helper_source
    assert "grad_recv_probs = chunk.recv_probs_base" in helper_source
    assert "cuda.synchronize" not in helper_source


def test_fused_reuses_arena_owned_expert_output_before_delayed_wgrad_flush(
    transformer_engine_import_stub,
):
    import inspect

    transformer_engine_import_stub()
    from megatron.lite.primitive.modules.moe_ep_chunk_overlap import (
        _EPChunkOperationBase,
    )

    source = inspect.getsource(_EPChunkOperationBase._full_recompute_fused_backward_v6)
    autograd = source.index("expert_grads = torch.autograd.grad(")
    flushed = source.index("flush_delayed_weight_grads", autograd)
    direct_unpermute = source.index("out=chunk.expert_out.detach()")
    alias_taken = source.index(
        'hidden_reuse_base = local_state.pop("grad_expert_out").detach()', autograd
    )
    graph_clear = source.index("chunk.expert_out = None", alias_taken)
    locals_clear = source.index("del expert_dispatched", graph_clear)
    overwrite = source.index("_dispatch_local_backward(", locals_clear)
    dispatch = source.index("submit_deepep_dispatch_backward(", overwrite)

    assert direct_unpermute < autograd < alias_taken
    assert alias_taken < graph_clear < locals_clear < overwrite < dispatch
    assert dispatch < flushed
    assert "del expert_input, expert_probs, metadata" in source
    assert "state.clear()" in source
    assert "del expert_dispatched, expert_probs_input" in source
    assert "del expert_inputs, expert_grads, expert_output" in source
    assert "cuda.synchronize" not in source


def test_forward_and_fused_expert_allocations_use_their_workspace_slot_arenas(
    transformer_engine_import_stub,
):
    import inspect

    transformer_engine_import_stub()
    from megatron.lite.primitive.modules.moe_ep_chunk_overlap import (
        _EPChunkOperationBase,
    )

    forward_source = inspect.getsource(_EPChunkOperationBase._forward_output_async)
    saved_forward_source = inspect.getsource(
        _EPChunkOperationBase._forward_saved_context_async
    )
    fused_source = inspect.getsource(
        _EPChunkOperationBase._full_recompute_fused_backward_v6
    )

    forward_arena = forward_source.index("with lease.allocation_arena.allocate():")
    forward_expert = forward_source.index("expert_out = self.experts(", forward_arena)
    saved_arena = saved_forward_source.index("with lease.allocation_arena.allocate():")
    saved_expert = saved_forward_source.index("expert_out = self.experts(", saved_arena)
    fused_arena = fused_source.index(
        "with workspace_lease.allocation_arena.allocate():"
    )
    fused_expert = fused_source.index("expert_out = self.experts(", fused_arena)

    assert forward_arena < forward_expert
    assert saved_arena < saved_expert
    assert fused_arena < fused_expert
    no_grad_finish = forward_source.index("finish_deepep_combine(state)")
    no_grad_release = forward_source.index("lease.release(consumed)", no_grad_finish)
    assert no_grad_finish < no_grad_release
    assert "cuda.synchronize" not in forward_source
    assert "cuda.synchronize" not in saved_forward_source
    assert "cuda.synchronize" not in fused_source


def test_manual_unpermute_writes_into_stable_workspace_slot(
    transformer_engine_import_stub,
):
    transformer_engine_import_stub()
    from megatron.lite.primitive.modules.moe_ep_chunk_overlap import (
        _manual_unpermute_backward,
    )

    class StableLease:
        def __init__(self):
            self.storage = None
            self.names = []

        def tensor(self, name, shape, *, dtype, device):
            self.names.append(name)
            if self.storage is None:
                self.storage = torch.empty(shape, dtype=dtype, device=device)
            return self.storage

    lease = StableLease()
    chunk = SimpleNamespace(
        expert_out_shape=torch.Size((4, 2)),
        expert_out_dtype=torch.float32,
        row_id_map=torch.tensor([2, 0, 3, 1]),
        workspace_lease=lease,
    )
    first_input = torch.arange(8, dtype=torch.float32).view(4, 2)
    first = _manual_unpermute_backward(chunk, first_input)
    first_ptr = first.data_ptr()
    second_input = first_input + 10
    second = _manual_unpermute_backward(chunk, second_input)

    assert first_ptr == second.data_ptr()
    assert lease.names == ["grad_expert_out", "grad_expert_out"]
    torch.testing.assert_close(second, second_input.index_select(0, chunk.row_id_map))


def test_fused_manual_unpermute_reuses_expert_output_without_workspace_scratch(
    transformer_engine_import_stub,
):
    transformer_engine_import_stub()
    from megatron.lite.primitive.modules.moe_ep_chunk_overlap import (
        _manual_unpermute_backward,
    )

    class NoScratchLease:
        def tensor(self, *args, **kwargs):
            raise AssertionError(
                "fused manual unpermute must not allocate workspace scratch"
            )

    expert_out = torch.full((4, 2), -1.0)
    chunk = SimpleNamespace(
        expert_out_shape=torch.Size((4, 2)),
        expert_out_dtype=torch.float32,
        row_id_map=torch.tensor([2, 0, 3, 1]),
        workspace_lease=NoScratchLease(),
    )
    grad_rank_grouped = torch.arange(8, dtype=torch.float32).view(4, 2)

    grad_expert_out = _manual_unpermute_backward(
        chunk,
        grad_rank_grouped,
        out=expert_out.detach(),
    )

    assert grad_expert_out.data_ptr() == expert_out.data_ptr()
    torch.testing.assert_close(
        grad_expert_out,
        grad_rank_grouped.index_select(0, chunk.row_id_map),
    )
