# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

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
    chunk = SimpleNamespace(
        idx=0,
        workspace_lease=lease,
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
    )

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

    scratch = {
        "grad_recv_hidden": torch.full((3, 2), 17.0),
        "grad_recv_probs": torch.full((3, 2), 19.0),
    }

    class ReusedLease:
        @staticmethod
        def tensor(name, _shape, *, dtype, device):
            return scratch[name].to(dtype=dtype, device=device)

    chunk = SimpleNamespace(
        workspace_lease=ReusedLease(),
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
    )
    torch.testing.assert_close(
        grad_hidden, torch.tensor([[1.0, 2.0], [0.0, 0.0], [3.0, 4.0]])
    )
    torch.testing.assert_close(
        grad_probs,
        torch.tensor([[0.0, 0.25], [0.0, 0.0], [0.0, 0.75]]),
    )

    scratch["grad_recv_hidden"].fill_(23.0)
    scratch["grad_recv_probs"].fill_(29.0)
    _grad_hidden, grad_probs_none = _dispatch_local_backward(
        chunk,
        torch.tensor([[5.0, 6.0], [7.0, 8.0]]),
        None,
    )
    torch.testing.assert_close(grad_probs_none, torch.zeros_like(grad_probs_none))


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
