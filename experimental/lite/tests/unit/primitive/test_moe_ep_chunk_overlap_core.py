# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import ast
import inspect
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


SOURCE = (
    Path(__file__).parents[3]
    / "megatron/lite/primitive/modules/moe_ep_chunk_overlap.py"
)


def test_finished_deepep_dispatch_shape_contract_executes_with_fake_dispatcher(
    transformer_engine_import_stub,
):
    transformer_engine_import_stub()
    from megatron.lite.primitive.modules.moe_ep_chunk_overlap import (
        EPChunkShapeProfile,
        _validate_finished_deepep_dispatch,
    )

    class FakeDispatcher:
        @staticmethod
        def finish(*, recv_rows, expert_rows):
            state = {
                "recv_hidden": torch.zeros(recv_rows, 4),
                "recv_probs": torch.zeros(recv_rows, 2),
            }
            return state, torch.zeros(expert_rows, 4)

    profile = EPChunkShapeProfile(
        max_input_rows=4,
        hidden_size=4,
        topk=2,
        ep_size=2,
    )
    dispatcher = FakeDispatcher()
    state, dispatched = dispatcher.finish(recv_rows=4, expert_rows=8)
    _validate_finished_deepep_dispatch(profile, state, dispatched)

    state, dispatched = dispatcher.finish(recv_rows=5, expert_rows=8)
    with pytest.raises(RuntimeError, match="recv rows"):
        _validate_finished_deepep_dispatch(profile, state, dispatched)

    state, dispatched = dispatcher.finish(recv_rows=4, expert_rows=9)
    with pytest.raises(RuntimeError, match="expert rows"):
        _validate_finished_deepep_dispatch(profile, state, dispatched)


def test_fused_pending_retirement_releases_dispatch_lease_before_next_expert(
    monkeypatch, transformer_engine_import_stub
):
    """CPU contract for dispatch-lease ordering, not physical allocator reuse."""
    transformer_engine_import_stub()
    from megatron.lite.primitive.modules import moe_ep_chunk_overlap as overlap

    events = []

    class FakeEvent:
        def record(self, _stream):
            events.append("lease-release-event")

    class FakeDispatcher:
        def finish_deepep_dispatch_backward(self, _state):
            events.append("retire-dispatch-bwd")
            return torch.ones(1, 1), torch.ones(1, 1)

    class FakeLease:
        def release(self, _event):
            events.append("lease-release")

    monkeypatch.setattr(overlap.torch.cuda, "stream", lambda _stream: nullcontext())
    monkeypatch.setattr(overlap.torch.cuda, "Event", FakeEvent)
    x = torch.ones(1, 1, requires_grad=True)
    scores = x * 2
    chunk = overlap._BackwardChunk(
        idx=0,
        start=0,
        end=1,
        x=x,
        scores=scores,
        handle=None,
        row_id_map=torch.zeros(1, dtype=torch.long),
        prob_flat_indices=torch.zeros(1, dtype=torch.long),
        recv_hidden_shape=torch.Size((1, 1)),
        recv_hidden_dtype=torch.float32,
        recv_probs_shape=torch.Size((1, 1)),
        recv_probs_dtype=torch.float32,
        recv_probs_base=None,
        dispatched=None,
        probs=None,
        expert_out=None,
        dispatcher=FakeDispatcher(),
        workspace_lease=FakeLease(),
        scores_shape=scores.shape,
        scores_dtype=scores.dtype,
    )
    pending = [(chunk, {"dispatch_bwd_state": object()})]
    grad_x_chunks = [None]

    events.append("prefetch-next")
    overlap._retire_one_fused_dispatch_bwd(
        pending,
        compute_stream=object(),
        grad_2d=torch.ones(1, 1),
        router_params=(),
        grad_x_chunks=grad_x_chunks,
        router_accum=[],
    )
    events.append("expert-start-next")

    assert pending == []
    torch.testing.assert_close(grad_x_chunks[0], torch.full((1, 1), 3.0))
    assert events == [
        "prefetch-next",
        "retire-dispatch-bwd",
        "lease-release-event",
        "lease-release",
        "expert-start-next",
    ]


@pytest.mark.parametrize(
    "state,dispatched,error",
    [
        (
            {"recv_hidden": torch.zeros(4), "recv_probs": torch.zeros(4, 2)},
            torch.zeros(8, 4),
            "recv_hidden must be a rank-2 tensor",
        ),
        (
            {"recv_hidden": torch.zeros(4, 5), "recv_probs": torch.zeros(4, 2)},
            torch.zeros(8, 4),
            "recv hidden size 5.*profile 4",
        ),
        (
            {"recv_hidden": torch.zeros(4, 4), "recv_probs": torch.zeros(4)},
            torch.zeros(8, 4),
            "recv_probs must be a rank-2 tensor",
        ),
        (
            {"recv_hidden": torch.zeros(4, 4), "recv_probs": torch.zeros(3, 2)},
            torch.zeros(8, 4),
            "recv_hidden and recv_probs rows must match",
        ),
        (
            {"recv_hidden": torch.zeros(4, 4), "recv_probs": torch.zeros(4, 3)},
            torch.zeros(8, 4),
            "recv_probs top-k 3.*profile 2",
        ),
        (
            {"recv_hidden": torch.zeros(4, 4), "recv_probs": torch.zeros(4, 2)},
            torch.zeros(8, 5),
            "dispatched expert input must be rank-2",
        ),
    ],
)
def test_finished_deepep_dispatch_rejects_shape_contract_mismatch(
    state, dispatched, error, transformer_engine_import_stub
):
    transformer_engine_import_stub()
    from megatron.lite.primitive.modules.moe_ep_chunk_overlap import (
        EPChunkShapeProfile,
        _validate_finished_deepep_dispatch,
    )

    profile = EPChunkShapeProfile(max_input_rows=4, hidden_size=4, topk=2, ep_size=2)
    with pytest.raises(RuntimeError, match=error):
        _validate_finished_deepep_dispatch(profile, state, dispatched)


def test_all_three_dispatch_finish_paths_validate_before_expert_or_arena(
    transformer_engine_import_stub,
):
    import inspect

    transformer_engine_import_stub()
    from megatron.lite.primitive.modules.moe_ep_chunk_overlap import (
        _EPChunkOperationBase,
    )

    for method in (
        _EPChunkOperationBase._forward_output_async,
        _EPChunkOperationBase._forward_saved_context_async,
        _EPChunkOperationBase._full_recompute_fused_backward_v6,
    ):
        source = inspect.getsource(method)
        validated = source.index("_validate_finished_deepep_dispatch")
        expert = source.index("self.experts(", validated)
        assert validated < expert


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
    for linear in (experts.fc1, experts.fc2):
        linear.wgrad_store._mlite_immediate_wgrad_contexts = 1
    experts.flush_delayed_weight_grads(num_contexts=1)
    assert all(
        linear.wgrad_store._mlite_immediate_wgrad_contexts == 0
        for linear in (experts.fc1, experts.fc2)
    )

    for linear in (experts.fc1, experts.fc2):
        linear.wgrad_store._mlite_immediate_wgrad_contexts = 2
    with pytest.raises(RuntimeError, match="immediate/deferred context accounting"):
        experts.flush_delayed_weight_grads(num_contexts=1)

    for linear in (experts.fc1, experts.fc2):
        linear.wgrad_store._mlite_immediate_wgrad_contexts = 1
        linear.wgrad_store.pending = 1
    with pytest.raises(RuntimeError, match="queue was not drained"):
        experts.flush_delayed_weight_grads(num_contexts=1)


def test_expert_wgrad_flush_records_nested_cuda_views_and_bases(
    monkeypatch, transformer_engine_import_stub
):
    transformer_engine_import_stub()
    from megatron.lite.primitive.modules import experts as experts_module

    original_is_tensor = experts_module.torch.is_tensor
    stream = object()
    recorded = []

    class FakeCudaTensor:
        is_cuda = True

        def __init__(self, name, *, base=None):
            self.name = name
            self._base = base

        def record_stream(self, actual_stream):
            recorded.append((self.name, actual_stream))

    class FakeStore:
        def __init__(self, linear, prefix):
            self.linear = linear
            self.pending = 1
            self.context = SimpleNamespace(empty=lambda: self.pending == 0)
            self.base = FakeCudaTensor(f"{prefix}.base")
            self.view = FakeCudaTensor(f"{prefix}.view", base=self.base)

        @staticmethod
        def delay_wgrad_compute():
            return True

        def pop(self):
            self.pending -= 1
            grads = [
                getattr(self.linear, f"weight{idx}").main_grad
                for idx in range(self.linear.num_gemms)
            ]
            result = ({"result": (self.view,)}, None, None)
            tensors = [{"input": self.view}, [self.view], grads]
            return result, tensors

    class FakeGroupedLinear(torch.nn.Module):
        next_idx = 0

        def __init__(self, num_gemms, *_args, bias, **_kwargs):
            super().__init__()
            self.num_gemms = num_gemms
            self.use_bias = bias
            prefix = f"linear{FakeGroupedLinear.next_idx}"
            FakeGroupedLinear.next_idx += 1
            for idx in range(num_gemms):
                param = torch.nn.Parameter(torch.zeros(2, 2))
                param.main_grad = torch.ones(2, 2)
                self.register_parameter(f"weight{idx}", param)
            self.wgrad_store = FakeStore(self, prefix)

    monkeypatch.setattr(
        experts_module.te, "GroupedLinear", FakeGroupedLinear, raising=False
    )
    monkeypatch.setattr(
        experts_module.torch,
        "is_tensor",
        lambda value: isinstance(value, FakeCudaTensor) or original_is_tensor(value),
    )
    experts = experts_module.Experts(
        SimpleNamespace(
            num_experts=2,
            hidden_size=2,
            moe_intermediate_size=2,
            swiglu_limit=0.0,
        ),
        SimpleNamespace(ep_size=1, etp_size=1, tp_size=1, etp_group=None),
        delay_wgrad_compute=True,
    )

    experts.flush_delayed_weight_grads(num_contexts=1, stream=stream)

    assert recorded == [
        ("linear0.view", stream),
        ("linear0.base", stream),
        ("linear1.view", stream),
        ("linear1.base", stream),
    ]
    assert experts.fc1.wgrad_store.context.empty()
    assert experts.fc2.wgrad_store.context.empty()


def test_delayed_expert_wgrads_request_fused_sink_reuse(
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


def _build_fake_delayed_grad_experts(monkeypatch, experts_module):
    class FakeStore:
        def __init__(self, linear):
            self.linear = linear
            self.pending = []
            self.context = SimpleNamespace(empty=lambda: not self.pending)

        @staticmethod
        def delay_wgrad_compute():
            return True

        def queue(self, value):
            sinks = [
                getattr(self.linear, f"weight{idx}").main_grad
                for idx in range(self.linear.num_gemms)
            ]
            self.pending.append((float(value), sinks))

        def pop(self):
            value, sinks = self.pending.pop(0)
            for sink in sinks:
                sink.add_(value)
            return (None, [None] * self.linear.num_gemms, None), [None, None, sinks]

    class FakeGroupedLinear(torch.nn.Module):
        def __init__(
            self,
            num_gemms,
            *_args,
            bias,
            fuse_wgrad_accumulation,
            **_kwargs,
        ):
            super().__init__()
            self.num_gemms = num_gemms
            self.use_bias = bias
            self.fuse_wgrad_accumulation = fuse_wgrad_accumulation
            self.output_size = _args[1]
            for idx in range(num_gemms):
                self.register_parameter(
                    f"weight{idx}", torch.nn.Parameter(torch.zeros(2, 2))
                )
            self.wgrad_store = FakeStore(self)

        def forward(self, x, _m_splits):
            return x.new_zeros((*x.shape[:-1], self.output_size))

    monkeypatch.setattr(
        experts_module.te, "GroupedLinear", FakeGroupedLinear, raising=False
    )
    return experts_module.Experts(
        SimpleNamespace(
            num_experts=2,
            hidden_size=2,
            moe_intermediate_size=2,
            swiglu_limit=0.0,
        ),
        SimpleNamespace(ep_size=1, etp_size=1, tp_size=1, etp_group=None),
        delay_wgrad_compute=True,
    )


@pytest.mark.parametrize("flush_schedule", ["normal", "full_fused"])
def test_delayed_expert_standard_grads_match_native_accumulation_without_main_grad(
    flush_schedule, monkeypatch, transformer_engine_import_stub
):
    transformer_engine_import_stub()
    from megatron.lite.primitive.modules import experts as experts_module

    experts = _build_fake_delayed_grad_experts(monkeypatch, experts_module)
    params = tuple(experts.fc1.parameters()) + tuple(experts.fc2.parameters())
    assert all(
        param.grad is None and not hasattr(param, "main_grad") for param in params
    )
    assert experts._owned_main_grad_aliases == {}
    native_params = [torch.nn.Parameter(torch.zeros_like(param)) for param in params]

    grad_storage = None
    for contribution in (1.25, 2.75):
        sum((param * contribution).sum() for param in native_params).backward()
        experts._prepare_delayed_weight_grad_sinks()
        if grad_storage is None:
            grad_storage = {id(param): param.grad for param in params}
        else:
            assert all(param.grad is grad_storage[id(param)] for param in params)
        for linear in (experts.fc1, experts.fc2):
            linear.wgrad_store.queue(contribution)
        if flush_schedule == "full_fused":
            experts.flush_delayed_weight_grads(num_contexts=1)
    if flush_schedule == "normal":
        experts.flush_delayed_weight_grads(num_contexts=2)

    for param, native_param in zip(params, native_params, strict=True):
        torch.testing.assert_close(param.grad, native_param.grad)
        assert not hasattr(param, "main_grad")
    assert experts._owned_main_grad_aliases == {}


def test_delayed_expert_forward_keeps_gradient_sinks_lazy_until_backward(
    monkeypatch, transformer_engine_import_stub
):
    transformer_engine_import_stub()
    from megatron.lite.primitive.modules import experts as experts_module

    experts = _build_fake_delayed_grad_experts(monkeypatch, experts_module)
    monkeypatch.setattr(
        experts_module,
        "swiglu_with_probs",
        lambda value, _probs, _limit: value[..., :2],
    )
    params = tuple(experts.fc1.parameters()) + tuple(experts.fc2.parameters())
    x = torch.zeros(3, 2)

    with torch.no_grad():
        experts(x, None, tokens_per_expert_list=[1, 2])
    assert all(
        param.grad is None and not hasattr(param, "main_grad") for param in params
    )

    with torch.enable_grad():
        experts(x.requires_grad_(), None, tokens_per_expert_list=[1, 2])
    assert all(
        param.grad is None and not hasattr(param, "main_grad") for param in params
    )

    experts._prepare_delayed_weight_grad_sinks()
    assert all(param.main_grad is param.grad for param in params)
    for linear in (experts.fc1, experts.fc2):
        linear.wgrad_store.queue(2.0)
    experts.flush_delayed_weight_grads(num_contexts=1)
    assert all(torch.equal(param.grad, torch.full_like(param, 2.0)) for param in params)
    assert all(not hasattr(param, "main_grad") for param in params)


def test_chunked_ep_backward_phases_prepare_sinks_before_expert_autograd(
    transformer_engine_import_stub,
):
    import inspect

    transformer_engine_import_stub()
    from megatron.lite.primitive.modules.moe_ep_chunk_overlap import (
        _EPChunkOperationBase,
    )

    for method in (
        _EPChunkOperationBase._full_recompute_fused_backward_v6,
        _EPChunkOperationBase._saved_context_backward,
    ):
        source = inspect.getsource(method)
        assert source.index("_prepare_delayed_weight_grad_sinks") < source.index(
            "expert_grads = torch.autograd.grad"
        )


def test_delayed_expert_external_main_grad_stays_zero_copy_without_parameter_grad(
    monkeypatch, transformer_engine_import_stub
):
    transformer_engine_import_stub()
    from megatron.lite.primitive.modules import experts as experts_module

    experts = _build_fake_delayed_grad_experts(monkeypatch, experts_module)
    params = tuple(experts.fc1.parameters()) + tuple(experts.fc2.parameters())
    external = {}
    for param in params:
        external[id(param)] = torch.zeros_like(param)
        param.main_grad = external[id(param)]

    experts._prepare_delayed_weight_grad_sinks()
    for linear in (experts.fc1, experts.fc2):
        linear.wgrad_store.queue(3.5)
    experts.flush_delayed_weight_grads(num_contexts=1)

    for param in params:
        assert param.grad is None
        assert param.main_grad is external[id(param)]
        torch.testing.assert_close(param.main_grad, torch.full_like(param, 3.5))
    assert experts._owned_main_grad_aliases == {}


def test_delayed_expert_fsdp_accessor_is_resolved_by_te_at_pop(
    monkeypatch, transformer_engine_import_stub
):
    transformer_engine_import_stub()
    from megatron.lite.primitive.modules import experts as experts_module

    experts = _build_fake_delayed_grad_experts(monkeypatch, experts_module)
    params = tuple(experts.fc1.parameters()) + tuple(experts.fc2.parameters())
    external = {id(param): torch.zeros_like(param) for param in params}
    accessor_calls = []
    for param in params:
        param.__fsdp_param__ = True

        def get_main_grad(p=param):
            accessor_calls.append(id(p))
            return external[id(p)]

        param.get_main_grad = get_main_grad

    experts._prepare_delayed_weight_grad_sinks()
    assert accessor_calls == []
    assert all(not hasattr(param, "main_grad") for param in params)

    # Frozen TE resolves the saved accessor during delayed backward and writes the
    # result back to origin_weight.main_grad before WeightGradStore.pop returns.
    for linear in (experts.fc1, experts.fc2):
        for idx in range(linear.num_gemms):
            param = getattr(linear, f"weight{idx}")
            param.main_grad = param.get_main_grad()
        linear.wgrad_store.queue(4.5)
    experts.flush_delayed_weight_grads(num_contexts=1)

    assert sorted(accessor_calls) == sorted(id(param) for param in params)
    for param in params:
        assert param.grad is None
        assert param.main_grad is external[id(param)]
        torch.testing.assert_close(param.main_grad, torch.full_like(param, 4.5))


def test_delayed_expert_fsdp_accessor_sink_is_validated_after_pop(
    monkeypatch, transformer_engine_import_stub
):
    transformer_engine_import_stub()
    from megatron.lite.primitive.modules import experts as experts_module

    experts = _build_fake_delayed_grad_experts(monkeypatch, experts_module)
    params = tuple(experts.fc1.parameters()) + tuple(experts.fc2.parameters())
    for param in params:
        param.__fsdp_param__ = True
        param.get_main_grad = lambda: torch.zeros(1)

    experts._prepare_delayed_weight_grad_sinks()
    for linear in (experts.fc1, experts.fc2):
        for idx in range(linear.num_gemms):
            param = getattr(linear, f"weight{idx}")
            param.main_grad = param.get_main_grad()
        linear.wgrad_store.queue(1.0)

    with pytest.raises(RuntimeError, match="matching shape and device"):
        experts.flush_delayed_weight_grads(num_contexts=1)


def test_callable_get_main_grad_without_fsdp_capability_uses_standard_grad(
    monkeypatch, transformer_engine_import_stub
):
    transformer_engine_import_stub()
    from megatron.lite.primitive.modules import experts as experts_module

    experts = _build_fake_delayed_grad_experts(monkeypatch, experts_module)
    params = tuple(experts.fc1.parameters()) + tuple(experts.fc2.parameters())
    calls = []
    for param in params:
        param.get_main_grad = lambda: calls.append(True)

    experts._prepare_delayed_weight_grad_sinks()

    assert calls == []
    assert all(param.main_grad is param.grad for param in params)


def test_owned_standard_grad_alias_obeys_zero_grad_and_explicit_release(
    monkeypatch, transformer_engine_import_stub
):
    transformer_engine_import_stub()
    from megatron.lite.primitive.modules import experts as experts_module

    experts = _build_fake_delayed_grad_experts(monkeypatch, experts_module)
    params = tuple(experts.fc1.parameters()) + tuple(experts.fc2.parameters())

    experts._prepare_delayed_weight_grad_sinks()
    first_grads = {id(param): param.grad for param in params}
    assert all(param.main_grad is param.grad for param in params)
    experts.release_delayed_weight_grad_aliases()
    assert all(param.grad is first_grads[id(param)] for param in params)
    assert all(not hasattr(param, "main_grad") for param in params)

    experts.zero_grad(set_to_none=False)
    experts._prepare_delayed_weight_grad_sinks()
    assert all(param.grad is first_grads[id(param)] for param in params)
    for linear in (experts.fc1, experts.fc2):
        linear.wgrad_store.queue(2.0)
    experts.flush_delayed_weight_grads(num_contexts=1)
    assert all(torch.equal(param.grad, torch.full_like(param, 2.0)) for param in params)

    experts.zero_grad(set_to_none=True)
    assert all(param.grad is None for param in params)
    experts._prepare_delayed_weight_grad_sinks()
    assert all(param.grad is not first_grads[id(param)] for param in params)
    assert all(param.main_grad is param.grad for param in params)


def test_chunked_ep_public_ops_have_no_optimizer_or_recompute_policy_knowledge(
    transformer_engine_import_stub,
):
    import inspect

    transformer_engine_import_stub()
    from megatron.lite.primitive.modules.moe_ep_chunk_overlap import (
        EPChunkBackwardOp,
        EPChunkForwardOp,
        EPChunkFusedForwardBackwardOp,
    )

    for op in (EPChunkForwardOp, EPChunkBackwardOp, EPChunkFusedForwardBackwardOp):
        public_contract = str(inspect.signature(op.__init__))
        for name, member in inspect.getmembers(op, predicate=inspect.isfunction):
            if not name.startswith("_"):
                public_contract += str(inspect.signature(member))
        assert "optimizer" not in public_contract
        assert "recompute_modules" not in public_contract
        assert "ep_chunk_full_recompute" not in public_contract


def test_pending_combine_guards_are_fail_loud_under_python_optimized_mode():
    source = SOURCE.read_text()

    assert "assert pending_combine is not None" not in source
    assert source.count("EP chunk combine pipeline produced no pending output") == 2


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


@pytest.mark.parametrize("chunk_count", [3, 4])
@pytest.mark.parametrize("saved_context", [False, True])
def test_logical_chunk_slot_reuse_finishes_combine_before_dispatch_reacquire(
    chunk_count, saved_context, monkeypatch, transformer_engine_import_stub
):
    """Exercise both forward pipelines with two strictly leased physical slots."""
    import gc
    import weakref

    transformer_engine_import_stub()
    from megatron.lite.primitive.modules import moe_ep_chunk_overlap as overlap

    events = []

    class FakeEvent:
        def record(self, _stream):
            pass

    class FakeStream:
        device = torch.device("cpu")

        def wait_event(self, _event):
            pass

    class FakeDispatcher:
        moe_permute_fusion = False

        def submit_deepep_dispatch(self, x, _scores, _indices, **_kwargs):
            return {
                "chunk_idx": int(x[0, 0]),
                "recv_hidden": x.clone(),
                "recv_probs": torch.ones(x.size(0), 1),
                "handle": object(),
            }

        def finish_deepep_dispatch(self, state, **_kwargs):
            gc.collect()
            chunk_idx = state["chunk_idx"] // 2
            events.append(f"finish:{chunk_idx}")
            dispatched = state["recv_hidden"].clone()
            weakref.finalize(
                dispatched, events.append, f"retire_dispatched:{chunk_idx}"
            )
            return (
                dispatched,
                [state["recv_hidden"].size(0)],
                state["recv_probs"],
            )

        def finish_deepep_dispatch_external_with_options(self, state, **_kwargs):
            return (
                state["recv_hidden"],
                [state["recv_hidden"].size(0)],
                state["recv_probs"],
                {
                    "local_tpe_list": [state["recv_hidden"].size(0)],
                    "manual_row_id_map": torch.arange(state["recv_hidden"].size(0)),
                    "manual_prob_flat_indices": torch.arange(
                        state["recv_hidden"].size(0)
                    ),
                },
            )

        def prepare_deepep_combine(self, expert_out):
            return expert_out, object()

        def submit_deepep_combine_prepared(self, rank_grouped, handle, **_kwargs):
            return rank_grouped, handle

        def finish_deepep_combine(self, state):
            return state[0]

    class FakeLease:
        def __init__(self, workspace, slot):
            self.workspace = workspace
            self.slot = slot
            self.dispatcher = FakeDispatcher()
            self.allocation_arena = SimpleNamespace(allocate=lambda: nullcontext())

        def deepep_recv_allocation(self):
            return nullcontext()

        def release(self, _event):
            events.append(f"release:{self.slot}")
            self.workspace.leased.remove(self.slot)

    class FakeActivationLease:
        def tensor(self, _name, shape, *, dtype, device):
            return torch.empty(shape, dtype=dtype, device=device)

        def allocate(self):
            return nullcontext()

        def release(self, _event):
            pass

    class FakeWorkspace:
        def __init__(self):
            self.key = SimpleNamespace(
                op="forward",
                shape_profile=SimpleNamespace(
                    chunk_count=chunk_count, validate_input=lambda _value: None
                ),
            )
            self.leased = set()

        def acquire(self, slot, **_kwargs):
            if slot in self.leased:
                raise RuntimeError(f"slot {slot} already leased")
            self.leased.add(slot)
            events.append(f"acquire:{slot}")
            return FakeLease(self, slot)

        def acquire_expert_activation(self, **_kwargs):
            return FakeActivationLease()

    compute_stream = FakeStream()
    workspace = FakeWorkspace()
    operation = overlap._EPChunkOperationBase(
        router=lambda x: (x * 2, torch.zeros(x.size(0), 1, dtype=torch.long)),
        experts=lambda x, *_args, **_kwargs: x + 1,
        workspace=workspace,
    )
    monkeypatch.setattr(
        operation, "_streams", lambda _device: (compute_stream, FakeStream())
    )
    monkeypatch.setattr(overlap.torch.cuda, "Event", FakeEvent)
    monkeypatch.setattr(overlap.torch.cuda, "stream", lambda _stream: nullcontext())
    monkeypatch.setattr(
        overlap.torch.cuda, "current_stream", lambda _device=None: compute_stream
    )
    monkeypatch.setattr(
        overlap, "_validate_finished_deepep_dispatch", lambda *_args: None
    )
    monkeypatch.setattr(
        overlap, "_record_state_tensors_current_stream", lambda _state: None
    )
    monkeypatch.setattr(
        overlap, "_record_ep_chunk_recv_tensors", lambda **_kwargs: None
    )
    monkeypatch.setattr(
        overlap, "unpermute", lambda expert_out, *_args, **_kwargs: expert_out
    )

    x = (
        torch.arange(chunk_count * 2, dtype=torch.float32)
        .view(-1, 1)
        .requires_grad_(True)
    )
    ranges = [(2 * index, 2 * index + 2) for index in range(chunk_count)]
    if saved_context:
        output, context = operation._forward_saved_context_async(
            x, ranges, x.shape, x.dtype
        )
        assert len(context.chunks) == chunk_count
    else:
        output = operation._forward_output_async(x, ranges, x.shape, x.dtype)

    assert output.shape == x.shape
    assert workspace.leased == set()
    if not saved_context:
        assert events.index("retire_dispatched:0") < events.index("finish:1"), events
    for slot in range(2):
        acquires = [
            index for index, event in enumerate(events) if event == f"acquire:{slot}"
        ]
        assert len(acquires) == (chunk_count + 1 - slot) // 2
        for reused in acquires[1:]:
            assert any(event == f"release:{slot}" for event in events[:reused]), events


def test_two_chunk_forward_keeps_next_dispatch_ahead_of_first_combine_release(
    transformer_engine_import_stub,
):
    """The n=2 schedule retains its existing dispatch/expert/combine overlap."""
    # The n=3/4 executable harness above would be redundant here; this source
    # order pins the n=2 fast path, where no physical slot is reused in-loop.
    transformer_engine_import_stub()
    from megatron.lite.primitive.modules.moe_ep_chunk_overlap import (
        _EPChunkOperationBase,
    )

    for method in (
        _EPChunkOperationBase._forward_output_async,
        _EPChunkOperationBase._forward_saved_context_async,
    ):
        source = inspect.getsource(method)
        first_dispatch = source.index("current_state = submit_dispatch(0)")
        next_dispatch = source.index("current_state = submit_dispatch(loop_idx + 1)")
        first_combine = source.index("finish_combine(pending_combine)", next_dispatch)
        assert first_dispatch < next_dispatch < first_combine


def test_saved_forward_routes_on_compute_then_hands_off_dispatch_to_comm_stream():
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

    assert stream_contexts == ["compute_stream", "comm_stream"]
    source = ast.unparse(submit)
    assert source.index("scores, indices = self._route") < source.index(
        "route_ready.record(compute_stream)"
    )
    assert source.index("route_ready.record(compute_stream)") < source.index(
        "comm_stream.wait_event(route_ready)"
    ) < source.index("dispatcher.submit_deepep_dispatch")


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


def test_fused_backward_keeps_activation_backing_until_explicit_reset(
    transformer_engine_import_stub,
):
    transformer_engine_import_stub()
    from megatron.lite.primitive.modules.moe_ep_chunk_overlap import (
        EPChunkFusedForwardBackwardOp,
        _EPChunkOperationBase,
    )

    fused_source = inspect.getsource(EPChunkFusedForwardBackwardOp.forward_backward)
    backward_source = inspect.getsource(
        _EPChunkOperationBase._full_recompute_fused_backward_v6
    )

    assert "torch.cuda.current_stream(grad_2d.device).wait_event(done)" in backward_source
    assert backward_source.index("wait_event(done)") < backward_source.index(
        "router_grads_out = _materialize"
    )
    assert fused_source.index("_full_recompute_fused_backward(") < fused_source.index(
        "grad_x = grad_x.view_as(x_saved)"
    )
    assert "self.workspace.park_expert_activations(" not in fused_source
    assert "reset_tensors" not in fused_source
    assert "synchronize" not in fused_source


def test_fused_backward_does_not_implicitly_park_between_steps(
    transformer_engine_import_stub,
):
    """The fused OP must leave reuse to explicit workspace lifecycle calls."""
    transformer_engine_import_stub()
    from megatron.lite.primitive.modules.moe_ep_chunk_overlap import (
        EPChunkFusedForwardBackwardOp,
    )

    class Workspace:
        def __init__(self):
            self.park_calls = 0

        def park_expert_activations(self, **_kwargs):
            self.park_calls += 1

    workspace = Workspace()
    op = EPChunkFusedForwardBackwardOp(
        router=torch.nn.Identity(),
        experts=object(),
        workspace=workspace,
    )
    op._full_recompute_fused_backward = lambda x, grad: (grad.clone(), [], [])
    x = torch.ones(2, 3)
    grad = torch.full_like(x, 2)

    for _ in range(2):
        grad_x, router_grads, expert_grads = op.forward_backward(x, grad)
        torch.testing.assert_close(grad_x, grad)
        assert router_grads == expert_grads == []
    assert workspace.park_calls == 0


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
    alias_saved = source.index('local_state["hidden_reuse_base"] = dispatched.detach()')
    alias_taken = source.index(
        'hidden_reuse_base = local_state.pop("hidden_reuse_base")', flush_waited
    )
    storage_overwritten = source.index("_dispatch_local_backward(", alias_taken)
    dispatch_submitted = source.index(
        "submit_deepep_dispatch_backward(", storage_overwritten
    )
    queued = source.index("pending_dispatch_bwd.append", grad_complete)

    assert grad_complete < alias_saved < queued < flushed < flush_waited < alias_taken
    assert alias_taken < storage_overwritten < dispatch_submitted
    assert 'local_state["hidden_reuse_base"] = dispatched.detach()' in source
    assert "Delayed grouped-linear Wgrad retains FC1 input" in source
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
    assert (
        'local_state["hidden_reuse_base"] = dispatched.detach()'
        in normal_backward_source
    )
    assert (
        'hidden_reuse_base = local_state.pop("hidden_reuse_base")'
        in normal_backward_source
    )
    assert "out=chunk.expert_out.detach()" in fused_source
    assert "hidden_reuse_base = expert_dispatched.detach()" in fused_source
    assert "chunk.workspace_lease.tensor(" in unpermute_source
    assert '"grad_expert_out"' in unpermute_source
    assert "torch.index_select(" in unpermute_source
    assert "out=grad_expert_out" in unpermute_source
    assert "chunk.workspace_lease.tensor(" not in helper_source
    assert "hidden_reuse_base.detach()" in helper_source
    assert ".view(-1)[:required_hidden_numel]" in helper_source
    assert "grad_recv_probs = chunk.recv_probs_base" in helper_source
    assert "cuda.synchronize" not in helper_source


def test_fused_flushes_each_chunk_before_reusing_delayed_wgrad_storage(
    transformer_engine_import_stub,
):
    import inspect

    transformer_engine_import_stub()
    from megatron.lite.primitive.modules.moe_ep_chunk_overlap import (
        _EPChunkOperationBase,
    )

    source = inspect.getsource(_EPChunkOperationBase._full_recompute_fused_backward_v6)
    autograd = source.index("expert_grads = torch.autograd.grad(")
    direct_unpermute = source.index("out=chunk.expert_out.detach()")
    next_dispatch = source.index(
        "next_state = submit_recompute_dispatch", direct_unpermute
    )
    dgrad_ready = source.index("dgrad_ready.record(compute_stream)", autograd)
    per_chunk_flush = source.index("flush_delayed_weight_grads", dgrad_ready)
    activation_release = source.index(
        "expert_activation_lease.release(local_bwd_ready)", per_chunk_flush
    )
    alias_taken = source.index(
        "hidden_reuse_base = expert_dispatched.detach()", autograd
    )
    graph_clear = source.index("chunk.expert_out = None", alias_taken)
    locals_clear = source.index("del expert_dispatched", graph_clear)
    overwrite = source.index("_dispatch_local_backward(", per_chunk_flush)
    local_ready = source.index("local_bwd_ready.record(wgrad_stream)", overwrite)
    dispatch = source.index("submit_deepep_dispatch_backward(", overwrite)
    pending_append = source.index("pending_dispatch_bwd.append", dispatch)

    assert direct_unpermute < next_dispatch < autograd < alias_taken
    assert alias_taken < graph_clear < locals_clear < dgrad_ready < per_chunk_flush
    assert per_chunk_flush < overwrite < local_ready < activation_release
    assert activation_release < dispatch < pending_append
    assert "num_contexts=1" in source[per_chunk_flush:overwrite]
    assert "stream=wgrad_stream" in source[per_chunk_flush:overwrite]
    activation_backward = source.rindex(
        "with expert_activation_lease.allocate():", 0, autograd
    )
    assert activation_backward < autograd < dgrad_ready
    assert "num_contexts=len(pending_dispatch_bwd)" not in source
    assert "compute_stream.wait_event(wgrad_done)" not in source
    assert "pending_local_bwd" not in source
    assert "_queue_backward_stream_wait(last_wgrad_done" in source
    assert "del expert_input, expert_probs, metadata" in source
    assert "state.clear()" in source
    assert "del expert_dispatched, expert_probs_input" in source
    assert "del expert_inputs, expert_grads, expert_output" in source
    for name in (
        "grad_dispatched",
        "grad_probs",
        "hidden_reuse_base",
        "chunk.recv_probs_base",
        "chunk.row_id_map",
        "chunk.prob_flat_indices",
    ):
        assert name in source[per_chunk_flush:overwrite]
    assert "cuda.synchronize" not in source


def test_dispatch_local_backward_rejects_source_destination_storage_alias(
    transformer_engine_import_stub,
):
    """The FC1-input reuse slot must not alias autograd's FC1 dgrad source."""
    transformer_engine_import_stub()
    from megatron.lite.primitive.modules.moe_ep_chunk_overlap import (
        _dispatch_local_backward,
    )

    shared = torch.empty(3, 2)
    chunk = SimpleNamespace(
        recv_probs_base=torch.empty(3, 2),
        row_id_map=torch.tensor([0, 2]),
        prob_flat_indices=torch.tensor([1, 5]),
        recv_hidden_shape=torch.Size((3, 2)),
        recv_hidden_dtype=torch.float32,
        recv_probs_shape=torch.Size((3, 2)),
        recv_probs_dtype=torch.float32,
    )

    with pytest.raises(RuntimeError, match="source and destination storage overlap"):
        _dispatch_local_backward(
            chunk,
            shared[:2],
            None,
            hidden_reuse_base=shared,
        )


def test_dispatch_local_backward_rejects_partial_byte_range_overlap(
    transformer_engine_import_stub,
):
    transformer_engine_import_stub()
    from megatron.lite.primitive.modules.moe_ep_chunk_overlap import (
        _dispatch_local_backward,
    )

    shared = torch.empty(4, 2)
    chunk = SimpleNamespace(
        recv_probs_base=torch.empty(3, 2),
        row_id_map=torch.tensor([0, 2]),
        prob_flat_indices=torch.tensor([1, 5]),
        recv_hidden_shape=torch.Size((3, 2)),
        recv_hidden_dtype=torch.float32,
        recv_probs_shape=torch.Size((3, 2)),
        recv_probs_dtype=torch.float32,
    )

    with pytest.raises(RuntimeError, match="source and destination storage overlap"):
        _dispatch_local_backward(
            chunk,
            shared[:2],
            None,
            hidden_reuse_base=shared[1:],
        )


def test_dispatch_local_backward_accepts_adjacent_nonoverlapping_ranges(
    transformer_engine_import_stub,
):
    transformer_engine_import_stub()
    from megatron.lite.primitive.modules.moe_ep_chunk_overlap import (
        _dispatch_local_backward,
    )

    shared = torch.empty(5, 2)
    grad_dispatched = shared[:2]
    grad_dispatched.copy_(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    hidden_reuse_base = shared[2:]
    chunk = SimpleNamespace(
        recv_probs_base=torch.empty(3, 2),
        row_id_map=torch.tensor([0, 2]),
        prob_flat_indices=torch.tensor([1, 5]),
        recv_hidden_shape=torch.Size((3, 2)),
        recv_hidden_dtype=torch.float32,
        recv_probs_shape=torch.Size((3, 2)),
        recv_probs_dtype=torch.float32,
    )

    grad_hidden, _ = _dispatch_local_backward(
        chunk, grad_dispatched, None, hidden_reuse_base=hidden_reuse_base
    )

    assert grad_hidden.data_ptr() != grad_dispatched.data_ptr()
    torch.testing.assert_close(
        grad_hidden, torch.tensor([[1.0, 2.0], [0.0, 0.0], [3.0, 4.0]])
    )


def test_dispatch_local_backward_rejects_noncontiguous_gradient_source(
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
    noncontiguous_source = torch.empty(2, 3)[:, :2]

    assert not noncontiguous_source.is_contiguous()
    with pytest.raises(RuntimeError, match="gradient source must be contiguous"):
        _dispatch_local_backward(
            chunk,
            noncontiguous_source,
            None,
            hidden_reuse_base=torch.empty(3, 2),
        )


def test_autograd_flush_then_local_scatter_reuses_distinct_fc1_input_storage(
    transformer_engine_import_stub,
):
    """CPU model of the fused FC1-dgrad/FC2-output coloring boundary."""
    transformer_engine_import_stub()
    from megatron.lite.primitive.modules.moe_ep_chunk_overlap import (
        _dispatch_local_backward,
    )

    events = []

    class CallerOwnedDgrad(torch.autograd.Function):
        @staticmethod
        def forward(ctx, value, dgrad_out):
            ctx.dgrad_out = dgrad_out
            return value * 2

        @staticmethod
        def backward(ctx, grad_output):
            events.append("autograd writes fc1_dgrad")
            ctx.dgrad_out.copy_(grad_output * 3)
            return ctx.dgrad_out, None

    # This is the physical FC2-output/FC1-dgrad color from the production
    # arena. FC1 input deliberately has a different physical address.
    fc2_output_and_fc1_dgrad = torch.empty(3, 2)
    expert_dispatched = torch.tensor([[2.0, 3.0], [5.0, 7.0]], requires_grad=True)
    expert_output = CallerOwnedDgrad.apply(
        expert_dispatched, fc2_output_and_fc1_dgrad[:2]
    )
    grad_dispatched = torch.autograd.grad(
        expert_output,
        expert_dispatched,
        torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
    )[0]
    hidden_reuse_base = expert_dispatched.detach()

    assert grad_dispatched.data_ptr() == fc2_output_and_fc1_dgrad.data_ptr()
    assert grad_dispatched.data_ptr() != hidden_reuse_base.data_ptr()

    class DelayedWgradQueue:
        def flush(self):
            events.append("flush delayed wgrad")

    # The delayed FC1 Wgrad still reads expert_dispatched, so this real queue
    # flush is the final operation before zero/scatter overwrites that storage.
    delayed_wgrad = DelayedWgradQueue()
    delayed_wgrad.flush()
    chunk = SimpleNamespace(
        recv_probs_base=torch.empty(2, 1),
        row_id_map=torch.tensor([1, 0]),
        prob_flat_indices=torch.tensor([0, 1]),
        recv_hidden_shape=torch.Size((2, 2)),
        recv_hidden_dtype=torch.float32,
        recv_probs_shape=torch.Size((2, 1)),
        recv_probs_dtype=torch.float32,
    )

    def dispatch_after_flush():
        events.append("zero/scatter fc1_input")
        return _dispatch_local_backward(
            chunk,
            grad_dispatched,
            None,
            hidden_reuse_base=hidden_reuse_base,
        )

    grad_hidden, _ = dispatch_after_flush()

    assert events == [
        "autograd writes fc1_dgrad",
        "flush delayed wgrad",
        "zero/scatter fc1_input",
    ]
    assert grad_hidden.data_ptr() == expert_dispatched.data_ptr()
    torch.testing.assert_close(grad_hidden, torch.tensor([[9.0, 12.0], [3.0, 6.0]]))


def test_fused_autograd_context_is_only_source_level_pool_routing_evidence(
    transformer_engine_import_stub,
):
    import inspect

    transformer_engine_import_stub()
    from megatron.lite.primitive.modules.moe_ep_chunk_overlap import (
        _EPChunkOperationBase,
    )

    source = inspect.getsource(_EPChunkOperationBase._full_recompute_fused_backward_v6)
    tree = ast.parse(inspect.cleandoc(source))
    activation_scopes = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.With)
        and any(
            isinstance(item.context_expr, ast.Call)
            and isinstance(item.context_expr.func, ast.Attribute)
            and isinstance(item.context_expr.func.value, ast.Name)
            and item.context_expr.func.value.id == "expert_activation_lease"
            and item.context_expr.func.attr == "allocate"
            for item in node.items
        )
    ]
    assert any(
        any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Attribute)
            and isinstance(node.func.value.value, ast.Name)
            and node.func.value.value.id == "torch"
            and node.func.value.attr == "autograd"
            and node.func.attr == "grad"
            for node in ast.walk(scope)
        )
        for scope in activation_scopes
    )
    # use_mem_pool is current-thread scoped; GPU allocator history must prove
    # whether autograd engine worker allocations actually use this pool.


def test_saved_backward_owns_one_activation_lease_until_all_local_backward_reads(
    transformer_engine_import_stub,
):
    import inspect

    transformer_engine_import_stub()
    from megatron.lite.primitive.modules.moe_ep_chunk_overlap import (
        _EPChunkOperationBase,
    )

    source = inspect.getsource(_EPChunkOperationBase._saved_context_backward)
    acquire = source.index("acquire_expert_activation(")
    chunk_loop = source.index("for saved in reversed(context.chunks):")
    activation_scope = source.index(
        "with expert_activation_lease.allocate():", chunk_loop
    )
    autograd = source.index("expert_grads = torch.autograd.grad(", activation_scope)
    pending = source.index("pending_dispatch_bwd.append", autograd)
    local_backward = source.index("_dispatch_local_backward(", pending)
    activation_done = source.index(
        "backward_activation_done.record(compute_stream)", local_backward
    )
    release = source.index(
        "expert_activation_lease.release(backward_activation_done)", activation_done
    )
    finish_dispatch = source.index("finish_deepep_dispatch_backward(", release)

    assert acquire < chunk_loop < activation_scope < autograd < pending
    assert pending < local_backward < activation_done < release < finish_dispatch
    assert "cuda.synchronize" not in source


def test_forward_and_fused_split_expert_activation_from_slot_output_arenas(
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

    forward_acquire = forward_source.index("acquire_expert_activation(")
    forward_expert = forward_source.index("expert_out = self.experts(", forward_acquire)
    forward_release = forward_source.index(
        "expert_activation_lease.release(expert_ready)", forward_expert
    )
    saved_arena = saved_forward_source.index("with lease.allocation_arena.allocate():")
    saved_expert = saved_forward_source.index("expert_out = self.experts(", saved_arena)
    fused_acquire = fused_source.index("acquire_expert_activation(")
    fused_dispatch_finish = fused_source.index(
        "finish_deepep_dispatch_external_with_options("
    )
    fused_expert = fused_source.index("expert_out = self.experts(", fused_acquire)

    assert forward_acquire < forward_expert < forward_release
    assert "activation_allocation=expert_activation_lease.allocate" in forward_source
    assert (
        'expert_activation_lease.tensor(\n                        "fc1_input"'
        in forward_source
    )
    assert "output_allocation=_expert_activation_output_allocation(" in forward_source
    assert "wgrad_done" not in forward_source
    assert saved_arena < saved_expert
    assert "activation_allocation=" not in saved_forward_source
    assert fused_dispatch_finish < fused_acquire < fused_expert
    assert "activation_allocation=expert_activation_lease.allocate" in fused_source
    assert (
        'expert_activation_lease.tensor(\n                    "fc1_input"'
        in fused_source
    )
    assert "expert_input = fc1_input.requires_grad_(True)" in fused_source
    assert "output_allocation=_expert_activation_output_allocation(" in fused_source
    no_grad_finish = forward_source.index("finish_deepep_combine(state)")
    no_grad_release = forward_source.index("lease.release(consumed)", no_grad_finish)
    assert no_grad_finish < no_grad_release
    assert "cuda.synchronize" not in forward_source
    assert "cuda.synchronize" not in saved_forward_source
    assert "cuda.synchronize" not in fused_source


def test_expert_activation_output_allocation_returns_owned_tensor(
    transformer_engine_import_stub,
):
    """Both no-grad forward and fused use the same Tensor-returning callback."""
    transformer_engine_import_stub()
    from megatron.lite.primitive.modules.moe_ep_chunk_overlap import (
        _expert_activation_output_allocation,
    )

    class FakeLease:
        def __init__(self):
            self.calls = []

        def tensor(self, name, shape, *, dtype, device):
            self.calls.append((name, shape, dtype, device))
            return torch.empty(shape, dtype=dtype, device=device)

    lease = FakeLease()
    input_tensor = torch.empty((3, 2), dtype=torch.bfloat16)

    allocation = _expert_activation_output_allocation(lease, input_tensor)
    output = allocation("fc1_output", (5, 4))

    assert isinstance(output, torch.Tensor)
    assert output.shape == (5, 4)
    assert output.dtype is input_tensor.dtype
    assert output.device == input_tensor.device
    assert lease.calls == [
        ("fc1_output", (5, 4), input_tensor.dtype, input_tensor.device)
    ]


def test_no_grad_and_fused_expert_compute_do_not_nest_slot_and_activation_arenas(
    transformer_engine_import_stub,
):
    """DeepEP receives use a slot pool; owned expert activations use one other pool."""
    import inspect

    transformer_engine_import_stub()
    from megatron.lite.primitive.modules.moe_ep_chunk_overlap import (
        _EPChunkOperationBase,
    )

    forward_source = inspect.getsource(_EPChunkOperationBase._forward_output_async)
    forward_expert = forward_source[
        forward_source.index("def run_expert") : forward_source.index(
            "def submit_combine"
        )
    ]
    fused_source = inspect.getsource(
        _EPChunkOperationBase._full_recompute_fused_backward_v6
    )
    fused_expert = fused_source[
        fused_source.index("def finish_recompute_expert") : fused_source.index(
            "def retire_pending_dispatch_bwd"
        )
    ]

    assert "allocation_arena.allocate" not in forward_expert
    assert "allocation_arena.allocate" not in fused_expert


def test_experts_split_fc1_and_swiglu_forward_activation_from_fc2_output(
    monkeypatch, transformer_engine_import_stub
):
    from contextlib import contextmanager

    transformer_engine_import_stub()
    from megatron.lite.primitive.modules import experts as experts_module

    active = []
    calls = []
    constructed = []

    @contextmanager
    def stage(name):
        active.append(name)
        try:
            yield
        finally:
            assert active.pop() == name

    class FakeGroupedLinear(torch.nn.Module):
        def __init__(self, _num_gemms, _in_features, out_features, **_kwargs):
            super().__init__()
            self.name = "fc1" if not constructed else "fc2"
            self.out_features = out_features
            constructed.append(self)

        def forward(self, x, _splits):
            calls.append((self.name, tuple(active)))
            return x.new_ones((x.size(0), self.out_features))

    def weighted(value, _probs, _limit):
        calls.append(("weighted_swiglu", tuple(active)))
        return value[:, :2]

    monkeypatch.setattr(
        experts_module.te, "GroupedLinear", FakeGroupedLinear, raising=False
    )
    monkeypatch.setattr(experts_module, "swiglu_with_probs", weighted)
    experts = experts_module.Experts(
        SimpleNamespace(
            num_experts=2,
            hidden_size=2,
            moe_intermediate_size=2,
            swiglu_limit=0.0,
        ),
        SimpleNamespace(ep_size=1, etp_size=1, tp_size=1, etp_group=None),
    )

    with stage("slot"):
        experts(
            torch.ones(3, 2),
            None,
            torch.ones(3),
            tokens_per_expert_list=[1, 2],
            activation_allocation=lambda: stage("activation"),
        )

    assert calls == [
        ("fc1", ("slot", "activation")),
        ("weighted_swiglu", ("slot", "activation")),
        ("fc2", ("slot",)),
    ]


def test_experts_owned_outputs_only_request_dgrad_for_each_grad_input(
    monkeypatch, transformer_engine_import_stub
):
    """The production adapter must not allocate FC1 dgrad for wgrad-only input."""
    transformer_engine_import_stub()
    from megatron.lite.primitive.modules import experts as experts_module

    constructed = []
    adapter_calls = []
    allocated = []

    class FakeGroupedLinear(torch.nn.Module):
        def __init__(self, _num_gemms, _in_features, out_features, **_kwargs):
            super().__init__()
            self.name = "fc1" if not constructed else "fc2"
            self.out_features = out_features
            constructed.append(self)

    def fake_owned_linear(linear, x, _m_splits, out, dgrad_out):
        adapter_calls.append((linear.name, x.requires_grad, dgrad_out is not None))
        if linear.name == "fc1":
            # FC1 parameter gradients make its output an autograd input to FC2,
            # even though the original expert input needs no dgrad.
            return out.requires_grad_(True)
        return out

    def allocation(name, shape):
        allocated.append(name)
        return torch.empty(shape, dtype=torch.bfloat16)

    monkeypatch.setattr(
        experts_module.te, "GroupedLinear", FakeGroupedLinear, raising=False
    )
    monkeypatch.setattr(
        experts_module, "_caller_owned_grouped_linear", fake_owned_linear
    )
    monkeypatch.setattr(experts_module, "swiglu_with_probs", lambda y, *_args: y[:, :2])
    experts = experts_module.Experts(
        SimpleNamespace(
            num_experts=2,
            hidden_size=2,
            moe_intermediate_size=2,
            swiglu_limit=0.0,
        ),
        SimpleNamespace(ep_size=1, etp_size=1, tp_size=1, etp_group=None),
        delay_wgrad_compute=True,
    )

    with torch.enable_grad():
        output = experts(
            torch.ones(3, 2, dtype=torch.bfloat16),
            None,
            tokens_per_expert_list=[1, 2],
            output_allocation=allocation,
        )

    assert output.shape == (3, 2)
    assert adapter_calls == [("fc1", False, False), ("fc2", True, True)]
    assert allocated == ["fc1_output", "fc2_output", "fc2_dgrad"]


def test_caller_owned_grouped_linear_keeps_te_delayed_wgrad_contract(
    transformer_engine_import_stub,
):
    """The output adapter is narrow: it must not silently become generic TE."""
    import inspect

    transformer_engine_import_stub()
    from megatron.lite.primitive.modules import experts as experts_module

    source = inspect.getsource(experts_module._CallerOwnedGroupedLinear)
    wrapper_source = inspect.getsource(experts_module._caller_owned_grouped_linear)
    assert "general_grouped_gemm" in source
    assert "wgrad_store.put" in source
    assert "main_grad sinks" in source
    assert "grad_added_to_main_grad" in source
    assert "overwrite_main_grad" in source
    assert "_2X_ACC_FPROP" in source
    assert "_2X_ACC_DGRAD" in source
    assert "_2X_ACC_WGRAD" in source
    assert "prepare_forward" in wrapper_source
    assert "end_forward" in wrapper_source
    assert "_get_weight_tensors" in wrapper_source
    assert "_get_quantizers" in wrapper_source
    assert "torch.bfloat16" in source
    assert "dgrad_out" in source
    assert "torch.empty_like(inp)" not in source


def test_caller_owned_dummy_wgrad_is_eager_scoped_and_capture_stable(
    monkeypatch, transformer_engine_import_stub
):
    """Eager dummies must not become TE's process-global allocation cache."""
    transformer_engine_import_stub()
    import gc
    import sys
    import weakref

    from megatron.lite.primitive.modules import experts as experts_module

    base = sys.modules["transformer_engine.pytorch.module.base"]
    cached = torch.full((2, 3), 9.0, dtype=torch.bfloat16)
    cache_calls = []

    def fake_te_dummy(shape, dtype, zero=False):
        cache_calls.append((shape, dtype, zero))
        if zero:
            cached.zero_()
        return cached

    monkeypatch.setattr(base, "get_dummy_wgrad", fake_te_dummy, raising=False)
    main_grad = torch.ones(2, 3, dtype=torch.float32)
    weight = torch.nn.Parameter(torch.ones(2, 3, dtype=torch.bfloat16))

    monkeypatch.setattr(experts_module, "_caller_owned_dummy_is_capturing", lambda _x: False)
    first = experts_module._caller_owned_dummy_wgrad(main_grad, weight, zero=True)
    second = experts_module._caller_owned_dummy_wgrad(main_grad, weight, zero=False)

    assert first is not None and second is not None
    assert first.shape == main_grad.shape == second.shape
    assert first.dtype == weight.dtype == second.dtype
    assert torch.count_nonzero(first) == 0
    assert first.data_ptr() != second.data_ptr()
    assert cache_calls == []
    first_ref, second_ref = weakref.ref(first), weakref.ref(second)
    del first, second
    gc.collect()
    assert first_ref() is None and second_ref() is None

    monkeypatch.setattr(experts_module, "_caller_owned_dummy_is_capturing", lambda _x: True)
    captured_first = experts_module._caller_owned_dummy_wgrad(main_grad, weight, zero=False)
    captured_second = experts_module._caller_owned_dummy_wgrad(main_grad, weight, zero=True)

    assert captured_first.data_ptr() == captured_second.data_ptr()
    assert torch.count_nonzero(captured_second) == 0
    assert cache_calls == [([2, 3], torch.bfloat16, False), ([2, 3], torch.bfloat16, True)]


def test_caller_owned_grouped_linear_executes_te_lifecycle_and_partial_overlap_wgrad(
    monkeypatch, transformer_engine_import_stub
):
    """CPU fake of TE2.15's narrow adapter: buffers, hooks, and delayed GEMM execute."""
    transformer_engine_import_stub()
    import sys

    base = sys.modules["transformer_engine.pytorch.module.base"]
    base._2X_ACC_FPROP = False
    base._2X_ACC_DGRAD = False
    base._2X_ACC_WGRAD = False
    base.get_dummy_wgrad = lambda shape, dtype, zero=False: (
        torch.zeros(shape, dtype=dtype) if zero else torch.ones(shape, dtype=dtype)
    )
    cpp = sys.modules["transformer_engine.pytorch.cpp_extensions"]
    trace = {
        "fc1_dgrad_ptr": None,
        "overlap_dgrad_ptr": None,
        "overlap_dgrad_end": None,
        "events": [],
    }

    def fake_grouped_gemm(
        weights, inputs, outputs, quantization_params=None, out_dtype=None, **kwargs
    ):
        if kwargs.get("layout") == "NN":
            pieces = [inp @ weight for weight, inp in zip(weights, inputs)]
        elif kwargs.get("layout") == "NT":
            if any(
                trace["overlap_dgrad_ptr"] < grad.data_ptr() + grad.nbytes
                and grad.data_ptr() < trace["overlap_dgrad_end"]
                for grad in inputs
                if trace["overlap_dgrad_ptr"] is not None
            ):
                trace["events"].append("overlap wgrad reads grad_output")
            pieces = [grad.T @ inp for inp, grad in zip(weights, inputs)]
            for dst, piece in zip(outputs, pieces, strict=True):
                if kwargs.get("accumulate"):
                    dst.add_(piece)
                else:
                    dst.copy_(piece)
            if (
                trace["events"]
                and trace["events"][-1] == "overlap wgrad reads grad_output"
            ):
                trace["events"].append("overlap wgrad completes")
            return None, [None] * len(pieces), None
        else:
            pieces = [inp @ weight.T for weight, inp in zip(weights, inputs)]
        outputs[0].copy_(torch.cat(pieces, dim=0))
        if (
            trace["overlap_dgrad_ptr"] is not None
            and trace["overlap_dgrad_ptr"] < outputs[0].data_ptr() + outputs[0].nbytes
            and outputs[0].data_ptr() < trace["overlap_dgrad_end"]
        ):
            trace["events"].append("overlap dgrad first write")
        if (
            kwargs.get("layout") == "NN"
            and outputs[0].data_ptr() == trace["fc1_dgrad_ptr"]
        ):
            trace["events"].append(
                "fc2_dgrad_first_write"
                if outputs[0].shape[1] == 3
                else "fc1_dgrad_first_write"
            )
        return None, [None] * len(pieces), None

    cpp.general_grouped_gemm = fake_grouped_gemm
    from megatron.lite.primitive.modules import experts as experts_module

    class Store:
        def __init__(self):
            self.entries = []
            self.context = SimpleNamespace(empty=lambda: not self.entries)

        @staticmethod
        def delay_wgrad_compute():
            return True

        def put(self, tensors, fn):
            self.entries.append((tensors, fn))

        def pop(self):
            tensors, fn = self.entries.pop(0)
            return fn(*tensors), tensors

    class Linear:
        def __init__(self):
            self.num_gemms = 2
            self.fp8 = False
            self.use_bias = False
            self.return_bias = False
            self.save_original_input = False
            self.fuse_wgrad_accumulation = True
            self.wgrad_store = Store()
            self.activation_dtype = torch.bfloat16
            self.weight0 = torch.nn.Parameter(
                torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=torch.bfloat16)
            )
            self.weight1 = torch.nn.Parameter(
                torch.tensor(
                    [[2.0, -1.0], [0.0, 3.0], [4.0, 1.0]], dtype=torch.bfloat16
                )
            )
            for weight in (self.weight0, self.weight1):
                weight.main_grad = torch.zeros_like(weight, dtype=torch.float32)
                weight.grad_added_to_main_grad = False
                weight.zero_out_wgrad = True
            self.calls = []

        def prepare_forward(self, value, *, num_gemms):
            self.calls.append(("prepare", num_gemms))
            return value

        def end_forward(self):
            self.calls.append(("end",))

        def _get_weight_tensors(self):
            self.calls.append(("weights",))
            return [self.weight0, self.weight1]

        def _get_bias_tensors(self):
            self.calls.append(("biases",))
            return []

        def _get_quantizers(self):
            self.calls.append(("quantizers",))
            return tuple([None, None] for _ in range(6))

    linear = Linear()
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.bfloat16, requires_grad=True)
    out = torch.empty(2, 3, dtype=torch.bfloat16)
    dgrad = torch.empty_like(x)
    result = experts_module._caller_owned_grouped_linear(linear, x, [1, 1], out, dgrad)
    assert result.data_ptr() == out.data_ptr()
    result.float().sum().backward()
    assert x.grad.data_ptr() == dgrad.data_ptr()
    torch.testing.assert_close(
        x.grad,
        torch.tensor([[9.0, 12.0], [6.0, 3.0]], dtype=torch.bfloat16),
    )
    assert linear.calls == [
        ("prepare", 2),
        ("weights",),
        ("biases",),
        ("quantizers",),
        ("end",),
    ]
    assert len(linear.wgrad_store.entries) == 1
    tensors, delayed = linear.wgrad_store.entries.pop()
    delayed(*tensors)
    assert all(
        weight.grad_added_to_main_grad for weight in (linear.weight0, linear.weight1)
    )
    assert all(
        torch.count_nonzero(weight.main_grad)
        for weight in (linear.weight0, linear.weight1)
    )
    torch.testing.assert_close(
        linear.weight0.main_grad,
        torch.tensor([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]]),
    )
    torch.testing.assert_close(
        linear.weight1.main_grad,
        torch.tensor([[3.0, 4.0], [3.0, 4.0], [3.0, 4.0]]),
    )

    # Fused 3-slot mode may use the FC2 output bytes as the FC2 dgrad output.
    # The explicit grad_output and dgrad_out below are different contiguous
    # views with a partial byte-range overlap, not a same-data_ptr shortcut.
    # Wgrad must execute before Dgrad overwrites the shared bytes, rather than
    # enqueueing a deferred callback that would read corruption during flush.
    overlap_linear = Linear()
    overlap_linear.weight0 = torch.nn.Parameter(
        torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.bfloat16)
    )
    overlap_linear.weight1 = torch.nn.Parameter(
        torch.tensor([[2.0, -1.0, 4.0], [0.0, 3.0, 5.0]], dtype=torch.bfloat16)
    )
    for weight in (overlap_linear.weight0, overlap_linear.weight1):
        weight.main_grad = torch.zeros_like(weight, dtype=torch.float32)
        weight.grad_added_to_main_grad = False
        weight.zero_out_wgrad = True
    overlap_x = torch.tensor(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    overlap_out = torch.empty(2, 2, dtype=torch.bfloat16)
    overlap_raw = torch.empty(8, dtype=torch.bfloat16)
    overlap_grad_output = overlap_raw[:4].view(2, 2)
    overlap_dgrad_out = overlap_raw[2:].view(2, 3)
    assert overlap_grad_output.data_ptr() != overlap_dgrad_out.data_ptr()
    overlap_result = experts_module._caller_owned_grouped_linear(
        overlap_linear,
        overlap_x,
        [1, 1],
        overlap_out,
        overlap_dgrad_out,
    )
    overlap_grad_output.copy_(
        torch.tensor([[2.0, 3.0], [5.0, 7.0]], dtype=torch.bfloat16)
    )
    trace["overlap_dgrad_ptr"] = overlap_dgrad_out.data_ptr()
    trace["overlap_dgrad_end"] = overlap_dgrad_out.data_ptr() + overlap_dgrad_out.nbytes
    overlap_dgrad = torch.autograd.grad(overlap_result, overlap_x, overlap_grad_output)[
        0
    ]

    assert overlap_dgrad.data_ptr() == overlap_dgrad_out.data_ptr()
    torch.testing.assert_close(
        overlap_dgrad,
        torch.tensor([[14.0, 19.0, 24.0], [10.0, 16.0, 55.0]], dtype=torch.bfloat16),
    )
    assert overlap_linear.wgrad_store.entries == []
    assert trace["events"] == [
        "overlap wgrad reads grad_output",
        "overlap wgrad completes",
        "overlap dgrad first write",
    ]
    assert all(
        weight.grad_added_to_main_grad
        for weight in (overlap_linear.weight0, overlap_linear.weight1)
    )
    torch.testing.assert_close(
        overlap_linear.weight0.main_grad,
        torch.tensor([[2.0, 4.0, 6.0], [3.0, 6.0, 9.0]]),
    )
    torch.testing.assert_close(
        overlap_linear.weight1.main_grad,
        torch.tensor([[20.0, 25.0, 30.0], [28.0, 35.0, 42.0]]),
    )
    trace["events"].clear()
    trace["overlap_dgrad_ptr"] = None
    trace["overlap_dgrad_end"] = None

    no_dgrad_x = torch.tensor([[2.0, 1.0], [4.0, 3.0]], dtype=torch.bfloat16)
    no_dgrad_out = torch.empty(2, 3, dtype=torch.bfloat16)
    no_dgrad_result = experts_module._caller_owned_grouped_linear(
        linear, no_dgrad_x, [1, 1], no_dgrad_out, None
    )
    assert no_dgrad_result.requires_grad
    no_dgrad_result.float().sum().backward()
    assert len(linear.wgrad_store.entries) == 1
    tensors, delayed = linear.wgrad_store.entries.pop()
    delayed(*tensors)

    # Full two-linear CPU probe: use the fused FC2 raw-byte coloring, run fake
    # grouped GEMMs, and prove that the colliding FC2 Wgrad is immediate while
    # FC1 remains deferred when SwiGLU supplies it a non-overlapping gradient.
    from megatron.lite.primitive.modules.moe_ep_chunk_overlap import (
        EPChunkShapeProfile,
        EPChunkWorkspaceKey,
        EPChunkWorkspaceRegistry,
    )

    workspace = EPChunkWorkspaceRegistry().get_or_create(
        EPChunkWorkspaceKey(
            op="fused_forward_backward",
            device_type="cpu",
            device_index=None,
            ep_group_id=101,
            dtype=torch.bfloat16,
            shape_profile=EPChunkShapeProfile(
                max_input_rows=8, hidden_size=2, topk=2, ep_size=2
            ),
        ),
        lambda slot: SimpleNamespace(slot=slot, use_deepep=True),
    )
    lease = workspace.acquire_expert_activation()
    colored_fc1_out = lease.tensor(
        "fc1_output", (2, 3), dtype=torch.bfloat16, device="cpu"
    )
    colored_fc2_out = lease.tensor(
        "fc2_output", (2, 2), dtype=torch.bfloat16, device="cpu"
    )
    colored_fc1_dgrad = lease.tensor(
        "fc1_dgrad", (2, 2), dtype=torch.bfloat16, device="cpu"
    )
    colored_fc2_dgrad = lease.tensor(
        "fc2_dgrad", (2, 3), dtype=torch.bfloat16, device="cpu"
    )
    assert colored_fc2_out.data_ptr() == colored_fc1_dgrad.data_ptr()
    assert colored_fc2_dgrad.data_ptr() == colored_fc1_dgrad.data_ptr()
    # This isolated adapter probe keeps FC1's caller-owned dgrad separate so
    # the non-overlap/deferred branch remains executable alongside FC2's
    # fused collision. Dedicated workspace tests retain the production 3-slot map.
    colored_fc1_dgrad = torch.empty_like(colored_fc1_dgrad)

    fc1 = Linear()
    fc2 = Linear()
    fc2.weight0 = torch.nn.Parameter(
        torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]], dtype=torch.bfloat16)
    )
    fc2.weight1 = torch.nn.Parameter(
        torch.tensor([[2.0, 0.0, 0.0], [0.0, 3.0, 0.0]], dtype=torch.bfloat16)
    )
    for weight in (fc2.weight0, fc2.weight1):
        weight.main_grad = torch.zeros_like(weight, dtype=torch.float32)
        weight.grad_added_to_main_grad = False
        weight.zero_out_wgrad = True
    colored_x = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0]], dtype=torch.bfloat16, requires_grad=True
    )
    trace["fc1_dgrad_ptr"] = colored_fc1_dgrad.data_ptr()
    colored_fc1 = experts_module._caller_owned_grouped_linear(
        fc1, colored_x, [1, 1], colored_fc1_out, colored_fc1_dgrad
    )

    class FakeSwiGLU(torch.autograd.Function):
        """CPU stand-in that records FC2-dgrad's final intermediate consumer."""

        @staticmethod
        def forward(ctx, value):
            return value.clone()

        @staticmethod
        def backward(ctx, grad):
            trace["events"].append("swiglu_intermediate_consumer")
            return grad.clone()

    colored_fc2 = experts_module._caller_owned_grouped_linear(
        fc2,
        FakeSwiGLU.apply(colored_fc1),
        [1, 1],
        colored_fc2_out,
        colored_fc2_dgrad,
    )
    # Match the fused production path: obtain the custom-Function edge before
    # writing the same FC2-output bytes used as the incoming gradient source.
    expert_out_edge = torch.autograd.graph.get_gradient_edge(colored_fc2)
    colored_fc2_grad = colored_fc2.detach()
    colored_fc2_grad.copy_(torch.tensor([[2.0, 3.0], [5.0, 7.0]], dtype=torch.bfloat16))
    trace["overlap_dgrad_ptr"] = colored_fc2_dgrad.data_ptr()
    trace["overlap_dgrad_end"] = colored_fc2_dgrad.data_ptr() + colored_fc2_dgrad.nbytes
    colored_x_dgrad = torch.autograd.grad(expert_out_edge, colored_x, colored_fc2_grad)[
        0
    ]
    assert trace["events"] == [
        "overlap wgrad reads grad_output",
        "overlap wgrad completes",
        "overlap dgrad first write",
        "swiglu_intermediate_consumer",
        "fc1_dgrad_first_write",
    ]
    assert colored_x_dgrad.data_ptr() == colored_fc1_dgrad.data_ptr()
    torch.testing.assert_close(
        colored_x_dgrad,
        torch.tensor([[36.0, 46.0], [20.0, 53.0]], dtype=torch.bfloat16),
    )
    assert fc2.wgrad_store.entries == []
    assert getattr(fc2.wgrad_store, "_mlite_immediate_wgrad_contexts", 0) == 1
    assert len(fc1.wgrad_store.entries) == 1
    assert getattr(fc1.wgrad_store, "_mlite_immediate_wgrad_contexts", 0) == 0

    flush_owner = SimpleNamespace(
        fc1=fc1,
        fc2=fc2,
        _validate_weight_grad_sink=experts_module.Experts._validate_weight_grad_sink,
        release_delayed_weight_grad_aliases=lambda: None,
    )
    experts_module.Experts.flush_delayed_weight_grads(flush_owner, num_contexts=1)
    assert fc1.wgrad_store.context.empty()
    assert fc2.wgrad_store.context.empty()
    assert fc1.wgrad_store._mlite_immediate_wgrad_contexts == 0
    assert fc2.wgrad_store._mlite_immediate_wgrad_contexts == 0
    torch.testing.assert_close(
        fc1.weight0.main_grad,
        torch.tensor([[2.0, 4.0], [3.0, 6.0], [5.0, 10.0]]),
    )
    torch.testing.assert_close(
        fc1.weight1.main_grad,
        torch.tensor([[30.0, 40.0], [63.0, 84.0], [0.0, 0.0]]),
    )
    torch.testing.assert_close(
        fc2.weight0.main_grad,
        torch.tensor([[10.0, 22.0, 34.0], [15.0, 33.0, 51.0]]),
    )
    torch.testing.assert_close(
        fc2.weight1.main_grad,
        torch.tensor([[10.0, 60.0, 80.0], [14.0, 84.0, 112.0]]),
    )
    trace["overlap_dgrad_ptr"] = None
    trace["overlap_dgrad_end"] = None

    class ReadyEvent:
        @staticmethod
        def query():
            return True

    lease.release(ReadyEvent())

    # TE2.15 must defer MCore-FSDP main_grad lookup until backward, after the
    # framework has materialized the sink.  The saved weight Tensor is not the
    # source of these Python-side hook attributes.
    for weight in (linear.weight0, linear.weight1):
        sink = weight.main_grad
        del weight.main_grad
        weight.__fsdp_param__ = True
        weight.get_main_grad = lambda sink=sink: sink
    fsdp_x = torch.tensor(
        [[1.0, 1.0], [2.0, 2.0]], dtype=torch.bfloat16, requires_grad=True
    )
    fsdp_out = torch.empty(2, 3, dtype=torch.bfloat16)
    fsdp_dgrad = torch.empty_like(fsdp_x)
    experts_module._caller_owned_grouped_linear(
        linear, fsdp_x, [1, 1], fsdp_out, fsdp_dgrad
    ).float().sum().backward()
    tensors, delayed = linear.wgrad_store.entries.pop()
    delayed(*tensors)
    assert fsdp_x.grad.data_ptr() == fsdp_dgrad.data_ptr()

    with pytest.raises(RuntimeError, match="non-grad"):
        experts_module._caller_owned_grouped_linear(
            linear, x, [1, 1], out.detach().requires_grad_(), dgrad
        )
    assert linear.calls[-1] == ("end",)


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
