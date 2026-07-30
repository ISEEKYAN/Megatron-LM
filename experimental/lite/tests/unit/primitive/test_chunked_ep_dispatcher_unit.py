# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import importlib
import sys
import types
from types import SimpleNamespace

import pytest
import torch


def _install_te_import_stub():
    def unavailable(*_args, **_kwargs):
        return None

    root = types.ModuleType("transformer_engine")
    pytorch = types.ModuleType("transformer_engine.pytorch")
    cpp_extensions = types.ModuleType("transformer_engine.pytorch.cpp_extensions")
    module = types.ModuleType("transformer_engine.pytorch.module")
    module_base = types.ModuleType("transformer_engine.pytorch.module.base")
    permutation = types.ModuleType("transformer_engine.pytorch.permutation")
    router = types.ModuleType("transformer_engine.pytorch.router")
    cpp_extensions.general_gemm = unavailable
    module_base.get_workspace = unavailable
    module.base = module_base
    permutation.moe_permute = unavailable
    permutation.moe_permute_and_pad_with_probs = unavailable
    permutation.moe_permute_with_probs = unavailable
    permutation.moe_unpermute = unavailable
    router.fused_compute_score_for_moe_aux_loss = unavailable
    router.fused_moe_aux_loss = unavailable
    router.fused_topk_with_score_function = unavailable
    root.pytorch = pytorch
    for name, value in {
        "transformer_engine": root,
        "transformer_engine.pytorch": pytorch,
        "transformer_engine.pytorch.cpp_extensions": cpp_extensions,
        "transformer_engine.pytorch.module": module,
        "transformer_engine.pytorch.module.base": module_base,
        "transformer_engine.pytorch.permutation": permutation,
        "transformer_engine.pytorch.router": router,
    }.items():
        sys.modules.setdefault(name, value)


_install_te_import_stub()

dispatcher_module = importlib.import_module(
    "megatron.lite.primitive.modules.dispatcher"
)


class _Event:
    def __init__(self):
        self.waited = False
        self.event = object()

    def current_stream_wait(self):
        self.waited = True


class _Buffer:
    def __init__(self):
        self.combine_calls = []
        self.dispatch_calls = []

    def combine(self, tensor, handle, **kwargs):
        self.combine_calls.append((tensor, handle, kwargs))
        event = _Event()
        return tensor + 1, tensor.new_zeros((tensor.size(0), 1)), event

    def dispatch(self, tensor, **kwargs):
        self.dispatch_calls.append((tensor, kwargs))
        event = _Event()
        return tensor + 2, None, None, None, None, event


def _dispatcher(buffer=None):
    value = object.__new__(dispatcher_module.TokenDispatcher)
    value.use_deepep = True
    value.buffer = buffer or _Buffer()
    value.num_experts = 4
    value.num_local_experts = 2
    value.ep_size = 2
    value.moe_permute_fusion = False
    value.ps = SimpleNamespace(ep_group=None)
    value._row_id_map = torch.tensor([0, 1])
    value._restore_shape = (2, 3)
    value._handle = "dispatch-handle"
    value._local_tpe_list = [1, 1]
    return value


def test_deepep_buffer_slot_reuses_only_matching_slot(monkeypatch):
    dispatcher_module._DEEPEP_BUFFER_CACHE.clear()
    built = []

    monkeypatch.setattr(dispatcher_module.dist, "get_world_size", lambda group: 8)
    monkeypatch.setattr(
        dispatcher_module,
        "_build_deepep_buffer",
        lambda group, hidden_size: built.append((group, hidden_size)) or object(),
    )
    group = object()

    first = dispatcher_module._get_deepep_buffer(group, 4096, buffer_slot="chunk-0")
    again = dispatcher_module._get_deepep_buffer(group, 4096, buffer_slot="chunk-0")
    other = dispatcher_module._get_deepep_buffer(group, 4096, buffer_slot="chunk-1")

    assert first is again
    assert other is not first
    assert built == [(group, 4096), (group, 4096)]


def test_deepep_dispatch_materializes_contiguous_router_outputs():
    class Buffer(_Buffer):
        def __init__(self):
            super().__init__()
            self.layout_indices = None

        def get_dispatch_layout(self, topk_indices, **_kwargs):
            self.layout_indices = topk_indices
            assert topk_indices.is_contiguous()
            return None, None, None, None, _Event()

    buffer = Buffer()
    value = _dispatcher(buffer)
    hidden = torch.zeros(3, 2)
    scores = torch.ones(2, 3).transpose(0, 1)
    indices = torch.zeros(2, 3, dtype=torch.long).transpose(0, 1)
    assert not scores.is_contiguous()
    assert not indices.is_contiguous()

    value.submit_deepep_dispatch(hidden, scores, indices)

    assert buffer.layout_indices is buffer.dispatch_calls[0][1]["topk_idx"]
    assert buffer.dispatch_calls[0][1]["topk_weights"].is_contiguous()


def test_fixed_capacity_dispatch_resolves_global_local_expert_counts(monkeypatch):
    class Work:
        waited = False

        def wait(self):
            self.waited = True

    class Buffer(_Buffer):
        def get_dispatch_layout(self, _topk_indices, **_kwargs):
            return (
                torch.zeros(2, dtype=torch.int32),
                None,
                torch.tensor([1, 2, 3, 4], dtype=torch.int32),
                torch.zeros(3, 2, dtype=torch.bool),
                _Event(),
            )

    work = Work()

    def all_reduce(counts, group, async_op):
        assert group is value.ps.tp_ep_group
        assert async_op
        assert counts.numel() == 6
        counts.add_(torch.tensor([5, 6, 10, 20, 30, 40], dtype=counts.dtype))
        return work

    monkeypatch.setattr(
        dispatcher_module, "_event_current_stream_wait", lambda _event: None
    )
    capacity_checks = []
    monkeypatch.setattr(
        torch,
        "_assert_async",
        lambda condition, message: capacity_checks.append(
            (bool(condition.item()), message)
        ),
    )
    value = _dispatcher(Buffer())
    value.ps = SimpleNamespace(tp_ep_group=object(), ep_rank=1)
    monkeypatch.setattr(dispatcher_module.dist, "all_reduce", all_reduce)
    hidden = torch.zeros(3, 2)
    scores = torch.ones(3, 1)
    indices = torch.zeros(3, 1, dtype=torch.long)

    state = value.submit_deepep_dispatch(hidden, scores, indices, num_worst_tokens=80)
    recv_per_expert = value._resolve_deepep_recv_per_expert(state)

    assert value.buffer.dispatch_calls[0][1]["num_worst_tokens"] == 80
    assert state["capacity_rows"] == 80
    assert recv_per_expert == [33, 44]
    assert work.waited
    assert capacity_checks == [
        (True, "DeepEP receive rows exceed the fixed dispatch capacity")
    ]


def test_fixed_capacity_dispatch_fails_before_deepep_when_receive_rows_exceed_bound(
    monkeypatch,
):
    class Work:
        def wait(self):
            return None

    class Buffer(_Buffer):
        def get_dispatch_layout(self, _topk_indices, **_kwargs):
            return (
                torch.tensor([40, 100], dtype=torch.int32),
                None,
                torch.tensor([1, 2, 3, 4], dtype=torch.int32),
                torch.zeros(3, 2, dtype=torch.bool),
                _Event(),
            )

    def fail_loud(condition, message):
        if not bool(condition.item()):
            raise RuntimeError(message)

    buffer = Buffer()
    value = _dispatcher(buffer)
    value.ps = SimpleNamespace(tp_ep_group=object(), ep_rank=1)
    monkeypatch.setattr(
        dispatcher_module, "_event_current_stream_wait", lambda _event: None
    )
    monkeypatch.setattr(
        dispatcher_module.dist,
        "all_reduce",
        lambda _counts, group, async_op: Work(),
    )
    monkeypatch.setattr(torch, "_assert_async", fail_loud)

    with pytest.raises(
        RuntimeError, match="DeepEP receive rows exceed the fixed dispatch capacity"
    ):
        value.submit_deepep_dispatch(
            torch.zeros(3, 2),
            torch.ones(3, 1),
            torch.zeros(3, 1, dtype=torch.long),
            num_worst_tokens=76,
        )

    assert buffer.dispatch_calls == []


def test_prepared_combine_preserves_manual_metadata_and_finishes(monkeypatch):
    value = _dispatcher()
    monkeypatch.setattr(
        dispatcher_module,
        "unpermute",
        lambda tensor, row_id_map, restore_shape, fused: tensor + 10,
    )
    expert_output = torch.zeros(2, 3)

    rank_grouped, handle = value.prepare_deepep_combine(expert_output)
    state = value.submit_deepep_combine_prepared(
        rank_grouped, handle, async_finish=True
    )

    assert handle == "dispatch-handle"
    event = state["event"]
    result = value.finish_deepep_combine(state)
    assert event.waited
    assert torch.equal(result, torch.full((2, 3), 11.0))
    assert state == {}
    assert value._handle is None


def test_manual_backward_submit_finish_pairs_wait_for_events():
    value = _dispatcher()
    grad = torch.zeros(2, 3)

    combine_state = value.submit_deepep_combine_backward(grad, "combine-handle")
    combine_event = combine_state["event"]
    assert torch.equal(
        value.finish_deepep_combine_backward(combine_state), torch.full((2, 3), 2.0)
    )
    assert combine_event.waited

    dispatch_state = value.submit_deepep_dispatch_backward(
        grad, torch.ones(2, 1), "dispatch-handle"
    )
    dispatch_event = dispatch_state["event"]
    grad_hidden, grad_scores = value.finish_deepep_dispatch_backward(dispatch_state)
    assert torch.equal(grad_hidden, torch.ones(2, 3))
    assert grad_scores.shape == (2, 1)
    assert dispatch_event.waited


def test_backward_only_combine_defers_transport_to_autograd(monkeypatch):
    value = _dispatcher()
    monkeypatch.setattr(
        dispatcher_module,
        "unpermute",
        lambda tensor, row_id_map, restore_shape, fused: tensor,
    )
    expert_output = torch.ones(2, 3, requires_grad=True)

    output = value.combine_deepep_backward_only(expert_output, (2, 3))
    output.sum().backward()

    assert torch.equal(expert_output.grad, torch.full((2, 3), 3.0))
    assert len(value.buffer.dispatch_calls) == 1
    assert value._handle is None


def test_external_finish_returns_manual_map_without_mutating_dispatcher():
    value = _dispatcher()
    value._row_id_map = None
    value._restore_shape = None
    recv_hidden = torch.tensor([[1.0], [2.0], [3.0]])
    recv_indices = torch.tensor([[1, -1], [0, 1], [0, -1]])
    recv_probs = torch.tensor([[0.2, 0.0], [0.3, 0.4], [0.5, 0.0]])

    dispatched, local_tpe, probs, metadata = value._finish_deepep_dispatch_external(
        recv_hidden,
        recv_indices,
        recv_probs,
        [2, 2],
        force_manual_map=True,
        force_direct_permute=True,
    )

    assert dispatched.squeeze(-1).tolist() == [2.0, 3.0, 1.0, 2.0]
    assert local_tpe.tolist() == [2, 2]
    assert probs.tolist() == pytest.approx([0.3, 0.5, 0.2, 0.4])
    assert metadata["manual_row_id_map"].tolist() == [1, 2, 0, 1]
    assert value._row_id_map is None


def test_external_finish_does_not_read_cuda_metadata_with_tensor_item(monkeypatch):
    value = _dispatcher()
    recv_hidden = torch.tensor([[1.0], [2.0], [3.0]])
    recv_indices = torch.tensor([[1, -1], [0, 1], [0, -1]])
    recv_probs = torch.tensor([[0.2, 0.0], [0.3, 0.4], [0.5, 0.0]])

    def fail_item(_tensor):
        raise AssertionError("dispatch finish must validate host metadata without a device sync")

    monkeypatch.setattr(torch.Tensor, "item", fail_item)
    dispatched, local_tpe, _probs, metadata = (
        value._finish_deepep_dispatch_external(
            recv_hidden,
            recv_indices,
            recv_probs,
            [2, 2],
            force_manual_map=True,
            force_direct_permute=True,
        )
    )

    assert dispatched.shape == (4, 1)
    assert local_tpe.shape == (2,)
    assert metadata["local_tpe_list"] == [2, 2]
