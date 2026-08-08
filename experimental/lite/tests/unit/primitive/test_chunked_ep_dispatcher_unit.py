# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import importlib
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture(autouse=True)
def _isolated_te_import(transformer_engine_import_stub):
    transformer_engine_import_stub()


def _dispatcher_module():
    return importlib.import_module("megatron.lite.primitive.modules.dispatcher")


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
    value = object.__new__(_dispatcher_module().TokenDispatcher)
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

    state = value.submit_deepep_dispatch(hidden, scores, indices)

    assert buffer.layout_indices is buffer.dispatch_calls[0][1]["topk_idx"]
    assert buffer.dispatch_calls[0][1]["topk_weights"].is_contiguous()
    assert state["_dispatch_inputs"][0] is hidden
    assert state["_dispatch_inputs"][1] is buffer.layout_indices


def test_chunked_dispatch_uses_native_deepep_layout_and_preserves_evidence():
    class Buffer(_Buffer):
        def get_dispatch_layout(self, topk_indices, **_kwargs):
            self.layout_indices = topk_indices
            return (
                torch.tensor([3, 2], dtype=torch.int32),
                None,
                torch.tensor([2, 2, 1, 1], dtype=torch.int32),
                torch.tensor([[True, True], [True, True], [True, False]]),
                _Event(),
            )

    buffer = Buffer()
    value = _dispatcher(buffer)
    hidden = torch.zeros(3, 2)
    scores = torch.ones(3, 2)
    indices = torch.tensor([[0, 2], [1, 3], [0, 1]])

    state = value.submit_deepep_dispatch(hidden, scores, indices)

    kwargs = buffer.dispatch_calls[0][1]
    torch.testing.assert_close(
        kwargs["num_tokens_per_rank"], torch.tensor([3, 2], dtype=torch.int32)
    )
    torch.testing.assert_close(
        kwargs["num_tokens_per_expert"], torch.tensor([2, 2, 1, 1], dtype=torch.int32)
    )
    torch.testing.assert_close(
        kwargs["is_token_in_rank"],
        torch.tensor([[True, True], [True, True], [True, False]]),
    )
    assert kwargs["num_tokens_per_rdma_rank"] is None
    assert kwargs["previous_event"] is not None
    assert "num_worst_tokens" not in kwargs
    dispatch_inputs = state["_dispatch_inputs"]
    assert dispatch_inputs[0] is hidden
    assert dispatch_inputs[1] is buffer.layout_indices
    assert dispatch_inputs[2] is scores
    assert dispatch_inputs[3] is kwargs["num_tokens_per_rank"]
    assert dispatch_inputs[4] is kwargs["num_tokens_per_rdma_rank"]
    assert dispatch_inputs[5] is kwargs["num_tokens_per_expert"]
    assert dispatch_inputs[6] is kwargs["is_token_in_rank"]


def test_dynamic_dispatch_uses_deepep_receive_counts():
    class Buffer(_Buffer):
        def get_dispatch_layout(self, _topk_indices, **_kwargs):
            return (
                torch.zeros(2, dtype=torch.int32),
                None,
                torch.tensor([1, 2, 3, 4], dtype=torch.int32),
                torch.zeros(3, 2, dtype=torch.bool),
                _Event(),
            )

        def dispatch(self, tensor, **kwargs):
            self.dispatch_calls.append((tensor, kwargs))
            return tensor + 2, None, None, [2], None, _Event()

    value = _dispatcher(Buffer())
    hidden = torch.zeros(3, 2)
    scores = torch.ones(3, 1)
    indices = torch.zeros(3, 1, dtype=torch.long)

    state = value.submit_deepep_dispatch(hidden, scores, indices)
    recv_per_expert = value._resolve_deepep_recv_per_expert(state)

    assert "num_worst_tokens" not in value.buffer.dispatch_calls[0][1]
    assert "capacity_rows" not in state
    assert recv_per_expert == [2]


def test_prepared_combine_preserves_manual_metadata_and_finishes(monkeypatch):
    value = _dispatcher()
    monkeypatch.setattr(
        _dispatcher_module(),
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
        raise AssertionError(
            "dispatch finish must validate host metadata without a device sync"
        )

    monkeypatch.setattr(torch.Tensor, "item", fail_item)
    dispatched, local_tpe, _probs, metadata = value._finish_deepep_dispatch_external(
        recv_hidden,
        recv_indices,
        recv_probs,
        [2, 2],
        force_manual_map=True,
        force_direct_permute=True,
    )

    assert dispatched.shape == (4, 1)
    assert local_tpe.shape == (2,)
    assert metadata["local_tpe_list"] == [2, 2]
