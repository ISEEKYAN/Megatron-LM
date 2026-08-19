import types

import pytest
import torch

import megatron.lite.primitive.modules.dispatcher as dispatcher_module
from megatron.lite.primitive.alignment.dispatcher_transports import (
    alltoall,
    deepep,
    hybridep,
)
from megatron.lite.primitive.modules.dispatcher import TokenDispatcher
from megatron.lite.primitive.parallel import ParallelState


def _install_fake_ep_gather(monkeypatch) -> None:
    import vllm.model_executor.layers.fused_moe.deep_gemm_utils as deep_gemm_utils

    def fake_ep_gather(
        input_tensor, recv_ids, recv_weights, input_index, expert_map, output
    ):
        assert expert_map is None
        output.zero_()
        for token in range(recv_ids.shape[0]):
            accumulator = torch.zeros(
                input_tensor.shape[1], dtype=torch.float32
            )
            for slot in range(recv_ids.shape[1]):
                row = int(input_index[token, slot])
                if row >= 0:
                    accumulator += (
                        input_tensor[row].float() * recv_weights[token, slot]
                    )
            output[token].copy_(accumulator.to(output.dtype))

    monkeypatch.setattr(deep_gemm_utils, "ep_gather", fake_ep_gather)


def test_deepep_disables_unsafe_deterministic_empty_fill(monkeypatch) -> None:
    monkeypatch.setattr(
        deepep.torch,
        "are_deterministic_algorithms_enabled",
        lambda: True,
    )
    monkeypatch.setattr(
        deepep.torch.utils.deterministic,
        "fill_uninitialized_memory",
        True,
    )

    deepep.configure_deterministic_allocator()

    assert not deepep.torch.utils.deterministic.fill_uninitialized_memory


def test_deepep_initialization_matches_mcore_num_sms(monkeypatch) -> None:
    calls = []

    class FakeBuffer:
        @staticmethod
        def set_num_sms(value):
            calls.append(value)

    group = object()
    monkeypatch.setattr(
        deepep,
        "deep_ep",
        type("FakeDeepEP", (), {"Buffer": FakeBuffer}),
    )
    ps = types.SimpleNamespace(ep_size=2, tp_ep_group=group)

    dispatcher = TokenDispatcher(4, 16, ps, moe_token_dispatcher_type="deepep")

    assert calls == [20]
    assert dispatcher.buffer is None


def test_explicit_hybrid_transport_fails_closed_without_runtime(monkeypatch) -> None:
    monkeypatch.setattr(
        hybridep,
        "deep_ep",
        type("FakeDeepEP", (), {"Buffer": object}),
    )

    with pytest.raises(RuntimeError, match="HybridEPBuffer"):
        TokenDispatcher(
            4,
            16,
            types.SimpleNamespace(ep_size=2),
            moe_token_dispatcher_type="hybridep",
            deepep_align_to_low_latency=True,
        )


def test_hybridep_never_falls_back_to_ordinary_alltoall() -> None:
    with pytest.raises(ValueError, match="DeepEP-LL-aligned"):
        TokenDispatcher(
            4,
            16,
            types.SimpleNamespace(ep_size=1),
            moe_token_dispatcher_type="hybridep",
        )


def test_hybridep_accepts_cross_node_nvlink_domain(monkeypatch) -> None:
    created = []

    class FakeHybridBuffer:
        def __init__(self, **kwargs):
            created.append(kwargs)
            self.runtime = object()

    group = object()
    monkeypatch.setenv(
        "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN", "8"
    )
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "4")
    monkeypatch.setattr(
        hybridep,
        "deep_ep",
        type(
            "FakeDeepEP",
            (),
            {"Buffer": object, "HybridEPBuffer": FakeHybridBuffer},
        ),
    )
    monkeypatch.setattr(
        hybridep.dist, "get_world_size", lambda *, group: 8
    )
    monkeypatch.setattr(
        hybridep, "detect_accessible_ranks", lambda group: 8
    )
    monkeypatch.setattr(hybridep, "_buffer", None)
    monkeypatch.setattr(hybridep, "_buffer_capacity", 0)
    monkeypatch.setattr(hybridep, "_buffer_signature", None)

    buffer = hybridep._get_buffer(group, 16, 2, 32)

    assert buffer.runtime is not None
    assert created[0]["max_num_of_tokens_per_rank"] == 32


def test_hybridep_rejects_invalid_nvlink_domain(monkeypatch) -> None:
    monkeypatch.setenv(
        "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN", "3"
    )
    monkeypatch.setattr(
        hybridep,
        "deep_ep",
        type(
            "FakeDeepEP",
            (),
            {"Buffer": object, "HybridEPBuffer": object},
        ),
    )
    monkeypatch.setattr(
        hybridep.dist, "get_world_size", lambda *, group: 8
    )

    with pytest.raises(RuntimeError, match="not divisible"):
        hybridep._get_buffer(object(), 16, 2, 32)


def test_hybridep_rejects_runtime_topology_mismatch(monkeypatch) -> None:
    monkeypatch.setenv(
        "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN", "8"
    )
    monkeypatch.setattr(
        hybridep,
        "deep_ep",
        type(
            "FakeDeepEP",
            (),
            {"Buffer": object, "HybridEPBuffer": object},
        ),
    )
    monkeypatch.setattr(
        hybridep.dist, "get_world_size", lambda *, group: 8
    )
    monkeypatch.setattr(
        hybridep, "detect_accessible_ranks", lambda group: 4
    )

    with pytest.raises(RuntimeError, match="requested=8, detected=4"):
        hybridep._get_buffer(object(), 16, 2, 32)


def test_explicit_alltoall_never_silently_selects_deepep() -> None:
    dispatcher = TokenDispatcher(
        4,
        16,
        types.SimpleNamespace(ep_size=2),
        moe_token_dispatcher_type="alltoall",
    )

    assert dispatcher.moe_token_dispatcher_type == "alltoall"
    assert dispatcher.transport_evidence["effective"] == "alltoall"
    assert dispatcher.transport_evidence["silent_fallback"] is False


def test_aligned_alltoall_preserves_duplicate_route_slots(monkeypatch) -> None:
    group = object()
    dispatcher = TokenDispatcher(
        4,
        16,
        types.SimpleNamespace(ep_size=2, ep_group=group),
        moe_token_dispatcher_type="alltoall",
        deepep_align_to_low_latency=True,
    )
    monkeypatch.setattr(
        alltoall.dist,
        "get_rank",
        lambda *, group: 0,
    )

    def fake_all_gather(output, value, *, group):
        assert value.tolist() == [4, 0]
        output.copy_(torch.tensor([4, 0, 0, 0], dtype=output.dtype))

    monkeypatch.setattr(
        alltoall.dist,
        "all_gather_into_tensor",
        fake_all_gather,
    )
    monkeypatch.setattr(
        alltoall._AllToAll,
        "apply",
        lambda value, *_args: value,
    )
    hidden = torch.stack(
        (
            torch.ones(16, dtype=torch.bfloat16),
            torch.full((16,), 2, dtype=torch.bfloat16),
        )
    )
    indices = torch.tensor([[0, 0], [1, 0]], dtype=torch.int64)
    weights = torch.tensor([[0.25, 0.75], [0.4, 0.6]], dtype=torch.float32)

    received, local_ids, received_weights, counts, output_index, all_valid = (
        dispatcher._dispatch_aligned_alltoall(hidden, weights, indices)
    )

    assert all_valid
    assert counts.tolist() == [128, 128]
    assert received.shape == (256, 16)
    assert local_ids.reshape(-1)[:4].tolist() == [0, 0, 1, 0]
    assert received_weights.reshape(-1)[:4].tolist() == pytest.approx(
        [0.25, 0.75, 0.4, 0.6]
    )
    assert torch.count_nonzero(received_weights[4:]) == 0
    assert output_index.tolist() == [[0, 1], [2, 3]]
    torch.testing.assert_close(received[0], hidden[0])
    torch.testing.assert_close(received[1], hidden[0])


def test_aligned_alltoall_forward_matches_ll_slot_semantics(
    monkeypatch,
) -> None:
    _install_fake_ep_gather(monkeypatch)
    group = object()
    monkeypatch.setattr(alltoall.dist, "get_rank", lambda *, group: 0)

    def fake_all_gather(output, value, *, group):
        output.copy_(torch.tensor([4, 0, 0, 0], dtype=output.dtype))

    monkeypatch.setattr(
        alltoall.dist, "all_gather_into_tensor", fake_all_gather
    )
    monkeypatch.setattr(
        alltoall._AllToAll, "apply", lambda value, *_args: value
    )
    dispatcher = TokenDispatcher(
        4,
        16,
        types.SimpleNamespace(ep_size=2, ep_group=group),
        moe_token_dispatcher_type="alltoall",
        deepep_align_to_low_latency=True,
    )
    hidden = torch.stack(
        (
            torch.ones(16, dtype=torch.bfloat16),
            torch.full((16,), 2, dtype=torch.bfloat16),
        )
    )
    indices = torch.tensor([[0, 0], [1, 0]], dtype=torch.int64)
    weights = torch.tensor([[0.25, 0.75], [0.4, 0.6]], dtype=torch.float32)

    dispatched, counts, _ = dispatcher.dispatch(hidden, weights, indices)
    assert counts.tolist() == [128, 128]
    expert_output = dispatched.clone()
    expert_output[:128].mul_(2)
    expert_output[128:].mul_(3)

    actual = dispatcher.combine(expert_output)
    expected = torch.stack(
        (
            torch.full((16,), 2.0, dtype=torch.bfloat16),
            torch.full((16,), 4.8, dtype=torch.bfloat16),
        )
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_aligned_hybridep_preserves_route_slots_and_pad128(monkeypatch) -> None:
    class FakeHybridBuffer:
        def dispatch_with_permute(
            self,
            *,
            hidden,
            routing_map,
            probs,
            num_of_experts_per_rank,
            **_kwargs,
        ):
            assert num_of_experts_per_rank == 2
            assert routing_map.shape == (4, 4)
            assert routing_map.sum(1).eq(1).all()
            assert probs.sum(1).eq(1).all()
            route_to_expert_row = torch.tensor(
                [0, 1, 128, 2], dtype=torch.long
            )
            dispatched = hidden.new_zeros((256, hidden.shape[1]))
            dispatched.index_copy_(0, route_to_expert_row, hidden)
            return (
                dispatched,
                None,
                None,
                torch.tensor([128, 128], dtype=torch.int64),
                (route_to_expert_row,),
            )

        def combine_with_unpermute(self, *, hidden, handle, **_kwargs):
            (route_to_expert_row,) = handle
            return hidden.index_select(0, route_to_expert_row), None

    buffer = FakeHybridBuffer()
    monkeypatch.setattr(hybridep, "_get_buffer", lambda *_args: buffer)
    hidden = torch.stack(
        (
            torch.ones(16, dtype=torch.bfloat16),
            torch.full((16,), 2, dtype=torch.bfloat16),
        )
    )
    indices = torch.tensor([[0, 0], [1, 0]], dtype=torch.int64)
    weights = torch.tensor([[0.25, 0.75], [0.4, 0.6]], dtype=torch.float32)

    result = hybridep.dispatch_routes(
        hidden,
        weights,
        indices,
        num_experts=4,
        num_local_experts=2,
        group=object(),
    )

    assert result.tokens_per_expert.tolist() == [128, 128]
    assert result.hidden.shape == (256, 16)
    assert result.state.source_output_index.tolist() == [[0, 1], [2, 3]]
    source_routes = hybridep.combine_routes(result.hidden, result.state)
    torch.testing.assert_close(source_routes[0], hidden[0])
    torch.testing.assert_close(source_routes[1], hidden[0])
    torch.testing.assert_close(source_routes[2], hidden[1])
    torch.testing.assert_close(source_routes[3], hidden[1])


def test_aligned_hybridep_forward_matches_ll_slot_semantics(
    monkeypatch,
) -> None:
    class FakeHybridBuffer:
        def dispatch_with_permute(
            self, *, hidden, routing_map, probs, **_kwargs
        ):
            assert routing_map.sum(1).eq(1).all()
            assert probs.sum(1).eq(1).all()
            route_to_expert_row = torch.tensor(
                [0, 1, 128, 2], dtype=torch.long
            )
            dispatched = hidden.new_zeros((256, hidden.shape[1]))
            dispatched.index_copy_(0, route_to_expert_row, hidden)
            return (
                dispatched,
                None,
                None,
                torch.tensor([128, 128], dtype=torch.int64),
                (route_to_expert_row,),
            )

        def combine_with_unpermute(self, *, hidden, handle, **_kwargs):
            (route_to_expert_row,) = handle
            return hidden.index_select(0, route_to_expert_row), None

    _install_fake_ep_gather(monkeypatch)
    monkeypatch.setattr(
        hybridep,
        "deep_ep",
        type(
            "FakeDeepEP",
            (),
            {"Buffer": object, "HybridEPBuffer": object},
        ),
    )
    monkeypatch.setenv(
        "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN", "2"
    )
    monkeypatch.setattr(
        hybridep.dist, "get_world_size", lambda *, group: 2
    )
    monkeypatch.setattr(
        hybridep, "_get_buffer", lambda *_args: FakeHybridBuffer()
    )
    dispatcher = TokenDispatcher(
        4,
        16,
        types.SimpleNamespace(ep_size=2, ep_group=object()),
        moe_token_dispatcher_type="hybridep",
        deepep_align_to_low_latency=True,
    )
    hidden = torch.stack(
        (
            torch.ones(16, dtype=torch.bfloat16),
            torch.full((16,), 2, dtype=torch.bfloat16),
        )
    )
    indices = torch.tensor([[0, 0], [1, 0]], dtype=torch.int64)
    weights = torch.tensor([[0.25, 0.75], [0.4, 0.6]], dtype=torch.float32)

    dispatched, counts, _ = dispatcher.dispatch(hidden, weights, indices)
    expert_output = dispatched.clone()
    expert_output[:128].mul_(2)
    expert_output[128:].mul_(3)

    actual = dispatcher.combine(expert_output)
    expected = torch.stack(
        (
            torch.full((16,), 2.0, dtype=torch.bfloat16),
            torch.full((16,), 4.8, dtype=torch.bfloat16),
        )
    )
    assert counts.tolist() == [128, 128]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_deepep_buffer_is_created_lazily_at_dispatch(monkeypatch) -> None:
    import megatron.lite.primitive.modules.dispatcher as dispatcher_module

    class FakeBuffer:
        @staticmethod
        def set_num_sms(_value):
            pass

    group = object()
    sentinel = object()
    monkeypatch.setattr(
        deepep,
        "deep_ep",
        type("FakeDeepEP", (), {"Buffer": FakeBuffer}),
    )
    monkeypatch.setattr(
        deepep,
        "get_buffer",
        lambda actual_group, _hidden_bytes: (
            sentinel if actual_group is group else None
        ),
    )
    dispatcher = TokenDispatcher(
        4,
        16,
        types.SimpleNamespace(ep_size=2, tp_ep_group=group),
        moe_token_dispatcher_type="deepep",
        deepep_align_to_low_latency=True,
    )
    dispatcher._ensure_deepep_buffer(
        torch.zeros(2, 16, dtype=torch.bfloat16)
    )

    assert dispatcher.buffer is sentinel


def _capture_aligned_dispatch_contract(dispatcher, monkeypatch):
    captured = {}

    def fake_aligned(self, hidden, scores, indices, *, source_fixed_topk_valid):
        captured["indices"] = indices
        captured["source_fixed_topk_valid"] = source_fixed_topk_valid
        return hidden, torch.empty(0, dtype=torch.int64), scores

    monkeypatch.setattr(
        dispatcher,
        "_dispatch_low_latency_aligned",
        types.MethodType(fake_aligned, dispatcher),
    )
    return captured


def test_aligned_dispatch_fixed_topk_contract_matches_slime(monkeypatch) -> None:
    dispatcher = TokenDispatcher.__new__(TokenDispatcher)
    dispatcher.capacity_factor = None
    dispatcher.deepep_align_to_low_latency = True
    captured = _capture_aligned_dispatch_contract(dispatcher, monkeypatch)
    hidden = torch.zeros(2, 16, dtype=torch.bfloat16)
    scores = torch.ones(2, 2, dtype=torch.float32)
    indices = torch.tensor([[0, 1], [1, 0]], dtype=torch.int64)

    dispatcher.dispatch(hidden, scores, indices)

    assert captured["source_fixed_topk_valid"] is True
    assert captured["indices"] is indices


def test_aligned_dispatch_masks_routes_like_slime(monkeypatch) -> None:
    dispatcher = TokenDispatcher.__new__(TokenDispatcher)
    dispatcher.capacity_factor = 1.0
    dispatcher.deepep_align_to_low_latency = True
    captured = _capture_aligned_dispatch_contract(dispatcher, monkeypatch)
    hidden = torch.zeros(2, 16, dtype=torch.bfloat16)
    scores = torch.tensor([[1.0, 0.0], [0.5, 0.5]], dtype=torch.float32)
    indices = torch.tensor([[0, 1], [1, 0]], dtype=torch.int64)
    token_mask = torch.tensor([False, True])

    dispatcher.dispatch(
        hidden,
        scores,
        indices,
        router_token_masks=token_mask,
    )

    assert captured["source_fixed_topk_valid"] is False
    assert torch.equal(
        captured["indices"],
        torch.tensor([[0, -1], [-1, -1]], dtype=torch.int64),
    )


def test_aligned_deepep_buffer_matches_mcore_process_wide_reuse(monkeypatch) -> None:
    import megatron.lite.primitive.modules.dispatcher as dispatcher_module

    class FakeConfig:
        def get_nvl_buffer_size_hint(self, hidden_bytes, group_size):
            return hidden_bytes * group_size

        def get_rdma_buffer_size_hint(self, hidden_bytes, group_size):
            return hidden_bytes * group_size * 2

    class FakeBuffer:
        created = 0

        @staticmethod
        def get_dispatch_config(group_size):
            return FakeConfig()

        @staticmethod
        def get_combine_config(group_size):
            return FakeConfig()

        def __init__(self, *, group, num_nvl_bytes, num_rdma_bytes, explicitly_destroy):
            type(self).created += 1
            self.group = group
            self.num_nvl_bytes = num_nvl_bytes
            self.num_rdma_bytes = num_rdma_bytes
            self.explicitly_destroy = explicitly_destroy
            self.runtime = object()

    group = object()
    monkeypatch.setattr(deepep, "deep_ep", type("FakeDeepEP", (), {"Buffer": FakeBuffer}))
    monkeypatch.setattr(deepep.dist, "get_world_size", lambda *, group: 4)
    monkeypatch.setattr(deepep.torch.cuda, "device_count", lambda: 4)
    monkeypatch.setattr(deepep, "_buffer", None)
    layer0_primary = deepep.build_buffer(group, 4096)
    layer0_metadata = deepep.build_buffer(group, 4096)
    layer1_primary = deepep.build_buffer(group, 4096)

    assert layer0_primary is layer0_metadata is layer1_primary
    assert FakeBuffer.created == 1
    assert layer0_primary.num_rdma_bytes == 0

    grown = deepep.build_buffer(group, 8192)
    assert grown is not layer0_primary
    assert FakeBuffer.created == 2


def test_deepep_receive_counts_keep_mcore_cpu_contract(monkeypatch) -> None:
    import megatron.lite.primitive.modules.dispatcher as dispatcher_module

    class FakeBuffer:
        def get_dispatch_layout(self, *_args, **_kwargs):
            return None, None, None, None, None

        def dispatch(self, hidden, **_kwargs):
            return hidden, None, None, [3, 5], (), None

    monkeypatch.setattr(deepep, "get_buffer", lambda *_args: FakeBuffer())
    ctx = types.SimpleNamespace()
    hidden = torch.zeros(2, 16, dtype=torch.bfloat16)
    indices = torch.zeros(2, 1, dtype=torch.int64)
    scores = torch.ones(2, 1, dtype=torch.float32)

    result = deepep.Dispatch.forward(
        ctx, object(), hidden, indices, scores, 2, False, False
    )

    counts = result[3]
    assert counts.device.type == "cpu"
    assert counts.dtype == torch.int64
    assert counts.tolist() == [3, 5]


def test_ll_alignment_preserves_duplicate_slots_and_fp32_gather(monkeypatch) -> None:
    _install_fake_ep_gather(monkeypatch)
    dispatcher = TokenDispatcher(
        num_experts=2,
        hidden_size=16,
        ps=ParallelState(ep_size=1, ep_rank=0),
        moe_token_dispatcher_type="alltoall",
        deepep_align_to_low_latency=True,
    )
    hidden = torch.stack(
        (
            torch.ones(16, dtype=torch.bfloat16),
            torch.full((16,), 2, dtype=torch.bfloat16),
        )
    )
    # Token 0 deliberately selects expert 0 twice.  Ordinary boolean routing
    # maps collapse these two slots; LL semantics must retain both.
    indices = torch.tensor([[0, 0], [1, 0]], dtype=torch.int64)
    weights = torch.tensor([[0.25, 0.75], [0.4, 0.6]], dtype=torch.float32)

    dispatched, tokens_per_expert, _ = dispatcher.dispatch(
        hidden, weights, indices
    )

    assert tokens_per_expert.tolist() == [3, 1]
    torch.testing.assert_close(dispatched[0], hidden[0])
    torch.testing.assert_close(dispatched[1], hidden[0])
    torch.testing.assert_close(dispatched[2], hidden[1])
    torch.testing.assert_close(dispatched[3], hidden[1])

    expert_output = dispatched.clone()
    expert_output[:3].mul_(2)  # expert 0
    expert_output[3:].mul_(3)  # expert 1
    actual = dispatcher.combine(expert_output)
    expected = torch.stack(
        (
            torch.full((16,), 2.0, dtype=torch.bfloat16),
            torch.full((16,), 4.8, dtype=torch.bfloat16),
        )
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_normal_deepep_finish_deduplicates_hash_routes() -> None:
    dispatcher = TokenDispatcher.__new__(TokenDispatcher)
    dispatcher.num_local_experts = 2
    dispatcher.moe_permute_fusion = False
    dispatcher._local_tpe_list = None
    dispatcher._row_id_map = None
    dispatcher._restore_shape = None

    hidden = torch.stack(
        (
            torch.ones(16, dtype=torch.bfloat16),
            torch.full((16,), 2, dtype=torch.bfloat16),
        )
    )
    indices = torch.tensor([[0, 0], [1, 0]], dtype=torch.int64)
    weights = torch.tensor([[0.25, 0.75], [0.4, 0.6]], dtype=torch.float32)

    dispatched, tokens_per_expert, routed_weights = (
        dispatcher._finish_deepep_dispatch(
            hidden,
            indices,
            weights,
            # DeepEP counts top-k slots before duplicate expert IDs are folded.
            [3, 1],
        )
    )

    assert tokens_per_expert.tolist() == [2, 1]
    assert dispatched.shape == (3, 16)
    torch.testing.assert_close(
        routed_weights,
        torch.tensor([1.0, 0.6, 0.4], dtype=torch.float32),
        rtol=0,
        atol=0,
    )
