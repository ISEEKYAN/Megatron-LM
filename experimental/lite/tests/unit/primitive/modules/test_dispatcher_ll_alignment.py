import torch

from megatron.lite.primitive.modules.dispatcher import TokenDispatcher
from megatron.lite.primitive.parallel import ParallelState


def test_normal_deepep_buffer_is_process_shared(monkeypatch) -> None:
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
    monkeypatch.setattr(dispatcher_module, "deep_ep", type("FakeDeepEP", (), {"Buffer": FakeBuffer}))
    monkeypatch.setattr(dispatcher_module.dist, "get_world_size", lambda *, group: 4)
    monkeypatch.setattr(dispatcher_module, "_deepep_buffer", None)

    layer0_primary = dispatcher_module._build_deepep_buffer(group, 4096)
    layer0_metadata = dispatcher_module._build_deepep_buffer(group, 4096)
    layer1_primary = dispatcher_module._build_deepep_buffer(group, 4096)

    assert layer0_primary is layer0_metadata is layer1_primary
    assert FakeBuffer.created == 1


def test_ll_alignment_preserves_duplicate_slots_and_fp32_gather(monkeypatch) -> None:
    import vllm.model_executor.layers.fused_moe.deep_gemm_utils as deep_gemm_utils

    def fake_ep_gather(
        input_tensor, recv_ids, recv_weights, input_index, expert_map, output
    ):
        assert expert_map is None
        output.zero_()
        for token in range(recv_ids.shape[0]):
            accumulator = torch.zeros(input_tensor.shape[1], dtype=torch.float32)
            for slot in range(recv_ids.shape[1]):
                row = int(input_index[token, slot])
                if row >= 0:
                    accumulator += input_tensor[row].float() * recv_weights[token, slot]
            output[token].copy_(accumulator.to(output.dtype))

    monkeypatch.setattr(deep_gemm_utils, "ep_gather", fake_ep_gather)
    dispatcher = TokenDispatcher(
        num_experts=2,
        hidden_size=16,
        ps=ParallelState(ep_size=1, ep_rank=0),
        use_deepep=False,
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
