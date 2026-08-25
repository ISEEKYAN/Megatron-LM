import inspect

import torch

from megatron.lite.model.deepseek_v4.vllm.primitive.moe.communication import (
    VLLMAlignedNormalDeepEPDispatcher,
    _scatter_deepep_routes_with_padding,
)
from megatron.lite.primitive.parallel import ParallelState

def test_route_alignment_preserves_duplicate_slots_and_fp32_gather(monkeypatch) -> None:
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
    dispatcher = VLLMAlignedNormalDeepEPDispatcher(
        num_experts=2,
        hidden_size=16,
        ps=ParallelState(ep_size=1, ep_rank=0),
        use_deepep=False,
    )
    hidden = torch.stack(
        (
            torch.ones(16, dtype=torch.bfloat16),
            torch.full((16,), 2, dtype=torch.bfloat16),
        )
    )
    indices = torch.tensor([[0, 0], [1, 0]], dtype=torch.int64)
    weights = torch.tensor([[0.25, 0.75], [0.4, 0.6]], dtype=torch.float32)

    dispatched, tokens_per_expert, _ = dispatcher.dispatch(hidden, weights, indices)

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


def test_normal_deepep_uses_one_route_slot_dispatch_and_host_counts() -> None:
    class _Buffer:
        def __init__(self):
            self.dispatch_calls = 0

        @staticmethod
        def get_dispatch_layout(*_args, **_kwargs):
            return None, None, None, None, None

        def dispatch(self, hidden, *, topk_idx, topk_weights, **_kwargs):
            self.dispatch_calls += 1
            assert topk_idx.shape == (4, 1)
            assert torch.equal(topk_idx.reshape(-1), torch.tensor([0, 0, 1, 0]))
            handle = (None, None, None, torch.empty(4, 1), None, None)
            return hidden, topk_idx, topk_weights, [3, 1], handle, None

    dispatcher = VLLMAlignedNormalDeepEPDispatcher(
        num_experts=2,
        hidden_size=16,
        ps=ParallelState(ep_size=1, ep_rank=0),
        use_deepep=False,
    )
    dispatcher.ep_size = 2
    buffer = _Buffer()
    dispatcher._ensure_deepep_buffer = lambda _hidden: buffer
    hidden = torch.stack(
        (
            torch.ones(16, dtype=torch.bfloat16),
            torch.full((16,), 2, dtype=torch.bfloat16),
        )
    )
    indices = torch.tensor([[0, 0], [1, 0]], dtype=torch.int64)
    weights = torch.tensor([[0.25, 0.75], [0.4, 0.6]], dtype=torch.float32)

    dispatched, counts, probs = dispatcher.dispatch(hidden, weights, indices)

    assert buffer.dispatch_calls == 1
    assert counts.device.type == "cpu"
    assert dispatcher._local_tpe_list == [3, 1]
    torch.testing.assert_close(
        dispatched,
        torch.stack((hidden[0], hidden[0], hidden[1], hidden[1])),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        probs,
        torch.tensor([0.25, 0.75, 0.6, 0.4]),
        rtol=0,
        atol=0,
    )


def test_deepep_scatter_fails_closed_without_host_counts() -> None:
    hidden = torch.ones(1, 16, dtype=torch.bfloat16)
    indices = torch.zeros(1, 1, dtype=torch.int64)
    weights = torch.ones(1, 1, dtype=torch.float32)
    non_host_counts = torch.ones(1, dtype=torch.int64, device="meta")
    try:
        _scatter_deepep_routes_with_padding(
            hidden,
            indices,
            weights,
            non_host_counts,
            expected_route_count=1,
        )
    except RuntimeError as exc:
        assert "CPU expert counts" in str(exc)
    else:
        raise AssertionError("non-host expert counts must fail closed")


def test_normal_deepep_hot_path_has_no_tensor_item_or_cpu_roundtrip() -> None:
    source = inspect.getsource(VLLMAlignedNormalDeepEPDispatcher._dispatch_aligned)
    assert ".item()" not in source
    assert ".cpu()" not in source
    assert "received_per_expert_cpu.to(" not in source
