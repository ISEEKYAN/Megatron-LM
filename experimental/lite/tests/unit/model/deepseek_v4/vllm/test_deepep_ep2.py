from __future__ import annotations

import importlib.util
import os
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.distributed as dist

from megatron.lite.primitive.kernels.vllm_ds4 import (
    DeepEPAdapter,
    DeepEPMode,
    HashRouteAdapter,
)
from megatron.lite.primitive.modules.dispatcher import TokenDispatcher


def test_deepep_adapter_cpu_dispatch_combine_contract() -> None:
    handle = SimpleNamespace(
        low_latency_dispatch=Mock(return_value=("expert_input", "dispatch_handle")),
        low_latency_combine=Mock(return_value=(torch.ones(2, 8), "event")),
    )
    adapter = DeepEPAdapter(handle, object(), DeepEPMode.LOW_LATENCY)
    x = torch.zeros(2, 8)
    ids = torch.zeros(2, 1, dtype=torch.int64)
    weights = torch.ones(2, 1)
    dispatched = adapter.dispatch(x, ids, max_tokens_per_rank=4, num_experts=2)
    assert dispatched == ("expert_input", "dispatch_handle")
    handle.low_latency_dispatch.assert_called_once_with(x, ids, 4, 2)
    combined = adapter.combine(
        torch.zeros(1, 4, 8),
        "dispatch_handle",
        topk_ids=ids,
        topk_weights=weights,
    )
    assert torch.equal(combined[0], torch.ones(2, 8))
    handle.low_latency_combine.assert_called_once()


def _ep2_skip_reason() -> str | None:
    if importlib.util.find_spec("deep_ep") is None:
        return "requires the DeepEP Python package and compiled kernels"
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        return "requires two visible CUDA GPUs"
    if int(os.environ.get("WORLD_SIZE", "1")) != 2:
        return "requires torchrun --standalone --nproc-per-node=2"
    return None


@pytest.mark.gpus(2)
def test_torchrun_ep2_real_low_latency_dispatch_combine() -> None:
    reason = _ep2_skip_reason()
    if reason:
        pytest.skip(reason)

    import deep_ep

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    created_group = not dist.is_initialized()
    if created_group:
        dist.init_process_group("nccl")
    group = dist.group.WORLD
    assert dist.get_world_size(group) == 2

    # DeepEP low-latency kernels compile a fixed supported hidden-size set;
    # 2048 is the smallest production shape.
    tokens, hidden, experts, topk = 4, 2048, 2, 1
    bytes_needed = deep_ep.Buffer.get_low_latency_rdma_size_hint(
        tokens, hidden, 2, experts
    )
    buffer = deep_ep.Buffer(
        group,
        num_rdma_bytes=bytes_needed,
        low_latency_mode=True,
        num_qps_per_rank=1,
        allow_nvlink_for_low_latency_mode=True,
        explicitly_destroy=True,
    )
    try:
        adapter = DeepEPAdapter(buffer, group, DeepEPMode.LOW_LATENCY)
        rank = dist.get_rank(group)
        x = torch.full(
            (tokens, hidden), rank + 1, dtype=torch.bfloat16, device="cuda"
        )
        ids = torch.tensor([[0], [1], [0], [1]], dtype=torch.int64, device="cuda")
        weights = torch.ones(tokens, topk, dtype=torch.float32, device="cuda")

        dispatched = adapter.dispatch(
            x,
            ids,
            max_tokens_per_rank=tokens,
            num_experts=experts,
            use_fp8=False,
            async_finish=False,
        )
        expert_input, _recv_count, dispatch_handle, _event, _hook = dispatched
        combined = adapter.combine(
            expert_input.clone(),
            dispatch_handle,
            topk_ids=ids,
            topk_weights=weights,
            async_finish=False,
        )
        output, _combine_event, _combine_hook = combined
        torch.testing.assert_close(output, x, rtol=0, atol=0)
        dist.barrier(group=group)
    finally:
        buffer.destroy()
        if created_group:
            dist.destroy_process_group()


@pytest.mark.gpus(2)
def test_torchrun_ep2_rl_shape_normal_dispatch_is_memory_safe() -> None:
    reason = _ep2_skip_reason()
    if reason:
        pytest.skip(reason)

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    created_group = not dist.is_initialized()
    if created_group:
        dist.init_process_group("nccl")
    group = dist.group.WORLD
    rank = dist.get_rank(group)
    torch.manual_seed(47 + rank)

    tokens, hidden, experts, topk = 640, 4096, 256, 6
    parallel_state = SimpleNamespace(ep_size=2, tp_ep_group=group)
    dispatcher = TokenDispatcher(
        experts,
        hidden,
        parallel_state,
        moe_token_dispatcher_type="deepep",
        deepep_align_to_low_latency=True,
    )
    hidden_states = torch.randn(
        tokens,
        hidden,
        device="cuda",
        dtype=torch.bfloat16,
    )
    topk_ids = torch.randint(
        0,
        experts,
        (tokens, topk),
        device="cuda",
        dtype=torch.int64,
    )
    topk_weights = torch.softmax(
        torch.randn(tokens, topk, device="cuda", dtype=torch.float32),
        dim=-1,
    )
    compact, counts, probs = dispatcher.dispatch(
        hidden_states,
        topk_weights,
        topk_ids,
    )
    torch.cuda.synchronize()
    assert compact.ndim == 2 and compact.shape[1] == hidden
    assert counts.shape == (experts // 2,)
    assert probs is not None and probs.dtype == torch.float32
    dist.barrier(group=group)
    destroy = getattr(dispatcher.buffer, "destroy", None)
    if callable(destroy):
        destroy()
    if created_group:
        dist.destroy_process_group()


@pytest.mark.gpus(8)
def test_torchrun_dp4_ep2_four_layer_dispatch_is_memory_safe() -> None:
    if importlib.util.find_spec("deep_ep") is None:
        pytest.skip("requires the DeepEP Python package and compiled kernels")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 8:
        pytest.skip("requires eight visible CUDA GPUs")
    if int(os.environ.get("WORLD_SIZE", "1")) != 8:
        pytest.skip("requires torchrun --standalone --nproc-per-node=8")

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    created_group = not dist.is_initialized()
    if created_group:
        dist.init_process_group("nccl")
    rank = dist.get_rank()
    ep_groups = [
        dist.new_group(ranks=[start, start + 1], backend="nccl")
        for start in range(0, 8, 2)
    ]
    ep_group = ep_groups[rank // 2]
    torch.manual_seed(53 + rank)

    hidden, experts, topk = 4096, 256, 6
    parallel_state = SimpleNamespace(ep_size=2, tp_ep_group=ep_group)
    dispatchers = [
        TokenDispatcher(
            experts,
            hidden,
            parallel_state,
            moe_token_dispatcher_type="deepep",
            deepep_align_to_low_latency=True,
        )
        for _ in range(4)
    ]
    # The RL actor executes all four dispatchers on variable-length ~2K-token
    # microbatches, combines through the route-fingerprint handle, and reuses
    # the same DeepEP buffers over many rollout/training cycles.  A one-shot
    # dispatch-only check misses corruption that is first observed on a later
    # route-buffer dispatch, so exercise the complete communication lifecycle.
    token_jitter = (109, 286, 171, 920, 249, 179, 337, 512)
    hash_table = torch.randint(
        0, experts, (129280, topk), device="cuda", dtype=torch.int32
    )
    for iteration in range(12):
        tokens = 2048 + token_jitter[(iteration + rank) % len(token_jitter)]
        hidden_states = torch.randn(
            tokens, hidden, device="cuda", dtype=torch.bfloat16
        )
        hash_logits = torch.randn(
            tokens, experts, device="cuda", dtype=torch.float32
        )
        token_ids = torch.randint(
            0, 129280, (tokens,), device="cuda", dtype=torch.int32
        )
        hash_weights, hash_ids = HashRouteAdapter()(
            hash_logits,
            token_ids,
            hash_table,
            topk=topk,
            renormalize=True,
            routed_scaling_factor=1.5,
        )
        for dispatcher in dispatchers:
            compact, counts, probs = dispatcher.dispatch(
                hidden_states, hash_weights, hash_ids
            )
            torch.cuda.synchronize()
            assert compact.ndim == 2 and compact.shape[1] == hidden
            assert counts.shape == (experts // 2,)
            assert probs is not None and probs.dtype == torch.float32

            combined = dispatcher.combine(compact)
            torch.cuda.synchronize()
            assert combined.shape == hidden_states.shape
            assert torch.isfinite(combined).all()
            del compact, counts, probs, combined
    dist.barrier()
    buffer = dispatchers[0].buffer
    assert all(dispatcher.buffer is buffer for dispatcher in dispatchers)
    destroy = getattr(buffer, "destroy", None)
    if callable(destroy) and getattr(buffer, "explicitly_destroy", False):
        destroy()
    if created_group:
        dist.destroy_process_group()
