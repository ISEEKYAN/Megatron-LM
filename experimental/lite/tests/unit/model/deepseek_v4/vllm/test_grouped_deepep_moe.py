from __future__ import annotations

import importlib.util
import os
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.distributed as dist

from megatron.lite.primitive.kernels import vllm_ds4
from megatron.lite.primitive.kernels.vllm_ds4 import (
    GroupedDeepGemmExpertsAdapter,
    GroupedMoEKernelBuilderAdapter,
)


def test_grouped_deepgemm_adapter_cpu_contract(monkeypatch) -> None:
    def pack(weights):
        weights = tuple(weights)
        return SimpleNamespace(
            qweight=torch.stack(
                [weight.detach().to(torch.float8_e4m3fn) for weight in weights]
            ),
            scales=torch.ones(len(weights), 1, 1),
        )

    apply = Mock(return_value=torch.ones(2, 128, dtype=torch.bfloat16))
    monkeypatch.setattr(
        vllm_ds4,
        "_symbol",
        lambda _module, name: (
            SimpleNamespace(SILU="silu") if name == "MoEActivation" else None
        ),
    )
    adapter = GroupedDeepGemmExpertsAdapter(pack_grouped_weight=pack)
    w13 = [torch.nn.Parameter(torch.ones(256, 128, dtype=torch.bfloat16))]
    w2 = [torch.nn.Parameter(torch.ones(128, 128, dtype=torch.bfloat16))]
    hidden = torch.zeros(2, 128, dtype=torch.bfloat16)
    weights = torch.ones(2, 1, dtype=torch.float32)
    ids = torch.zeros(2, 1, dtype=torch.int64)
    expert_map = torch.tensor([0], dtype=torch.int32)

    def build(packed):
        experts = type("BatchedDeepGemmExperts", (), {})()
        experts.w1_scale = packed.w13_scale
        experts.w2_scale = packed.w2_scale
        return SimpleNamespace(
            fused_experts=experts,
            prepare_finalize=type("DeepEPLLPrepareAndFinalize", (), {})(),
            apply=apply,
        )

    output = adapter(
        hidden,
        w13,
        w2,
        weights,
        ids,
        build_kernel=build,
        global_num_experts=1,
        expert_map=expert_map,
    )

    assert torch.equal(output, torch.ones_like(hidden))
    kwargs = apply.call_args.kwargs
    assert kwargs["w1"].shape == (1, 256, 128)
    assert kwargs["w2"].shape == (1, 128, 128)
    assert kwargs["w1"].dtype == torch.float8_e4m3fn
    assert kwargs["topk_ids"].dtype == torch.int64
    assert kwargs["expert_map"].dtype == torch.int32


def test_grouped_deepgemm_adapter_rejects_nonofficial_kernel() -> None:
    adapter = GroupedDeepGemmExpertsAdapter(
        pack_grouped_weight=lambda weights: SimpleNamespace(
            qweight=torch.stack(
                [weight.detach().to(torch.float8_e4m3fn) for weight in weights]
            ),
            scales=torch.ones(len(tuple(weights)), 1, 1),
        )
    )
    weight = torch.nn.Parameter(torch.ones(128, 128, dtype=torch.bfloat16))
    with pytest.raises(RuntimeError, match="BatchedDeepGemmExperts"):
        adapter(
            torch.zeros(1, 128, dtype=torch.bfloat16),
            [weight],
            [weight],
            torch.ones(1, 1),
            torch.zeros(1, 1, dtype=torch.int64),
            build_kernel=lambda _packed: SimpleNamespace(
                fused_experts=object(), prepare_finalize=object()
            ),
            global_num_experts=1,
            expert_map=torch.tensor([0], dtype=torch.int32),
        )


def _ep2_skip_reason() -> str | None:
    required = ("deep_ep", "deep_gemm", "vllm")
    missing = [name for name in required if importlib.util.find_spec(name) is None]
    if missing:
        return f"requires compiled packages: {', '.join(missing)}"
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        return "requires two visible CUDA GPUs"
    if int(os.environ.get("WORLD_SIZE", "1")) != 2:
        return "requires torchrun --standalone --nproc-per-node=2"
    return None


@pytest.mark.gpus(2)
def test_torchrun_ep2_grouped_deepgemm_matches_official_kernel_bitwise() -> None:
    reason = _ep2_skip_reason()
    if reason:
        pytest.skip(reason)

    import deep_ep
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    created_group = not dist.is_initialized()
    if created_group:
        dist.init_process_group("nccl")
    group = dist.group.WORLD
    rank = dist.get_rank(group)

    if (
        torch.cuda.get_device_capability()[0] >= 10
        and os.environ.get("VLLM_BATCH_INVARIANT_KERNEL_LIB")
    ):
        # The SM100 batch-invariant kernel's scheduler
        # requires the real 2048-wide expert intermediate rather than the old
        # 128-wide unit fixture. Keep the Hopper fixture unchanged.
        tokens, hidden, intermediate, experts, topk = 4, 4096, 2048, 2, 1
    else:
        tokens, hidden, intermediate, experts, topk = 4, 2048, 128, 2, 1
    max_tokens = 8
    bytes_needed = deep_ep.Buffer.get_low_latency_rdma_size_hint(
        max_tokens, hidden, 2, experts
    )
    buffer = deep_ep.Buffer(
        group,
        num_rdma_bytes=bytes_needed,
        low_latency_mode=True,
        num_qps_per_rank=1,
        allow_nvlink_for_low_latency_mode=True,
        explicitly_destroy=True,
    )

    local_experts = experts // 2
    torch.manual_seed(17 + rank)
    w13 = [
        torch.nn.Parameter(
            torch.randn(
                2 * intermediate, hidden, device="cuda", dtype=torch.bfloat16
            )
            / 100
        )
        for _ in range(local_experts)
    ]
    w2 = [
        torch.nn.Parameter(
            torch.randn(
                hidden, intermediate, device="cuda", dtype=torch.bfloat16
            )
            / 100
        )
        for _ in range(local_experts)
    ]
    hidden_states = torch.randn(
        tokens, hidden, device="cuda", dtype=torch.bfloat16
    )
    topk_ids = torch.tensor([[0], [1], [0], [1]], device="cuda", dtype=torch.int64)
    topk_weights = torch.ones(tokens, topk, device="cuda", dtype=torch.float32)
    expert_map = torch.full((experts,), -1, device="cuda", dtype=torch.int32)
    expert_map[rank] = 0

    build = GroupedMoEKernelBuilderAdapter(
        buffer,
        device=torch.device("cuda", local_rank),
        num_experts=experts,
        num_local_experts=local_experts,
        experts_per_token=topk,
        hidden_dim=hidden,
        intermediate_size=intermediate,
        max_tokens_per_rank=max_tokens,
        num_dispatchers=2,
    )

    adapter = GroupedDeepGemmExpertsAdapter()
    try:
        with set_current_vllm_config(VllmConfig()):
            candidate = adapter(
                hidden_states,
                w13,
                w2,
                topk_weights,
                topk_ids,
                build_kernel=build,
                global_num_experts=experts,
                expert_map=expert_map,
            )
            packed = adapter.pack(w13, w2)
            reference = build(packed).apply(
                hidden_states=hidden_states,
                w1=packed.w13,
                w2=packed.w2,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                activation=MoEActivation.SILU,
                global_num_experts=experts,
                expert_map=expert_map,
                apply_router_weight_on_input=False,
            )
        torch.testing.assert_close(candidate, reference, rtol=0, atol=0)
        dist.barrier(group=group)
    finally:
        buffer.destroy()
        if created_group:
            dist.destroy_process_group()
