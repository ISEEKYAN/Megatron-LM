"""Compare mLite training MoE forwards with a vLLM DeepEP-LL golden.

The golden path is the official vLLM ``DeepEPLLPrepareAndFinalize`` plus
``BatchedDeepGemmExperts`` forward. Candidate paths retain mLite's BF16 master
parameters and BF16 backward contract, but their visible expert forward is the
same vLLM/DeepGEMM block-FP8 computation.

Run one golden first, then each candidate with the same world size:

    torchrun ... vllm_training_moe_gate.py golden --golden-dir /path
    torchrun ... vllm_training_moe_gate.py alltoall --golden-dir /path
    torchrun ... vllm_training_moe_gate.py deepep --golden-dir /path
    torchrun ... vllm_training_moe_gate.py hybridep --golden-dir /path
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist

from megatron.lite.model.deepseek_v4.config import DeepseekV4Config
from megatron.lite.model.deepseek_v4.vllm.moe import DeepseekV4MoE
from megatron.lite.primitive.kernels.vllm_ds4 import (
    GroupedDeepGemmExpertsAdapter,
    GroupedMoEKernelBuilderAdapter,
)
from megatron.lite.primitive.parallel import ParallelState


_HIDDEN = 4096
_INTERMEDIATE = 2048
_LOCAL_EXPERTS = 2
_TOKENS_PER_RANK = 128
_TOPK = 6
_SEED = 20260819


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        choices=(
            "golden",
            "golden-bf16-dispatch",
            "alltoall",
            "deepep",
            "hybridep",
        ),
    )
    parser.add_argument(
        "--reference-dispatch",
        choices=("fp8", "bf16"),
        default="fp8",
    )
    parser.add_argument("--golden-dir", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    return parser.parse_args()


def _fixed_capsule(
    rank: int,
    world_size: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    tuple[torch.nn.Parameter, ...],
    tuple[torch.nn.Parameter, ...],
]:
    torch.manual_seed(_SEED + rank)
    num_experts = world_size * _LOCAL_EXPERTS
    hidden = (
        torch.randn(
            _TOKENS_PER_RANK,
            _HIDDEN,
            dtype=torch.bfloat16,
            device="cuda",
        )
        / 16
    ).requires_grad_(True)
    token_rows = torch.arange(
        _TOKENS_PER_RANK, dtype=torch.int64, device="cuda"
    ).unsqueeze(1)
    slots = torch.arange(_TOPK, dtype=torch.int64, device="cuda").unsqueeze(0)
    topk_ids = (
        rank * _LOCAL_EXPERTS + token_rows + slots * 3
    ).remainder(num_experts).contiguous()
    raw_weights = (
        torch.arange(
            1,
            _TOPK + 1,
            dtype=torch.float32,
            device="cuda",
        )
        .unsqueeze(0)
        .expand(_TOKENS_PER_RANK, -1)
    )
    topk_weights = (
        raw_weights / raw_weights.sum(dim=1, keepdim=True)
    ).contiguous()
    w13 = tuple(
        torch.nn.Parameter(
            torch.randn(
                2 * _INTERMEDIATE,
                _HIDDEN,
                dtype=torch.bfloat16,
                device="cuda",
            )
            / 128
        )
        for _ in range(_LOCAL_EXPERTS)
    )
    w2 = tuple(
        torch.nn.Parameter(
            torch.randn(
                _HIDDEN,
                _INTERMEDIATE,
                dtype=torch.bfloat16,
                device="cuda",
            )
            / 128
        )
        for _ in range(_LOCAL_EXPERTS)
    )
    return hidden, topk_weights, topk_ids, w13, w2


def _time_forward(
    operation,
    *,
    warmup: int,
    iterations: int,
) -> tuple[torch.Tensor, float]:
    output = None
    for _ in range(warmup):
        output = operation()
    torch.cuda.synchronize()
    dist.barrier()
    started = time.perf_counter()
    for _ in range(iterations):
        output = operation()
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - started) * 1000 / iterations
    if output is None:
        raise RuntimeError("forward benchmark produced no output")
    return output, elapsed_ms


def _golden_forward(
    hidden: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    w13: tuple[torch.nn.Parameter, ...],
    w2: tuple[torch.nn.Parameter, ...],
    *,
    group: dist.ProcessGroup,
    rank: int,
    world_size: int,
    use_fp8_dispatch: bool,
):
    import deep_ep
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation

    num_experts = world_size * _LOCAL_EXPERTS
    bytes_needed = deep_ep.Buffer.get_low_latency_rdma_size_hint(
        _TOKENS_PER_RANK,
        _HIDDEN,
        world_size,
        num_experts,
    )
    buffer = deep_ep.Buffer(
        group,
        num_rdma_bytes=bytes_needed,
        low_latency_mode=True,
        num_qps_per_rank=_LOCAL_EXPERTS,
        allow_nvlink_for_low_latency_mode=True,
        explicitly_destroy=True,
    )
    expert_map = torch.full(
        (num_experts,), -1, dtype=torch.int32, device="cuda"
    )
    local_start = rank * _LOCAL_EXPERTS
    expert_map[local_start : local_start + _LOCAL_EXPERTS] = torch.arange(
        _LOCAL_EXPERTS, dtype=torch.int32, device="cuda"
    )
    adapter = GroupedDeepGemmExpertsAdapter()
    packed = adapter.pack(w13, w2)
    build = GroupedMoEKernelBuilderAdapter(
        buffer,
        device=hidden.device,
        num_experts=num_experts,
        num_local_experts=_LOCAL_EXPERTS,
        experts_per_token=_TOPK,
        hidden_dim=_HIDDEN,
        intermediate_size=_INTERMEDIATE,
        max_tokens_per_rank=_TOKENS_PER_RANK,
        num_dispatchers=2,
        # BatchedDeepGemm allocates this capacity once per dispatcher. Across
        # both dispatchers, a local expert must hold one route per source token
        # from every EP rank.
        max_tokens_per_dispatcher_expert=(
            _TOKENS_PER_RANK * world_size // 2
        ),
        use_fp8_dispatch=use_fp8_dispatch,
    )
    kernel = build(packed)

    def operation() -> torch.Tensor:
        with torch.no_grad():
            return kernel.apply(
                hidden_states=hidden.detach(),
                w1=packed.w13,
                w2=packed.w2,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                activation=MoEActivation.SILU,
                global_num_experts=num_experts,
                expert_map=expert_map,
                apply_router_weight_on_input=False,
            )

    return operation, buffer


def _candidate_forward(
    backend: str,
    hidden: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    w13: tuple[torch.nn.Parameter, ...],
    w2: tuple[torch.nn.Parameter, ...],
    *,
    group: dist.ProcessGroup,
    rank: int,
    world_size: int,
):
    parallel_state = ParallelState(
        ep_size=world_size,
        ep_rank=rank,
        ep_group=group,
        tp_ep_group=group,
    )
    moe = DeepseekV4MoE(
        DeepseekV4Config(
            hidden_size=_HIDDEN,
            moe_intermediate_size=_INTERMEDIATE,
            n_routed_experts=world_size * _LOCAL_EXPERTS,
            n_shared_experts=0,
            num_experts_per_tok=_TOPK,
            num_hash_layers=0,
        ),
        parallel_state,
        layer_idx=0,
        selected_stages=frozenset(("router_moe", "deepep")),
        moe_token_dispatcher_type=backend,
    )
    moe.experts.w13 = torch.nn.ParameterList(w13)
    moe.experts.w2 = torch.nn.ParameterList(w2)

    def operation() -> torch.Tensor:
        with torch.no_grad():
            return moe.forward_with_fixed_routes(
                hidden.detach(),
                topk_weights,
                topk_ids,
            )

    return operation, moe.dispatcher


def main() -> None:
    args = _arguments()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        "nccl", device_id=torch.device("cuda", local_rank)
    )
    world_size = dist.get_world_size()
    group = dist.new_group(list(range(world_size)), backend="nccl")
    rank = dist.get_rank(group)
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.utils.deep_gemm import DeepGemmQuantScaleFMT

    DeepGemmQuantScaleFMT.init_oracle_cache()
    hidden, topk_weights, topk_ids, w13, w2 = _fixed_capsule(
        rank, world_size
    )
    args.golden_dir.mkdir(parents=True, exist_ok=True)
    reference_dispatch = (
        "bf16" if args.mode == "golden-bf16-dispatch"
        else args.reference_dispatch
    )
    golden_file = args.golden_dir / (
        f"vllm-deepep-ll-{reference_dispatch}-dispatch-"
        f"ep{world_size}-rank{rank}.pt"
    )
    legacy_fp8_golden = args.golden_dir / (
        f"vllm-deepep-ll-ep{world_size}-rank{rank}.pt"
    )
    with set_current_vllm_config(VllmConfig()):
        if args.mode in ("golden", "golden-bf16-dispatch"):
            operation, buffer = _golden_forward(
                hidden,
                topk_weights,
                topk_ids,
                w13,
                w2,
                group=group,
                rank=rank,
                world_size=world_size,
                use_fp8_dispatch=(reference_dispatch == "fp8"),
            )
            output, elapsed_ms = _time_forward(
                operation,
                warmup=args.warmup,
                iterations=args.iterations,
            )
            torch.save(output.detach().cpu(), golden_file)
            transport = {
                "effective": "vllm-deepep-low-latency",
                "dispatch_dtype": reference_dispatch,
            }
            buffer.destroy()
            bitwise = True
            max_abs = 0.0
        else:
            if (
                reference_dispatch == "fp8"
                and not golden_file.is_file()
                and legacy_fp8_golden.is_file()
            ):
                golden_file = legacy_fp8_golden
            if not golden_file.is_file():
                raise FileNotFoundError(
                    f"missing vLLM golden for rank {rank}: {golden_file}"
                )
            operation, dispatcher = _candidate_forward(
                args.mode,
                hidden,
                topk_weights,
                topk_ids,
                w13,
                w2,
                group=group,
                rank=rank,
                world_size=world_size,
            )
            output, elapsed_ms = _time_forward(
                operation,
                warmup=args.warmup,
                iterations=args.iterations,
            )
            golden = torch.load(
                golden_file, map_location=output.device, weights_only=True
            )
            bitwise = torch.equal(output.detach(), golden)
            max_abs = float(
                (output.detach().float() - golden.float()).abs().max().item()
            )
            transport = dispatcher.transport_evidence

    passed = torch.tensor([int(bitwise)], dtype=torch.int32, device="cuda")
    max_abs_tensor = torch.tensor(
        [max_abs], dtype=torch.float32, device="cuda"
    )
    elapsed_tensor = torch.tensor(
        [elapsed_ms], dtype=torch.float64, device="cuda"
    )
    dist.all_reduce(passed, op=dist.ReduceOp.MIN)
    dist.all_reduce(max_abs_tensor, op=dist.ReduceOp.MAX)
    dist.all_reduce(elapsed_tensor, op=dist.ReduceOp.MAX)
    if rank == 0:
        result = {
            "mode": args.mode,
            "reference": (
                "vllm-deepep-low-latency-"
                f"{reference_dispatch}-dispatch-fp8-experts"
            ),
            "world_size": world_size,
            "device": torch.cuda.get_device_name(local_rank),
            "tokens_per_rank": _TOKENS_PER_RANK,
            "hidden": _HIDDEN,
            "intermediate": _INTERMEDIATE,
            "experts": world_size * _LOCAL_EXPERTS,
            "topk": _TOPK,
            "warmup": args.warmup,
            "iterations": args.iterations,
            "candidate_vs_vllm_bitwise": bool(passed.item()),
            "max_abs_all_ranks": float(max_abs_tensor.item()),
            "forward_ms_max_rank": float(elapsed_tensor.item()),
            "transport": transport,
        }
        serialized = json.dumps(result, sort_keys=True)
        print(serialized, flush=True)
        (args.golden_dir / f"{args.mode}-fp8-moe-ep{world_size}.json").write_text(
            serialized + "\n", encoding="utf-8"
        )
    if not bool(passed.item()):
        raise AssertionError(
            f"{args.mode} candidate FP8 MoE forward differs from vLLM "
            f"DeepEP-LL; rank={rank} max_abs={max_abs}"
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
