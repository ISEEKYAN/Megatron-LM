"""Benchmark aligned MoE dispatch/combine without expert GEMM."""

from __future__ import annotations

import argparse
import json
import time

import torch
import torch.distributed as dist

from megatron.lite.primitive.modules.dispatcher import TokenDispatcher
from megatron.lite.primitive.parallel import ParallelState


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("backend", choices=("alltoall", "deepep", "hybridep"))
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--topk", type=int, default=6)
    parser.add_argument("--local-experts", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    local_rank = int(__import__("os").environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        "nccl", device_id=torch.device("cuda", local_rank)
    )
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    group = dist.new_group(list(range(world_size)), backend="nccl")
    num_experts = world_size * args.local_experts
    dispatcher = TokenDispatcher(
        num_experts,
        args.hidden,
        ParallelState(
            ep_size=world_size,
            ep_rank=rank,
            ep_group=group,
            tp_ep_group=group,
        ),
        moe_token_dispatcher_type=args.backend,
        deepep_align_to_low_latency=True,
    )
    generator = torch.Generator(device="cuda")
    generator.manual_seed(20260819 + rank)
    hidden = torch.randn(
        args.tokens,
        args.hidden,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    rows = torch.arange(args.tokens, device="cuda").unsqueeze(1)
    slots = torch.arange(args.topk, device="cuda").unsqueeze(0)
    topk_ids = (
        rank * args.local_experts + rows + slots * 3
    ).remainder(num_experts)
    weights = torch.arange(
        1, args.topk + 1, dtype=torch.float32, device="cuda"
    )
    topk_weights = (weights / weights.sum()).unsqueeze(0).expand(
        args.tokens, -1
    ).contiguous()

    def operation() -> torch.Tensor:
        dispatched, _, _ = dispatcher.dispatch(
            hidden, topk_weights, topk_ids
        )
        return dispatcher.combine(dispatched)

    for _ in range(args.warmup):
        output = operation()
    torch.cuda.synchronize()
    dist.barrier()
    started = time.perf_counter()
    for _ in range(args.iterations):
        output = operation()
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - started) * 1000 / args.iterations

    expected = hidden
    bitwise = torch.equal(output, expected)
    max_abs = float((output.float() - expected.float()).abs().max().item())
    elapsed = torch.tensor(elapsed_ms, dtype=torch.float64, device="cuda")
    dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
    if rank == 0:
        print(
            json.dumps(
                {
                    "backend": args.backend,
                    "world_size": world_size,
                    "tokens_per_rank": args.tokens,
                    "hidden": args.hidden,
                    "topk": args.topk,
                    "warmup": args.warmup,
                    "iterations": args.iterations,
                    "dispatch_combine_ms_max_rank": float(elapsed.item()),
                    "identity_bitwise": bitwise,
                    "max_abs": max_abs,
                    "transport": dispatcher.transport_evidence,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
