"""Distributed route-contract sanity gate for the three MoE dispatchers.

This checks dispatch/combine against a mathematical route-slot reference.  It
does not claim end-to-end parity with the vLLM DeepEP-LL inference forward.

Run with, for example:
    torchrun --standalone --nproc-per-node=4 dispatcher_alignment_gate.py alltoall

HybridEP also requires an explicit
``NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN`` matching the deployment.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist

from megatron.lite.primitive.modules.dispatcher import TokenDispatcher
from megatron.lite.primitive.parallel import ParallelState


def _expected_output(
    hidden: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_scores: torch.Tensor,
) -> torch.Tensor:
    rows = []
    for token in range(hidden.shape[0]):
        accumulated = torch.zeros(
            hidden.shape[1], dtype=torch.float32, device=hidden.device
        )
        for slot in range(topk_indices.shape[1]):
            expert_factor = int(topk_indices[token, slot]) + 1
            route_output = (
                hidden[token] * expert_factor
            ).to(torch.bfloat16)
            accumulated.add_(
                route_output.float() * topk_scores[token, slot]
            )
        rows.append(accumulated.to(torch.bfloat16))
    return torch.stack(rows)


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in {
        "alltoall",
        "deepep",
        "hybridep",
    }:
        raise SystemExit(
            "usage: dispatcher_alignment_gate.py "
            "alltoall|deepep|hybridep"
        )
    backend = sys.argv[1]
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        "nccl", device_id=torch.device("cuda", local_rank)
    )
    world_size = dist.get_world_size()
    group = dist.new_group(list(range(world_size)), backend="nccl")
    rank = dist.get_rank(group)
    num_local_experts = 2
    num_experts = world_size * num_local_experts
    hidden_size = 16
    parallel_state = ParallelState(
        ep_size=world_size,
        ep_rank=rank,
        ep_group=group,
        tp_ep_group=group,
    )
    dispatcher = TokenDispatcher(
        num_experts,
        hidden_size,
        parallel_state,
        moe_token_dispatcher_type=backend,
        deepep_align_to_low_latency=True,
    )
    hidden = (
        torch.arange(
            2 * hidden_size,
            device="cuda",
            dtype=torch.float32,
        ).reshape(2, hidden_size)
        + rank * 32
        + 1
    ).to(torch.bfloat16)
    local_expert = rank * num_local_experts
    topk_indices = torch.tensor(
        [
            [local_expert, local_expert],
            [
                local_expert + 1,
                ((rank + 1) % world_size) * num_local_experts,
            ],
        ],
        dtype=torch.int64,
        device="cuda",
    )
    topk_scores = torch.tensor(
        [[0.25, 0.75], [0.4, 0.6]],
        dtype=torch.float32,
        device="cuda",
    )

    dist.barrier()
    torch.cuda.synchronize()
    started = time.perf_counter()
    dispatched, tokens_per_expert, _ = dispatcher.dispatch(
        hidden, topk_scores, topk_indices
    )
    counts = [int(value) for value in tokens_per_expert.tolist()]
    offset = 0
    expert_output = dispatched.clone()
    for local_index, count in enumerate(counts):
        expert_output[offset : offset + count].mul_(
            rank * num_local_experts + local_index + 1
        )
        offset += count
    if offset != expert_output.shape[0]:
        raise RuntimeError(
            "tokens_per_expert does not cover dispatched rows: "
            f"{offset} != {expert_output.shape[0]}"
        )
    actual = dispatcher.combine(expert_output)
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - started) * 1000
    expected = _expected_output(hidden, topk_indices, topk_scores)
    bitwise = torch.equal(actual, expected)
    max_abs = float((actual.float() - expected.float()).abs().max().item())
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
        evidence = {
            "backend": backend,
            "world_size": world_size,
            "device": torch.cuda.get_device_name(local_rank),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "route_contract_bitwise": bool(passed.item()),
            "max_abs_all_ranks": float(max_abs_tensor.item()),
            "elapsed_ms_max_rank": float(elapsed_tensor.item()),
            "transport": dispatcher.transport_evidence,
        }
        serialized = json.dumps(evidence, sort_keys=True)
        print(serialized, flush=True)
        evidence_dir = os.environ.get(
            "MLITE_DISPATCHER_EVIDENCE_DIR"
        )
        if evidence_dir:
            output_dir = Path(evidence_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / f"{backend}-ep{world_size}.json").write_text(
                serialized + "\n", encoding="utf-8"
            )
    if not bool(passed.item()):
        raise AssertionError(
            f"{backend} route contract does not match the reference; "
            f"rank={rank} max_abs={max_abs}"
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
