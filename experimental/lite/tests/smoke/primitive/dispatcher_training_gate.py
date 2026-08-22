"""Distributed forward/backward gate for regular mLite MoE dispatchers."""

from __future__ import annotations

import argparse
import json
import os

import torch
import torch.distributed as dist

from megatron.lite.primitive.modules.dispatcher import TokenDispatcher
from megatron.lite.primitive.parallel import ParallelState


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "backend", choices=("alltoall", "deepep", "hybridep")
    )
    return parser.parse_args()


def _expert_forward(
    dispatched: torch.Tensor,
    probs: torch.Tensor,
    counts: torch.Tensor,
    *,
    rank: int,
    local_experts: int,
) -> torch.Tensor:
    chunks = []
    offset = 0
    for local_expert, count in enumerate(counts.tolist()):
        end = offset + int(count)
        factor = rank * local_experts + local_expert + 1
        chunks.append(
            (
                dispatched[offset:end].float()
                * probs[offset:end].float().unsqueeze(-1)
                * factor
            ).to(torch.bfloat16)
        )
        offset = end
    if offset != dispatched.shape[0]:
        raise RuntimeError("expert counts do not cover dispatched rows")
    return torch.cat(chunks, dim=0)


def _reference(
    hidden: torch.Tensor,
    scores: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    routes = []
    for slot in range(indices.shape[1]):
        factors = indices[:, slot].float().add(1).unsqueeze(-1)
        routes.append(
            (
                hidden.float()
                * scores[:, slot].float().unsqueeze(-1)
                * factors
            ).to(torch.bfloat16)
        )
    output = routes[0]
    for route in routes[1:]:
        output = output + route
    return output


def main() -> None:
    args = _arguments()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        "nccl", device_id=torch.device("cuda", local_rank)
    )
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    group = dist.new_group(list(range(world_size)), backend="nccl")
    local_experts = 2
    # DeepEP's internode kernels require a 512-byte-aligned hidden row.
    hidden_size = 256
    dispatcher = TokenDispatcher(
        world_size * local_experts,
        hidden_size,
        ParallelState(
            ep_size=world_size,
            ep_rank=rank,
            ep_group=group,
            tp_ep_group=group,
        ),
        moe_token_dispatcher_type=args.backend,
    )

    generator = torch.Generator(device="cuda")
    generator.manual_seed(20260823 + rank)
    hidden = torch.randn(
        8,
        hidden_size,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
        requires_grad=True,
    )
    scores = torch.tensor(
        [[0.25, 0.75], [0.4, 0.6]],
        device="cuda",
        dtype=torch.float32,
    ).repeat(4, 1)
    scores.requires_grad_(True)
    optimizer = torch.optim.SGD([hidden, scores], lr=1e-4)
    rows = torch.arange(8, device="cuda")
    indices = torch.stack(
        (
            (rank * local_experts + rows).remainder(
                world_size * local_experts
            ),
            (
                (rank + 1) * local_experts
                + rows
                + 1
            ).remainder(world_size * local_experts),
        ),
        dim=1,
    )

    dispatched, counts, probs = dispatcher.dispatch(
        hidden, scores, indices
    )
    dispatcher.wait_dispatch_event()
    if probs is None:
        raise RuntimeError(f"{args.backend} did not return router weights")
    actual = dispatcher.combine(
        _expert_forward(
            dispatched,
            probs,
            counts,
            rank=rank,
            local_experts=local_experts,
        )
    )
    actual.float().sum().backward()
    actual_hidden_grad = hidden.grad.detach().clone()
    actual_score_grad = scores.grad.detach().clone()
    if not (
        torch.isfinite(actual).all()
        and torch.isfinite(actual_hidden_grad).all()
        and torch.isfinite(actual_score_grad).all()
    ):
        raise AssertionError(f"{args.backend} produced non-finite training values")

    reference_hidden = hidden.detach().clone().requires_grad_(True)
    reference_scores = scores.detach().clone().requires_grad_(True)
    expected = _reference(reference_hidden, reference_scores, indices)
    expected.float().sum().backward()

    output_max_abs = float(
        (actual.float() - expected.float()).abs().max().item()
    )
    hidden_grad_max_abs = float(
        (
            actual_hidden_grad.float()
            - reference_hidden.grad.float()
        )
        .abs()
        .max()
        .item()
    )
    score_grad_max_abs = float(
        (actual_score_grad - reference_scores.grad)
        .abs()
        .max()
        .item()
    )
    metrics = torch.tensor(
        [output_max_abs, hidden_grad_max_abs, score_grad_max_abs],
        device="cuda",
        dtype=torch.float64,
    )
    dist.all_reduce(metrics, op=dist.ReduceOp.MAX)
    if metrics[0] > 0 or metrics[1] > 0 or metrics[2] > 1e-3:
        raise AssertionError(
            f"{args.backend} regular MoE gate failed: "
            f"output={metrics[0].item()} "
            f"hidden_grad={metrics[1].item()} "
            f"score_grad={metrics[2].item()}"
        )
    optimizer.step()
    if not (torch.isfinite(hidden).all() and torch.isfinite(scores).all()):
        raise AssertionError(f"{args.backend} optimizer step became non-finite")
    if rank == 0:
        print(
            json.dumps(
                {
                    "backend": args.backend,
                    "world_size": world_size,
                    "output_bitwise": True,
                    "hidden_grad_bitwise": True,
                    "score_grad_max_abs": float(metrics[2].item()),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
