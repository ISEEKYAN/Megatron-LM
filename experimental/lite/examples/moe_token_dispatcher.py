# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Select a MoE token-dispatch backend with ``moe_token_dispatcher_type``.

The option takes one of ``alltoall``, ``deepep`` or ``hybridep``. ``alltoall`` needs no
optional dependency; ``deepep`` and ``hybridep`` both come from DeepEP, with ``hybridep``
requiring the branch that exports ``HybridEPBuffer``. A backend that is asked for but not
installed raises rather than falling back, so a missing dependency cannot be mistaken for
a backend that was simply not enabled.

Run under torchrun, one rank per expert-parallel rank::

    torchrun --nproc_per_node=8 examples/moe_token_dispatcher.py --dispatcher-type hybridep

Pass ``--compare-with alltoall`` to additionally check the chosen backend against the
AllToAll reference on identical inputs -- the check this example is derived from produces
bitwise-equal forward and backward results on 8 ranks.

Two HybridEP-specific notes, both learned the hard way on H100:

* ``hidden_size`` must be a multiple of 512. HybridEP JIT-specialises its kernel on the
  hidden dimension and static-asserts that a token's scaling-factor row is a multiple of
  16B (one fp32 scale per 128 elements). Production widths clear this -- DeepSeek-V4 7168,
  Qwen3-MoE 4096, GLM 5120 -- but small toy widths do not, and the failure surfaces as an
  nvcc static assertion rather than a Python error.
* ``CUDA_HOME`` must be set in the environment; HybridEP's C++ runtime resolves against it
  and raises if it is missing.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist

_EXPERIMENTAL_LITE_ROOT = Path(__file__).resolve().parents[1]
if str(_EXPERIMENTAL_LITE_ROOT) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTAL_LITE_ROOT))

from megatron.lite.primitive.modules.dispatcher import TokenDispatcher  # noqa: E402
from megatron.lite.primitive.parallel import init_parallel  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dispatcher-type", default="alltoall", choices=["alltoall", "deepep", "hybridep"]
    )
    p.add_argument("--compare-with", default=None, choices=["alltoall", "deepep", "hybridep"])
    p.add_argument("--num-experts", type=int, default=16)
    p.add_argument("--hidden-size", type=int, default=1024, help="multiple of 512 for hybridep")
    p.add_argument("--topk", type=int, default=2)
    p.add_argument("--num-tokens", type=int, default=256)
    return p.parse_args()


def apply_experts(dispatched, tokens_per_expert, probs, num_local_experts, ep_rank):
    """Stand in for the expert MLPs with a permutation-equivariant op.

    Each row's result depends only on its expert and its routing probability, never on
    where the backend placed it, so two backends' combined outputs are comparable even
    though their permuted layouts differ.
    """
    local_ids = torch.repeat_interleave(
        torch.arange(num_local_experts, device=dispatched.device), tokens_per_expert
    )
    global_ids = ep_rank * num_local_experts + local_ids
    weight = (global_ids.to(dispatched.dtype) + 1.0).unsqueeze(1) * 0.01
    return dispatched * weight * probs.to(dispatched.dtype).unsqueeze(1)


def run(dispatcher_type, args, ps, hidden, topk_scores, topk_indices, ep_rank):
    dispatcher = TokenDispatcher(
        num_experts=args.num_experts,
        hidden_size=args.hidden_size,
        ps=ps,
        moe_token_dispatcher_type=dispatcher_type,
    )
    x = hidden.clone().detach().requires_grad_(True)
    dispatched, tokens_per_expert, probs = dispatcher.dispatch(x, topk_scores, topk_indices)
    expert_out = apply_experts(
        dispatched, tokens_per_expert, probs, dispatcher.num_local_experts, ep_rank
    )
    combined = dispatcher.combine(expert_out)
    combined.sum().backward()
    return combined.detach(), x.grad.detach()


def main() -> int:
    args = parse_args()
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")

    ep_size = dist.get_world_size()
    ps = init_parallel(SimpleNamespace(tp=1, ep=ep_size, cp=1, pp=1, etp=None))
    ep_rank = dist.get_rank(group=ps.ep_group)

    torch.manual_seed(1234 + rank)
    hidden = torch.randn(args.num_tokens, args.hidden_size, device="cuda", dtype=torch.bfloat16)
    logits = torch.randn(args.num_tokens, args.num_experts, device="cuda", dtype=torch.float32)
    topk_scores, topk_indices = torch.topk(logits.softmax(dim=-1), args.topk, dim=-1)

    out, grad = run(args.dispatcher_type, args, ps, hidden, topk_scores, topk_indices, ep_rank)
    if rank == 0:
        print(f"{args.dispatcher_type}: combined={tuple(out.shape)} ep_size={ep_size}", flush=True)

    status = 0
    if args.compare_with:
        ref, ref_grad = run(args.compare_with, args, ps, hidden, topk_scores, topk_indices, ep_rank)
        fwd = (out.float() - ref.float()).abs().max().item()
        bwd = (grad.float() - ref_grad.float()).abs().max().item()
        agreed = torch.tensor([1 if fwd == 0.0 and bwd == 0.0 else 0], device="cuda")
        dist.all_reduce(agreed, op=dist.ReduceOp.MIN)
        print(f"[rank{rank}] vs {args.compare_with}: fwd={fwd:.3e} bwd={bwd:.3e}", flush=True)
        dist.barrier()
        if rank == 0:
            print("bitwise-equal" if int(agreed.item()) else "MISMATCH", flush=True)
        status = 0 if int(agreed.item()) else 1

    dist.destroy_process_group()
    return status


if __name__ == "__main__":
    raise SystemExit(main())
