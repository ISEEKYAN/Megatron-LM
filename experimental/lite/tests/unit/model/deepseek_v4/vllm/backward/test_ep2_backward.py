"""Two-rank ordered-adjoint gate for the DS4 whole-MoE bridge."""

from __future__ import annotations

import os

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F

from megatron.lite.model.deepseek_v4.vllm.primitive.moe import deep_ep_moe


def _ready() -> bool:
    return torch.cuda.is_available() and int(os.environ.get("WORLD_SIZE", "1")) == 2


@pytest.mark.gpus(2)
def test_ep2_hidden_probability_and_local_weight_adjoint() -> None:
    if not _ready():
        pytest.skip("requires torchrun with two CUDA ranks")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    created = not dist.is_initialized()
    if created:
        dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = torch.device("cuda", local_rank)
    tokens, hidden_size, intermediate = 2 + rank, 8, 5
    torch.manual_seed(20260813 + rank)
    hidden = torch.randn(tokens, hidden_size, device=device, requires_grad=True)
    probs = torch.rand(tokens, 2, device=device, requires_grad=True)
    ids = torch.stack(
        (torch.zeros(tokens, device=device), torch.ones(tokens, device=device)), dim=-1
    ).long()
    w13 = torch.randn(2 * intermediate, hidden_size, device=device, requires_grad=True)
    w2 = torch.randn(hidden_size, intermediate, device=device, requires_grad=True)

    def visible(hidden_, probs_, ids_, local_w13, local_w2):
        gathered_w13 = [torch.empty_like(local_w13) for _ in range(2)]
        gathered_w2 = [torch.empty_like(local_w2) for _ in range(2)]
        dist.all_gather(gathered_w13, local_w13.detach())
        dist.all_gather(gathered_w2, local_w2.detach())
        output = torch.zeros_like(hidden_)
        for expert in range(2):
            gate, up = F.linear(hidden_.float(), gathered_w13[expert].float()).chunk(2, -1)
            expert_out = F.linear(F.silu(gate) * up, gathered_w2[expert].float())
            slot = (ids_ == expert).nonzero(as_tuple=False)[:, 1]
            output = output + expert_out.to(output.dtype) * probs_[
                torch.arange(tokens, device=device), slot
            ].unsqueeze(-1)
        return output

    output = deep_ep_moe(
        visible,
        hidden,
        probs,
        ids,
        (w13,),
        (w2,),
        group=dist.group.WORLD,
        global_expert_start=rank,
    )
    grad = torch.arange(output.numel(), device=device).reshape_as(output).float() / 17
    output.backward(grad.to(output.dtype))
    for tensor in (hidden.grad, probs.grad, w13.grad, w2.grad):
        assert tensor is not None and torch.isfinite(tensor).all()
    # Same input must produce bitwise deterministic gradients.
    first = tuple(tensor.grad.detach().clone() for tensor in (hidden, probs, w13, w2))
    for tensor in (hidden, probs, w13, w2):
        tensor.grad = None
    output = deep_ep_moe(
        visible, hidden, probs, ids, (w13,), (w2,),
        group=dist.group.WORLD, global_expert_start=rank,
    )
    output.backward(grad.to(output.dtype))
    for before, tensor in zip(first, (hidden, probs, w13, w2), strict=True):
        assert torch.equal(before, tensor.grad)
    dist.barrier()
    if created:
        dist.destroy_process_group()
