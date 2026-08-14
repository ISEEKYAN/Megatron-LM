"""Four-rank dense-DP/expert-DP gradient finalization gate."""

from __future__ import annotations

import os

import pytest
import torch
import torch.distributed as dist

from megatron.lite.model.deepseek_v4.vllm.protocol import _finalize_replica_grads
from megatron.lite.primitive.parallel import init_parallel
from megatron.lite.runtime.contracts import ParallelConfig


class _ReplicaParameters(torch.nn.Module):
    def __init__(self, device: torch.device):
        super().__init__()
        self.dense = torch.nn.Parameter(torch.zeros(3, device=device))
        self.layer = torch.nn.Module()
        self.layer.experts = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.zeros(3, device=device))]
        )


@pytest.mark.gpus(4)
def test_ep2_expert_dp2_and_dense_replica_grad_averages() -> None:
    if not torch.cuda.is_available() or int(os.environ.get("WORLD_SIZE", "1")) != 4:
        pytest.skip("requires torchrun with four CUDA ranks")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    created = not dist.is_initialized()
    if created:
        dist.init_process_group("nccl")
    rank = dist.get_rank()
    ps = init_parallel(ParallelConfig(tp=1, ep=2))
    assert ps.ep_size == 2
    assert ps.expert_dp_size == 2

    model = _ReplicaParameters(torch.device("cuda", local_rank))
    model.dense.grad = torch.full_like(model.dense, rank + 1.0)
    model.layer.experts[0].grad = torch.full_like(
        model.layer.experts[0], rank + 1.0
    )
    _finalize_replica_grads(model, ps)

    torch.testing.assert_close(model.dense.grad, torch.full_like(model.dense, 2.5))
    expert_expected = 2.0 if rank % 2 == 0 else 3.0
    torch.testing.assert_close(
        model.layer.experts[0].grad,
        torch.full_like(model.layer.experts[0], expert_expected),
    )
    dist.barrier()
    if created:
        dist.destroy_process_group()
