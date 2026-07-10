# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import os

import pytest
import torch

from megatron.lite.primitive.parallel.cp import (
    gather_contiguous_for_cp,
    roll_contiguous_left_for_cp,
)

pytestmark = pytest.mark.mlite


def test_gather_contiguous_for_cp_cp1_is_identity():
    local = torch.arange(4, dtype=torch.float32).reshape(1, 4).requires_grad_()
    assert gather_contiguous_for_cp(local, cp_size=1, cp_group=None, seq_dim=1) is local


def test_cp_contiguous_helpers_require_group_for_cp_gt_1():
    with pytest.raises(ValueError, match="cp_group"):
        gather_contiguous_for_cp(torch.arange(4), cp_size=2, cp_group=None, seq_dim=0)
    with pytest.raises(ValueError, match="cp_group"):
        roll_contiguous_left_for_cp(
            torch.arange(4), cp_rank=0, cp_size=2, cp_group=None, seq_dim=0
        )


def test_roll_contiguous_left_for_cp_cp1_matches_global_roll():
    tensor = torch.arange(1, 9).reshape(1, -1)
    rolled, token_sum = roll_contiguous_left_for_cp(
        tensor, cp_rank=0, cp_size=1, cp_group=None, seq_dim=1
    )
    assert torch.equal(rolled, torch.tensor([[2, 3, 4, 5, 6, 7, 8, 0]]))
    assert token_sum.item() == 35


def test_cp_contiguous_gather_and_roll_cross_rank_boundaries():
    import torch.distributed as dist

    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        pytest.skip("Run with torchrun so CP ranks are available.")
    if not dist.is_initialized():
        dist.init_process_group("gloo")
    world, rank = dist.get_world_size(), dist.get_rank()
    if world < 2:
        pytest.skip("Contiguous CP gather requires at least two ranks.")

    local_len = 3
    local = torch.arange(
        rank * local_len, (rank + 1) * local_len, dtype=torch.float32
    ).reshape(1, local_len).requires_grad_()
    gathered = gather_contiguous_for_cp(
        local, cp_size=world, cp_group=dist.group.WORLD, seq_dim=1
    )
    expected = torch.arange(world * local_len, dtype=torch.float32).reshape(1, -1)
    assert torch.equal(gathered, expected)
    gathered.narrow(1, rank * local_len, local_len).sum().backward()
    assert torch.equal(local.grad, torch.ones_like(local))

    rolled_local, _ = roll_contiguous_left_for_cp(
        local.detach(), cp_rank=rank, cp_size=world,
        cp_group=dist.group.WORLD, seq_dim=1,
    )
    parts = [torch.empty_like(rolled_local) for _ in range(world)]
    dist.all_gather(parts, rolled_local, group=dist.group.WORLD)
    expected = torch.roll(expected, shifts=-1, dims=1)
    expected[:, -1] = 0
    assert torch.equal(torch.cat(parts, dim=1), expected)
    if rank == 0:
        print(f"NON_SKIP_CONTIGUOUS_CP_GATHER_PASSED world_size={world}")
