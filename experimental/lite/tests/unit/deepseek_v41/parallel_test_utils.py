# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Shared test setup and exact comparisons; reference computations stay local."""

from datetime import timedelta
from functools import partial
from pathlib import Path

import torch

assert_exact = partial(torch.testing.assert_close, atol=0, rtol=0)


def seed_engram(model):
    with torch.no_grad():
        for block in model.layers:
            if block is not None and block.engram is not None:
                table = block.engram.embed
                rows = torch.linspace(
                    -0.25, 0.25, table.weight.numel(), device=table.weight.device
                ).reshape(table.weight.shape)
                table.weight.copy_(rows.to(table.weight.dtype))
                if table.master is not None:
                    table.master.copy_(table.weight.float())


def init_world(rank, directory, *, world, timeout, rendezvous):
    torch.distributed.init_process_group(
        'nccl',
        init_method=(Path(directory) / rendezvous).as_uri(),
        rank=rank,
        world_size=world,
        timeout=timedelta(seconds=timeout),
    )
