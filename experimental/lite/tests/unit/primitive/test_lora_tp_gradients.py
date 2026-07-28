# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Tensor-parallel gradient parity for row-parallel LoRA."""

from __future__ import annotations

import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from megatron.lite.primitive.modules.lora import LinearLoRA

pytestmark = [pytest.mark.mlite, pytest.mark.distributed]


def _linear_proj_tp_worker(rank: int, world_size: int, init_file: str, queue) -> None:
    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=world_size
    )
    try:
        torch.manual_seed(20260728)
        batch, in_features, out_features, lora_rank = 3, 8, 6, 4
        x_full = torch.randn(batch, in_features, dtype=torch.float64)
        a_full = torch.randn(lora_rank, in_features, dtype=torch.float64)
        b_full = torch.randn(out_features, lora_rank, dtype=torch.float64)
        grad_out = torch.randn(batch, out_features, dtype=torch.float64)

        x_ref = x_full.clone().requires_grad_(True)
        a_ref = a_full.clone().requires_grad_(True)
        b_ref = b_full.clone().requires_grad_(True)
        y_ref = (x_ref @ a_ref.t()) @ b_ref.t()
        (y_ref * grad_out).sum().backward()

        local_in = in_features // world_size
        local_out = out_features // world_size
        in_slice = slice(rank * local_in, (rank + 1) * local_in)
        out_slice = slice(rank * local_out, (rank + 1) * local_out)
        adapter = LinearLoRA(
            local_in,
            out_features,
            lora_rank,
            alpha=lora_rank,
            tp_group=dist.group.WORLD,
            tp_rank=rank,
            input_parallel_reduce=True,
            output_partition_size=world_size,
            output_partitioned_b=True,
        ).to(torch.float64)
        with torch.no_grad():
            adapter.lora_a.copy_(a_full[:, in_slice])
            adapter.lora_b.copy_(b_full[out_slice, :])

        x_local = x_full[:, in_slice].clone().requires_grad_(True)
        y_tp = adapter(x_local)
        (y_tp * grad_out).sum().backward()
        queue.put(
            {
                "rank": rank,
                "forward": (y_tp - y_ref).abs().max().item(),
                "a_grad": (adapter.lora_a.grad - a_ref.grad[:, in_slice])
                .abs()
                .max()
                .item(),
                "b_grad": (adapter.lora_b.grad - b_ref.grad[out_slice, :])
                .abs()
                .max()
                .item(),
                "x_grad": (x_local.grad - x_ref.grad[:, in_slice]).abs().max().item(),
                "a_grad_ratio": (
                    adapter.lora_a.grad.norm() / a_ref.grad[:, in_slice].norm()
                ).item(),
            }
        )
    finally:
        dist.destroy_process_group()


def test_linear_proj_tp2_gradients_match_tp1(tmp_path):
    """The backward SUM combines output shards; it must not scale TP2 gradients."""
    world_size = 2
    init_file = tmp_path / "lora-tp-gradient-init"
    ctx = mp.get_context("spawn")
    queue = ctx.SimpleQueue()
    mp.spawn(
        _linear_proj_tp_worker,
        args=(world_size, str(init_file), queue),
        nprocs=world_size,
        join=True,
    )
    results = sorted(
        (queue.get() for _ in range(world_size)), key=lambda item: item["rank"]
    )

    for result in results:
        assert result["forward"] < 1e-12
        assert result["a_grad"] < 1e-12
        assert result["b_grad"] < 1e-12
        assert result["x_grad"] < 1e-12
        assert result["a_grad_ratio"] == pytest.approx(1.0, abs=1e-12)

    if init_file.exists():
        os.unlink(init_file)
