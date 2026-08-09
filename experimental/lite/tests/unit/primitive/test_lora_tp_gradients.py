# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Tensor-parallel gradient parity for row-parallel LoRA."""

from __future__ import annotations

import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from megatron.lite.primitive.modules import multi_lora_kernel
from megatron.lite.primitive.modules.lora import (
    LinearLoRA,
    SharedGroupedLinearLoRA,
    _all_gather_last_dim,
    _gather_sequence_parallel,
)
from megatron.lite.primitive.modules.lora_apply import LoRAWrappedLinear
from megatron.lite.primitive.modules.multi_lora_bank import (
    DenseLoraBank,
    LoraBankPartition,
    apply_batched_lora_delta,
)

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


def _shared_grouped_tp_worker(
    rank: int, world_size: int, init_file: str, queue
) -> None:
    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=world_size
    )
    try:
        torch.manual_seed(20260729)
        tokens, in_features, out_features, lora_rank = 4, 6, 8, 3
        x = torch.randn(tokens, in_features, dtype=torch.float64)
        a = torch.randn(lora_rank, in_features, dtype=torch.float64)
        b = torch.randn(out_features, lora_rank, dtype=torch.float64)
        grad_out = torch.randn(tokens, out_features, dtype=torch.float64)

        a_ref = a.clone().requires_grad_(True)
        b_ref = b.clone().requires_grad_(True)
        y_ref = (x @ a_ref.t()) @ b_ref.t()
        (y_ref * grad_out).sum().backward()

        adapter = SharedGroupedLinearLoRA(
            2,
            in_features,
            out_features,
            lora_rank,
            alpha=lora_rank,
            tp_group=dist.group.WORLD,
        ).to(torch.float64)
        with torch.no_grad():
            adapter.lora_a.copy_(a)
            adapter.lora_b.copy_(b)

        local_grad_out = torch.zeros_like(grad_out)
        local_out = out_features // world_size
        out_slice = slice(rank * local_out, (rank + 1) * local_out)
        local_grad_out[:, out_slice] = grad_out[:, out_slice]
        y_tp = adapter(x, [tokens // 2, tokens // 2])
        (y_tp * local_grad_out).sum().backward()
        queue.put(
            {
                "rank": rank,
                "a_grad": adapter.lora_a.grad.tolist(),
                "b_grad": adapter.lora_b.grad.tolist(),
                "a_ref": a_ref.grad.tolist(),
                "b_ref": b_ref.grad.tolist(),
            }
        )
    finally:
        dist.destroy_process_group()


def test_shared_grouped_tp2_gradients_match_tp1(tmp_path):
    """Replicated expert adapters must sum partial TP gradients."""
    world_size = 2
    init_file = tmp_path / "shared-grouped-lora-tp-gradient-init"
    ctx = mp.get_context("spawn")
    queue = ctx.SimpleQueue()
    mp.spawn(
        _shared_grouped_tp_worker,
        args=(world_size, str(init_file), queue),
        nprocs=world_size,
        join=True,
    )
    results = sorted(
        (queue.get() for _ in range(world_size)), key=lambda item: item["rank"]
    )

    a_grads = [
        torch.tensor(result["a_grad"], dtype=torch.float64) for result in results
    ]
    b_grads = [
        torch.tensor(result["b_grad"], dtype=torch.float64) for result in results
    ]
    a_ref = torch.tensor(results[0]["a_ref"], dtype=torch.float64)
    b_ref = torch.tensor(results[0]["b_ref"], dtype=torch.float64)
    torch.testing.assert_close(a_grads[0], a_grads[1], rtol=0, atol=1e-12)
    torch.testing.assert_close(b_grads[0], b_grads[1], rtol=0, atol=1e-12)
    for a_grad, b_grad in zip(a_grads, b_grads):
        torch.testing.assert_close(a_grad, a_ref, rtol=0, atol=1e-12)
        torch.testing.assert_close(b_grad, b_ref, rtol=0, atol=1e-12)

    if init_file.exists():
        os.unlink(init_file)


class _RecordingLinearLoRA(LinearLoRA):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.adapter_input = x.detach().clone()
        return super().forward(x)


class _SPAwareQKVBase(torch.nn.Module):
    """CPU proxy for TE LayerNormLinear's normalize-local, gather, GEMM path."""

    def __init__(self, weight: torch.Tensor, gamma: torch.Tensor, eps: float, tp_group):
        super().__init__()
        self.weight = torch.nn.Parameter(weight, requires_grad=False)
        self.gamma = torch.nn.Parameter(gamma, requires_grad=False)
        self.eps = eps
        self.tp_group = tp_group

    def forward_with_normalized_input(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        variance = x.float().pow(2).mean(dim=-1, keepdim=True)
        normalized_local = (
            x.float() * torch.rsqrt(variance + self.eps) * self.gamma.float()
        ).to(x.dtype)
        normalized_full = torch.empty(
            (normalized_local.shape[0] * dist.get_world_size(self.tp_group),)
            + normalized_local.shape[1:],
            dtype=normalized_local.dtype,
        )
        dist.all_gather_into_tensor(
            normalized_full, normalized_local.contiguous(), group=self.tp_group
        )
        return normalized_full @ self.weight.t(), normalized_local

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_with_normalized_input(x)[0]


def _qkv_sp_tp_worker(rank: int, world_size: int, init_file: str, queue) -> None:
    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=world_size
    )
    try:
        torch.manual_seed(20260728)
        seq, batch, hidden, out_features, lora_rank = 6, 2, 8, 12, 4
        eps = 1e-6
        x_full = torch.randn(seq, batch, hidden, dtype=torch.float64)
        gamma = torch.randn(hidden, dtype=torch.float64)
        base_weight = torch.randn(out_features, hidden, dtype=torch.float64)
        lora_a = torch.randn(lora_rank, hidden, dtype=torch.float64)
        lora_b = torch.randn(out_features, lora_rank, dtype=torch.float64)

        local_seq = seq // world_size
        local_out = out_features // world_size
        seq_slice = slice(rank * local_seq, (rank + 1) * local_seq)
        out_slice = slice(rank * local_out, (rank + 1) * local_out)
        rank_slice = slice(
            rank * (lora_rank // world_size), (rank + 1) * (lora_rank // world_size)
        )

        base = _SPAwareQKVBase(
            base_weight[out_slice].clone(), gamma.clone(), eps, dist.group.WORLD
        )
        adapter = _RecordingLinearLoRA(
            hidden,
            local_out,
            lora_rank,
            alpha=lora_rank,
            sequence_parallel_input=True,
            tp_group=dist.group.WORLD,
            rank_partition_size=world_size,
            rank_partitioned_a=True,
        ).to(torch.float64)
        with torch.no_grad():
            adapter.lora_a.copy_(lora_a[rank_slice])
            adapter.lora_b.copy_(lora_b[out_slice])

        wrapped = LoRAWrappedLinear(base, adapter, use_base_normalized_input=True)
        actual = wrapped(x_full[seq_slice])

        variance = x_full.float().pow(2).mean(dim=-1, keepdim=True)
        normalized_reference = (
            x_full.float() * torch.rsqrt(variance + eps) * gamma.float()
        ).to(x_full.dtype)
        normalized_local_reference = normalized_reference[seq_slice]
        merged_weight = base_weight[out_slice] + lora_b[out_slice] @ lora_a
        merged_reference = normalized_reference @ merged_weight.t()
        with torch.no_grad():
            materialized_delta = adapter.materialized_delta_weight()

        queue.put(
            {
                "rank": rank,
                "adapter_input": (adapter.adapter_input - normalized_local_reference)
                .abs()
                .max()
                .item(),
                "merged_forward": (actual - merged_reference).abs().max().item(),
                "materialized_delta": (
                    materialized_delta - (lora_b[out_slice] @ lora_a)
                )
                .abs()
                .max()
                .item(),
            }
        )
    finally:
        dist.destroy_process_group()


def test_qkv_tp2_sp_uses_base_normalized_input_and_matches_merged_export(tmp_path):
    world_size = 2
    init_file = tmp_path / "lora-qkv-sp-init"
    ctx = mp.get_context("spawn")
    queue = ctx.SimpleQueue()
    mp.spawn(
        _qkv_sp_tp_worker,
        args=(world_size, str(init_file), queue),
        nprocs=world_size,
        join=True,
    )
    results = sorted(
        (queue.get() for _ in range(world_size)), key=lambda item: item["rank"]
    )

    for result in results:
        assert result["adapter_input"] < 1e-12
        assert result["merged_forward"] < 1e-12
        assert result["materialized_delta"] < 1e-12

    if init_file.exists():
        os.unlink(init_file)


def _multi_lora_qkv_tp_worker(
    rank: int, world_size: int, init_file: str, queue
) -> None:
    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=world_size
    )
    try:
        torch.manual_seed(20260809)
        tokens, hidden, out_features, lora_rank = 4, 4, 6, 4
        slots = torch.tensor([0, 0, 1, 1], dtype=torch.int64)
        x_full = torch.randn(tokens, hidden, dtype=torch.float32)
        a_full = torch.randn(2, lora_rank, hidden, dtype=torch.float32)
        b_full = torch.randn(2, out_features, lora_rank, dtype=torch.float32)
        grad_full = torch.randn(tokens, out_features, dtype=torch.float32)

        x_ref = x_full.clone().requires_grad_(True)
        a_ref = a_full.clone().requires_grad_(True)
        b_ref = b_full.clone().requires_grad_(True)
        # The production carrier retains the established BGMV FP32 A-stage
        # scratch before the rank gather, including when its inputs are FP64
        # in this CPU oracle.
        y_ref = apply_batched_lora_delta(
            DenseLoraBank(a_ref, b_ref), x_ref, slots, scale=1.0
        )
        (y_ref * grad_full).sum().backward()

        local_rank = lora_rank // world_size
        local_out = out_features // world_size
        token_slice = slice(
            rank * (tokens // world_size), (rank + 1) * (tokens // world_size)
        )
        rank_slice = slice(rank * local_rank, (rank + 1) * local_rank)
        out_slice = slice(rank * local_out, (rank + 1) * local_out)
        a_local = a_full[:, rank_slice].clone().requires_grad_(True)
        b_local = b_full[:, out_slice].clone().requires_grad_(True)
        x_local = x_full[token_slice].clone().requires_grad_(True)
        bank = DenseLoraBank(
            a_local,
            b_local,
            LoraBankPartition(tp_size=world_size, rank_partitioned_a=True),
        )
        actual = apply_batched_lora_delta(
            bank,
            x_local,
            slots[token_slice],
            scale=1.0,
            tp_group=dist.group.WORLD,
            sequence_parallel_input=True,
        )
        (actual * grad_full[:, out_slice]).sum().backward()
        queue.put(
            {
                "rank": rank,
                "forward": (actual - y_ref[:, out_slice]).abs().max().item(),
                "x_grad": (x_local.grad - x_ref.grad[token_slice]).abs().max().item(),
                "a_grad": (a_local.grad - a_ref.grad[:, rank_slice]).abs().max().item(),
                "b_grad": (b_local.grad - b_ref.grad[:, out_slice]).abs().max().item(),
            }
        )
    finally:
        dist.destroy_process_group()


def _multi_lora_qkv_tp_backward_probe_worker(
    rank: int, world_size: int, init_file: str, queue
) -> None:
    """Isolate the hidden all-gather backward before changing production."""
    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=world_size
    )
    try:
        torch.manual_seed(20260810)
        slots = torch.tensor([0, 0, 1, 1], dtype=torch.int64)
        x_full = torch.randn(4, 4)
        a_full = torch.randn(2, 4, 4)
        b_full = torch.randn(2, 6, 4)
        grad_full = torch.randn(4, 6)
        token_slice = slice(rank * 2, (rank + 1) * 2)
        rank_slice = slice(rank * 2, (rank + 1) * 2)
        out_slice = slice(rank * 3, (rank + 1) * 3)
        rows = _gather_sequence_parallel(
            x_full[token_slice].clone().requires_grad_(), dist.group.WORLD
        )
        hidden = multi_lora_kernel.batched_lora_linear_stage(
            rows,
            a_full[:, rank_slice].clone().requires_grad_(),
            slots,
            output_dtype=torch.float32,
            max_g_size_hint=4,
        )
        hidden.retain_grad()
        hidden_full = _all_gather_last_dim(
            hidden, dist.group.WORLD, reduce_backward=True
        )
        out = multi_lora_kernel.batched_lora_linear_stage(
            hidden_full,
            b_full[:, out_slice].clone().requires_grad_(),
            slots,
            max_g_size_hint=4,
        )
        (out * grad_full[:, out_slice]).sum().backward()
        expected = torch.stack(
            [b_full[slot].t() @ grad for slot, grad in zip(slots, grad_full)]
        )[:, rank_slice]
        queue.put({"rank": rank, "hidden": (hidden.grad - expected).abs().max().item()})
    finally:
        dist.destroy_process_group()


def test_multi_lora_qkv_tp2_probe_hidden_gather_backward(tmp_path):
    world_size = 2
    init_file = tmp_path / "multi-lora-qkv-tp-probe-init"
    queue = mp.get_context("spawn").SimpleQueue()
    mp.spawn(
        _multi_lora_qkv_tp_backward_probe_worker,
        args=(world_size, str(init_file), queue),
        nprocs=world_size,
        join=True,
    )
    results = sorted(
        (queue.get() for _ in range(world_size)), key=lambda item: item["rank"]
    )
    assert [item["hidden"] for item in results] == pytest.approx([0.0, 0.0], abs=1e-6)
    if init_file.exists():
        os.unlink(init_file)


def _all_gather_last_dim_no_reduce_worker(rank, world_size, init_file, queue):
    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=world_size
    )
    try:
        local = (
            torch.arange(4, dtype=torch.float32)
            .reshape(2, 2)
            .add_(rank * 10)
            .requires_grad_()
        )
        gathered = _all_gather_last_dim(local, dist.group.WORLD, reduce_backward=False)
        grad = torch.arange(gathered.numel(), dtype=torch.float32).reshape_as(gathered)
        (gathered * grad).sum().backward()
        expected = grad[:, rank * 2 : (rank + 1) * 2]
        queue.put((rank, (local.grad - expected).abs().max().item()))
    finally:
        dist.destroy_process_group()


def test_all_gather_last_dim_tp2_without_reduce_backward_returns_local_chunk(tmp_path):
    init_file = tmp_path / "all-gather-last-dim-no-reduce-init"
    queue = mp.get_context("spawn").SimpleQueue()
    mp.spawn(
        _all_gather_last_dim_no_reduce_worker,
        args=(2, str(init_file), queue),
        nprocs=2,
        join=True,
    )
    assert [
        error for _, error in sorted(queue.get() for _ in range(2))
    ] == pytest.approx([0.0, 0.0])
    if init_file.exists():
        os.unlink(init_file)


def test_multi_lora_qkv_tp2_sp_matches_tp1_forward_and_gradients(tmp_path):
    """The bank carrier must retain the single-LoRA QKV TP/SP contract."""
    world_size = 2
    init_file = tmp_path / "multi-lora-qkv-tp-init"
    queue = mp.get_context("spawn").SimpleQueue()
    mp.spawn(
        _multi_lora_qkv_tp_worker,
        args=(world_size, str(init_file), queue),
        nprocs=world_size,
        join=True,
    )
    results = sorted(
        (queue.get() for _ in range(world_size)), key=lambda result: result["rank"]
    )
    for result in results:
        for key in ("forward", "x_grad", "a_grad", "b_grad"):
            assert result[key] < 1e-6, results
    if init_file.exists():
        os.unlink(init_file)
