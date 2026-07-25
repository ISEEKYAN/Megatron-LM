# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from __future__ import annotations

import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from megatron.lite.primitive.ops.cross_entropy import vocab_parallel_cross_entropy

pytestmark = pytest.mark.mlite


def _tp_worker(rank, world_size, port):
    dist.init_process_group(
        "gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world_size
    )
    try:
        torch.manual_seed(123)
        full_logits = torch.randn(7, 2, 12, dtype=torch.float16)
        labels = torch.randint(0, 12, (7, 2))
        local_logits = (
            full_logits.chunk(world_size, dim=-1)[rank].clone().requires_grad_()
        )

        loss = vocab_parallel_cross_entropy(
            local_logits, labels, dist.group.WORLD, chunk_size=3
        )
        loss.sum().backward()

        full_logits_ref = full_logits.clone().requires_grad_()
        expected = F.cross_entropy(
            full_logits_ref.float().reshape(-1, 12),
            labels.reshape(-1),
            reduction="none",
        ).view_as(labels)
        expected.sum().backward()

        torch.testing.assert_close(loss, expected)
        torch.testing.assert_close(
            local_logits.grad, full_logits_ref.grad.chunk(world_size, dim=-1)[rank]
        )
    finally:
        dist.destroy_process_group()


def test_chunked_cross_entropy_bounds_saved_fp32_state_and_matches_gradients():
    logits = torch.randn(7, 2, 11, dtype=torch.float16, requires_grad=True)
    labels = torch.randint(0, 11, (7, 2))
    logits_ref = logits.detach().clone().requires_grad_()
    saved_tensors = []

    def record_saved(tensor):
        saved_tensors.append(tensor)
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(record_saved, lambda tensor: tensor):
        loss = vocab_parallel_cross_entropy(logits, labels, chunk_size=3)
    loss.sum().backward()

    expected = F.cross_entropy(
        logits_ref.float().reshape(-1, 11), labels.reshape(-1), reduction="none"
    ).view_as(labels)
    expected.sum().backward()

    assert not any(
        tensor.dtype == torch.float32 and tensor.shape == logits.shape
        for tensor in saved_tensors
    )
    torch.testing.assert_close(loss, expected)
    torch.testing.assert_close(logits.grad, logits_ref.grad)


def test_chunked_cross_entropy_matches_two_rank_tp():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    mp.spawn(_tp_worker, args=(2, port), nprocs=2, join=True)
