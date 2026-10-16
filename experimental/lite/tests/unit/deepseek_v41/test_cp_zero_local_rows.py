# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""A contiguous-CP rank can own no rows of a short packed document.

Native launches with an empty grid leave a CUDA error that the next checked
launch reports (seen as ``invalid argument`` at an unrelated torch.cat), so the
per-document attention path must not launch them, yet must keep the autograd
graph so every CP rank runs the same backward collectives.
"""
import os

import pytest
import torch

PREFIX = os.environ.get('DS41_GPU_PARITY_PREFIX')


def test_sparse_attention_without_local_rows_keeps_graph():
    from megatron.lite.primitive.modules import deployment_math

    q = torch.zeros(1, 0, 4, 32, requires_grad=True)
    kv = torch.randn(1, 12, 32, requires_grad=True)
    sink = torch.zeros(4, requires_grad=True)
    out = deployment_math.sparse_attention(
        q,
        kv,
        sink,
        torch.zeros(1, 0, 12, dtype=torch.bool),
        None,
        main_size=0,
        window_size=8,
        width=8,
        scale=0.1,
    )
    assert out.shape == q.shape and out.dtype == q.dtype
    out.sum().backward()
    # Zero, not missing: the KV gather's backward still runs on this rank.
    assert torch.equal(kv.grad, torch.zeros_like(kv))
    assert torch.equal(sink.grad, torch.zeros_like(sink))


def _residual_cuda_error():
    """The error a later checked launch would report, if any."""
    torch.cuda.synchronize()
    torch.ones(1, device='cuda').add_(1)


@pytest.mark.gpus(1, min_architecture='blackwell')
@pytest.mark.skipif(PREFIX is None, reason='set DS41_GPU_PARITY_PREFIX to a 4L prefix')
def test_native_attention_pieces_skip_empty_launches():
    from megatron.lite.model.deepseek_v41.config import DeepseekV41Config
    from megatron.lite.primitive.modules import deployment_math
    from megatron.lite.primitive.modules.attention import csa
    from megatron.lite.primitive.modules.deployment_math import cache

    c = DeepseekV41Config.from_hf(PREFIX).attention_config()
    cos_sin = cache.rotary_cache(c, 0, 'cuda')
    q = torch.zeros(1, 0, c.heads, c.head_dim, device='cuda', dtype=torch.bfloat16)
    kv = torch.randn(1, 8, c.head_dim, device='cuda').bfloat16()
    empty = torch.arange(0, device='cuda')
    rotated, window = cache.swa_cache(
        q, kv, empty, cos_sin, c.eps, lambda a, b: (a, b), start=8
    )
    assert rotated.shape == q.shape and window.shape[1] == 8
    _residual_cuda_error()
    out = deployment_math.sparse_attention(
        q,
        kv,
        torch.zeros(c.heads, device='cuda'),
        torch.zeros(1, 0, 8, dtype=torch.bool, device='cuda'),
        None,
        main_size=0,
        window_size=8,
        width=128,
        scale=c.head_dim**-0.5,
    )
    assert out.shape == q.shape
    _residual_cuda_error()
    weight = torch.randn(
        c.groups * c.o_rank, c.heads * c.head_dim // c.groups, device='cuda'
    )
    projected = csa._bf16_wo_a(q, empty, c, 0, weight)
    assert projected.shape == (1, 0, c.groups * c.o_rank)
    _residual_cuda_error()
