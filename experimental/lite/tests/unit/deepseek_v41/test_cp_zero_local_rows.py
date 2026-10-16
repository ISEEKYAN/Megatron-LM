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


def test_bf16_wo_a_reference_without_local_rows(v41_core_te):
    """The empty-rank reference path keeps explicit widths (no -1 over 0 rows)."""
    from megatron.lite.model.deepseek_v41.config import DeepseekV41Config
    from megatron.lite.primitive.modules.attention import csa
    from test_w4a8_fp32 import tiny_config

    cfg = tiny_config().to_hf_dict()
    cfg['text_config']['wo_a_dtype'] = 'bf16'
    c = DeepseekV41Config(cfg).attention_config()
    q = torch.zeros(1, 0, c.heads, c.head_dim, dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(c.groups * c.o_rank, c.heads * c.head_dim // c.groups)
    projected = csa._bf16_wo_a(q, torch.arange(0), c, 0, weight)
    assert projected.shape == (1, 0, c.groups * c.o_rank)
    projected.sum().backward()
    assert q.grad.shape == q.shape


def test_visible_forward_without_rows_skips_the_native_kernel():
    from megatron.lite.primitive.modules.deployment_math.autograd import visible_forward

    def native(x, w):
        raise AssertionError('empty native launch')

    def reference(x, w):
        return x * w, (x * w).sum(-1)

    x = torch.zeros(0, 4, requires_grad=True)
    w = torch.ones(4, requires_grad=True)
    out, total = visible_forward(native, reference, x, w)
    assert out.shape == (0, 4) and total.shape == (0,)
    (out.sum() + total.sum()).backward()
    # Zero, not missing: the empty rank still joins the backward.
    assert torch.equal(w.grad, torch.zeros(4)) and x.grad.shape == (0, 4)


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


def test_every_visible_forward_site_accepts_a_rank_without_rows(
    v41_core_te, monkeypatch
):
    """All 13 deployment_math.visible_forward sites, fed zero local rows.

    The native side is never reached for an empty first input, so this runs on
    CPU; each site's reference must produce empty rows and a backward. The two
    CUDA-gated cache writers are fed through their own zero-row paths:
    swa_cache in test_swa_cache_cp_split, compressed_cache only ever sees the
    gathered (non-empty) compressed sequence.
    """
    import importlib
    from types import SimpleNamespace

    from megatron.lite.primitive.modules import deployment_math as dm
    from megatron.lite.primitive.modules.attention import csa

    # The package re-exports the function under the module's name.
    comp = importlib.import_module(
        'megatron.lite.primitive.modules.deployment_math.compressor'
    )
    from test_w4a8_fp32 import tiny_config

    monkeypatch.setattr(
        comp, 'bf16_fp32_linear', lambda x, w: torch.nn.functional.linear(x, w)
    )
    copies, width = 4, 8
    mixes_width = 2 * copies + copies**2
    mixes = SimpleNamespace(
        fn=torch.randn(mixes_width, copies * width, requires_grad=True),
        scale=torch.ones(3, requires_grad=True),
        base=torch.zeros(mixes_width, requires_grad=True),
        norm_eps=1e-6,
        hc_eps=1e-6,
        iterations=3,
        copies=copies,
        broadcast_projection=False,
    )
    hidden = torch.zeros(1, 0, copies, width, dtype=torch.bfloat16, requires_grad=True)
    rows = torch.zeros(1, 0, width, dtype=torch.bfloat16, requires_grad=True)
    weight = torch.ones(width, requires_grad=True)
    pre = torch.zeros(1, 0, copies)
    post = torch.zeros(1, 0, copies)
    comb = torch.zeros(1, 0, copies, copies)
    cfg = tiny_config().to_hf_dict()
    cfg['text_config']['wo_a_dtype'] = 'bf16'
    from megatron.lite.model.deepseek_v41.config import DeepseekV41Config

    c = DeepseekV41Config(cfg).attention_config()
    sites = {
        'mhc_coefficients': lambda: dm.mhc_coefficients(
            hidden,
            mixes.fn,
            mixes.scale,
            mixes.base,
            copies=copies,
            norm_eps=1e-6,
            hc_eps=1e-6,
            iterations=3,
            broadcast=False,
        ),
        'mhc_post': lambda: dm.mhc_post(rows, hidden, post, comb),
        'mhc_joint_first': lambda: dm.mhc_joint(hidden, pre, weight, mixes),
        'mhc_joint_pending': lambda: dm.mhc_joint(
            hidden, pre, weight, mixes, (rows, hidden, post, comb)
        ),
        'hc_collapse': lambda: dm.hc_collapse(hidden, pre),
        'rms_norm': lambda: dm.rms_norm(rows, weight, (width,), 1e-6),
        'qkv_rms_norm': lambda: dm.qkv_rms_norm(rows, weight, (width,), 1e-6),
        'compressor': lambda: dm.compressor(
            torch.zeros(1, 0, width, requires_grad=True),
            torch.randn(512, width),
            torch.randn(512, width),
            torch.ones(512),
            2,
            1e-6,
        ),
        'engram_post': lambda: dm.engram_post(
            hidden,
            torch.zeros(1, 0, (copies + 1) * width),
            torch.ones(width, requires_grad=True),
            torch.ones(width, requires_grad=True),
            eps=1e-6,
        ),
        'shared_swiglu': lambda: dm.shared_swiglu(
            torch.zeros(0, 2 * width, requires_grad=True), 10.0
        ),
        'router_topk': lambda: dm.router_topk(
            torch.zeros(0, 16, requires_grad=True),
            torch.zeros(16),
            topk=2,
            scaling_factor=1.0,
        )[0],
        'bf16_wo_a': lambda: csa._bf16_wo_a(
            torch.zeros(1, 0, c.heads, c.head_dim, requires_grad=True),
            torch.arange(0),
            c,
            0,
            torch.randn(c.groups * c.o_rank, c.heads * c.head_dim // c.groups),
        ),
    }
    for name, call in sites.items():
        out = call()
        outs = out if isinstance(out, tuple) else (out,)
        assert all(t.shape[0] == 0 or t.shape[1] == 0 for t in outs), name
        total = sum(t.float().sum() for t in outs if t.requires_grad)
        if isinstance(total, torch.Tensor):
            total.backward()
