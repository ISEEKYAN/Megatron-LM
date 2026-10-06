# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Native CUDA forward, immutable Torch reference VJP, and default preservation."""
import pytest
import torch
from engram_reference import Engram as ReferenceEngram
from megatron.lite.primitive.modules import deployment_math as deployment
from megatron.lite.primitive.modules.engram_lookup import Engram


def operands(width, mask_kind, *, device='cpu', tokens=33):
    generator = torch.Generator(device=device).manual_seed(593)
    hidden = torch.randn(
        1, tokens, 4, width, generator=generator, device=device
    ).bfloat16()
    kv = torch.randn(
        1, tokens, 5 * width, generator=generator, device=device
    ).bfloat16()
    query = torch.randn(4, width, generator=generator, device=device)
    key = torch.randn(4, width, generator=generator, device=device)
    mask = None
    if mask_kind != 'none':
        mask = torch.ones(1, tokens, device=device, dtype=torch.bool)
        mask[:, ::3] = False
        if mask_kind == 'inactive':
            mask.zero_()
    return tuple(x.requires_grad_() for x in (hidden, kv, query, key)), mask


def reference(inputs, mask, *, decoded):
    h, v, q, k = inputs
    module = ReferenceEngram(
        h.shape[-1], 4, torch.nn.Identity(), torch.nn.Identity(), eps=1e-20
    )
    # Decoded BF16 values are FP32 leaves: expected master VJP is the STE's
    # identity pullback, not a BF16 gradient cast or the new implementation.
    module.q_weight = torch.nn.Parameter(
        q.detach().bfloat16().float() if decoded else q.detach()
    )
    module.k_weight = torch.nn.Parameter(
        k.detach().bfloat16().float() if decoded else k.detach()
    )
    h = h.detach().requires_grad_()
    v = v.detach().requires_grad_()
    result = module(h, v.unsqueeze(-2), mask)
    return result, (h, v, module.q_weight, module.k_weight)


@pytest.mark.parametrize('width', [128, 1024, 5120])
@pytest.mark.parametrize('mask_kind', ['none', 'mixed', 'inactive'])
def test_declared_reference_vjp_and_default_off_are_immutable(width, mask_kind):
    inputs, mask = operands(width, mask_kind, tokens=3)
    expected, leaves = reference(inputs, mask, decoded=True)
    actual = deployment.engram_reference(*inputs, eps=1e-20, token_mask=mask)
    assert torch.equal(actual, expected)
    upstream = torch.linspace(-1, 1, actual.numel()).reshape_as(actual).bfloat16()
    wanted = torch.autograd.grad(expected, leaves, upstream)
    got = torch.autograd.grad(actual, inputs, upstream)
    assert all(torch.equal(a, b) for a, b in zip(got, wanted))
    assert got[2].dtype == got[3].dtype == torch.float32
    assert all(torch.isfinite(x).all() for x in got)
    baseline, base_leaves = reference(inputs, mask, decoded=False)
    module = Engram(width, 4, torch.nn.Identity(), torch.nn.Identity(), eps=1e-20)
    module.q_weight = torch.nn.Parameter(inputs[2].detach())
    module.k_weight = torch.nn.Parameter(inputs[3].detach())
    h = inputs[0].detach().requires_grad_()
    v = inputs[1].detach().requires_grad_()
    default = module(h, v.unsqueeze(-2), mask)
    assert not module.deployment_math and torch.equal(default, baseline)
    default_grads = torch.autograd.grad(
        default, (h, v, module.q_weight, module.k_weight), upstream
    )
    baseline_grads = torch.autograd.grad(baseline, base_leaves, upstream)
    assert all(torch.equal(a, b) for a, b in zip(default_grads, baseline_grads))


@pytest.mark.gpus(1)
@pytest.mark.parametrize('width', [128, 1024, 5120])
@pytest.mark.parametrize('mask_kind', ['none', 'mixed', 'inactive'])
def test_native_forward_partitions_and_owned_vjp(width, mask_kind):
    if not torch.cuda.is_available():
        pytest.skip('Real native CUDA oracle required')
    import triton
    import vllm.models.deepseek_v41.common.engram as _imports_engram

    _fused_engram_post_wkv_kernel = _imports_engram._fused_engram_post_wkv_kernel

    inputs, mask = operands(width, mask_kind, device='cuda')
    h, v, q, k = inputs
    residual = h.reshape(-1, 4, width)
    kv = v.reshape(-1, 5 * width)
    query, key = q.bfloat16(), k.bfloat16()
    token_mask = None if mask is None else mask.reshape(-1)
    expected = torch.empty_like(residual)
    block = triton.next_power_of_2(width)
    _fused_engram_post_wkv_kernel[(33 * 4,)](
        residual,
        kv,
        query,
        key,
        token_mask if token_mask is not None else residual,
        expected,
        33,
        *residual.stride(),
        *kv.stride(),
        *query.stride(),
        *key.stride(),
        token_mask.stride(0) if token_mask is not None else 0,
        *expected.stride(),
        1e-20,
        1e-6,
        DIM=width,
        HC_MULT=4,
        BLOCK_SIZE=block,
        HAS_MASK=token_mask is not None,
        num_warps=8 if block >= 2048 else 4,
    )
    actual = deployment.engram_post(*inputs, eps=1e-20, token_mask=mask)
    assert torch.equal(actual, expected.reshape_as(h))
    chunks = []
    for start, end in [(0, 1), (1, 16), (16, 33)]:
        chunks.append(
            deployment.engram_post(
                h[:, start:end],
                v[:, start:end],
                q,
                k,
                eps=1e-20,
                token_mask=None if mask is None else mask[:, start:end],
            )
        )
    assert torch.equal(actual, torch.cat(chunks, dim=1))
    if mask is not None:
        assert torch.equal(actual[~mask], h[~mask])
    ref, ref_leaves = reference(inputs, mask, decoded=True)
    upstream = (
        torch.linspace(-1, 1, actual.numel(), device='cuda')
        .reshape_as(actual)
        .bfloat16()
    )
    wanted = torch.autograd.grad(ref, ref_leaves, upstream)
    got = torch.autograd.grad(actual, inputs, upstream)
    assert all(torch.equal(a, b) for a, b in zip(got, wanted))
    assert all(torch.isfinite(x).all() for x in got)
    assert got[2].dtype == got[3].dtype == torch.float32
