# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""CPU dispatch proof; serving bit equality is covered by separate GPU gates."""

import sys
from types import SimpleNamespace

import pytest
import torch


@pytest.mark.parametrize('paged', [False, True])
def test_sparse_backward_policy_preserves_forward_and_global_determinism(monkeypatch, paged):
    from megatron.lite.model.deepseek_v41.vllm import attention
    from megatron.lite.model.deepseek_v41.vllm.protocol import ImplConfig, validate_impl_cfg_keys

    assert ImplConfig().attention_deterministic_backward is True
    validate_impl_cfg_keys({'attention_deterministic_backward': False})
    scratch_calls, adjoints, flash_calls = [], [], []

    def flash(**kw):
        flash_calls.append(kw)
        out = kw['q'] * 0.5 + kw['kv'][0, 0] * 0.25
        if 'out' in kw:
            kw['out'].copy_(out)
        return out, None, out.new_zeros(out.shape[:2])

    def scratch(*args, **kwargs):
        assert kwargs['requested'] and torch.are_deterministic_algorithms_enabled()
        scratch_calls.append(True)
        return {'deterministic': True, 'workspace': torch.empty(1)}

    def backward(q, kv, out, grad, lse, sink, indices, **kw):
        adjoints.append(kw.copy())
        assert torch.are_deterministic_algorithms_enabled()
        return dict(dq=torch.ones_like(q), dkv=torch.ones_like(kv), d_sink=torch.ones_like(sink))

    monkeypatch.setitem(
        sys.modules, 'vllm.v1.attention.ops.flashmla', SimpleNamespace(flash_mla_sparse_fwd=flash)
    )
    native = SimpleNamespace(
        _ensure_dsa_namespace=lambda: None,
        _deterministic_sparse_bwd_kwargs=scratch,
        _DSA=SimpleNamespace(sparse_attention_backward_wrapper=backward),
    )
    monkeypatch.setitem(
        sys.modules,
        'megatron.core.transformer.experimental_attention_variant.csa_utils',
        SimpleNamespace(fused_sparse_attention=native),
    )
    # Only admit CPU descriptors to the mocked CUDA interface; no CUDA initialization.
    monkeypatch.setattr(torch.Tensor, 'is_cuda', property(lambda self: True))
    before = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
    )
    outputs = []
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
        for policy in [True, False, None]:
            q = torch.arange(2 * 2 * 512, dtype=torch.float32).reshape(2, 2, 512).requires_grad_()
            kv = torch.ones(4, 512, requires_grad=True)
            sink = torch.zeros(2, requires_grad=True)
            args = [q, kv, sink, torch.zeros(2, 4, dtype=torch.int32), None, 0.5]
            if paged:
                args.append(lambda: q * 0.5 + kv[0] * 0.25)
            if policy is not None:
                args.append(policy)
            value = (attention._PagedSparse if paged else attention._Sparse).apply(*args)
            outputs.append(value.detach().flatten().log_softmax(0).view(torch.int32))
            value.sum().backward()
            assert all(
                x.grad is not None and x.grad.isfinite().all() and x.grad.count_nonzero()
                for x in (q, kv, sink)
            )
        assert torch.equal(outputs[0], outputs[1]) and torch.equal(outputs[0], outputs[2])
        # Gather reuses its forward output/LSE; paged still replays once.
        assert len(flash_calls) == 3
        assert len(scratch_calls) == 2
        assert [x.get('deterministic') for x in adjoints] == [True, False, True]
        assert 'workspace' in adjoints[0] and 'workspace' not in adjoints[1]
        assert torch.are_deterministic_algorithms_enabled()
    finally:
        torch.use_deterministic_algorithms(before[0], warn_only=before[1])


@pytest.mark.parametrize('heads', [2, 64])
@pytest.mark.parametrize('length_limited', [False, True])
def test_sparse_reuses_only_compatible_forward_statistics(monkeypatch, heads, length_limited):
    from megatron.lite.model.deepseek_v41.vllm import attention

    calls, captured = [], []

    def flash(**kw):
        output = kw.get('out')
        if output is None:
            output = torch.empty_like(kw['q'])
        output.copy_(kw['q'] * 0.5)
        lse = output.new_full(output.shape[:2], float(len(calls) + 1))
        calls.append((output, lse, kw['topk_length']))
        return output, None, lse

    def backward(q, kv, out, grad, lse, sink, indices, **kw):
        captured.append((out, lse, kw['topk_length']))
        return dict(dq=grad * 0.5, dkv=torch.zeros_like(kv), d_sink=torch.zeros_like(sink))

    monkeypatch.setitem(
        sys.modules, 'vllm.v1.attention.ops.flashmla', SimpleNamespace(flash_mla_sparse_fwd=flash)
    )
    monkeypatch.setitem(
        sys.modules,
        'megatron.core.transformer.experimental_attention_variant.csa_utils',
        SimpleNamespace(
            fused_sparse_attention=SimpleNamespace(
                _ensure_dsa_namespace=lambda: None,
                _DSA=SimpleNamespace(sparse_attention_backward_wrapper=backward),
            )
        ),
    )
    monkeypatch.setattr(torch.Tensor, 'is_cuda', property(lambda self: True))
    q = torch.randn(2, heads, 512, requires_grad=True)
    kv = torch.randn(3, 512, requires_grad=True)
    sink = torch.randn(heads, requires_grad=True)
    lengths = torch.tensor([2, 3], dtype=torch.int32) if length_limited else None
    output = attention._Sparse.apply(
        q, kv, sink, torch.zeros(2, 4, dtype=torch.int32), lengths, 0.5, False
    )
    assert torch.equal(output, q * 0.5)
    output.sum().backward()
    assert torch.equal(q.grad, torch.full_like(q, 0.5))
    assert len(calls) == (2 if length_limited else 1)
    expected = calls[-1]
    assert captured[0][0] is expected[0] and captured[0][1] is expected[1]
    assert captured[0][2] is None  # Preserve the existing length-limited replay adjoint.
    assert calls[0][2] is lengths
