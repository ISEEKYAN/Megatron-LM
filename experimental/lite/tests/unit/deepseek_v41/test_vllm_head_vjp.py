# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""CPU derivative controls for the chunked deployment head and its projection VJP."""
from types import SimpleNamespace

import pytest
import torch

from megatron.lite.model.deepseek_v41.vllm.head import _HeadLogProbs, head_log_probs
from megatron.lite.model.deepseek_v41.vllm.linear import _BF16Linear


@pytest.mark.parametrize("rows", [1, 3, 8])
@pytest.mark.parametrize("entropy", [False, True])
def test_chunked_head_owned_vjp_matches_dense_fp32_reference(rows, entropy):
    torch.manual_seed(42)
    hidden = torch.randn(7, 16, requires_grad=True)
    weight = torch.randn(32, 16, requires_grad=True)
    labels = torch.tensor([0, 31, 2, 4, 11, 8, 3])
    picked, spread = _HeadLogProbs.apply(hidden, weight, labels, 0.7, entropy, rows)
    # Deployment logits are BF16 (vLLM head_dtype); rounding is an identity STE.
    logits = hidden @ weight.T
    logits = logits + (logits.bfloat16().float() - logits).detach()
    distribution = (logits / 0.7).log_softmax(-1)
    target = distribution.gather(1, labels[:, None]).squeeze(1)
    loss = picked.sum()
    reference = target.sum()
    if entropy:
        loss = loss + spread.sum() * 0.3
        reference = reference - (distribution.exp() * distribution).sum() * 0.3
    expected = torch.autograd.grad(reference, (hidden, weight))
    actual = torch.autograd.grad(loss, (hidden, weight))
    torch.testing.assert_close(picked, target)
    for value, gradient in zip(actual, expected):
        torch.testing.assert_close(value, gradient, atol=2e-5, rtol=2e-5)


def test_public_head_has_no_deployment_cpu_fallback():
    with pytest.raises(RuntimeError, match="CUDA"):
        head_log_probs(
            torch.zeros(1, 16),
            torch.zeros(32, 16),
            torch.zeros(1, dtype=torch.long),
            1,
            entropy=False,
        )


def test_head_projection_vjp_uses_compute_weight_and_master_dtype():
    torch.manual_seed(71)
    x = torch.randn(5, 7, dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(11, 7, dtype=torch.float32, requires_grad=True)
    grad = torch.randn(5, 11)
    compute = weight + (weight.bfloat16().float() - weight).detach()
    output = torch.nn.functional.linear(x.float(), compute)
    # Backward GEMMs take BF16 operands (tensor cores): the upstream gradient is rounded.
    expected = torch.autograd.grad(output, (x, weight), grad.bfloat16().float())
    ctx = SimpleNamespace(
        saved_tensors=(x.detach(), weight.detach().bfloat16()), owners=(weight,), like=weight
    )
    actual = _BF16Linear.backward(ctx, grad)
    assert all(torch.equal(a, b) for a, b in zip(actual, expected))
    assert actual[0].dtype == x.dtype and actual[1].dtype == weight.dtype


def test_deployment_projection_rejects_cpu_forward():
    with pytest.raises(RuntimeError, match='requires CUDA'):
        _BF16Linear.apply(torch.ones(2, 8), torch.ones(16, 8))
