# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from __future__ import annotations

import pytest
import torch

from megatron.lite.primitive.ops.linear_cross_entropy import linear_cross_entropy

pytestmark = [pytest.mark.mlite, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")]


def _reference(hidden, weight, labels, temperature):
    logits = hidden.reshape(-1, hidden.shape[-1]).float() @ weight.float().T
    logits = logits / temperature
    flat_labels = labels.reshape(-1)
    log_probs = -torch.nn.functional.cross_entropy(logits, flat_labels, reduction="none")
    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    entropy = torch.logsumexp(logits, dim=-1) - (probabilities * logits).sum(dim=-1)
    return log_probs.reshape(labels.shape), entropy.reshape(labels.shape)


def test_qwen35_linear_cross_entropy_matches_explicit_logits():
    pytest.importorskip("verl.utils.kernel.linear_cross_entropy")
    torch.manual_seed(20260711)
    temperature = 1.5
    tokens, batch, hidden_size, vocab_size = 256, 1, 2048, 248320

    hidden = torch.empty(tokens, batch, hidden_size, device="cuda", dtype=torch.bfloat16)
    weight = torch.empty(vocab_size, hidden_size, device="cuda", dtype=torch.bfloat16)
    hidden.uniform_(-0.02, 0.02).requires_grad_()
    weight.uniform_(-0.02, 0.02).requires_grad_()
    labels = torch.randint(vocab_size, (tokens, batch), device="cuda")

    reference_log_probs, reference_entropy = _reference(hidden, weight, labels, temperature)
    fused_log_probs, fused_entropy = linear_cross_entropy(
        hidden, weight, labels, temperature=temperature
    )

    torch.testing.assert_close(reference_log_probs, fused_log_probs, atol=1e-3, rtol=2e-4)
    torch.testing.assert_close(reference_entropy, fused_entropy, atol=5e-3, rtol=5e-4)

    log_prob_grad = torch.empty_like(reference_log_probs).uniform_(-1, 1)
    entropy_grad = torch.empty_like(reference_entropy).uniform_(-0.5, 0.5)
    reference_grads = torch.autograd.grad(
        (reference_log_probs, reference_entropy),
        (hidden, weight),
        (log_prob_grad, entropy_grad),
    )
    fused_grads = torch.autograd.grad(
        (fused_log_probs, fused_entropy),
        (hidden, weight),
        (log_prob_grad, entropy_grad),
    )
    for reference_grad, fused_grad in zip(reference_grads, fused_grads, strict=True):
        torch.testing.assert_close(reference_grad, fused_grad, atol=2e-2, rtol=4e-2)

    print(
        "QWEN35_LINEAR_CE_PRECISION "
        f"log_probs_max_abs={(reference_log_probs - fused_log_probs).abs().max().item():.8e} "
        f"entropy_max_abs={(reference_entropy - fused_entropy).abs().max().item():.8e} "
        f"hidden_grad_max_abs={(reference_grads[0] - fused_grads[0]).abs().max().item():.8e} "
        f"weight_grad_max_abs={(reference_grads[1] - fused_grads[1]).abs().max().item():.8e}"
    )
