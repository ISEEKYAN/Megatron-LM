# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The same GDN gate inputs must survive sequence-length compile history."""

import pytest
import torch


@pytest.mark.gpus(1)
def test_gate_backward_is_independent_of_shape_history():
    from megatron.lite.primitive.modules.gated_delta_net import GatedDeltaNet

    assert torch.cuda.is_available(), "This regression requires a real GPU"
    gate = GatedDeltaNet._compute_g_and_beta
    # Place dA near a BF16 rounding boundary to expose FP32 codegen changes.
    midpoint = (3.790855407714844e-5 + 3.814697265625e-5) / 2
    for seed in range(16):
        torch._dynamo.reset()
        generator = torch.Generator().manual_seed(seed)
        inputs = [
            torch.zeros(3, dtype=torch.bfloat16),
            torch.ones(3, dtype=torch.bfloat16),
            torch.randn(1, 256, 3, generator=generator).to(torch.bfloat16),
            torch.zeros(1, 256, 3, dtype=torch.bfloat16),
        ]
        inputs = [x.cuda().requires_grad_() for x in inputs]
        output = gate(*inputs)
        dg = torch.randn(1, 256, 3, generator=generator) * 1e-5
        factors = output[0].detach().cpu().double()
        dg[0, -1, 1] = (
            midpoint - (factors[0, :-1, 1] * dg[0, :-1, 1].double()).sum()
        ) / factors[0, -1, 1]
        cotangents = (dg.cuda(), torch.zeros_like(output[1]))
        before = torch.autograd.grad(output, inputs, cotangents)
        for length in (240, 144):
            changed = [
                (
                    x.detach().clone().requires_grad_()
                    if x.ndim == 1
                    else x[:, :length].detach().contiguous().requires_grad_()
                )
                for x in inputs
            ]
            warm = gate(*changed)
            torch.autograd.grad(warm, changed, tuple(torch.ones_like(x) for x in warm))
        repeated = gate(*inputs)
        after = torch.autograd.grad(repeated, inputs, cotangents)
        for name, old, new in zip(("g", "beta"), output, repeated):
            assert torch.equal(old, new), (seed, name)
        for name, old, new in zip(("dA", "dt_bias", "alpha", "beta"), before, after):
            assert torch.equal(old, new), (seed, name, (old - new).abs().max().item())
    torch._dynamo.reset()
