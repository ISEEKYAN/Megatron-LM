# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""CUDA semantic smoke for the observable recompute contract."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from megatron.lite.primitive.recompute import apply_recompute


pytestmark = [pytest.mark.mlite, pytest.mark.smoke, pytest.mark.gpu]


class _CountingSquare(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return x.square()


def test_recompute_replays_cuda_forward_and_preserves_gradient() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for recompute semantic smoke.")

    layer = nn.Module().cuda()
    layer.inner = _CountingSquare().cuda()
    result = apply_recompute(
        nn.ModuleList([layer]), ["inner"], {"inner": lambda module: module.inner}
    )

    x = torch.tensor([2.0, -3.0], device="cuda", requires_grad=True)
    layer.inner(x).sum().backward()

    assert (result.units, result.matched, result.wrapped) == (1, 1, 1)
    assert layer.inner.calls == 2
    torch.testing.assert_close(x.grad, torch.tensor([4.0, -6.0], device="cuda"))
    print("CUDA_RECOMPUTE_SEMANTIC_SMOKE_PASSED units=1 matched=1 wrapped=1")
