# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Indexer wk / weights_proj use vLLM's batch-invariant BF16 GEMM, not cuBLAS."""
import os
from types import SimpleNamespace

import pytest
import torch


def test_indexer_linear_dispatches_native_gemm_on_cuda(monkeypatch):
    from megatron.lite.primitive.modules import deployment_math
    from megatron.lite.primitive.modules.attention import csa

    calls = []

    def native(x, weight):
        calls.append((x, weight))
        return torch.ones(2, 3, dtype=torch.float32)

    monkeypatch.setattr(deployment_math, 'bf16_fp32_linear', native)
    weight = torch.zeros(3, 4, dtype=torch.bfloat16)
    linear = SimpleNamespace(weight=weight)
    x = SimpleNamespace(is_cuda=True, dtype=torch.bfloat16)
    out = csa._indexer_linear(linear, x, True)
    assert calls == [(x, weight)]
    assert out.dtype == torch.bfloat16

    fallback = torch.nn.Linear(4, 3, bias=False, dtype=torch.bfloat16)
    cpu = torch.randn(2, 4, dtype=torch.bfloat16)
    assert torch.equal(csa._indexer_linear(fallback, cpu, True), fallback(cpu))
    assert len(calls) == 1


@pytest.mark.gpus(1, min_architecture="blackwell")
@pytest.mark.parametrize("rows", [1, 7, 12928])
@pytest.mark.parametrize("shape", [(512, 128), (5120, 32)], ids=["wk", "weights_proj"])
def test_indexer_linear_matches_vllm_replicated_linear(rows, shape):
    if os.environ.get("MEGATRON_LITE_REQUIRE_CUDA_TESTS") == "1":
        assert torch.cuda.is_available()
    elif not torch.cuda.is_available():
        pytest.skip("CUDA batch-invariant GEMM")
    from megatron.lite.primitive.modules.attention import csa
    from vllm.model_executor.determinism.batch_invariant import linear_batch_invariant

    torch.manual_seed(rows)
    k, n = shape
    linear = torch.nn.Linear(k, n, bias=False, dtype=torch.bfloat16, device="cuda")
    x = torch.randn(rows, k, device="cuda", dtype=torch.bfloat16)
    expected = linear_batch_invariant(x, linear.weight)
    actual = csa._indexer_linear(linear, x, True)
    assert actual.dtype == expected.dtype == torch.bfloat16
    assert torch.equal(actual, expected)
