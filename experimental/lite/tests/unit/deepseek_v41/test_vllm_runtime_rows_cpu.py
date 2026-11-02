# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Dispatch controls only; native CUDA bit equality is a separate test."""
import ast
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

from megatron.lite.model.deepseek_v41.vllm import linear


def _install(monkeypatch, bi):
    parent = ModuleType('vllm.model_executor.kernels.linear.cute_dsl')
    parent.batch_invariant = bi
    monkeypatch.setitem(sys.modules, parent.__name__, parent)


@pytest.mark.parametrize('tensor_core', [False, True])
def test_runtime_rows_dispatch_once_without_serving_sweep(monkeypatch, tensor_core):
    calls = []

    def gemm(a, weight, out, *, compiled_dims):
        assert compiled_dims == 'nk'  # M is a runtime kernel argument.
        calls.append(a.shape[0])
        # A row-local CPU oracle, not a surrogate for native GEMM bit parity.
        out.copy_((a.float()[:, None, :] * weight.float()[None]).sum(-1))

    def fallback(a, weight):
        if tensor_core:
            raise AssertionError("public BI wrapper would enter serving prewarm")
        calls.append(a.shape[0])
        return (a.float()[:, None, :] * weight.float()[None]).sum(-1)

    def forbidden(*args, **kwargs):
        raise AssertionError('training entered serving prewarm')

    bi = SimpleNamespace(
        _uses_tensor_core=lambda weight: tensor_core,
        _deterministic_deep_gemm=lambda: SimpleNamespace(bf16_gemm_nt=gemm),
        bf16_fixed_mm=fallback,
        _compile_tensor_core_configs=forbidden,
        compile_tensor_core_rows=forbidden,
    )
    _install(monkeypatch, bi)
    gen = torch.Generator().manual_seed(310)
    x = torch.randn(1031, 8, generator=gen).bfloat16()
    w = torch.randn(4, 8, generator=gen).bfloat16()
    expected = (x.float()[:, None, :] * w.float()[None]).sum(-1)
    calls.clear()
    rows = [1, 7, 31, 32, 33, 65, 129, 257, 1031, 0]
    for m in rows:
        got = linear._bf16_fixed_mm(x[:m], w)
        assert got.shape == (m, 4)
        assert torch.equal(got, expected[:m])
    assert calls == rows  # exactly one real dispatch per request, not 1..M.


def test_projection_never_imports_serving_prewarmer():
    tree = ast.parse(Path(linear.__file__).with_name('attention.py').read_text())
    projection = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == '_FP8OutputProjection'
    )
    calls = [
        n.func.id
        for n in ast.walk(projection)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
    ]
    assert '_compile_einsum_configs' not in calls
    assert calls.count('fp8_einsum') == 1
