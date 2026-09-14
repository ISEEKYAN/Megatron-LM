"""Prevent profiler attribution from including upstream attention operands."""

import importlib.util
from pathlib import Path

import torch


def test_qsa_profile_scope_excludes_upstream_index():
    path = (
        Path(__file__).resolve().parents[2]
        / 'smoke/workflows/training/qwen38_qsa_profile_observer.py'
    )
    spec = importlib.util.spec_from_file_location('qsa_profile_observer', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def native(q, k, v, routes):
        return q * (k[routes] + v[routes])

    source = torch.randn(8, 4)
    observed = source.clone().requires_grad_()
    plain = source.clone().requires_grad_()

    def run(x, function):
        for _ in range(2):
            upstream = x[torch.tensor([0, 2, 4, 6])]
            function(
                upstream * 2, upstream * 3, upstream * 4, torch.arange(4)
            ).sum().backward()

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU]
    ) as prof:
        run(observed, module.observed_attention(native))
    run(plain, native)
    markers = [e for e in prof.events() if e.name == 'QSA_KV_INDEX_BACKWARD']
    assert len(markers) == 4, (
        'QSA_PROFILE_SCOPE_EXCLUDES_UPSTREAM_INDEX',
        len(markers),
    )
    assert torch.equal(observed.grad, plain.grad), 'QSA_PROFILE_SCOPE_GRAD_BITWISE'
