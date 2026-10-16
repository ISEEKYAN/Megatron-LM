# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Recompute in backward must launch vLLM kernels on autograd's stream.

vLLM keeps a thread-local current stream and creates a fresh one on first use
in a thread; backward runs on the autograd thread. On GB200 this produced
gradients off by 1e5x for the batch-invariant head GEMM recomputed in backward.
"""

import sys
import threading
import types

import torch


def test_recompute_pins_vllm_stream_and_restores_it(monkeypatch):
    from megatron.lite.primitive import recompute

    tls = threading.local()
    tls.value = 'previous'
    fake = types.ModuleType('vllm.utils.torch_utils')
    fake._current_stream_tls = tls
    monkeypatch.setitem(sys.modules, 'vllm.utils.torch_utils', fake)
    autograd_stream = object()
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'is_initialized', lambda: True)
    monkeypatch.setattr(torch.cuda, 'current_stream', lambda *a, **k: autograd_stream)
    seen = []

    def run(x):
        if torch.is_grad_enabled():
            seen.append(getattr(tls, 'value', None))
        return x * 2

    x = torch.ones(3, requires_grad=True)
    out = recompute.CheckpointFunction.apply(run, False, x)
    tls.value = 'previous'
    out.sum().backward()
    # The backward-time recomputation ran with vLLM following autograd's stream.
    assert seen and all(value is autograd_stream for value in seen)
    assert getattr(tls, 'value', None) == 'previous'
    assert torch.equal(x.grad, torch.full((3,), 2.0))
