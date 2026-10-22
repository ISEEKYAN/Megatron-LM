# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The documented entry performs a step and collectively drains adapter export."""
import importlib.util
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.optional


def test_example_calls_training_then_adapter_export(tmp_path, monkeypatch):
    source = Path(__file__).resolve().parents[3] / 'examples/deepseek_v41/lora_vllm.py'
    spec = importlib.util.spec_from_file_location('ds41_example', source)
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
    events = []
    key = 'layers.20.attn.wq_b.lora_B.weight'
    weight = torch.zeros(2, 2, dtype=torch.bfloat16)

    class Runtime:
        def train_mode(self, handle):
            return nullcontext()

        def zero_grad(self, handle):
            events.append('zero')

        def forward_backward(self, handle, data, loss_fn):
            assert list(data) == ['batch'] and loss_fn is None
            events.append('forward_backward')
            return SimpleNamespace(metrics={'loss': 1.0})

        def optimizer_step(self, handle):
            weight.add_(1)
            events.append('step')
            return True, 1.0, None

        def export_weights(self, handle, *, target):
            assert target == 'lora'
            events.append('export')
            yield key, weight

    monkeypatch.setattr(example.dist, 'get_rank', lambda: 0)
    monkeypatch.setattr(example.dist, 'barrier', lambda: events.append('barrier'))
    output = tmp_path / 'adapter.safetensors'
    example.train_and_export(Runtime(), object(), 'batch', 2, str(output))
    from safetensors.torch import load_file

    actual = load_file(output)
    assert list(actual) == [key] and torch.equal(actual[key], torch.full_like(weight, 2))
    assert events == ['zero', 'forward_backward', 'step'] * 2 + ['export', 'barrier']
