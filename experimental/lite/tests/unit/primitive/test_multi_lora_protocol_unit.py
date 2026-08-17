# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Protocol-level contracts for model-owned multi-LoRA activation."""

import importlib
import sys
import types
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from megatron.lite.primitive.quantization.qat import QATSpec, apply_qat_to_chunks


def _protocol(monkeypatch):
    fake_model = types.ModuleType("megatron.lite.model.qwen3_moe.lite.model")
    fake_model.MTPLossAutoScaler = type("MTPLossAutoScaler", (), {})
    fake_model.Qwen3MoEModel = nn.Module
    monkeypatch.setitem(
        sys.modules, "megatron.lite.model.qwen3_moe.lite.model", fake_model
    )
    return importlib.import_module("megatron.lite.model.qwen3_moe.lite.protocol")


def test_model_owned_multi_lora_requires_slots_but_no_adapter_remains_a_noop(
    monkeypatch,
):
    protocol = _protocol(monkeypatch)

    class MissingSlotsBatch:
        extras = {}

    with pytest.raises(
        ValueError, match=r"enabled impl_cfg\.multi_lora requires multi_lora_slots"
    ):
        protocol._inject_multi_lora_sidecars(
            {}, MissingSlotsBatch(), SimpleNamespace(local_layer_indices=())
        )

    no_adapter_kwargs = {}
    protocol._inject_multi_lora_sidecars(no_adapter_kwargs, MissingSlotsBatch(), None)
    assert no_adapter_kwargs == {}

    class ValidSlotsBatch:
        extras = {"multi_lora_slots": {}}

    enabled_kwargs = {}
    protocol._inject_multi_lora_sidecars(
        enabled_kwargs, ValidSlotsBatch(), SimpleNamespace(local_layer_indices=())
    )
    assert enabled_kwargs == {"multi_lora_sidecars": {}}


def test_multi_lora_trainability_is_exactly_the_model_owned_bank_parameter_ids(
    monkeypatch,
):
    """Multi-LoRA must not silently turn the production bundle into full FT."""
    protocol = _protocol(monkeypatch)
    chunks = [nn.Linear(3, 4), nn.Linear(4, 2)]
    state = nn.Module()
    state.register_parameter("bank_a", nn.Parameter(torch.empty(2, 3)))
    state.register_parameter("bank_b", nn.Parameter(torch.empty(3, 2)))
    chunks[0].add_module("multi_lora_training_state", state)

    protocol._set_multi_lora_trainable_parameters(chunks, state)

    expected_ids = {id(parameter) for parameter in state.parameters()}
    trainable = [
        parameter
        for chunk in chunks
        for parameter in chunk.parameters()
        if parameter.requires_grad
    ]
    assert {id(parameter) for parameter in trainable} == expected_ids
    assert sum(parameter.numel() for parameter in trainable) == sum(
        parameter.numel() for parameter in state.parameters()
    )

    disabled_chunk = nn.Linear(2, 2)
    protocol._set_multi_lora_trainable_parameters([disabled_chunk], None)
    assert all(parameter.requires_grad for parameter in disabled_chunk.parameters())

    with pytest.raises(RuntimeError, match="must be registered"):
        protocol._set_multi_lora_trainable_parameters([nn.Linear(2, 2)], state)


def test_multi_lora_freeze_runs_after_qat_and_freezes_qat_master_weight(monkeypatch):
    protocol = _protocol(monkeypatch)

    class Chunk(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(32, 32, bias=False).bfloat16()

    chunk = Chunk()
    state = nn.Module()
    state.register_parameter("bank", nn.Parameter(torch.empty(2, 2, dtype=torch.bfloat16)))
    chunk.add_module("multi_lora_training_state", state)
    assert apply_qat_to_chunks([chunk], QATSpec(enabled=True, format="int8"))["quantized_modules"] == 1

    protocol._set_multi_lora_trainable_parameters([chunk], state)

    assert chunk.linear.parametrizations.weight.original.requires_grad is False
    assert all(parameter.requires_grad for parameter in state.parameters())
