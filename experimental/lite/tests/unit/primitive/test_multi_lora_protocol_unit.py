# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Protocol-level contracts for model-owned multi-LoRA activation."""

import importlib
import sys
import types
from types import SimpleNamespace

import pytest
import torch.nn as nn


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
