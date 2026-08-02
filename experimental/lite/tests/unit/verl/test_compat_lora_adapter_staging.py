# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""LoRA adapter staging patch: which device the adapter is staged on.

VERL hardcodes ``device="cpu"`` when handing adapter tensors to vLLM, and vLLM
turns pinned-host staging on for exactly that value. On a fused-MoE model the
adapter arrives as one tensor per (expert, projection, factor), so the per-tensor
cost is paid tens of thousands of times per weight sync.

These tests pin the *decision* the patch makes -- which device string reaches
vLLM and whether pinning stays on -- without needing vllm or verl importable,
because the interesting behaviour is entirely in the wrapper.

The fake vLLM module is rebuilt per test: the patch marks the function it has
already wrapped, so a shared fake would carry that mark into the next test and
make the idempotence check pass for the wrong reason.
"""

from __future__ import annotations

import sys
import types

import pytest
from verl_mlite import compat


@pytest.fixture
def fake_vllm(monkeypatch):
    """Install a fresh minimal ``vllm.lora.models`` and pretend vLLM is importable."""
    module = types.ModuleType("vllm.lora.models")
    calls: list[dict] = []

    class _FakeLoRAModel:
        @classmethod
        def from_lora_tensors(cls, lora_model_id, tensors, peft_helper, **kwargs):
            calls.append(
                {
                    "device": kwargs.get("device"),
                    # captured at call time: the patch toggles this for cpu_nopin
                    "pin_available": module.is_pin_memory_available(),
                }
            )
            return "lora-model"

    module.LoRAModel = _FakeLoRAModel
    module.is_pin_memory_available = lambda: True
    module.calls = calls

    pkg = types.ModuleType("vllm.lora")
    pkg.models = module
    monkeypatch.setitem(sys.modules, "vllm", types.ModuleType("vllm"))
    monkeypatch.setitem(sys.modules, "vllm.lora", pkg)
    monkeypatch.setitem(sys.modules, "vllm.lora.models", module)
    monkeypatch.setattr(compat, "_vllm_importable", lambda: True)
    yield module


def _invoke(fake_vllm):
    return fake_vllm.LoRAModel.from_lora_tensors(1, {}, object())


def test_default_mode_leaves_verl_behaviour_untouched(fake_vllm, monkeypatch):
    """The baseline arm must be a genuine no-op, or the A/B has no control."""
    monkeypatch.delenv(compat._LORA_ADAPTER_DEVICE_ENV, raising=False)
    assert compat._patch_vllm_lora_adapter_staging() is False
    _invoke(fake_vllm)
    # untouched: whatever VERL passes is what vLLM sees
    assert fake_vllm.calls[0]["device"] is None


def test_cpu_nopin_keeps_host_staging_but_disables_pinning(fake_vllm, monkeypatch):
    monkeypatch.setenv(compat._LORA_ADAPTER_DEVICE_ENV, "cpu_nopin")
    assert compat._patch_vllm_lora_adapter_staging() is True
    _invoke(fake_vllm)
    call = fake_vllm.calls[0]
    assert call["device"] == "cpu"
    assert call["pin_available"] is False, "pinning must be off inside the call"


def test_cpu_nopin_restores_pin_memory_probe_afterwards(fake_vllm, monkeypatch):
    """The override is scoped to the call; leaking it would change unrelated code."""
    monkeypatch.setenv(compat._LORA_ADAPTER_DEVICE_ENV, "cpu_nopin")
    compat._patch_vllm_lora_adapter_staging()
    _invoke(fake_vllm)
    assert fake_vllm.is_pin_memory_available() is True


def test_cpu_nopin_restores_probe_even_when_the_call_raises(fake_vllm, monkeypatch):
    """A failed adapter load must not leave pinning globally disabled."""
    monkeypatch.setenv(compat._LORA_ADAPTER_DEVICE_ENV, "cpu_nopin")

    def boom(cls, *a, **k):
        raise RuntimeError("adapter load failed")

    fake_vllm.LoRAModel.from_lora_tensors = classmethod(boom)
    assert compat._patch_vllm_lora_adapter_staging() is True
    with pytest.raises(RuntimeError):
        _invoke(fake_vllm)
    assert fake_vllm.is_pin_memory_available() is True


def test_cuda_mode_names_the_current_device(fake_vllm, monkeypatch):
    """Naming the device explicitly is what turns vLLM's pinning off upstream."""
    monkeypatch.setenv(compat._LORA_ADAPTER_DEVICE_ENV, "cuda")
    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = types.SimpleNamespace(
        is_available=lambda: True, current_device=lambda: 3
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    assert compat._patch_vllm_lora_adapter_staging() is True
    _invoke(fake_vllm)
    assert fake_vllm.calls[0]["device"] == "cuda:3"


def test_patch_is_idempotent(fake_vllm, monkeypatch):
    """apply_runtime_patches can run more than once per process."""
    monkeypatch.setenv(compat._LORA_ADAPTER_DEVICE_ENV, "cpu_nopin")
    assert compat._patch_vllm_lora_adapter_staging() is True
    assert compat._patch_vllm_lora_adapter_staging() is False


def test_unknown_mode_fails_loud(fake_vllm, monkeypatch):
    """A typo must not silently fall back to the baseline and void the arm."""
    monkeypatch.setenv(compat._LORA_ADAPTER_DEVICE_ENV, "gpu")
    with pytest.raises(ValueError, match=compat._LORA_ADAPTER_DEVICE_ENV):
        compat._patch_vllm_lora_adapter_staging()
