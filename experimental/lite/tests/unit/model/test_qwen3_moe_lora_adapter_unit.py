# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""rsLoRA round-trip awareness in the Qwen3-MoE LoRA adapter import/validate path.

Guards the blocker that rsLoRA breaks scale->alpha inversion: native module scale is
alpha/sqrt(rank) under rsLoRA, so _infer_native_alpha must invert with sqrt(rank), and
_validate_adapter_config must check use_rslora rather than silently accepting a mismatch.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from megatron.lite.model.qwen3_moe.lite.lora_adapter import (
    _infer_native_alpha,
    _infer_native_use_rslora,
    _validate_adapter_config,
)
from megatron.lite.primitive.modules.lora import LinearLoRA

pytestmark = pytest.mark.mlite


def _fake_chunk(lora_module):
    attn = SimpleNamespace(qkv_lora=lora_module, proj_lora=None)
    moe = SimpleNamespace(experts=SimpleNamespace(fc1_lora=None, fc2_lora=None))
    return SimpleNamespace(layers=[SimpleNamespace(attn=attn, moe=moe)])


def test_infer_native_alpha_inverts_rslora_scale():
    # rank=4, alpha=8: rsLoRA scale = 8/2 = 4.0 -> must recover alpha=8 (not 16 from scale*rank).
    rs_chunk = _fake_chunk(LinearLoRA(3, 2, rank=4, alpha=8, use_rslora=True))
    assert _infer_native_alpha([rs_chunk]) == 8
    assert _infer_native_use_rslora([rs_chunk]) is True

    # standard LoRA scale = 8/4 = 2.0 -> recover alpha=8 as before.
    std_chunk = _fake_chunk(LinearLoRA(3, 2, rank=4, alpha=8))
    assert _infer_native_alpha([std_chunk]) == 8
    assert _infer_native_use_rslora([std_chunk]) is False


def test_validate_adapter_config_checks_use_rslora():
    rs_chunk = _fake_chunk(LinearLoRA(3, 2, rank=4, alpha=8, use_rslora=True))
    state: dict = {}
    matching = {"peft_type": "LORA", "lora_alpha": 8, "use_rslora": True}
    # matching rsLoRA flag + correctly-inverted alpha -> no error
    _validate_adapter_config([rs_chunk], state, matching)

    # adapter saved as standard LoRA loaded into an rsLoRA model must fail loudly
    with pytest.raises(ValueError, match="use_rslora"):
        _validate_adapter_config([rs_chunk], state, {"peft_type": "LORA", "use_rslora": False})

    # and the expected-config branch flags a mismatch too
    with pytest.raises(ValueError, match="use_rslora"):
        _validate_adapter_config(
            [rs_chunk], state, {"peft_type": "LORA", "use_rslora": True},
            lora_config={"rank": 4, "alpha": 8, "use_rslora": False},
        )
