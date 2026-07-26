# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Adapter-only LoRA export: naming contract and scale compensation.

The rollout engine applies its own LoRA scaling, which does *not* always match
the training-side scaling. These tests pin the invariant that matters:

    exported_B @ exported_A * (scaling the consumer will apply) == training delta

so a change on either side surfaces as a failing test rather than as a quietly
weakened rollout policy.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from megatron.lite.primitive.ckpt.hf_weights import (
    VLLM_LORA_NAME_PREFIX,
    export_hf_lora_adapter,
    vllm_applied_lora_scaling,
)
from megatron.lite.primitive.modules.lora import LinearLoRA, SharedGroupedLinearLoRA
from megatron.lite.primitive.modules.lora_apply import (
    LoRAWrappedGroupedLinear,
    LoRAWrappedLinear,
)

pytestmark = pytest.mark.mlite

NUM_EXPERTS = 2
RANK = 4
ALPHA = 16
IN_FEATURES = 8
DENSE_OUT = 12
EXPERT_OUT = 16


class _SplittingSpec:
    """Mimics a real spec: the expert surface fans out into two HF modules."""

    num_experts = NUM_EXPERTS

    def native_to_hf(self, native_name, tensor):
        if native_name.startswith("experts.weight"):
            expert_idx = int(native_name.rsplit("weight", 1)[1])
            gate, up = tensor.chunk(2, dim=0)
            return [
                (f"model.layers.0.mlp.experts.{expert_idx}.gate_proj.weight", gate),
                (f"model.layers.0.mlp.experts.{expert_idx}.up_proj.weight", up),
            ]
        return [("model.layers.0.self_attn.q_proj.weight", tensor)]

    def tp_spec(self, native_name):
        return None

    def is_expert(self, native_name):
        return native_name.startswith("experts.")


def _single_rank_ps():
    return SimpleNamespace(
        tp_size=1,
        tp_rank=0,
        etp_size=1,
        etp_rank=0,
        ep_size=1,
        ep_rank=0,
        pp_size=1,
        pp_rank=0,
        tp_group=None,
        etp_group=None,
        ep_group=None,
        pp_group=None,
    )


class _Wrapped(nn.Module):
    def __init__(self, out_features, in_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)

    def forward(self, x):
        return self.linear(x)


class _GroupedBase(nn.Module):
    def __init__(self, num_experts, out_features, in_features, generator):
        super().__init__()
        for expert_idx in range(num_experts):
            setattr(
                self,
                f"weight{expert_idx}",
                nn.Parameter(torch.randn(out_features, in_features, generator=generator)),
            )

    def forward(self, x, splits):
        del x, splits
        raise AssertionError("Exporter tests do not call forward")


class _TinyWrappedModel(nn.Module):
    def __init__(self, *, use_rslora: bool):
        super().__init__()
        generator = torch.Generator().manual_seed(0)
        dense_adapter = LinearLoRA(
            IN_FEATURES, DENSE_OUT, RANK, alpha=ALPHA, use_rslora=use_rslora
        )
        grouped_adapter = SharedGroupedLinearLoRA(
            NUM_EXPERTS, IN_FEATURES, EXPERT_OUT, RANK, alpha=ALPHA, use_rslora=use_rslora
        )
        with torch.no_grad():
            # lora_b initializes to zero; a zero adapter would pass any scaling.
            dense_adapter.lora_b.copy_(
                torch.randn(dense_adapter.lora_b.shape, generator=generator)
            )
            grouped_adapter.lora_b.copy_(
                torch.randn(grouped_adapter.lora_b.shape, generator=generator)
            )
        self.qkv = LoRAWrappedLinear(_Wrapped(DENSE_OUT, IN_FEATURES), dense_adapter)
        self.experts = LoRAWrappedGroupedLinear(
            _GroupedBase(NUM_EXPERTS, EXPERT_OUT, IN_FEATURES, generator), grouped_adapter
        )


def _export_dict(model):
    return dict(
        export_hf_lora_adapter(model, _SplittingSpec(), _single_rank_ps())
    )


def test_scaling_model_matches_documented_consumer_formulas():
    # Non-MoE honours rsLoRA; the FusedMoE packing path does not.
    assert vllm_applied_lora_scaling(
        RANK, ALPHA, use_rslora=True, packed_moe=False
    ) == pytest.approx(ALPHA / math.sqrt(RANK))
    assert vllm_applied_lora_scaling(
        RANK, ALPHA, use_rslora=False, packed_moe=False
    ) == pytest.approx(ALPHA / RANK)
    assert vllm_applied_lora_scaling(
        RANK, ALPHA, use_rslora=True, packed_moe=True
    ) == pytest.approx(ALPHA / RANK)
    assert vllm_applied_lora_scaling(
        RANK, ALPHA, use_rslora=False, packed_moe=True
    ) == pytest.approx(ALPHA / RANK)


def test_adapter_export_emits_peft_names_for_every_expert():
    exported = _export_dict(_TinyWrappedModel(use_rslora=True))

    expected = {f"{VLLM_LORA_NAME_PREFIX}model.layers.0.self_attn.q_proj.lora_A.weight"}
    expected.add(f"{VLLM_LORA_NAME_PREFIX}model.layers.0.self_attn.q_proj.lora_B.weight")
    for expert_idx in range(NUM_EXPERTS):
        for proj in ("gate_proj", "up_proj"):
            stem = f"model.layers.0.mlp.experts.{expert_idx}.{proj}"
            expected.add(f"{VLLM_LORA_NAME_PREFIX}{stem}.lora_A.weight")
            expected.add(f"{VLLM_LORA_NAME_PREFIX}{stem}.lora_B.weight")
    assert set(exported) == expected
    assert not any("lora_a" in name or "lora_b" in name for name in exported)


@pytest.mark.parametrize("use_rslora", [False, True])
def test_dense_adapter_reproduces_training_delta_after_consumer_scaling(use_rslora):
    model = _TinyWrappedModel(use_rslora=use_rslora)
    exported = _export_dict(model)
    stem = f"{VLLM_LORA_NAME_PREFIX}model.layers.0.self_attn.q_proj"

    consumer_scale = vllm_applied_lora_scaling(
        RANK, ALPHA, use_rslora=use_rslora, packed_moe=False
    )
    reconstructed = (
        exported[f"{stem}.lora_B.weight"] @ exported[f"{stem}.lora_A.weight"]
    ) * consumer_scale

    torch.testing.assert_close(
        reconstructed, model.qkv.adapter.materialized_delta_weight()
    )


@pytest.mark.parametrize("use_rslora", [False, True])
def test_expert_adapter_compensates_the_dropped_rslora_scaling(use_rslora):
    """The MoE path ignores rsLoRA, so lora_B must carry the correction."""
    model = _TinyWrappedModel(use_rslora=use_rslora)
    exported = _export_dict(model)

    training_delta = model.experts.adapter.materialized_delta_weight()
    gate_delta, up_delta = training_delta.chunk(2, dim=0)
    consumer_scale = vllm_applied_lora_scaling(
        RANK, ALPHA, use_rslora=use_rslora, packed_moe=True
    )

    for expert_idx in range(NUM_EXPERTS):
        for proj, expected_delta in (("gate_proj", gate_delta), ("up_proj", up_delta)):
            stem = f"{VLLM_LORA_NAME_PREFIX}model.layers.0.mlp.experts.{expert_idx}.{proj}"
            reconstructed = (
                exported[f"{stem}.lora_B.weight"] @ exported[f"{stem}.lora_A.weight"]
            ) * consumer_scale
            torch.testing.assert_close(reconstructed, expected_delta)


def test_rslora_compensation_is_actually_load_bearing():
    """Guard against a no-op compensation silently passing the other tests."""
    model = _TinyWrappedModel(use_rslora=True)
    exported = _export_dict(model)
    stem = f"{VLLM_LORA_NAME_PREFIX}model.layers.0.mlp.experts.0.gate_proj"

    # What vLLM would produce if we had shipped raw factors: alpha/rank instead
    # of the training-side alpha/sqrt(rank).
    uncompensated = (
        model.experts.adapter.lora_b @ model.experts.adapter.lora_a
    ).chunk(2, dim=0)[0] * (ALPHA / RANK)
    compensated = (
        exported[f"{stem}.lora_B.weight"] @ exported[f"{stem}.lora_A.weight"]
    ) * vllm_applied_lora_scaling(RANK, ALPHA, use_rslora=True, packed_moe=True)

    ratio = (compensated.norm() / uncompensated.norm()).item()
    assert ratio == pytest.approx(math.sqrt(RANK), rel=1e-5)
    assert not torch.allclose(compensated, uncompensated)
