# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""LoRA wrapper contract for the generic HF exporter."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from megatron.lite.primitive.ckpt.hf_weights import (
    _is_adapter_param,
    _lora_export_surfaces,
    export_hf_weights,
)
from megatron.lite.primitive.modules.lora import LinearLoRA, SharedGroupedLinearLoRA
from megatron.lite.primitive.modules.lora import apply_olora_tail_init
from megatron.lite.primitive.modules.lora_apply import (
    LoRAWrappedGroupedLinear,
    LoRAWrappedLinear,
)

pytestmark = pytest.mark.mlite


class _IdentitySpec:
    num_experts = 2

    def native_to_hf(self, native_name, tensor):
        return [(native_name, tensor)]

    def tp_spec(self, native_name):
        return None

    def is_expert(self, native_name):
        return False


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
    def __init__(self):
        super().__init__()
        generator = torch.Generator().manual_seed(0)
        dense_adapter = LinearLoRA(8, 12, 2, alpha=4)
        grouped_adapter = SharedGroupedLinearLoRA(2, 8, 16, 2, alpha=4)
        with torch.no_grad():
            dense_adapter.lora_b.copy_(
                torch.randn(dense_adapter.lora_b.shape, generator=generator)
            )
            grouped_adapter.lora_b.copy_(
                torch.randn(grouped_adapter.lora_b.shape, generator=generator)
            )
        self.qkv = LoRAWrappedLinear(_Wrapped(12, 8), dense_adapter)
        self.experts = LoRAWrappedGroupedLinear(
            _GroupedBase(2, 16, 8, generator), grouped_adapter
        )


def _export_dict(model, *, merge_lora):
    return dict(
        export_hf_weights(
            model,
            _IdentitySpec(),
            _single_rank_ps(),
            merge_lora=merge_lora,
        )
    )


def test_wrapper_names_are_canonical_and_adapter_params_never_leak():
    model = _TinyWrappedModel()
    plain = _export_dict(model, merge_lora=False)

    assert set(plain) == {
        "qkv.linear.weight",
        "experts.weight0",
        "experts.weight1",
    }
    assert not any(_is_adapter_param(name) for name in plain)
    assert not any(".base." in name for name in plain)


def test_merge_lora_adds_dense_and_shared_expert_deltas():
    model = _TinyWrappedModel()
    plain = _export_dict(model, merge_lora=False)
    merged = _export_dict(model, merge_lora=True)

    dense_delta = model.qkv.adapter.materialized_delta_weight()
    assert torch.equal(merged["qkv.linear.weight"], plain["qkv.linear.weight"] + dense_delta)

    shared_delta = model.experts.adapter.materialized_delta_weight()
    for expert_idx in range(2):
        name = f"experts.weight{expert_idx}"
        assert torch.equal(merged[name], plain[name] + shared_delta)


def test_export_surface_map_covers_wrapper_base_weights():
    model = _TinyWrappedModel()
    canonical_names, delta_resolvers = _lora_export_surfaces(model)

    assert canonical_names == {
        "qkv.base.linear.weight": "qkv.linear.weight",
        "experts.base.weight0": "experts.weight0",
        "experts.base.weight1": "experts.weight1",
    }
    assert set(delta_resolvers) == {
        "qkv.linear.weight",
        "experts.weight0",
        "experts.weight1",
    }


@pytest.mark.parametrize("adapter_kind", ["dense", "grouped"])
def test_static_delta_uses_eval_semantics_with_training_dropout(adapter_kind):
    def make_adapter(dropout):
        if adapter_kind == "dense":
            return LinearLoRA(8, 16, 2, alpha=4, dropout=dropout)
        return SharedGroupedLinearLoRA(2, 8, 16, 2, alpha=4, dropout=dropout)

    without_dropout = make_adapter(0.0)
    with_dropout = make_adapter(0.1)
    with_dropout.load_state_dict(without_dropout.state_dict())

    torch.testing.assert_close(
        with_dropout.materialized_delta_weight(),
        without_dropout.materialized_delta_weight(),
        rtol=0,
        atol=0,
    )


def test_olora_tail_warns_when_grouped_expert_adapters_are_skipped():
    model = _TinyWrappedModel()

    with pytest.warns(
        UserWarning,
        match="OLoRA-tail does not support MoE grouped expert adapters",
    ):
        stats = apply_olora_tail_init(model)

    assert stats == {"initialized": 1, "skipped": 1}


def test_olora_tail_merged_export_matches_external_pretrained_base():
    """Residual-base training and merged rollout must represent one weight.

    The pre-init tensor is the independent reference: it comes from the plain
    model before either the OLoRA-tail transform or the export path runs.  This
    catches a self-consistent but jointly wrong residual/export pair.
    """
    model = _TinyWrappedModel()
    # Powers of two make subtract-then-add bitwise reversible, so this test can
    # distinguish a contract violation from ordinary floating-point roundoff.
    with torch.no_grad():
        model.qkv.base.linear.weight.fill_(3.0)
        model.qkv.adapter.lora_a.fill_(0.25)
        model.qkv.adapter.lora_b.fill_(0.5)
    external_base = _export_dict(model, merge_lora=False)["qkv.linear.weight"].clone()
    delta = model.qkv.adapter.materialized_delta_weight().clone()

    with pytest.warns(UserWarning, match="OLoRA-tail"):
        apply_olora_tail_init(model)

    residual_base = _export_dict(model, merge_lora=False)["qkv.linear.weight"]
    rollout_weight = _export_dict(model, merge_lora=True)["qkv.linear.weight"]
    training_weight = residual_base + delta

    assert not torch.equal(residual_base, external_base)
    torch.testing.assert_close(training_weight, external_base, rtol=0, atol=0)
    torch.testing.assert_close(rollout_weight, training_weight, rtol=0, atol=0)
