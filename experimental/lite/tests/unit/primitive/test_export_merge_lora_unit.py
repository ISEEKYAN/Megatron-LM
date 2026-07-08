# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Generic-exporter merge_lora contract (RL rollout weight sync on MoE namespaces).

The serving engine must see the CURRENT policy: every adapted base weight is
exported as ``base + scale·B@A`` (per LOCAL expert for grouped adapters), and
adapter/delta_mem parameters never leak into the HF stream.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from megatron.lite.primitive.ckpt.hf_weights import (
    _is_adapter_param,
    _lora_delta_resolvers,
    export_hf_weights,
)
from megatron.lite.primitive.modules.lora import (
    GroupedLinearLoRA,
    LinearLoRA,
    SharedGroupedLinearLoRA,
)

pytestmark = pytest.mark.mlite


class _IdentitySpec:
    """Minimal HFWeights spec: native names pass through, nothing is an expert."""

    def native_to_hf(self, native_name, tensor):
        return [(native_name, tensor)]

    def tp_spec(self, native_name):
        return None

    def qkv_spec(self, native_name):
        return None

    def is_expert(self, native_name):
        return False

    def expert_global_id(self, native_name):
        return None


def _single_rank_ps():
    return SimpleNamespace(
        tp_size=1, tp_rank=0, etp_size=1, etp_rank=0, ep_size=1, ep_rank=0,
        pp_size=1, pp_rank=0, tp_group=None, etp_group=None, ep_group=None, pp_group=None,
    )


class _Wrapped(nn.Module):
    def __init__(self, out_f, in_f):
        super().__init__()
        self.linear = nn.Linear(in_f, out_f, bias=False)


class _GroupedBase(nn.Module):
    def __init__(self, n, out_f, in_f, gen):
        super().__init__()
        for e in range(n):
            setattr(self, f"weight{e}", nn.Parameter(torch.randn(out_f, in_f, generator=gen)))


class _TinyMoEModel(nn.Module):
    """Mimics the mlite attribute convention the delta walk pairs on."""

    def __init__(self, shared_experts=False):
        super().__init__()
        gen = torch.Generator().manual_seed(0)
        attn = nn.Module()
        attn.qkv = _Wrapped(12, 8)
        attn.qkv_lora = LinearLoRA(8, 12, 2, alpha=4)
        attn.proj = _Wrapped(8, 8)
        attn.proj_lora = LinearLoRA(8, 8, 2, alpha=4)
        experts = nn.Module()
        experts.num_local_experts = 2
        experts.fc1 = _GroupedBase(2, 16, 8, gen)
        experts.fc2 = _GroupedBase(2, 8, 8, gen)
        if shared_experts:
            experts.fc1_lora = SharedGroupedLinearLoRA(2, 8, 16, 2, alpha=4)
            experts.fc2_lora = SharedGroupedLinearLoRA(2, 8, 8, 2, alpha=4)
        else:
            experts.fc1_lora = GroupedLinearLoRA(2, 8, 16, 2, alpha=4)
            experts.fc2_lora = GroupedLinearLoRA(2, 8, 8, 2, alpha=4)
        self.attn = attn
        self.experts = experts
        with torch.no_grad():  # nonzero B so deltas are nonzero
            for lora in (attn.qkv_lora, attn.proj_lora, experts.fc1_lora, experts.fc2_lora):
                lora.lora_b.copy_(torch.randn(lora.lora_b.shape, generator=gen))


def _export_dict(model, merge_lora):
    return dict(
        export_hf_weights(model, _IdentitySpec(), _single_rank_ps(), merge_lora=merge_lora)
    )


def test_adapter_params_never_leak_and_merge_adds_delta():
    model = _TinyMoEModel()
    plain = _export_dict(model, merge_lora=False)
    merged = _export_dict(model, merge_lora=True)

    assert not any(_is_adapter_param(name) for name in plain)
    assert plain.keys() == merged.keys()

    # Dense surfaces: merged == base + scale·B@A, bitwise.
    for base_attr, lora in (("qkv", model.attn.qkv_lora), ("proj", model.attn.proj_lora)):
        name = f"attn.{base_attr}.linear.weight"
        expected = plain[name] + lora.materialized_delta_weight()
        assert torch.equal(merged[name], expected)
        assert torch.count_nonzero(merged[name] - plain[name]) > 0

    # Grouped experts: per-LOCAL-expert delta.
    for e in range(2):
        name = f"experts.fc1.weight{e}"
        expected = plain[name] + model.experts.fc1_lora.materialized_delta_weight(e)
        assert torch.equal(merged[name], expected)
    # Distinct experts get distinct deltas (per-expert factors).
    d0 = model.experts.fc1_lora.materialized_delta_weight(0)
    d1 = model.experts.fc1_lora.materialized_delta_weight(1)
    assert torch.count_nonzero(d0 - d1) > 0


def test_shared_grouped_delta_is_identical_across_experts():
    model = _TinyMoEModel(shared_experts=True)
    merged = _export_dict(model, merge_lora=True)
    plain = _export_dict(model, merge_lora=False)
    shared_delta = model.experts.fc2_lora.materialized_delta_weight()
    for e in range(2):
        name = f"experts.fc2.weight{e}"
        # Same construction as the exporter (base + D) — bitwise; note (a+D)−a
        # is NOT bitwise D in floating point, so we compare sums, not diffs.
        assert torch.equal(merged[name], plain[name] + shared_delta)
    assert torch.equal(
        shared_delta, model.experts.fc2_lora.materialized_delta_weight(expert_idx=1)
    )


def test_delta_resolver_map_covers_all_adapted_surfaces():
    model = _TinyMoEModel()
    resolvers = _lora_delta_resolvers(model)
    assert set(resolvers) == {
        "attn.qkv.linear.weight",
        "attn.proj.linear.weight",
        "experts.fc1.weight0",
        "experts.fc1.weight1",
        "experts.fc2.weight0",
        "experts.fc2.weight1",
    }


def test_grouped_delta_rejects_dropout():
    lora = GroupedLinearLoRA(2, 8, 16, 2, alpha=4, dropout=0.1)
    with pytest.raises(NotImplementedError, match="dropout"):
        lora.materialized_delta_weight(0)
