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
    _iter_expert_adapter_placements,
    expected_global_expert_count,
    export_hf_lora_adapter,
    guard_expert_adapter_completeness,
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





def test_lora_spec_rollout_sync_is_declarative_and_round_trips():
    """Pin the *declared* default and that an explicit value survives normalization.

    This test deliberately does not claim to pin rollout behaviour. Nothing in
    the export path reads ``LoraSpec.rollout_sync``: the mode is resolved from
    the raw engine config by ``MegatronLiteEngine._lora_rollout_sync_is_merge``,
    pinned in tests/unit/verl/test_mlite_engine_lora_sync.py.

    A previous version asserted a ``merge`` default here and read as a guarantee
    of merged sync at runtime. It was not: the resolver defaulted to ``adapter``
    the whole time, so this suite stayed green while every run took the other
    path. The two defaults are now equal, and this test's job is to keep them
    equal -- not to stand in for the resolver's.
    """
    from megatron.lite.primitive.modules.lora import LoraSpec, normalize_lora_spec

    assert LoraSpec().rollout_sync == "adapter"
    assert normalize_lora_spec({"enabled": True, "rank": 8}).rollout_sync == "adapter"
    assert normalize_lora_spec({"enabled": True, "rank": 8, "rollout_sync": "merge"}).rollout_sync == "merge"

# --- expert identity under expert parallelism -------------------------------
#
# The adapter of an EP rank belongs to the experts that rank owns, and to no
# others: each rank initializes its own lora_A and grows its own lora_B from its
# own experts' gradients. Emitting a rank's adapter under the whole global expert
# range keeps the tensor count exactly right while misattributing seven of every
# eight experts, so these tests assert identity, not quantity.


class _FakePS:
    """Parallel state with EP only; gathering is simulated, no torch.distributed."""

    def __init__(self, ep_size, ep_rank, shards):
        self.ep_size = ep_size
        self.ep_rank = ep_rank
        self.ep_group = "fake" if ep_size > 1 else None
        self._shards = shards


def _placements(ps, num_experts, monkeypatch, rank_tag):
    """Expert ids this rank emits, with the adapter identity it used."""
    import megatron.lite.primitive.ckpt.hf_weights as hw

    monkeypatch.setattr(
        hw,
        "_gather_lora_factors_across_ep",
        lambda a, b, ps_: [(f"A{i}", f"B{i}") for i in range(ps_.ep_size)]
        if ps_.ep_size > 1
        else [(a, b)],
    )
    out = []
    for name, a, b in hw._iter_expert_adapter_placements(
        "layers.0.mlp.experts.fc1.weight0",
        f"A{rank_tag}",
        f"B{rank_tag}",
        is_grouped=True,
        num_experts=num_experts,
        ps=ps,
    ):
        out.append((int(name.rsplit("weight", 1)[1]), a, b))
    return out


def test_each_ep_rank_emits_only_the_experts_it_owns(monkeypatch):
    ep_size, num_experts = 8, 128
    per_rank = num_experts // ep_size
    for rank in range(ep_size):
        got = _placements(_FakePS(ep_size, rank, None), num_experts, monkeypatch, rank)
        by_identity = {}
        for eid, a, b in got:
            by_identity.setdefault((a, b), []).append(eid)
        # every adapter identity lands exactly on its owning rank's contiguous block
        for (a, _b), ids in by_identity.items():
            owner = int(a[1:])
            assert ids == list(range(owner * per_rank, (owner + 1) * per_rank)), (
                f"adapter {a} placed on {ids[:4]}..., expected block of rank {owner}"
            )


def test_global_expert_ids_are_covered_exactly_once(monkeypatch):
    ep_size, num_experts = 8, 128
    got = _placements(_FakePS(ep_size, 0, None), num_experts, monkeypatch, 0)
    ids = [eid for eid, _, _ in got]
    assert sorted(ids) == list(range(num_experts))       # complete
    assert len(ids) == len(set(ids))                     # no duplicates


def test_coverage_guard_cannot_see_the_ep_broadcast_bug():
    """Negative control that FAILED first, and what it taught.

    The broadcast bug emits, from every EP rank, all 128 global expert ids using
    that rank's own adapter. Within a single rank's stream that is one tensor per
    expert -- identical to correct output. So the coverage guard passes, exactly
    as the count guard did before it. Coverage is a real invariant (it catches
    missing and duplicated ids) but it is not this one, and pretending otherwise
    would leave the same blind spot one layer up.

    Identity is defended instead by the self-check inside
    ``_gather_lora_factors_across_ep`` (covered below), and by the placement
    tests above.
    """
    num_experts = 128
    broadcast_from_one_rank = [
        (f"base_model.model.model.layers.0.mlp.experts.{eid}.gate_proj.lora_A.weight", None)
        for eid in range(num_experts)
    ]
    # documents the limitation rather than asserting a capability it lacks
    assert (
        len(list(guard_expert_adapter_completeness(iter(broadcast_from_one_rank), num_experts)))
        == num_experts
    )


def test_ep_gather_self_identity_check_fires_on_foreign_data(monkeypatch):
    """The assertion that does defend expert identity."""
    import megatron.lite.primitive.ckpt.hf_weights as hw

    mine_a = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    mine_b = torch.arange(4, dtype=torch.float32).reshape(2, 2)
    ps = _FakePS(2, 1, None)

    def _bad_gather(outputs, tensor, group):
        for out in outputs:
            out.fill_(-1.0)  # nobody's slot holds our data

    monkeypatch.setattr(hw, "_ep_all_gather", _bad_gather)
    with pytest.raises(RuntimeError, match="foreign data in our own slot"):
        hw._gather_lora_factors_across_ep(mine_a, mine_b, ps)

    def _good_gather(outputs, tensor, group):
        for i, out in enumerate(outputs):
            out.copy_(tensor if i == ps.ep_rank else torch.full_like(tensor, float(i)))

    monkeypatch.setattr(hw, "_ep_all_gather", _good_gather)
    shards = hw._gather_lora_factors_across_ep(mine_a, mine_b, ps)
    assert len(shards) == 2
    assert torch.equal(shards[ps.ep_rank][0], mine_a)


def test_guard_accepts_correct_coverage_and_still_catches_a_missing_expert():
    num_experts = 128
    good = [
        (f"base_model.model.model.layers.0.mlp.experts.{eid}.gate_proj.lora_A.weight", None)
        for eid in range(num_experts)
    ]
    assert len(list(guard_expert_adapter_completeness(iter(good), num_experts))) == num_experts

    with pytest.raises(RuntimeError, match="exactly once"):
        list(guard_expert_adapter_completeness(iter(good[:-1]), num_experts))


def test_expected_global_expert_count_tracks_target_modules():
    assert expected_global_expert_count(num_experts=128, target_modules=["linear_fc1"]) == 128
    assert expected_global_expert_count(
        num_experts=128, target_modules=["linear_qkv", "linear_proj"]
    ) is None
    assert expected_global_expert_count(num_experts=None, target_modules=["linear_fc1"]) is None
