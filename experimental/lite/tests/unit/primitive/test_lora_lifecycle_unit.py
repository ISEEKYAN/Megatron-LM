# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Fail-loud mirror-operation contracts for the generic LoRA primitive."""

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn

from megatron.lite.primitive.modules.lora import (
    LinearLoRA,
    LoraSpec,
    SharedGroupedLinearLoRA,
)
from megatron.lite.primitive.modules.lora_apply import (
    LoRAWrappedGroupedLinear,
    LoRAWrappedLinear,
    LoraTargetRule,
    apply_lora_to_chunks,
    load_lora_adapter_state,
    merge_lora_in_chunks,
    remove_lora_from_chunks,
    save_lora_adapter_state,
    unmerge_lora_in_chunks,
)

pytestmark = pytest.mark.mlite


@pytest.fixture(autouse=True)
def _te_stub(transformer_engine_import_stub):
    transformer_engine_import_stub()


class _ToyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(4, 3, bias=False)

    def forward(self, x):
        return self.proj(x)


_TOY_TARGETS = (LoraTargetRule("_ToyBlock", "proj", "linear_proj"),)


def _apply_toy(model):
    return apply_lora_to_chunks(
        [model],
        LoraSpec(enabled=True, rank=2, alpha=2, dropout=0.0),
        model_targets=_TOY_TARGETS,
    )


def _make_effective(model):
    with torch.no_grad():
        model.proj.adapter.lora_b.fill_(0.25)


def test_apply_remove_round_trip_is_observable_and_fails_loud():
    torch.manual_seed(7)
    model = _ToyBlock()
    x = torch.randn(2, 4)
    base_state = copy.deepcopy(model.state_dict())
    base_output = model(x)

    with pytest.raises(RuntimeError, match="no LoRA wrappers"):
        remove_lora_from_chunks([model])

    stats = _apply_toy(model)
    _make_effective(model)
    assert stats["attached_modules"] == 1
    assert isinstance(model.proj, LoRAWrappedLinear)
    assert not torch.equal(model(x), base_output)

    removed = remove_lora_from_chunks([model])
    assert removed["removed_modules"] == 1
    assert isinstance(model.proj, nn.Linear)
    torch.testing.assert_close(model.state_dict(), base_state)
    torch.testing.assert_close(model(x), base_output)

    with pytest.raises(RuntimeError, match="no LoRA wrappers"):
        remove_lora_from_chunks([model])


def test_apply_no_matching_target_fails_without_freezing_the_model():
    model = _ToyBlock()
    before = copy.deepcopy(model.state_dict())
    wrong_targets = (LoraTargetRule("AbsentBlock", "proj", "linear_proj"),)

    with pytest.raises(RuntimeError, match="no declared target modules matched"):
        apply_lora_to_chunks(
            [model], LoraSpec(enabled=True, rank=2), model_targets=wrong_targets
        )

    torch.testing.assert_close(model.state_dict(), before)
    assert all(parameter.requires_grad for parameter in model.parameters())
    assert not hasattr(model, "_mlite_lora_requires_grad_state")


def test_merge_unmerge_round_trip_preserves_dense_adapter_output_and_fails_loud():
    torch.manual_seed(11)
    model = _ToyBlock()
    x = torch.randn(3, 4)

    with pytest.raises(RuntimeError, match="no LoRA wrappers"):
        merge_lora_in_chunks([model])

    _apply_toy(model)
    _make_effective(model)
    adapted = model(x).detach().clone()
    base_weight = model.proj.base.weight.detach().clone()

    assert merge_lora_in_chunks([model]) == {"merged_modules": 1}
    torch.testing.assert_close(model(x), adapted)
    assert not torch.equal(model.proj.base.weight, base_weight)
    with pytest.raises(RuntimeError, match="already merged"):
        merge_lora_in_chunks([model])

    assert unmerge_lora_in_chunks([model]) == {"unmerged_modules": 1}
    torch.testing.assert_close(model(x), adapted)
    torch.testing.assert_close(model.proj.base.weight, base_weight)
    with pytest.raises(RuntimeError, match="not merged"):
        unmerge_lora_in_chunks([model])


class _GroupedBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight0 = nn.Parameter(torch.randn(3, 4))
        self.weight1 = nn.Parameter(torch.randn(3, 4))

    def forward(self, x, splits):
        parts = x.split(splits, dim=0)
        return torch.cat(
            [
                torch.nn.functional.linear(part, getattr(self, f"weight{index}"))
                for index, part in enumerate(parts)
            ],
            dim=0,
        )


def test_merge_unmerge_covers_grouped_expert_weights():
    base = _GroupedBase()
    adapter = SharedGroupedLinearLoRA(2, 4, 3, 2, alpha=2)
    with torch.no_grad():
        adapter.lora_b.fill_(0.5)
    model = nn.Module()
    model.experts = LoRAWrappedGroupedLinear(base, adapter)
    x = torch.randn(4, 4)
    splits = [1, 3]
    adapted = model.experts(x, splits).detach().clone()
    before = [base.weight0.detach().clone(), base.weight1.detach().clone()]

    merge_lora_in_chunks([model])
    torch.testing.assert_close(model.experts(x, splits), adapted)
    unmerge_lora_in_chunks([model])
    torch.testing.assert_close(model.experts(x, splits), adapted)
    torch.testing.assert_close(base.weight0, before[0])
    torch.testing.assert_close(base.weight1, before[1])


def test_save_load_round_trip_and_negative_paths(tmp_path):
    torch.manual_seed(13)
    source = _ToyBlock()
    target = _ToyBlock()
    target.load_state_dict(source.state_dict())
    path = tmp_path / "adapter.pt"

    with pytest.raises(RuntimeError, match="no LoRA adapters"):
        save_lora_adapter_state([source], path)
    with pytest.raises(FileNotFoundError, match="LoRA adapter checkpoint"):
        load_lora_adapter_state([source], tmp_path / "missing.pt")

    _apply_toy(source)
    _apply_toy(target)
    _make_effective(source)
    save_stats = save_lora_adapter_state([source], path)
    assert save_stats["saved_tensors"] == 2
    load_stats = load_lora_adapter_state([target], path)
    assert load_stats == {"loaded_tensors": 2}
    x = torch.randn(2, 4)
    torch.testing.assert_close(target(x), source(x))

    incompatible = _ToyBlock()
    with pytest.raises(RuntimeError, match="does not match attached adapters"):
        load_lora_adapter_state([incompatible], path)


def test_unmerge_fails_before_any_partial_mutation():
    model = _ToyBlock()
    _apply_toy(model)
    before = model.proj.base.weight.detach().clone()
    with pytest.raises(RuntimeError, match="not merged"):
        unmerge_lora_in_chunks([model])
    torch.testing.assert_close(model.proj.base.weight, before)


def test_qwen_protocol_rejects_lora_etp_at_entry_before_parallel_init(monkeypatch):
    from megatron.lite.model.qwen3_moe.config import Qwen3MoEConfig
    from megatron.lite.model.qwen3_moe.lite import protocol
    from megatron.lite.runtime.contracts import ParallelConfig

    monkeypatch.setattr(
        protocol,
        "init_parallel",
        lambda *_args, **_kwargs: pytest.fail("parallel init must not run"),
    )
    impl_cfg = protocol.ImplConfig(
        parallel=ParallelConfig(etp=2),
        optimizer=None,
        lora={"enabled": True, "rank": 2},
    )
    with pytest.raises(NotImplementedError, match="LoRA does not support ETP>1"):
        protocol.build_model(Qwen3MoEConfig(), impl_cfg=impl_cfg)
