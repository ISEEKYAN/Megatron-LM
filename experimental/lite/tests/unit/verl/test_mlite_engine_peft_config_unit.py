# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""LoRA config parity at the training-to-vLLM boundary."""

from __future__ import annotations

import ast
import copy
import sys
import types
from pathlib import Path

import pytest
from megatron.lite.primitive.ckpt.hf_weights import vllm_applied_lora_scaling
from megatron.lite.primitive.modules import lora as lora_module


def _load_production_vllm_peft_builder():
    """Load the production method without importing optional VERL dependencies."""
    engine_path = (
        Path(__file__).parents[3]
        / "examples/verl/verl_mlite/engine/mlite_engine.py"
    )
    tree = ast.parse(engine_path.read_text())
    engine_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "MegatronLiteEngine"
    )
    method = copy.deepcopy(
        next(
            node
            for node in engine_class.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_build_vllm_peft_config"
        )
    )
    namespace = {
        name: getattr(lora_module, name)
        for name in (
            "LORA_DEFAULT_ALPHA",
            "LORA_DEFAULT_DROPOUT",
            "LORA_DEFAULT_RANK",
            "LORA_DEFAULT_TARGET_MODULES",
            "LORA_DEFAULT_USE_RSLORA",
            "assert_lora_alpha_scaling_consistent",
            "resolve_lora_alpha",
        )
        if hasattr(lora_module, name)
    }
    module = ast.fix_missing_locations(ast.Module([method], []))
    exec(compile(module, engine_path, "exec"), namespace)
    return namespace[method.name]


def _install_identity_verl_target_converter(monkeypatch):
    verl = types.ModuleType("verl")
    utils = types.ModuleType("verl.utils")
    peft_utils = types.ModuleType("verl.utils.megatron_peft_utils")
    peft_utils.convert_megatron_to_hf_target_modules = list
    verl.utils = utils
    utils.megatron_peft_utils = peft_utils
    monkeypatch.setitem(sys.modules, "verl", verl)
    monkeypatch.setitem(sys.modules, "verl.utils", utils)
    monkeypatch.setitem(sys.modules, "verl.utils.megatron_peft_utils", peft_utils)


def test_omitted_alpha_matches_training_and_production_vllm_builder(monkeypatch):
    """One raw config must resolve identically on both sides."""
    _install_identity_verl_target_converter(monkeypatch)
    builder = _load_production_vllm_peft_builder()
    spec = lora_module.LoraSpec(enabled=True, rank=8)

    peft_config = builder(None, {"enabled": True, "rank": spec.rank})
    rollout_alpha = peft_config["lora_alpha"]
    rollout_scale = vllm_applied_lora_scaling(
        spec.rank,
        rollout_alpha,
        use_rslora=spec.use_rslora,
        packed_moe=False,
    )

    assert rollout_alpha == spec.rank
    assert rollout_scale == spec.scale
    assert peft_config["r"] == spec.rank
    assert peft_config["lora_dropout"] == spec.dropout
    assert peft_config["target_modules"] == list(spec.target_modules)


@pytest.mark.parametrize(
    "applied_alpha,applied_use_rslora",
    [
        pytest.param(32, False, id="old-alpha-default"),
        pytest.param(8, True, id="scaling-mode-mismatch"),
    ],
)
def test_alpha_scaling_contract_fails_loud_on_injected_rollout_mismatch(
    applied_alpha,
    applied_use_rslora,
):
    """Reverse check: an injected alpha or scaling mismatch must stop immediately."""
    with pytest.raises(RuntimeError, match="alpha/scaling disagrees"):
        lora_module.assert_lora_alpha_scaling_consistent(
            8,
            None,
            applied_alpha=applied_alpha,
            applied_use_rslora=applied_use_rslora,
        )
