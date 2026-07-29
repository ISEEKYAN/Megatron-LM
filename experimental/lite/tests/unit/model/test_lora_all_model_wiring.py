# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Static coverage contracts for LoRA across every MLite model protocol."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest


MODEL_ROOT = Path(__file__).parents[3] / "megatron" / "lite" / "model"
MODEL_NAMES = ("qwen3_moe", "qwen3_5", "deepseek_v4", "glm5", "kimi_k2")


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_every_model_declares_lora_targets(model_name: str):
    adapter_path = MODEL_ROOT / model_name / "lite" / "lora_adapter.py"
    tree = ast.parse(adapter_path.read_text())
    assignments = {
        target.id
        for node in tree.body
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        for target in (
            node.targets
            if isinstance(node, ast.Assign)
            else (node.target,)
        )
        if isinstance(target, ast.Name)
    }
    assert "LORA_TARGETS" in assignments


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_every_model_applies_lora_before_optimizer_construction(model_name: str):
    protocol_path = MODEL_ROOT / model_name / "lite" / "protocol.py"
    tree = ast.parse(protocol_path.read_text())
    impl_config = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "ImplConfig"
    )
    fields = {
        node.target.id
        for node in impl_config.body
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
    }
    assert "lora" in fields

    build_model = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "build_model"
    )
    calls = [node for node in ast.walk(build_model) if isinstance(node, ast.Call)]
    lora_calls = [
        node
        for node in calls
        if isinstance(node.func, ast.Name) and node.func.id == "apply_lora_to_chunks"
    ]
    optimizer_calls = [
        node
        for node in calls
        if (
            isinstance(node.func, ast.Name)
            and (
                "optimizer" in node.func.id
                or node.func.id.startswith("_build_dist_opt")
            )
        )
        or (isinstance(node.func, ast.Attribute) and "optimizer" in node.func.attr)
    ]
    assert lora_calls
    assert optimizer_calls
    assert min(call.lineno for call in lora_calls) < min(
        call.lineno for call in optimizer_calls
    )
