# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Static coverage contracts for LoRA across every MLite model protocol."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest


MODEL_ROOT = Path(__file__).parents[3] / "megatron" / "lite" / "model"
MODEL_NAMES = ("qwen3_moe", "qwen3_5", "deepseek_v4", "glm5", "kimi_k2")


def _has_model_targets_lora_binding(tree: ast.AST) -> bool:
    """Return whether an apply call binds ``model_targets`` to ``LORA_TARGETS``."""
    return any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "apply_lora_to_chunks"
        and any(
            keyword.arg == "model_targets"
            and isinstance(keyword.value, ast.Name)
            and keyword.value.id == "LORA_TARGETS"
            for keyword in node.keywords
        )
        for node in ast.walk(tree)
    )

@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_every_model_binds_lora_targets_to_apply_call(model_name: str):
    adapter_path = MODEL_ROOT / model_name / "lite" / "lora_adapter.py"
    tree = ast.parse(adapter_path.read_text())
    assert any(
        isinstance(node, (ast.Assign, ast.AnnAssign))
        and any(
            isinstance(target, ast.Name) and target.id == "LORA_TARGETS"
            for target in (node.targets if isinstance(node, ast.Assign) else (node.target,))
        )
        for node in tree.body
    )

    protocol_path = MODEL_ROOT / model_name / "lite" / "protocol.py"
    assert _has_model_targets_lora_binding(ast.parse(protocol_path.read_text()))


def test_model_targets_binding_rejects_another_constant():
    tree = ast.parse(
        "apply_lora_to_chunks(chunks, lora_config, model_targets=OTHER_TARGETS)"
    )

    assert not _has_model_targets_lora_binding(tree)


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
    validation_calls = [
        node
        for node in calls
        if isinstance(node.func, ast.Name)
        and node.func.id == "validate_lora_parallel_support"
    ]
    init_parallel_calls = [
        node
        for node in calls
        if isinstance(node.func, ast.Name) and node.func.id == "init_parallel"
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
    assert validation_calls
    assert init_parallel_calls
    assert min(call.lineno for call in validation_calls) < min(
        call.lineno for call in init_parallel_calls
    )
    assert optimizer_calls
    assert min(call.lineno for call in lora_calls) < min(
        call.lineno for call in optimizer_calls
    )
    assert _has_model_targets_lora_binding(tree)


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_every_model_exports_lora_adapter(model_name: str):
    """Adapter rollout must be implemented uniformly, not just declared by runtime."""
    protocol_path = MODEL_ROOT / model_name / "lite" / "protocol.py"
    tree = ast.parse(protocol_path.read_text())
    adapter_export = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "export_hf_lora_adapter"
        ),
        None,
    )

    assert adapter_export is not None
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_export_hf_lora_adapter_impl"
        for node in ast.walk(adapter_export)
    )

    checkpoint_path = MODEL_ROOT / model_name / "lite" / "checkpoint.py"
    checkpoint_tree = ast.parse(checkpoint_path.read_text())
    checkpoint_export = next(
        (
            node
            for node in checkpoint_tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "export_hf_lora_adapter"
        ),
        None,
    )
    assert checkpoint_export is not None
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_export_adapter"
        for node in ast.walk(checkpoint_export)
    )
