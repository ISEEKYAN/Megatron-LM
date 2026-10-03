# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Architecture guard for model HF checkpoint loaders."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_MODEL_CHECKPOINTS = (
    "qwen3_moe/lite/checkpoint.py",
    "qwen3_5/lite/checkpoint.py",
    "kimi_k2/lite/checkpoint.py",
    "glm5/lite/checkpoint.py",
    "deepseek_v4/lite/checkpoint.py",
)


@pytest.mark.parametrize("relative_path", _MODEL_CHECKPOINTS)
def test_model_hf_loaders_only_configure_the_primitive(relative_path: str) -> None:
    model_root = Path(__file__).resolve().parents[3] / "megatron" / "lite" / "model"
    source = (model_root / relative_path).read_text()
    tree = ast.parse(source)
    loader = next(
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "load_hf_weights"
    )

    assert not any(
        isinstance(node, (ast.For, ast.AsyncFor, ast.While, ast.With, ast.Dict))
        for node in ast.walk(loader)
    )
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_load"
        for node in ast.walk(loader)
    )
    assert not {
        "SafeTensorReader",
        "StreamingStateLoader",
        "TensorLoadSink",
    }.intersection(source)


def test_qwen_bounded_contract_and_public_primitive_boundary():
    import inspect

    from megatron.lite.primitive.ckpt import hf_weights
    from megatron.lite.primitive.ckpt.binding_records import TensorBinding
    from megatron.lite.primitive.ckpt.row_stream import stream_rows

    assert not hasattr(hf_weights, "export_raw_tensors")
    assert "role" not in TensorBinding.__dataclass_fields__
    assert "boundaries" not in inspect.signature(stream_rows).parameters
    for method in ("bindings", "encode", "decode", "validate_keys"):
        assert callable(getattr(hf_weights.BoundHFWeights, method))
    path = (
        Path(__file__).resolve().parents[3]
        / "megatron/lite/model/qwen3_moe/lite/checkpoint.py"
    )
    tree = ast.parse(path.read_text())
    primitive = "megatron.lite.primitive.ckpt.hf_weights"
    imports = {
        alias.asname or alias.name: alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == primitive
        for alias in node.names
    }
    assert not any(name.startswith("_") for name in imports.values())
    for function, bounded, ordinary in (
        ("load_hf_weights", "load_bound_model", "load_hf_weights"),
        ("export_hf_weights", "export_bound_tensors", "export_hf_weights"),
        ("save_hf_weights", "save_bound_model", "save_hf_weights"),
    ):
        node = next(
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == function
        )
        branch = next(
            n
            for n in node.body
            if isinstance(n, ast.If) and "bounded" in ast.unparse(n.test)
        )
        calls = lambda nodes: {
            imports.get(n.func.id)
            for statement in nodes
            for n in ast.walk(statement)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        }
        assert bounded in calls(branch.body)
        assert ordinary in calls([n for n in node.body if n is not branch])
        assert not any(
            isinstance(n, (ast.For, ast.While, ast.With)) for n in ast.walk(branch)
        )
