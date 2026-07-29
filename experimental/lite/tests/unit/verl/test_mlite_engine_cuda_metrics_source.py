# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import ast
from pathlib import Path


def test_reduced_outputs_path_collects_cuda_allocator_metrics():
    source_path = (
        Path(__file__).parents[3]
        / "examples"
        / "verl"
        / "verl_mlite"
        / "engine"
        / "mlite_engine.py"
    )
    tree = ast.parse(source_path.read_text())
    method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "_forward_backward_batch_with_runtime"
    )
    reduced_outputs_branch = next(
        node
        for node in method.body
        if isinstance(node, ast.If)
        and "reduced_outputs is not None" in ast.unparse(node.test)
    )
    calls = {
        node.func.id
        for node in ast.walk(reduced_outputs_branch)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }

    assert {"cuda_allocator_metrics", "pop_workspace_shape_metrics"} <= calls


def test_full_train_batch_is_wrapped_by_nsys_step_range():
    source_path = (
        Path(__file__).parents[3]
        / "examples"
        / "verl"
        / "verl_mlite"
        / "engine"
        / "mlite_engine.py"
    )
    tree = ast.parse(source_path.read_text())
    method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "train_batch"
    )
    train_call = next(
        node
        for node in ast.walk(method)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "train_batch"
    )
    enclosing_with = next(
        node
        for node in ast.walk(method)
        if isinstance(node, ast.With) and train_call in list(ast.walk(node))
    )

    assert any(
        isinstance(item.context_expr, ast.Call)
        and isinstance(item.context_expr.func, ast.Name)
        and item.context_expr.func.id == "_nsys_profile_step"
        for item in enclosing_with.items
    )
