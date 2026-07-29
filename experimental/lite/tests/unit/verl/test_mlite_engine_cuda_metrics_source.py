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
