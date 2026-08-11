# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch
from megatron.lite.primitive.optimizers.fp32_adamw import FP32AdamW


@pytest.mark.parametrize("slice_capacity", [None, 3])
def test_fp32_adamw_matches_scalar_torch_adamw(slice_capacity: int | None):
    initial = [
        torch.tensor([1.0, -2.0, 3.0, -4.0, 5.0]),
        torch.tensor([0.25, -0.75]),
        torch.empty(0),
    ]
    candidate = [torch.nn.Parameter(value.clone()) for value in initial]
    reference = [torch.nn.Parameter(value.clone()) for value in initial]
    candidate_optimizer = FP32AdamW(
        [
            {"params": [candidate[0], candidate[2]], "weight_decay": 0.1},
            {"params": [candidate[1]], "weight_decay": 0.0},
        ],
        lr=0.03,
        weight_decay=0.1,
        betas=(0.8, 0.95),
        eps=1.0e-6,
        slice_capacity=slice_capacity,
    )
    reference_optimizer = torch.optim.AdamW(
        [
            {"params": [reference[0]], "weight_decay": 0.1},
            {"params": [reference[1]], "weight_decay": 0.0},
        ],
        lr=0.03,
        betas=(0.8, 0.95),
        eps=1.0e-6,
        foreach=False,
    )

    grads = [
        (torch.tensor([0.5, -0.25, 0.0, 1.0, -0.5]), None),
        (None, torch.tensor([0.125, -0.375])),
        (torch.tensor([-0.2, 0.4, -0.6, 0.8, -1.0]), torch.tensor([0.3, 0.1])),
    ]
    for grad0, grad1 in grads:
        candidate[0].grad = None if grad0 is None else grad0.clone()
        candidate[1].grad = None if grad1 is None else grad1.clone()
        candidate[2].grad = torch.empty(0)
        reference[0].grad = None if grad0 is None else grad0.clone()
        reference[1].grad = None if grad1 is None else grad1.clone()
        candidate_optimizer.step()
        reference_optimizer.step()

    for actual, expected in zip(candidate[:2], reference[:2], strict=True):
        torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)
    assert candidate_optimizer.state[candidate[0]]["step"] == 2
    assert candidate_optimizer.state[candidate[1]]["step"] == 2
    assert candidate_optimizer.state[candidate[2]]["step"] == 0


def test_sliced_update_advances_step_once_and_matches_unsliced():
    initial = torch.linspace(-1.0, 1.0, 11)
    sliced_param = torch.nn.Parameter(initial.clone())
    whole_param = torch.nn.Parameter(initial.clone())
    sliced = FP32AdamW(
        [sliced_param],
        lr=0.01,
        weight_decay=0.2,
        betas=(0.9, 0.99),
        eps=1.0e-8,
        slice_capacity=4,
    )
    whole = FP32AdamW(
        [whole_param], lr=0.01, weight_decay=0.2, betas=(0.9, 0.99), eps=1.0e-8
    )

    grad = torch.linspace(0.3, -0.4, 11)
    sliced_param.grad = grad.clone()
    whole_param.grad = grad.clone()
    sliced.step()
    whole.step()

    torch.testing.assert_close(sliced_param, whole_param, atol=0.0, rtol=0.0)
    for key in ("master_param", "exp_avg", "exp_avg_sq"):
        torch.testing.assert_close(
            sliced.state[sliced_param][key],
            whole.state[whole_param][key],
            atol=0.0,
            rtol=0.0,
        )
    assert sliced.state[sliced_param]["step"] == 1


def test_shared_fp32_adamw_has_no_backend_imports():
    source = Path(
        "experimental/lite/megatron/lite/primitive/optimizers/fp32_adamw.py"
    ).read_text()
    imports = [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    rendered = "\n".join(ast.unparse(node) for node in imports)

    for forbidden in (
        "fsdp2",
        "mfsdp",
        "DTensor",
        "torch.distributed",
        "megatron.lite.model",
    ):
        assert forbidden not in rendered
