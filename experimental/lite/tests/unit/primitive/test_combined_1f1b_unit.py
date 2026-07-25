# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""CPU contract tests for the combined-1F1B EP-overlap driver."""

from copy import deepcopy
from dataclasses import dataclass

import pytest
import torch
from megatron.lite.primitive.parallel.combined_1f1b import (
    Combined1F1BConfig,
    Combined1F1BModelPlan,
    build_combined_1f1b_trace,
    run_combined_1f1b,
)
from torch import nn


@dataclass
class _FakePlan:
    microbatch: int
    calls: list[tuple]

    def forward(self):
        self.calls.append(("forward", self.microbatch))
        return {"loss": self.microbatch}

    def combined_forward_backward(self, backward_plan):
        self.calls.append(("combined", self.microbatch, backward_plan.microbatch))
        return {"loss": self.microbatch}

    def backward(self):
        self.calls.append(("backward", self.microbatch))


def test_combined_1f1b_trace_matches_megatron_phase_order():
    assert build_combined_1f1b_trace(4) == [
        (0, None),
        (1, 0),
        (2, 1),
        (3, 2),
        (None, 3),
    ]


def test_combined_1f1b_driver_runs_adjacent_microbatches():
    calls = []
    plans = [_FakePlan(i, calls) for i in range(4)]

    outputs = run_combined_1f1b(plans)

    assert calls == [
        ("forward", 0),
        ("combined", 1, 0),
        ("combined", 2, 1),
        ("combined", 3, 2),
        ("backward", 3),
    ]
    assert outputs == [{"loss": 0}, {"loss": 1}, {"loss": 2}, {"loss": 3}]


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList(
            [nn.ModuleList([nn.Linear(4, 4) for _ in range(4)]) for _ in range(2)]
        )
        self.head = nn.Linear(4, 1)

    def forward(self, x):
        for layer in self.layers:
            for node in layer:
                x = torch.tanh(node(x))
        return self.head(x)


def _model_plan(model, x, target, num_microbatches):
    layer_callables = [
        [lambda value, node=node: torch.tanh(node(value)) for node in layer]
        for layer in model.layers
    ]
    return Combined1F1BModelPlan(
        preprocess=lambda: x,
        layer_callables=layer_callables,
        postprocess=lambda hidden: {
            "loss": torch.nn.functional.mse_loss(model.head(hidden), target)
        },
        num_microbatches=num_microbatches,
        use_cuda=False,
    )


def test_combined_1f1b_gradients_match_sequential_microbatch_backward():
    torch.manual_seed(123)
    reference = _TinyModel()
    combined = deepcopy(reference)
    batches = [
        (torch.randn(3, 4), torch.randn(3, 1)),
        (torch.randn(3, 4), torch.randn(3, 1)),
        (torch.randn(3, 4), torch.randn(3, 1)),
    ]

    for x, target in batches:
        (torch.nn.functional.mse_loss(reference(x), target) / len(batches)).backward()

    run_combined_1f1b(
        [_model_plan(combined, x, target, len(batches)) for x, target in batches]
    )

    for ref_param, combined_param in zip(
        reference.parameters(), combined.parameters(), strict=True
    ):
        torch.testing.assert_close(combined_param.grad, ref_param.grad)


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"num_microbatches": 1}, "at least 2 microbatches"),
        ({"ep_size": 1}, "EP > 1"),
        ({"pp_size": 2}, "PP=1"),
        ({"use_deepep": True}, "alltoall"),
        ({"recompute": ("full",)}, "recompute"),
        ({"moe_permute_fusion": True}, "permute fusion"),
    ],
)
def test_combined_1f1b_config_fails_loud_outside_first_profile(override, message):
    values = {
        "num_microbatches": 2,
        "ep_size": 8,
        "pp_size": 1,
        "use_deepep": False,
        "recompute": (),
        "moe_permute_fusion": False,
    }
    values.update(override)

    with pytest.raises(ValueError, match=message):
        Combined1F1BConfig(**values).validate()
