# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from megatron.lite.primitive.optimizers.fsdp2.adamw import FP32AdamW

pytestmark = pytest.mark.mlite


def _build_optimizer(
    param: nn.Parameter, *, model_dtype: torch.dtype | None, cpu_update: bool = False
) -> FP32AdamW:
    model_param_dtypes = {id(param): model_dtype} if model_dtype is not None else None
    return FP32AdamW(
        [param],
        lr=0.01,
        weight_decay=0.1,
        betas=(0.9, 0.99),
        eps=1.0e-8,
        cpu_update=cpu_update,
        model_param_dtypes=model_param_dtypes,
    )


@pytest.mark.parametrize(
    ("param_dtype", "model_dtype", "cpu_update", "aliases_param"),
    [
        pytest.param(torch.float32, torch.float32, False, True, id="fp32-shards"),
        pytest.param(
            torch.float32, torch.bfloat16, False, True, id="bf16-model-fp32-shards"
        ),
        pytest.param(torch.bfloat16, None, False, False, id="non-fp32-shards"),
        pytest.param(torch.float32, torch.float32, True, False, id="cpu-update"),
    ],
)
def test_fp32_adamw_master_storage_ownership_cpu(
    param_dtype: torch.dtype,
    model_dtype: torch.dtype | None,
    cpu_update: bool,
    aliases_param: bool,
):
    param = nn.Parameter(torch.tensor([1.0, -2.0], dtype=param_dtype))
    optimizer = _build_optimizer(param, model_dtype=model_dtype, cpu_update=cpu_update)
    master = optimizer.state[param]["master_param"]

    assert (master.data_ptr() == param.data_ptr()) is aliases_param
    assert master.nbytes == param.numel() * torch.float32.itemsize


def test_fp32_shard_shared_master_skips_bf16_copyback_cpu():
    initial = torch.tensor([1.0], dtype=torch.float32)
    shared_param = nn.Parameter(initial.clone())
    reference_param = nn.Parameter(initial.clone())
    shared_optimizer = _build_optimizer(shared_param, model_dtype=torch.bfloat16)
    reference_optimizer = _build_optimizer(reference_param, model_dtype=torch.bfloat16)
    reference_optimizer.state[reference_param][
        "master_param"
    ] = reference_param.detach().clone()
    shared_master = shared_optimizer.state[shared_param]["master_param"]
    reference_master = reference_optimizer.state[reference_param]["master_param"]

    assert shared_master.data_ptr() == shared_param.data_ptr()
    grad = torch.tensor([0.1234567], dtype=torch.float32)
    shared_param.grad = grad.clone()
    reference_param.grad = grad.clone()
    shared_optimizer.step()
    reference_optimizer.step()

    assert torch.equal(shared_master, reference_master)
    assert torch.equal(shared_param, shared_master)
    assert torch.equal(
        reference_param, reference_master.to(torch.bfloat16).to(torch.float32)
    )
    assert not torch.equal(reference_param, reference_master)


def test_fp32_adamw_shared_master_matches_cloned_reference_for_20_steps_cpu():
    torch.manual_seed(1234)
    initial = torch.randn(8, dtype=torch.float32)
    shared_param = nn.Parameter(initial.clone())
    reference_param = nn.Parameter(initial.clone())
    shared_optimizer = _build_optimizer(shared_param, model_dtype=torch.float32)
    reference_optimizer = _build_optimizer(reference_param, model_dtype=torch.float32)
    reference_optimizer.state[reference_param][
        "master_param"
    ] = reference_param.detach().clone()
    inputs = torch.randn(20, 8, dtype=torch.float32)
    targets = torch.randn(20, dtype=torch.float32)

    shared_master = shared_optimizer.state[shared_param]["master_param"]
    assert shared_master.data_ptr() == shared_param.data_ptr()
    for step in range(20):
        shared_loss = (shared_param.dot(inputs[step]) - targets[step]).square()
        reference_loss = (reference_param.dot(inputs[step]) - targets[step]).square()
        assert torch.equal(shared_loss, reference_loss), f"loss differs at step {step}"

        shared_loss.backward()
        reference_loss.backward()
        shared_optimizer.step()
        reference_optimizer.step()
        shared_optimizer.zero_grad(set_to_none=True)
        reference_optimizer.zero_grad(set_to_none=True)

        reference_master = reference_optimizer.state[reference_param]["master_param"]
        assert torch.equal(
            shared_master, reference_master
        ), f"master differs at step {step}"
        assert torch.equal(
            shared_param, reference_param
        ), f"param differs at step {step}"
