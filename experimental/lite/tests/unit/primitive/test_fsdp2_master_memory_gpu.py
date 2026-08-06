# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from __future__ import annotations

import gc
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn
from megatron.lite.primitive.optimizers.fsdp2.adamw import FP32AdamW

pytestmark = [
    pytest.mark.mlite,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required."),
]


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


def test_fp32_master_memory_saves_one_parameter_cuda():
    device = torch.device("cuda")
    numel = 16 * 1024 * 1024
    expected_saved_bytes = numel * torch.float32.itemsize

    def measure(*, legacy_clone: bool) -> tuple[int, int]:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        param = nn.Parameter(torch.zeros(numel, device=device, dtype=torch.float32))
        param_bytes = torch.cuda.memory_stats(device)["allocated_bytes.all.current"]

        if legacy_clone:

            def legacy_init_master(optimizer, model_param):
                del optimizer
                return model_param.detach().to(dtype=torch.float32).clone()

            context = patch.object(FP32AdamW, "_init_master_param", legacy_init_master)
        else:
            context = patch.object(
                FP32AdamW, "_init_master_param", FP32AdamW._init_master_param
            )
        with context:
            optimizer = _build_optimizer(param, model_dtype=torch.float32)
        torch.cuda.synchronize(device)
        stats = torch.cuda.memory_stats(device)
        resident = stats["allocated_bytes.all.current"] - param_bytes
        peak = stats["allocated_bytes.all.peak"] - param_bytes

        del optimizer, param
        gc.collect()
        torch.cuda.empty_cache()
        return resident, peak

    legacy_resident, legacy_peak = measure(legacy_clone=True)
    shared_resident, shared_peak = measure(legacy_clone=False)
    resident_saved = legacy_resident - shared_resident
    peak_saved = legacy_peak - shared_peak

    print(
        "fp32 master memory: "
        f"legacy_resident={legacy_resident} shared_resident={shared_resident} "
        f"resident_saved={resident_saved} legacy_peak={legacy_peak} "
        f"shared_peak={shared_peak} peak_saved={peak_saved} "
        f"saved_gib={resident_saved / 1024**3:.6f} bytes_per_param=4"
    )
    assert resident_saved == expected_saved_bytes
    assert peak_saved == expected_saved_bytes


def test_fp32_master_three_configurations_cuda():
    fp32_param = nn.Parameter(torch.ones(8, device="cuda", dtype=torch.float32))
    fp32_optimizer = _build_optimizer(fp32_param, model_dtype=torch.float32)
    assert (
        fp32_optimizer.state[fp32_param]["master_param"].data_ptr()
        == fp32_param.data_ptr()
    )

    bf16_param = nn.Parameter(torch.ones(8, device="cuda", dtype=torch.bfloat16))
    bf16_optimizer = _build_optimizer(bf16_param, model_dtype=None)
    assert (
        bf16_optimizer.state[bf16_param]["master_param"].data_ptr()
        != bf16_param.data_ptr()
    )

    offloaded_param = nn.Parameter(torch.ones(8, device="cuda", dtype=torch.float32))
    offloaded_optimizer = _build_optimizer(
        offloaded_param, model_dtype=torch.float32, cpu_update=True
    )
    offloaded_master = offloaded_optimizer.state[offloaded_param]["master_param"]
    assert offloaded_master.device.type == "cpu"
    assert offloaded_master.data_ptr() != offloaded_param.data_ptr()


def test_fp32_shared_master_matches_cloned_reference_for_20_steps_cuda():
    torch.manual_seed(1234)
    initial = torch.randn(8, device="cuda", dtype=torch.float32)
    shared_param = nn.Parameter(initial.clone())
    reference_param = nn.Parameter(initial.clone())
    shared_optimizer = _build_optimizer(shared_param, model_dtype=torch.float32)
    reference_optimizer = _build_optimizer(reference_param, model_dtype=torch.float32)
    reference_optimizer.state[reference_param][
        "master_param"
    ] = reference_param.detach().clone()
    inputs = torch.randn(20, 8, device="cuda", dtype=torch.float32)
    targets = torch.randn(20, device="cuda", dtype=torch.float32)

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
