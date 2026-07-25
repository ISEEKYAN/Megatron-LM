# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Real-CUDA offload lifecycle for the FSDP2 Muon facade (AC#4).

The CPU correctness suite (``test_muon_fsdp2_unit.py``) drives the shared
offload movement path, but ``offload_state_to_cpu`` only moves tensors that are
*on CUDA* (``state.move_optimizer_state_to_cpu`` guards on ``is_cuda``), so a CPU
run is a genuine no-op and cannot prove the Muon state actually survives a
GPU->CPU->GPU round-trip.

These tests wrap a tiny model with FSDP2 so the Muon-managed matrices become
real DTensor shards on CUDA, build the production ``build_fsdp2_muon`` facade,
and assert that the **FP32Muon child's own** ``master_param`` and
``momentum_buffer`` DTensor state is offloaded to CPU and reloaded back onto CUDA
as a DTensor, with the parameter values preserved bit-for-bit and continued
training matching an uninterrupted run.

Run (single node, >=1 GPU):

    PYTHONPATH="$(pwd):$(pwd)/experimental/lite" \
      pytest experimental/lite/tests/unit/primitive/test_muon_fsdp2_offload_gpu.py
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn

from megatron.lite.primitive.optimizers.fsdp2 import (
    FSDP2Config,
    build_fsdp2_device_mesh,
    fsdp2_available,
    wrap_fsdp2,
)
from megatron.lite.primitive.optimizers.fsdp2.adamw import (
    iter_torch_optimizers,
    to_local_tensor,
)
from megatron.lite.primitive.optimizers.fsdp2.muon import FP32Muon
from megatron.lite.primitive.optimizers.fsdp2.optimizer import build_fsdp2_muon
from megatron.lite.primitive.optimizers.muon_routing import tag_muon_parameter_metadata
from megatron.lite.primitive.parallel.state import ParallelState
from megatron.lite.runtime.backends.mlite.runtime import MegatronLiteRuntime
from megatron.lite.runtime.contracts.handle import ModelHandle


class TinyUnit(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(8, 8)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.linear(x))


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.unit0 = TinyUnit()
        self.unit1 = TinyUnit()
        self.out = nn.Linear(8, 4)

    def forward(self, x):
        return self.out(self.unit1(self.unit0(x)))


@pytest.fixture(scope="module", autouse=True)
def _single_node_cuda_dist():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for FSDP2 Muon offload tests.")
    if not fsdp2_available():
        pytest.skip("Installed PyTorch does not expose FSDP2 fully_shard.")

    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29531")

    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    created_pg = False
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")
        created_pg = True
    yield
    if created_pg and dist.is_initialized():
        dist.destroy_process_group()


def _parallel_state() -> ParallelState:
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    return ParallelState(
        dp_group=dist.group.WORLD,
        dp_cp_group=dist.group.WORLD,
        dp_size=world_size,
        dp_cp_size=world_size,
        dp_rank=rank,
        dp_cp_rank=rank,
    )


def _muon_opt(offload_fraction: float = 0.0) -> SimpleNamespace:
    return SimpleNamespace(
        optimizer="muon",
        lr=1.0e-2,
        weight_decay=0.05,
        clip_grad=1.0,
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_eps=1.0e-8,
        muon_momentum=0.9,
        muon_nesterov=False,
        muon_split_qkv=False,
        muon_num_ns_steps=5,
        muon_coefficient_type="quintic",
        muon_scale_mode="spectral",
        muon_extra_scale_factor=1.0,
        muon_fp32_matmul_prec="medium",
        offload_fraction=offload_fraction,
    )


def _build_muon_fsdp2_model(
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[nn.Module, ParallelState]:
    torch.manual_seed(1234)
    model = TinyModel().cuda().to(dtype=dtype)
    ps = _parallel_state()
    # Tag Muon routing BEFORE wrapping so ``fully_shard`` preserves the metadata
    # onto the DTensor shards (matches ``build_fsdp2_training_optimizer``).
    tag_muon_parameter_metadata([model], is_expert_param=lambda name: False)
    config = FSDP2Config(unit_modules=(TinyUnit,), reshard_after_forward=True)
    mesh = build_fsdp2_device_mesh(ps, config)
    return wrap_fsdp2(model, ps, config, mesh=mesh), ps


def _build_muon_optimizer(
    model: nn.Module, ps: ParallelState, *, offload_fraction: float = 0.0
):
    return build_fsdp2_muon(
        [model], _muon_opt(offload_fraction), ps, use_fp32_master=True
    )


def _muon_child(optimizer) -> FP32Muon:
    for child in iter_torch_optimizers(optimizer.optimizer):
        if isinstance(child, FP32Muon):
            return child
    raise AssertionError("FP32Muon child not found in the chained optimizer.")


def _muon_state_devices(optimizer) -> set[str]:
    child = _muon_child(optimizer)
    devices: set[str] = set()
    for param_state in child.state.values():
        for key in ("master_param", "momentum_buffer"):
            value = param_state[key]
            assert isinstance(value, torch.Tensor)
            devices.add(to_local_tensor(value).device.type)
    return devices


def _assert_muon_state_is_dtensor(optimizer) -> None:
    from megatron.lite.primitive.optimizers.fsdp2.adamw import is_dtensor_like

    child = _muon_child(optimizer)
    assert child.state, "Muon child has no per-parameter state."
    for param_state in child.state.values():
        for key in ("master_param", "momentum_buffer"):
            assert is_dtensor_like(param_state[key]), (
                f"Muon {key} is not a DTensor; the FSDP2 shard did not lower to DTensor."
            )


def _local_param_devices(model: nn.Module) -> set[str]:
    return {to_local_tensor(param.detach()).device.type for param in model.parameters()}


def _muon_master_snapshot(optimizer) -> list[torch.Tensor]:
    child = _muon_child(optimizer)
    return [
        to_local_tensor(state["master_param"]).detach().cpu().clone()
        for state in child.state.values()
    ]


def _train_step(model: nn.Module, optimizer, x: torch.Tensor, target: torch.Tensor):
    optimizer.zero_grad()
    loss = torch.nn.functional.mse_loss(model(x).float(), target.float())
    loss.backward()
    success, grad_norm, _ = optimizer.step()
    assert success
    assert torch.isfinite(torch.tensor(grad_norm))
    return float(grad_norm)


def _local_named_params(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: to_local_tensor(param.detach()).cpu().clone()
        for name, param in model.named_parameters()
    }


def test_muon_optimizer_state_offload_roundtrip_single_gpu():
    """FP32Muon master+momentum DTensor state: CUDA -> CPU -> CUDA, values intact."""
    model, ps = _build_muon_fsdp2_model()
    optimizer = _build_muon_optimizer(model, ps, offload_fraction=0.0)
    handle = ModelHandle(
        model=model,
        optimizer=optimizer,
        parallel_state=ps,
        _extras={"model_chunks": [model]},
    )
    runtime = MegatronLiteRuntime.__new__(MegatronLiteRuntime)

    # One real step so the momentum buffer is populated (non-zero) before offload.
    x = torch.randn(4, 8, device="cuda", dtype=torch.bfloat16)
    target = torch.randn(4, 4, device="cuda", dtype=torch.bfloat16)
    _train_step(model, optimizer, x, target)

    # Pre-offload: Muon state lives on CUDA as DTensor shards.
    _assert_muon_state_is_dtensor(optimizer)
    assert _muon_state_devices(optimizer) == {"cuda"}
    before = _muon_master_snapshot(optimizer)

    # Offload the whole optimizer state (incl. the Muon child) to CPU.
    runtime.to(handle, "cpu", model=True, optimizer=True, grad=True)
    assert _muon_state_devices(optimizer) == {"cpu"}
    assert _local_param_devices(model) == {"cpu"}

    # Reload onto CUDA; the Muon state must be a DTensor again, values unchanged.
    runtime.to(handle, "cuda", model=True, optimizer=True, grad=True)
    _assert_muon_state_is_dtensor(optimizer)
    assert _muon_state_devices(optimizer) == {"cuda"}
    assert _local_param_devices(model) == {"cuda"}

    after = _muon_master_snapshot(optimizer)
    assert len(before) == len(after) and before
    for lhs, rhs in zip(before, after, strict=True):
        torch.testing.assert_close(lhs, rhs, atol=0.0, rtol=0.0)


def test_muon_offload_reload_then_resume_matches_uninterrupted_single_gpu():
    """A GPU->CPU->GPU offload mid-run must not perturb subsequent Muon updates."""
    torch.manual_seed(4321)
    xs = [torch.randn(4, 8, device="cuda", dtype=torch.bfloat16) for _ in range(3)]
    ys = [torch.randn(4, 4, device="cuda", dtype=torch.bfloat16) for _ in range(3)]

    # Reference: three uninterrupted steps.
    ref_model, ref_ps = _build_muon_fsdp2_model()
    ref_opt = _build_muon_optimizer(ref_model, ref_ps)
    for x, y in zip(xs, ys, strict=True):
        _train_step(ref_model, ref_opt, x, y)

    # Interrupted: step, offload to CPU + reload to CUDA (real DTensor movement),
    # then finish the remaining steps.
    run_model, run_ps = _build_muon_fsdp2_model()
    run_opt = _build_muon_optimizer(run_model, run_ps)
    handle = ModelHandle(
        model=run_model,
        optimizer=run_opt,
        parallel_state=run_ps,
        _extras={"model_chunks": [run_model]},
    )
    runtime = MegatronLiteRuntime.__new__(MegatronLiteRuntime)

    _train_step(run_model, run_opt, xs[0], ys[0])
    runtime.to(handle, "cpu", model=True, optimizer=True, grad=True)
    assert _muon_state_devices(run_opt) == {"cpu"}
    runtime.to(handle, "cuda", model=True, optimizer=True, grad=True)
    _assert_muon_state_is_dtensor(run_opt)
    for x, y in zip(xs[1:], ys[1:], strict=True):
        _train_step(run_model, run_opt, x, y)

    ref_params = _local_named_params(ref_model)
    run_params = _local_named_params(run_model)
    assert ref_params.keys() == run_params.keys()
    for name in ref_params:
        torch.testing.assert_close(
            run_params[name], ref_params[name], atol=0.0, rtol=0.0
        )
