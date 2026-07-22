# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn

from megatron.lite.primitive.optimizers.fsdp2 import (
    FSDP2Config,
    build_fsdp2_adamw,
    build_fsdp2_device_mesh,
    fsdp2_available,
    wrap_fsdp2,
)
from megatron.lite.primitive.optimizers.fsdp2.adamw import iter_torch_optimizers, to_local_tensor
from megatron.lite.primitive.parallel.state import ParallelState
from megatron.lite.runtime.backends.mlite.runtime import MegatronLiteRuntime
from megatron.lite.runtime.contracts.handle import ModelHandle


class TinyFP8Model(nn.Module):
    """A blockwise TE MLP whose parameters are constructed as FP8 weights."""

    def __init__(self):
        super().__init__()
        from megatron.lite.primitive.modules.mlp import SwiGLUMLP
        from megatron.lite.primitive.precision import (
            PrecisionCoverage,
            precision_model_init_context,
            resolve_precision,
        )

        implementation = resolve_precision("hopper_blockwise_fp8_weight")
        assert implementation is not None
        coverage = PrecisionCoverage(implementation)
        with precision_model_init_context(implementation):
            self.mlp = SwiGLUMLP(
                128,
                128,
                precision_coverage=coverage,
            )
            self.coverage_manifest = coverage.seal()
        self.precision_implementation = implementation

    def forward(self, x):
        from megatron.lite.primitive.precision import precision_forward_context

        with precision_forward_context(self.precision_implementation):
            return self.mlp(x)


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
        pytest.skip("CUDA is required for FSDP2 offload tests.")
    if not fsdp2_available():
        pytest.skip("Installed PyTorch does not expose FSDP2 fully_shard.")

    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29501")

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


def _build_fsdp2_model(dtype: torch.dtype = torch.bfloat16) -> tuple[nn.Module, ParallelState]:
    torch.manual_seed(1234)
    model = TinyModel().cuda().to(dtype=dtype)
    ps = _parallel_state()
    config = FSDP2Config(unit_modules=(TinyUnit,), reshard_after_forward=True)
    mesh = build_fsdp2_device_mesh(ps, config)
    return wrap_fsdp2(model, ps, config, mesh=mesh), ps


def _build_optimizer(model: nn.Module, ps: ParallelState, *, offload_fraction: float):
    return build_fsdp2_adamw(
        [model],
        SimpleNamespace(
            optimizer="adam",
            lr=1.0e-3,
            weight_decay=0.0,
            adam_beta1=0.9,
            adam_beta2=0.95,
            adam_eps=1.0e-8,
            clip_grad=1.0,
            offload_fraction=offload_fraction,
        ),
        ps,
        use_fp32_master=True,
    )


def _build_fp8_fsdp2_model() -> tuple[nn.Module, ParallelState, dict[str, torch.Tensor]]:
    torch.manual_seed(1234)
    model = TinyFP8Model().cuda().to(dtype=torch.bfloat16)
    sources = {}
    for name, param in model.named_parameters():
        get_source = getattr(param, "get_high_precision_init_val", None)
        assert callable(get_source), f"{name} is missing TE's preserved FP32 source"
        source = get_source()
        assert isinstance(source, torch.Tensor)
        sources[name] = source.detach().float().cpu().clone()

    ps = _parallel_state()
    config = FSDP2Config(unit_modules=(type(model.mlp),), reshard_after_forward=True)
    mesh = build_fsdp2_device_mesh(ps, config)
    return wrap_fsdp2(model, ps, config, mesh=mesh), ps, sources


def _local_param_devices(model: nn.Module) -> set[str]:
    return {to_local_tensor(param.detach()).device.type for param in model.parameters()}


def _optimizer_state_devices(optimizer) -> set[str]:
    devices: set[str] = set()
    for child in iter_torch_optimizers(optimizer.optimizer):
        for param_state in getattr(child, "state", {}).values():
            if not isinstance(param_state, dict):
                continue
            for value in param_state.values():
                if isinstance(value, torch.Tensor):
                    devices.add(to_local_tensor(value).device.type)
    return devices


def _fp8_step(model: nn.Module, optimizer, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    optimizer.zero_grad()
    loss = torch.nn.functional.mse_loss(model(x).float(), target.float())
    loss.backward()
    success, grad_norm, _ = optimizer.step()

    assert success
    assert torch.isfinite(loss)
    assert torch.isfinite(torch.tensor(grad_norm))
    return loss.detach()


def _fp32_masters(model: nn.Module, optimizer) -> dict[str, torch.Tensor]:
    return {
        name: optimizer.optimizer.state[param]["master_param"].detach().float().cpu().clone()
        for name, param in model.named_parameters()
    }


def test_fsdp2_runtime_model_and_optimizer_offload_roundtrip_single_gpu():
    model, ps = _build_fsdp2_model()
    optimizer = _build_optimizer(model, ps, offload_fraction=0.0)
    handle = ModelHandle(
        model=model, optimizer=optimizer, parallel_state=ps, _extras={"model_chunks": [model]}
    )
    runtime = MegatronLiteRuntime.__new__(MegatronLiteRuntime)

    assert _local_param_devices(model) == {"cuda"}
    assert _optimizer_state_devices(optimizer) == {"cuda"}

    runtime.to(handle, "cpu", model=True, optimizer=True, grad=True)
    assert _local_param_devices(model) == {"cpu"}
    assert _optimizer_state_devices(optimizer) == {"cpu"}

    runtime.to(handle, "cuda", model=True, optimizer=True, grad=True)
    assert _local_param_devices(model) == {"cuda"}
    assert _optimizer_state_devices(optimizer) == {"cuda"}


def test_fsdp2_offload_fraction_keeps_optimizer_update_state_on_cpu_single_gpu():
    model, ps = _build_fsdp2_model()
    optimizer = _build_optimizer(model, ps, offload_fraction=1.0)

    assert _optimizer_state_devices(optimizer) == {"cpu"}

    x = torch.randn(4, 8, device="cuda", dtype=torch.bfloat16)
    target = torch.randn(4, 4, device="cuda", dtype=torch.bfloat16)
    optimizer.zero_grad()
    loss = torch.nn.functional.mse_loss(model(x).float(), target.float())
    loss.backward()
    success, grad_norm, _ = optimizer.step()

    assert success
    assert torch.isfinite(torch.tensor(grad_norm))
    assert _local_param_devices(model) == {"cuda"}
    assert _optimizer_state_devices(optimizer) == {"cpu"}


def test_fsdp2_fp8_weight_uses_te_source_for_fp32_master_and_updates_single_gpu():
    model, ps, sources = _build_fp8_fsdp2_model()
    optimizer = _build_optimizer(model, ps, offload_fraction=0.0)

    masters = {
        name: optimizer.optimizer.state[param]["master_param"].detach().float().cpu()
        for name, param in model.named_parameters()
    }
    assert masters.keys() == sources.keys()
    for name in sources:
        torch.testing.assert_close(masters[name], sources[name], atol=0.0, rtol=0.0)
        assert getattr(model.get_parameter(name), "get_high_precision_init_val")() is None

    x = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16)
    target = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16)
    _fp8_step(model, optimizer, x, target)


def test_fsdp2_fp8_weight_local_checkpoint_resume_matches_uninterrupted_single_gpu(tmp_path):
    direct_model, direct_ps, _ = _build_fp8_fsdp2_model()
    direct_optimizer = _build_optimizer(direct_model, direct_ps, offload_fraction=0.0)
    saved_model, saved_ps, _ = _build_fp8_fsdp2_model()
    saved_optimizer = _build_optimizer(saved_model, saved_ps, offload_fraction=0.0)

    torch.manual_seed(4321)
    x0 = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16)
    target0 = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16)
    x1 = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16)
    target1 = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16)

    torch.testing.assert_close(
        _fp8_step(direct_model, direct_optimizer, x0, target0),
        _fp8_step(saved_model, saved_optimizer, x0, target0),
        atol=0.0,
        rtol=0.0,
    )

    runtime = MegatronLiteRuntime.__new__(MegatronLiteRuntime)
    runtime.save_checkpoint(
        ModelHandle(
            model=saved_model,
            optimizer=saved_optimizer,
            parallel_state=saved_ps,
            _extras={"model_chunks": [saved_model]},
        ),
        str(tmp_path),
        step=1,
        use_dcp=False,
    )

    resumed_model, resumed_ps, _ = _build_fp8_fsdp2_model()
    resumed_optimizer = _build_optimizer(resumed_model, resumed_ps, offload_fraction=0.0)
    assert runtime.load_checkpoint(
        ModelHandle(
            model=resumed_model,
            optimizer=resumed_optimizer,
            parallel_state=resumed_ps,
            _extras={"model_chunks": [resumed_model]},
        ),
        str(tmp_path),
        use_dcp=False,
    ) == 1

    for name, master in _fp32_masters(saved_model, saved_optimizer).items():
        torch.testing.assert_close(
            _fp32_masters(resumed_model, resumed_optimizer)[name], master, atol=0.0, rtol=0.0
        )

    _fp8_step(direct_model, direct_optimizer, x1, target1)
    _fp8_step(resumed_model, resumed_optimizer, x1, target1)
    for name, master in _fp32_masters(direct_model, direct_optimizer).items():
        torch.testing.assert_close(
            _fp32_masters(resumed_model, resumed_optimizer)[name], master, atol=0.0, rtol=0.0
        )
