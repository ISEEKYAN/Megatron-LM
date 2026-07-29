# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from __future__ import annotations

import gc
import json
import os
import statistics
import sys
import time
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils._python_dispatch import TorchDispatchMode

from megatron.lite.primitive.optimizers.fsdp2 import (
    build_fsdp2_training_optimizer,
    fsdp2_available,
)
from megatron.lite.primitive.optimizers.mfsdp import build_mfsdp_training_optimizer
from megatron.lite.primitive.parallel import init_parallel
from megatron.lite.primitive.recompute import apply_offload
from megatron.lite.runtime.backends.mlite.runtime import MegatronLiteRuntime
from megatron.lite.runtime.contracts.config import OptimizerConfig, ParallelConfig
from megatron.lite.runtime.contracts.data import PackedBatch
from megatron.lite.runtime.contracts.handle import ModelHandle

pytestmark = [
    pytest.mark.mlite,
    pytest.mark.smoke,
    pytest.mark.gpu,
    pytest.mark.distributed,
]

_MFSDP_SHARDING_STRATEGY = "optim_grads_params"
_PRECISION_STEPS = 50
_LOSS_REL_TOL = 1.0e-2
_TENSOR_RTOL = 1.0e-2
_TENSOR_ATOL = 1.0e-5
_BENCH_WARMUP_STEPS = 5
_BENCH_MEASURE_STEPS = 20
_BENCH_TOKENS_PER_RANK = 1024
_MIN_SPEEDUP = 1.0
_FULL_PARALLEL_WORLD_SIZE = 8
_FULL_PARALLEL_STEPS = 50
_FULL_PARALLEL_CURVE_INTERVAL = 10
_FULL_PARALLEL_MICROBATCHES = 2
_FULL_PARALLEL_SEQ_LEN = 64


class TinyUnit(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(8, 8, bias=False)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.linear(x))


class TinyDenseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.unit0 = TinyUnit()
        self.unit1 = TinyUnit()
        self.out = nn.Linear(8, 4, bias=False)

    def forward(self, x):
        return self.out(self.unit1(self.unit0(x)))


class TinyExperts(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(8, 8, bias=False)

    def forward(self, x):
        return torch.nn.functional.gelu(self.linear(x))


class TinyParallelModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.unit = TinyUnit()
        self.experts = TinyExperts()
        self.tp_bias = nn.Parameter(torch.zeros(8))
        self.sp_params = [self.tp_bias]
        self.out = nn.Linear(8, 4, bias=False)
        self.unit.linear.weight.tensor_model_parallel = True
        self.unit.linear.weight.partition_dim = 0
        self.out.weight.tensor_model_parallel = True
        self.out.weight.partition_dim = 1

    def forward(self, x):
        hidden = self.unit(x) + self.tp_bias
        return self.out(self.experts(hidden))


class BenchmarkUnit(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.up = nn.Linear(hidden_size, hidden_size * 2, bias=False)
        self.down = nn.Linear(hidden_size * 2, hidden_size, bias=False)

    def forward(self, x):
        return x + self.down(torch.nn.functional.gelu(self.up(x)))


class BenchmarkModel(nn.Module):
    hidden_size = 1024
    num_layers = 12

    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList(
            BenchmarkUnit(self.hidden_size) for _ in range(self.num_layers)
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


@pytest.fixture(scope="module", autouse=True)
def _cuda_dist():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if not torch.cuda.is_available():
        if world_size > 1:
            pytest.fail(
                "Distributed M-FSDP parity was launched without visible CUDA devices."
            )
        pytest.skip("CUDA is required for M-FSDP parity smoke tests.")
    if not fsdp2_available():
        if world_size > 1:
            pytest.fail("Distributed M-FSDP parity requires PyTorch FSDP2 fully_shard.")
        pytest.skip("Installed PyTorch does not expose FSDP2 fully_shard.")
    if world_size < 2:
        pytest.skip(
            "M-FSDP sharding parity smoke requires at least 2 distributed ranks."
        )
    if world_size > _FULL_PARALLEL_WORLD_SIZE:
        pytest.skip(
            "M-FSDP smoke tests support at most the dedicated 8-rank full-parallel signoff."
        )

    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29571")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("NVTE_ALLOW_NONDETERMINISTIC_ALGO", "0")

    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    if int(os.environ["RANK"]) == 0:
        print(
            "[MFSDP_ENV] "
            f"torch={torch.__version__} "
            f"world_size={world_size} "
            f"slurm_nnodes={os.environ.get('SLURM_NNODES', '1')} "
            f"cuda_devices={torch.cuda.device_count()} "
            f"fsdp2_available={fsdp2_available()}",
            flush=True,
        )
    _install_transformer_engine_import_stub_if_needed()
    created_pg = False
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")
        created_pg = True
    yield
    if created_pg and dist.is_initialized():
        dist.destroy_process_group()


def _install_transformer_engine_import_stub_if_needed() -> None:
    try:
        import transformer_engine  # noqa: F401

        return
    except (ImportError, OSError):
        pass

    for name in list(sys.modules):
        if name == "transformer_engine" or name.startswith("transformer_engine."):
            sys.modules.pop(name, None)
        if name == "transformer_engine_torch" or name.startswith(
            "transformer_engine_torch."
        ):
            sys.modules.pop(name, None)

    sys.modules["transformer_engine"] = None
    sys.modules["transformer_engine_torch"] = None


def _model_cfg() -> SimpleNamespace:
    return SimpleNamespace(
        num_hidden_layers=2,
        hidden_size=8,
        num_attention_heads=1,
        add_bias_linear=False,
    )


def _optimizer_cfg(*, use_fused_optimizer: bool = True) -> OptimizerConfig:
    cfg = OptimizerConfig(
        optimizer="adam",
        lr=1.0e-3,
        min_lr=0.0,
        weight_decay=0.0,
        clip_grad=1000.0,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_eps=1.0e-8,
    )
    cfg.override_optimizer_config = {
        "mfsdp_sharding_strategy": _MFSDP_SHARDING_STRATEGY,
        "use_fused_optimizer": use_fused_optimizer,
    }
    return cfg


def _new_model(seed: int, model_type: type[nn.Module]) -> nn.Module:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return model_type().cuda().to(torch.bfloat16)


def _parallel_state(config: ParallelConfig) -> Any:
    return init_parallel(config)


def _build_fsdp2_pair(
    seed: int,
    *,
    parallel: ParallelConfig,
    model_type: type[nn.Module] = TinyDenseModel,
    expert_classifier=None,
    unit_modules: tuple[type[nn.Module], ...] = (TinyUnit,),
    offload_fraction: float = 0.0,
    activation_offload: bool = False,
):
    ps = _parallel_state(parallel)
    chunks = [_new_model(seed, model_type)]
    if activation_offload:
        apply_offload(chunks[0].layers, ["full"], {})
    optimizer_config = _optimizer_cfg(use_fused_optimizer=False)
    optimizer_config.offload_fraction = offload_fraction
    optimizer = build_fsdp2_training_optimizer(
        chunks,
        optimizer_config,
        ps,
        unit_modules=unit_modules,
        expert_classifier=expert_classifier,
        deterministic=True,
        use_fp32_master=True,
    )
    return chunks, optimizer, None


def _build_mfsdp_pair(
    seed: int,
    *,
    parallel: ParallelConfig,
    model_type: type[nn.Module] = TinyDenseModel,
    expert_classifier=None,
    unit_modules: tuple[type[nn.Module], ...] = (TinyUnit,),
    offload_fraction: float = 0.0,
    activation_offload: bool = False,
    use_fused_optimizer: bool = True,
):
    ps = _parallel_state(parallel)
    chunks = [_new_model(seed, model_type)]
    if activation_offload:
        apply_offload(chunks[0].layers, ["full"], {})
    optimizer_config = _optimizer_cfg(use_fused_optimizer=use_fused_optimizer)
    optimizer_config.offload_fraction = offload_fraction
    impl_cfg = SimpleNamespace(
        parallel=parallel,
        optimizer_config=optimizer_config,
    )
    optimizer, finalize = build_mfsdp_training_optimizer(
        chunks,
        impl_cfg=impl_cfg,
        ps=ps,
        is_expert=expert_classifier or (lambda _name: False),
        fsdp_unit_modules=unit_modules,
    )
    return chunks, optimizer, finalize


def _memory_triple() -> dict[str, float]:
    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    values = torch.tensor(
        [allocated, reserved, reserved - allocated],
        device="cuda",
        dtype=torch.float64,
    )
    dist.all_reduce(values, op=dist.ReduceOp.MAX)
    return dict(
        zip(
            ("allocated_gib", "reserved_gib", "scratch_gib"),
            (float(value) / (1024**3) for value in values),
            strict=True,
        )
    )


def _optimizer_tensor_devices(optimizer: Any) -> list[str]:
    devices: set[str] = set()
    seen: set[int] = set()

    def visit(value: Any, depth: int = 0) -> None:
        if value is None or depth > 8 or id(value) in seen:
            return
        seen.add(id(value))
        if isinstance(value, torch.Tensor):
            devices.add(value.device.type)
            return
        if isinstance(value, dict):
            for item in value.values():
                visit(item, depth + 1)
            return
        if isinstance(value, (list, tuple)):
            for item in value:
                visit(item, depth + 1)
            return
        for attribute in (
            "state",
            "optimizer",
            "_inner_optimizer",
            "_cpu_optimizer",
            "cpu_group",
            "_cpu_params",
        ):
            visit(getattr(value, attribute, None), depth + 1)

    visit(optimizer)
    return sorted(devices)


def _mfsdp_cpu_master_devices(optimizer: Any) -> list[str]:
    inner = getattr(optimizer, "_inner_optimizer", None)
    cpu_group = getattr(inner, "cpu_group", None)
    cpu_params = getattr(cpu_group, "_cpu_params", ())
    return sorted({param.device.type for param in cpu_params})


def _profiled_train_step(chunks, optimizer, finalize, x, target):
    optimizer.zero_grad()
    torch.cuda.synchronize()
    started = time.perf_counter()
    output = chunks[0](x)
    loss = torch.nn.functional.mse_loss(output.float(), target.float())
    torch.cuda.synchronize()
    training_memory = _memory_triple()
    loss.backward()
    if finalize is not None:
        finalize()
    success, grad_norm, _num_zeros = optimizer.step()
    torch.cuda.synchronize()
    optimizer_memory = _memory_triple()
    elapsed = torch.tensor(time.perf_counter() - started, device="cuda")
    dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
    return (
        bool(success),
        float(loss.detach()),
        float(grad_norm),
        float(elapsed),
        training_memory,
        optimizer_memory,
    )


def _run_offload_arm(
    *,
    backend: str,
    offload_fraction: float,
    activation_offload: bool,
    x: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, Any]:
    gc.collect()
    torch.cuda.empty_cache()
    builder = _build_mfsdp_pair if backend == "mfsdp" else _build_fsdp2_pair
    kwargs = dict(
        seed=4567,
        parallel=_dense_parallel_config(),
        model_type=BenchmarkModel,
        unit_modules=(BenchmarkUnit,),
        offload_fraction=offload_fraction,
        activation_offload=activation_offload,
    )
    if backend == "mfsdp":
        kwargs["use_fused_optimizer"] = False
    chunks, optimizer, finalize = builder(**kwargs)

    losses = []
    for _ in range(2):
        success, loss, _grad_norm = _train_step(chunks, optimizer, finalize, x, target)
        assert success
        losses.append(loss)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    step_times = []
    training_memory = []
    optimizer_memory = []
    for _ in range(5):
        success, loss, _grad_norm, step_s, training, optimizer_side = (
            _profiled_train_step(chunks, optimizer, finalize, x, target)
        )
        assert success
        losses.append(loss)
        step_times.append(step_s)
        training_memory.append(training)
        optimizer_memory.append(optimizer_side)

    params = _named_model_tensors(chunks)
    result = {
        "backend": backend,
        "optimizer_offload": bool(offload_fraction),
        "activation_offload": activation_offload,
        "losses": losses,
        "step_ms": statistics.median(step_times) * 1000.0,
        "tokens_per_s_per_gpu": x.shape[0] / statistics.median(step_times),
        "training_memory": training_memory[-1],
        "optimizer_memory": optimizer_memory[-1],
        "optimizer_tensor_devices": _optimizer_tensor_devices(optimizer),
        "cpu_master_devices": _mfsdp_cpu_master_devices(optimizer),
        "params": params,
    }
    del optimizer, chunks
    gc.collect()
    torch.cuda.empty_cache()
    return result


def _qualified_type(value: Any) -> str:
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


def _optimizer_chain(optimizer: Any) -> tuple[str, ...]:
    chain = []
    seen = set()
    current = optimizer
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        chain.append(_qualified_type(current))
        current = getattr(current, "_inner_optimizer", None) or getattr(
            current, "optimizer", None
        )
    return tuple(chain)


def _assert_distinct_backend_identities(
    fsdp2_optimizer: Any, mfsdp_optimizer: Any
) -> None:
    fsdp2_chain = _optimizer_chain(fsdp2_optimizer)
    mfsdp_chain = _optimizer_chain(mfsdp_optimizer)
    assert fsdp2_chain[0].startswith("megatron.lite.primitive.optimizers.fsdp2."), (
        fsdp2_chain
    )
    assert mfsdp_chain[0].startswith("megatron.lite.primitive.optimizers.mfsdp."), (
        mfsdp_chain
    )
    assert fsdp2_chain[0] != mfsdp_chain[0]
    if dist.get_rank() == 0:
        print(
            "[MFSDP_BACKEND] "
            f"configured=fsdp2 optimizer_chain={' > '.join(fsdp2_chain)}",
            flush=True,
        )
        print(
            "[MFSDP_BACKEND] "
            f"configured=mfsdp optimizer_chain={' > '.join(mfsdp_chain)}",
            flush=True,
        )


def _train_once(chunks, optimizer, finalize, x: torch.Tensor, target: torch.Tensor):
    optimizer.zero_grad()
    output = chunks[0](x)
    loss = torch.nn.functional.mse_loss(output.float(), target.float())
    loss.backward()
    if finalize is not None:
        finalize()
    success, grad_norm, _num_zeros = optimizer.step()
    grads = _named_optimizer_grads(optimizer)
    optimizer.zero_grad()
    return bool(success), float(loss.detach().cpu()), float(grad_norm), grads


def _train_step(chunks, optimizer, finalize, x: torch.Tensor, target: torch.Tensor):
    optimizer.zero_grad()
    output = chunks[0](x)
    loss = torch.nn.functional.mse_loss(output.float(), target.float())
    loss.backward()
    if finalize is not None:
        finalize()
    success, grad_norm, _num_zeros = optimizer.step()
    return bool(success), float(loss.detach()), float(grad_norm)


def _timed_train_step(chunks, optimizer, finalize, x, target) -> float:
    torch.cuda.synchronize()
    started = time.perf_counter()
    success, _loss, _grad_norm = _train_step(chunks, optimizer, finalize, x, target)
    torch.cuda.synchronize()
    assert success
    elapsed = torch.tensor(time.perf_counter() - started, device="cuda")
    dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
    return float(elapsed)


def _relative_difference_curve(lhs: list[float], rhs: list[float]) -> list[float]:
    assert len(lhs) == len(rhs)
    assert lhs
    return [
        abs(left - right) / max(abs(left), abs(right), 1.0e-12)
        for left, right in zip(lhs, rhs)
    ]


def _max_relative_difference(lhs: list[float], rhs: list[float]) -> float:
    return max(_relative_difference_curve(lhs, rhs))


def _full_tensor(tensor: torch.Tensor) -> torch.Tensor:
    full_tensor = getattr(tensor, "full_tensor", None)
    if callable(full_tensor):
        return full_tensor()
    return tensor


def _named_model_tensors(chunks) -> dict[str, torch.Tensor]:
    params: dict[str, torch.Tensor] = {}
    for chunk_idx, chunk in enumerate(chunks):
        for name, param in chunk.named_parameters():
            if not param.requires_grad:
                continue
            canonical_name = name.replace("_orig_mod.", "").replace("module.", "")
            full = _full_tensor(param.detach())
            params[f"{chunk_idx}.{canonical_name}"] = full.cpu().float().clone()
    return params


def _canonical_optimizer_name(name: str) -> str:
    chunk_name, separator, remainder = name.partition(".")
    if separator and chunk_name.startswith("chunk") and chunk_name[5:].isdigit():
        chunk_index = chunk_name[5:]
        name = f"{chunk_index}.{remainder}"
    return name.replace("_orig_mod.", "").replace("module.", "")


def _named_optimizer_grads(optimizer) -> dict[str, torch.Tensor]:
    model_chunks = getattr(optimizer, "_model_chunks", None)
    if model_chunks is not None:
        return _named_mfsdp_optimizer_grads(model_chunks)

    grads: dict[str, torch.Tensor] = {}
    for param in optimizer.params:
        if param.grad is None:
            continue
        name = _canonical_optimizer_name(optimizer.param_names[id(param)])
        grads[name] = _full_tensor(param.grad.detach()).cpu().float().clone()
    return grads


def _named_mfsdp_optimizer_grads(model_chunks) -> dict[str, torch.Tensor]:
    grads: dict[str, torch.Tensor] = {}
    for chunk_index, chunk in enumerate(model_chunks):
        for bucket in chunk.param_sync.buckets:
            rank_major = torch.empty(
                bucket.full_numel,
                dtype=bucket.main_grad_buffer.dtype,
                device=bucket.device,
            )
            if bucket.world_size == 1:
                rank_major.copy_(bucket.grad_shard_buffer)
            else:
                dist.all_gather_into_tensor(
                    rank_major,
                    bucket.grad_shard_buffer,
                    group=bucket.process_group,
                )
            for spec in bucket.specs:
                grads[f"{chunk_index}.{spec.name}"] = (
                    rank_major.narrow(0, spec.full_offset, spec.numel)
                    .view(spec.shape)
                    .cpu()
                    .float()
                    .clone()
                )
    return grads


def _assert_tensor_sets_close(
    lhs: dict[str, torch.Tensor], rhs: dict[str, torch.Tensor]
) -> tuple[float, float]:
    assert lhs.keys() == rhs.keys()
    max_abs = 0.0
    max_rel = 0.0
    for name in lhs:
        diff = (lhs[name] - rhs[name]).abs()
        max_abs = max(max_abs, float(diff.max().item()))
        denominator = torch.maximum(lhs[name].abs(), rhs[name].abs()).clamp_min(
            _TENSOR_ATOL
        )
        max_rel = max(max_rel, float((diff / denominator).max().item()))
        torch.testing.assert_close(
            lhs[name],
            rhs[name],
            rtol=_TENSOR_RTOL,
            atol=_TENSOR_ATOL,
            msg=lambda message: f"{name}: {message}",
        )
    return max_abs, max_rel


def _tensor_set_metrics(
    lhs: dict[str, torch.Tensor], rhs: dict[str, torch.Tensor]
) -> dict[str, float]:
    assert lhs.keys() == rhs.keys()
    max_abs = 0.0
    max_rel = 0.0
    min_cosine = 1.0
    for name in lhs:
        left = lhs[name].double().reshape(-1)
        right = rhs[name].double().reshape(-1)
        diff = (left - right).abs()
        max_abs = max(max_abs, float(diff.max().item()))
        denominator = torch.maximum(left.abs(), right.abs()).clamp_min(_TENSOR_ATOL)
        max_rel = max(max_rel, float((diff / denominator).max().item()))
        cosine = torch.nn.functional.cosine_similarity(left, right, dim=0, eps=1.0e-12)
        min_cosine = min(min_cosine, float(cosine.item()))
    return {
        "max_param_abs_diff": max_abs,
        "max_param_rel_diff": max_rel,
        "min_param_cosine": min_cosine,
    }


def _dense_parallel_config() -> ParallelConfig:
    return ParallelConfig(tp=1, ep=1, etp=1, pp=1, vpp=1, cp=1)


def _tp_ep_parallel_config() -> ParallelConfig:
    return ParallelConfig(tp=2, ep=2, etp=1, pp=1, vpp=1, cp=1)


def _is_tiny_expert(name: str) -> bool:
    return name.startswith("experts.")


def test_mfsdp_precision_curve_matches_fsdp2():
    torch.manual_seed(4026 + dist.get_rank())
    torch.cuda.manual_seed_all(4026 + dist.get_rank())
    x = torch.randn(32, 8, device="cuda", dtype=torch.bfloat16)
    target = torch.randn(32, 4, device="cuda", dtype=torch.bfloat16)
    parallel = _dense_parallel_config()

    fsdp2_chunks, fsdp2_optimizer, fsdp2_finalize = _build_fsdp2_pair(
        seed=3456, parallel=parallel
    )
    mfsdp_chunks, mfsdp_optimizer, mfsdp_finalize = _build_mfsdp_pair(
        seed=3456, parallel=parallel
    )
    _assert_distinct_backend_identities(fsdp2_optimizer, mfsdp_optimizer)

    fsdp2_losses = []
    mfsdp_losses = []
    for _step in range(_PRECISION_STEPS):
        fsdp2_success, fsdp2_loss, _ = _train_step(
            fsdp2_chunks, fsdp2_optimizer, fsdp2_finalize, x, target
        )
        mfsdp_success, mfsdp_loss, _ = _train_step(
            mfsdp_chunks, mfsdp_optimizer, mfsdp_finalize, x, target
        )
        assert fsdp2_success
        assert mfsdp_success
        fsdp2_losses.append(fsdp2_loss)
        mfsdp_losses.append(mfsdp_loss)

    max_loss_rel_diff = torch.tensor(
        _max_relative_difference(fsdp2_losses, mfsdp_losses), device="cuda"
    )
    dist.all_reduce(max_loss_rel_diff, op=dist.ReduceOp.MAX)
    first_window_fsdp2 = statistics.mean(fsdp2_losses[:10])
    final_window_fsdp2 = statistics.mean(fsdp2_losses[-10:])
    first_window_mfsdp = statistics.mean(mfsdp_losses[:10])
    final_window_mfsdp = statistics.mean(mfsdp_losses[-10:])

    if dist.get_rank() == 0:
        print(
            "[MFSDP_PRECISION] "
            f"world_size={dist.get_world_size()} "
            f"steps={_PRECISION_STEPS} "
            f"loss_rel_tol={_LOSS_REL_TOL:.3e} "
            f"max_loss_rel_diff={float(max_loss_rel_diff):.8e} "
            f"first_loss_fsdp2={first_window_fsdp2:.8f} "
            f"final_loss_fsdp2={final_window_fsdp2:.8f} "
            f"first_loss_mfsdp={first_window_mfsdp:.8f} "
            f"final_loss_mfsdp={final_window_mfsdp:.8f}",
            flush=True,
        )

    assert torch.isfinite(max_loss_rel_diff)
    assert float(max_loss_rel_diff) <= _LOSS_REL_TOL
    assert final_window_fsdp2 < first_window_fsdp2
    assert final_window_mfsdp < first_window_mfsdp


def test_mfsdp_throughput_exceeds_fsdp2():
    torch.manual_seed(5026 + dist.get_rank())
    torch.cuda.manual_seed_all(5026 + dist.get_rank())
    x = torch.randn(
        _BENCH_TOKENS_PER_RANK,
        BenchmarkModel.hidden_size,
        device="cuda",
        dtype=torch.bfloat16,
    )
    target = torch.zeros_like(x)
    parallel = _dense_parallel_config()

    fsdp2_chunks, fsdp2_optimizer, fsdp2_finalize = _build_fsdp2_pair(
        seed=4567,
        parallel=parallel,
        model_type=BenchmarkModel,
        unit_modules=(BenchmarkUnit,),
    )
    mfsdp_chunks, mfsdp_optimizer, mfsdp_finalize = _build_mfsdp_pair(
        seed=4567,
        parallel=parallel,
        model_type=BenchmarkModel,
        unit_modules=(BenchmarkUnit,),
    )
    _assert_distinct_backend_identities(fsdp2_optimizer, mfsdp_optimizer)

    for step in range(_BENCH_WARMUP_STEPS):
        pairs = (
            (
                fsdp2_chunks,
                fsdp2_optimizer,
                fsdp2_finalize,
            ),
            (
                mfsdp_chunks,
                mfsdp_optimizer,
                mfsdp_finalize,
            ),
        )
        if step % 2:
            pairs = tuple(reversed(pairs))
        for chunks, optimizer, finalize in pairs:
            _train_step(chunks, optimizer, finalize, x, target)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    fsdp2_times = []
    mfsdp_times = []
    for step in range(_BENCH_MEASURE_STEPS):
        pairs = [
            (
                "fsdp2",
                fsdp2_chunks,
                fsdp2_optimizer,
                fsdp2_finalize,
                fsdp2_times,
            ),
            (
                "mfsdp",
                mfsdp_chunks,
                mfsdp_optimizer,
                mfsdp_finalize,
                mfsdp_times,
            ),
        ]
        if step % 2:
            pairs.reverse()
        for _name, chunks, optimizer, finalize, samples in pairs:
            samples.append(_timed_train_step(chunks, optimizer, finalize, x, target))

    fsdp2_step_s = statistics.median(fsdp2_times)
    mfsdp_step_s = statistics.median(mfsdp_times)
    global_tokens = _BENCH_TOKENS_PER_RANK * dist.get_world_size()
    fsdp2_tokens_per_s = global_tokens / fsdp2_step_s
    mfsdp_tokens_per_s = global_tokens / mfsdp_step_s
    speedup = mfsdp_tokens_per_s / fsdp2_tokens_per_s
    peak_memory_bytes = torch.tensor(
        torch.cuda.max_memory_allocated(), device="cuda", dtype=torch.float64
    )
    dist.all_reduce(peak_memory_bytes, op=dist.ReduceOp.MAX)
    peak_memory_gib = float(peak_memory_bytes) / (1024**3)

    if dist.get_rank() == 0:
        print(
            "[MFSDP_THROUGHPUT] "
            f"world_size={dist.get_world_size()} "
            f"warmup_steps={_BENCH_WARMUP_STEPS} "
            f"measure_steps={_BENCH_MEASURE_STEPS} "
            f"tokens_per_rank={_BENCH_TOKENS_PER_RANK} "
            f"fsdp2_step_ms={fsdp2_step_s * 1000.0:.4f} "
            f"mfsdp_step_ms={mfsdp_step_s * 1000.0:.4f} "
            f"fsdp2_tokens_per_s={fsdp2_tokens_per_s:.2f} "
            f"mfsdp_tokens_per_s={mfsdp_tokens_per_s:.2f} "
            f"mfsdp_speedup={speedup:.4f} "
            f"peak_memory_gib={peak_memory_gib:.4f}",
            flush=True,
        )

    assert speedup > _MIN_SPEEDUP


def test_mfsdp_offload_matrix_matches_fsdp2_precision_speed_and_memory():
    torch.manual_seed(6060 + dist.get_rank())
    torch.cuda.manual_seed_all(6060 + dist.get_rank())
    x = torch.randn(
        128,
        BenchmarkModel.hidden_size,
        device="cuda",
        dtype=torch.bfloat16,
    )
    target = torch.zeros_like(x)

    commit = os.environ.get("MLITE_VALIDATION_COMMIT")
    wandb_project = os.environ.get("WANDB_PROJECT")
    if not commit:
        pytest.fail("offload matrix requires MLITE_VALIDATION_COMMIT provenance")
    if not wandb_project:
        pytest.fail(
            "offload matrix requires WANDB_PROJECT; offline fallback is not accepted"
        )

    wandb_run = None
    if dist.get_rank() == 0:
        try:
            import wandb

            wandb_run = wandb.init(
                project=wandb_project,
                entity=os.environ.get("WANDB_ENTITY"),
                name=os.environ.get("WANDB_NAME", f"mfsdp-offload-{commit[:10]}"),
                config={
                    "commit": commit,
                    "world_size": dist.get_world_size(),
                    "optimizer_offload_values": [False, True],
                    "activation_offload_values": [False, True],
                    "reference": "fsdp2",
                },
            )
            if wandb_run is None:
                pytest.fail("wandb.init returned no run")
            print(f"[MFSDP_WANDB] url={wandb_run.url}", flush=True)
        except Exception as exc:
            pytest.fail(f"mandatory W&B initialization failed: {exc}")

    matrix = {}
    for backend in ("fsdp2", "mfsdp"):
        for optimizer_offload in (False, True):
            for activation_offload in (False, True):
                key = (backend, optimizer_offload, activation_offload)
                matrix[key] = _run_offload_arm(
                    backend=backend,
                    offload_fraction=float(optimizer_offload),
                    activation_offload=activation_offload,
                    x=x,
                    target=target,
                )

    if dist.get_rank() == 0:
        for (backend, optimizer_offload, activation_offload), result in matrix.items():
            name = (
                f"{backend}_optimizer_{int(optimizer_offload)}"
                f"_activation_{int(activation_offload)}"
            )
            raw = {key: value for key, value in result.items() if key != "params"}
            print(
                "[MFSDP_OFFLOAD_RAW] "
                + json.dumps({"commit": commit, "arm": name, **raw}, sort_keys=True),
                flush=True,
            )
            wandb_run.log(
                {
                    f"{name}/step_ms": raw["step_ms"],
                    f"{name}/tokens_per_s_per_gpu": raw["tokens_per_s_per_gpu"],
                    f"{name}/loss_final": raw["losses"][-1],
                    **{
                        f"{name}/{side}/{metric}": value
                        for side in ("training_memory", "optimizer_memory")
                        for metric, value in raw[side].items()
                    },
                }
            )

    comparisons = {}
    offload_effects = {}
    for optimizer_offload in (False, True):
        for activation_offload in (False, True):
            fsdp2 = matrix[("fsdp2", optimizer_offload, activation_offload)]
            mfsdp = matrix[("mfsdp", optimizer_offload, activation_offload)]
            assert mfsdp["losses"] == pytest.approx(
                fsdp2["losses"],
                rel=_LOSS_REL_TOL,
                abs=_TENSOR_ATOL,
            )
            comparisons[(optimizer_offload, activation_offload)] = {
                "max_loss_rel_diff": _max_relative_difference(
                    fsdp2["losses"],
                    mfsdp["losses"],
                ),
                **_tensor_set_metrics(fsdp2["params"], mfsdp["params"]),
            }

    for backend in ("fsdp2", "mfsdp"):
        baseline = matrix[(backend, False, False)]
        for optimizer_offload in (False, True):
            for activation_offload in (False, True):
                arm = matrix[(backend, optimizer_offload, activation_offload)]
                assert arm["losses"] == pytest.approx(
                    baseline["losses"],
                    rel=_LOSS_REL_TOL,
                    abs=_TENSOR_ATOL,
                )
                offload_effects[(backend, optimizer_offload, activation_offload)] = {
                    "max_loss_rel_diff": _max_relative_difference(
                        baseline["losses"],
                        arm["losses"],
                    ),
                    **_tensor_set_metrics(baseline["params"], arm["params"]),
                }
        for activation_offload in (False, True):
            optimizer_off = matrix[(backend, False, activation_offload)]
            optimizer_on = matrix[(backend, True, activation_offload)]
            assert (
                optimizer_on["optimizer_memory"]["allocated_gib"]
                < optimizer_off["optimizer_memory"]["allocated_gib"]
            )
            assert "cpu" in optimizer_on["optimizer_tensor_devices"]
            if backend == "mfsdp":
                assert optimizer_off["cpu_master_devices"] == []
                assert optimizer_on["cpu_master_devices"] == ["cpu"]
        for optimizer_offload in (False, True):
            activation_off = matrix[(backend, optimizer_offload, False)]
            activation_on = matrix[(backend, optimizer_offload, True)]
            assert (
                activation_on["training_memory"]["allocated_gib"]
                < activation_off["training_memory"]["allocated_gib"]
            )

    serializable_matrix = {}
    for (backend, optimizer_offload, activation_offload), result in matrix.items():
        name = (
            f"{backend}_optimizer_{int(optimizer_offload)}"
            f"_activation_{int(activation_offload)}"
        )
        serializable_matrix[name] = {
            key: value for key, value in result.items() if key != "params"
        }
        if dist.get_rank() == 0:
            print(
                "[MFSDP_OFFLOAD_MATRIX] "
                + json.dumps(
                    {
                        "commit": commit,
                        "arm": name,
                        **serializable_matrix[name],
                        "vs_fsdp2": comparisons[
                            (optimizer_offload, activation_offload)
                        ],
                        "vs_backend_no_offload": offload_effects[
                            (backend, optimizer_offload, activation_offload)
                        ],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    if wandb_run is not None:
        try:
            for arm, result in serializable_matrix.items():
                payload = {
                    f"{arm}/step_ms": result["step_ms"],
                    f"{arm}/tokens_per_s_per_gpu": result["tokens_per_s_per_gpu"],
                    f"{arm}/loss_final": result["losses"][-1],
                }
                for side in ("training_memory", "optimizer_memory"):
                    for metric, value in result[side].items():
                        payload[f"{arm}/{side}/{metric}"] = value
                wandb_run.log(payload)
            wandb_run.finish()
        except Exception as exc:
            pytest.fail(f"mandatory W&B logging failed: {exc}")


def test_mfsdp_matches_fsdp2_tiny_dense_single_step():
    torch.manual_seed(2026 + dist.get_rank())
    torch.cuda.manual_seed_all(2026 + dist.get_rank())
    x = torch.randn(4, 8, device="cuda", dtype=torch.bfloat16)
    target = torch.randn(4, 4, device="cuda", dtype=torch.bfloat16)

    parallel = _dense_parallel_config()
    fsdp2_chunks, fsdp2_optimizer, fsdp2_finalize = _build_fsdp2_pair(
        seed=1234, parallel=parallel
    )
    mfsdp_chunks, mfsdp_optimizer, mfsdp_finalize = _build_mfsdp_pair(
        seed=1234, parallel=parallel
    )

    fsdp2_success, fsdp2_loss, fsdp2_grad_norm, fsdp2_grads = _train_once(
        fsdp2_chunks, fsdp2_optimizer, fsdp2_finalize, x, target
    )
    mfsdp_success, mfsdp_loss, mfsdp_grad_norm, mfsdp_grads = _train_once(
        mfsdp_chunks, mfsdp_optimizer, mfsdp_finalize, x, target
    )

    assert fsdp2_success
    assert mfsdp_success
    assert fsdp2_loss == pytest.approx(mfsdp_loss, rel=_LOSS_REL_TOL)
    assert fsdp2_grad_norm == pytest.approx(mfsdp_grad_norm, rel=_LOSS_REL_TOL)
    max_grad_abs, max_grad_rel = _assert_tensor_sets_close(fsdp2_grads, mfsdp_grads)
    max_param_abs, max_param_rel = _assert_tensor_sets_close(
        _named_model_tensors(fsdp2_chunks), _named_model_tensors(mfsdp_chunks)
    )

    if dist.get_rank() == 0:
        print(
            "[MFSDP_COMPOSITION] "
            f"world_size={dist.get_world_size()} "
            f"strategy={_MFSDP_SHARDING_STRATEGY} "
            f"loss_fsdp2={fsdp2_loss:.8f} "
            f"loss_mfsdp={mfsdp_loss:.8f} "
            f"grad_norm_fsdp2={fsdp2_grad_norm:.8f} "
            f"grad_norm_mfsdp={mfsdp_grad_norm:.8f} "
            f"max_grad_abs_diff={max_grad_abs:.8e} "
            f"max_grad_rel_diff={max_grad_rel:.8e} "
            f"max_param_abs_diff={max_param_abs:.8e} "
            f"max_param_rel_diff={max_param_rel:.8e}",
            flush=True,
        )


def test_mfsdp_matches_fsdp2_tp_ep_single_step():
    torch.manual_seed(3026 + dist.get_rank())
    torch.cuda.manual_seed_all(3026 + dist.get_rank())
    x = torch.randn(4, 8, device="cuda", dtype=torch.bfloat16)
    target = torch.randn(4, 4, device="cuda", dtype=torch.bfloat16)
    parallel = _tp_ep_parallel_config()

    fsdp2_chunks, fsdp2_optimizer, fsdp2_finalize = _build_fsdp2_pair(
        seed=2345,
        parallel=parallel,
        model_type=TinyParallelModel,
        expert_classifier=_is_tiny_expert,
    )
    mfsdp_chunks, mfsdp_optimizer, mfsdp_finalize = _build_mfsdp_pair(
        seed=2345,
        parallel=parallel,
        model_type=TinyParallelModel,
        expert_classifier=_is_tiny_expert,
    )

    fsdp2_success, fsdp2_loss, fsdp2_grad_norm, fsdp2_grads = _train_once(
        fsdp2_chunks, fsdp2_optimizer, fsdp2_finalize, x, target
    )
    mfsdp_success, mfsdp_loss, mfsdp_grad_norm, mfsdp_grads = _train_once(
        mfsdp_chunks, mfsdp_optimizer, mfsdp_finalize, x, target
    )

    assert fsdp2_success
    assert mfsdp_success
    assert fsdp2_loss == pytest.approx(mfsdp_loss, rel=_LOSS_REL_TOL)
    assert fsdp2_grad_norm == pytest.approx(mfsdp_grad_norm, rel=_LOSS_REL_TOL)
    max_grad_abs, max_grad_rel = _assert_tensor_sets_close(fsdp2_grads, mfsdp_grads)
    max_param_abs, max_param_rel = _assert_tensor_sets_close(
        _named_model_tensors(fsdp2_chunks), _named_model_tensors(mfsdp_chunks)
    )

    if dist.get_rank() == 0:
        print(
            "[MFSDP_COMPOSITION] "
            f"world_size={dist.get_world_size()} "
            "topology=tp2_ep2 "
            f"strategy={_MFSDP_SHARDING_STRATEGY} "
            f"loss_fsdp2={fsdp2_loss:.8f} "
            f"loss_mfsdp={mfsdp_loss:.8f} "
            f"grad_norm_fsdp2={fsdp2_grad_norm:.8f} "
            f"grad_norm_mfsdp={mfsdp_grad_norm:.8f} "
            f"max_grad_abs_diff={max_grad_abs:.8e} "
            f"max_grad_rel_diff={max_grad_rel:.8e} "
            f"max_param_abs_diff={max_param_abs:.8e} "
            f"max_param_rel_diff={max_param_rel:.8e}",
            flush=True,
        )


def _full_parallel_config() -> ParallelConfig:
    return ParallelConfig(tp=2, ep=2, etp=1, pp=2, vpp=1, cp=2)


def _tiny_qwen3_moe_config():
    from megatron.lite.model.qwen3_moe.config import Qwen3MoEConfig

    return Qwen3MoEConfig(
        num_hidden_layers=2,
        hidden_size=256,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        vocab_size=64,
        num_experts=4,
        num_experts_per_tok=1,
        moe_intermediate_size=64,
        max_position_embeddings=4096,
        layer_types=["full_attention", "full_attention"],
    )


def _build_full_parallel_handle(backend: str, *, seed: int):
    from megatron.lite.model.qwen3_moe.lite import protocol

    parallel = _full_parallel_config()
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    impl_cfg = protocol.ImplConfig(
        parallel=parallel,
        optimizer=backend,
        optimizer_config=_optimizer_cfg(use_fused_optimizer=False),
        use_deepep=False,
        use_thd=True,
        deterministic=True,
    )
    model_cfg = _tiny_qwen3_moe_config()
    bundle = protocol.build_model(model_cfg, impl_cfg=impl_cfg)
    initial_params = _named_model_tensors(bundle.chunks)
    assert bundle.extras.get("optimizer_backend") == backend
    post_load_hook = bundle.extras.get("post_model_load_hook")
    assert callable(post_load_hook), (
        f"{backend} did not expose its production post-load hook."
    )
    updates = post_load_hook()
    assert isinstance(updates, dict)
    optimizer = updates.get("optimizer", bundle.optimizer)
    finalize_grads = updates.get("finalize_grads", bundle.finalize_grads)
    assert optimizer is not None

    extras = dict(bundle.extras)
    extras.update(
        {
            "model_chunks": bundle.chunks,
            "model_cfg": model_cfg,
            "forward_step": bundle.forward_step,
            "finalize_grads": finalize_grads,
        }
    )
    handle = ModelHandle(
        model=bundle.chunks,
        optimizer=optimizer,
        parallel_state=bundle.parallel_state,
        config=SimpleNamespace(parallel=parallel),
        _extras=extras,
    )
    return handle, initial_params


def _fixed_packed_batches(vocab_size: int, *, seed: int) -> list[PackedBatch]:
    batches = []
    for microbatch in range(_FULL_PARALLEL_MICROBATCHES):
        generator = torch.Generator(device="cuda").manual_seed(seed + microbatch)
        batches.append(
            PackedBatch(
                input_ids=torch.randint(
                    0,
                    vocab_size,
                    (_FULL_PARALLEL_SEQ_LEN,),
                    device="cuda",
                    generator=generator,
                ),
                labels=torch.randint(
                    0,
                    vocab_size,
                    (_FULL_PARALLEL_SEQ_LEN,),
                    device="cuda",
                    generator=generator,
                ),
                seq_lens=torch.tensor(
                    [_FULL_PARALLEL_SEQ_LEN], dtype=torch.int64, device="cuda"
                ),
            )
        )
    return batches


_COMM_TOKENS = (
    "all_gather",
    "allgather",
    "reduce_scatter",
    "reducescatter",
    "all_reduce",
    "allreduce",
    "all_to_all",
    "alltoall",
    "broadcast",
    "send",
    "recv",
)


class _CollectiveDispatchTap(TorchDispatchMode):
    def __init__(self, sequence: list[str]):
        super().__init__()
        self.sequence = sequence

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        del types
        name = str(func)
        if any(token in name.lower() for token in _COMM_TOKENS):
            self.sequence.append(f"dispatch::{name}")
        return func(*args, **(kwargs or {}))


@contextmanager
def _record_collectives():
    sequence = []
    originals = {}
    api_names = (
        "all_gather",
        "all_gather_into_tensor",
        "_all_gather_base",
        "reduce_scatter",
        "reduce_scatter_tensor",
        "_reduce_scatter_base",
        "all_reduce",
        "all_to_all",
        "all_to_all_single",
        "broadcast",
        "batch_isend_irecv",
        "send",
        "recv",
    )
    for name in api_names:
        original = getattr(dist, name, None)
        if original is None:
            continue
        originals[name] = original

        def wrapped(*args, _name=name, _original=original, **kwargs):
            sequence.append(f"python::torch.distributed.{_name}")
            return _original(*args, **kwargs)

        setattr(dist, name, wrapped)

    try:
        with _CollectiveDispatchTap(sequence):
            yield sequence
    finally:
        for name, original in originals.items():
            setattr(dist, name, original)


def _contains_collective(sequence: tuple[str, ...], *tokens: str) -> bool:
    return any(any(token in name.lower() for token in tokens) for name in sequence)


def _run_full_parallel_step(
    handle: ModelHandle, *, batch_seed: int, record_collectives: bool
) -> tuple[float, float, tuple[str, ...]]:
    runtime = MegatronLiteRuntime.__new__(MegatronLiteRuntime)
    runtime.zero_grad(handle)
    batches = _fixed_packed_batches(64, seed=batch_seed)
    sequence = []
    if record_collectives:
        with _record_collectives() as sequence:
            result = runtime.forward_backward(
                handle,
                iter(batches),
                None,
                num_microbatches=_FULL_PARALLEL_MICROBATCHES,
            )
        torch.cuda.synchronize()
    else:
        result = runtime.forward_backward(
            handle, iter(batches), None, num_microbatches=_FULL_PARALLEL_MICROBATCHES
        )

    success, grad_norm, _num_zeros = runtime.optimizer_step(handle)
    assert success
    loss = result.model_output.loss
    assert loss is not None
    return float(loss.detach().float().cpu()), float(grad_norm), tuple(sequence)


def _max_snapshot_abs_diff(
    lhs: dict[str, torch.Tensor], rhs: dict[str, torch.Tensor]
) -> float:
    assert lhs.keys() == rhs.keys()
    assert lhs
    return max(float((lhs[name] - rhs[name]).abs().max()) for name in lhs)


def test_mfsdp_matches_fsdp2_full_parallel_precision_curve(monkeypatch):
    if dist.get_world_size() != _FULL_PARALLEL_WORLD_SIZE:
        pytest.skip("M-FSDP TP2/EP2/ETP1/PP2/CP2 signoff requires exactly 8 ranks.")

    if int(os.environ.get("SLURM_NNODES", "0")) != 1:
        pytest.fail("M-FSDP full-parallel signoff requires exactly one Slurm node.")

    try:
        import transformer_engine.pytorch as te
    except (ImportError, OSError) as exc:
        pytest.fail(
            f"full-parallel M-FSDP signoff requires real Transformer Engine: {exc}"
        )
    assert hasattr(te, "Linear"), "full-parallel signoff requires real TE Linear."

    from megatron.lite.model import protocol_utils

    cp_splits = []
    original_prepare_cp = protocol_utils.prepare_packed_thd_kwargs_for_context_parallel

    def record_cp_split(model, kwargs):
        full_tokens = int(kwargs["input_ids"].numel())
        result = original_prepare_cp(model, kwargs)
        local_tokens = int(kwargs["input_ids"].numel())
        cp_splits.append((full_tokens, local_tokens))
        return result

    monkeypatch.setattr(
        protocol_utils,
        "prepare_packed_thd_kwargs_for_context_parallel",
        record_cp_split,
    )

    fsdp2_handle, fsdp2_initial = _build_full_parallel_handle("fsdp2", seed=7345)
    mfsdp_handle, mfsdp_initial = _build_full_parallel_handle("mfsdp", seed=7345)
    _assert_distinct_backend_identities(
        fsdp2_handle._optimizer, mfsdp_handle._optimizer
    )

    for handle in (fsdp2_handle, mfsdp_handle):
        ps = handle._parallel_state
        assert (ps.tp_size, ps.ep_size, ps.etp_size, ps.pp_size, ps.cp_size) == (
            2,
            2,
            1,
            2,
            2,
        )

    initial_max_abs = torch.tensor(
        _max_snapshot_abs_diff(fsdp2_initial, mfsdp_initial), device="cuda"
    )
    dist.all_reduce(initial_max_abs, op=dist.ReduceOp.MAX)
    assert float(initial_max_abs) == 0.0

    losses = {"fsdp2": [], "mfsdp": []}
    grad_norms = {"fsdp2": [], "mfsdp": []}
    traces = {}
    for step in range(_FULL_PARALLEL_STEPS):
        for backend, handle in (("fsdp2", fsdp2_handle), ("mfsdp", mfsdp_handle)):
            dist.barrier()
            loss, grad_norm, trace = _run_full_parallel_step(
                handle,
                batch_seed=8345 + step,
                record_collectives=step == 0,
            )
            losses[backend].append(loss)
            grad_norms[backend].append(grad_norm)
            if trace:
                traces[backend] = trace

    for backend in ("fsdp2", "mfsdp"):
        trace = traces.get(backend, ())
        assert trace, f"{backend} tap recorded no distributed collectives."
        assert _contains_collective(trace, "all_gather", "allgather")
        assert _contains_collective(trace, "reduce_scatter", "reducescatter")

    assert cp_splits
    assert all(
        full_tokens == 2 * local_tokens for full_tokens, local_tokens in cp_splits
    )

    loss_rel_curve = torch.tensor(
        _relative_difference_curve(losses["fsdp2"], losses["mfsdp"]),
        device="cuda",
    )
    grad_norm_rel_curve = torch.tensor(
        _relative_difference_curve(grad_norms["fsdp2"], grad_norms["mfsdp"]),
        device="cuda",
    )
    dist.all_reduce(loss_rel_curve, op=dist.ReduceOp.MAX)
    dist.all_reduce(grad_norm_rel_curve, op=dist.ReduceOp.MAX)
    max_loss_rel_diff = loss_rel_curve.max()
    max_grad_norm_rel_diff = grad_norm_rel_curve.max()

    if dist.get_rank() == 0:
        ps = fsdp2_handle._parallel_state
        print(
            "[MFSDP_FULL_PARALLEL] "
            "topology=tp2_ep2_etp1_pp2_cp2 "
            f"world_size={dist.get_world_size()} "
            f"dp_cp_size={ps.dp_cp_size} "
            f"microbatches={_FULL_PARALLEL_MICROBATCHES} "
            f"steps={_FULL_PARALLEL_STEPS} "
            f"initial_max_abs_diff={float(initial_max_abs):.8e} "
            f"max_loss_rel_diff={float(max_loss_rel_diff):.8e} "
            f"max_grad_norm_rel_diff={float(max_grad_norm_rel_diff):.8e} "
            f"fsdp2_losses={','.join(f'{value:.8f}' for value in losses['fsdp2'])} "
            f"mfsdp_losses={','.join(f'{value:.8f}' for value in losses['mfsdp'])} "
            f"cp_split={cp_splits[0][0]}->{cp_splits[0][1]} "
            f"cp_split_calls={len(cp_splits)}",
            flush=True,
        )
        for curve_index in range(
            _FULL_PARALLEL_CURVE_INTERVAL - 1,
            _FULL_PARALLEL_STEPS,
            _FULL_PARALLEL_CURVE_INTERVAL,
        ):
            print(
                "[MFSDP_FULL_PARALLEL_CURVE] "
                f"step={curve_index + 1} "
                f"loss_rel_diff={float(loss_rel_curve[curve_index]):.8e} "
                f"grad_norm_rel_diff={float(grad_norm_rel_curve[curve_index]):.8e} "
                f"loss_fsdp2={losses['fsdp2'][curve_index]:.8f} "
                f"loss_mfsdp={losses['mfsdp'][curve_index]:.8f} "
                f"grad_norm_fsdp2={grad_norms['fsdp2'][curve_index]:.8f} "
                f"grad_norm_mfsdp={grad_norms['mfsdp'][curve_index]:.8f}",
                flush=True,
            )
        for backend in ("fsdp2", "mfsdp"):
            trace = traces[backend]
            print(
                "[MFSDP_COMM_TRACE] "
                f"backend={backend} phase=forward_backward events={len(trace)} "
                f"sequence={' > '.join(trace[:96])}",
                flush=True,
            )

    assert torch.isfinite(loss_rel_curve).all()
    assert torch.isfinite(grad_norm_rel_curve).all()
    assert float(max_loss_rel_diff) <= _LOSS_REL_TOL
    assert float(max_grad_norm_rel_diff) <= _LOSS_REL_TOL
    assert losses["fsdp2"][-1] < losses["fsdp2"][0]
    assert losses["mfsdp"][-1] < losses["mfsdp"][0]
