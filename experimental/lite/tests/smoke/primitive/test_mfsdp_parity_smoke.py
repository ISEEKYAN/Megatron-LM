# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from __future__ import annotations

import os
import hashlib
import math
import statistics
import sys
import time
from contextlib import contextmanager
from pathlib import Path
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
_FULL_PARALLEL_MICROBATCHES = 2
_FULL_PARALLEL_SEQ_LEN = 64
_MCORE_REFERENCE_COMMIT = "c178e1c3e2ff47e359c56e9b86c9d40c6fddfb7b"
_MCORE_FULLY_SHARD_SHA256 = "4a8db9861f726f1ad3582c5c0e0f1b7545bf0d768167be6b6775ee6424d6e556"
_PRECISION_BACKENDS = ("mcore_mfsdp", "mfsdp", "fsdp2")
_PRECISION_CHECKPOINT_STEPS = (1, 10, 20, 30, 40, 50)
_TENSOR_COSINE_TOL = 0.999
_OPTIMIZER_STATE_SAMPLE_COUNT = 3


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
):
    ps = _parallel_state(parallel)
    chunks = [_new_model(seed, model_type)]
    optimizer = build_fsdp2_training_optimizer(
        chunks,
        _optimizer_cfg(),
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
):
    ps = _parallel_state(parallel)
    chunks = [_new_model(seed, model_type)]
    impl_cfg = SimpleNamespace(
        parallel=parallel,
        optimizer_config=_optimizer_cfg(),
    )
    optimizer, finalize = build_mfsdp_training_optimizer(
        chunks,
        impl_cfg=impl_cfg,
        ps=ps,
        is_expert=expert_classifier or (lambda _name: False),
        fsdp_unit_modules=unit_modules,
    )
    return chunks, optimizer, finalize


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
    """Replicate the FSDP dimension while preserving the local TP/ETP shard."""
    placements = getattr(tensor, "placements", None)
    redistribute = getattr(tensor, "redistribute", None)
    to_local = getattr(tensor, "to_local", None)
    if placements and callable(redistribute) and callable(to_local):
        from torch.distributed.tensor import Replicate

        replicated = list(placements)
        replicated[0] = Replicate()
        return redistribute(placements=tuple(replicated)).to_local()
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
    named_grad_tensors = getattr(optimizer, "named_grad_tensors", None)
    if callable(named_grad_tensors):
        return named_grad_tensors()
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


def _singleton_process_group() -> dist.ProcessGroup:
    selected = None
    rank = dist.get_rank()
    for candidate in range(dist.get_world_size()):
        group = dist.new_group([candidate])
        if candidate == rank:
            selected = group
    assert selected is not None
    return selected


def _mcore_reference_meshes(ps):
    from torch.distributed import DeviceMesh

    world_ranks = torch.arange(dist.get_world_size())
    dense_mesh = world_ranks.reshape(
        ps.pp_size, ps.dp_size * ps.cp_size, ps.tp_size
    )[ps.pp_rank]
    dense = DeviceMesh.from_group(
        [ps.dp_cp_group, ps.tp_group],
        device_type="cuda",
        mesh=dense_mesh.tolist(),
        mesh_dim_names=("dp_cp", "tp"),
    )

    etp_group = ps.etp_group or _singleton_process_group()
    expert_mesh = world_ranks.reshape(
        ps.pp_size, ps.expert_dp_size, ps.ep_size, ps.etp_size
    )[ps.pp_rank, :, ps.ep_rank, :]
    expert = DeviceMesh.from_group(
        [ps.ep_dp_group, etp_group],
        device_type="cuda",
        mesh=expert_mesh.tolist(),
        mesh_dim_names=("dp_cp", "tp"),
    )
    return dense, expert


def _mark_mcore_reference_params(module: nn.Module, ps) -> None:
    from megatron.lite.model.qwen3_moe.common import is_expert_param

    for name, param in module.named_parameters():
        expert = is_expert_param(name)
        param.allreduce = not expert
        active_tp_size = ps.etp_size if expert else ps.tp_size
        if bool(getattr(param, "tensor_model_parallel", False)):
            param._mcore_tp = True
            param._tp_partition_dim = int(getattr(param, "partition_dim", 0))
        elif active_tp_size > 1:
            param._mcore_tp = True
            param._tp_duplicated = True


@contextmanager
def _mcore_stage_local_dtensor_validation(module: nn.Module, ps):
    """Scope DTensor metadata checks to the model-parallel group.

    PyTorch's check_tensor_meta uses WORLD even when DTensor.from_local receives
    a stage-local DeviceMesh. Pipeline stages legitimately have different
    parameter counts, so validate the same metadata on TP/ETP first and suppress
    only the redundant WORLD check while NVIDIA MCore constructs its DTensors.
    """
    from torch.distributed.tensor import _api as dtensor_api

    for _name, param in module.named_parameters():
        expert = not bool(getattr(param, "allreduce", True))
        group = ps.etp_group if expert else ps.tp_group
        if group is None or dist.get_world_size(group) == 1:
            continue
        local_metadata = (param.dtype, param.requires_grad)
        gathered = [None] * dist.get_world_size(group)
        dist.all_gather_object(gathered, local_metadata, group=group)
        assert all(metadata == local_metadata for metadata in gathered)

    original_check_tensor_meta = dtensor_api.check_tensor_meta
    dtensor_api.check_tensor_meta = lambda *_args, **_kwargs: None
    try:
        yield
    finally:
        dtensor_api.check_tensor_meta = original_check_tensor_meta


class _MCoreReferenceOptimizer:
    """Adapt NVIDIA MCore's standalone M-FSDP API to the smoke runtime contract."""

    name = "mcore_mfsdp"

    def __init__(self, optimizer, model_chunks, ps, param_names):
        self.optimizer = optimizer
        self.model_chunks = list(model_chunks)
        self.ps = ps
        self.param_names = dict(param_names)
        self.params = [
            param for group in optimizer.param_groups for param in group["params"]
        ]
        self._grad_sync_enabled = False
        self.grad_sync_enabled = False

    @property
    def grad_sync_enabled(self) -> bool:
        return self._grad_sync_enabled

    @grad_sync_enabled.setter
    def grad_sync_enabled(self, enabled: bool) -> None:
        self._grad_sync_enabled = bool(enabled)
        for chunk in self.model_chunks:
            chunk.set_model_auto_sync(self._grad_sync_enabled)

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    def zero_grad(self) -> None:
        self.optimizer.zero_grad(set_to_none=True)
        self.grad_sync_enabled = False

    def named_grad_tensors(self) -> dict[str, torch.Tensor]:
        grads = {}
        for param in self.params:
            if param.grad is None:
                continue
            name = self.param_names[id(param)]
            grads[name] = _full_tensor(param.grad.detach()).cpu().float().clone()
        return grads

    def named_state_tensors(self) -> dict[str, dict[str, torch.Tensor]]:
        states = {}
        for param in self.params:
            state = self.optimizer.state.get(param, {})
            tensors = {
                key: _full_tensor(value.detach()).cpu().float().clone()
                for key, value in state.items()
                if key in {"exp_avg", "exp_avg_sq"} and isinstance(value, torch.Tensor)
            }
            if tensors:
                states[self.param_names[id(param)]] = tensors
        return states

    def _grad_norm(self, grads: dict[str, torch.Tensor]) -> float:
        total = torch.zeros((), device="cuda", dtype=torch.float64)
        params_by_name = {
            self.param_names[id(param)]: param for param in self.params
        }
        for name, grad in grads.items():
            param = params_by_name[name]
            expert = not bool(getattr(param, "allreduce", True))
            if expert:
                include = self.ps.expert_dp_rank == 0
            else:
                include = self.ps.dp_cp_rank == 0
            if bool(getattr(param, "_tp_duplicated", False)):
                include = include and (self.ps.etp_rank == 0 if expert else self.ps.tp_rank == 0)
            if include:
                total += grad.to(device="cuda", dtype=torch.float64).pow(2).sum()
        dist.all_reduce(total, op=dist.ReduceOp.SUM)
        return float(total.sqrt().float())

    def step(self) -> tuple[bool, float, int]:
        grads = self.named_grad_tensors()
        grad_norm = self._grad_norm(grads)
        if not math.isfinite(grad_norm):
            self.grad_sync_enabled = False
            return False, grad_norm, 0
        if grad_norm > 1000.0:
            scale = 1000.0 / (grad_norm + 1.0e-6)
            for param in self.params:
                if param.grad is not None:
                    local_grad = getattr(param.grad, "_local_tensor", param.grad)
                    local_grad.mul_(scale)
        self.optimizer.step()
        self.grad_sync_enabled = False
        return True, grad_norm, 0


class _MCoreRuntimeChunk(nn.Module):
    """Keep MLite's generic unwrapping from bypassing MegatronFSDP.forward."""

    def __init__(self, wrapped: nn.Module):
        super().__init__()
        object.__setattr__(self, "_wrapped", wrapped)

    def forward(self, *args, **kwargs):
        return self._wrapped(*args, **kwargs)

    def set_input_tensor(self, input_tensor) -> None:
        self._wrapped.module.set_input_tensor(input_tensor)

    def named_parameters(self, *args, **kwargs):
        return self._wrapped.named_parameters(*args, **kwargs)

    def parameters(self, *args, **kwargs):
        return self._wrapped.parameters(*args, **kwargs)


def _build_mcore_reference_optimizer(chunks, ps):
    from megatron.core.distributed.fsdp.src.megatron_fsdp.fully_shard import (
        fully_shard_model,
        fully_shard_optimizer,
    )
    from megatron.lite.model.qwen3_moe.lite.model import TransformerLayer

    source_path = Path(sys.modules[fully_shard_model.__module__].__file__).resolve()
    source_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
    assert os.environ.get("MLITE_MCORE_COMMIT") == _MCORE_REFERENCE_COMMIT
    assert source_sha256 == _MCORE_FULLY_SHARD_SHA256
    assert "megatron/core/distributed/fsdp" in source_path.as_posix()
    print(
        f"[MFSDP_BUILD] rank={dist.get_rank()} pp_rank={ps.pp_rank} "
        "backend=mcore_mfsdp phase=mesh_start",
        flush=True,
    )
    dense_mesh, expert_mesh = _mcore_reference_meshes(ps)
    print(
        f"[MFSDP_BUILD] rank={dist.get_rank()} pp_rank={ps.pp_rank} "
        "backend=mcore_mfsdp phase=mesh_done",
        flush=True,
    )
    wrapped_chunks = []
    for chunk_index, chunk in enumerate(chunks):
        _mark_mcore_reference_params(chunk, ps)
        print(
            f"[MFSDP_BUILD] rank={dist.get_rank()} pp_rank={ps.pp_rank} "
            f"backend=mcore_mfsdp phase=wrap_start chunk={chunk_index}",
            flush=True,
        )
        with _mcore_stage_local_dtensor_validation(chunk, ps):
            wrapped_chunks.append(
                fully_shard_model(
                    module=chunk,
                    device_mesh=dense_mesh,
                    dp_shard_dim="dp_cp",
                    tp_dim="tp",
                    expt_device_mesh=expert_mesh,
                    fsdp_unit_modules=(TransformerLayer,),
                    zero_dp_strategy="optim_grads_params",
                    grad_reduce_in_fp32=True,
                    preserve_fp32_weights=True,
                    overlap_grad_reduce=True,
                    overlap_param_gather=True,
                    sync_model_each_microbatch=False,
                    preproc_state_dict_for_dcp_ckpt=False,
                    average_in_collective=True,
                )
            )
        print(
            f"[MFSDP_BUILD] rank={dist.get_rank()} pp_rank={ps.pp_rank} "
            f"backend=mcore_mfsdp phase=wrap_done chunk={chunk_index}",
            flush=True,
        )
    chunks[:] = wrapped_chunks

    print(
        f"[MFSDP_BUILD] rank={dist.get_rank()} pp_rank={ps.pp_rank} "
        "backend=mcore_mfsdp phase=optimizer_start",
        flush=True,
    )
    raw_optimizer = torch.optim.AdamW(
        [param for chunk in chunks for param in chunk.parameters()],
        lr=1.0e-3,
        betas=(0.9, 0.999),
        eps=1.0e-8,
        weight_decay=0.0,
        foreach=False,
    )
    fully_shard_optimizer(raw_optimizer, preproc_state_dict_for_dcp_ckpt=False)
    param_names = {}
    for chunk_idx, chunk in enumerate(chunks):
        for name, param in chunk.named_parameters():
            param_names[id(param)] = f"{chunk_idx}.{_canonical_optimizer_name(name)}"
    optimizer = _MCoreReferenceOptimizer(raw_optimizer, chunks, ps, param_names)
    chunks[:] = [_MCoreRuntimeChunk(chunk) for chunk in chunks]
    print(
        f"[MFSDP_BUILD] rank={dist.get_rank()} pp_rank={ps.pp_rank} "
        "backend=mcore_mfsdp phase=optimizer_done",
        flush=True,
    )
    if dist.get_rank() == 0:
        print(
            "[MFSDP_REFERENCE] "
            "primary=mcore_mfsdp "
            "dtensor_validation=stage_local_tp_etp "
            f"commit={_MCORE_REFERENCE_COMMIT} "
            f"source_sha256={source_sha256} "
            f"source={source_path}",
            flush=True,
        )
    return optimizer


def _build_full_parallel_handle(backend: str, *, seed: int):
    from megatron.lite.model.qwen3_moe.lite import protocol

    parallel = _full_parallel_config()
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    impl_cfg = protocol.ImplConfig(
        parallel=parallel,
        optimizer=None if backend == "mcore_mfsdp" else backend,
        optimizer_config=_optimizer_cfg(use_fused_optimizer=False),
        use_deepep=False,
        use_thd=True,
        deterministic=True,
    )
    model_cfg = _tiny_qwen3_moe_config()
    bundle = protocol.build_model(model_cfg, impl_cfg=impl_cfg)
    initial_params = _named_model_tensors(bundle.chunks)
    if backend == "mcore_mfsdp":
        assert bundle.extras.get("optimizer_backend") == "none"
        optimizer = _build_mcore_reference_optimizer(
            bundle.chunks, bundle.parallel_state
        )
        finalize_grads = None
    else:
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
    handle: ModelHandle,
    *,
    batch_seed: int,
    record_collectives: bool,
    collect_grads: bool,
) -> tuple[float, float, tuple[str, ...], dict[str, torch.Tensor]]:
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

    grads = _named_optimizer_grads(handle._optimizer) if collect_grads else {}
    success, grad_norm, _num_zeros = runtime.optimizer_step(handle)
    assert success
    loss = result.model_output.loss
    assert loss is not None
    return (
        float(loss.detach().float().cpu()),
        float(grad_norm),
        tuple(sequence),
        grads,
    )


def _max_snapshot_abs_diff(
    lhs: dict[str, torch.Tensor], rhs: dict[str, torch.Tensor]
) -> float:
    assert lhs.keys() == rhs.keys()
    assert lhs
    return max(float((lhs[name] - rhs[name]).abs().max()) for name in lhs)


def _named_optimizer_states(optimizer) -> dict[str, dict[str, torch.Tensor]]:
    named_state_tensors = getattr(optimizer, "named_state_tensors", None)
    if callable(named_state_tensors):
        return named_state_tensors()

    model_chunks = getattr(optimizer, "_model_chunks", None)
    if model_chunks is not None:
        torch_optimizer = optimizer._inner_optimizer.optimizer
        states: dict[str, dict[str, torch.Tensor]] = {}
        for chunk_index, chunk in enumerate(model_chunks):
            for bucket in chunk.param_sync.buckets:
                for state_name in ("exp_avg", "exp_avg_sq"):
                    local_bucket = torch.zeros(
                        bucket.local_numel,
                        dtype=torch.float32,
                        device=bucket.device,
                    )
                    for spec in bucket.specs:
                        state = torch_optimizer.state.get(spec.shard_param, {})
                        value = state.get(state_name)
                        if isinstance(value, torch.Tensor) and spec.shard_numel:
                            local_bucket.narrow(
                                0, spec.local_offset, spec.shard_numel
                            ).copy_(value.detach().reshape(-1).float())
                    full_bucket = torch.empty(
                        bucket.full_numel,
                        dtype=torch.float32,
                        device=bucket.device,
                    )
                    if bucket.world_size == 1:
                        full_bucket.copy_(local_bucket)
                    else:
                        dist.all_gather_into_tensor(
                            full_bucket,
                            local_bucket,
                            group=bucket.process_group,
                        )
                    for spec in bucket.specs:
                        name = f"{chunk_index}.{spec.name}"
                        states.setdefault(name, {})[state_name] = (
                            full_bucket.narrow(0, spec.full_offset, spec.numel)
                            .view(spec.shape)
                            .cpu()
                            .clone()
                        )
        return states

    torch_optimizer = optimizer.optimizer
    states = {}
    for param in optimizer.params:
        name = _canonical_optimizer_name(optimizer.param_names[id(param)])
        state = torch_optimizer.state.get(param, {})
        tensors = {
            key: _full_tensor(value.detach()).cpu().float().clone()
            for key, value in state.items()
            if key in {"exp_avg", "exp_avg_sq"} and isinstance(value, torch.Tensor)
        }
        if tensors:
            states[name] = tensors
    return states


def _tensor_statistics(reference: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    assert reference.shape == target.shape
    reference = reference.float().reshape(-1)
    target = target.float().reshape(-1)
    diff = (reference - target).abs()
    max_abs = float(diff.max()) if diff.numel() else 0.0
    denominator = torch.maximum(reference.abs(), target.abs()).clamp_min(_TENSOR_ATOL)
    max_rel = float((diff / denominator).max()) if diff.numel() else 0.0
    reference_norm = float(torch.linalg.vector_norm(reference))
    target_norm = float(torch.linalg.vector_norm(target))
    if reference_norm <= _TENSOR_ATOL and target_norm <= _TENSOR_ATOL:
        cosine = 1.0 if max_abs <= _TENSOR_ATOL else 0.0
    else:
        cosine = float(
            torch.nn.functional.cosine_similarity(reference, target, dim=0, eps=1.0e-12)
        )
    return {
        "max_abs": max_abs,
        "max_rel": max_rel,
        "cosine": cosine,
        "reference_norm": reference_norm,
        "target_norm": target_norm,
    }


def _record_tensor_evidence(
    *,
    step: int,
    kind: str,
    reference_name: str,
    reference: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
) -> dict[str, Any] | None:
    assert reference.keys() == target.keys(), (
        reference_name,
        kind,
        sorted(reference),
        sorted(target),
    )
    first = None
    for name in sorted(reference):
        stats = _tensor_statistics(reference[name], target[name])
        exceeded = stats["max_abs"] > _TENSOR_ATOL and stats["max_rel"] > _TENSOR_RTOL
        print(
            "[MFSDP_TENSOR_EVIDENCE] "
            f"rank={dist.get_rank()} step={step} kind={kind} "
            f"reference={reference_name} target=mfsdp tensor={name} "
            f"shape={tuple(target[name].shape)} "
            f"max_abs={stats['max_abs']:.8e} max_rel={stats['max_rel']:.8e} "
            f"cosine={stats['cosine']:.8e} "
            f"reference_norm={stats['reference_norm']:.8e} "
            f"target_norm={stats['target_norm']:.8e} "
            f"atol={_TENSOR_ATOL:.3e} rtol={_TENSOR_RTOL:.3e} "
            f"cosine_tol={_TENSOR_COSINE_TOL:.6f} exceeded={str(exceeded).lower()}",
            flush=True,
        )
        if exceeded and first is None:
            first = {
                "step": step,
                "rank": dist.get_rank(),
                "kind": kind,
                "reference": reference_name,
                "tensor": name,
                **stats,
            }
    return first


def _record_optimizer_state_evidence(
    *,
    step: int,
    reference_name: str,
    reference: dict[str, dict[str, torch.Tensor]],
    target: dict[str, dict[str, torch.Tensor]],
) -> None:
    common_names = sorted(reference.keys() & target.keys())
    assert common_names, (reference_name, sorted(reference), sorted(target))
    for name in common_names[:_OPTIMIZER_STATE_SAMPLE_COUNT]:
        for state_name in ("exp_avg", "exp_avg_sq"):
            assert state_name in reference[name]
            assert state_name in target[name]
            stats = _tensor_statistics(reference[name][state_name], target[name][state_name])
            print(
                "[MFSDP_OPTIMIZER_STATE] "
                f"rank={dist.get_rank()} step={step} reference={reference_name} "
                f"target=mfsdp tensor={name} state={state_name} "
                f"max_abs={stats['max_abs']:.8e} max_rel={stats['max_rel']:.8e} "
                f"cosine={stats['cosine']:.8e} "
                f"reference_norm={stats['reference_norm']:.8e} "
                f"target_norm={stats['target_norm']:.8e}",
                flush=True,
            )


def _global_first_exceedance(local_first: dict[str, Any] | None):
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, local_first)
    candidates = [item for item in gathered if item is not None]
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda item: (
            item["step"],
            item["rank"],
            item["kind"],
            item["reference"],
            item["tensor"],
        ),
    )


def _run_full_parallel_precision(monkeypatch, *, batch_mode: str):
    assert batch_mode in {"matched", "fixed"}
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

    handles = {}
    initial_params = {}
    for backend in _PRECISION_BACKENDS:
        print(
            f"[MFSDP_BUILD] rank={dist.get_rank()} backend={backend} phase=handle_start",
            flush=True,
        )
        handles[backend], initial_params[backend] = _build_full_parallel_handle(
            backend, seed=7345
        )
        print(
            f"[MFSDP_BUILD] rank={dist.get_rank()} backend={backend} phase=handle_done",
            flush=True,
        )
        # Every init_parallel() call creates process groups collectively over WORLD.
        # Pipeline stages have different parameter sets and therefore finish wrapper
        # construction at different times; rendezvous before any rank creates the
        # next reference arm's groups.
        dist.barrier()
        print(
            f"[MFSDP_BUILD] rank={dist.get_rank()} backend={backend} phase=all_ranks_done",
            flush=True,
        )
    mcore_handle = handles["mcore_mfsdp"]
    mfsdp_handle = handles["mfsdp"]
    fsdp2_handle = handles["fsdp2"]
    _assert_distinct_backend_identities(
        fsdp2_handle._optimizer, mfsdp_handle._optimizer
    )
    assert type(mcore_handle._optimizer).__name__ == "_MCoreReferenceOptimizer"

    for handle in handles.values():
        ps = handle._parallel_state
        assert (ps.tp_size, ps.ep_size, ps.etp_size, ps.pp_size, ps.cp_size) == (
            2,
            2,
            1,
            2,
            2,
        )

    initial_diffs = {}
    for reference_name in ("mcore_mfsdp", "fsdp2"):
        value = torch.tensor(
            _max_snapshot_abs_diff(
                initial_params[reference_name], initial_params["mfsdp"]
            ),
            device="cuda",
        )
        dist.all_reduce(value, op=dist.ReduceOp.MAX)
        initial_diffs[reference_name] = float(value)
        assert initial_diffs[reference_name] == 0.0

    losses = {backend: [] for backend in _PRECISION_BACKENDS}
    grad_norms = {backend: [] for backend in _PRECISION_BACKENDS}
    traces = {}
    first_exceedance = None
    for step in range(_FULL_PARALLEL_STEPS):
        step_number = step + 1
        checkpoint = step_number in _PRECISION_CHECKPOINT_STEPS
        grad_snapshots = {}
        for backend in _PRECISION_BACKENDS:
            handle = handles[backend]
            dist.barrier()
            loss, grad_norm, trace, grads = _run_full_parallel_step(
                handle,
                batch_seed=8345 + (step if batch_mode == "matched" else 0),
                record_collectives=step == 0,
                collect_grads=checkpoint,
            )
            losses[backend].append(loss)
            grad_norms[backend].append(grad_norm)
            if checkpoint:
                grad_snapshots[backend] = grads
            if trace:
                traces[backend] = trace
            if dist.get_rank() == 0:
                print(
                    "[MFSDP_STEP] "
                    f"batch_mode={batch_mode} step={step_number} backend={backend} "
                    f"loss={loss:.8e} grad_norm={grad_norm:.8e}",
                    flush=True,
                )

        if checkpoint:
            param_snapshots = {
                backend: _named_model_tensors(handles[backend]._extras["model_chunks"])
                for backend in _PRECISION_BACKENDS
            }
            state_snapshots = {
                backend: _named_optimizer_states(handles[backend]._optimizer)
                for backend in _PRECISION_BACKENDS
            }
            for reference_name in ("mcore_mfsdp", "fsdp2"):
                for kind, snapshots in (
                    ("gradient", grad_snapshots),
                    ("parameter", param_snapshots),
                ):
                    exceedance = _record_tensor_evidence(
                        step=step_number,
                        kind=kind,
                        reference_name=reference_name,
                        reference=snapshots[reference_name],
                        target=snapshots["mfsdp"],
                    )
                    if first_exceedance is None and exceedance is not None:
                        first_exceedance = exceedance
                _record_optimizer_state_evidence(
                    step=step_number,
                    reference_name=reference_name,
                    reference=state_snapshots[reference_name],
                    target=state_snapshots["mfsdp"],
                )

    for backend in _PRECISION_BACKENDS:
        trace = traces.get(backend, ())
        assert trace, f"{backend} tap recorded no distributed collectives."
        assert _contains_collective(trace, "all_gather", "allgather")
        assert _contains_collective(trace, "reduce_scatter", "reducescatter")

    assert cp_splits
    assert all(
        full_tokens == 2 * local_tokens for full_tokens, local_tokens in cp_splits
    )

    loss_rel_curves = {}
    grad_norm_rel_curves = {}
    for reference_name in ("mcore_mfsdp", "fsdp2"):
        loss_curve = torch.tensor(
            _relative_difference_curve(losses[reference_name], losses["mfsdp"]),
            device="cuda",
        )
        grad_curve = torch.tensor(
            _relative_difference_curve(
                grad_norms[reference_name], grad_norms["mfsdp"]
            ),
            device="cuda",
        )
        dist.all_reduce(loss_curve, op=dist.ReduceOp.MAX)
        dist.all_reduce(grad_curve, op=dist.ReduceOp.MAX)
        loss_rel_curves[reference_name] = loss_curve
        grad_norm_rel_curves[reference_name] = grad_curve

    global_first = _global_first_exceedance(first_exceedance)

    if dist.get_rank() == 0:
        ps = mfsdp_handle._parallel_state
        print(
            "[MFSDP_FULL_PARALLEL] "
            f"batch_mode={batch_mode} "
            "topology=tp2_ep2_etp1_pp2_cp2 "
            f"world_size={dist.get_world_size()} "
            f"dp_cp_size={ps.dp_cp_size} "
            f"microbatches={_FULL_PARALLEL_MICROBATCHES} "
            f"steps={_FULL_PARALLEL_STEPS} "
            f"initial_mcore_max_abs_diff={initial_diffs['mcore_mfsdp']:.8e} "
            f"initial_fsdp2_max_abs_diff={initial_diffs['fsdp2']:.8e} "
            f"cp_split={cp_splits[0][0]}->{cp_splits[0][1]} "
            f"cp_split_calls={len(cp_splits)}",
            flush=True,
        )
        for reference_name in ("mcore_mfsdp", "fsdp2"):
            print(
                "[MFSDP_REFERENCE_CURVE] "
                f"batch_mode={batch_mode} reference={reference_name} target=mfsdp "
                f"max_loss_rel={float(loss_rel_curves[reference_name].max()):.8e} "
                f"max_grad_norm_rel={float(grad_norm_rel_curves[reference_name].max()):.8e} "
                f"losses_reference={','.join(f'{value:.8f}' for value in losses[reference_name])} "
                f"losses_target={','.join(f'{value:.8f}' for value in losses['mfsdp'])} "
                f"grad_norms_reference={','.join(f'{value:.8f}' for value in grad_norms[reference_name])} "
                f"grad_norms_target={','.join(f'{value:.8f}' for value in grad_norms['mfsdp'])}",
                flush=True,
            )
            for curve_index in (step - 1 for step in _PRECISION_CHECKPOINT_STEPS):
                print(
                    "[MFSDP_FULL_PARALLEL_CURVE] "
                    f"batch_mode={batch_mode} reference={reference_name} "
                    f"step={curve_index + 1} "
                    f"loss_rel_diff={float(loss_rel_curves[reference_name][curve_index]):.8e} "
                    f"grad_norm_rel_diff={float(grad_norm_rel_curves[reference_name][curve_index]):.8e} "
                    f"loss_reference={losses[reference_name][curve_index]:.8f} "
                    f"loss_mfsdp={losses['mfsdp'][curve_index]:.8f} "
                    f"grad_norm_reference={grad_norms[reference_name][curve_index]:.8f} "
                    f"grad_norm_mfsdp={grad_norms['mfsdp'][curve_index]:.8f}",
                    flush=True,
                )
        if global_first is None:
            print(
                "[MFSDP_FIRST_THRESHOLD_EXCEEDANCE] status=none "
                f"atol={_TENSOR_ATOL:.3e} rtol={_TENSOR_RTOL:.3e}",
                flush=True,
            )
        else:
            print(
                "[MFSDP_FIRST_THRESHOLD_EXCEEDANCE] status=found "
                + " ".join(f"{key}={value}" for key, value in global_first.items()),
                flush=True,
            )
        for backend in _PRECISION_BACKENDS:
            trace = traces[backend]
            print(
                "[MFSDP_COMM_TRACE] "
                f"backend={backend} phase=forward_backward events={len(trace)} "
                f"sequence={' > '.join(trace[:96])}",
                flush=True,
            )

    for reference_name in ("mcore_mfsdp", "fsdp2"):
        assert torch.isfinite(loss_rel_curves[reference_name]).all()
        assert torch.isfinite(grad_norm_rel_curves[reference_name]).all()

    if batch_mode == "matched":
        for reference_name in ("mcore_mfsdp", "fsdp2"):
            assert float(loss_rel_curves[reference_name].max()) <= _LOSS_REL_TOL
            assert float(grad_norm_rel_curves[reference_name].max()) <= _LOSS_REL_TOL
        for backend in _PRECISION_BACKENDS:
            assert losses[backend][-1] < losses[backend][0]
    else:
        scalar_exceeded = any(
            float(loss_rel_curves[reference_name].max()) > _LOSS_REL_TOL
            or float(grad_norm_rel_curves[reference_name].max()) > _LOSS_REL_TOL
            for reference_name in ("mcore_mfsdp", "fsdp2")
        )
        assert scalar_exceeded, "Fixed-batch regression no longer reproduces the 50-step drift."
        if dist.get_rank() == 0:
            first_scalar = next(
                (
                    (reference_name, index)
                    for index in range(_FULL_PARALLEL_STEPS)
                    for reference_name in ("mcore_mfsdp", "fsdp2")
                    if float(loss_rel_curves[reference_name][index]) > _LOSS_REL_TOL
                    or float(grad_norm_rel_curves[reference_name][index]) > _LOSS_REL_TOL
                ),
                None,
            )
            assert first_scalar is not None
            reference_name, index = first_scalar
            loss_abs = abs(losses[reference_name][index] - losses["mfsdp"][index])
            loss_scale = max(
                abs(losses[reference_name][index]), abs(losses["mfsdp"][index])
            )
            print(
                "[MFSDP_FIXED_BATCH_EXPLANATION] "
                "batch_mode=fixed repeated_batch_overfits_toward_zero=true "
                f"first_scalar_step={index + 1} reference={reference_name} "
                f"loss_abs={loss_abs:.8e} loss_scale={loss_scale:.8e} "
                f"loss_rel={float(loss_rel_curves[reference_name][index]):.8e} "
                f"tensor_absolute_exceedance={str(global_first is not None).lower()} "
                "conclusion=relative_error_is_amplified_near_zero_but_nonzero_tensor_abs_diff_proves_real_trajectory_drift",
                flush=True,
            )


def test_mfsdp_three_arm_full_parallel_precision(monkeypatch):
    _run_full_parallel_precision(monkeypatch, batch_mode="matched")


def test_mfsdp_fixed_batch_full_parallel_regression(monkeypatch):
    _run_full_parallel_precision(monkeypatch, batch_mode="fixed")
