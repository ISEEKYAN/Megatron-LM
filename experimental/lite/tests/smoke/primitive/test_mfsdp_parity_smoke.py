# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from __future__ import annotations

import os
import sys
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn

from megatron.lite.primitive.optimizers.fsdp2 import (
    build_fsdp2_training_optimizer,
    fsdp2_available,
)
from megatron.lite.primitive.optimizers.mfsdp import build_mfsdp_training_optimizer
from megatron.lite.primitive.parallel import init_parallel
from megatron.lite.runtime.contracts.config import OptimizerConfig, ParallelConfig

pytestmark = [
    pytest.mark.mlite,
    pytest.mark.smoke,
    pytest.mark.gpu,
    pytest.mark.distributed,
]

_MFSDP_SHARDING_STRATEGY = "optim_grads_params"


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


@pytest.fixture(scope="module", autouse=True)
def _single_node_cuda_dist():
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
    if world_size > 8:
        pytest.skip("Megatron Lite smoke tests are capped at single-node 8 GPUs.")

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
    try:
        from megatron.core import parallel_state as mpu

        if mpu.is_initialized():
            mpu.destroy_model_parallel()
    finally:
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


def _optimizer_cfg() -> OptimizerConfig:
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
        "mfsdp_sharding_strategy": _MFSDP_SHARDING_STRATEGY
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
):
    ps = _parallel_state(parallel)
    chunks = [_new_model(seed, model_type)]
    optimizer = build_fsdp2_training_optimizer(
        chunks,
        _optimizer_cfg(),
        ps,
        unit_modules=(TinyUnit,),
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
):
    ps = _parallel_state(parallel)
    chunks = [_new_model(seed, model_type)]
    impl_cfg = SimpleNamespace(
        parallel=parallel,
        optimizer_config=_optimizer_cfg(),
    )
    optimizer, finalize = build_mfsdp_training_optimizer(
        chunks,
        model_cfg=_model_cfg(),
        impl_cfg=impl_cfg,
        ps=ps,
        is_expert=expert_classifier or (lambda _name: False),
        fsdp_unit_modules=(TinyUnit,),
        deterministic=True,
    )
    return chunks, optimizer, finalize


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
    model_chunks = getattr(optimizer, "model_chunks", None)
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
            rank_major = torch.empty_like(bucket.grad_comm_buffer)
            if bucket.world_size == 1:
                rank_major.copy_(bucket.grad_shard_buffer)
            else:
                dist.all_gather_into_tensor(
                    rank_major,
                    bucket.grad_shard_buffer,
                    group=bucket.process_group,
                )
            for spec in bucket.specs:
                full_grad = torch.zeros(
                    spec.padded_numel,
                    dtype=bucket.grad_dtype,
                    device=bucket.device,
                )
                for rank in range(bucket.world_size):
                    source_offset = rank * bucket.local_numel + spec.local_offset
                    destination_offset = rank * spec.shard_numel
                    full_grad.narrow(0, destination_offset, spec.shard_numel).copy_(
                        rank_major.narrow(0, source_offset, spec.shard_numel)
                    )
                grads[f"{chunk_index}.{spec.name}"] = (
                    full_grad[: spec.numel].view(spec.shape).cpu().float().clone()
                )
    return grads


def _assert_tensor_sets_equal(
    lhs: dict[str, torch.Tensor], rhs: dict[str, torch.Tensor]
) -> float:
    assert lhs.keys() == rhs.keys()
    max_abs = 0.0
    for name in lhs:
        diff = (lhs[name] - rhs[name]).abs()
        max_abs = max(max_abs, float(diff.max().item()))
        assert torch.equal(lhs[name], rhs[name]), name
    return max_abs


def _dense_parallel_config() -> ParallelConfig:
    return ParallelConfig(tp=1, ep=1, etp=1, pp=1, vpp=1, cp=1)


def _tp_ep_parallel_config() -> ParallelConfig:
    return ParallelConfig(tp=2, ep=2, etp=1, pp=1, vpp=1, cp=1)


def _is_tiny_expert(name: str) -> bool:
    return name.startswith("experts.")


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
    assert fsdp2_loss == mfsdp_loss
    assert fsdp2_grad_norm == mfsdp_grad_norm
    max_grad_abs = _assert_tensor_sets_equal(fsdp2_grads, mfsdp_grads)
    max_param_abs = _assert_tensor_sets_equal(
        _named_model_tensors(fsdp2_chunks), _named_model_tensors(mfsdp_chunks)
    )
    assert max_grad_abs == 0.0
    assert max_param_abs == 0.0

    if dist.get_rank() == 0:
        print(
            "[MFSDP_PARITY] "
            f"world_size={dist.get_world_size()} "
            f"strategy={_MFSDP_SHARDING_STRATEGY} "
            f"loss_fsdp2={fsdp2_loss:.8f} "
            f"loss_mfsdp={mfsdp_loss:.8f} "
            f"grad_norm_fsdp2={fsdp2_grad_norm:.8f} "
            f"grad_norm_mfsdp={mfsdp_grad_norm:.8f} "
            f"max_grad_abs_diff={max_grad_abs:.8e} "
            f"max_param_abs_diff={max_param_abs:.8e}",
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
    assert fsdp2_loss == mfsdp_loss
    assert fsdp2_grad_norm == mfsdp_grad_norm
    max_grad_abs = _assert_tensor_sets_equal(fsdp2_grads, mfsdp_grads)
    max_param_abs = _assert_tensor_sets_equal(
        _named_model_tensors(fsdp2_chunks), _named_model_tensors(mfsdp_chunks)
    )
    assert max_grad_abs == 0.0
    assert max_param_abs == 0.0

    if dist.get_rank() == 0:
        print(
            "[MFSDP_PARITY] "
            f"world_size={dist.get_world_size()} "
            "topology=tp2_ep2 "
            f"strategy={_MFSDP_SHARDING_STRATEGY} "
            f"loss_fsdp2={fsdp2_loss:.8f} "
            f"loss_mfsdp={mfsdp_loss:.8f} "
            f"grad_norm_fsdp2={fsdp2_grad_norm:.8f} "
            f"grad_norm_mfsdp={mfsdp_grad_norm:.8f} "
            f"max_grad_abs_diff={max_grad_abs:.8e} "
            f"max_param_abs_diff={max_param_abs:.8e}",
            flush=True,
        )
