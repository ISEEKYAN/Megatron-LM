# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from __future__ import annotations

import json
import math
import os
import sys
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from megatron.lite.primitive.modules.experts import Experts
from megatron.lite.primitive.optimizers.fsdp2 import fsdp2_available
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
_FULL_PARALLEL_WORLD_SIZE = 8


class TinyTELinear(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        import transformer_engine.pytorch as te

        self.linear = te.Linear(
            in_features, out_features, bias=False, params_dtype=torch.bfloat16
        )
    def forward(self, x):
        return self.linear(x)


class TinyTETransformerLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense = TinyTELinear(8, 8)
        self.expert = TinyTELinear(8, 8)

    def forward(self, x):
        return torch.tanh(self.dense(x) + self.expert(x))


class TinyTEQwen3MoE(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList(
            [TinyTETransformerLayer(), TinyTETransformerLayer()]
        )
        self.out = TinyTELinear(8, 4)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.out(x)


class TinyTEGroupedExpertLayer(nn.Module):
    def __init__(self, ps):
        super().__init__()
        config = SimpleNamespace(
            num_experts=2, hidden_size=8, moe_intermediate_size=16, swiglu_limit=0.0
        )
        self.experts = Experts(config, ps)

    def forward(self, x):
        first_expert_tokens = x.shape[0] // 2
        tokens_per_expert = torch.tensor(
            [first_expert_tokens, x.shape[0] - first_expert_tokens],
            device=x.device,
            dtype=torch.int64,
        )
        return torch.tanh(self.experts(x, tokens_per_expert))


class TinyTEGroupedMoE(nn.Module):
    def __init__(self, ps):
        super().__init__()
        self.layers = nn.ModuleList(
            [TinyTEGroupedExpertLayer(ps), TinyTEGroupedExpertLayer(ps)]
        )
        self.out = TinyTELinear(8, 4)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.out(x)


class TinyNonFusedLinear(nn.Module):
    """Regular autograd linear covering M-FSDP's non-TE gradient path."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4, bias=False, dtype=torch.bfloat16)

    def forward(self, x):
        return self.linear(x)


class RecordingSGD(torch.optim.SGD):
    def __init__(self, param_groups):
        super().__init__(param_groups, lr=0.0)
        self.consumed_grad_groups: list[list[torch.Tensor]] = []
        self.consumed_grad_by_param: dict[int, torch.Tensor] = {}

    def step(self, closure=None):
        self.consumed_grad_groups = [
            [
                param.grad.detach().clone()
                for param in group["params"]
                if param.grad is not None
            ]
            for group in self.param_groups
        ]
        self.consumed_grad_by_param = {
            id(param): param.grad.detach().clone()
            for group in self.param_groups
            for param in group["params"]
            if param.grad is not None
        }
        return super().step(closure)


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
            f"local_world_size={os.environ.get('LOCAL_WORLD_SIZE', world_size)} "
            f"cuda_devices={torch.cuda.device_count()} "
            f"fsdp2_available={fsdp2_available()}",
            flush=True,
        )
    _install_transformer_engine_import_stub_if_needed()
    created_pg = False
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")
        created_pg = True
    try:
        yield
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


def _bf16_roundtrip_difference_fraction(tensors: list[torch.Tensor]) -> float:
    values = torch.cat([tensor.detach().float().reshape(-1) for tensor in tensors])
    assert values.numel() > 0
    rounded = values.to(torch.bfloat16).to(torch.float32)
    assert not torch.equal(values, rounded)
    return float((values != rounded).float().mean().item())


def _is_bf16_roundtrip_exact(tensor: torch.Tensor) -> bool:
    values = tensor.detach().float()
    return torch.equal(values, values.to(torch.bfloat16).to(torch.float32))


def _dense_parallel_config() -> ParallelConfig:
    return ParallelConfig(tp=1, ep=1, etp=1, pp=1, vpp=1, cp=1)


def _grouped_expert_key(name: str) -> str | None:
    layer_name, marker, parameter_name = name.partition("experts.")
    if not marker:
        return None
    weight_name = parameter_name.rsplit(".", 1)[-1]
    if not weight_name.startswith("weight") or not weight_name[6:].isdigit():
        return None
    return f"{layer_name}expert{weight_name[6:]}"


def test_mfsdp_native_fp32_fused_wgrad_reaches_optimizer_groups():
    parallel = _dense_parallel_config()
    ps = _parallel_state(parallel)
    chunks = [_new_model(7319, TinyTEQwen3MoE)]
    optimizer_config = _optimizer_cfg(use_fused_optimizer=False)
    optimizer_config.override_optimizer_config["gradient_accumulation_fusion"] = True
    impl_cfg = SimpleNamespace(parallel=parallel, optimizer_config=optimizer_config)

    def build_recording_optimizer(param_groups, _optimizer_config):
        params = [param for group in param_groups for param in group["params"]]
        assert len(params) == 5
        return RecordingSGD(
            [
                {"params": params[::2], "weight_decay": 0.0},
                {"params": params[1::2], "weight_decay": 0.0},
            ]
        )

    optimizer, finalize = build_mfsdp_training_optimizer(
        chunks,
        impl_cfg=impl_cfg,
        ps=ps,
        is_expert=lambda name: ".expert." in name,
        fsdp_unit_modules=(TinyTETransformerLayer,),
        optimizer_factory=build_recording_optimizer,
    )
    chunk = chunks[0]
    assert len(chunk.param_sync.buckets) == 5
    assert len(optimizer.param_groups) == 2

    optimizer.zero_grad()
    generator = torch.Generator(device="cuda")
    generator.manual_seed(8107 + dist.get_rank())
    value = torch.randn(64, 8, device="cuda", dtype=torch.bfloat16, generator=generator)
    chunk(value).float().square().mean().backward()

    # MCore de-references the temporary full-parameter ``main_grad`` staging as
    # soon as this bucket's reduce-scatter completes.  The persistent FP32
    # result exposed to the optimizer is the sharded parameter gradient.
    for bucket in chunk.param_sync.buckets:
        for spec in bucket.specs:
            assert spec.shard_param is not None
            assert spec.full_param.grad_added_to_main_grad is False
            assert spec.full_param.grad is None
            assert spec.full_param.main_grad.numel() == 0

    finalize()
    sharded_main_grad_fractions = [
        _bf16_roundtrip_difference_fraction(
            [param.grad for param in group["params"] if param.grad is not None]
        )
        for group in optimizer.param_groups
    ]
    optimizer_grad_fractions = [
        _bf16_roundtrip_difference_fraction(
            [param.grad for param in group["params"] if param.grad is not None]
        )
        for group in optimizer.param_groups
    ]
    success, _grad_norm, _num_zeros = optimizer.step()
    recording_optimizer = optimizer._inner_optimizer.optimizer
    consumed_grad_fractions = [
        _bf16_roundtrip_difference_fraction(group)
        for group in recording_optimizer.consumed_grad_groups
    ]

    assert success
    assert all(fraction > 0.0 for fraction in sharded_main_grad_fractions)
    assert all(fraction > 0.0 for fraction in optimizer_grad_fractions)
    assert all(fraction > 0.0 for fraction in consumed_grad_fractions)
    if dist.get_rank() == 0:
        print(
            "[MFSDP_NATIVE_FP32_WGRAD] "
            + json.dumps(
                {
                    "world_size": dist.get_world_size(),
                    "bucket_count": len(chunk.param_sync.buckets),
                    "optimizer_group_count": len(optimizer.param_groups),
                    "sharded_main_grad_low_bit_fractions": sharded_main_grad_fractions,
                    "optimizer_grad_low_bit_fractions": optimizer_grad_fractions,
                    "consumed_grad_low_bit_fractions": consumed_grad_fractions,
                },
                sort_keys=True,
            ),
            flush=True,
        )


def test_mfsdp_native_fp32_grouped_expert_wgrad_reaches_optimizer():
    parallel = _dense_parallel_config()
    ps = _parallel_state(parallel)
    torch.manual_seed(9127)
    torch.cuda.manual_seed_all(9127)
    chunks = [TinyTEGroupedMoE(ps).cuda().to(torch.bfloat16)]
    optimizer_config = _optimizer_cfg(use_fused_optimizer=False)
    optimizer_config.override_optimizer_config["gradient_accumulation_fusion"] = True
    impl_cfg = SimpleNamespace(parallel=parallel, optimizer_config=optimizer_config)

    def build_recording_optimizer(param_groups, _optimizer_config):
        params = [param for group in param_groups for param in group["params"]]
        return RecordingSGD(
            [{"params": [param], "weight_decay": 0.0} for param in params]
        )

    optimizer, finalize = build_mfsdp_training_optimizer(
        chunks,
        impl_cfg=impl_cfg,
        ps=ps,
        is_expert=lambda name: "experts." in name,
        fsdp_unit_modules=(TinyTEGroupedExpertLayer,),
        optimizer_factory=build_recording_optimizer,
    )
    chunk = chunks[0]
    grouped_linear_modules = [
        module for module in chunk.modules() if type(module).__name__ == "GroupedLinear"
    ]
    assert len(grouped_linear_modules) == 4
    assert all(module.fuse_wgrad_accumulation for module in grouped_linear_modules)

    expert_specs = {}
    for bucket in chunk.param_sync.buckets:
        for spec in bucket.specs:
            expert_key = _grouped_expert_key(spec.name)
            if expert_key is not None:
                expert_specs.setdefault(expert_key, []).append(spec)
    assert len(expert_specs) == 4
    assert all(len(specs) == 2 for specs in expert_specs.values())

    optimizer.zero_grad()
    generator = torch.Generator(device="cuda")
    generator.manual_seed(9131 + dist.get_rank())
    value = torch.randn(64, 8, device="cuda", dtype=torch.bfloat16, generator=generator)
    chunk(value).float().square().mean().backward()

    local_error = None
    local_specs = []
    try:
        for specs in expert_specs.values():
            for spec in specs:
                full_param = spec.full_param
                main_grad = getattr(full_param, "main_grad", None)
                local_specs.append(
                    {
                        "name": spec.name,
                        "has_fsdp_param": hasattr(full_param, "__fsdp_param__"),
                        "overwrite_main_grad": getattr(
                            full_param, "overwrite_main_grad", None
                        ),
                        "grad_added": getattr(
                            full_param, "grad_added_to_main_grad", None
                        ),
                        "main_grad": (
                            None
                            if main_grad is None
                            else {
                                "shape": tuple(main_grad.shape),
                                "numel": main_grad.numel(),
                                "dtype": str(main_grad.dtype),
                                "contiguous": main_grad.is_contiguous(),
                            }
                        ),
                        "full_param_grad": (
                            None
                            if full_param.grad is None
                            else str(full_param.grad.dtype)
                        ),
                    }
                )
                if spec.shard_param is None:
                    raise AssertionError(f"{spec.name}: missing shard_param")
                if getattr(full_param, "grad_added_to_main_grad", None) is not False:
                    raise AssertionError(
                        f"{spec.name}: grad_added_to_main_grad was not consumed"
                    )
                if main_grad is not None and main_grad.numel() != 0:
                    raise AssertionError(
                        f"{spec.name}: full main_grad staging was not released"
                    )
                if full_param.grad is not None:
                    raise AssertionError(f"{spec.name}: full_param.grad is populated")
    except Exception as exc:
        local_error = f"{type(exc).__name__}: {exc}"
    gathered_errors = [None] * dist.get_world_size()
    dist.all_gather_object(gathered_errors, local_error)
    dist.barrier()
    if any(gathered_errors):
        pytest.fail(
            "GroupedLinear rank-local diagnostics: " + json.dumps(gathered_errors)
        )

    finalize()
    optimizer_grad_fractions, optimizer_records, optimizer_error = {}, [], None
    try:
        for expert_key, specs in expert_specs.items():
            owned = []
            for spec in specs:
                grad = spec.shard_param.grad
                record = {
                    "expert": expert_key,
                    "name": spec.name,
                    "shard_numel": spec.shard_param.numel(),
                    "grad": (
                        None
                        if grad is None
                        else {"shape": tuple(grad.shape), "dtype": str(grad.dtype)}
                    ),
                    "optimizer_owned": any(
                        spec.shard_param is p
                        for g in optimizer.param_groups
                        for p in g["params"]
                    ),
                }
                optimizer_records.append(record)
                if grad is not None and grad.numel() > 0:
                    assert record["optimizer_owned"]
                    assert grad.dtype is torch.float32
                    owned.append(grad)
            if owned:
                optimizer_grad_fractions[expert_key] = (
                    _bf16_roundtrip_difference_fraction(owned)
                )
    except Exception as exc:
        optimizer_error = f"{type(exc).__name__}: {exc}"
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, optimizer_error)
    all_records = [None] * dist.get_world_size()
    dist.all_gather_object(all_records, optimizer_records)
    if any(gathered):
        pytest.fail("Grouped optimizer-grad diagnostics: " + json.dumps(gathered))
    expected = {spec.name for specs in expert_specs.values() for spec in specs}
    covered = {
        record["name"]
        for records in all_records
        for record in records
        if record["optimizer_owned"]
        and record["grad"] is not None
        and record["shard_numel"] > 0
    }
    if covered != expected:
        pytest.fail(
            f"Grouped optimizer-grad global coverage missing: {expected - covered}"
        )
    success, _grad_norm, _num_zeros = optimizer.step()
    recording_optimizer = optimizer._inner_optimizer.optimizer
    consumed_grad_fractions, consumed_records, consumed_error = {}, [], None
    try:
        for expert_key, specs in expert_specs.items():
            owned = []
            for spec in specs:
                grad = recording_optimizer.consumed_grad_by_param.get(
                    id(spec.shard_param)
                )
                consumed_records.append(
                    {
                        "name": spec.name,
                        "shard_numel": spec.shard_param.numel(),
                        "entry": grad is not None,
                        "dtype": None if grad is None else str(grad.dtype),
                    }
                )
                if grad is not None:
                    owned.append(grad)
            owned = [grad for grad in owned if grad.numel() > 0]
            if owned:
                assert all(grad.dtype is torch.float32 for grad in owned)
                consumed_grad_fractions[expert_key] = (
                    _bf16_roundtrip_difference_fraction(owned)
                )
    except Exception as exc:
        consumed_error = f"{type(exc).__name__}: {exc}"
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, consumed_error)
    if any(gathered):
        pytest.fail("Grouped consumed-grad diagnostics: " + json.dumps(gathered))
    all_consumed = [None] * dist.get_world_size()
    dist.all_gather_object(all_consumed, consumed_records)
    consumed_covered = {
        record["name"]
        for records in all_consumed
        for record in records
        if record["entry"] and record["shard_numel"] > 0
    }
    if consumed_covered != expected:
        pytest.fail(
            f"Grouped consumed-grad global coverage missing: {expected - consumed_covered}"
        )

    assert success
    assert all(fraction > 0.0 for fraction in optimizer_grad_fractions.values())
    assert all(fraction > 0.0 for fraction in consumed_grad_fractions.values())
    if dist.get_rank() == 0:
        print(
            "[MFSDP_NATIVE_FP32_GROUPED_EXPERT_WGRAD] "
            + json.dumps(
                {
                    "world_size": dist.get_world_size(),
                    "expert_group_count": len(expert_specs),
                    "grouped_linear_count": len(grouped_linear_modules),
                    "full_main_grad_staging_released": True,
                    "optimizer_grad_low_bit_fractions": optimizer_grad_fractions,
                    "consumed_grad_low_bit_fractions": consumed_grad_fractions,
                },
                sort_keys=True,
            ),
            flush=True,
        )


def test_mfsdp_non_fused_wgrad_accumulates_in_fp32_per_microbatch():
    parallel = _dense_parallel_config()
    ps = _parallel_state(parallel)
    torch.manual_seed(10103)
    torch.cuda.manual_seed_all(10103)
    model = TinyNonFusedLinear().cuda()
    assert not hasattr(model.linear, "fuse_wgrad_accumulation")
    chunks = [model]
    optimizer_config = _optimizer_cfg(use_fused_optimizer=False)
    optimizer_config.clip_grad = 10000.0
    impl_cfg = SimpleNamespace(parallel=parallel, optimizer_config=optimizer_config)

    def build_recording_optimizer(param_groups, _optimizer_config):
        return RecordingSGD(param_groups)

    optimizer, finalize = build_mfsdp_training_optimizer(
        chunks,
        impl_cfg=impl_cfg,
        ps=ps,
        is_expert=lambda _name: False,
        fsdp_unit_modules=(TinyNonFusedLinear,),
        optimizer_factory=build_recording_optimizer,
    )
    chunk = chunks[0]
    assert len(chunk.param_sync.buckets) == 1
    bucket = chunk.param_sync.buckets[0]
    spec = bucket.specs[0]
    assert spec.shard_param is not None
    assert spec.full_param.grad_added_to_main_grad is False

    # Capture the ephemeral unsharded staging at the exact ready boundary,
    # before the asynchronous reduce-scatter and MCore-style release.  This
    # distinguishes autograd accumulation defects from collective/output
    # placement defects without extending staging lifetime in production.
    staging_snapshots = []
    reduced_before_staging_release = []
    reduce_ready = bucket.grad_ready_callback
    release_staging = bucket._release_full_main_grads
    assert reduce_ready is not None

    def capture_then_reduce(ready_bucket):
        assert ready_bucket is bucket
        staging_snapshots.append(ready_bucket.full_main_grad_buffer.detach().clone())
        reduce_ready(ready_bucket)

    def capture_then_release_staging():
        if bucket._full_main_grad_lease is not None:
            reduced_before_staging_release.append(
                {
                    "grad": bucket.main_grad_buffer.detach().clone(),
                    "input_ptr": bucket.full_main_grad_buffer.untyped_storage().data_ptr(),
                    "output_ptr": bucket.main_grad_buffer.untyped_storage().data_ptr(),
                }
            )
        release_staging()

    bucket.grad_ready_callback = capture_then_reduce
    bucket._release_full_main_grads = capture_then_release_staging

    optimizer.zero_grad()
    chunk(torch.ones(1, 4, device="cuda", dtype=torch.bfloat16)).sum().backward()
    assert spec.shard_param.grad is not None
    single_backward_grad = spec.shard_param.grad.detach().clone()
    assert spec.full_param.grad is None
    assert spec.full_param.main_grad.numel() == 0
    assert single_backward_grad.dtype is torch.float32
    assert len(staging_snapshots) == 1
    assert len(reduced_before_staging_release) == 1
    assert (
        reduced_before_staging_release[-1]["input_ptr"]
        != reduced_before_staging_release[-1]["output_ptr"]
    )
    assert torch.equal(
        reduced_before_staging_release[-1]["grad"],
        torch.cat(
            [
                torch.zeros(spec.local_offset, device="cuda"),
                torch.ones(spec.shard_numel, device="cuda"),
                torch.zeros(
                    bucket.local_numel - spec.local_offset - spec.shard_numel,
                    device="cuda",
                ),
            ]
        ),
    )
    assert torch.equal(
        staging_snapshots[-1].narrow(0, spec.full_offset, spec.numel),
        torch.ones(spec.numel, device="cuda"),
    )
    assert torch.equal(single_backward_grad, torch.ones_like(single_backward_grad))
    assert _is_bf16_roundtrip_exact(single_backward_grad)

    optimizer.zero_grad()
    chunk(
        torch.full((1, 4), 256.0, device="cuda", dtype=torch.bfloat16)
    ).sum().backward()
    assert spec.full_param.grad is None
    assert spec.full_param.main_grad.numel() == 0
    assert spec.shard_param.grad is not None
    assert len(staging_snapshots) == 2
    assert torch.equal(
        staging_snapshots[-1].narrow(0, spec.full_offset, spec.numel),
        torch.full((spec.numel,), 256.0, device="cuda"),
    )
    assert torch.equal(
        spec.shard_param.grad, torch.full_like(spec.shard_param.grad, 256.0)
    )
    assert _is_bf16_roundtrip_exact(spec.shard_param.grad)

    chunk(torch.ones(1, 4, device="cuda", dtype=torch.bfloat16)).sum().backward()
    expected = torch.full((4, 4), 257.0, device="cuda")
    assert spec.full_param.grad is None
    assert spec.full_param.main_grad.numel() == 0
    assert len(staging_snapshots) == 3
    assert torch.equal(
        staging_snapshots[-1].narrow(0, spec.full_offset, spec.numel),
        torch.ones(spec.numel, device="cuda"),
    )

    finalize()
    expected_shard = expected.reshape(-1).narrow(0, spec.param_offset, spec.shard_numel)
    assert spec.shard_param.grad is not None
    assert spec.shard_param.grad.dtype is torch.float32
    assert torch.equal(spec.shard_param.grad, expected_shard)
    success, _grad_norm, _num_zeros = optimizer.step()
    recording_optimizer = optimizer._inner_optimizer.optimizer
    consumed_grad = recording_optimizer.consumed_grad_by_param[id(spec.shard_param)]
    assert success
    assert torch.equal(consumed_grad, expected_shard)
    if dist.get_rank() == 0:
        print(
            "[MFSDP_NON_FUSED_FP32_ACCUMULATION] "
            + json.dumps(
                {
                    "world_size": dist.get_world_size(),
                    "single_backward_bf16_roundtrip_exact": True,
                    "multi_microbatch_value": float(consumed_grad[0]),
                    "multi_microbatch_bf16_roundtrip_exact": False,
                },
                sort_keys=True,
            ),
            flush=True,
        )


def test_mfsdp_training_and_rollout_offload_distributed_smoke():
    """Exercise both offload contracts with real NCCL sharding on every rank."""
    parallel = _dense_parallel_config()
    ps = _parallel_state(parallel)
    chunks = [_new_model(17321, TinyTEQwen3MoE)]
    optimizer_config = _optimizer_cfg(use_fused_optimizer=False)
    optimizer_config.offload_fraction = 1.0
    optimizer_config.override_optimizer_config.update(
        {"gradient_accumulation_fusion": True, "bucket_size": 17}
    )
    impl_cfg = SimpleNamespace(parallel=parallel, optimizer_config=optimizer_config)
    optimizer, finalize = build_mfsdp_training_optimizer(
        chunks,
        impl_cfg=impl_cfg,
        ps=ps,
        is_expert=lambda name: ".expert." in name,
        fsdp_unit_modules=(TinyTETransformerLayer,),
    )
    chunk = chunks[0]
    cpu_group = optimizer._inner_optimizer.cpu_group
    assert cpu_group is not None

    def run_step(seed: int) -> float:
        optimizer.zero_grad()
        generator = torch.Generator(device="cuda")
        generator.manual_seed(seed + dist.get_rank())
        value = torch.randn(
            64, 8, device="cuda", dtype=torch.bfloat16, generator=generator
        )
        loss = chunk(value).float().square().mean()
        loss.backward()
        finalize()
        success, grad_norm, _num_zeros = optimizer.step()
        assert success and math.isfinite(grad_norm)
        return float(loss.item())

    first_loss = run_step(17331)
    assert cpu_group.d2h_bytes > 0 and cpu_group.h2d_bytes > 0
    assert cpu_group.live_transfer_leases == 0
    assert cpu_group.ring_high_water_elements <= min(
        sum(param.numel() for param in cpu_group.gpu_params), 2 * 17
    )
    assert all(
        state["master_param"].device.type == "cpu"
        and state["exp_avg"].device.type == "cpu"
        and state["exp_avg_sq"].device.type == "cpu"
        for state in cpu_group._optimizer.state.values()
    )

    optimizer.offload_for_rollout()
    assert all(bucket.device.type == "cpu" for bucket in chunk.param_sync.buckets)
    assert all(bucket.main_grad_buffer.numel() == 0 for bucket in chunk.param_sync.buckets)
    optimizer.load_from_rollout()
    assert all(bucket.device.type == "cuda" for bucket in chunk.param_sync.buckets)
    second_loss = run_step(17341)

    if dist.get_rank() == 0:
        print(
            "[MFSDP_TWO_OFFLOADS] "
            + json.dumps(
                {
                    "world_size": dist.get_world_size(),
                    "first_loss": first_loss,
                    "second_loss": second_loss,
                    "d2h_bytes": cpu_group.d2h_bytes,
                    "h2d_bytes": cpu_group.h2d_bytes,
                    "ring_high_water_elements": cpu_group.ring_high_water_elements,
                },
                sort_keys=True,
            ),
            flush=True,
        )
