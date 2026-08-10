# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from __future__ import annotations

import json
import os
import statistics
import sys
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from megatron.lite.primitive.modules.experts import Experts
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
from torch.utils._python_dispatch import TorchDispatchMode

pytestmark = [
    pytest.mark.mlite,
    pytest.mark.smoke,
    pytest.mark.gpu,
    pytest.mark.distributed,
]

_MFSDP_SHARDING_STRATEGY = "optim_grads_params"
_PRECISION_STEPS = 50
_LOSS_REL_TOL = 1.0e-2
_E2E_SFT_LOSS_REL_TOL = 5.09e-3
_TENSOR_RTOL = 1.0e-2
_TENSOR_ATOL = 1.0e-5
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


def _model_cfg() -> SimpleNamespace:
    return SimpleNamespace(
        num_hidden_layers=2, hidden_size=8, num_attention_heads=1, add_bias_linear=False
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
    activation_offload: bool = False,
):
    ps = _parallel_state(parallel)
    chunks = [_new_model(seed, model_type)]
    if activation_offload:
        apply_offload(chunks[0].layers, ["full"], {})
    optimizer_config = _optimizer_cfg(use_fused_optimizer=False)
    optimizer_config.offload_fraction = 0.0
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
    activation_offload: bool = False,
    use_fused_optimizer: bool = True,
):
    ps = _parallel_state(parallel)
    chunks = [_new_model(seed, model_type)]
    if activation_offload:
        apply_offload(chunks[0].layers, ["full"], {})
    optimizer_config = _optimizer_cfg(use_fused_optimizer=use_fused_optimizer)
    optimizer_config.offload_fraction = 0.0
    impl_cfg = SimpleNamespace(parallel=parallel, optimizer_config=optimizer_config)
    optimizer, finalize = build_mfsdp_training_optimizer(
        chunks,
        impl_cfg=impl_cfg,
        ps=ps,
        is_expert=expert_classifier or (lambda _name: False),
        fsdp_unit_modules=unit_modules,
    )
    return chunks, optimizer, finalize


def _bf16_roundtrip_difference_fraction(tensors: list[torch.Tensor]) -> float:
    values = torch.cat([tensor.detach().float().reshape(-1) for tensor in tensors])
    assert values.numel() > 0
    rounded = values.to(torch.bfloat16).to(torch.float32)
    assert not torch.equal(values, rounded)
    return float((values != rounded).float().mean().item())


def _is_bf16_roundtrip_exact(tensor: torch.Tensor) -> bool:
    values = tensor.detach().float()
    return torch.equal(values, values.to(torch.bfloat16).to(torch.float32))


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
    assert fsdp2_chain[0].startswith(
        "megatron.lite.primitive.optimizers.fsdp2."
    ), fsdp2_chain
    assert mfsdp_chain[0].startswith(
        "megatron.lite.primitive.optimizers.mfsdp."
    ), mfsdp_chain
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
                    rank_major, bucket.grad_shard_buffer, group=bucket.process_group
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
    optimizer = bundle.optimizer
    finalize_grads = bundle.finalize_grads
    if callable(post_load_hook):
        updates = post_load_hook()
        assert isinstance(updates, dict)
        optimizer = updates.get("optimizer", optimizer)
        finalize_grads = updates.get("finalize_grads", finalize_grads)
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


@pytest.mark.parametrize("reference_backend", ("dist_opt", "fsdp2"))
def test_mfsdp_matches_reference_full_parallel_precision_curve(
    monkeypatch, reference_backend
):
    if dist.get_world_size() != _FULL_PARALLEL_WORLD_SIZE:
        pytest.skip("M-FSDP TP2/EP2/ETP1/PP2/CP2 signoff requires exactly 8 ranks.")

    if int(os.environ.get("LOCAL_WORLD_SIZE", "0")) != _FULL_PARALLEL_WORLD_SIZE:
        pytest.fail("M-FSDP full-parallel signoff requires all 8 ranks on one node.")

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

    reference_handle, reference_initial = _build_full_parallel_handle(
        reference_backend, seed=7345
    )
    mfsdp_handle, mfsdp_initial = _build_full_parallel_handle("mfsdp", seed=7345)
    if reference_backend == "fsdp2":
        _assert_distinct_backend_identities(
            reference_handle._optimizer, mfsdp_handle._optimizer
        )
    else:
        assert reference_handle._optimizer is not mfsdp_handle._optimizer
        assert _qualified_type(reference_handle._optimizer).startswith(
            "megatron.core.optimizer."
        )
        assert _qualified_type(mfsdp_handle._optimizer).startswith(
            "megatron.lite.primitive.optimizers.mfsdp."
        )
        if dist.get_rank() == 0:
            print(
                "[MFSDP_BACKEND] "
                f"configured={reference_backend} optimizer_type="
                f"{_qualified_type(reference_handle._optimizer)}",
                flush=True,
            )

    for handle in (reference_handle, mfsdp_handle):
        ps = handle._parallel_state
        assert (ps.tp_size, ps.ep_size, ps.etp_size, ps.pp_size, ps.cp_size) == (
            2,
            2,
            1,
            2,
            2,
        )

    initial_max_abs = torch.tensor(
        _max_snapshot_abs_diff(reference_initial, mfsdp_initial), device="cuda"
    )
    dist.all_reduce(initial_max_abs, op=dist.ReduceOp.MAX)
    assert float(initial_max_abs) == 0.0

    losses = {reference_backend: [], "mfsdp": []}
    grad_norms = {reference_backend: [], "mfsdp": []}
    wandb_run = None
    if dist.get_rank() == 0:
        import wandb

        wandb_run = wandb.init(
            project=os.environ.get("WANDB_PROJECT", "mlite-mfsdp-precision"),
            entity=os.environ.get("WANDB_ENTITY", "megatron-core-moe-dev"),
            name=os.environ.get("WANDB_NAME", f"mfsdp-sft-curve-{reference_backend}"),
            config={
                "reference_backend": reference_backend,
                "target_backend": "mfsdp",
                "seed": 7345,
                "steps": _FULL_PARALLEL_STEPS,
                "microbatches": _FULL_PARALLEL_MICROBATCHES,
                "same_input_sft": True,
            },
        )
        print(f"[MFSDP_WANDB] url={wandb_run.url}", flush=True)
    traces = {}
    for step in range(_FULL_PARALLEL_STEPS):
        for backend, handle in (
            (reference_backend, reference_handle),
            ("mfsdp", mfsdp_handle),
        ):
            dist.barrier()
            loss, grad_norm, trace = _run_full_parallel_step(
                handle, batch_seed=8345 + step, record_collectives=step == 0
            )
            losses[backend].append(loss)
            grad_norms[backend].append(grad_norm)
            if trace:
                traces[backend] = trace
        if wandb_run is not None:
            wandb_run.log(
                {
                    "step": step + 1,
                    f"{reference_backend}/sft_loss": losses[reference_backend][-1],
                    "mfsdp/sft_loss": losses["mfsdp"][-1],
                    f"{reference_backend}/grad_norm": grad_norms[reference_backend][-1],
                    "mfsdp/grad_norm": grad_norms["mfsdp"][-1],
                },
                step=step + 1,
            )

    for backend in (reference_backend, "mfsdp"):
        trace = traces.get(backend, ())
        assert trace, f"{backend} tap recorded no distributed collectives."
        assert _contains_collective(trace, "all_gather", "allgather")
        assert _contains_collective(trace, "reduce_scatter", "reducescatter")

    assert cp_splits
    assert all(
        full_tokens == 2 * local_tokens for full_tokens, local_tokens in cp_splits
    )

    loss_rel_curve = torch.tensor(
        _relative_difference_curve(losses[reference_backend], losses["mfsdp"]),
        device="cuda",
    )
    grad_norm_rel_curve = torch.tensor(
        _relative_difference_curve(grad_norms[reference_backend], grad_norms["mfsdp"]),
        device="cuda",
    )
    dist.all_reduce(loss_rel_curve, op=dist.ReduceOp.MAX)
    dist.all_reduce(grad_norm_rel_curve, op=dist.ReduceOp.MAX)
    max_loss_rel_diff = loss_rel_curve.max()
    max_grad_norm_rel_diff = grad_norm_rel_curve.max()

    if dist.get_rank() == 0:
        ps = reference_handle._parallel_state
        print(
            "[MFSDP_FULL_PARALLEL] "
            "topology=tp2_ep2_etp1_pp2_cp2 "
            f"world_size={dist.get_world_size()} "
            f"dp_cp_size={ps.dp_cp_size} "
            f"microbatches={_FULL_PARALLEL_MICROBATCHES} "
            f"steps={_FULL_PARALLEL_STEPS} "
            f"reference={reference_backend} "
            f"sft_loss_rel_tol={_E2E_SFT_LOSS_REL_TOL:.3e} "
            f"initial_max_abs_diff={float(initial_max_abs):.8e} "
            f"max_loss_rel_diff={float(max_loss_rel_diff):.8e} "
            f"max_grad_norm_rel_diff={float(max_grad_norm_rel_diff):.8e} "
            f"{reference_backend}_losses={','.join(f'{value:.8f}' for value in losses[reference_backend])} "
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
                f"loss_{reference_backend}={losses[reference_backend][curve_index]:.8f} "
                f"loss_mfsdp={losses['mfsdp'][curve_index]:.8f} "
                f"grad_norm_{reference_backend}={grad_norms[reference_backend][curve_index]:.8f} "
                f"grad_norm_mfsdp={grad_norms['mfsdp'][curve_index]:.8f}",
                flush=True,
            )
        for backend in (reference_backend, "mfsdp"):
            trace = traces[backend]
            print(
                "[MFSDP_COMM_TRACE] "
                f"backend={backend} phase=forward_backward events={len(trace)} "
                f"sequence={' > '.join(trace[:96])}",
                flush=True,
            )

    assert torch.isfinite(loss_rel_curve).all()
    assert torch.isfinite(grad_norm_rel_curve).all()
    assert float(max_loss_rel_diff) <= _E2E_SFT_LOSS_REL_TOL
    assert float(max_grad_norm_rel_diff) <= _LOSS_REL_TOL
    assert losses[reference_backend][-1] < losses[reference_backend][0]
    assert losses["mfsdp"][-1] < losses["mfsdp"][0]
    if wandb_run is not None:
        wandb_run.finish()
