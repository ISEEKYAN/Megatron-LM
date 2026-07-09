# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from __future__ import annotations

import ast
import copy
from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from megatron.lite.primitive.optimizers import get_optimizer_backend
from megatron.lite.primitive.optimizers.mfsdp import config as mfsdp_config
from megatron.lite.primitive.optimizers.mfsdp import grad_norm as mfsdp_grad_norm
from megatron.lite.primitive.optimizers.mfsdp import optimizer as mfsdp_optimizer
from megatron.lite.primitive.optimizers.mfsdp.patches import (
    should_skip_tp_duplicate_sync,
)
from megatron.lite.runtime.contracts.config import ParallelConfig


class _GlooUnit(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, out_features, bias=False)

    def forward(self, value):
        return torch.nn.functional.gelu(self.linear(value))


class _GlooModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.unit0 = _GlooUnit(4, 5)
        self.unit1 = _GlooUnit(5, 3)
        self.out = torch.nn.Linear(3, 2, bias=False)

    def forward(self, value):
        return self.out(self.unit1(self.unit0(value)))


class _DirectParamModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.unit = _GlooUnit(4, 4)
        self.projection = torch.nn.Linear(4, 3, bias=False)

    def forward(self, value):
        hidden = self.unit(value)
        return torch.nn.functional.linear(hidden, self.projection.weight)


def test_mfsdp_has_no_megatron_core_imports():
    package = Path(mfsdp_config.__file__).parent
    violations = []
    for source_path in package.rglob("*.py"):
        tree = ast.parse(source_path.read_text(), filename=str(source_path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [node.module or ""]
            else:
                continue
            for module in modules:
                if module == "megatron.core" or module.startswith("megatron.core."):
                    relative_path = source_path.relative_to(package)
                    violations.append(f"{relative_path}:{node.lineno}: {module}")

    assert violations == []


def test_mfsdp_is_standalone_without_vendored_mcore_or_fsdp2_dependencies():
    package = Path(mfsdp_config.__file__).parent

    assert not list((package / "impl").glob("*.py")), (
        "Do not vendor the MCore FSDP implementation."
    )
    violations = []
    for source_path in package.rglob("*.py"):
        tree = ast.parse(source_path.read_text(), filename=str(source_path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [node.module or ""]
            else:
                continue
            for module in modules:
                if module.startswith("megatron.") and not module.startswith(
                    "megatron.lite.primitive.optimizers.mfsdp"
                ):
                    relative_path = source_path.relative_to(package)
                    violations.append(f"{relative_path}:{node.lineno}: {module}")

    assert violations == []


def test_mfsdp_imports_when_megatron_core_is_blocked():
    script = """
import builtins
original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    forbidden = (
        name == 'megatron.core'
        or name.startswith('megatron.core.')
        or name == 'megatron.lite.primitive.optimizers.fsdp2'
        or name.startswith('megatron.lite.primitive.optimizers.fsdp2.')
    )
    if forbidden:
        raise RuntimeError(f'forbidden import: {name}')
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
import megatron.lite.primitive.optimizers.mfsdp
"""
    subprocess.run([sys.executable, "-c", script], check=True)


@dataclass
class _FakeDDPConfig:
    use_distributed_optimizer: bool = False
    use_megatron_fsdp: bool = False
    data_parallel_sharding_strategy: str = "no_shard"
    bucket_size: int | None = None
    overlap_grad_reduce: bool = False
    overlap_param_gather: bool = False
    num_distributed_optimizer_instances: int = 1
    nccl_ub: bool = False
    fsdp_double_buffer: bool = False
    megatron_fsdp_main_params_dtype: torch.dtype | None = None
    megatron_fsdp_main_grads_dtype: torch.dtype | None = None
    megatron_fsdp_grad_comm_dtype: torch.dtype | None = None


def _engine_cfg(**overrides):
    cfg = SimpleNamespace(
        model_name="qwen3_5",
        parallel=ParallelConfig(tp=1, ep=1, etp=1, pp=1, vpp=1, cp=1),
        optimizer=SimpleNamespace(optimizer="adam", override_optimizer_config={}),
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def test_mfsdp_backend_registry_resolves_backend():
    backend = get_optimizer_backend("mfsdp")

    assert backend.name == "megatron_fsdp"
    assert backend.runtime_backend == "megatron_fsdp"


def test_mfsdp_config_lowers_aliases_and_preserves_optimizer_keys():
    opt = SimpleNamespace(
        override_optimizer_config={
            "mfsdp_sharding_strategy": "optim_grads",
            "bucket_size": 1234,
            "adam_beta1": 0.8,
        }
    )

    opt_overrides, ddp_overrides = mfsdp_config.split_mfsdp_overrides(
        opt,
        _FakeDDPConfig,
    )

    assert opt_overrides == {"adam_beta1": 0.8}
    assert ddp_overrides["data_parallel_sharding_strategy"] == "optim_grads"
    assert ddp_overrides["bucket_size"] == 1234

    cfg = mfsdp_config.build_mfsdp_ddp_config(
        _FakeDDPConfig,
        {
            "megatron_fsdp_main_params_dtype": "bf16",
            "megatron_fsdp_grad_comm_dtype": "torch.float16",
        },
    )

    assert cfg.use_distributed_optimizer is True
    assert cfg.use_megatron_fsdp is True
    assert cfg.data_parallel_sharding_strategy == "optim_grads_params"
    assert cfg.megatron_fsdp_main_params_dtype is torch.bfloat16
    assert cfg.megatron_fsdp_main_grads_dtype is None
    assert cfg.megatron_fsdp_grad_comm_dtype is torch.float16


def test_mfsdp_config_rejects_unsupported_optimizer():
    engine_cfg = _engine_cfg(
        optimizer=SimpleNamespace(optimizer="adamw", override_optimizer_config={})
    )
    with pytest.raises(ValueError, match="adam/sgd"):
        mfsdp_config.validate_mfsdp_config(engine_cfg)


def test_mfsdp_grad_clip_scales_unique_grads_and_decoupled_grads():
    shared = torch.nn.Parameter(torch.ones(2))
    shared.grad = torch.ones(2)
    other = torch.nn.Parameter(torch.ones(2))
    other.grad = torch.ones(2)
    decoupled = torch.nn.Parameter(torch.ones(2))
    decoupled.grad = torch.ones(2)
    decoupled.decoupled_grad = torch.ones(2)

    class _Leaf:
        def __init__(self, params, *, use_decoupled_grad=False):
            self.is_stub_optimizer = False
            self.config = SimpleNamespace(
                clip_grad=1.0,
                use_precision_aware_optimizer_no_fp8_or_ds_fp8=use_decoupled_grad,
            )
            self.param_groups = [{"params": params}]

        def get_parameters(self):
            return []

    optimizer = SimpleNamespace(
        chained_optimizers=[
            _Leaf([shared, other]),
            _Leaf([shared]),
            _Leaf([decoupled], use_decoupled_grad=True),
        ],
    )

    mfsdp_grad_norm._clip_mfsdp_grads_by_total_norm(optimizer, grad_norm=4.0)

    assert torch.allclose(shared.grad, torch.full_like(shared.grad, 0.25))
    assert torch.allclose(other.grad, torch.full_like(other.grad, 0.25))
    assert torch.equal(decoupled.grad, torch.ones(2))
    assert torch.allclose(
        decoupled.decoupled_grad, torch.full_like(decoupled.grad, 0.25)
    )


def test_mfsdp_metadata_infers_tp_partition_attrs_for_known_weights():
    class _Leaf(torch.nn.Module):
        def __init__(self, *, param_name: str = "weight"):
            super().__init__()
            param = torch.nn.Parameter(torch.randn(4, 3))
            param.tensor_model_parallel = True
            setattr(self, param_name, param)

    class _Layer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = torch.nn.Module()
            self.attn.qkv = torch.nn.Module()
            self.attn.qkv.linear = _Leaf()
            self.attn.proj = torch.nn.Module()
            self.attn.proj.linear = _Leaf()
            self.moe = torch.nn.Module()
            self.moe.experts = torch.nn.Module()
            self.moe.experts.fc1 = _Leaf(param_name="weight0")
            self.moe.experts.fc2 = _Leaf(param_name="weight0")

    model = torch.nn.Module()
    model.layers = torch.nn.ModuleList([_Layer()])

    mfsdp_optimizer.ensure_mfsdp_tp_partition_attrs(model)

    layer = model.layers[0]
    assert layer.attn.qkv.linear.weight.partition_dim == 0
    assert layer.attn.proj.linear.weight.partition_dim == 1
    assert layer.moe.experts.fc1.weight0.partition_dim == 0
    assert layer.moe.experts.fc2.weight0.partition_dim == 1
    assert should_skip_tp_duplicate_sync(layer.attn.qkv.linear.weight)


def test_mfsdp_marks_sequence_parallel_shards_for_tp_gradient_sync():
    model = _GlooModel()
    model.sp_params = [model.out.weight]
    ps = SimpleNamespace(
        dp_cp_group=None,
        dp_group=None,
        ep_dp_group=None,
        ep_group=None,
        etp_group=None,
        tp_group=None,
        pp_group=None,
        dp_cp_size=1,
        expert_dp_size=1,
    )
    opt = SimpleNamespace(
        optimizer="adam",
        lr=1.0e-3,
        min_lr=0.0,
        weight_decay=0.0,
        clip_grad=1.0,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_eps=1.0e-8,
        override_optimizer_config={"mfsdp_sharding_strategy": "optim_grads_params"},
    )

    _chunks, optimizer = mfsdp_optimizer.build_mfsdp_stack(
        [model],
        model_cfg=SimpleNamespace(),
        engine_cfg=SimpleNamespace(
            model_name="qwen3_5",
            parallel=ParallelConfig(tp=2, ep=1, etp=1, pp=1, vpp=1, cp=1),
            optimizer=opt,
        ),
        ps=ps,
        is_expert=lambda _name: False,
        fsdp_unit_modules=(_GlooUnit,),
    )

    out_shard = next(
        param
        for param in optimizer.params
        if optimizer.param_names[id(param)].endswith("out.weight")
    )
    assert out_shard.sequence_parallel is True
    assert out_shard.tensor_model_parallel is False
    assert optimizer._inner_optimizer.tp_replicated_params == [out_shard]


def test_mfsdp_syncs_average_gradients_across_tp_domain_params(monkeypatch):
    vision_param = torch.nn.Parameter(torch.ones(1))
    vision_param.average_gradients_across_tp_domain = True
    vision_param.grad = torch.ones(1)
    tp_group = object()
    reduced = []

    def record_grad_reduction(grad, group, *, average=False):
        reduced.append((grad, group, average))

    monkeypatch.setattr(
        mfsdp_optimizer, "_all_reduce_grad_if_distributed", record_grad_reduction
    )
    adapter = mfsdp_optimizer._StandaloneOptimizer(
        torch.optim.SGD([vision_param], lr=0.0),
        [vision_param],
        {id(vision_param): "vision.weight"},
        ps=SimpleNamespace(
            dp_cp_group=None,
            dp_group=None,
            ep_dp_group=None,
            ep_group=None,
            etp_group=None,
            tp_group=tp_group,
            pp_group=None,
        ),
        clip_grad=0.0,
        grad_norm_accum_dtype=torch.float32,
        expert_params=[],
        expert_grad_scale=1.0,
    )

    adapter.step()

    assert adapter.tp_replicated_params == [vision_param]
    assert reduced == [(vision_param.grad, tp_group, True)]


def test_mfsdp_tp_gradient_average_divides_the_collective_sum(monkeypatch):
    grad = torch.ones(2)
    group = object()

    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda _group: 2)
    monkeypatch.setattr(
        dist,
        "all_reduce",
        lambda value, *, op, group: value.mul_(2.0),
    )

    mfsdp_optimizer._all_reduce_grad_if_distributed(grad, group, average=True)

    assert torch.equal(grad, torch.ones_like(grad))


def test_mfsdp_standalone_step_covers_tp_and_expert_norm_domains(monkeypatch):
    dense = torch.nn.Parameter(torch.ones(1))
    dense.grad = torch.ones(1)
    tp_replicated = torch.nn.Parameter(torch.ones(1))
    tp_replicated.sequence_parallel = True
    tp_replicated.grad = torch.ones(1)
    expert = torch.nn.Parameter(torch.ones(1))
    expert.grad = torch.ones(1)
    ps = SimpleNamespace(
        dp_cp_group="dense_dp",
        dp_group=None,
        ep_dp_group="expert_dp",
        ep_group="expert",
        etp_group="expert_tp",
        tp_group="tensor",
        pp_group="pipeline",
    )
    scalar_reductions = []
    grad_reductions = []

    def record_scalar_reduction(_value, group):
        scalar_reductions.append(group)

    def record_grad_reduction(grad, group, *, average=False):
        grad_reductions.append(group)
        grad.mul_(2.0)

    monkeypatch.setattr(mfsdp_optimizer, "_sum_if_distributed", record_scalar_reduction)
    monkeypatch.setattr(
        mfsdp_optimizer, "_all_reduce_grad_if_distributed", record_grad_reduction
    )
    adapter = mfsdp_optimizer._StandaloneOptimizer(
        torch.optim.SGD([dense, tp_replicated, expert], lr=0.0),
        [dense, tp_replicated, expert],
        {id(dense): "dense", id(tp_replicated): "sp", id(expert): "expert"},
        ps=ps,
        clip_grad=0.0,
        grad_norm_accum_dtype=torch.float32,
        expert_params=[expert],
        expert_grad_scale=1.0,
    )

    success, grad_norm, _ = adapter.step()

    assert success
    assert grad_norm == pytest.approx((1.0 + 4.0 + 1.0) ** 0.5)
    assert grad_reductions == ["tensor"]
    assert scalar_reductions == [
        "dense_dp",
        "tensor",
        "dense_dp",
        "expert_dp",
        "expert_tp",
        "expert",
        "pipeline",
    ]


def test_mfsdp_cpu_single_rank_matches_torch_adamw_optimizer_step():
    class _Unit(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(4, 4, bias=False)

        def forward(self, value):
            return torch.nn.functional.gelu(self.linear(value))

    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.unit0 = _Unit()
            self.unit1 = _Unit()
            self.out = torch.nn.Linear(4, 2, bias=False)

        def forward(self, value):
            return self.out(self.unit1(self.unit0(value)))

    torch.manual_seed(123)
    reference = _Model()
    candidate = copy.deepcopy(reference)
    ps = SimpleNamespace(
        dp_cp_group=None,
        dp_group=None,
        ep_dp_group=None,
        ep_group=None,
        tp_group=None,
        pp_group=None,
        dp_cp_size=1,
        expert_dp_size=1,
    )
    opt = SimpleNamespace(
        optimizer="adam",
        lr=1.0e-3,
        min_lr=0.0,
        weight_decay=0.0,
        clip_grad=1000.0,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_eps=1.0e-8,
        override_optimizer_config={"mfsdp_sharding_strategy": "optim_grads_params"},
    )
    engine_cfg = SimpleNamespace(
        model_name="qwen3_5",
        parallel=ParallelConfig(tp=1, ep=1, etp=1, pp=1, vpp=1, cp=1),
        optimizer=opt,
    )

    reference_optimizer = torch.optim.AdamW(
        reference.parameters(),
        lr=opt.lr,
        betas=(opt.adam_beta1, opt.adam_beta2),
        eps=opt.adam_eps,
        weight_decay=opt.weight_decay,
        foreach=False,
    )
    chunks, candidate_optimizer = mfsdp_optimizer.build_mfsdp_stack(
        [candidate],
        model_cfg=SimpleNamespace(),
        engine_cfg=engine_cfg,
        ps=ps,
        is_expert=lambda _name: False,
        fsdp_unit_modules=(_Unit,),
    )

    value = torch.randn(3, 4)
    target = torch.randn(3, 2)
    reference_optimizer.zero_grad()
    torch.nn.functional.mse_loss(reference(value), target).backward()

    candidate_optimizer.zero_grad()
    torch.nn.functional.mse_loss(chunks[0](value), target).backward()
    mfsdp_optimizer.finalize_mfsdp_grads(chunks, candidate_optimizer)

    reference_grads = {
        name: param.grad.detach().reshape(-1)
        for name, param in reference.named_parameters()
    }
    candidate_grads = {
        candidate_optimizer.param_names[id(param)].removeprefix(
            "chunk0.module."
        ): param.grad.detach().reshape(-1)
        for param in candidate_optimizer.params
    }
    assert reference_grads.keys() == candidate_grads.keys()
    for name, reference_grad in reference_grads.items():
        assert torch.equal(reference_grad, candidate_grads[name]), name

    reference_optimizer.step()
    success, _grad_norm, _num_zeros = candidate_optimizer.step()
    assert success

    reference_params = dict(reference.named_parameters())
    candidate_params = {
        name.removeprefix("module."): param
        for name, param in chunks[0].named_parameters()
    }
    assert reference_params.keys() == candidate_params.keys()
    for name, reference_param in reference_params.items():
        assert torch.equal(reference_param, candidate_params[name]), name

    saved_model = {
        name: value.clone() for name, value in chunks[0].state_dict().items()
    }
    with torch.no_grad():
        for param in chunks[0].parameters():
            param.add_(7.0)
    chunks[0].load_state_dict(saved_model)
    assert candidate_optimizer.sync_model_weights_to_main_weights()
    restored_model = chunks[0].state_dict()
    assert saved_model.keys() == restored_model.keys()
    for name, value in saved_model.items():
        assert torch.equal(value, restored_model[name]), name


def test_mfsdp_accumulates_all_microbatches_before_grad_reduce():
    torch.manual_seed(321)
    reference = _GlooModel()
    candidate = copy.deepcopy(reference)
    ps = SimpleNamespace(
        dp_cp_group=None,
        dp_group=None,
        ep_dp_group=None,
        ep_group=None,
        etp_group=None,
        tp_group=None,
        pp_group=None,
        dp_cp_size=1,
        expert_dp_size=1,
    )
    opt = SimpleNamespace(
        optimizer="adam",
        lr=1.0e-3,
        min_lr=0.0,
        weight_decay=0.0,
        clip_grad=1000.0,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_eps=1.0e-8,
        override_optimizer_config={"mfsdp_sharding_strategy": "optim_grads_params"},
    )
    reference_optimizer = torch.optim.AdamW(
        reference.parameters(),
        lr=opt.lr,
        betas=(opt.adam_beta1, opt.adam_beta2),
        eps=opt.adam_eps,
        weight_decay=opt.weight_decay,
        foreach=False,
    )
    chunks, candidate_optimizer = mfsdp_optimizer.build_mfsdp_stack(
        [candidate],
        model_cfg=SimpleNamespace(),
        engine_cfg=SimpleNamespace(
            model_name="qwen3_5",
            parallel=ParallelConfig(tp=1, ep=1, etp=1, pp=1, vpp=1, cp=1),
            optimizer=opt,
        ),
        ps=ps,
        is_expert=lambda _name: False,
        fsdp_unit_modules=(_GlooUnit,),
    )
    microbatches = [
        (torch.randn(3, 4), torch.randn(3, 2)),
        (torch.randn(2, 4), torch.randn(2, 2)),
    ]

    reference_optimizer.zero_grad()
    candidate_optimizer.zero_grad()
    for microbatch_idx, (value, target) in enumerate(microbatches):
        reference_loss = torch.nn.functional.mse_loss(reference(value), target)
        (reference_loss / len(microbatches)).backward()

        candidate_loss = torch.nn.functional.mse_loss(chunks[0](value), target)
        if microbatch_idx == len(microbatches) - 1:
            candidate_optimizer.grad_sync_enabled = True
        (candidate_loss / len(microbatches)).backward()

    mfsdp_optimizer.finalize_mfsdp_grads(chunks, candidate_optimizer)

    reference_grads = {
        name: param.grad.detach().reshape(-1)
        for name, param in reference.named_parameters()
    }
    candidate_grads = {
        candidate_optimizer.param_names[id(param)].removeprefix("chunk0.module."): (
            param.grad.detach().reshape(-1)
        )
        for param in candidate_optimizer.params
    }
    assert reference_grads.keys() == candidate_grads.keys()
    for name, reference_grad in reference_grads.items():
        assert torch.equal(reference_grad, candidate_grads[name]), name


def test_mfsdp_materializes_root_params_used_without_calling_their_leaf_module():
    torch.manual_seed(654)
    reference = _DirectParamModel()
    candidate = copy.deepcopy(reference)
    ps = SimpleNamespace(
        dp_cp_group=None,
        dp_group=None,
        ep_dp_group=None,
        ep_group=None,
        etp_group=None,
        tp_group=None,
        pp_group=None,
        dp_cp_size=1,
        expert_dp_size=1,
    )
    opt = SimpleNamespace(
        optimizer="adam",
        lr=1.0e-3,
        min_lr=0.0,
        weight_decay=0.0,
        clip_grad=1000.0,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_eps=1.0e-8,
        override_optimizer_config={"mfsdp_sharding_strategy": "optim_grads_params"},
    )
    chunks, optimizer = mfsdp_optimizer.build_mfsdp_stack(
        [candidate],
        model_cfg=SimpleNamespace(),
        engine_cfg=SimpleNamespace(
            model_name="qwen3_5",
            parallel=ParallelConfig(tp=1, ep=1, etp=1, pp=1, vpp=1, cp=1),
            optimizer=opt,
        ),
        ps=ps,
        is_expert=lambda _name: False,
        fsdp_unit_modules=(_GlooUnit,),
    )

    value = torch.randn(2, 4)
    reference_output = reference(value)
    candidate_output = chunks[0](value)
    assert torch.equal(reference_output, candidate_output)

    reference_output.square().mean().backward()
    candidate_output.square().mean().backward()
    mfsdp_optimizer.finalize_mfsdp_grads(chunks, optimizer)
    projection_shard = next(
        param
        for param in optimizer.params
        if optimizer.param_names[id(param)].endswith("projection.weight")
    )
    assert torch.equal(
        reference.projection.weight.grad.reshape(-1), projection_shard.grad.reshape(-1)
    )


def test_mfsdp_keeps_fp32_shards_for_bfloat16_compute_parameters():
    model = _GlooModel().to(torch.bfloat16)
    ps = SimpleNamespace(
        dp_cp_group=None,
        dp_group=None,
        ep_dp_group=None,
        ep_group=None,
        tp_group=None,
        pp_group=None,
        dp_cp_size=1,
        expert_dp_size=1,
    )
    opt = SimpleNamespace(
        optimizer="adam",
        lr=1.0e-3,
        min_lr=0.0,
        weight_decay=0.0,
        clip_grad=1.0,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_eps=1.0e-8,
        override_optimizer_config={"mfsdp_sharding_strategy": "optim_grads_params"},
    )
    chunks, optimizer = mfsdp_optimizer.build_mfsdp_stack(
        [model],
        model_cfg=SimpleNamespace(),
        engine_cfg=SimpleNamespace(
            model_name="qwen3_5",
            parallel=ParallelConfig(tp=1, ep=1, etp=1, pp=1, vpp=1, cp=1),
            optimizer=opt,
        ),
        ps=ps,
        is_expert=lambda _name: False,
        fsdp_unit_modules=(_GlooUnit,),
    )

    assert all(param.dtype is torch.float32 for param in optimizer.params)
    assert all(
        param.dtype is torch.bfloat16 for _name, param in chunks[0].named_parameters()
    )
    assert all(
        bucket.grad_shard_buffer.dtype is torch.float32
        for bucket in chunks[0].param_sync.buckets
    )

    before = [param.detach().clone() for param in optimizer.params]
    value = torch.randn(3, 4, dtype=torch.bfloat16)
    optimizer.zero_grad()
    chunks[0](value).float().square().mean().backward()
    mfsdp_optimizer.finalize_mfsdp_grads(chunks, optimizer)
    success, grad_norm, _ = optimizer.step()

    assert success
    assert grad_norm > 0.0
    assert all(param.dtype is torch.float32 for param in optimizer.params)
    assert any(not torch.equal(old, new) for old, new in zip(before, optimizer.params))
    assert torch.isfinite(chunks[0](value).float()).all()


def _run_mfsdp_gloo_parity(rank: int, world_size: int, init_file: str) -> None:
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        torch.manual_seed(456)
        reference = _GlooModel()
        candidate = copy.deepcopy(reference)
        ps = SimpleNamespace(
            dp_cp_group=dist.group.WORLD,
            dp_group=dist.group.WORLD,
            ep_dp_group=dist.group.WORLD,
            ep_group=None,
            tp_group=None,
            pp_group=None,
            dp_cp_size=world_size,
            expert_dp_size=world_size,
        )
        opt = SimpleNamespace(
            optimizer="adam",
            lr=1.0e-3,
            min_lr=0.0,
            weight_decay=0.0,
            clip_grad=1000.0,
            adam_beta1=0.9,
            adam_beta2=0.999,
            adam_eps=1.0e-8,
            override_optimizer_config={"mfsdp_sharding_strategy": "optim_grads_params"},
        )
        engine_cfg = SimpleNamespace(
            model_name="qwen3_5",
            parallel=ParallelConfig(tp=1, ep=1, etp=1, pp=1, vpp=1, cp=1),
            optimizer=opt,
        )

        reference_optimizer = torch.optim.AdamW(
            reference.parameters(),
            lr=opt.lr,
            betas=(opt.adam_beta1, opt.adam_beta2),
            eps=opt.adam_eps,
            weight_decay=opt.weight_decay,
            foreach=False,
        )
        chunks, candidate_optimizer = mfsdp_optimizer.build_mfsdp_stack(
            [candidate],
            model_cfg=SimpleNamespace(),
            engine_cfg=engine_cfg,
            ps=ps,
            is_expert=lambda _name: False,
            fsdp_unit_modules=(_GlooUnit,),
        )

        torch.manual_seed(900 + rank)
        value = torch.randn(3, 4)
        target = torch.randn(3, 2)
        reference_optimizer.zero_grad()
        torch.nn.functional.mse_loss(reference(value), target).backward()
        candidate_optimizer.zero_grad()
        torch.nn.functional.mse_loss(chunks[0](value), target).backward()
        for param in reference.parameters():
            assert param.grad is not None
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
            param.grad.div_(world_size)
        mfsdp_optimizer.finalize_mfsdp_grads(chunks, candidate_optimizer)

        reference_norm = torch.linalg.vector_norm(
            torch.cat([param.grad.reshape(-1) for param in reference.parameters()])
        ).item()
        reference_optimizer.step()
        candidate_success, candidate_norm, _ = candidate_optimizer.step()
        assert candidate_success
        assert reference_norm == candidate_norm

        reference_params = dict(reference.named_parameters())
        candidate_params = {
            name.removeprefix("module."): param
            for name, param in chunks[0].named_parameters()
        }
        assert reference_params.keys() == candidate_params.keys()
        for name, reference_param in reference_params.items():
            assert torch.equal(reference_param, candidate_params[name]), name
    finally:
        dist.destroy_process_group()


def test_mfsdp_cpu_two_rank_gloo_matches_replicated_reference(tmp_path):
    init_file = str(tmp_path / "mfsdp-gloo-init")
    mp.spawn(_run_mfsdp_gloo_parity, args=(2, init_file), nprocs=2, join=True)
