# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from __future__ import annotations

import ast
import copy
import inspect
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from megatron.lite.primitive.optimizers.mfsdp import config as mfsdp_config
from megatron.lite.primitive.optimizers.mfsdp import buffer as mfsdp_buffer
from megatron.lite.primitive.optimizers.mfsdp import optimizer as mfsdp_optimizer
from megatron.lite.primitive.optimizers import get_optimizer_backend
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


def _optimizer_params(optimizer) -> list[torch.nn.Parameter]:
    return optimizer._inner_optimizer.params


def _optimizer_named_shards(optimizer) -> dict[str, torch.nn.Parameter]:
    result = {}
    for chunk in optimizer._model_chunks:
        for bucket in chunk.param_sync.buckets:
            for spec in bucket.specs:
                assert spec.shard_param is not None
                result[spec.name] = spec.shard_param
    return result


def test_mfsdp_zero_grad_supports_fused_optimizer_contract():
    class NoArgZeroGradOptimizer:
        def __init__(self):
            self.called = False

        def zero_grad(self):
            self.called = True

    fused_optimizer = NoArgZeroGradOptimizer()
    adapter = mfsdp_optimizer._StandaloneOptimizer(
        fused_optimizer,
        [],
        ps=SimpleNamespace(),
        clip_grad=0.0,
        grad_norm_accum_dtype=torch.float32,
        expert_params=[],
        expert_grad_scale=1.0,
    )

    adapter.zero_grad()

    assert fused_optimizer.called is True


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


def test_mfsdp_standalone_smoke_has_no_megatron_core_imports():
    smoke_path = (
        Path(__file__).parents[2] / "smoke" / "primitive" / "test_mfsdp_parity_smoke.py"
    )
    tree = ast.parse(smoke_path.read_text(), filename=str(smoke_path))
    violations = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            modules = [node.module or ""]
        else:
            continue
        for module in modules:
            if module == "megatron.core" or module.startswith("megatron.core."):
                violations.append(f"{smoke_path.name}:{node.lineno}: {module}")

    assert violations == []


def test_mfsdp_full_parallel_signoff_is_single_node_50_step_curve():
    tests_root = Path(__file__).parents[2]
    smoke_path = tests_root / "smoke" / "primitive" / "test_mfsdp_parity_smoke.py"
    smoke_source = smoke_path.read_text()
    smoke_tree = ast.parse(smoke_source, filename=str(smoke_path))
    constants = {}
    for node in smoke_tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name) and isinstance(node.value, ast.Constant):
            constants[target.id] = node.value.value

    assert constants["_FULL_PARALLEL_WORLD_SIZE"] == 8
    assert constants["_FULL_PARALLEL_STEPS"] == 50
    assert constants["_FULL_PARALLEL_CURVE_INTERVAL"] == 10
    assert "[MFSDP_FULL_PARALLEL_CURVE]" in smoke_source
    assert "[MFSDP_FULL_PARALLEL_STEP]" in smoke_source
    assert "[MFSDP_FULL_PARALLEL_WORST]" in smoke_source
    assert "_tensor_set_max_differences" in smoke_source
    assert "_named_model_tensors(handle._model)" in smoke_source
    assert "_named_model_tensors(handle.model)" not in smoke_source

    runner_source = (tests_root / "run_mfsdp_hopper_validation.sh").read_text()
    assert "full-parallel mode requires NNODES=1 and NPROC_PER_NODE=8." in runner_source


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


def test_mfsdp_source_layout_is_bounded():
    package = Path(mfsdp_config.__file__).parent
    modules = {path.name for path in package.glob("*.py")}

    assert modules == {
        "__init__.py",
        "backend.py",
        "buffer.py",
        "config.py",
        "fully_shard.py",
        "fused_ops.py",
        "grad_norm.py",
        "optimizer.py",
        "wrapper.py",
    }
    assert len(modules) <= 9
    assert not (package / "impl").exists()


def test_mfsdp_primitive_has_no_model_specific_knowledge():
    package = Path(mfsdp_config.__file__).parent
    violations = []
    for source_path in package.rglob("*.py"):
        source = source_path.read_text()
        for forbidden in ("_SUPPORTED_MODELS", "model_name", "Qwen3", "qwen3"):
            if forbidden in source:
                relative_path = source_path.relative_to(package)
                violations.append(f"{relative_path}: {forbidden}")

    assert violations == []
    assert (
        "model_name"
        not in inspect.signature(
            mfsdp_optimizer.build_mfsdp_training_optimizer
        ).parameters
    )


def test_mfsdp_config_validation_does_not_require_or_filter_model_name():
    engine_cfg = _engine_cfg()
    mfsdp_config.validate_mfsdp_config(engine_cfg)

    engine_cfg.model_name = "arbitrary_2d_transformer"
    mfsdp_config.validate_mfsdp_config(engine_cfg)


def test_mfsdp_reference_rewrite_modules_are_live():
    package = Path(mfsdp_config.__file__).parent
    required_modules = {
        "backend.py",
        "buffer.py",
        "config.py",
        "fully_shard.py",
        "fused_ops.py",
        "grad_norm.py",
        "optimizer.py",
        "wrapper.py",
    }
    missing = sorted(
        name for name in required_modules if not (package / name).is_file()
    )
    assert missing == []

    # These are production edges, not files kept alive only by tests.  The
    # optimizer constructs the wrapper, and the wrapper owns the M-FSDP buffer
    # plus both communication pipelines.
    optimizer_source = (package / "optimizer.py").read_text()
    wrapper_source = (package / "wrapper.py").read_text()
    assert "mfsdp.wrapper" in optimizer_source
    assert "ParamAndGradBuffer" in wrapper_source
    assert "AllGatherPipeline" in wrapper_source
    assert "GradReducePipeline" in wrapper_source


def test_mfsdp_has_no_legacy_test_only_production_surface():
    package = Path(mfsdp_config.__file__).parent
    forbidden_top_level = {
        "config.py": {
            "_collect_mfsdp_overrides",
            "_distributed_optimizer_instance_override",
            "_validate_ddp_knob_value",
            "coerce_dtype",
            "split_mfsdp_overrides",
            "validate_mfsdp_topology_optimizer_combo",
        },
        "grad_norm.py": {
            "CanonicalGradNormMegatronFSDPOptimizer",
            "GradNormBreakdown",
            "_clip_mfsdp_grads_by_total_norm",
            "_leaf_optimizers",
            "_sum_if_distributed",
            "_unique_parameters",
            "compute_mfsdp_grad_norm",
        },
        "optimizer.py": {
            "_default_expert_classifier",
            "finalize_mfsdp_grads",
        },
    }
    forbidden_methods = {
        "optimizer.py": {"sync_model_weights_to_main_weights"},
    }

    violations = []
    for module_name, names in forbidden_top_level.items():
        tree = ast.parse((package / module_name).read_text())
        defined = {
            node.name
            for node in tree.body
            if isinstance(node, ast.FunctionDef | ast.ClassDef)
        }
        violations.extend(
            f"test-only definition: {module_name}:{name}"
            for name in sorted(defined & names)
        )
    for module_name, names in forbidden_methods.items():
        tree = ast.parse((package / module_name).read_text())
        defined = {
            child.name
            for node in tree.body
            if isinstance(node, ast.ClassDef)
            for child in node.body
            if isinstance(child, ast.FunctionDef)
        }
        violations.extend(
            f"test-only method: {module_name}:{name}"
            for name in sorted(defined & names)
        )

    stack_params = inspect.signature(mfsdp_optimizer.build_mfsdp_stack).parameters
    training_params = inspect.signature(
        mfsdp_optimizer.build_mfsdp_training_optimizer
    ).parameters
    for name in ("model_cfg", "proto", "skip_fsdp_wrap"):
        if name in stack_params:
            violations.append(f"unused build_mfsdp_stack argument: {name}")
    for name in ("model_cfg", "deterministic"):
        if name in training_params:
            violations.append(f"unused build_mfsdp_training_optimizer argument: {name}")

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


def _engine_cfg(**overrides):
    cfg = SimpleNamespace(
        parallel=ParallelConfig(tp=1, ep=1, etp=1, pp=1, vpp=1, cp=1),
        optimizer=SimpleNamespace(optimizer="adam", override_optimizer_config={}),
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def test_mfsdp_config_rejects_unsupported_optimizer():
    engine_cfg = _engine_cfg(
        optimizer=SimpleNamespace(optimizer="adamw", override_optimizer_config={})
    )
    with pytest.raises(ValueError, match="adam/sgd"):
        mfsdp_config.validate_mfsdp_config(engine_cfg)


def test_mfsdp_accepts_an_optional_optimizer_factory():
    model = _GlooModel()
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
        optimizer="muon",
        lr=1.0e-3,
        weight_decay=0.0,
        clip_grad=1.0,
        override_optimizer_config={"mfsdp_sharding_strategy": "optim_grads_params"},
    )
    calls = []

    def optimizer_factory(param_groups, optimizer_config):
        calls.append((param_groups, optimizer_config))
        return torch.optim.SGD(param_groups, lr=optimizer_config.lr)

    chunks = [model]
    optimizer, finalize_grads = mfsdp_optimizer.build_mfsdp_training_optimizer(
        chunks,
        impl_cfg=SimpleNamespace(
            parallel=ParallelConfig(tp=1, ep=1, etp=1, pp=1, vpp=1, cp=1),
            optimizer_config=opt,
        ),
        ps=ps,
        is_expert=lambda _name: False,
        fsdp_unit_modules=(_GlooUnit,),
        optimizer_factory=optimizer_factory,
    )

    assert len(calls) == 1
    assert calls[0][1] is opt
    assert isinstance(optimizer._inner_optimizer.optimizer, torch.optim.SGD)
    finalize_grads()


def test_mfsdp_config_enables_double_buffer_for_nccl_user_buffers():
    config = mfsdp_config.build_mfsdp_config(
        SimpleNamespace(
            override_optimizer_config={
                "mfsdp_sharding_strategy": "optim_grads_params",
                "nccl_ub": True,
            }
        )
    )

    assert config.nccl_ub is True
    assert config.fsdp_double_buffer is True


def test_mfsdp_nccl_user_buffer_falls_back_when_apex_is_missing(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    def missing_apex(_name):
        raise ImportError("optional allocator missing")

    monkeypatch.setattr(mfsdp_buffer.importlib, "import_module", missing_apex)
    user_buffer = mfsdp_buffer.NCCLUserBuffer(
        enabled=True,
        groups=(),
        symmetric=True,
    )

    assert user_buffer.active is False


def test_mfsdp_backend_is_registered():
    backend = get_optimizer_backend("mfsdp")
    assert backend.name == "mfsdp"


def test_mfsdp_parallel_metadata_uses_topology_and_explicit_classifier():
    model = torch.nn.Module()
    model.dense_matrix = torch.nn.Parameter(torch.ones(4, 4))
    model.routed_matrix = torch.nn.Parameter(torch.ones(4, 4))
    model.replicated_matrix = torch.nn.Parameter(torch.ones(4, 4))
    model.replicated_matrix.average_gradients_across_tp_domain = True

    mfsdp_optimizer._mark_mfsdp_parallel_attrs(
        model,
        lambda name: name == "routed_matrix",
        tp_size=2,
        etp_size=1,
    )

    assert model.dense_matrix.tensor_model_parallel is True
    assert model.routed_matrix.tensor_model_parallel is False
    assert model.routed_matrix.allreduce is False
    assert model.replicated_matrix.tensor_model_parallel is False


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
        engine_cfg=SimpleNamespace(
            parallel=ParallelConfig(tp=2, ep=1, etp=1, pp=1, vpp=1, cp=1),
            optimizer=opt,
        ),
        ps=ps,
        is_expert=lambda _name: False,
        fsdp_unit_modules=(_GlooUnit,),
    )

    out_shard = _optimizer_named_shards(optimizer)["out.weight"]
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
    candidate_optimizer.finish_grad_sync()

    reference_grads = {
        name: param.grad.detach().reshape(-1)
        for name, param in reference.named_parameters()
    }
    candidate_grads = {
        name: param.grad.detach().reshape(-1)
        for name, param in _optimizer_named_shards(candidate_optimizer).items()
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
    restored_model = chunks[0].state_dict()
    assert saved_model.keys() == restored_model.keys()
    for name, value in saved_model.items():
        assert torch.equal(value, restored_model[name]), name


def test_mfsdp_bucket_policy_splits_one_unit_without_splitting_parameters():
    model = torch.nn.Module()
    model.weight0 = torch.nn.Parameter(torch.ones(4))
    model.weight1 = torch.nn.Parameter(torch.ones(4))
    model.weight2 = torch.nn.Parameter(torch.ones(4))
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
        optimizer="sgd",
        lr=0.0,
        weight_decay=0.0,
        clip_grad=0.0,
        override_optimizer_config={
            "mfsdp_sharding_strategy": "optim_grads_params",
            "bucket_size": 4,
        },
    )

    chunks, _optimizer = mfsdp_optimizer.build_mfsdp_stack(
        [model],
        engine_cfg=SimpleNamespace(
            parallel=ParallelConfig(tp=1, ep=1, etp=1, pp=1, vpp=1, cp=1),
            optimizer=opt,
        ),
        ps=ps,
        is_expert=lambda _name: False,
        fsdp_unit_modules=(),
    )

    buckets = chunks[0].param_sync.buckets
    assert len(buckets) == 3
    assert [[spec.name for spec in bucket.specs] for bucket in buckets] == [
        ["weight0"],
        ["weight1"],
        ["weight2"],
    ]


def test_mfsdp_empty_uneven_shard_stays_inside_local_buffer(monkeypatch):
    monkeypatch.setattr(mfsdp_buffer, "group_size", lambda _group: 4)
    monkeypatch.setattr(mfsdp_buffer, "group_rank", lambda _group: 0)
    model = torch.nn.Module()
    model.weight0 = torch.nn.Parameter(torch.arange(4.0))
    model.weight1 = torch.nn.Parameter(torch.arange(4.0))
    groups = mfsdp_buffer.MFSDPProcessGroups(
        dense_dp=None,
        expert_dp=None,
        dense_ag=None,
        expert_ag=None,
        tp=None,
        etp=None,
        ep=None,
        pp=None,
    )
    config = mfsdp_config.MFSDPConfig(bucket_size=None)

    buffers = mfsdp_buffer.ParamAndGradBuffer(
        model,
        groups=groups,
        config=config,
        is_expert=lambda _name: False,
        unit_modules=(),
    )

    bucket = buffers.buckets[0]
    assert bucket.local_numel == 2
    assert bucket.specs[1].shard_numel == 0
    assert bucket.specs[1].local_offset == bucket.local_numel
    assert bucket.specs[1].shard_param is not None
    assert bucket.specs[1].shard_param.numel() == 0


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
        engine_cfg=SimpleNamespace(
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

    candidate_optimizer.finish_grad_sync()

    reference_grads = {
        name: param.grad.detach().reshape(-1)
        for name, param in reference.named_parameters()
    }
    candidate_grads = {
        name: param.grad.detach().reshape(-1)
        for name, param in _optimizer_named_shards(candidate_optimizer).items()
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
        engine_cfg=SimpleNamespace(
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
    optimizer.finish_grad_sync()
    projection_shard = _optimizer_named_shards(optimizer)["projection.weight"]
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
        engine_cfg=SimpleNamespace(
            parallel=ParallelConfig(tp=1, ep=1, etp=1, pp=1, vpp=1, cp=1),
            optimizer=opt,
        ),
        ps=ps,
        is_expert=lambda _name: False,
        fsdp_unit_modules=(_GlooUnit,),
    )

    optimizer_params = _optimizer_params(optimizer)
    assert all(param.dtype is torch.float32 for param in optimizer_params)
    assert all(
        param.dtype is torch.bfloat16 for _name, param in chunks[0].named_parameters()
    )
    assert all(
        bucket.grad_shard_buffer.dtype is torch.float32
        for bucket in chunks[0].param_sync.buckets
    )
    for bucket in chunks[0].param_sync.buckets:
        main_storage = bucket.main_param_buffer.untyped_storage().data_ptr()
        assert all(
            spec.shard_param is not None
            and spec.shard_param.untyped_storage().data_ptr() == main_storage
            for spec in bucket.specs
        )

    before = [param.detach().clone() for param in optimizer_params]
    value = torch.randn(3, 4, dtype=torch.bfloat16)
    optimizer.zero_grad()
    chunks[0](value).float().square().mean().backward()
    optimizer.finish_grad_sync()
    for bucket in chunks[0].param_sync.buckets:
        grad_storage = bucket.main_grad_buffer.untyped_storage().data_ptr()
        assert all(
            spec.shard_param is not None
            and spec.shard_param.grad is not None
            and spec.shard_param.grad.untyped_storage().data_ptr() == grad_storage
            for spec in bucket.specs
        )
    success, grad_norm, _ = optimizer.step()

    assert success
    assert grad_norm > 0.0
    assert all(param.dtype is torch.float32 for param in optimizer_params)
    assert any(not torch.equal(old, new) for old, new in zip(before, optimizer_params))
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
        candidate_optimizer.finish_grad_sync()

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
