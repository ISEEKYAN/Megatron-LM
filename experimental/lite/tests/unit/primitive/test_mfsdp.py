# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from __future__ import annotations

import ast
import copy
import inspect
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from megatron.lite.primitive.optimizers.mfsdp import buffer as mfsdp_buffer
from megatron.lite.primitive.optimizers.mfsdp import config as mfsdp_config
from megatron.lite.primitive.optimizers.mfsdp import cpu_offload as mfsdp_cpu_offload
from megatron.lite.primitive.optimizers.mfsdp import optimizer as mfsdp_optimizer
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


class _FusedMainGradLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, weight, fuse_wgrad_accumulation):
        ctx.save_for_backward(value, weight)
        ctx.weight = weight
        ctx.fuse_wgrad_accumulation = fuse_wgrad_accumulation
        return torch.nn.functional.linear(value, weight)

    @staticmethod
    def backward(ctx, grad_output):
        value, weight = ctx.saved_tensors
        grad_input = grad_output.matmul(weight)
        grad_weight = (
            grad_output.float()
            .reshape(-1, grad_output.shape[-1])
            .t()
            .matmul(value.float().reshape(-1, value.shape[-1]))
        )
        if ctx.fuse_wgrad_accumulation:
            ctx.weight.main_grad.add_(grad_weight)
            ctx.weight.grad_added_to_main_grad = True
            grad_weight = torch.zeros_like(weight)
        return grad_input, grad_weight, None


class _FusedMainGradLinear(torch.nn.Module):
    """CPU stand-in for TE's dynamic ``weight.main_grad`` backward contract."""

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(4, 4, dtype=torch.bfloat16))
        self.fuse_wgrad_accumulation = False

    def forward(self, value):
        return _FusedMainGradLinearFunction.apply(
            value, self.weight, self.fuse_wgrad_accumulation
        )


class _RegularMainGradLinear(torch.nn.Module):
    """Non-TE linear used to exercise the autograd gradient fallback."""

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(4, 4, dtype=torch.bfloat16))

    def forward(self, value):
        return torch.nn.functional.linear(value, self.weight)


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
    assert "batch_seed=8345 + step" in smoke_source

    runner_source = (tests_root / "run_mfsdp_hopper_validation.sh").read_text()
    assert "full-parallel mode requires NNODES=1 and NPROC_PER_NODE=8." in runner_source


def test_mfsdp_is_standalone_without_vendored_mcore_or_fsdp2_dependencies():
    package = Path(mfsdp_config.__file__).parent

    assert not list(
        (package / "impl").glob("*.py")
    ), "Do not vendor the MCore FSDP implementation."
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
        "cpu_offload.py",
        "fully_shard.py",
        "fused_ops.py",
        "grad_norm.py",
        "optimizer.py",
        "wrapper.py",
    }
    assert len(modules) <= 10
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
        "buffer.py",
        "config.py",
        "cpu_offload.py",
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
        "optimizer.py": {"_default_expert_classifier", "finalize_mfsdp_grads"},
    }
    forbidden_methods = {"optimizer.py": {"sync_model_weights_to_main_weights"}}

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
    user_buffer = mfsdp_buffer.NCCLUserBuffer(enabled=True, groups=(), symmetric=True)

    assert user_buffer.active is False


def test_mfsdp_parallel_metadata_uses_topology_and_explicit_classifier():
    model = torch.nn.Module()
    model.dense_matrix = torch.nn.Parameter(torch.ones(4, 4))
    model.routed_matrix = torch.nn.Parameter(torch.ones(4, 4))
    model.replicated_matrix = torch.nn.Parameter(torch.ones(4, 4))
    model.replicated_matrix.average_gradients_across_tp_domain = True

    mfsdp_optimizer._mark_mfsdp_parallel_attrs(
        model, lambda name: name == "routed_matrix", tp_size=2, etp_size=1
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
            parallel=ParallelConfig(tp=2, ep=1, etp=1, pp=1, vpp=1, cp=1), optimizer=opt
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
    monkeypatch.setattr(dist, "all_reduce", lambda value, *, op, group: value.mul_(2.0))

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
        parallel=ParallelConfig(tp=1, ep=1, etp=1, pp=1, vpp=1, cp=1), optimizer=opt
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


def _build_offload_stack(offload_fraction: float):
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
        offload_fraction=offload_fraction,
        override_optimizer_config={"mfsdp_sharding_strategy": "optim_grads_params"},
    )
    engine_cfg = SimpleNamespace(
        parallel=ParallelConfig(tp=1, ep=1, etp=1, pp=1, vpp=1, cp=1), optimizer=opt
    )
    return _Model, _Unit, ps, engine_cfg


def test_mfsdp_offload_fraction_keeps_optimizer_state_on_cpu():
    _Model, _Unit, ps, engine_cfg = _build_offload_stack(offload_fraction=1.0)

    torch.manual_seed(42)
    chunks, optimizer = mfsdp_optimizer.build_mfsdp_stack(
        [_Model()],
        engine_cfg=engine_cfg,
        ps=ps,
        is_expert=lambda _name: False,
        fsdp_unit_modules=(_Unit,),
    )

    value = torch.randn(3, 4)
    target = torch.randn(3, 2)
    optimizer.zero_grad()
    torch.nn.functional.mse_loss(chunks[0](value), target).backward()
    optimizer.finish_grad_sync()
    success, _grad_norm, _ = optimizer.step()
    assert success

    inner = optimizer._inner_optimizer
    cpu_group = inner.cpu_group
    assert (
        cpu_group is not None
    ), "Expected cpu_group to be set for offload_fraction=1.0"
    assert len(cpu_group._cpu_optimizer.optimizers) == len(cpu_group._cpu_params)
    for cpu_p in cpu_group._cpu_params:
        assert cpu_p.device.type == "cpu", "cpu_param should be on CPU"
    cpu_opt_state = cpu_group._cpu_optimizer.state
    for cpu_p in cpu_group._cpu_params:
        if cpu_p in cpu_opt_state:
            state = cpu_opt_state[cpu_p]
            assert state["exp_avg"].device.type == "cpu"
            assert state["exp_avg_sq"].device.type == "cpu"

    for gpu_param in inner.params:
        assert gpu_param.device.type != "cuda" or gpu_param.data is not None


def test_mfsdp_cpu_offload_overlaps_per_param_d2h_cpu_step_and_h2d():
    source = inspect.getsource(mfsdp_cpu_offload.CpuAdamGroup.step)

    assert source.count("non_blocking=True") >= 2
    assert "self._d2h_stream.record_event()" in source
    assert "d2h_event.synchronize()" in source
    assert "with torch.cuda.stream(self._h2d_stream)" in source
    assert "self._h2d_stream.record_event().wait(current_stream)" in source


def test_mfsdp_full_offload_has_six_gpu_and_twelve_cpu_bytes_per_param():
    _Model, _Unit, ps, engine_cfg = _build_offload_stack(offload_fraction=1.0)

    torch.manual_seed(43)
    model = _Model().to(dtype=torch.bfloat16)
    chunks, optimizer = mfsdp_optimizer.build_mfsdp_stack(
        [model],
        engine_cfg=engine_cfg,
        ps=ps,
        is_expert=lambda _name: False,
        fsdp_unit_modules=(_Unit,),
    )

    value = torch.randn(3, 4, dtype=torch.bfloat16)
    target = torch.randn(3, 2, dtype=torch.bfloat16)
    optimizer.zero_grad()
    torch.nn.functional.mse_loss(chunks[0](value), target).backward()
    optimizer.finish_grad_sync()
    success, _grad_norm, _ = optimizer.step()
    assert success

    inner = optimizer._inner_optimizer
    cpu_group = inner.cpu_group
    assert cpu_group is not None
    total_numel = sum(param.numel() for param in inner.params)
    buckets = [bucket for chunk in chunks for bucket in chunk.param_sync.buckets]
    assert all(
        bucket.local_compute_buffer is bucket.main_param_buffer for bucket in buckets
    )

    gpu_param_bytes = sum(
        param.numel() * param.element_size() for param in inner.params
    )
    gpu_main_grad_bytes = sum(
        param.main_grad.numel() * param.main_grad.element_size()
        for param in inner.params
        if hasattr(param, "main_grad")
    )
    cpu_master_bytes = sum(
        param.numel() * param.element_size() for param in cpu_group._cpu_params
    )
    cpu_exp_avg_bytes = sum(
        cpu_group._cpu_optimizer.state[param]["exp_avg"].numel()
        * cpu_group._cpu_optimizer.state[param]["exp_avg"].element_size()
        for param in cpu_group._cpu_params
    )
    cpu_exp_avg_sq_bytes = sum(
        cpu_group._cpu_optimizer.state[param]["exp_avg_sq"].numel()
        * cpu_group._cpu_optimizer.state[param]["exp_avg_sq"].element_size()
        for param in cpu_group._cpu_params
    )

    assert all(param.dtype == torch.bfloat16 for param in inner.params)
    assert all(
        param.grad is None and param.main_grad.dtype == torch.float32
        for param in inner.params
    )
    assert all(
        param.device.type == "cpu" and param.dtype == torch.float32
        for param in cpu_group._cpu_params
    )
    assert gpu_param_bytes + gpu_main_grad_bytes == 6 * total_numel
    assert cpu_master_bytes == 4 * total_numel
    assert cpu_exp_avg_bytes == 4 * total_numel
    assert cpu_exp_avg_sq_bytes == 4 * total_numel


def test_mfsdp_offload_fraction_numerically_matches_no_offload():
    _Model, _Unit, ps, engine_cfg_offload = _build_offload_stack(offload_fraction=1.0)
    _, _, _, engine_cfg_gpu = _build_offload_stack(offload_fraction=0.0)

    torch.manual_seed(77)
    model_ref = _Model()
    model_cpu = copy.deepcopy(model_ref)

    chunks_ref, opt_ref = mfsdp_optimizer.build_mfsdp_stack(
        [model_ref],
        engine_cfg=engine_cfg_gpu,
        ps=ps,
        is_expert=lambda _name: False,
        fsdp_unit_modules=(_Unit,),
    )
    chunks_cpu, opt_cpu = mfsdp_optimizer.build_mfsdp_stack(
        [model_cpu],
        engine_cfg=engine_cfg_offload,
        ps=ps,
        is_expert=lambda _name: False,
        fsdp_unit_modules=(_Unit,),
    )

    value = torch.randn(3, 4)
    target = torch.randn(3, 2)

    opt_ref.zero_grad()
    torch.nn.functional.mse_loss(chunks_ref[0](value), target).backward()
    opt_ref.finish_grad_sync()
    success_ref, _norm_ref, _ = opt_ref.step()
    assert success_ref

    opt_cpu.zero_grad()
    torch.nn.functional.mse_loss(chunks_cpu[0](value), target).backward()
    opt_cpu.finish_grad_sync()
    success_cpu, _norm_cpu, _ = opt_cpu.step()
    assert success_cpu

    ref_params = {
        name.removeprefix("module."): param
        for name, param in chunks_ref[0].named_parameters()
    }
    cpu_params = {
        name.removeprefix("module."): param
        for name, param in chunks_cpu[0].named_parameters()
    }
    assert ref_params.keys() == cpu_params.keys()
    for name in ref_params:
        assert torch.equal(
            ref_params[name], cpu_params[name]
        ), f"Parameter {name} diverges between GPU-only and CPU-offload runs"


def test_mfsdp_offload_fraction_partial_splits_by_numel():
    _Model, _Unit, ps, engine_cfg = _build_offload_stack(offload_fraction=0.5)

    torch.manual_seed(7)
    chunks, optimizer = mfsdp_optimizer.build_mfsdp_stack(
        [_Model()],
        engine_cfg=engine_cfg,
        ps=ps,
        is_expert=lambda _name: False,
        fsdp_unit_modules=(_Unit,),
    )

    inner = optimizer._inner_optimizer
    cpu_group = inner.cpu_group
    assert cpu_group is not None

    total_numel = sum(p.numel() for p in inner.params)
    cpu_numel = sum(p.numel() for p in cpu_group._gpu_params)
    gpu_numel = total_numel - cpu_numel
    assert cpu_numel > 0, "Expected some params to be CPU-offloaded at fraction=0.5"
    assert gpu_numel > 0, "Expected some params to remain on GPU at fraction=0.5"
    ratio = cpu_numel / total_numel
    # The greedy split assigns whole params; exact ratio depends on model shape.
    # For fraction=0.5 with 3 params of unequal size the ratio will be ≥0.4.
    assert (
        ratio >= 0.4
    ), f"Expected substantial CPU offload at fraction=0.5, got {ratio:.2%}"
    assert ratio <= 0.95, f"Expected some GPU params at fraction=0.5, got {ratio:.2%}"


def test_mfsdp_full_offload_includes_trailing_empty_shards():
    nonempty = torch.nn.Parameter(torch.ones(4))
    trailing_empty = torch.nn.Parameter(torch.empty(0))

    gpu_groups, cpu_groups = mfsdp_optimizer._split_param_groups_by_fraction(
        [{"params": [nonempty, trailing_empty], "weight_decay": 0.0}], 1.0
    )

    assert gpu_groups == []
    assert cpu_groups[0]["params"] == [nonempty, trailing_empty]


def test_mfsdp_offload_fraction_zero_has_no_cpu_group():
    _Model, _Unit, ps, engine_cfg = _build_offload_stack(offload_fraction=0.0)

    torch.manual_seed(9)
    _chunks, optimizer = mfsdp_optimizer.build_mfsdp_stack(
        [_Model()],
        engine_cfg=engine_cfg,
        ps=ps,
        is_expert=lambda _name: False,
        fsdp_unit_modules=(_Unit,),
    )
    assert optimizer._inner_optimizer.cpu_group is None


def test_mfsdp_offload_fraction_zero_preserves_optimizer_state_dict_contract():
    _Model, _Unit, ps, engine_cfg = _build_offload_stack(offload_fraction=0.0)

    torch.manual_seed(10)
    chunks, optimizer = mfsdp_optimizer.build_mfsdp_stack(
        [_Model()],
        engine_cfg=engine_cfg,
        ps=ps,
        is_expert=lambda _name: False,
        fsdp_unit_modules=(_Unit,),
    )
    value = torch.randn(3, 4)
    target = torch.randn(3, 2)
    optimizer.zero_grad()
    torch.nn.functional.mse_loss(chunks[0](value), target).backward()
    optimizer.finish_grad_sync()
    optimizer.step()

    assert optimizer.state_dict().keys() == {
        "state",
        "param_groups",
        "_mfsdp_param_values",
    }


def test_mfsdp_optimizer_checkpoint_round_trips_fp32_shard_values():
    _Model, _Unit, ps, engine_cfg = _build_offload_stack(offload_fraction=0.0)

    torch.manual_seed(11)
    _chunks, optimizer = mfsdp_optimizer.build_mfsdp_stack(
        [_Model()],
        engine_cfg=engine_cfg,
        ps=ps,
        is_expert=lambda _name: False,
        fsdp_unit_modules=(_Unit,),
    )
    expected = [param.detach().clone() for param in _optimizer_params(optimizer)]
    saved = copy.deepcopy(optimizer.state_dict())

    with torch.no_grad():
        for param in _optimizer_params(optimizer):
            param.add_(17.0)

    optimizer.load_state_dict(saved)

    for param, expected_param in zip(
        _optimizer_params(optimizer), expected, strict=True
    ):
        assert torch.equal(param, expected_param)


@pytest.mark.parametrize("offload_fraction", [-0.01, 1.01])
def test_mfsdp_offload_fraction_rejects_out_of_range_values(offload_fraction):
    _Model, _Unit, ps, engine_cfg = _build_offload_stack(
        offload_fraction=offload_fraction
    )

    with pytest.raises(ValueError, match="offload_fraction"):
        mfsdp_optimizer.build_mfsdp_stack(
            [_Model()],
            engine_cfg=engine_cfg,
            ps=ps,
            is_expert=lambda _name: False,
            fsdp_unit_modules=(_Unit,),
        )


def test_mfsdp_cpu_offload_rejects_non_adam_optimizer():
    _Model, _Unit, ps, engine_cfg = _build_offload_stack(offload_fraction=1.0)
    engine_cfg.optimizer.optimizer = "sgd"

    with pytest.raises(ValueError, match="only supports Adam"):
        mfsdp_optimizer.build_mfsdp_stack(
            [_Model()],
            engine_cfg=engine_cfg,
            ps=ps,
            is_expert=lambda _name: False,
            fsdp_unit_modules=(_Unit,),
        )


def test_mfsdp_cpu_offload_rejects_custom_optimizer_factory():
    _Model, _Unit, ps, engine_cfg = _build_offload_stack(offload_fraction=1.0)

    with pytest.raises(ValueError, match="optimizer_factory"):
        mfsdp_optimizer.build_mfsdp_stack(
            [_Model()],
            engine_cfg=engine_cfg,
            ps=ps,
            is_expert=lambda _name: False,
            fsdp_unit_modules=(_Unit,),
            optimizer_factory=lambda groups, opt: torch.optim.SGD(groups, lr=opt.lr),
        )


def test_mfsdp_offload_fraction_checkpoint_round_trips():
    _Model, _Unit, ps, engine_cfg = _build_offload_stack(offload_fraction=1.0)

    torch.manual_seed(55)
    chunks, optimizer = mfsdp_optimizer.build_mfsdp_stack(
        [_Model()],
        engine_cfg=engine_cfg,
        ps=ps,
        is_expert=lambda _name: False,
        fsdp_unit_modules=(_Unit,),
    )

    value = torch.randn(3, 4)
    target = torch.randn(3, 2)
    optimizer.zero_grad()
    torch.nn.functional.mse_loss(chunks[0](value), target).backward()
    optimizer.finish_grad_sync()
    optimizer.step()

    saved = copy.deepcopy(optimizer.state_dict())
    assert "gpu" in saved
    assert "cpu" in saved
    assert "master_params" in saved["cpu"]

    cpu_group = optimizer._inner_optimizer.cpu_group
    assert cpu_group is not None
    expected_masters = [
        param.detach().clone() for param in saved["cpu"]["master_params"]
    ]
    with torch.no_grad():
        for cpu_param, gpu_param in zip(cpu_group._cpu_params, cpu_group._gpu_params):
            cpu_param.add_(17.0)
            gpu_param.add_(23.0)

    optimizer.load_state_dict(saved)
    for cpu_param, gpu_param, expected in zip(
        cpu_group._cpu_params, cpu_group._gpu_params, expected_masters
    ):
        assert torch.equal(cpu_param, expected)
        assert torch.equal(gpu_param, expected.to(dtype=gpu_param.dtype))

    optimizer.zero_grad()
    torch.nn.functional.mse_loss(chunks[0](value), target).backward()
    optimizer.finish_grad_sync()
    success, _, _ = optimizer.step()
    assert success

    legacy_optimizer = {"state": {}, "param_groups": []}
    expected_exp_avg = []
    for index, cpu_optimizer in enumerate(cpu_group._cpu_optimizer.optimizers):
        local = copy.deepcopy(cpu_optimizer.state_dict())
        group = local["param_groups"][0]
        group["params"] = [index]
        legacy_optimizer["param_groups"].append(group)
        if 0 in local["state"]:
            legacy_optimizer["state"][index] = local["state"][0]
            expected_exp_avg.append(local["state"][0]["exp_avg"].clone())
    legacy_saved = {
        "optimizer": legacy_optimizer,
        "master_params": [param.detach().clone() for param in cpu_group._cpu_params],
    }
    for state in cpu_group._cpu_optimizer.state.values():
        state["exp_avg"].zero_()
    cpu_group.load_state_dict(legacy_saved)
    restored_exp_avg = [
        state["exp_avg"]
        for cpu_optimizer in cpu_group._cpu_optimizer.optimizers
        for state in cpu_optimizer.state.values()
    ]
    assert len(restored_exp_avg) == len(expected_exp_avg)
    assert all(
        torch.equal(restored, expected)
        for restored, expected in zip(restored_exp_avg, expected_exp_avg)
    )


def _single_rank_mfsdp_stack():
    """Build a trained single-rank M-FSDP stack for scratch-release tests."""

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
    chunks, optimizer = mfsdp_optimizer.build_mfsdp_stack(
        [_Model()],
        engine_cfg=SimpleNamespace(
            parallel=ParallelConfig(tp=1, ep=1, etp=1, pp=1, vpp=1, cp=1), optimizer=opt
        ),
        ps=ps,
        is_expert=lambda _name: False,
        fsdp_unit_modules=(_Unit,),
    )
    # One real train step so the double-buffer allocator has cached scratch and
    # the optimizer holds shard-aliased params in their steady sharded state.
    value = torch.randn(3, 4)
    target = torch.randn(3, 2)
    optimizer.zero_grad()
    torch.nn.functional.mse_loss(chunks[0](value), target).backward()
    optimizer.finish_grad_sync()
    assert optimizer.step()[0]
    return chunks[0], optimizer


def test_mfsdp_release_export_scratch_keeps_weights_and_aliases():
    chunk, _optimizer = _single_rank_mfsdp_stack()

    # Byte-equivalent export reference: the full (materialized) parameters before
    # the scratch release. Materializing installs full-parameter leases, so
    # return to the sharded steady state (what the pre-wake production call sees,
    # post optimizer.step) before releasing the scratch.
    full_before = {
        name: param.detach().clone() for name, param in chunk.named_parameters()
    }
    chunk.param_sync.release_all()
    chunk.param_sync.discard_full_parameter_views()

    # Record the sharded weight storage so we can prove it is not moved/rebuilt.
    weight_storage = {
        id(bucket): bucket.main_param_buffer.data_ptr()
        for bucket in chunk.param_sync.buckets
    }
    weight_values = {
        id(bucket): bucket.main_param_buffer.detach().clone()
        for bucket in chunk.param_sync.buckets
    }

    chunk.release_export_scratch()

    for bucket in chunk.param_sync.buckets:
        # Sharded weights stay put (same storage, same values, still on CPU).
        assert bucket.main_param_buffer.data_ptr() == weight_storage[id(bucket)]
        assert bucket.main_param_buffer.device.type == "cpu"
        assert torch.equal(bucket.main_param_buffer, weight_values[id(bucket)])
        # The double-buffer scratch was handed back to the driver.
        assert getattr(bucket.allocator, "_slots", {}) == {}
        # Optimizer aliases still view the resident shard (no dangling storage).
        for spec in bucket.specs:
            assert spec.shard_param is not None
            if spec.shard_numel:
                expected = bucket.main_param_buffer.narrow(
                    0, spec.local_offset, spec.shard_numel
                )
                assert spec.shard_param.data.data_ptr() == expected.data_ptr()
            # Full-parameter views are dropped, not left aliasing freed scratch.
            assert spec.full_param.data.numel() == 0

    # Export after the release reproduces the pre-release parameters byte-for-byte.
    full_after = {
        name: param.detach().clone() for name, param in chunk.named_parameters()
    }
    assert full_before.keys() == full_after.keys()
    for name in full_before:
        assert torch.equal(full_before[name], full_after[name]), name


def test_mfsdp_release_export_scratch_is_idempotent():
    chunk, optimizer = _single_rank_mfsdp_stack()
    chunk.release_export_scratch()
    # A second release on already-sharded state must not raise (the busy-buffer
    # guard passes: no collective is in flight) and must keep training usable.
    chunk.release_export_scratch()
    value = torch.randn(3, 4)
    target = torch.randn(3, 2)
    optimizer.zero_grad()
    torch.nn.functional.mse_loss(chunk(value), target).backward()
    optimizer.finish_grad_sync()
    assert optimizer.step()[0]


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
            parallel=ParallelConfig(tp=1, ep=1, etp=1, pp=1, vpp=1, cp=1), optimizer=opt
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


def test_mfsdp_dtype_conversion_buffers_are_collective_scoped():
    """Cast/communication storage must not remain resident through optimizer.step."""
    model = torch.nn.Linear(4, 3, bias=False, dtype=torch.bfloat16)
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
    config = mfsdp_config.MFSDPConfig(
        bucket_size=None,
        main_params_dtype=torch.float32,
        main_grads_dtype=torch.float32,
        grad_comm_dtype=torch.bfloat16,
    )
    bucket = mfsdp_buffer.ParamAndGradBuffer(
        model,
        groups=groups,
        config=config,
        is_expert=lambda _name: False,
        unit_modules=(),
    ).buckets[0]

    assert bucket.local_compute_buffer.numel() == 0
    assert bucket.local_grad_comm_buffer.numel() == 0

    bucket.release_full_parameters()
    _, local_compute = bucket.prepare_param_gather()
    assert local_compute.numel() == bucket.local_numel
    bucket.wait_param_gather()
    assert bucket.local_compute_buffer.numel() == 0

    bucket.install_full_parameters()
    for spec in bucket.specs:
        spec.full_param.grad = torch.ones_like(spec.full_param)
    grad_reduce = mfsdp_buffer.GradReducePipeline([bucket])
    grad_reduce.reduce_gradients(bucket, force=True)
    assert bucket.local_grad_comm_buffer.numel() == bucket.local_numel
    grad_reduce.finish()
    assert bucket.local_grad_comm_buffer.numel() == 0
    assert torch.count_nonzero(bucket.main_grad_buffer) == bucket.local_numel


def test_mfsdp_reduce_scatters_each_microbatch_into_sharded_accumulation():
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
        override_optimizer_config={
            "mfsdp_sharding_strategy": "optim_grads_params",
            "bucket_size": 4,
        },
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
            parallel=ParallelConfig(tp=1, ep=1, etp=1, pp=1, vpp=1, cp=1), optimizer=opt
        ),
        ps=ps,
        is_expert=lambda _name: False,
        fsdp_unit_modules=(_GlooUnit,),
    )
    assert (
        len({bucket.allocator_layout_key for bucket in chunks[0].param_sync.buckets})
        > 1
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

        candidate_output = chunks[0](value)
        if microbatch_idx:
            buckets = chunks[0].param_sync.buckets
            assert all(bucket._full_main_grad_lease is None for bucket in buckets)
            assert any(
                torch.count_nonzero(bucket.main_grad_buffer) for bucket in buckets
            )
        candidate_loss = torch.nn.functional.mse_loss(candidate_output, target)
        if microbatch_idx == len(microbatches) - 1:
            candidate_optimizer.grad_sync_enabled = True
        (candidate_loss / len(microbatches)).backward()

        live_full_grads = sum(
            bucket._full_main_grad_lease is not None
            for bucket in chunks[0].param_sync.buckets
        )
        assert live_full_grads <= 2

    candidate_optimizer.finish_grad_sync()

    allocator = chunks[0].param_and_grad_buffer.allocator
    gradient_pool_keys = [
        key for key in allocator._slots if key[0] in {"main_grad", "grad", "grad-local"}
    ]
    assert {key[0] for key in gradient_pool_keys} == {"main_grad", "grad-local"}
    assert len(gradient_pool_keys) == 2

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
            parallel=ParallelConfig(tp=1, ep=1, etp=1, pp=1, vpp=1, cp=1), optimizer=opt
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
            parallel=ParallelConfig(tp=1, ep=1, etp=1, pp=1, vpp=1, cp=1), optimizer=opt
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


def test_mfsdp_routes_fused_wgrad_through_bucketed_fp32_main_grad():
    model = _FusedMainGradLinear()
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
        override_optimizer_config={"mfsdp_sharding_strategy": "optim_grads_params"},
    )
    chunks, optimizer = mfsdp_optimizer.build_mfsdp_stack(
        [model],
        engine_cfg=SimpleNamespace(
            parallel=ParallelConfig(tp=1, ep=1, etp=1, pp=1, vpp=1, cp=1), optimizer=opt
        ),
        ps=ps,
        is_expert=lambda _name: False,
        fsdp_unit_modules=(_FusedMainGradLinear,),
    )

    chunk = chunks[0]
    bucket = chunk.param_sync.buckets[0]
    optimizer.zero_grad()
    value = torch.randn(8, 4, dtype=torch.bfloat16)
    chunk(value).float().sum().backward()

    spec = bucket.specs[0]
    assert model.fuse_wgrad_accumulation is True
    assert spec.full_param.grad_added_to_main_grad is True
    assert spec.full_param.grad is None
    assert spec.full_param.main_grad.dtype is torch.float32
    assert spec.full_param.main_grad.untyped_storage().data_ptr() == (
        bucket.full_main_grad_buffer.untyped_storage().data_ptr()
    )
    assert not torch.equal(
        spec.full_param.main_grad,
        spec.full_param.main_grad.to(torch.bfloat16).to(torch.float32),
    )
    first_main_grad = spec.full_param.main_grad.detach().clone()
    main_grad_storage = spec.full_param.main_grad.untyped_storage().data_ptr()

    second_value = torch.randn(8, 4, dtype=torch.bfloat16)
    chunk(second_value).float().sum().backward()
    assert spec.full_param.grad is None
    second_expected = second_value.float().sum(dim=0).repeat(4, 1)
    assert spec.full_param.main_grad.untyped_storage().data_ptr() == main_grad_storage
    assert torch.equal(spec.full_param.main_grad, second_expected)

    optimizer.finish_grad_sync()
    assert torch.equal(
        bucket.main_grad_buffer.view_as(second_expected),
        first_main_grad + second_expected,
    )
    expected_main_grad = (first_main_grad + second_expected).reshape(-1)

    optimizer.finish_grad_sync()
    consumed_grad = _optimizer_params(optimizer)[0].grad
    assert consumed_grad is not None
    assert consumed_grad.dtype is torch.float32
    assert torch.equal(consumed_grad, expected_main_grad)
    assert not torch.equal(
        consumed_grad, consumed_grad.to(torch.bfloat16).to(torch.float32)
    )


def test_mfsdp_routes_regular_autograd_grad_through_fp32_main_grad():
    chunk, optimizer = _single_rank_mfsdp_stack()
    optimizer.zero_grad()
    value = torch.randn(3, 4)
    chunk(value).square().mean().backward()
    expected = {}
    for bucket in chunk.param_sync.buckets:
        for spec in bucket.specs:
            grad = spec.full_param.main_grad
            if grad.numel() == 0:
                grad = bucket.main_grad_buffer.narrow(
                    0, spec.local_offset, spec.shard_numel
                ).view_as(spec.shard_param)
            expected[spec.name] = grad.detach().clone()
    for bucket in chunk.param_sync.buckets:
        for spec in bucket.specs:
            assert spec.full_param.grad is None
            assert spec.full_param.main_grad.dtype is torch.float32

    optimizer.finish_grad_sync()

    for name, shard in _optimizer_named_shards(optimizer).items():
        assert shard.grad is not None
        assert torch.equal(shard.grad.reshape(-1), expected[name].reshape(-1))


def test_mfsdp_accumulates_regular_wgrad_in_fp32_per_microbatch():
    model = _RegularMainGradLinear()
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
        override_optimizer_config={"mfsdp_sharding_strategy": "optim_grads_params"},
    )
    chunks, optimizer = mfsdp_optimizer.build_mfsdp_stack(
        [model],
        engine_cfg=SimpleNamespace(
            parallel=ParallelConfig(tp=1, ep=1, etp=1, pp=1, vpp=1, cp=1), optimizer=opt
        ),
        ps=ps,
        is_expert=lambda _name: False,
        fsdp_unit_modules=(_RegularMainGradLinear,),
    )

    chunk = chunks[0]
    bucket = chunk.param_sync.buckets[0]
    spec = bucket.specs[0]
    optimizer.zero_grad()

    chunk(torch.full((1, 4), 256.0, dtype=torch.bfloat16)).sum().backward()
    assert spec.full_param.grad is None
    assert torch.equal(spec.full_param.main_grad, torch.full((4, 4), 256.0))

    chunk(torch.ones(1, 4, dtype=torch.bfloat16)).sum().backward()
    assert spec.full_param.grad is None
    expected = torch.full((4, 4), 257.0)
    assert torch.equal(spec.full_param.main_grad, torch.ones((4, 4)))

    optimizer.finish_grad_sync()
    assert torch.equal(bucket.main_grad_buffer.view_as(expected), expected)
    assert not torch.equal(
        bucket.main_grad_buffer,
        bucket.main_grad_buffer.to(torch.bfloat16).to(torch.float32),
    )
    consumed_grad = _optimizer_params(optimizer)[0].grad
    assert consumed_grad is not None
    assert consumed_grad.dtype is torch.float32
    assert torch.equal(consumed_grad, expected.reshape(-1))


def _run_mfsdp_gloo_parity(rank: int, world_size: int, init_file: str) -> None:
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=world_size
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
            parallel=ParallelConfig(tp=1, ep=1, etp=1, pp=1, vpp=1, cp=1), optimizer=opt
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
        reference_optimizer.zero_grad()
        candidate_optimizer.zero_grad()
        for _microbatch in range(2):
            value = torch.randn(3, 4)
            target = torch.randn(3, 2)
            (torch.nn.functional.mse_loss(reference(value), target) / 2).backward()
            (torch.nn.functional.mse_loss(chunks[0](value), target) / 2).backward()
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
