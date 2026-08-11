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
from megatron.lite.primitive.ckpt import dcp as checkpoint_dcp
from megatron.lite.primitive.optimizers.mfsdp import buffer as mfsdp_buffer
from megatron.lite.primitive.optimizers.mfsdp import config as mfsdp_config
from megatron.lite.primitive.optimizers.mfsdp import fused_ops as mfsdp_fused_ops
from megatron.lite.primitive.optimizers.mfsdp import optimizer as mfsdp_optimizer
from megatron.lite.primitive.optimizers.mfsdp import wrapper as mfsdp_wrapper
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


class _FineGrainedContainer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 4, bias=False)

    def forward(self, value):
        return self.linear(value)


class _FineGrainedUnit(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fine = _FineGrainedContainer()
        self.output = torch.nn.Linear(4, 4, bias=False)

    def forward(self, value):
        return self.output(self.fine(value))


class _FineGrainedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.unit = _FineGrainedUnit()

    def forward(self, value):
        return self.unit(value)


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
            main_grad = (
                ctx.weight.get_main_grad()
                if hasattr(ctx.weight, "__fsdp_param__")
                else ctx.weight.main_grad
            )
            main_grad.add_(grad_weight)
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


def test_mfsdp_fused_optimizer_accepts_disjoint_views_of_flat_storage(monkeypatch):
    flat = torch.zeros(8)
    params = [
        torch.nn.Parameter(flat[:4]),
        torch.nn.Parameter(flat[4:]),
    ]
    expected = object()
    calls = []

    monkeypatch.setattr(mfsdp_fused_ops, "_has_cuda_params", lambda _groups: True)

    def build_fused(name, groups, opt, *, lr, weight_decay, use_decoupled_grad):
        calls.append((name, groups, opt, lr, weight_decay, use_decoupled_grad))
        return expected

    monkeypatch.setattr(mfsdp_fused_ops, "_build_fused_optimizer", build_fused)
    opt = SimpleNamespace(
        optimizer="adam",
        lr=2.0e-4,
        weight_decay=0.1,
        override_optimizer_config={},
    )

    actual = mfsdp_fused_ops.build_optimizer([{"params": params}], opt)

    assert actual is expected
    assert len(calls) == 1
    assert calls[0][-1] is False
    assert params[0].untyped_storage().data_ptr() == params[1].untyped_storage().data_ptr()


def test_mfsdp_fused_optimizer_prefers_transformer_engine(monkeypatch):
    expected = object()
    imports = []

    class TEOptimizers:
        @staticmethod
        def FusedAdam(*_args, **_kwargs):
            return expected

    def import_module(name):
        imports.append(name)
        if name == "transformer_engine.pytorch.optimizers":
            return TEOptimizers
        raise AssertionError(f"unexpected fallback import: {name}")

    monkeypatch.setattr(mfsdp_fused_ops.importlib, "import_module", import_module)
    opt = SimpleNamespace(optimizer="adam")

    actual = mfsdp_fused_ops._build_fused_optimizer(
        "adam",
        [{"params": [torch.nn.Parameter(torch.zeros(1))]}],
        opt,
        lr=1e-4,
        weight_decay=0.01,
    )

    assert actual is expected
    assert imports == ["transformer_engine.pytorch.optimizers"]


def test_mfsdp_precision_aware_fused_adam_uses_decoupled_grad(monkeypatch):
    captured = {}

    class TEOptimizers:
        @staticmethod
        def FusedAdam(*_args, **kwargs):
            captured.update(kwargs)
            return object()

    monkeypatch.setattr(
        mfsdp_fused_ops.importlib,
        "import_module",
        lambda name: TEOptimizers
        if name == "transformer_engine.pytorch.optimizers"
        else (_ for _ in ()).throw(AssertionError(name)),
    )
    opt = SimpleNamespace(optimizer="adam")

    mfsdp_fused_ops._build_fused_optimizer(
        "adam",
        [{"params": [torch.nn.Parameter(torch.zeros(1))]}],
        opt,
        lr=1e-4,
        weight_decay=0.01,
        use_decoupled_grad=True,
    )

    assert captured["use_decoupled_grad"] is True
    assert captured["master_weights"] is False


def test_mfsdp_decoupled_grad_reuses_main_grad_without_fp32_copy():
    param = torch.nn.Parameter(torch.zeros(4, dtype=torch.float32))
    main_grad = torch.ones(4, dtype=torch.bfloat16)
    param.main_grad = main_grad

    class _Optimizer:
        param_groups = [{"params": [param]}]

    adapter = mfsdp_optimizer._StandaloneOptimizer(
        _Optimizer(),
        [param],
        ps=SimpleNamespace(),
        clip_grad=0.0,
        grad_norm_accum_dtype=torch.float32,
        expert_params=[],
        expert_grad_scale=1.0,
        use_decoupled_grad=True,
    )

    adapter.update_optimizer_grads()

    assert param.grad is None
    assert param.decoupled_grad is main_grad


def test_mfsdp_fine_grained_hook_reuses_outer_unit_bucket_and_window():
    model = _FineGrainedModel()
    chunks, _optimizer = mfsdp_optimizer.build_mfsdp_stack(
        [model],
        engine_cfg=_engine_cfg(),
        ps=SimpleNamespace(),
        fsdp_unit_modules=(_FineGrainedUnit,),
        enable_fine_grained_param_gather_hook=True,
        enable_fine_grained_param_gather_backward_hook=True,
        fine_grained_recurse_module_types=(_FineGrainedContainer,),
        suggested_communication_unit_size=17,
    )
    wrapped = chunks[0]
    fine_ids = wrapped.param_and_grad_buffer.bucket_ids_for_module(
        model.unit.fine, recurse=True
    )

    assert fine_ids
    assert (
        wrapped.param_and_grad_buffer.bucket_ids_for_module(
            model.unit.fine, recurse=False
        )
        == ()
    )
    assert id(model.unit) in wrapped.param_and_grad_buffer.owners
    assert id(model.unit.fine) not in wrapped.param_and_grad_buffer.owners
    assert all(wrapped.param_sync.buckets[index].is_fsdp_unit for index in fine_ids)
    assert wrapped.mfsdp_config.suggested_communication_unit_size == 17
    assert len(model.unit.fine._forward_pre_hooks) == 1
    assert len(model.unit.fine._forward_hooks) == 1

    output = wrapped(torch.randn(2, 4, requires_grad=True))
    output.sum().backward()


def test_mfsdp_fine_grained_backward_keeps_unit_unshard_marker():
    """Fine-grained gather augments rather than replaces MCore's unit marker."""

    class DirectUnit(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.randn(4, 4))

        def forward(self, value):
            return torch.nn.functional.linear(value, self.weight)

    model = torch.nn.Sequential(DirectUnit())
    chunks, _optimizer = mfsdp_optimizer.build_mfsdp_stack(
        [model],
        engine_cfg=_engine_cfg(),
        ps=SimpleNamespace(),
        fsdp_unit_modules=(DirectUnit,),
        enable_fine_grained_param_gather_hook=True,
        enable_fine_grained_param_gather_backward_hook=True,
    )
    wrapped = chunks[0]
    unit_ids = tuple(wrapped.param_and_grad_buffer.owners[id(model[0])])
    acquisitions = []
    acquire_backward_ids = wrapped.param_sync.acquire_backward_ids

    def record_acquisition(bucket_ids):
        acquisitions.append(tuple(bucket_ids))
        acquire_backward_ids(bucket_ids)

    wrapped.param_sync.acquire_backward_ids = record_acquisition
    wrapped(torch.randn(2, 4, requires_grad=True)).sum().backward()

    # MCore installs both markers: the fine-grained marker for the unit's
    # shallow parameter and the enclosing FSDP-unit pre-backward marker.
    assert acquisitions.count(unit_ids) == 2


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


def test_mfsdp_hotpath_excludes_lite_owner_scheduler_and_keeps_mcore_hooks():
    package = Path(mfsdp_config.__file__).parent
    source = "\n".join(path.read_text() for path in package.glob("*.py"))
    forbidden = (
        "TrainingState",
        "process_post_backward(owner_id",
        "release_post_backward_owner",
        "register_full_backward_hook",
        "_grad_generation",
        "owner_cursor",
    )

    assert {token for token in forbidden if token in source} == set()
    assert "register_post_accumulate_grad_hook" in source
    assert "queue_callback" in source
    assert "finish_microbatch" in source
    assert "_coalescing_manager" in source


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


def test_mfsdp_precision_aware_config_enables_decoupled_grad():
    config = mfsdp_config.build_mfsdp_config(
        SimpleNamespace(
            optimizer="adam",
            use_precision_aware_optimizer=True,
            override_optimizer_config={},
        )
    )

    assert config.use_decoupled_grad is True


def test_mfsdp_per_token_loss_uses_sum_gradient_collective():
    config = mfsdp_config.build_mfsdp_config(
        SimpleNamespace(override_optimizer_config={}),
        calculate_per_token_loss=True,
    )

    assert config.calculate_per_token_loss is True
    assert config.average_gradients is False


def test_mfsdp_per_token_finalize_scales_by_global_token_count():
    scales = []

    class _Chunk:
        def scale_gradients(self, scale):
            scales.append(scale.detach().clone())

    mfsdp_optimizer._scale_gradients_by_global_tokens(
        [_Chunk()],
        torch.tensor(8, dtype=torch.int64),
        SimpleNamespace(pp_group=None, dp_cp_group=None),
    )

    torch.testing.assert_close(scales[0], torch.tensor(0.125))


def test_mfsdp_can_scale_persistent_grads_before_installing_optimizer_grads():
    events = []

    class _Inner:
        def update_optimizer_grads(self):
            events.append("install")

    class _Chunk:
        def finish_grad_sync(self):
            events.append("finish")

    optimizer = mfsdp_optimizer.MFSdpOptimizer(_Inner(), [_Chunk()])

    optimizer.finish_grad_sync(update_optimizer_grads=False)
    events.append("scale")
    optimizer.update_optimizer_grads()

    assert events == ["finish", "scale", "install"]


def test_mfsdp_config_supports_recommended_manual_nccl_registration():
    config = mfsdp_config.build_mfsdp_config(
        SimpleNamespace(
            override_optimizer_config={
                "nccl_ub": True,
                "fsdp_manual_registration": True,
            }
        )
    )

    assert config.nccl_ub is True
    assert config.fsdp_double_buffer is True
    assert config.fsdp_manual_registration is True


def test_mfsdp_config_rejects_manual_registration_without_nccl_ub():
    with pytest.raises(ValueError, match="requires nccl_ub=True"):
        mfsdp_config.build_mfsdp_config(
            SimpleNamespace(
                override_optimizer_config={"fsdp_manual_registration": True}
            )
        )


def test_mfsdp_config_keeps_double_buffer_opt_in_without_nccl_user_buffers():
    config = mfsdp_config.build_mfsdp_config(
        SimpleNamespace(
            override_optimizer_config={"mfsdp_sharding_strategy": "optim_grads_params"}
        )
    )

    assert config.nccl_ub is False
    assert config.fsdp_double_buffer is False


def test_mfsdp_config_max_pool_enables_double_buffer_for_hybrid_units():
    config = mfsdp_config.build_mfsdp_config(
        SimpleNamespace(
            override_optimizer_config={
                "megatron_fsdp_max_pool_double_buffer": True
            }
        )
    )

    assert config.maxpool_double_buffer is True
    assert config.fsdp_double_buffer is True
    allocators = mfsdp_buffer.build_temporary_allocator(config, ())
    assert isinstance(
        allocators.weight_allocator, mfsdp_buffer.MaxPoolBufferAllocator
    )
    assert isinstance(allocators.grad_allocator, mfsdp_buffer.MaxPoolBufferAllocator)


def test_mfsdp_config_uses_recipe_wgrad_policy_and_allows_explicit_opt_in():
    recommended = mfsdp_config.build_mfsdp_config(SimpleNamespace())
    fused = mfsdp_config.build_mfsdp_config(
        SimpleNamespace(
            override_optimizer_config={"gradient_accumulation_fusion": True}
        )
    )

    assert recommended.gradient_accumulation_fusion is False
    assert fused.gradient_accumulation_fusion is True


def test_mfsdp_temporary_lease_physically_releases_storage():
    allocator = mfsdp_buffer.TemporaryBufferAllocator()
    lease = allocator.allocate(
        32, dtype=torch.float32, device=torch.device("cpu"), group=None, key=("grad",)
    )

    assert lease.tensor.untyped_storage().nbytes() == 32 * 4
    lease.release()
    assert lease.tensor.untyped_storage().nbytes() == 0


def test_mfsdp_storage_resize_allocator_reuses_bucket_tensor_object():
    allocator = mfsdp_buffer.StorageResizeBufferAllocator()
    args = dict(
        numel=32,
        dtype=torch.float32,
        device=torch.device("cpu"),
        group=None,
        key=("param", 7),
    )

    first = allocator.allocate(**args)
    tensor = first.tensor
    saved_view = tensor.view(4, 8)
    first.release()
    assert tensor.untyped_storage().nbytes() == 0

    second = allocator.allocate(**args)
    assert second.tensor is tensor
    assert second.tensor.untyped_storage().nbytes() == 32 * 4
    second.tensor.fill_(3.0)
    torch.testing.assert_close(saved_view, torch.full((4, 8), 3.0))
    second.release()


def test_mfsdp_storage_resize_allocator_rejects_concurrent_bucket_reuse():
    allocator = mfsdp_buffer.StorageResizeBufferAllocator()
    args = dict(
        numel=8,
        dtype=torch.float32,
        device=torch.device("cpu"),
        group=None,
        key=("main_grad", 3),
    )
    lease = allocator.allocate(**args)

    with pytest.raises(RuntimeError, match="already in use"):
        allocator.allocate(**args)

    lease.release()


def test_mfsdp_non_double_buffer_routes_weights_to_storage_resize_only():
    config = mfsdp_config.MFSDPConfig(fsdp_double_buffer=False, nccl_ub=False)
    allocator = mfsdp_buffer.build_temporary_allocator(config, ())

    assert isinstance(
        allocator.weight_allocator, mfsdp_buffer.StorageResizeBufferAllocator
    )
    assert type(allocator.grad_allocator) is mfsdp_buffer.TemporaryBufferAllocator


def test_mfsdp_double_buffer_rejects_a_third_in_flight_lease():
    allocator = mfsdp_buffer.DoubleBufferAllocator()
    buckets = [
        SimpleNamespace(
            full_numel=8,
            is_fsdp_unit=True,
            policy=SimpleNamespace(compute_dtype=torch.float32),
        )
        for _ in range(3)
    ]
    allocator.configure(buckets, ((0,), (1,), (2,)))
    args = dict(numel=8, dtype=torch.float32, device=torch.device("cpu"), group=None)
    allocator.allocate(**args, key=("grad", 0))
    allocator.allocate(**args, key=("grad", 1))

    with pytest.raises(RuntimeError, match="capacity exhausted"):
        allocator.allocate(**args, key=("grad", 2))


def test_mfsdp_double_buffer_only_pools_modal_symmetric_units():
    allocator = mfsdp_buffer.DoubleBufferAllocator()
    buckets = [
        SimpleNamespace(
            full_numel=numel,
            is_fsdp_unit=True,
            policy=SimpleNamespace(compute_dtype=torch.float32),
        )
        for numel in (8, 8, 8, 32)
    ]
    allocator.configure(buckets, ((0,), (1,), (2,), (3,)))

    pooled = allocator.allocate(
        8,
        dtype=torch.float32,
        device=torch.device("cpu"),
        group=None,
        key=("param", 0),
    )
    asymmetric = allocator.allocate(
        32,
        dtype=torch.float32,
        device=torch.device("cpu"),
        group=None,
        key=("param", 3),
    )

    assert pooled.owner is allocator
    assert isinstance(asymmetric.owner, mfsdp_buffer.StorageResizeBufferAllocator)
    pooled.release()
    asymmetric.release()


def test_mfsdp_max_pool_covers_asymmetric_units_with_dtype_maxima():
    allocator = mfsdp_buffer.MaxPoolBufferAllocator()
    buckets = [
        SimpleNamespace(
            full_numel=numel,
            is_fsdp_unit=True,
            policy=SimpleNamespace(compute_dtype=dtype),
        )
        for numel, dtype in (
            (8, torch.float32),
            (4, torch.float32),
            (32, torch.float32),
            (6, torch.bfloat16),
        )
    ]
    allocator.configure(buckets, ((0, 1), (2, 3)))

    small = allocator.allocate(
        8,
        dtype=torch.float32,
        device=torch.device("cpu"),
        group=None,
        key=("param", 0),
    )
    large = allocator.allocate(
        32,
        dtype=torch.float32,
        device=torch.device("cpu"),
        group=None,
        key=("param", 2),
    )
    bf16 = allocator.allocate(
        6,
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
        group=None,
        key=("param", 3),
    )

    assert small.owner is allocator
    assert large.owner is allocator
    assert bf16.owner is allocator
    assert small.tensor.numel() == 8
    assert small.tensor.untyped_storage().nbytes() == 32 * 4
    assert large.tensor.numel() == 32
    assert bf16.tensor.numel() == 6
    small.release()
    large.release()
    bf16.release()


def test_mfsdp_fixed_pool_uses_complete_dense_and_expert_unit_layouts():
    class MixedUnit(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dense = torch.nn.Parameter(torch.ones(8))
            self.expert = torch.nn.Parameter(torch.ones(16))

    model = torch.nn.Sequential(MixedUnit(), MixedUnit(), MixedUnit())
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
    buffers = mfsdp_buffer.ParamAndGradBuffer(
        model,
        groups=groups,
        config=mfsdp_config.MFSDPConfig(fsdp_double_buffer=True),
        is_expert=lambda name: name.endswith("expert"),
        unit_modules=(MixedUnit,),
    )

    # Collective scheduling separates dense/expert process groups, while the
    # MCore FixedPool layout keeps both offsets in each complete FSDP unit.
    assert all(len(group) == 1 for group in buffers.collective_groups)
    assert all(len(group) == 2 for group in buffers.owners.values())
    assert buffers.allocator.weight_allocator._eligible_bucket_ids == set(
        range(len(buffers.buckets))
    )


def test_mfsdp_double_buffer_named_parameters_and_state_dict_stay_sharded():
    chunk, _optimizer = _single_rank_mfsdp_stack(
        override_optimizer_config={"fsdp_double_buffer": True}
    )
    allocator = chunk.param_and_grad_buffer.allocator
    slots_before = {
        key: tuple(id(tensor) if tensor is not None else None for tensor in slots)
        for key, slots in allocator._slots.items()
    }
    assert slots_before
    assert not any(allocator._busy.values())

    idle_params = dict(chunk.named_parameters())
    optimizer_params = dict(chunk.named_optimizer_parameters())
    assert idle_params.keys() == {f"module.{name}" for name in optimizer_params}
    for name, optimizer_param in optimizer_params.items():
        assert idle_params[f"module.{name}"] is optimizer_param
    # MCore keeps the original compute Parameter installed in the module and
    # only releases/rebinds its data storage.  Optimizer shards are exposed via
    # the wrapper interface; they are not swapped into the model lifecycle.
    for bucket in chunk.param_sync.buckets:
        for spec in bucket.specs:
            for binding in spec.bindings:
                assert getattr(binding.module, binding.attribute) is spec.full_param
            assert spec.full_param.data.numel() == spec.numel
    assert not any(allocator._busy.values())

    state = chunk.state_dict()
    assert state.keys() == idle_params.keys()
    for bucket in chunk.param_sync.buckets:
        for spec in bucket.specs:
            for binding in spec.bindings:
                assert getattr(binding.module, binding.attribute) is spec.full_param
    slots_after = {
        key: tuple(id(tensor) if tensor is not None else None for tensor in slots)
        for key, slots in allocator._slots.items()
    }
    assert slots_after == slots_before
    assert not any(allocator._busy.values())


def test_mfsdp_double_buffer_reuse_matches_non_double_gradients():
    class Unit(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(4, 4, bias=False)

        def forward(self, value):
            return torch.nn.functional.gelu(self.linear(value))

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.units = torch.nn.ModuleList(Unit() for _ in range(4))

        def forward(self, value):
            for unit in self.units:
                value = unit(value)
            return value

    def run(*, double_buffer):
        torch.manual_seed(567)
        model = Model()
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
            override_optimizer_config={"fsdp_double_buffer": double_buffer},
        )
        chunks, optimizer = mfsdp_optimizer.build_mfsdp_stack(
            [model],
            engine_cfg=SimpleNamespace(
                parallel=ParallelConfig(tp=1, ep=1, etp=1, pp=1, vpp=1, cp=1),
                optimizer=opt,
            ),
            ps=ps,
            is_expert=lambda _name: False,
            fsdp_unit_modules=(Unit,),
        )
        optimizer.zero_grad()
        value = torch.arange(12, dtype=torch.float32).view(3, 4) / 10
        chunks[0](value).square().sum().backward()
        optimizer.finish_grad_sync()
        return {
            name: param.grad.detach().clone()
            for name, param in chunks[0].named_optimizer_parameters()
        }

    reference = run(double_buffer=False)
    candidate = run(double_buffer=True)

    assert candidate.keys() == reference.keys()
    for name in reference:
        torch.testing.assert_close(candidate[name], reference[name], rtol=0, atol=0)


def test_mfsdp_nccl_user_buffer_fails_loudly_when_allocators_are_missing(
    monkeypatch,
):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    def missing_apex(_name):
        raise ImportError("optional allocator missing")

    monkeypatch.setattr(mfsdp_buffer.importlib, "import_module", missing_apex)
    with pytest.raises(RuntimeError, match="NCCL user-buffer allocation unavailable"):
        mfsdp_buffer.NCCLUserBuffer(
            enabled=True,
            groups=(),
            symmetric=True,
            manual_registration=False,
        )


def test_mfsdp_nccl_user_buffer_prefers_mcore_manual_registration(monkeypatch):
    calls = []

    class FakeContext:
        def __enter__(self):
            calls.append("enter")

        def __exit__(self, *_args):
            calls.append("exit")

    class FakeAllocator:
        @staticmethod
        def init():
            calls.append("init")

        @staticmethod
        def create_nccl_mem_pool(*, symmetric):
            calls.append(("pool", symmetric))
            return "pool"

        @staticmethod
        def MemPoolAllocatorWithoutRegistration(pool):
            calls.append(("allocate", pool))
            return FakeContext()

        @staticmethod
        def register_mem_pool(pool, group, *, symmetric):
            calls.append(("register", pool, group, symmetric))

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: calls.append("sync"))
    monkeypatch.setattr(dist, "is_initialized", lambda: False)
    monkeypatch.setattr(
        mfsdp_buffer.importlib,
        "import_module",
        lambda name: FakeAllocator if name == "megatron.core.nccl_allocator" else None,
    )
    user_buffer = mfsdp_buffer.NCCLUserBuffer(
        enabled=True,
        groups=("dense", "expert"),
        symmetric=True,
        manual_registration=True,
    )

    with user_buffer.allocation_context():
        calls.append("allocated")
    user_buffer.manual_buffer_registration()
    user_buffer.manual_buffer_registration()

    assert user_buffer.allocator_kind == "mcore"
    assert calls.count(("register", "pool", "dense", True)) == 1
    assert calls.count(("register", "pool", "expert", True)) == 1


def test_mfsdp_optimizer_manually_registers_buffers_after_first_step_once():
    calls = []

    class FakeInner:
        param_groups = []

        def step(self):
            calls.append("optimizer_step")
            return True, 1.0, 0

    class FakeSync:
        def copy_main_weights_to_model_weights(self):
            calls.append("copy_weights")

        def invalidate_parameters(self):
            calls.append("invalidate")

        def set_grad_sync_enabled(self, _enabled):
            pass

    chunk = SimpleNamespace(
        param_sync=FakeSync(),
        mfsdp_config=SimpleNamespace(fsdp_manual_registration=True),
        manual_buffer_registration=lambda: calls.append("manual_register"),
    )
    optimizer = mfsdp_optimizer.MFSdpOptimizer(FakeInner(), [chunk])

    optimizer.step()
    optimizer.step()

    assert calls.count("optimizer_step") == 2
    assert calls.count("manual_register") == 1
    assert calls.index("manual_register") > calls.index("optimizer_step")


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


def test_mfsdp_intersecting_parallel_groups_disable_param_gather_overlap(
    monkeypatch, caplog
):
    config = mfsdp_config.MFSDPConfig(overlap_param_gather=True)
    ps = SimpleNamespace(
        tp_size=2,
        cp_size=2,
        ep_size=2,
        etp_size=2,
        dp_cp_size=2,
        expert_dp_size=1,
    )
    monkeypatch.setattr(dist, "is_initialized", lambda: False)

    with caplog.at_level("WARNING"):
        ordered_config = mfsdp_optimizer._order_param_gathers_for_parallel_collectives(
            config, ps
        )

    assert ordered_config.overlap_param_gather is False
    assert "intersecting process groups" in caplog.text


def test_mfsdp_start_param_sync_materializes_synchronously_when_overlap_is_disabled():
    events = []
    module = SimpleNamespace(
        mfsdp_config=SimpleNamespace(
            overlap_param_gather=False,
            all_gather_in_start_param_sync=True,
        ),
        param_sync=SimpleNamespace(
            buckets=[object()],
            materialize_all=lambda: events.append("materialize"),
        ),
        all_gather_pipeline=SimpleNamespace(
            async_bucket_gather=lambda bucket_id: events.append(
                ("async_bucket_gather", bucket_id)
            )
        ),
    )

    mfsdp_wrapper.MFSdpModule.start_param_sync(module)

    assert events == ["materialize"]


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
    candidate_params = dict(chunks[0].stream_full_parameters())
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


def _build_optimizer_stack(offload_fraction: float = 0.0):
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


def test_mfsdp_offload_fraction_zero_preserves_optimizer_state_dict_contract():
    _Model, _Unit, ps, engine_cfg = _build_optimizer_stack()

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
    _Model, _Unit, ps, engine_cfg = _build_optimizer_stack()

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


@pytest.mark.parametrize("offload_fraction", [-0.01, 0.5, 1.0, 1.01])
def test_mfsdp_rejects_optimizer_offload(offload_fraction):
    _Model, _Unit, ps, engine_cfg = _build_optimizer_stack(offload_fraction)

    with pytest.raises(ValueError, match="offload_fraction"):
        mfsdp_optimizer.build_mfsdp_stack(
            [_Model()],
            engine_cfg=engine_cfg,
            ps=ps,
            is_expert=lambda _name: False,
            fsdp_unit_modules=(_Unit,),
        )


def _single_rank_mfsdp_stack(*, override_optimizer_config=None):
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
        override_optimizer_config={
            "mfsdp_sharding_strategy": "optim_grads_params",
            **(override_optimizer_config or {}),
        },
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


def test_mfsdp_scale_gradients_updates_persistent_shards():
    chunk, _optimizer = _single_rank_mfsdp_stack()
    for bucket in chunk.param_sync.buckets:
        bucket.main_grad_buffer.fill_(4.0)

    chunk.scale_gradients(torch.tensor(0.25))

    for bucket in chunk.param_sync.buckets:
        torch.testing.assert_close(
            bucket.main_grad_buffer, torch.ones_like(bucket.main_grad_buffer)
        )


def test_mfsdp_borrowed_export_stream_aliases_until_advance():
    chunk, _optimizer = _single_rank_mfsdp_stack()
    stream = chunk.stream_borrowed_full_parameters()

    name, borrowed = next(stream)
    bucket, spec = next(
        (bucket, spec)
        for bucket in chunk.param_sync.buckets
        for spec in bucket.specs
        if spec.name == name
    )
    assert borrowed is spec.full_param
    assert borrowed.data_ptr() == spec.full_param.data_ptr()
    assert borrowed.numel() == spec.numel

    stream.close()
    assert bucket._full_lease is None
    assert bucket.full_buffer.numel() == 0


def test_mfsdp_owning_export_stream_close_releases_current_bucket():
    chunk, _optimizer = _single_rank_mfsdp_stack()
    stream = chunk.stream_full_parameters()

    name, _owned = next(stream)
    bucket = next(
        bucket
        for bucket in chunk.param_sync.buckets
        if any(spec.name == name for spec in bucket.specs)
    )
    assert bucket._full_lease is not None

    stream.close()
    assert bucket._full_lease is None
    assert bucket.full_buffer.numel() == 0


def test_mfsdp_release_export_scratch_keeps_weights_and_aliases():
    chunk, _optimizer = _single_rank_mfsdp_stack()

    # Byte-equivalent export reference from MCore's bounded full-weight stream.
    full_before = {
        name: param.detach().clone() for name, param in chunk.stream_full_parameters()
    }

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
            # MCore preserves non-unit full compute storage across module
            # boundaries; only explicit FSDP-unit views alias releasable
            # all-gather scratch.
            if bucket.is_fsdp_unit:
                assert spec.full_param.data.numel() == 0
            else:
                assert spec.full_param.data.numel() == spec.numel

    # Export after the release reproduces the pre-release parameters byte-for-byte.
    full_after = {
        name: param.detach().clone() for name, param in chunk.stream_full_parameters()
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


def test_mfsdp_bucket_size_is_independent_from_communication_unit_size():
    opt = SimpleNamespace(
        override_optimizer_config={
            "bucket_size": 17,
            "suggested_communication_unit_size": 123,
        }
    )

    config = mfsdp_config.build_mfsdp_config(opt)

    assert config.bucket_size == 17
    assert config.suggested_communication_unit_size == 123


def test_mfsdp_grouped_expert_chunk_factors_split_but_share_one_collective_group():
    class ExpertUnit(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.first = torch.nn.Parameter(torch.ones(2, 3, 4))
            self.second = torch.nn.Parameter(torch.ones(2, 2, 4))

    model = ExpertUnit()
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
    buffers = mfsdp_buffer.ParamAndGradBuffer(
        model,
        groups=groups,
        config=mfsdp_config.MFSDPConfig(bucket_size=None),
        is_expert=lambda _name: True,
        unit_modules=(ExpertUnit,),
    )

    assert [bucket.chunk_size_factor for bucket in buffers.buckets] == [12, 8]
    assert buffers.collective_groups == ((0, 1),)


def test_mfsdp_shared_embedding_gets_a_dedicated_bucket():
    model = torch.nn.Module()
    model.first = torch.nn.Parameter(torch.ones(2, 4))
    model.embedding = torch.nn.Parameter(torch.ones(2, 4))
    model.embedding.shared_embedding = True
    model.last = torch.nn.Parameter(torch.ones(2, 4))
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
    buffers = mfsdp_buffer.ParamAndGradBuffer(
        model,
        groups=groups,
        config=mfsdp_config.MFSDPConfig(bucket_size=None),
        is_expert=lambda _name: False,
        unit_modules=(),
    )

    assert [[spec.name for spec in bucket.specs] for bucket in buffers.buckets] == [
        ["first"],
        ["embedding"],
        ["last"],
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

    # A completed reduce-scatter installs gradients only for non-empty local
    # shards.  FusedAdam must see ``grad is None`` for empty inputs.
    bucket._grad_reduce_launched = True
    bucket.wait_grad_reduce()
    assert bucket.specs[0].shard_param is not None
    assert bucket.specs[0].shard_param.grad is not None
    assert bucket.specs[1].shard_param.grad is None
    assert bucket.specs[1].shard_param.main_grad is None


def test_mfsdp_frozen_bucket_has_compute_shard_without_master_or_grad_storage():
    model = torch.nn.Linear(4, 3, bias=False, dtype=torch.bfloat16)
    model.weight.requires_grad_(False)
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
    bucket = mfsdp_buffer.ParamAndGradBuffer(
        model,
        groups=groups,
        config=mfsdp_config.MFSDPConfig(
            main_params_dtype=torch.float32, main_grads_dtype=torch.float32
        ),
        is_expert=lambda _name: False,
        unit_modules=(torch.nn.Linear,),
    ).buckets[0]

    assert bucket.requires_grad is False
    assert bucket.main_param_buffer.dtype is torch.bfloat16
    assert bucket.main_grad_buffer.numel() == 0
    pipeline = mfsdp_buffer.GradReducePipeline([bucket], ((bucket.bucket_id,),))
    assert pipeline._bucket_groups == ()
    pipeline.finish_microbatch()


def test_mfsdp_model_shard_is_persistent_and_grad_conversion_is_collective_scoped():
    """MCore keeps model shards resident while gradient conversion stays scratch."""
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
        unit_modules=(torch.nn.Linear,),
    ).buckets[0]

    assert bucket.local_compute_buffer.numel() == bucket.local_numel
    assert bucket.local_compute_buffer is bucket.model_param_buffer
    assert bucket.local_compute_buffer is not bucket.main_param_buffer
    assert bucket.local_grad_comm_buffer.numel() == 0

    bucket.release_full_parameters()
    _, local_compute = bucket.prepare_param_gather()
    assert local_compute.numel() == bucket.local_numel
    bucket.wait_param_gather()
    assert bucket.local_compute_buffer.numel() == bucket.local_numel

    with torch.no_grad():
        bucket.main_param_buffer.add_(1.0)
    assert not torch.equal(
        bucket.model_param_buffer,
        bucket.main_param_buffer.to(bucket.model_param_buffer.dtype),
    )
    bucket.copy_main_weights_to_model_weights()
    assert torch.equal(
        bucket.model_param_buffer,
        bucket.main_param_buffer.to(bucket.model_param_buffer.dtype),
    )

    bucket.install_full_parameters()
    for spec in bucket.specs:
        spec.full_param.grad = torch.ones_like(spec.full_param)
    grad_reduce = mfsdp_buffer.GradReducePipeline([bucket])
    grad_reduce.reduce_gradients(bucket, force=True)
    assert bucket.local_grad_comm_buffer.numel() == bucket.local_numel
    grad_reduce.finish()
    assert bucket.local_grad_comm_buffer.numel() == 0
    assert torch.count_nonzero(bucket.main_grad_buffer) == bucket.local_numel


def test_mfsdp_maxpool_reuses_full_grad_shard_as_reduce_scatter_output():
    """A later microbatch must not allocate a second pool lease for local RS output."""
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
    bucket = mfsdp_buffer.ParamAndGradBuffer(
        model,
        groups=groups,
        config=mfsdp_config.MFSDPConfig(
            main_params_dtype=torch.float32,
            main_grads_dtype=torch.bfloat16,
            grad_comm_dtype=torch.bfloat16,
            fsdp_double_buffer=True,
            maxpool_double_buffer=True,
        ),
        is_expert=lambda _name: False,
        unit_modules=(torch.nn.Linear,),
    ).buckets[0]
    pipeline = mfsdp_buffer.GradReducePipeline(
        [bucket], capacity_owner_bucket_ids=((bucket.bucket_id,),)
    )

    # Model a later microbatch: the persistent local shard already contains an
    # earlier reduction, while this GraphTask needs one fresh unsharded bucket.
    bucket._has_accumulated_grad = True
    bucket.prepare_param_gather()
    bucket.install_full_parameters()
    for spec in bucket.specs:
        spec.full_param.grad = torch.ones_like(spec.full_param)
    pipeline.reduce_gradients(bucket, force=True)

    assert bucket.local_grad_comm_buffer.untyped_storage().data_ptr() == (
        bucket.full_main_grad_buffer.untyped_storage().data_ptr()
    )
    grad_allocator = bucket.allocator.grad_allocator
    assert sum(len(busy) for busy in grad_allocator._busy.values()) == 1
    pipeline.finish()
    assert torch.count_nonzero(bucket.main_grad_buffer) == bucket.local_numel


def test_mfsdp_full_param_installs_mcore_te_lazy_main_grad_protocol():
    """TE must recognize and materialize the released full-param lazy grad."""
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
    bucket = mfsdp_buffer.ParamAndGradBuffer(
        model,
        groups=groups,
        config=mfsdp_config.MFSDPConfig(bucket_size=None),
        is_expert=lambda _name: False,
        unit_modules=(),
    ).buckets[0]
    spec = bucket.specs[0]
    assert spec.full_param.__fsdp_param__ is True
    assert spec.full_param.overwrite_main_grad is True
    assert callable(spec.full_param.get_main_grad)

    # Simulate a released full parameter and the next all-gather/install cycle.
    bucket.release_full_parameters()
    bucket.wait_param_gather()
    spec.full_param.__fsdp_param__ = False
    spec.full_param.overwrite_main_grad = False
    bucket.install_full_parameters()

    assert spec.full_param.__fsdp_param__ is True
    assert spec.full_param.overwrite_main_grad is True
    getter = spec.full_param.get_main_grad
    calls = 0

    def counted_getter():
        nonlocal calls
        calls += 1
        return getter()

    spec.full_param.get_main_grad = counted_getter
    main_grad = spec.full_param.get_main_grad()

    assert calls == 1
    assert main_grad is spec.full_param.main_grad
    assert main_grad.shape == spec.full_param.shape
    assert main_grad.dtype is torch.float32
    assert main_grad.is_contiguous()
    assert main_grad.numel() == spec.full_param.numel()


def test_mfsdp_sharded_grad_hook_overwrites_each_microbatch_staging_buffer():
    """Match MCore's data-distributed _grad_acc copy_ semantics."""
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
    bucket = mfsdp_buffer.ParamAndGradBuffer(
        model,
        groups=groups,
        config=mfsdp_config.MFSDPConfig(bucket_size=None),
        is_expert=lambda _name: False,
        unit_modules=(),
    ).buckets[0]
    spec = bucket.specs[0]
    staged = spec.full_param.get_main_grad()
    staged.fill_(7.0)
    expected = torch.full_like(spec.full_param, 3.0)
    spec.full_param.grad = expected

    bucket._make_grad_ready_hook(spec)(spec.full_param)

    assert torch.equal(spec.full_param.main_grad, expected)
    assert spec.full_param.grad_added_to_main_grad is False


def test_mfsdp_nested_units_use_outermost_owner():
    """Match MCore's first-match assignment for nested FSDP units."""

    class InnerUnit(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(4, 4, bias=False)

        def forward(self, value):
            return self.linear(value)

    class OuterUnit(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.inner = InnerUnit()

        def forward(self, value):
            return self.inner(value)

    model = torch.nn.Sequential(OuterUnit())
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
    param_buffer = mfsdp_buffer.ParamAndGradBuffer(
        model,
        groups=groups,
        config=mfsdp_config.MFSDPConfig(bucket_size=None),
        is_expert=lambda _name: False,
        unit_modules=(OuterUnit, InnerUnit),
    )

    outer = model[0]
    assert set(param_buffer.owners) == {id(outer)}
    assert all(bucket.is_fsdp_unit for bucket in param_buffer.buckets)


def test_mfsdp_grad_ready_hook_does_not_release_unit_parameters_early():
    """Only the FSDP-unit post-backward boundary may release full weights."""
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
    bucket = mfsdp_buffer.ParamAndGradBuffer(
        model,
        groups=groups,
        config=mfsdp_config.MFSDPConfig(bucket_size=None),
        is_expert=lambda _name: False,
        unit_modules=(torch.nn.Linear,),
    ).buckets[0]
    bucket.release_full_parameters()
    bucket.wait_param_gather()
    bucket.install_full_parameters()
    spec = bucket.specs[0]
    spec.full_param.grad = torch.ones_like(spec.full_param)

    bucket._make_grad_ready_hook(spec)(spec.full_param)

    assert bucket._full_ready is True
    assert bucket._full_lease is not None
    assert spec.full_param.data.numel() == spec.numel

    bucket.release_full_parameters()
    bucket.discard_full_parameter_views()
    assert bucket._full_ready is False
    assert bucket._full_lease is None
    assert spec.full_param.data.numel() == 0


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
        (torch.randn(5, 4), torch.randn(5, 2)),
        (torch.randn(4, 4), torch.randn(4, 2)),
    ]

    reference_optimizer.zero_grad()
    candidate_optimizer.zero_grad()
    for microbatch_idx, (value, target) in enumerate(microbatches):
        reference_loss = torch.nn.functional.mse_loss(reference(value), target)
        (reference_loss / len(microbatches)).backward()

        candidate_output = chunks[0](value)
        candidate_loss = torch.nn.functional.mse_loss(candidate_output, target)
        if microbatch_idx == len(microbatches) - 1:
            candidate_optimizer.grad_sync_enabled = True
        (candidate_loss / len(microbatches)).backward()
        # MCore's root post-backward callback resets the RS pipeline on every
        # backward: sharded accumulation persists, unsharded staging does not.
        assert chunks[0].grad_reduce_pipeline._pending == []
        assert all(
            bucket._full_main_grad_lease is None
            for bucket in chunks[0].param_sync.buckets
        )
        assert any(
            torch.count_nonzero(bucket.main_grad_buffer)
            for bucket in chunks[0].param_sync.buckets
        )

    candidate_optimizer.finish_grad_sync()

    assert all(
        bucket._full_main_grad_lease is None for bucket in chunks[0].param_sync.buckets
    )

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


def test_mfsdp_get_main_grad_lazily_materializes_only_the_requested_bucket():
    chunk, _optimizer = _single_rank_mfsdp_stack()
    first, second = chunk.param_sync.buckets[:2]
    first_param = first.specs[0].full_param

    assert first._full_main_grad_lease is None
    assert second._full_main_grad_lease is None

    main_grad = first_param.get_main_grad()

    assert main_grad.shape == first.specs[0].shape
    assert first._full_main_grad_lease is not None
    assert second._full_main_grad_lease is None
    assert main_grad.untyped_storage().data_ptr() == (
        first.full_main_grad_buffer.untyped_storage().data_ptr()
    )


def test_mfsdp_double_buffer_nonunit_fallback_does_not_consume_a_unit_slot():
    """MCore excludes non-unit dynamic buckets from its two-unit pool limit."""
    chunk, _optimizer = _single_rank_mfsdp_stack(
        override_optimizer_config={"bucket_size": 4, "nccl_ub": True}
    )
    pipeline = chunk.grad_reduce_pipeline

    assert chunk.mfsdp_config.nccl_ub is True
    assert chunk.mfsdp_config.fsdp_double_buffer is True
    buckets = chunk.param_sync.buckets
    assert len({bucket.full_numel for bucket in buckets}) > 1
    assert len(buckets) > 2
    first, second = buckets[:2]
    third = next(
        bucket for bucket in buckets[2:] if bucket.full_numel != first.full_numel
    )

    # Model two completed-but-not-yet-retired unit reductions. The differently
    # sized root/output bucket is not an FSDP unit and uses MCore's dynamic
    # fallback allocator, so it must not evict either pooled unit.
    first.get_main_grad(first.specs[0])
    second.get_main_grad(second.specs[0])
    first._grad_reduce_launched = True
    second._grad_reduce_launched = True
    first._grad_reduce_finished = False
    second._grad_reduce_finished = False
    pipeline._pending = [(first, first.full_numel), (second, second.full_numel)]
    pipeline._pending_elements = sum(
        element_count for _bucket, element_count in pipeline._pending
    )
    assert len(pipeline._pending) == 2
    assert third.config.fsdp_double_buffer is True
    assert third.before_main_grad_allocate is not None
    assert third.before_main_grad_allocate.__self__ is pipeline

    third.get_main_grad(third.specs[0])

    assert first._full_main_grad_lease is not None
    assert third._full_main_grad_lease is not None
    assert len(pipeline._pending) == 2


def test_mfsdp_grad_double_buffer_limit_counts_fsdp_units_not_buckets(monkeypatch):
    """Port MCore's unit-ID queue limit for multi-bucket hybrid FSDP units."""

    class Bucket:
        def __init__(self, bucket_id: int):
            self.bucket_id = bucket_id
            self.device = torch.device("cpu")
            self.full_numel = 10
            self.unpadded_numel = 10
            self.is_fsdp_unit = True
            self.requires_grad = True
            self.config = SimpleNamespace(
                fsdp_double_buffer=True,
                suggested_communication_unit_size=None,
            )
            self.grad_ready_callback = None
            self.before_main_grad_allocate = None

    buckets = [Bucket(bucket_id) for bucket_id in range(4)]
    pipeline = mfsdp_buffer.GradReducePipeline(
        buckets,
        owner_bucket_ids=((0,), (1,), (2,), (3,)),
        capacity_owner_bucket_ids=((0, 1), (2,), (3,)),
    )
    retired = []

    def retire_oldest():
        retired.append(pipeline._pending.pop(0)[0].bucket_id)

    monkeypatch.setattr(pipeline, "_retire_oldest", retire_oldest)

    # Two queued buckets from one enclosing unit plus an incoming second unit
    # are still only two live FSDP units. Bucket-counting incorrectly retired 0.
    pipeline._pending = [(buckets[0], 10), (buckets[1], 10)]
    pipeline._enforce_double_buffer_limit(buckets[2])
    assert retired == []
    assert [bucket.bucket_id for bucket, _ in pipeline._pending] == [0, 1]

    # MCore scans newest-to-oldest. A third unit retires every queued bucket
    # belonging to the oldest unit, leaving the newer unit in flight.
    pipeline._pending.append((buckets[2], 10))
    pipeline._enforce_double_buffer_limit(buckets[3])
    assert retired == [0, 1]
    assert [bucket.bucket_id for bucket, _ in pipeline._pending] == [2]


def test_mfsdp_wrapper_aborts_busy_double_buffer_slots_after_forward_exception():
    """The production wrapper, not a test-only call, owns force teardown."""
    chunk, _optimizer = _single_rank_mfsdp_stack(
        override_optimizer_config={"fsdp_double_buffer": True}
    )

    def fail_after_mfsdp_acquire(_module, _inputs):
        raise RuntimeError("intentional forward failure")

    handle = chunk.module.unit1.register_forward_pre_hook(fail_after_mfsdp_acquire)
    try:
        with pytest.raises(RuntimeError, match="intentional forward failure"):
            chunk(torch.randn(3, 4))
    finally:
        handle.remove()

    allocator = chunk.param_and_grad_buffer.allocator
    assert getattr(allocator, "_slots", {}) == {}
    assert getattr(allocator, "_busy", {}) == {}


def test_mfsdp_wrapper_preserves_primary_forward_failure_when_cleanup_fails(
    monkeypatch,
):
    """Secondary teardown errors must never replace the module failure."""
    chunk, _optimizer = _single_rank_mfsdp_stack(
        override_optimizer_config={"fsdp_double_buffer": True}
    )

    def fail_after_mfsdp_acquire(_module, _inputs):
        raise RuntimeError("PRIMARY_FORWARD_FAILURE")

    def fail_cleanup():
        raise RuntimeError("SECONDARY_CLEANUP_FAILURE")

    handle = chunk.module.unit1.register_forward_pre_hook(fail_after_mfsdp_acquire)
    monkeypatch.setattr(chunk.param_sync, "abort", fail_cleanup)
    monkeypatch.setattr(chunk.param_sync, "end_forward", fail_cleanup)
    try:
        with pytest.raises(RuntimeError, match="PRIMARY_FORWARD_FAILURE"):
            chunk(torch.randn(3, 4))
    finally:
        handle.remove()


def test_mfsdp_abort_waits_inflight_work_before_shared_allocator_clear(monkeypatch):
    """Abort drains work and releases leases before clearing a shared allocator."""
    chunk, _optimizer = _single_rank_mfsdp_stack(
        override_optimizer_config={"fsdp_double_buffer": True}
    )
    bucket = chunk.param_sync.buckets[0]
    bucket.get_main_grad(bucket.specs[0])

    class InflightWork:
        waited = False

        def wait(self):
            self.waited = True

    class InflightEvent:
        synchronized = False

        def synchronize(self):
            self.synchronized = True

    work = InflightWork()
    event = InflightEvent()
    bucket._grad_reduce_work = work
    bucket._grad_reduce_event = event
    allocator = chunk.param_and_grad_buffer.allocator
    release_cached = allocator.release_cached

    def assert_drained_before_clear(*, force=False):
        assert force is True
        assert work.waited is True
        assert event.synchronized is True
        assert bucket._full_main_grad_lease is None
        return release_cached(force=force)

    monkeypatch.setattr(allocator, "release_cached", assert_drained_before_clear)
    chunk.param_sync.abort()

    assert bucket._grad_reduce_work is None
    assert bucket._grad_reduce_event is None
    assert getattr(allocator, "_slots", {}) == {}
    assert getattr(allocator, "_busy", {}) == {}


def test_mfsdp_grad_reduce_waits_at_launch_using_mcore_element_capacity():
    class PendingBucket:
        def __init__(self, full_numel: int):
            self.device = torch.device("cpu")
            self.full_numel = full_numel
            self.policy = SimpleNamespace(grad_comm_dtype=torch.float32)
            self.config = SimpleNamespace(
                fsdp_double_buffer=False, suggested_communication_unit_size=32
            )
            self.world_size = 1
            self.process_group = None
            self.main_grad_buffer = torch.zeros(full_numel)
            self.launched = False
            self.waited = False
            self.grad_ready_callback = None

        def prepare_grad_reduce(self, *, force=False):
            self.launched = True
            return torch.empty(self.full_numel), torch.ones(self.full_numel)

        def mark_grad_reduce_launched(self, work, completion_event=None):
            self.work = work
            self.completion_event = completion_event

        def wait_grad_reduce(self):
            self.waited = True

    first = PendingBucket(8)
    second = PendingBucket(32)
    third = PendingBucket(32)
    pipeline = mfsdp_buffer.GradReducePipeline([first, second, third])

    pipeline.reduce_gradients(first, force=True)
    pipeline.reduce_gradients(second, force=True)
    pipeline.reduce_gradients(third, force=True)

    # MCore measures the already-queued elements immediately before launching
    # the next reduction.  The queue may therefore exceed the suggestion by
    # the just-launched bucket; it is trimmed at the following launch.
    assert pipeline._pending_elements == 32 + 32
    assert pipeline._pending_capacity_elements == 32
    assert first.waited is True
    assert second.waited is False


def test_mfsdp_mcore_capacity_counts_only_explicit_units_and_unpadded_elements():
    buckets = [
        SimpleNamespace(
            is_fsdp_unit=False, unpadded_numel=9_000_000_000, full_numel=9_000_000_008
        ),
        SimpleNamespace(
            is_fsdp_unit=True, unpadded_numel=600_000_000, full_numel=600_000_008
        ),
        SimpleNamespace(
            is_fsdp_unit=True, unpadded_numel=800_000_000, full_numel=800_000_008
        ),
    ]

    capacity = mfsdp_buffer._resolve_suggested_communication_unit_size(
        buckets, ((0,), (1,), (2,)), explicit=None
    )

    # MCore computes 2 * average explicit FSDP-unit elements.  The much larger
    # non-unit root and DP-padding are both excluded from that average.
    assert capacity == 1_400_000_000


def test_mfsdp_all_gather_launches_whole_owner_before_wait_and_prefetches_by_budget():
    events = []

    class GatherBucket:
        def __init__(self, bucket_id: int):
            self.bucket_id = bucket_id
            self.device = torch.device("cpu")
            self.full_numel = 10
            self.config = SimpleNamespace(
                overlap_param_gather=True,
                fsdp_double_buffer=False,
                suggested_communication_unit_size=30,
            )

        def install_full_parameters(self):
            events.append(("install", self.bucket_id))

    buckets = [GatherBucket(index) for index in range(4)]
    pipeline = mfsdp_buffer.AllGatherPipeline(buckets)
    pipeline.async_bucket_gather = lambda bucket_id, bwd=False: events.append(
        ("launch_bwd" if bwd else "launch_fwd", bucket_id)
    )
    pipeline.wait_bucket_ready = lambda bucket_id, bwd=False: events.append(
        ("wait_bwd" if bwd else "wait_fwd", bucket_id)
    )

    pipeline.acquire_forward((0, 1))

    assert events[:4] == [
        ("launch_fwd", 0),
        ("launch_fwd", 1),
        ("launch_fwd", 2),
        ("launch_fwd", 3),
    ]
    assert events[4:] == [
        ("wait_fwd", 0),
        ("install", 0),
        ("wait_fwd", 1),
        ("install", 1),
    ]


def test_mfsdp_double_buffer_does_not_prefetch_from_non_unit_request():
    events = []

    class GatherBucket:
        def __init__(self, bucket_id: int, *, is_fsdp_unit: bool):
            self.bucket_id = bucket_id
            self.device = torch.device("cpu")
            self.full_numel = 10
            self.unpadded_numel = 10
            self.is_fsdp_unit = is_fsdp_unit
            self.config = SimpleNamespace(
                overlap_param_gather=True,
                fsdp_double_buffer=True,
                suggested_communication_unit_size=30,
            )

        def install_full_parameters(self):
            events.append(("install", self.bucket_id))

    buckets = [
        GatherBucket(0, is_fsdp_unit=False),
        GatherBucket(1, is_fsdp_unit=True),
        GatherBucket(2, is_fsdp_unit=True),
    ]
    pipeline = mfsdp_buffer.AllGatherPipeline(
        buckets,
        ((0,), (1,), (2,)),
        fsdp_unit_bucket_ids=((1,), (2,)),
    )
    pipeline.async_bucket_gather = lambda bucket_id, bwd=False: events.append(
        ("launch_bwd" if bwd else "launch_fwd", bucket_id)
    )
    pipeline.wait_bucket_ready = lambda bucket_id, bwd=False: events.append(
        ("wait_bwd" if bwd else "wait_fwd", bucket_id)
    )

    pipeline.acquire_forward((0,))

    # MCore skips prefetch when a double-buffer request contains no FSDP unit;
    # otherwise a root-owned norm can occupy head/embed/layer slots out of order.
    assert events == [("launch_fwd", 0), ("wait_fwd", 0), ("install", 0)]


def test_mfsdp_double_buffer_counts_complete_owner_not_collective_groups():
    events = []

    class GatherBucket:
        def __init__(self, bucket_id: int):
            self.bucket_id = bucket_id
            self.device = torch.device("cpu")
            self.full_numel = 10
            self.unpadded_numel = 10
            self.is_fsdp_unit = True
            self.config = SimpleNamespace(
                overlap_param_gather=True,
                fsdp_double_buffer=True,
                suggested_communication_unit_size=30,
            )

        def install_full_parameters(self):
            events.append(("install", self.bucket_id))

    buckets = [GatherBucket(index) for index in range(3)]
    pipeline = mfsdp_buffer.AllGatherPipeline(
        buckets,
        # Dense and expert collectives are separate, but buckets 0 and 1 are
        # offsets of one enclosing FSDP unit and consume one lifecycle slot.
        ((0,), (1,), (2,)),
        fsdp_unit_bucket_ids=((0, 1), (2,)),
    )
    pipeline.async_bucket_gather = lambda bucket_id, bwd=False: events.append(
        ("launch_bwd" if bwd else "launch_fwd", bucket_id)
    )
    pipeline.wait_bucket_ready = lambda bucket_id, bwd=False: events.append(
        ("wait_bwd" if bwd else "wait_fwd", bucket_id)
    )

    pipeline.acquire_forward((0, 1))

    assert events[:3] == [
        ("launch_fwd", 0),
        ("launch_fwd", 1),
        ("launch_fwd", 2),
    ]
    assert events[3:] == [
        ("wait_fwd", 0),
        ("install", 0),
        ("wait_fwd", 1),
        ("install", 1),
    ]


def test_mfsdp_all_gather_wait_is_bucket_scoped_not_whole_stream():
    class GatherBucket:
        bucket_id = 0
        device = torch.device("cpu")
        full_numel = 10
        config = SimpleNamespace(
            overlap_param_gather=True,
            fsdp_double_buffer=False,
            suggested_communication_unit_size=30,
        )
        _full_ready = False
        _param_gather_work = object()

        def wait_param_gather(self):
            self.waited = True

    bucket = GatherBucket()
    pipeline = mfsdp_buffer.AllGatherPipeline([bucket])
    pipeline.comm_stream.wait_for_current = lambda: (_ for _ in ()).throw(
        AssertionError("waited for unrelated prefetched collectives")
    )

    pipeline.wait_bucket_ready(0)

    assert bucket.waited is True


def test_mfsdp_grad_reduce_waits_for_owner_bucket_group_in_bucket_id_order():
    config = SimpleNamespace(
        suggested_communication_unit_size=32, fsdp_double_buffer=False
    )

    def fake_bucket(bucket_id):
        bucket = SimpleNamespace(
            bucket_id=bucket_id,
            specs=[object()],
            _grad_ready_ids=set(),
            device=torch.device("cpu"),
            full_numel=4,
            config=config,
        )
        bucket.start_microbatch = lambda: None
        return bucket

    buckets = [fake_bucket(0), fake_bucket(1)]
    pipeline = mfsdp_buffer.GradReducePipeline(buckets, owner_bucket_ids=[(1, 0)])
    launched = []
    pipeline._reduce_bucket_group = lambda bucket_ids, force=False: launched.extend(
        (bucket_id, force) for bucket_id in bucket_ids
    )

    buckets[1]._grad_ready_ids.add(id(buckets[1].specs[0]))
    pipeline.reduce_gradients(buckets[1])
    assert launched == []

    buckets[0]._grad_ready_ids.add(id(buckets[0].specs[0]))
    pipeline.reduce_gradients(buckets[0])
    assert launched == [(0, False), (1, False)]

    # Readiness from the preceding microbatch must not make a partial current
    # group launch early while the prior reductions are still queued.
    pipeline.start_microbatch()
    launched.clear()
    pipeline.reduce_gradients(buckets[0])
    assert launched == []
    pipeline.reduce_gradients(buckets[1])
    assert launched == [(0, False), (1, False)]


def test_mfsdp_begin_backward_is_once_only_until_post_backward_reset():
    pipeline = mfsdp_buffer.CommunicationPipelines([])
    events = []
    pipeline.grad_reduce.start_microbatch = lambda: events.append("grad")
    pipeline.grad_reduce.finish_microbatch = lambda: events.append("root")
    pipeline.all_gather.begin_backward = lambda: events.append("param")

    assert pipeline.begin_backward() is True
    assert pipeline.begin_backward() is False
    assert events == ["grad", "param"]

    pipeline.end_backward()
    assert events == ["grad", "param", "root"]
    assert pipeline.begin_backward() is True
    assert events == ["grad", "param", "root", "grad", "param"]


def test_mfsdp_root_callback_drains_grad_reduce_before_next_forward():
    chunk, optimizer = _single_rank_mfsdp_stack()
    value = torch.randn(3, 4)
    target = torch.randn(3, 2)

    optimizer.zero_grad()
    torch.nn.functional.mse_loss(chunk(value), target).backward()
    assert not chunk.grad_reduce_pipeline._pending
    grad_before = tuple(
        bucket.main_grad_buffer.clone() for bucket in chunk.param_sync.buckets
    )
    assert any(torch.count_nonzero(grad) for grad in grad_before)

    chunk(value)

    assert not chunk.grad_reduce_pipeline._pending
    assert all(
        torch.equal(bucket.main_grad_buffer, expected)
        for bucket, expected in zip(chunk.param_sync.buckets, grad_before, strict=True)
    )
    optimizer.finish_grad_sync()


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
    root_buckets = [
        bucket for bucket in chunks[0].param_sync.buckets if not bucket.is_fsdp_unit
    ]
    assert root_buckets
    assert all(bucket._full_lease is None for bucket in root_buckets)
    assert all(bucket._full_ready for bucket in root_buckets)
    assert all(
        group == (bucket.bucket_id,)
        for bucket in root_buckets
        for group in chunks[0].param_and_grad_buffer.collective_groups
        if bucket.bucket_id in group
    )

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
        param.dtype is torch.bfloat16
        for _name, param in chunks[0].stream_full_parameters()
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
    for bucket in chunks[0].param_sync.buckets:
        assert torch.equal(
            bucket.model_param_buffer,
            bucket.main_param_buffer.to(bucket.model_param_buffer.dtype),
        )
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
        override_optimizer_config={
            "mfsdp_sharding_strategy": "optim_grads_params",
            "gradient_accumulation_fusion": True,
        },
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
    # Match MCore's _grad_acc contract: this is a per-backward signal from TE,
    # not optimizer-step state, so the post-accumulate hook must consume and
    # clear it before the next microbatch.
    assert spec.full_param.grad_added_to_main_grad is False
    assert spec.full_param.grad is None
    assert spec.full_param.main_grad.numel() == 0
    first_main_grad = bucket.main_grad_buffer.view(spec.shape).detach().clone()
    assert not torch.equal(
        first_main_grad, first_main_grad.to(torch.bfloat16).to(torch.float32)
    )
    second_value = torch.randn(8, 4, dtype=torch.bfloat16)
    chunk(second_value).float().sum().backward()
    assert spec.full_param.grad is None
    second_expected = second_value.float().sum(dim=0).repeat(4, 1)
    assert spec.full_param.main_grad.numel() == 0

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
    assert spec.full_param.main_grad.numel() == 0
    assert torch.equal(bucket.main_grad_buffer.view(4, 4), torch.full((4, 4), 256.0))

    chunk(torch.ones(1, 4, dtype=torch.bfloat16)).sum().backward()
    assert spec.full_param.grad is None
    expected = torch.full((4, 4), 257.0)
    assert spec.full_param.main_grad.numel() == 0

    optimizer.finish_grad_sync()
    assert torch.equal(bucket.main_grad_buffer.view_as(expected), expected)
    assert not torch.equal(
        bucket.main_grad_buffer,
        bucket.main_grad_buffer.to(torch.bfloat16).to(torch.float32),
    )
    consumed_grad = _optimizer_params(optimizer)[0].grad
    assert consumed_grad is not None
    assert consumed_grad.dtype is torch.float32


def test_mfsdp_rejects_bf16_main_grad_without_precision_aware_optimizer():
    with pytest.raises(ValueError, match="must be FP32"):
        mfsdp_config.build_mfsdp_config(
            SimpleNamespace(
                use_precision_aware_optimizer=False,
                override_optimizer_config={
                    "megatron_fsdp_main_grads_dtype": "bf16",
                    "megatron_fsdp_grad_comm_dtype": "bf16",
                },
            )
        )


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
        streamed_params = dict(chunks[0].stream_full_parameters())
        assert reference_params.keys() == streamed_params.keys()
        for name, reference_param in reference_params.items():
            assert torch.equal(reference_param, streamed_params[name]), name
        idle_params = dict(chunks[0].named_parameters())
        optimizer_params = dict(chunks[0].named_optimizer_parameters())
        assert idle_params.keys() == {
            f"module.{name}" for name in optimizer_params
        }
        for name, optimizer_param in optimizer_params.items():
            assert idle_params[f"module.{name}"] is optimizer_param
    finally:
        dist.destroy_process_group()


def test_mfsdp_cpu_two_rank_gloo_matches_replicated_reference(tmp_path):
    init_file = str(tmp_path / "mfsdp-gloo-init")
    mp.spawn(_run_mfsdp_gloo_parity, args=(2, init_file), nprocs=2, join=True)


def test_mfsdp_dcp_same_topology_restores_flat_master_and_adam_state(tmp_path):
    init_file = str(tmp_path / "mfsdp-dcp-init")
    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=0, world_size=1
    )
    try:
        torch.manual_seed(765)
        model = _GlooModel()
        ps = SimpleNamespace(
            dp_cp_group=dist.group.WORLD,
            dp_group=dist.group.WORLD,
            ep_dp_group=dist.group.WORLD,
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
            [model],
            engine_cfg=SimpleNamespace(
                parallel=ParallelConfig(tp=1, ep=1, etp=1, pp=1, vpp=1, cp=1),
                optimizer=opt,
            ),
            ps=ps,
            is_expert=lambda _name: False,
            fsdp_unit_modules=(_GlooUnit,),
        )
        value = torch.randn(3, 4)
        target = torch.randn(3, 2)
        optimizer.zero_grad()
        torch.nn.functional.mse_loss(chunks[0](value), target).backward()
        optimizer.finish_grad_sync()
        assert optimizer.step()[0]

        # TransformerEngine FusedAdam owns the update counter outside each
        # per-parameter state mapping, so its mapping contains moments but no
        # ``step`` key. DCP must preserve that schema instead of inventing a
        # torch.optim-style per-parameter counter during load.
        torch_optimizer = optimizer._inner_optimizer.optimizer
        for state in torch_optimizer.state.values():
            state.pop("step", None)
        expected_group_steps = []
        for index, group in enumerate(torch_optimizer.param_groups):
            # TE FusedAdam lazily creates this group-level counter on its first
            # update. A newly built load target has no such key, so the DCP
            # destination template must request it explicitly.
            group["step"] = 17 + index
            expected_group_steps.append(group["step"])

        expected_params = {
            name: param.detach().clone()
            for name, param in chunks[0].stream_full_parameters()
        }
        expected_state = {
            spec.name: {
                key: tensor.detach().clone()
                for key, tensor in torch_optimizer.state[spec.shard_param].items()
                if torch.is_tensor(tensor)
            }
            for bucket in chunks[0].param_sync.buckets
            for spec in bucket.specs
            if spec.shard_param is not None and spec.full_param.requires_grad
        }

        checkpoint_dcp.save_training_checkpoint(
            chunks[0], optimizer, 11, str(tmp_path / "checkpoint"), use_dcp=True
        )
        for group in torch_optimizer.param_groups:
            group.pop("step")
        with torch.no_grad():
            for bucket in chunks[0].param_sync.buckets:
                bucket.main_param_buffer.add_(9.0)
            for state in torch_optimizer.state.values():
                for tensor in state.values():
                    if torch.is_tensor(tensor):
                        tensor.zero_()

        step = checkpoint_dcp.load_training_checkpoint(
            chunks[0], optimizer, str(tmp_path / "checkpoint"), use_dcp=True
        )

        assert step == 11
        assert [group.get("step") for group in torch_optimizer.param_groups] == (
            expected_group_steps
        )
        actual_params = dict(chunks[0].stream_full_parameters())
        assert actual_params.keys() == expected_params.keys()
        for name, expected in expected_params.items():
            assert torch.equal(actual_params[name], expected), name
        for bucket in chunks[0].param_sync.buckets:
            for spec in bucket.specs:
                if spec.shard_param is None or not spec.full_param.requires_grad:
                    continue
                actual = torch_optimizer.state[spec.shard_param]
                assert actual.keys() == expected_state[spec.name].keys()
                for key, expected in expected_state[spec.name].items():
                    assert torch.equal(actual[key], expected), (spec.name, key)
    finally:
        dist.destroy_process_group()


def _dcp_test_stack(world_size: int, rank: int):
    torch.manual_seed(2468)
    model = _GlooModel()
    ps = SimpleNamespace(
        dp_cp_group=dist.group.WORLD,
        dp_group=dist.group.WORLD,
        ep_dp_group=dist.group.WORLD,
        ep_group=None,
        etp_group=None,
        tp_group=None,
        pp_group=None,
        dp_cp_size=world_size,
        expert_dp_size=world_size,
        dp_rank=rank,
        pp_rank=0,
        cp_rank=0,
        tp_rank=0,
        ep_rank=0,
        etp_rank=0,
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
        [model],
        engine_cfg=SimpleNamespace(
            parallel=ParallelConfig(tp=1, ep=1, etp=1, pp=1, vpp=1, cp=1), optimizer=opt
        ),
        ps=ps,
        is_expert=lambda _name: False,
        fsdp_unit_modules=(_GlooUnit,),
    )
    return model, chunks[0], optimizer, ps, opt


def test_mfsdp_dcp_model_only_round_trip_does_not_require_optimizer(tmp_path):
    init_file = str(tmp_path / "mfsdp-dcp-model-only-init")
    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=0, world_size=1
    )
    try:
        _model, chunk, _optimizer, ps, _opt = _dcp_test_stack(1, 0)
        expected = {
            name: param.detach().clone()
            for name, param in chunk.stream_full_parameters()
        }
        checkpoint_dcp.save_training_checkpoint(
            chunk,
            None,
            23,
            str(tmp_path / "model-only-checkpoint"),
            ps=ps,
            use_dcp=True,
            save_model=True,
            save_optimizer=False,
            save_rng=False,
        )
        with torch.no_grad():
            for bucket in chunk.param_sync.buckets:
                bucket.main_param_buffer.add_(1.0)
                bucket.copy_main_weights_to_model_weights()
                bucket.invalidate_full_parameters()

        step = checkpoint_dcp.load_training_checkpoint(
            chunk,
            None,
            str(tmp_path / "model-only-checkpoint"),
            ps=ps,
            use_dcp=True,
            load_model=True,
            load_optimizer=False,
            load_rng=False,
        )
        actual = dict(chunk.stream_full_parameters())
        assert step == 23
        assert actual.keys() == expected.keys()
        for name, tensor in actual.items():
            assert torch.equal(tensor, expected[name]), name
    finally:
        dist.destroy_process_group()


def test_mfsdp_dcp_optimizer_only_template_omits_model_weights(
    tmp_path, monkeypatch
):
    init_file = str(tmp_path / "mfsdp-dcp-optimizer-only-init")
    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=0, world_size=1
    )
    try:
        _model, chunk, optimizer, ps, _opt = _dcp_test_stack(1, 0)
        optimizer.zero_grad()
        torch.nn.functional.mse_loss(
            chunk(torch.randn(3, 4)), torch.randn(3, 2)
        ).backward()
        optimizer.finish_grad_sync()
        assert optimizer.step()[0]

        captured = {}

        def capture_save(state_dict, *, checkpoint_id):
            captured["state_dict"] = state_dict
            captured["checkpoint_id"] = checkpoint_id

        monkeypatch.setattr(checkpoint_dcp.dcp, "save", capture_save)
        checkpoint_dcp.save_training_checkpoint(
            chunk,
            optimizer,
            29,
            str(tmp_path / "optimizer-only-checkpoint"),
            ps=ps,
            use_dcp=True,
            save_model=False,
            save_optimizer=True,
            save_rng=False,
        )
        bucket_states = [
            bucket_state
            for domain in captured["state_dict"]["mfsdp"]["domains"].values()
            for chunk_state in domain.values()
            for bucket_state in chunk_state.values()
        ]
        assert bucket_states
        assert all("main_param" not in state for state in bucket_states)
        assert all("exp_avg" in state for state in bucket_states)
    finally:
        dist.destroy_process_group()


def _run_mfsdp_dcp_source(rank: int, world_size: int, init_file: str, root: str):
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=world_size
    )
    try:
        _model, chunk, optimizer, ps, _opt = _dcp_test_stack(world_size, rank)
        torch.manual_seed(3000 + rank)
        value = torch.randn(3, 4)
        target = torch.randn(3, 2)
        optimizer.zero_grad()
        torch.nn.functional.mse_loss(chunk(value), target).backward()
        optimizer.finish_grad_sync()
        assert optimizer.step()[0]

        torch_optimizer = optimizer._inner_optimizer.optimizer
        expected = {
            "params": {
                name: param.detach().cpu().clone()
                for name, param in chunk.stream_full_parameters()
            },
            "optimizer": {},
        }
        for bucket in chunk.param_sync.buckets:
            exp_avg = checkpoint_dcp._mfsdp_gather_padded_bucket(
                bucket,
                checkpoint_dcp._mfsdp_pack_optimizer_tensor(
                    bucket, torch_optimizer, "exp_avg"
                ),
            )
            exp_avg_sq = checkpoint_dcp._mfsdp_gather_padded_bucket(
                bucket,
                checkpoint_dcp._mfsdp_pack_optimizer_tensor(
                    bucket, torch_optimizer, "exp_avg_sq"
                ),
            )
            (
                _initialized,
                _step_present,
                steps,
            ) = checkpoint_dcp._mfsdp_optimizer_metadata(bucket, torch_optimizer)
            for index, spec in enumerate(bucket.specs):
                expected["optimizer"][spec.name] = {
                    "step": steps[index].cpu().clone(),
                    "exp_avg": exp_avg.narrow(0, spec.full_offset, spec.numel)
                    .view(spec.shape)
                    .cpu()
                    .clone(),
                    "exp_avg_sq": exp_avg_sq.narrow(0, spec.full_offset, spec.numel)
                    .view(spec.shape)
                    .cpu()
                    .clone(),
                }
        if rank == 0:
            torch.save(expected, os.path.join(root, "expected.pt"))
        checkpoint_dcp.save_training_checkpoint(
            chunk,
            optimizer,
            13,
            os.path.join(root, "checkpoint"),
            ps=ps,
            use_dcp=True,
            save_rng=False,
        )
    finally:
        dist.destroy_process_group()


def _run_mfsdp_dcp_target(rank: int, world_size: int, init_file: str, root: str):
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=world_size
    )
    try:
        _model, chunk, optimizer, ps, opt = _dcp_test_stack(world_size, rank)
        torch_optimizer = optimizer._inner_optimizer.optimizer
        for group in torch_optimizer.param_groups:
            group["lr"] = 0.25
        step = checkpoint_dcp.load_training_checkpoint(
            chunk,
            optimizer,
            os.path.join(root, "checkpoint"),
            ps=ps,
            use_dcp=True,
            load_rng=False,
        )
        assert step == 13
        assert all(group["lr"] == opt.lr for group in torch_optimizer.param_groups)
        expected = torch.load(
            os.path.join(root, "expected.pt"), map_location="cpu", weights_only=False
        )
        actual = dict(chunk.stream_full_parameters())
        assert actual.keys() == expected["params"].keys()
        for name, tensor in actual.items():
            assert torch.equal(tensor.cpu(), expected["params"][name]), name

        reference = _GlooModel()
        reference.load_state_dict(expected["params"])
        reference_optimizer = torch.optim.AdamW(
            reference.parameters(),
            lr=opt.lr,
            betas=(opt.adam_beta1, opt.adam_beta2),
            eps=opt.adam_eps,
            weight_decay=opt.weight_decay,
            foreach=False,
        )
        for name, param in reference.named_parameters():
            saved = expected["optimizer"][name]
            reference_optimizer.state[param] = {
                "step": saved["step"].to(dtype=torch.float32),
                "exp_avg": saved["exp_avg"].clone(),
                "exp_avg_sq": saved["exp_avg_sq"].clone(),
            }

        torch.manual_seed(4000 + rank)
        value = torch.randn(4, 4)
        target = torch.randn(4, 2)
        optimizer.zero_grad()
        reference_optimizer.zero_grad()
        torch.nn.functional.mse_loss(chunk(value), target).backward()
        torch.nn.functional.mse_loss(reference(value), target).backward()
        for param in reference.parameters():
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
            param.grad.div_(world_size)
        optimizer.finish_grad_sync()
        assert optimizer.step()[0]
        reference_optimizer.step()

        actual_after = dict(chunk.stream_full_parameters())
        for name, reference_param in reference.named_parameters():
            assert torch.equal(actual_after[name].cpu(), reference_param), name
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize(("source_world", "target_world"), [(2, 4), (4, 2)])
def test_mfsdp_dcp_cross_dp_reshard_matches_next_step(
    tmp_path, source_world, target_world
):
    root = str(tmp_path)
    mp.spawn(
        _run_mfsdp_dcp_source,
        args=(source_world, str(tmp_path / "source-init"), root),
        nprocs=source_world,
        join=True,
    )
    mp.spawn(
        _run_mfsdp_dcp_target,
        args=(target_world, str(tmp_path / "target-init"), root),
        nprocs=target_world,
        join=True,
    )


def test_mfsdp_local_checkpoint_resume_matches_uninterrupted_next_step(tmp_path):
    init_file = str(tmp_path / "mfsdp-local-checkpoint-init")
    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=0, world_size=1
    )
    try:
        _model, uninterrupted, uninterrupted_optimizer, _ps, _opt = _dcp_test_stack(
            1, 0
        )
        first_value = torch.randn(3, 4)
        first_target = torch.randn(3, 2)
        next_value = torch.randn(4, 4)
        next_target = torch.randn(4, 2)

        uninterrupted_optimizer.zero_grad()
        torch.nn.functional.mse_loss(
            uninterrupted(first_value), first_target
        ).backward()
        uninterrupted_optimizer.finish_grad_sync()
        assert uninterrupted_optimizer.step()[0]

        checkpoint_dcp.save_training_checkpoint(
            uninterrupted,
            uninterrupted_optimizer,
            17,
            str(tmp_path / "local-checkpoint"),
            use_dcp=False,
            save_rng=False,
        )

        uninterrupted_optimizer.zero_grad()
        torch.nn.functional.mse_loss(uninterrupted(next_value), next_target).backward()
        uninterrupted_optimizer.finish_grad_sync()
        uninterrupted_result = uninterrupted_optimizer.step()
        expected = {
            name: param.detach().clone()
            for name, param in uninterrupted.stream_full_parameters()
        }

        _model, resumed, resumed_optimizer, _ps, _opt = _dcp_test_stack(1, 0)
        step = checkpoint_dcp.load_training_checkpoint(
            resumed,
            resumed_optimizer,
            str(tmp_path / "local-checkpoint"),
            use_dcp=False,
            load_rng=False,
        )
        assert step == 17

        resumed_optimizer.zero_grad()
        torch.nn.functional.mse_loss(resumed(next_value), next_target).backward()
        resumed_optimizer.finish_grad_sync()
        resumed_result = resumed_optimizer.step()
        actual = dict(resumed.stream_full_parameters())

        assert resumed_result == uninterrupted_result
        assert actual.keys() == expected.keys()
        for name, tensor in actual.items():
            assert torch.equal(tensor, expected[name]), name
    finally:
        dist.destroy_process_group()
