# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from __future__ import annotations

import os
import subprocess
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from megatron.lite.runtime import create_runtime
from megatron.lite.runtime.backends.mlite.config import MegatronLiteConfig
from megatron.lite.runtime.backends.mlite.runtime import (
    MegatronLiteRuntime,
    _apply_attention_backend_env,
    _build_impl_cfg,
    _pipeline_callbacks,
)
from megatron.lite.runtime.contracts.config import OptimizerConfig, ParallelConfig, RuntimeConfig
from megatron.lite.runtime.contracts.handle import ModelHandle
from megatron.lite.runtime.contracts.loss import LossContext, get_loss_context, use_loss_context

pytestmark = pytest.mark.mlite


def test_runtime_returns_loss_separately_from_microbatch_metrics():
    model = nn.Linear(1, 1, bias=False)
    handle = ModelHandle(
        model=model,
        parallel_state=types.SimpleNamespace(pp_size=1),
        _extras={"forward_step": lambda module, batch: {"value": module(batch["x"])}},
    )
    batches = iter([{"x": torch.ones(1, 1), "micro": i} for i in range(2)])
    result = MegatronLiteRuntime.__new__(MegatronLiteRuntime).forward_backward(
        handle,
        batches,
        lambda out, batch: (out["value"].sum(), {"micro": batch["micro"]}),
        num_microbatches=2,
    )

    assert result.metrics == {"micro": [0, 1]}
    assert result.model_output.loss is not None


def test_pipeline_callbacks_accept_wrapped_and_presplit_context():
    context = LossContext(source_batch="source")
    seen = []
    forward, loss = _pipeline_callbacks(
        lambda _model, batch: seen.append((batch, get_loss_context()))
        or {"loss": torch.tensor(1.0)},
        lambda out, batch, ctx: (out["loss"], {"batch": batch, "source": ctx.source_batch}),
    )

    output = forward(None, ("wrapped", context))
    with use_loss_context(context):
        forward(None, "presplit")
    _, metrics = loss(output, "presplit", context)

    assert seen == [("wrapped", context), ("presplit", context)]
    assert metrics == {"batch": "presplit", "source": "source"}


def test_runtime_config_defaults_to_mlite_backend():
    cfg = RuntimeConfig()

    assert cfg.backend == "mlite"
    assert cfg.hf_path == ""
    assert isinstance(cfg.backend_cfg, dict)


def test_runtime_config_accepts_mlite_backend_cfg():
    cfg = RuntimeConfig(
        backend="mlite",
        hf_path="/models/Qwen3",
        backend_cfg={"model_name": "qwen3", "impl": "lite", "tp": 2, "ep": 4},
    )

    assert cfg.backend == "mlite"
    assert cfg.backend_cfg["model_name"] == "qwen3"
    assert cfg.backend_cfg["tp"] == 2


def test_mlite_config_defaults_and_parallel_fields():
    cfg = MegatronLiteConfig(
        model_name="qwen3_moe", parallel=ParallelConfig(tp=4, etp=1, ep=8, pp=2, vpp=2, cp=2)
    )

    assert cfg.model_name == "qwen3_moe"
    assert cfg.impl == "lite"
    assert cfg.parallel.tp == 4
    assert cfg.parallel.ep == 8
    assert cfg.parallel.pp == 2
    assert cfg.parallel.cp == 2


def test_mlite_config_impl_cfg_optimizer_and_load_gate():
    hook = lambda cfg: cfg  # noqa: E731
    cfg = MegatronLiteConfig(
        model_name="qwen3_moe",
        impl_cfg={"recompute": "full", "use_deepep": True},
        optimizer=OptimizerConfig(lr=1e-4, weight_decay=0.1, adam_beta1=0.9),
        load_hf_weights=False,
        model_config_hook=hook,
    )

    assert cfg.impl_cfg["recompute"] == "full"
    assert cfg.impl_cfg["use_deepep"] is True
    assert cfg.optimizer.lr == 1e-4
    assert cfg.optimizer.adam_beta1 == 0.9
    assert cfg.load_hf_weights is False
    assert cfg.model_config_hook is hook


def test_mlite_config_from_dict_accepts_optimizer_override_config():
    cfg = MegatronLiteConfig.from_dict(
        "/models/Qwen3",
        {
            "optimizer": {
                "override_optimizer_config": {
                    "fsdp2_use_fp32_master": False,
                    "offload_fraction": 1.0,
                }
            }
        },
    )

    assert cfg.optimizer.override_optimizer_config == {
        "fsdp2_use_fp32_master": False,
        "offload_fraction": 1.0,
    }


def test_mlite_config_from_dict_rejects_num_microbatches():
    with pytest.raises(ValueError, match="num_microbatches"):
        MegatronLiteConfig.from_dict(
            "/models/Qwen3", {"model_name": "qwen3", "tp": 4, "num_microbatches": 2}
        )


@dataclass
class _FakeImplConfig:
    parallel: object
    hf_path: str = ""
    optimizer_config: object = None
    attention_backend_override: str | None = None


def test_build_impl_cfg_backfills_top_level_hf_path_and_runtime_fields():
    proto = type("Proto", (), {"ImplConfig": _FakeImplConfig})
    cfg = MegatronLiteConfig(
        model_name="qwen3", hf_path="/models/top", attention_backend_override="local"
    )

    impl_cfg = _build_impl_cfg(proto, cfg)

    assert impl_cfg.parallel is cfg.parallel
    assert impl_cfg.hf_path == "/models/top"
    assert impl_cfg.optimizer_config is cfg.optimizer
    assert impl_cfg.attention_backend_override == "local"


def test_build_impl_cfg_preserves_explicit_impl_hf_path():
    proto = type("Proto", (), {"ImplConfig": _FakeImplConfig})
    cfg = MegatronLiteConfig(
        model_name="qwen3", hf_path="/models/top", impl_cfg={"hf_path": "/models/impl"}
    )

    impl_cfg = _build_impl_cfg(proto, cfg)

    assert impl_cfg.hf_path == "/models/impl"


@dataclass
class _PrecisionImplConfig:
    parallel: object
    precision_implementation: object = None
    precision_parameter_contract: object = None
    precision_coverage: object = None


def test_build_impl_cfg_injects_typed_precision_contract_and_coverage():
    from megatron.lite.primitive.precision import PrecisionCoverage, resolve_precision

    proto = type("Proto", (), {"ImplConfig": _PrecisionImplConfig})
    cfg = MegatronLiteConfig(precision="hopper_blockwise_bf16_weight")
    implementation = resolve_precision(cfg.precision)
    assert implementation is not None
    coverage = PrecisionCoverage(implementation)

    impl_cfg = _build_impl_cfg(
        proto,
        cfg,
        precision_implementation=implementation,
        precision_coverage=coverage,
    )

    assert impl_cfg.precision_implementation is implementation
    assert impl_cfg.precision_parameter_contract is implementation.parameter_contract
    assert impl_cfg.precision_coverage is coverage


def test_build_impl_cfg_rejects_protocol_without_precision_injection_fields():
    from megatron.lite.primitive.precision import PrecisionCoverage, resolve_precision

    proto = type("Proto", (), {"ImplConfig": _FakeImplConfig})
    cfg = MegatronLiteConfig(precision="hopper_blockwise_bf16_weight")
    implementation = resolve_precision(cfg.precision)
    assert implementation is not None

    with pytest.raises(ValueError, match="precision_implementation"):
        _build_impl_cfg(
            proto,
            cfg,
            precision_implementation=implementation,
            precision_coverage=PrecisionCoverage(implementation),
        )


def _patch_cpu_precision_build(monkeypatch, runtime, proto):
    from megatron.lite.primitive.precision import hopper_blockwise

    monkeypatch.setattr(runtime, "_load_protocol", lambda _cfg: proto)
    monkeypatch.setattr(
        "megatron.lite.runtime.backends.mlite.runtime.validate_hopper_environment",
        lambda: SimpleNamespace(compute_capability=(9, 0)),
    )
    monkeypatch.setattr(
        hopper_blockwise, "_build_hopper_blockwise_recipe", lambda: SimpleNamespace()
    )
    monkeypatch.setattr(
        hopper_blockwise, "_validate_hopper_blockwise_recipe", lambda _recipe: None
    )
    monkeypatch.setattr(dist := torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_rank", lambda: 0)
    monkeypatch.setattr(dist, "get_world_size", lambda: 1)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "set_device", lambda _device: None)
    monkeypatch.setattr(torch.cuda, "manual_seed", lambda _seed: None)


def _precision_protocol(*, seal: bool, events: list[str]):
    from megatron.lite.primitive.bundle import ModelBundle
    from megatron.lite.primitive.precision import (
        PrecisionPhase,
        PrimitiveCapability,
        SemanticSite,
        active_precision,
    )

    class Proto:
        ImplConfig = _PrecisionImplConfig

        @staticmethod
        def build_model_config(_source):
            return SimpleNamespace(hidden_size=1)

        @staticmethod
        def build_model(_model_cfg, *, impl_cfg):
            implementation = active_precision(PrecisionPhase.MODEL_INIT)
            assert implementation is impl_cfg.precision_implementation
            assert (
                impl_cfg.precision_parameter_contract
                is implementation.parameter_contract
            )
            projection = object()
            core = object()
            impl_cfg.precision_coverage.require(
                projection,
                SemanticSite.ATTENTION_PROJECTION,
                frozenset({PrimitiveCapability.TE_LINEAR}),
                diagnostic="unit projection",
            )
            impl_cfg.precision_coverage.claim(
                projection,
                SemanticSite.ATTENTION_PROJECTION,
                PrimitiveCapability.TE_LINEAR,
            )
            impl_cfg.precision_coverage.require(
                core, SemanticSite.ATTENTION_CORE, diagnostic="unit attention core"
            )
            if seal:
                impl_cfg.precision_coverage.seal()

            model = nn.Linear(1, 1, bias=False)

            def forward_step(module, batch):
                events.append(active_precision(PrecisionPhase.FORWARD).name)
                return {"logits": module(batch)}

            ps = SimpleNamespace(pp_size=1, tp_size=1, cp_size=1)
            return ModelBundle(
                chunks=[model],
                parallel_state=ps,
                optimizer=None,
                forward_step=forward_step,
            )

    return Proto


def test_runtime_injects_and_seals_precision_on_the_production_build_and_forward_path(
    monkeypatch,
):
    events: list[str] = []
    proto = _precision_protocol(seal=True, events=events)
    cfg = MegatronLiteConfig(
        model_name="unit",
        precision="hopper_blockwise_bf16_weight",
        load_hf_weights=False,
        attention_backend_override=None,
    )
    runtime = MegatronLiteRuntime("", cfg)
    _patch_cpu_precision_build(monkeypatch, runtime, proto)

    handle = runtime.build_model()
    implementation = handle._extras["precision_implementation"]
    assert implementation.name == "hopper_blockwise_bf16_weight"
    assert (
        handle._extras["precision_parameter_contract"]
        is implementation.parameter_contract
    )
    assert (
        handle._extras["precision_coverage"].implementation_name == implementation.name
    )

    result = runtime.forward_backward(
        handle,
        iter([torch.ones(1, 1)]),
        None,
        forward_only=True,
    )

    assert result.model_output.vocab_parallel_logits is not None
    assert events == [implementation.name]


def test_runtime_rejects_unsealed_precision_coverage(monkeypatch):
    proto = _precision_protocol(seal=False, events=[])
    cfg = MegatronLiteConfig(
        model_name="unit",
        precision="hopper_blockwise_bf16_weight",
        load_hf_weights=False,
        attention_backend_override=None,
    )
    runtime = MegatronLiteRuntime("", cfg)
    _patch_cpu_precision_build(monkeypatch, runtime, proto)

    with pytest.raises(RuntimeError, match="not sealed before optimizer"):
        runtime.build_model()


def test_runtime_runs_hopper_preflight_before_distributed_initialization(monkeypatch):
    cfg = MegatronLiteConfig(precision="hopper_blockwise_bf16_weight")
    runtime = MegatronLiteRuntime("", cfg)
    init_process_group = MagicMock()
    monkeypatch.setattr(torch.distributed, "init_process_group", init_process_group)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    monkeypatch.setattr(
        "megatron.lite.runtime.backends.mlite.runtime.validate_hopper_environment",
        MagicMock(side_effect=RuntimeError("preflight rejected environment")),
    )

    with pytest.raises(RuntimeError, match="preflight rejected environment"):
        runtime.build_model()

    init_process_group.assert_not_called()


@pytest.mark.parametrize(
    ("backend", "expected"),
    [
        ("auto", ("1", "1", "1")),
        ("flash", ("1", "0", "0")),
        ("fused", ("0", "1", "0")),
        ("unfused", ("0", "0", "1")),
        ("local", ("0", "0", "0")),
    ],
)
def test_attention_backend_override_sets_expected_env(monkeypatch, backend, expected):
    for name in ("NVTE_FLASH_ATTN", "NVTE_FUSED_ATTN", "NVTE_UNFUSED_ATTN"):
        monkeypatch.delenv(name, raising=False)

    _apply_attention_backend_env(backend, tag="unit")

    assert (
        os.environ["NVTE_FLASH_ATTN"],
        os.environ["NVTE_FUSED_ATTN"],
        os.environ["NVTE_UNFUSED_ATTN"],
    ) == expected


def test_attention_backend_override_rejects_unknown_backend():
    with pytest.raises(ValueError, match="attention_backend_override"):
        _apply_attention_backend_env("invalid", tag="unit")


class HookedOptimizer:
    def __init__(self):
        self.calls: list[str] = []

    def offload_state_to_cpu(self):
        self.calls.append("offload")

    def load_state_to_device(self):
        self.calls.append("load")


def test_runtime_to_prefers_optimizer_specific_offload_hooks():
    optimizer = HookedOptimizer()
    handle = ModelHandle(model=nn.Linear(2, 2), optimizer=optimizer, _extras={"model_chunks": []})
    runtime = MegatronLiteRuntime.__new__(MegatronLiteRuntime)

    runtime.to(handle, "cpu", model=False, optimizer=True, grad=False)
    runtime.to(handle, "cuda", model=False, optimizer=True, grad=False)

    assert optimizer.calls == ["offload", "load"]


class _FakeStorage:
    def __init__(self, size: int):
        self._size = size
        self.resize_calls: list[int] = []

    def size(self):
        return self._size

    def resize_(self, size: int):
        self.resize_calls.append(size)
        self._size = size
        return self


class _FakeBufferData:
    def __init__(self, size: int):
        self._storage = _FakeStorage(size)
        self.cpu_called = False
        self.pinned = False
        self.copied_from = None
        self.copy_non_blocking = None
        self.zero_calls = 0

    @property
    def data(self):
        return self

    def cpu(self):
        self.cpu_called = True
        return self

    def pin_memory(self):
        self.pinned = True
        return self

    def storage(self):
        return self._storage

    def copy_(self, other, *, non_blocking: bool):
        self.copied_from = other
        self.copy_non_blocking = non_blocking
        return self

    def zero_(self):
        self.zero_calls += 1
        return self


class _FakeBuffer:
    def __init__(self):
        self.param_data = _FakeBufferData(3)
        self.grad_data = _FakeBufferData(5)


class _FakeModule:
    def parameters(self):
        return []


class _FakeMegatronDDP:
    def __init__(self):
        self.buffer = _FakeBuffer()
        self.buffers = [self.buffer]
        self.expert_parallel_buffers = []
        self.module = _FakeModule()
        self.to_calls: list[str] = []

    def to(self, device):
        self.to_calls.append(device)
        raise AssertionError("DDP model chunks must use the buffer offload path")


class _FakeMegatronDDPSubclass(_FakeMegatronDDP):
    pass


class _FakeNativeModel:
    def __init__(self):
        self.calls: list[str] = []

    def to(self, device):
        self.calls.append(device)
        return self


def _install_fake_megatron_ddp(monkeypatch) -> None:
    core = types.ModuleType("megatron.core")
    distributed = types.ModuleType("megatron.core.distributed")
    distributed.DistributedDataParallel = _FakeMegatronDDP
    core.distributed = distributed
    monkeypatch.setitem(sys.modules, "megatron.core", core)
    monkeypatch.setitem(sys.modules, "megatron.core.distributed", distributed)


def test_megatron_ddp_detection_accepts_ddp_and_subclasses(monkeypatch):
    from megatron.lite.runtime.megatron_utils import _is_megatron_ddp

    _install_fake_megatron_ddp(monkeypatch)

    assert _is_megatron_ddp(_FakeMegatronDDP()) is True
    assert _is_megatron_ddp(_FakeMegatronDDPSubclass()) is True
    assert _is_megatron_ddp(_FakeNativeModel()) is False


@pytest.mark.parametrize("model_cls", [_FakeMegatronDDP, _FakeMegatronDDPSubclass])
def test_megatron_ddp_model_move_helpers_use_buffer_path(monkeypatch, model_cls):
    from megatron.lite.runtime.megatron_utils import load_model_to_gpu, offload_model_to_cpu

    _install_fake_megatron_ddp(monkeypatch)
    model = model_cls()
    buffer = model.buffer

    offload_model_to_cpu([model])

    assert model.to_calls == []
    assert buffer.param_data.cpu_called is True
    assert buffer.param_data.pinned is True
    assert buffer.param_data_size == 3
    assert buffer.grad_data_size == 5
    assert buffer.param_data.storage().size() == 0
    assert buffer.grad_data.storage().size() == 0

    load_model_to_gpu([model])

    assert model.to_calls == []
    assert buffer.param_data.storage().size() == 3
    assert buffer.grad_data.storage().size() == 5
    assert buffer.param_data.copied_from is buffer.param_data.cpu_data
    assert buffer.param_data.copy_non_blocking is True
    assert buffer.grad_data.zero_calls == 1


def test_native_model_move_helpers_do_not_require_megatron_core(monkeypatch):
    from megatron.lite.runtime.megatron_utils import load_model_to_gpu, offload_model_to_cpu

    monkeypatch.setitem(sys.modules, "megatron.core", None)
    monkeypatch.setitem(sys.modules, "megatron.core.distributed", None)
    model = _FakeNativeModel()

    offload_model_to_cpu([model])
    load_model_to_gpu([model])

    assert model.calls == ["cpu", "cuda"]


def test_model_handle_dp_defaults():
    handle = ModelHandle(model=MagicMock())

    assert handle.dp_rank == 0
    assert handle.dp_size == 1
    assert handle.dp_group is None


def test_model_handle_dp_from_parallel_state():
    ps = MagicMock()
    ps.dp_rank = 3
    ps.dp_size = 8
    ps.dp_group = "fake_group"

    handle = ModelHandle(model=MagicMock(), parallel_state=ps)

    assert handle.dp_rank == 3
    assert handle.dp_size == 8
    assert handle.dp_group == "fake_group"


def test_model_handle_cp_range_and_config_properties():
    cfg = {"tp": 8, "ep": 4}
    default_handle = ModelHandle(model=MagicMock())
    configured_handle = ModelHandle(model=MagicMock(), config=cfg, _extras={"cp_range": (1, 8)})

    assert default_handle.cp_range == (1, 1)
    assert configured_handle.cp_range == (1, 8)
    assert configured_handle.config is cfg


def test_runtime_dispatch_creates_mlite_backend():
    with patch("megatron.lite.runtime.backends.mlite.create") as mock_create:
        backend = MagicMock()
        mock_create.return_value = backend

        runtime = create_runtime(
            RuntimeConfig(
                backend="mlite", hf_path="/models/test", backend_cfg={"model_name": "qwen3"}
            )
        )

    assert runtime is backend
    mock_create.assert_called_once_with("/models/test", {"model_name": "qwen3"})


def test_runtime_dispatch_unknown_backend_raises():
    with pytest.raises(KeyError):
        create_runtime(RuntimeConfig(backend="nonexistent"))


def _run_verl_sft_dry_run(script: Path, tmp_path: Path, **env_overrides: str) -> str:
    env = {
        **os.environ,
        "MODEL_PATH": "/tmp/mlite-model",
        "TRAIN_FILES": "/tmp/mlite-train.parquet",
        "OUTPUT_ROOT": str(tmp_path),
        "DRY_RUN": "1",
        "NUM_GPUS": "1",
        "NPROC_PER_NODE": "1",
        "TP_SIZE": "1",
        "PP_SIZE": "1",
        "CP_SIZE": "1",
        "EP_SIZE": "1",
        "ETP_SIZE": "1",
        **env_overrides,
    }
    completed = subprocess.run([str(script)], env=env, text=True, capture_output=True, check=True)
    return completed.stdout


def test_verl_sft_script_maps_offload_env_to_backend_args(tmp_path):
    script = (
        Path(__file__).resolve().parents[3]
        / "examples"
        / "verl"
        / "scripts"
        / "run_qwen3moe_sft.sh"
    )

    command = _run_verl_sft_dry_run(
        script,
        tmp_path,
        PARAM_OFFLOAD="True",
        OPTIMIZER_OFFLOAD="True",
        OPTIMIZER_STATE_OFFLOAD_FRACTION="0.75",
    )

    assert "engine.param_offload=True" in command
    assert "engine.optimizer_offload=True" in command
    assert "+optim.override_optimizer_config.offload_fraction=0.75" in command
    assert "+optim.override_optimizer_config.use_precision_aware_optimizer=True" in command


def test_verl_sft_script_does_not_emit_optimizer_state_offload_when_disabled(tmp_path):
    script = (
        Path(__file__).resolve().parents[3]
        / "examples"
        / "verl"
        / "scripts"
        / "run_qwen3moe_sft.sh"
    )

    command = _run_verl_sft_dry_run(
        script, tmp_path, PARAM_OFFLOAD="False", OPTIMIZER_OFFLOAD="False"
    )

    assert "engine.param_offload=False" in command
    assert "engine.optimizer_offload=False" in command
    assert "override_optimizer_config.offload_fraction" not in command
