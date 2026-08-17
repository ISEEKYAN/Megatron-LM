# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""LoRA config parity at the training-to-vLLM boundary."""

from __future__ import annotations

import ast
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
from megatron.lite.primitive.ckpt.hf_weights import vllm_applied_lora_scaling
from megatron.lite.primitive.modules import lora as lora_module


def _load_production_vllm_peft_builder():
    """Load the production method without importing optional VERL dependencies."""
    engine_path = (
        Path(__file__).parents[3] / "examples/verl/verl_mlite/engine/mlite_engine.py"
    )
    tree = ast.parse(engine_path.read_text())
    engine_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "MegatronLiteEngine"
    )
    method = copy.deepcopy(
        next(
            node
            for node in engine_class.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_build_vllm_peft_config"
        )
    )
    namespace = {
        name: getattr(lora_module, name)
        for name in (
            "LORA_DEFAULT_ALPHA",
            "LORA_DEFAULT_DROPOUT",
            "LORA_DEFAULT_RANK",
            "LORA_DEFAULT_TARGET_MODULES",
            "LORA_DEFAULT_USE_RSLORA",
            "resolve_lora_alpha",
        )
        if hasattr(lora_module, name)
    }
    module = ast.fix_missing_locations(ast.Module([method], []))
    exec(compile(module, engine_path, "exec"), namespace)
    return namespace[method.name]


def _load_production_weight_exporter():
    """Execute the production method without importing optional VERL modules."""
    engine_path = (
        Path(__file__).parents[3] / "examples/verl/verl_mlite/engine/mlite_engine.py"
    )
    tree = ast.parse(engine_path.read_text())
    engine_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "MegatronLiteEngine"
    )
    method = copy.deepcopy(
        next(
            node
            for node in engine_class.body
            if isinstance(node, ast.FunctionDef) and node.name == "get_per_tensor_param"
        )
    )
    namespace = {"qat_export": SimpleNamespace(export_qat_weights=lambda weights, _: weights)}
    module = ast.fix_missing_locations(ast.Module([method], []))
    exec(compile(module, engine_path, "exec"), namespace)
    return namespace[method.name]


def test_model_owned_registry_routes_verl_resync_to_the_named_adapter():
    """The adapter resync contract must not silently export the frozen base."""
    export = _load_production_weight_exporter()
    validated_names = []

    class Registry:
        rank = 8
        alpha = 16
        lora_spec = None

        @staticmethod
        def slot_for(name):
            validated_names.append(name)
            if name != "alpha":
                raise KeyError(f"unknown adapter {name}")
            return 0

    registry = Registry()
    captured_base = []
    captured_adapter = []
    capability_configs = []
    checked_streams = []

    class Runtime:
        @staticmethod
        def multi_lora_registry(handle):
            assert handle is engine.handle
            return registry

        @staticmethod
        def export_weights(handle, **kwargs):
            captured_base.append((handle, kwargs))
            return iter(())

        @staticmethod
        def export_lora_adapter(handle, **kwargs):
            captured_adapter.append((handle, kwargs))
            return iter(())

    engine = SimpleNamespace(
        _require_initialized=lambda: None,
        is_param_offload_enabled=False,
        _initial_sync_cache_cleared=True,
        engine_config=SimpleNamespace(
            resync_format=None,
            resync_config=None,
            export_dtype="bfloat16",
            qat={"enable": False},
            multi_lora_name="alpha",
        ),
        _mlite_config=SimpleNamespace(impl_cfg={}),
        runtime=Runtime(),
        handle=object(),
        _resolve_model_name=lambda: "qwen3_moe",
        _build_vllm_peft_config=lambda config: {"r": config["rank"]},
        _assert_adapter_rollout_contract=lambda config: capability_configs.append(config),
        _checked_adapter_stream=lambda stream: checked_streams.append(stream) or stream,
    )

    # Initial sync must still export the base, without selecting an adapter.
    weights, metadata = export(engine, base_sync_done=False)
    assert list(weights) == []
    assert metadata == {"r": 8}
    assert validated_names == []
    assert captured_base[-1][0] is engine.handle
    assert "multi_lora_name" not in captured_base[-1][1]

    # VERL's real caller does not pass a multi_lora_name to this method. The
    # engine config is the explicit selection surface for the named adapter.
    weights, metadata = export(engine, base_sync_done=True)

    assert list(weights) == []
    assert metadata == {"r": 8}
    assert validated_names == ["alpha"]
    assert capability_configs == [
        {"enabled": True, "rank": 8, "alpha": 16, "use_rslora": False}
    ]
    assert captured_adapter == [
        (engine.handle, {"multi_lora_name": "alpha", "export_dtype": "bfloat16"})
    ]
    assert len(checked_streams) == 1
    with pytest.raises(ValueError, match="disagrees"):
        export(engine, base_sync_done=True, multi_lora_name="bravo")
    engine.engine_config.multi_lora_name = "unknown"
    with pytest.raises(KeyError, match="unknown adapter"):
        export(engine, base_sync_done=True)
    engine.engine_config.multi_lora_name = None
    with pytest.raises(ValueError, match="requires multi_lora_name"):
        export(engine, base_sync_done=True)


@pytest.mark.parametrize(
    "alpha,use_rslora",
    [
        pytest.param(None, False, id="default-alpha-standard"),
        pytest.param(None, True, id="default-alpha-rslora"),
        pytest.param(24, False, id="explicit-alpha-standard"),
        pytest.param(24, True, id="explicit-alpha-rslora"),
    ],
)
def test_single_and_named_adapter_sync_share_the_pin_peft_metadata(alpha, use_rslora):
    """Both exporter branches must give vLLM the same placeholder contract."""

    export = _load_production_weight_exporter()
    builder = _load_production_vllm_peft_builder()

    class Registry:
        rank = 8

        def __init__(self):
            self.alpha = alpha
            self.lora_spec = SimpleNamespace(use_rslora=use_rslora)

        @staticmethod
        def slot_for(name):
            assert name == "alpha"
            return 0

    def metadata_for(*, registry):
        class Runtime:
            @staticmethod
            def multi_lora_registry(_handle):
                return registry

            @staticmethod
            def export_weights(_handle, **_kwargs):
                return iter(())

            @staticmethod
            def export_lora_adapter(_handle, **_kwargs):
                return iter(())

        engine = SimpleNamespace(
            _require_initialized=lambda: None,
            is_param_offload_enabled=False,
            _initial_sync_cache_cleared=True,
            engine_config=SimpleNamespace(
                resync_format=None,
                resync_config=None,
                export_dtype="bfloat16",
                qat={"enable": False},
                multi_lora_name="alpha",
            ),
            _mlite_config=SimpleNamespace(
                impl_cfg={
                    "lora": {
                        "enabled": True,
                        "rank": 8,
                        "alpha": alpha,
                        "use_rslora": use_rslora,
                        "target_modules": ["linear_qkv"],
                        "exclude_modules": ["linear_fc2"],
                    }
                }
            ),
            runtime=Runtime(),
            handle=object(),
            _resolve_model_name=lambda: "qwen3_moe",
            _build_vllm_peft_config=lambda config: builder(None, config),
            _lora_rollout_sync_is_merge=lambda _config: False,
            _assert_adapter_rollout_contract=lambda _config: None,
            _checked_adapter_stream=lambda stream: stream,
        )
        _, metadata = export(engine, base_sync_done=False)
        return metadata

    single = metadata_for(registry=None)
    named = metadata_for(registry=Registry())

    assert named == single
    assert named["target_modules"] == "all-linear"
    assert "exclude_modules" not in named
    assert {key: named[key] for key in ("r", "lora_alpha", "use_rslora")} == {
        "r": 8,
        "lora_alpha": 8 if alpha is None else alpha,
        "use_rslora": use_rslora,
    }


@pytest.mark.parametrize(
    "alpha,use_rslora",
    [
        pytest.param(None, False, id="omitted-alpha"),
        pytest.param(24, False, id="explicit-alpha"),
        pytest.param(24, True, id="rslora"),
    ],
)
def test_alpha_scaling_matches_training_and_production_vllm_builder(alpha, use_rslora):
    """One raw config must resolve identically on both sides."""
    builder = _load_production_vllm_peft_builder()
    spec = lora_module.LoraSpec(
        enabled=True,
        rank=8,
        alpha=alpha,
        use_rslora=use_rslora,
    )

    peft_config = builder(
        None,
        {
            "enabled": True,
            "rank": spec.rank,
            "alpha": spec.alpha,
            "use_rslora": spec.use_rslora,
        },
    )
    rollout_alpha = peft_config["lora_alpha"]
    rollout_scale = vllm_applied_lora_scaling(
        spec.rank,
        rollout_alpha,
        use_rslora=spec.use_rslora,
        packed_moe=False,
    )

    assert rollout_alpha == (spec.rank if alpha is None else alpha)
    assert rollout_scale == spec.scale
    assert peft_config["r"] == spec.rank
    assert peft_config["use_rslora"] is spec.use_rslora
    assert peft_config["lora_dropout"] == spec.dropout
    assert peft_config["target_modules"] == "all-linear"
    assert "exclude_modules" not in peft_config


def test_production_vllm_builder_uses_pin_placeholder_not_target_metadata():
    builder = _load_production_vllm_peft_builder()

    config = builder(
        None,
        {
            "rank": 16,
            "alpha": 32,
            "use_rslora": True,
            "target_modules": ["linear_qkv", "linear_fc1", "linear_qkv", "router"],
            "exclude_modules": ["linear_fc2", "linear_fc2"],
        },
    )

    assert config["target_modules"] == "all-linear"
    assert "exclude_modules" not in config
    assert config["lora_alpha"] == 32
    assert config["use_rslora"] is True
