# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from __future__ import annotations

import pytest

from megatron.lite.runtime import RuntimeConfig, create_runtime
from megatron.lite.runtime.backends.mlite.config import DebugConfig, MegatronLiteConfig
from megatron.lite.runtime.contracts.config import OptimizerConfig, ParallelConfig

pytestmark = pytest.mark.mlite


def test_mlite_config_defaults_are_stable():
    cfg = MegatronLiteConfig(model_name="qwen3_moe")

    assert cfg.model_name == "qwen3_moe"
    assert cfg.impl == "lite"
    assert cfg.parallel.tp == 1
    assert cfg.parallel.ep == 1
    assert cfg.parallel.pp == 1
    assert cfg.parallel.cp == 1
    assert isinstance(cfg.optimizer, OptimizerConfig)
    assert isinstance(cfg.debug, DebugConfig)
    assert cfg.precision == "bf16"


@pytest.mark.parametrize(
    "precision",
    ["bf16", "hopper_blockwise_bf16_weight"],
)
def test_mlite_config_round_trips_closed_precision_names(precision):
    direct = MegatronLiteConfig(model_name="qwen3_moe", precision=precision)
    parsed = MegatronLiteConfig.from_dict(
        "/models/qwen", {"model_name": "qwen3_moe", "precision": precision}
    )

    assert direct.precision == precision
    assert parsed.precision == precision


def test_mlite_config_rejects_unknown_precision_name_at_construction():
    with pytest.raises(ValueError, match="hopper_blockwise_bf16_weight"):
        MegatronLiteConfig(precision="blockwise")
    with pytest.raises(ValueError, match="hopper_blockwise_bf16_weight"):
        MegatronLiteConfig.from_dict("/models/qwen", {"precision": "fp8"})


def test_mlite_config_rejects_reserved_unimplemented_fp8_weight_profile():
    # The reserved FP8-weight profile is not sold as usable: selecting it fails
    # loud at config construction rather than advertising a working profile.
    with pytest.raises(NotImplementedError, match="not implemented"):
        MegatronLiteConfig(precision="hopper_blockwise_fp8_weight")
    with pytest.raises(NotImplementedError, match="not implemented"):
        MegatronLiteConfig.from_dict(
            "/models/qwen", {"precision": "hopper_blockwise_fp8_weight"}
        )


@pytest.mark.parametrize(
    "override",
    [
        {"fp8_recipe": "blockwise"},
        {"recipe": "blockwise"},
        {"targets": ["attention"]},
        {"weight_dtype": "fp8"},
    ],
)
def test_mlite_config_rejects_ad_hoc_precision_overrides(override):
    with pytest.raises(ValueError, match="closed precision names"):
        MegatronLiteConfig.from_dict(
            "/models/qwen",
            {"precision": "hopper_blockwise_bf16_weight", **override},
        )


@pytest.mark.parametrize("parallel", [ParallelConfig(pp=2), ParallelConfig(cp=2)])
def test_hopper_profiles_reject_unvalidated_parallel_combinations(parallel):
    with pytest.raises(ValueError, match="pp=1 and cp=1"):
        MegatronLiteConfig(precision="hopper_blockwise_bf16_weight", parallel=parallel)


@pytest.mark.parametrize(
    "impl_cfg",
    [
        {"cuda_graph": True},
        {"fp8_param_gather": True},
        {"fp8_communication": True},
        {"mxfp8": True},
    ],
)
def test_hopper_profiles_reject_unvalidated_runtime_features(impl_cfg):
    with pytest.raises(ValueError, match="does not support"):
        MegatronLiteConfig(
            precision="hopper_blockwise_bf16_weight",
            impl_cfg=impl_cfg,
        )


def test_hopper_profiles_reject_fp8_parameter_gather_in_optimizer_overrides():
    with pytest.raises(ValueError, match="does not support fp8_param_gather"):
        MegatronLiteConfig.from_dict(
            "/models/qwen",
            {
                "precision": "hopper_blockwise_bf16_weight",
                "optimizer": {
                    "override_optimizer_config": {"fp8_param_gather": True}
                },
            },
        )


def test_mlite_config_from_dict_preserves_parallel_optimizer_and_impl_cfg():
    cfg = MegatronLiteConfig.from_dict(
        "/models/qwen",
        {
            "model_name": "qwen3_moe",
            "impl": "lite",
            "tp": 2,
            "ep": 4,
            "pp": 2,
            "cp": 2,
            "optimizer": {
                "lr": 1.0e-4,
                "weight_decay": 0.1,
                "adam_beta1": 0.9,
                "offload_fraction": 1.0,
            },
            "impl_cfg": {"attn_impl": "mcore", "moe_impl": "ml"},
            "use_thd": True,
            "precision_aware_opt": True,
        },
    )

    assert cfg.hf_path == "/models/qwen"
    assert cfg.parallel == ParallelConfig(tp=2, etp=None, ep=4, pp=2, vpp=1, cp=2)
    assert cfg.optimizer.lr == 1.0e-4
    assert cfg.optimizer.weight_decay == 0.1
    assert cfg.optimizer.adam_beta1 == 0.9
    assert cfg.optimizer.offload_fraction == 1.0
    assert cfg.impl_cfg["attn_impl"] == "mcore"
    assert cfg.impl_cfg["moe_impl"] == "ml"
    assert cfg.impl_cfg["use_thd"] is True
    assert cfg.impl_cfg["precision_aware_opt"] is True


def test_mlite_config_rejects_num_microbatches_in_backend_config():
    with pytest.raises(ValueError, match="num_microbatches"):
        MegatronLiteConfig.from_dict(
            "/models/qwen", {"model_name": "qwen3_moe", "num_microbatches": 2}
        )


def test_create_runtime_uses_mlite_backend_registry():
    runtime = create_runtime(
        RuntimeConfig(
            backend="mlite",
            hf_path="/models/qwen",
            backend_cfg={"model_name": "qwen3_moe", "load_hf_weights": False},
        )
    )

    assert type(runtime).__name__ == "MegatronLiteRuntime"
    assert runtime.tier == "rl_best"
