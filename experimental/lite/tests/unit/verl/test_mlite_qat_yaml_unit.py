# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Dependency-free contract for the two QAT owners in MLite's verl config."""

from pathlib import Path

import yaml


def test_default_yaml_keeps_export_and_training_qat_disabled() -> None:
    config_path = (
        Path(__file__).parents[3]
        / "examples"
        / "verl"
        / "verl_mlite"
        / "config"
        / "engine"
        / "mlite.yaml"
    )
    config = yaml.safe_load(config_path.read_text())

    assert config["qat"] == {
        "enable": False,
        "apply_modelopt_fake_quant": False,
        "mode": "mxfp4",
        "group_size": 32,
        "ignore_patterns": [
            "lm_head",
            "embed_tokens",
            "re:.*mlp.gate$",
        ],
    }
    assert config["impl_cfg"]["qat"] == {
        "enabled": False,
        "format": "mxfp4",
        "group_size": 32,
        "symmetric": True,
        "ste_clip": True,
        "ignore_patterns": [],
        "export_mode": "fake",
    }

    # These are deliberately separate schemas: verl owns online export, while
    # MLite owns training parametrization. Both remain opt-in by default.
    assert config["qat"]["enable"] is False
    assert config["impl_cfg"]["qat"]["enabled"] is False
