# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import json
import os
import shlex
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.optional


SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "examples"
    / "verl"
    / "scripts"
    / "run_deepseek_v4_dapo.sh"
)


@pytest.mark.parametrize(
    (
        "weight_bits",
        "enable_r3",
        "expert_dtype",
        "resync_format",
        "moe_backend",
        "scale_fmt",
        "router_mode",
        "rollout_replay",
    ),
    [
        ("4", "True", "fp4", "mxfp4", "marlin", "ue8m0", "R3", "True"),
        (
            "8",
            "False",
            "fp8",
            "block_fp8",
            "flashinfer_cutlass",
            "float32",
            "disabled",
            "False",
        ),
    ],
)
def test_ds4_dapo_weight_and_r3_knobs(
    tmp_path: Path,
    weight_bits: str,
    enable_r3: str,
    expert_dtype: str,
    resync_format: str,
    moe_backend: str,
    scale_fmt: str,
    router_mode: str,
    rollout_replay: str,
) -> None:
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({"o_groups": 8}), encoding="utf-8")
    env = {
        **os.environ,
        "MODEL_PATH": str(model),
        "OUTPUT_ROOT": str(tmp_path / "output"),
        "TRAIN_FILES": str(tmp_path / "train.parquet"),
        "VAL_FILES": str(tmp_path / "val.parquet"),
        "ROLLOUT_WEIGHT_BITS": weight_bits,
        "ENABLE_R3": enable_r3,
        "DRY_RUN": "1",
    }
    result = subprocess.run(
        ["bash", str(SCRIPT)], env=env, text=True, capture_output=True, check=True
    )
    command = result.stdout
    assert f"resync_config.expert_dtype={expert_dtype}" in command
    assert f"engine.resync_format={resync_format}" in command
    assert f"hf_overrides.expert_dtype={expert_dtype}" in command
    assert f"moe_backend={moe_backend}" in command
    assert f"quantization_config.scale_fmt={scale_fmt}" in command
    assert f"router_replay_mode={router_mode}" in command
    assert f"enable_rollout_routing_replay={rollout_replay}" in command
    assert "actor_rollout_ref.actor.engine.impl_cfg.optimizer=fsdp2" in command
    assert "actor_rollout_ref.actor.engine.attention_backend_override=fused" in command
    assert "actor_rollout_ref.rollout.enforce_eager=True" in command
    assert "algorithm.rollout_correction.bypass_mode=False" in command
    assert "data.max_response_length=6144" in command
    assert "actor_rollout_ref.model.custom_chat_template=" in command
    assert "+data.apply_chat_template_kwargs.chat_template=" in command
    assert "data.custom_cls" not in command
    assert "data.label_key" not in command
    assert "data.default_data_source" not in command


@pytest.mark.parametrize(
    ("enable_qat", "expected_enabled", "expected_format"),
    [
        ("False", "false", None),
        ("True", "true", "mxfp4"),
    ],
)
def test_ds4_dapo_qat_toggle_is_explicit_and_matches_mxfp4_rollout(
    tmp_path: Path,
    enable_qat: str,
    expected_enabled: str,
    expected_format: str | None,
) -> None:
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({"o_groups": 8}), encoding="utf-8")
    env = {
        **os.environ,
        "MODEL_PATH": str(model),
        "OUTPUT_ROOT": str(tmp_path / "output"),
        "TRAIN_FILES": str(tmp_path / "train.parquet"),
        "VAL_FILES": str(tmp_path / "val.parquet"),
        "ROLLOUT_WEIGHT_BITS": "4",
        "ENABLE_QAT": enable_qat,
        "DRY_RUN": "1",
    }
    result = subprocess.run(
        ["bash", str(SCRIPT)], env=env, text=True, capture_output=True, check=True
    )
    command = result.stdout
    assert (
        f"actor_rollout_ref.actor.engine.impl_cfg.qat.enabled={expected_enabled}"
        in command
    )
    format_override = "actor_rollout_ref.actor.engine.impl_cfg.qat.format="
    if expected_format is None:
        assert format_override not in command
    else:
        assert f"{format_override}{expected_format}" in command
        assert "actor_rollout_ref.actor.engine.impl_cfg.qat.group_size=32" in command


def test_ds4_dapo_rejects_qat_with_non_mxfp4_rollout(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({"o_groups": 8}), encoding="utf-8")
    env = {
        **os.environ,
        "MODEL_PATH": str(model),
        "OUTPUT_ROOT": str(tmp_path / "output"),
        "TRAIN_FILES": str(tmp_path / "train.parquet"),
        "VAL_FILES": str(tmp_path / "val.parquet"),
        "ROLLOUT_WEIGHT_BITS": "8",
        "ENABLE_QAT": "True",
        "DRY_RUN": "1",
    }
    result = subprocess.run(
        ["bash", str(SCRIPT)], env=env, text=True, capture_output=True
    )
    assert result.returncode == 2
    assert "ENABLE_QAT requires ROLLOUT_WEIGHT_BITS=4" in result.stderr


def test_ds4_dapo_qat_ab_keeps_rollout_configuration_identical(
    tmp_path: Path,
) -> None:
    commands = {}
    for enable_qat in ("False", "True"):
        model = tmp_path / f"model-{enable_qat}"
        model.mkdir()
        (model / "config.json").write_text(
            json.dumps({"o_groups": 8}), encoding="utf-8"
        )
        env = {
            **os.environ,
            "MODEL_PATH": str(model),
            "OUTPUT_ROOT": str(tmp_path / f"output-{enable_qat}"),
            "TRAIN_FILES": str(tmp_path / "train.parquet"),
            "VAL_FILES": str(tmp_path / "val.parquet"),
            "ROLLOUT_WEIGHT_BITS": "4",
            "ENABLE_QAT": enable_qat,
            "DRY_RUN": "1",
        }
        result = subprocess.run(
            ["bash", str(SCRIPT)],
            env=env,
            text=True,
            capture_output=True,
            check=True,
        )
        commands[enable_qat] = sorted(
            token
            for token in shlex.split(result.stdout)
            if token.startswith("actor_rollout_ref.rollout.")
        )

    assert commands["False"] == commands["True"]


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("ROLLOUT_WEIGHT_BITS", "16", "must be 4 or 8"),
        ("ENABLE_R3", "maybe", "must be a boolean"),
        ("ENABLE_QAT", "maybe", "must be a boolean"),
    ],
)
def test_ds4_dapo_rejects_invalid_feature_knob(
    tmp_path: Path, name: str, value: str, message: str
) -> None:
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({"o_groups": 8}), encoding="utf-8")
    env = {
        **os.environ,
        "MODEL_PATH": str(model),
        "OUTPUT_ROOT": str(tmp_path / "output"),
        "DRY_RUN": "1",
        name: value,
    }
    result = subprocess.run(
        ["bash", str(SCRIPT)], env=env, text=True, capture_output=True
    )
    assert result.returncode == 2
    assert message in result.stderr
