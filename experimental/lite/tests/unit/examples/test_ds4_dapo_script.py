from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "examples"
    / "verl"
    / "scripts"
    / "run_deepseek_v4_dapo.sh"
)


@pytest.mark.parametrize(
    ("weight_bits", "enable_r3", "expert_dtype", "router_mode", "rollout_replay"),
    [
        ("4", "True", "fp4", "R3", "True"),
        ("8", "False", "fp8", "disabled", "False"),
    ],
)
def test_ds4_dapo_weight_and_r3_knobs(
    tmp_path: Path,
    weight_bits: str,
    enable_r3: str,
    expert_dtype: str,
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
    assert f"hf_overrides.expert_dtype={expert_dtype}" in command
    assert f"router_replay_mode={router_mode}" in command
    assert f"enable_rollout_routing_replay={rollout_replay}" in command


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("ROLLOUT_WEIGHT_BITS", "16", "must be 4 or 8"),
        ("ENABLE_R3", "maybe", "must be a boolean"),
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
    result = subprocess.run(["bash", str(SCRIPT)], env=env, text=True, capture_output=True)
    assert result.returncode == 2
    assert message in result.stderr
