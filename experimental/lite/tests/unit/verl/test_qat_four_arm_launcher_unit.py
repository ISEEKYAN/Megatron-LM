# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest


_SCRIPT = (
    Path(__file__).parents[3]
    / "examples"
    / "verl"
    / "scripts"
    / "run_qwen3moe_mxfp4_qat.sh"
)


def _render_arm(mode: str, tmp_path: Path) -> list[str]:
    env = os.environ.copy()
    env.update(
        {
            "DRY_RUN": "1",
            "OUTPUT_ROOT": str(tmp_path),
            "TRAIN_FILES": "public/train",
            "VAL_FILES": "public/validation",
        }
    )
    result = subprocess.run(
        ["bash", str(_SCRIPT), "--mode", mode],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    return shlex.split(result.stdout)


def _override(tokens: list[str], name: str) -> str | None:
    prefix = f"{name}="
    return next(
        (token.removeprefix(prefix) for token in tokens if token.startswith(prefix)),
        None,
    )


def _has_verl() -> bool:
    result = subprocess.run(
        [sys.executable, "-c", "import verl"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


@pytest.mark.parametrize(
    ("mode", "rollout_quantization", "training_qat", "router_replay"),
    [
        ("baseline", None, "false", "false"),
        ("qat_off", "mxfp4", "false", "false"),
        ("qat_on", "mxfp4", "true", "false"),
        ("r3", "mxfp4", "true", "true"),
    ],
)
def test_four_arm_launcher_selects_only_the_intended_features(
    mode: str,
    rollout_quantization: str | None,
    training_qat: str,
    router_replay: str,
    tmp_path: Path,
):
    tokens = _render_arm(mode, tmp_path)

    assert (
        _override(tokens, "actor_rollout_ref.rollout.quantization")
        == rollout_quantization
    )
    assert _override(tokens, "actor_rollout_ref.actor.engine.qat.enable") is None
    assert (
        _override(tokens, "actor_rollout_ref.actor.engine.impl_cfg.qat.enabled")
        == training_qat
    )
    assert (
        _override(tokens, "actor_rollout_ref.actor.engine.router_replay_mode")
        == ("R3" if router_replay == "true" else "disabled")
    )
    assert _override(
        tokens, "actor_rollout_ref.rollout.enable_rollout_routing_replay"
    ) == ("true" if mode == "r3" else "false")
    assert _override(tokens, "actor_rollout_ref.actor.engine.pp") == "1"
    assert _override(tokens, "actor_rollout_ref.actor.engine.tp") == "2"
    assert _override(tokens, "actor_rollout_ref.actor.engine.ep") == "8"
    assert _override(tokens, "actor_rollout_ref.actor.engine.cp") == "1"
    assert not any(
        token.startswith("actor_rollout_ref.actor.megatron.") for token in tokens
    )


def test_qat_off_and_qat_on_keep_the_rollout_configuration_identical(tmp_path: Path):
    qat_off = _render_arm("qat_off", tmp_path)
    qat_on = _render_arm("qat_on", tmp_path)

    rollout_prefix = "actor_rollout_ref.rollout."
    assert sorted(
        token for token in qat_off if token.startswith(rollout_prefix)
    ) == sorted(token for token in qat_on if token.startswith(rollout_prefix))


@pytest.mark.parametrize("mode", ["baseline", "qat_off", "qat_on", "r3"])
def test_four_arm_launcher_hydra_composes(mode: str, tmp_path: Path) -> None:
    """Compose the real verl job config so every launcher key is schema-checked."""
    if not _has_verl():
        pytest.skip("requires the target verl runtime for Hydra composition")

    env = os.environ.copy()
    env.update(
        {
            "OUTPUT_ROOT": str(tmp_path / mode),
            "TRAIN_FILES": "public/train.parquet",
            "VAL_FILES": "public/validation.parquet",
        }
    )
    result = subprocess.run(
        ["bash", str(_SCRIPT), "--mode", mode, "--cfg", "job", "--resolve"],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode == 0, result.stderr
