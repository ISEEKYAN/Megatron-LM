# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from __future__ import annotations

import os
import subprocess
from pathlib import Path


LITE_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = LITE_ROOT / "examples/verl/scripts/run_deepseek_v4_hopper_smoke.sh"


def _fake_environment(tmp_path: Path) -> dict[str, str]:
    sm90_site = tmp_path / "sm90/site-packages"
    thin_site = tmp_path / "thin/site-packages"
    megatron_root = tmp_path / "Megatron-LM"
    verl_root = tmp_path / "verl"
    shim = tmp_path / "thin/abi_shim/libvllm_torch212_abi_shim.so"

    for path in (
        sm90_site,
        thin_site,
        megatron_root / "megatron/core",
        verl_root,
        shim.parent,
    ):
        path.mkdir(parents=True, exist_ok=True)
    shim.touch()

    for package in (
        "flash_mla-1.0.0.dist-info",
        "nvidia_cudnn_frontend-1.25.0.dist-info",
        "nvidia_cutlass_dsl-4.5.2.dist-info",
    ):
        (sm90_site / package).mkdir()
    for package in (
        "apache_tvm_ffi-0.1.9.dist-info",
        "tilelang-0.1.9.dist-info",
        "transformers-5.12.1.dist-info",
        "vllm-0.20.2.dist-info",
    ):
        (thin_site / package).mkdir()

    env = os.environ.copy()
    env.update(
        {
            "DS4_VLLM_SHIM": str(shim),
            "DS4_VLLM_SITE": str(thin_site),
            "MEGATRON_ROOT": str(megatron_root),
            "MLITE_SM90_SITE": str(sm90_site),
            "VERL_ROOT": str(verl_root),
        }
    )
    return env


def _run(
    tmp_path: Path, mode: str, **overrides: str
) -> subprocess.CompletedProcess[str]:
    env = _fake_environment(tmp_path)
    env.update(overrides)
    return subprocess.run(
        ["bash", str(SCRIPT), mode],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )


def test_training_check_uses_sm90_stack_without_rollout_thin_overlay(tmp_path):
    result = _run(tmp_path, "training", CHECK_ONLY="1")

    assert result.returncode == 0, result.stderr
    assert "DS4_HOPPER_ENV_CHECK_PASSED mode=training" in result.stdout
    pythonpath = next(
        line for line in result.stdout.splitlines() if line.startswith("PYTHONPATH=")
    )
    assert str(tmp_path / "sm90/site-packages") in pythonpath
    assert str(tmp_path / "thin/site-packages") not in pythonpath
    assert str(tmp_path / "Megatron-LM") in pythonpath


def test_rollout_check_prepends_thin_overlay_and_abi_shim(tmp_path):
    result = _run(tmp_path, "rollout-probe", CHECK_ONLY="1")

    assert result.returncode == 0, result.stderr
    assert "DS4_HOPPER_ENV_CHECK_PASSED mode=rollout-probe" in result.stdout
    pythonpath = next(
        line for line in result.stdout.splitlines() if line.startswith("PYTHONPATH=")
    )
    assert pythonpath.split("=", 1)[1].split(":", 1)[0] == str(
        tmp_path / "thin/site-packages"
    )
    assert (
        f"LD_PRELOAD={tmp_path}/thin/abi_shim/libvllm_torch212_abi_shim.so"
        in result.stdout
    )


def test_training_dry_run_targets_only_deepseek_v4_runtime_smoke(tmp_path):
    result = _run(tmp_path, "training", DRY_RUN="1", NPROC_PER_NODE="2")

    assert result.returncode == 0, result.stderr
    assert "torchrun --standalone --nproc_per_node=2" in result.stdout
    assert "test_mlite_engine_cp_smoke.py" in result.stdout
    assert "-k deepseek_v4" in result.stdout


def test_rollout_dry_run_uses_checked_probe_module(tmp_path):
    result = _run(tmp_path, "rollout-probe", DRY_RUN="1")

    assert result.returncode == 0, result.stderr
    assert "python" in result.stdout
    assert "deepseek_v4_hopper_vllm_probe.py" in result.stdout


def test_rollout_check_rejects_missing_vllm_overlay(tmp_path):
    result = _run(
        tmp_path,
        "rollout-probe",
        CHECK_ONLY="1",
        DS4_VLLM_SITE=str(tmp_path / "missing"),
    )

    assert result.returncode != 0
    assert "DS4_VLLM_SITE is not a directory" in result.stderr


def test_training_check_rejects_missing_megatron_core(tmp_path):
    result = _run(
        tmp_path,
        "training",
        CHECK_ONLY="1",
        MEGATRON_ROOT=str(tmp_path / "missing"),
    )

    assert result.returncode != 0
    assert "MEGATRON_ROOT is not a directory" in result.stderr
