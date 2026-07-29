# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest


LITE_ROOT = Path(__file__).parents[3]
SCRIPTS = LITE_ROOT / "examples" / "verl" / "scripts"


def _dry_run(script: str, tmp_path: Path, omp_num_threads: str | None) -> str:
    model_path = tmp_path / "model"
    model_path.mkdir()
    (model_path / "config.json").write_text(
        json.dumps({"o_groups": 8}),
        encoding="utf-8",
    )

    env = os.environ.copy()
    env.update(
        {
            "DRY_RUN": "1",
            "MODEL_PATH": str(model_path),
            "OUTPUT_ROOT": str(tmp_path / "output"),
            "TRAIN_FILES": str(tmp_path / "train.parquet"),
            "VAL_FILES": str(tmp_path / "val.parquet"),
        }
    )
    if omp_num_threads is None:
        env.pop("OMP_NUM_THREADS", None)
    else:
        env["OMP_NUM_THREADS"] = omp_num_threads

    return subprocess.run(
        ["bash", str(SCRIPTS / script)],
        check=True,
        capture_output=True,
        env=env,
        text=True,
    ).stdout


@pytest.mark.parametrize(
    "script",
    [
        "run_qwen3moe_gsm8k_grpo.sh",
        "run_deepseek_v4_dapo.sh",
    ],
)
@pytest.mark.parametrize(
    ("configured", "expected"),
    [
        (None, "1"),
        ("3", "3"),
    ],
)
def test_ray_runtime_env_matches_torchrun_omp_contract(
    script: str,
    configured: str | None,
    expected: str,
    tmp_path: Path,
) -> None:
    command = _dry_run(script, tmp_path, configured)

    assert (
        f"+ray_kwargs.ray_init.runtime_env.env_vars.OMP_NUM_THREADS=\\'{expected}\\'"
        in command
    )
