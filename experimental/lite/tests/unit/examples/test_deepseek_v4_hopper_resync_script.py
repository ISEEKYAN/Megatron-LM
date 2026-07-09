# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from __future__ import annotations

from pathlib import Path


LITE_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = LITE_ROOT / "examples/verl/scripts/run_deepseek_v4_hopper_resync.sh"
SBATCH = LITE_ROOT / "examples/verl/slurm/run_ds4_hopper_resync.sbatch"


def test_hopper_resync_runner_freezes_source_and_requires_official_checkpoint() -> None:
    script = SCRIPT.read_text()

    assert "CHECKPOINT_DIR:?set CHECKPOINT_DIR" in script
    assert "MLITE_COMMIT:?set MLITE_COMMIT" in script
    assert 'git -C "${REPO_ROOT}" rev-parse HEAD' in script
    assert "refusing to reuse existing output directory" in script


def test_hopper_resync_runner_covers_training_fp8_proxy_and_rollout_probe() -> None:
    script = SCRIPT.read_text()

    assert '"${HOPPER_SMOKE}" training' in script
    assert 'PYTHONPATH="${LITE_ROOT}:${COMMON_PYTHONPATH}' in script
    assert 'python "${EXAMPLE_ROOT}/ds4_hopper_resync_proxy.py"' in script
    assert '"${HOPPER_SMOKE}" rollout-probe' in script
    assert "CUDA_VISIBLE_DEVICES=0" in script
    assert "--checkpoint" in script
    assert "--output-dir" in script


def test_hopper_resync_runner_requires_non_skip_artifacts() -> None:
    script = SCRIPT.read_text()

    assert "DS4_HOPPER_RESYNC_PROXY_COMPLETE" in script
    assert '"${OUTPUT_DIR}/report.json"' in script
    assert '"${OUTPUT_DIR}/bf16-trained.pt"' in script
    assert '"${OUTPUT_DIR}/fp8-export.pt"' in script
    assert "DS4_HOPPER_TRAIN_RESYNC_PASSED" in script


def test_hopper_resync_sbatch_uses_two_h100_slots_and_the_container_runner() -> None:
    script = SBATCH.read_text()

    assert "#SBATCH --gres=gpu:2" in script
    assert "#SBATCH --time=00:30:00" in script
    assert '--container-image="${BASE_IMAGE}"' in script
    assert (
        "CHECK_ONLY=1 bash experimental/lite/examples/verl/scripts/run_deepseek_v4_hopper_resync.sh"
        in script
    )
    assert (
        "NPROC_PER_NODE=2 bash experimental/lite/examples/verl/scripts/run_deepseek_v4_hopper_resync.sh"
        in script
    )
