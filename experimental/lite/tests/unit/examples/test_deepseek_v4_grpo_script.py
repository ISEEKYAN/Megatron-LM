# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


LITE_ROOT = Path(__file__).resolve().parents[3]
EXAMPLE_ROOT = LITE_ROOT / "examples/verl"
RUNNER = EXAMPLE_ROOT / "scripts/run_deepseek_v4_gsm8k_grpo.sh"
SBATCH = EXAMPLE_ROOT / "slurm/run_ds4_gsm8k_grpo.sbatch"


def test_sitecustomize_skips_application_patches_for_ray_infrastructure() -> None:
    env = {
        "PATH": os.environ["PATH"],
        "PYTHONPATH": str(EXAMPLE_ROOT),
        "VERL_MLITE_SKIP_RUNTIME_PATCHES": "1",
    }
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; assert 'verl_mlite.compat' not in sys.modules",
        ],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_math_smoke_rows_follow_verl_gsm8k_schema() -> None:
    from examples.verl.prepare_math_smoke_data import build_rows

    rows = build_rows(size=8, split="train", index_offset=0)

    assert len(rows) == 8
    assert len({row["extra_info"]["index"] for row in rows}) == 8
    for row in rows:
        assert row["data_source"] == "openai/gsm8k"
        assert row["ability"] == "math"
        assert row["prompt"][0]["role"] == "user"
        assert 'after "####"' in row["prompt"][0]["content"]
        assert row["reward_model"]["style"] == "rule"
        assert row["reward_model"]["ground_truth"].lstrip("-").isdigit()


def test_ds4_grpo_dry_run_freezes_fused_fsdp_and_fp8_resync(tmp_path: Path) -> None:
    env = {
        **os.environ,
        "MODEL_PATH": str(tmp_path / "mixed-model"),
        "TRAIN_FILES": str(tmp_path / "train.parquet"),
        "VAL_FILES": str(tmp_path / "test.parquet"),
        "OUTPUT_ROOT": str(tmp_path / "output"),
        "CKPT_DIR": str(tmp_path / "checkpoints"),
        "NNODES": "8",
        "NGPUS_PER_NODE": "8",
        "ACTOR_PP": "2",
        "ACTOR_EP": "8",
        "ACTOR_CP": "2",
        "ROLLOUT_TP": "16",
        "TOTAL_TRAINING_STEPS": "3",
        "COMPOSE_ONLY": "1",
        "DRY_RUN": "1",
    }
    result = subprocess.run(
        ["bash", str(RUNNER)],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    command = result.stdout
    assert "actor_rollout_ref.actor.engine.impl_cfg.optimizer=fsdp2" in command
    assert "actor_rollout_ref.actor.engine.pp=2" in command
    assert "actor_rollout_ref.actor.engine.ep=8" in command
    assert "actor_rollout_ref.actor.engine.cp=2" in command
    assert "actor_rollout_ref.actor.engine.attention_backend_override=fused" in command
    assert "actor_rollout_ref.actor.engine.impl_cfg.recompute=full" in command
    assert "actor_rollout_ref.actor.engine.impl_cfg.mtp_enable_train=True" in command
    assert "actor_rollout_ref.model.use_fused_kernels=True" in command
    assert "actor_rollout_ref.model.custom_chat_template=" in command
    runner = RUNNER.read_text()
    assert "<｜User｜>" in runner
    assert "<｜Assistant｜>" in runner
    assert "+actor_rollout_ref.actor.engine.cross_entropy_fusion=True" in command
    assert "actor_rollout_ref.actor.engine.resync_format=vllm_checkpoint" in command
    assert "actor_rollout_ref.actor.engine.resync_config.expert_dtype=fp8" in command
    assert "VllmCheckpointWorkerExtension" in command
    assert "hf_overrides.expert_dtype=fp8" in command
    assert "actor_rollout_ref.rollout.tensor_model_parallel_size=16" in command
    assert "actor_rollout_ref.rollout.load_format=dummy" in command
    assert "trainer.total_training_steps=3" in command
    assert "trainer.use_legacy_worker_impl=disable" in command
    assert "--cfg job" in command


def test_ds4_grpo_sbatch_is_multinode_resumable_and_fail_closed() -> None:
    script = SBATCH.read_text()

    assert "#SBATCH --nodes=16" in script
    assert "#SBATCH --gres=gpu:8" in script
    assert "#SBATCH --time=14-00:00:00" in script
    assert 'git -C "${MLITE_SRC}" rev-parse HEAD' in script
    assert 'srun --nodes="${SLURM_NNODES}"' in script
    assert "MASTER_ADDR=$(hostname -i)" in script
    assert "MASTER_ADDR=${MASTER_ADDR%% *}" in script
    assert (
        "RAY_CLI=(env VERL_MLITE_SKIP_RUNTIME_PATCHES=1 "
        "python -m ray.scripts.scripts)"
    ) in script
    assert '"${RAY_CLI[@]}" --help' in script
    assert '"${RAY_CLI[@]}" start --head' in script
    assert '"${RAY_CLI[@]}" start --address="${MASTER_ADDR}:${RAY_PORT}"' in script
    assert "RAY_raylet_start_wait_time_s" in script
    assert '--temp-dir="${RAY_TEMP_DIR}"' in script
    assert 'RAY_TEMP_DIR="/tmp/ds4-grpo-${SLURM_JOB_ID}-ray"' in script
    assert '"${RUN_ROOT}/ray-logs/node-${NODE_RANK}"' in script
    assert '"${RAY_CLI[@]}" job submit' in script
    assert "RUNTIME_ENV_JSON=$(VERL_MLITE_SKIP_RUNTIME_PATCHES=1 python" in script
    assert 'env["VERL_MLITE_SKIP_RUNTIME_PATCHES"] = "0"' in script
    assert 'RAY_CLUSTER_NODES ${#alive[@]}' not in script
    assert "RAY_CLUSTER_NODES" in script
    assert "PHASE1_STEPS" in script
    assert "TOTAL_STEPS" in script
    assert "CONFIG_ONLY" in script
    assert "RAY_ONLY" in script
    assert "DS4_GRPO_CONFIG_COMPOSE_PASSED" in script
    assert '"use_legacy_worker_impl: disable"' in script
    assert '"custom_chat_template: null"' in script
    assert '"<｜User｜>"' in script
    assert '"<｜Assistant｜>"' in script
    assert "DS4_RAY_CLUSTER_PASSED" in script
    assert "DS4_RAY_JOB_RUNTIME_ENV_PASSED" in script
    assert script.index("RUNTIME_ENV_JSON=$(") < script.index(
        'if [[ "${RAY_ONLY}" == "1" ]]'
    )
    assert 'VERL_MLITE_VLLM_SITE="${DS4_VLLM_SITE}"' in script
    assert 'VERL_MLITE_VLLM_LD_PRELOAD="${DS4_VLLM_SHIM}"' in script
    assert 'export PYTHONPATH="${MLITE_SM90_SITE}' in script
    assert ':${DS4_VLLM_SITE}:' in script
    assert 'export LD_PRELOAD="${DS4_VLLM_SHIM}"' in script
    assert 'export HF_HOME="${RUN_ROOT}/hf-cache"' in script
    assert 'export HF_DATASETS_CACHE="${HF_HOME}/datasets"' in script
    assert '"HF_HOME",' in script
    assert '"HF_DATASETS_CACHE",' in script
    assert "export CC=/usr/bin/gcc" in script
    assert "export CXX=/usr/bin/g++" in script
    assert "Error in sitecustomize" in script
    assert "Failed to import Triton kernels" in script
    assert "Traceback" in script
    assert '> "${RUN_ROOT}/resolved-config.yaml" 2>&1' in script
    assert 'RESUME_MODE="${resume_mode}"' in script
    assert 'run_phase phase1 "${PHASE1_STEPS}" disable' in script
    assert 'run_phase resume "${TOTAL_STEPS}" auto' in script
    assert "DS4_GRPO_PHASE1_COMPLETE" in script
    assert "DS4_GRPO_RESUME_COMPLETE" in script
    assert "validate_grpo_metrics.py" in script
    assert "metrics-report.json" in script
    assert "DS4_GRPO_RUN_COMPLETE" in script
    assert "DRY_RUN=0" in script


def test_ds4_grpo_metrics_validator_requires_real_update_and_throughput(
    tmp_path: Path,
) -> None:
    from examples.verl.validate_grpo_metrics import validate_metrics

    phase1 = tmp_path / "phase1.jsonl"
    resume = tmp_path / "resume.jsonl"
    records = []
    for step in range(1, 7):
        record = {
            "step": step,
            "data": {
                "training/global_step": step,
                "critic/score/mean": 0.25 + step / 100,
                "critic/score/min": 0.0,
                "critic/score/max": 1.0,
                "actor/pg_loss": -0.01 * step,
                "actor/grad_norm": 0.1 * step,
                "perf/time_per_step": 2.0,
                "perf/throughput": 3.0 + step,
            },
        }
        records.append(json.dumps(record))
    phase1.write_text("\n".join(records[:3]) + "\n")
    resume.write_text("\n".join(records[3:]) + "\n")

    report = validate_metrics([phase1, resume], expected_steps=6)

    assert report["steps"] == [1, 2, 3, 4, 5, 6]
    assert report["max_grad_norm"] == pytest.approx(0.6)
    assert report["min_throughput"] == 4.0
