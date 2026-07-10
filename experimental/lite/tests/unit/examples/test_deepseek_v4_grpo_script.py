# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from __future__ import annotations

import ast
import importlib.util
import json
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest


LITE_ROOT = Path(__file__).resolve().parents[3]
EXAMPLE_ROOT = LITE_ROOT / "examples/verl"
README = EXAMPLE_ROOT / "README.md"
RUNNER = EXAMPLE_ROOT / "scripts/run_deepseek_v4_gsm8k_grpo.sh"
SBATCH = EXAMPLE_ROOT / "slurm/run_ds4_gsm8k_grpo.sbatch"
DATASET_MODULE = EXAMPLE_ROOT / "verl_mlite/dataset.py"
ENGINE_MODULE = EXAMPLE_ROOT / "verl_mlite/engine/mlite_engine.py"
VLLM_LOAD_ONLY_MODULE = EXAMPLE_ROOT / "deepseek_v4_rollout_load_only.py"


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
    model_path = tmp_path / "mixed-model"
    model_path.mkdir()
    (model_path / "config.json").write_text(
        json.dumps({"model_type": "deepseek_v4", "o_groups": 8})
    )
    env = {
        **os.environ,
        "MODEL_PATH": str(model_path),
        "TRAIN_FILES": str(tmp_path / "train.parquet"),
        "VAL_FILES": str(tmp_path / "test.parquet"),
        "OUTPUT_ROOT": str(tmp_path / "output"),
        "HYDRA_RUN_DIR": str(tmp_path / "hydra/phase1"),
        "CKPT_DIR": str(tmp_path / "checkpoints"),
        "NNODES": "8",
        "NGPUS_PER_NODE": "8",
        "ACTOR_PP": "2",
        "ACTOR_EP": "8",
        "ACTOR_CP": "2",
        "ROLLOUT_TP": "8",
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
    assert "data.custom_cls.name=ChatTemplateRLHFDataset" in command
    assert "data.chat_template=" in command
    runner = RUNNER.read_text()
    assert "<｜User｜>" in runner
    assert "<｜Assistant｜>" in runner
    assert "+actor_rollout_ref.actor.engine.cross_entropy_fusion=True" in command
    assert "actor_rollout_ref.actor.engine.resync_format=vllm_checkpoint" in command
    assert "actor_rollout_ref.actor.engine.resync_config.expert_dtype=fp8" in command
    assert "VllmCheckpointWorkerExtension" in command
    assert "hf_overrides.expert_dtype=fp8" in command
    assert "DS4_ROLLOUT_TP_PREFLIGHT_PASSED" in command
    assert "actor_rollout_ref.rollout.tensor_model_parallel_size=8" in command
    assert "actor_rollout_ref.rollout.gpu_memory_utilization=0.60" in command
    assert "actor_rollout_ref.rollout.load_format=dummy" in command
    assert "trainer.total_training_steps=3" in command
    assert "trainer.use_legacy_worker_impl=disable" in command
    assert f"hydra.run.dir={tmp_path / 'hydra/phase1'}" in command
    assert "--cfg job" in command


def test_ds4_grpo_rejects_rollout_tp_larger_than_output_groups(
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "mixed-model"
    model_path.mkdir()
    (model_path / "config.json").write_text(
        json.dumps({"model_type": "deepseek_v4", "o_groups": 8})
    )
    env = {
        **os.environ,
        "MODEL_PATH": str(model_path),
        "TRAIN_FILES": str(tmp_path / "train.parquet"),
        "VAL_FILES": str(tmp_path / "test.parquet"),
        "OUTPUT_ROOT": str(tmp_path / "output"),
        "ROLLOUT_TP": "16",
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

    assert result.returncode != 0
    assert "o_groups=8 must be divisible by rollout_tp=16" in result.stderr


def test_chat_template_dataset_sets_controller_tokenizer_before_loading(
    monkeypatch,
) -> None:
    class StubRLHFDataset:
        def __init__(self, *args, **kwargs) -> None:
            self.base_args = args
            self.base_kwargs = kwargs

    for name in ("verl", "verl.utils", "verl.utils.dataset"):
        module = types.ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
    rl_dataset = types.ModuleType("verl.utils.dataset.rl_dataset")
    rl_dataset.RLHFDataset = StubRLHFDataset
    monkeypatch.setitem(sys.modules, "verl.utils.dataset.rl_dataset", rl_dataset)

    spec = importlib.util.spec_from_file_location("ds4_test_dataset", DATASET_MODULE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    tokenizer = types.SimpleNamespace(chat_template=None)
    processor = types.SimpleNamespace(chat_template=None)
    dataset = module.ChatTemplateRLHFDataset(
        "train.parquet",
        tokenizer,
        {"chat_template": "template"},
        processor=processor,
        max_samples=8,
    )

    assert tokenizer.chat_template == "template"
    assert processor.chat_template == "template"
    assert dataset.base_args == ("train.parquet", tokenizer, {"chat_template": "template"})
    assert dataset.base_kwargs == {"processor": processor, "max_samples": 8}


def test_ds4_grpo_sbatch_is_multinode_resumable_and_fail_closed() -> None:
    script = SBATCH.read_text()
    import_program = script.split("    python - <<'PY'\n", 1)[1].split(
        "\nPY\n  exit 0\nfi", 1
    )[0]

    ast.parse(import_program)
    assert "#SBATCH --nodes=16" in script
    assert "#SBATCH --gres=gpu:8" in script
    directives = script.splitlines()
    assert "#SBATCH --partition=batch" in directives
    assert "#SBATCH --time=04:00:00" in directives
    assert 'git -C "${MLITE_SRC}" rev-parse HEAD' in script
    assert 'srun --export=ALL \\\n    --nodes="${SLURM_NNODES}"' in script
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
    assert "IMPORT_ONLY" in script
    assert "RAY_ONLY" in script
    assert "DS4_GRPO_CONFIG_COMPOSE_PASSED" in script
    assert '"gpu_memory_utilization: 0.6"' in script
    assert "os.path.isdir(cache_root)" in script
    assert "from tilelang import env as tilelang_env" in script
    assert "tilelang_env.TILELANG_CACHE_DIR" in script
    assert "tilelang_env.TILELANG_TMP_DIR" in script
    assert "DS4_TILELANG_CACHE_PREFLIGHT_PASSED" in script
    assert "vllm_cache_root=${VLLM_CACHE_ROOT}" in script
    assert "tilelang_cache_dir=${TILELANG_CACHE_DIR}" in script
    assert '"use_legacy_worker_impl: disable"' in script
    assert '"custom_chat_template: null"' in script
    assert '"<｜User｜>"' in script
    assert '"<｜Assistant｜>"' in script
    assert "DS4_RAY_CLUSTER_PASSED" in script
    assert "DS4_RAY_JOB_RUNTIME_ENV_PASSED" in script
    assert "DS4_RAY_GPU_ENV_PASSED" in script
    assert "DS4_RAY_HF_CONFIG_PASSED" in script
    assert "DS4_SERVER_IMPORT_STATE" in script
    assert "DS4_SERVER_IMPORT_PASSED" in script
    assert "import sitecustomize" not in import_program
    assert "compat._patch_transformers_vision2seq_alias()" not in import_program
    assert import_program.index('"DS4_SERVER_IMPORT_STATE "') < (
        import_program.index("assert startup_sitecustomize is not None")
    )
    assert script.index('if [[ "${IMPORT_ONLY}" == "1" ]]') < script.index(
        "RAY_CLI=("
    )
    assert script.index("RUNTIME_ENV_JSON=$(") < script.index(
        'if [[ "${RAY_ONLY}" == "1" ]]'
    )
    assert 'VERL_MLITE_VLLM_SITE="${DS4_VLLM_SITE}"' in script
    assert 'VERL_MLITE_VLLM_LD_PRELOAD="${DS4_VLLM_SHIM}"' in script
    assert "DS4_WEIGHT_SYNC_PROFILE_DISABLED" in script
    probe_unset = "unset MLITE_WEIGHT_SYNC_PROBE MLITE_WEIGHT_SYNC_PROBE_BACKEND"
    assert probe_unset in script
    assert script.index(probe_unset) < script.index(
        'if [[ "${IMPORT_ONLY}" == "1" ]]'
    )
    assert "weight_sync_probe=" in import_program
    assert "skip_runtime_patches=" in import_program
    assert "headless_api_server_count_patch=" in import_program
    assert "_verl_mlite_api_server_count_patch" in import_program
    assert "transformers_id=" in import_program
    assert "transformers_version=" in import_program
    assert "vllm_site=" in import_program
    assert script.count("VERL_MLITE_RUNTIME_PATCH_TRACE=1") == 1
    assert '"VERL_MLITE_RUNTIME_PATCH_TRACE",' not in script
    assert "export VERL_MLITE_HF_CONFIG_MODEL_TYPE=deepseek_v4" in script
    assert '"VERL_MLITE_HF_CONFIG_MODEL_TYPE",' in script
    assert '"CHECKPOINT_DIR",' in script
    assert 'export PYTHONPATH="${MLITE_SM90_SITE}' in script
    assert ':${DS4_VLLM_SITE}:' in script
    assert 'export LD_PRELOAD="${DS4_VLLM_SHIM}"' in script
    assert 'export VLLM_CACHE_ROOT="${RUN_ROOT}/vllm-cache"' in script
    assert 'export TILELANG_CACHE_DIR="${RUN_ROOT}/tilelang-cache"' in script
    assert 'export TILELANG_TMP_DIR="${TILELANG_CACHE_DIR}/tmp"' in script
    assert 'export VLLM_WORKER_MULTIPROC_METHOD="spawn"' in script
    assert 'export VLLM_DEEP_GEMM_WARMUP="skip"' in script
    assert 'export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"' in script
    assert '"${VLLM_CACHE_ROOT}"' in script
    for name in (
        "VLLM_CACHE_ROOT",
        "TILELANG_CACHE_DIR",
        "TILELANG_TMP_DIR",
        "VLLM_WORKER_MULTIPROC_METHOD",
        "VLLM_DEEP_GEMM_WARMUP",
        "PYTORCH_CUDA_ALLOC_CONF",
    ):
        assert f'"{name}",' in script
    assert "unset HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES" in script
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
    assert 'HYDRA_RUN_DIR="${RUN_ROOT}/hydra/${phase}"' in script
    assert 'run_phase phase1 "${PHASE1_STEPS}" disable' in script
    assert 'run_phase resume "${TOTAL_STEPS}" auto' in script
    assert "DS4_GRPO_PHASE1_COMPLETE" in script
    assert "DS4_GRPO_RESUME_COMPLETE" in script
    assert "validate_grpo_metrics.py" in script
    assert "metrics-report.json" in script
    assert "DS4_GRPO_RUN_COMPLETE" in script
    assert "DRY_RUN=0" in script


def test_ds4_grpo_sbatch_has_bounded_vllm_load_only_gate() -> None:
    script = SBATCH.read_text()
    load_program = VLLM_LOAD_ONLY_MODULE.read_text()

    ast.parse(load_program)
    assert 'if [[ "${VLLM_LOAD_ONLY}" == "1" ]]' in script
    assert "VLLM_LOAD_ONLY requires one node with eight GPUs" in script
    assert "deepseek_v4_rollout_load_only.py" in script
    assert "tensor_parallel_size=rollout_tp" in load_program
    assert 'load_format="dummy"' in load_program
    assert '"expert_dtype": "fp8"' in load_program
    assert "DS4_VLLM_LOAD_ONLY_PASSED" in load_program


def test_ds4_vllm_load_only_uses_production_fp8_shape(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    checkpoint_dir = tmp_path / "mixed-model"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "config.json").write_text(json.dumps({"o_groups": 8}))
    calls = []

    class FakeLLM:
        def __init__(self, **kwargs) -> None:
            calls.append(kwargs)

    vllm = types.ModuleType("vllm")
    vllm.LLM = FakeLLM
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setenv("CHECKPOINT_DIR", str(checkpoint_dir))
    monkeypatch.setenv("ROLLOUT_TP", "8")

    spec = importlib.util.spec_from_file_location(
        "ds4_vllm_load_only_test", VLLM_LOAD_ONLY_MODULE
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main()

    assert calls[0]["tensor_parallel_size"] == 8
    assert calls[0]["load_format"] == "dummy"
    assert calls[0]["hf_overrides"]["expert_dtype"] == "fp8"
    assert calls[0]["hf_overrides"]["quantization_config"]["weight_block_size"] == [
        128,
        128,
    ]
    assert "DS4_VLLM_LOAD_ONLY_PASSED rollout_tp=8 o_groups=8 local_groups=1" in (
        capsys.readouterr().out
    )


def test_ds4_grpo_ray_only_probes_every_local_device_uuid_mapping() -> None:
    script = SBATCH.read_text()
    ray_program = script.split("    python -c '\n", 1)[1].split(
        "\n'\n  echo \"DS4_RAY_CLUSTER_PASSED", 1
    )[0]

    ast.parse(ray_program)
    assert "'" not in ray_program
    assert "RAY_ONLY requires one node with eight GPUs" in script
    assert "class DeviceProbe:" in script
    assert "_RayActorClassProfile" in script
    assert "_vllm_server_profile_env" in script
    assert "from verl.workers.config import HFModelConfig" in script
    assert "ray.remote(num_gpus=1)(DeviceProbe)" in script
    assert "_RayActorClassProfile(" in script
    assert "AutoModelForImageTextToText, AutoModelForVision2Seq" in script
    assert "import compressed_tensors" in script
    assert "import outlines_core" in script
    assert "import tvm_ffi" in script
    assert "import xgrammar" in script
    assert "import transformers" in script
    assert "import vllm" in script
    assert "vLLMHttpServer" in script
    assert "vLLMReplica" in script
    assert "for _ in range(8)" in script
    assert "get_accelerator_ids()" in script
    assert 'os.environ.get("CUDA_VISIBLE_DEVICES")' in script
    assert "torch.cuda.current_device()" in script
    assert "torch.cuda.device_count()" in script
    assert (
        "from verl.workers.rollout.vllm_rollout.vllm_rollout import "
        "get_device_uuid" in script
    )
    assert "get_device_uuid(get_device_id())" in script
    assert "get_device_uuid(int(accelerator_ids[0]))" in script
    assert 'record["accelerator_ids"]' in script
    assert 'record["visible_devices"]' in script
    assert 'record["logical_device_uuid"]' in script
    assert 'record["physical_device_uuid"]' in script
    assert 'record["transformers_origin"]' in script
    assert 'record["vllm_origin"]' in script
    assert 'record["dependency_origins"]' in script
    assert 'record["dependency_versions"]' in script
    assert 'record["server_profile_patch_applied"]' in script
    assert 'record["pythonpath"]' in script
    assert (
        'record["logical_device_uuid"] == record["physical_device_uuid"]' in script
    )
    assert "accelerator id must be a decimal CUDA device id" in script
    assert "len(set(physical_ids)) == 8" in script
    assert "len(set(device_uuids)) == 8" in script
    assert "DS4_RAY_SERVER_PROFILE_PASSED" in script
    assert "DS4_RAY_DEVICE_UUID_PROBE_PASSED actors=8" in script


def test_mlite_engine_reapplies_vllm_device_uuid_patch_before_registration() -> None:
    engine = ENGINE_MODULE.read_text()

    assert "_patch_verl_vllm_device_uuid" in engine
    assert (
        engine.index("_patch_verl_vllm_device_uuid()")
        < engine.index("load_verl_engine_api()")
    )


def test_ds4_grpo_readme_uses_the_smoke_partition_limit() -> None:
    readme = README.read_text()

    assert "sbatch --partition=batch --nodes=8 --time=04:00:00" in readme


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
