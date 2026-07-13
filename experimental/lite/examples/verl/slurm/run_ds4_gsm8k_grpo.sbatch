#!/usr/bin/env bash
#SBATCH --job-name=ds4-grpo
#SBATCH --account=coreai_devtech_all
#SBATCH --partition=batch
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=04:00:00

set -euo pipefail

BASE_IMAGE=${BASE_IMAGE:?set BASE_IMAGE to the Hopper-qualified container}
MLITE_SRC=${MLITE_SRC:?set MLITE_SRC to a committed Megatron Lite snapshot}
MLITE_COMMIT=${MLITE_COMMIT:?set MLITE_COMMIT to the snapshot commit}
MEGATRON_ROOT=${MEGATRON_ROOT:?set MEGATRON_ROOT to a compatible MCore snapshot}
VERL_ROOT=${VERL_ROOT:?set VERL_ROOT to a committed VERL snapshot}
MLITE_SM90_SITE=${MLITE_SM90_SITE:?set MLITE_SM90_SITE to the SM90 training overlay}
DS4_VLLM_SITE=${DS4_VLLM_SITE:?set DS4_VLLM_SITE to the DS4 vLLM thin overlay}
# DS4_VLLM_SHIM is the Torch/vLLM ABI LD_PRELOAD shim required by the cw
# H100/NGC26.04 thin-0.20.2 overlay. On a natively-built overlay (e.g. the GB200
# cd0de48/NGC26.06 build) there is no ABI gap, so the shim is optional: leave it
# empty and no LD_PRELOAD is injected. It stays exported (possibly empty) because
# the Ray runtime-env snapshot indexes os.environ["DS4_VLLM_SHIM"] directly.
DS4_VLLM_SHIM=${DS4_VLLM_SHIM:-}
CHECKPOINT_DIR=${CHECKPOINT_DIR:?set CHECKPOINT_DIR to the official mixed checkpoint}
RUN_ROOT=${RUN_ROOT:?set RUN_ROOT to a fresh result directory}

for path in \
  "${BASE_IMAGE}" \
  "${MLITE_SRC}" \
  "${MEGATRON_ROOT}" \
  "${VERL_ROOT}" \
  "${MLITE_SM90_SITE}" \
  "${DS4_VLLM_SITE}" \
  "${CHECKPOINT_DIR}"; do
  [[ -e "${path}" ]] || {
    echo "required DS4 GRPO path does not exist: ${path}" >&2
    exit 2
  }
done
# The ABI shim is only checked when supplied (native overlays leave it empty).
if [[ -n "${DS4_VLLM_SHIM}" && ! -e "${DS4_VLLM_SHIM}" ]]; then
  echo "DS4_VLLM_SHIM set but does not exist: ${DS4_VLLM_SHIM}" >&2
  exit 2
fi
[[ "$(git -C "${MLITE_SRC}" rev-parse HEAD)" == "${MLITE_COMMIT}" ]] || {
  echo "MLite source is not at ${MLITE_COMMIT}" >&2
  exit 7
}
[[ -n "${SLURM_JOB_ID:-}" ]] || {
  echo "submit this script with sbatch" >&2
  exit 2
}

SCRIPT_PATH="${MLITE_SRC}/experimental/lite/examples/verl/slurm/run_ds4_gsm8k_grpo.sbatch"
if [[ -z "${MASTER_ADDR:-}" ]]; then
  MASTER_ADDR=$(hostname -i)
  MASTER_ADDR=${MASTER_ADDR%% *}
fi
RAY_PORT=${RAY_PORT:-6379}
RAY_DASHBOARD_PORT=${RAY_DASHBOARD_PORT:-8265}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
CONFIG_TARGET_NNODES=${CONFIG_TARGET_NNODES:-16}
CONFIG_TARGET_GPUS_PER_NODE=${CONFIG_TARGET_GPUS_PER_NODE:-8}
PHASE1_STEPS=${PHASE1_STEPS:-3}
TOTAL_STEPS=${TOTAL_STEPS:-6}
CONFIG_ONLY=${CONFIG_ONLY:-0}
IMPORT_ONLY=${IMPORT_ONLY:-0}
RAY_ONLY=${RAY_ONLY:-0}
VLLM_LOAD_ONLY=${VLLM_LOAD_ONLY:-0}
VLLM_CHECKPOINT_SYNC_PROBE=${VLLM_CHECKPOINT_SYNC_PROBE:-0}
ACTOR_TP=${ACTOR_TP:-1}
ACTOR_PP=${ACTOR_PP:-4}
ACTOR_CP=${ACTOR_CP:-4}
ACTOR_EP=${ACTOR_EP:-8}
ROLLOUT_TP=${ROLLOUT_TP:-8}

if (( PHASE1_STEPS < 1 || TOTAL_STEPS <= PHASE1_STEPS )); then
  echo "expected 1 <= PHASE1_STEPS < TOTAL_STEPS" >&2
  exit 2
fi
WORLD_SIZE=$((SLURM_NNODES * GPUS_PER_NODE))
ACTOR_MODEL_PARALLEL=$((ACTOR_TP * ACTOR_PP * ACTOR_CP * ACTOR_EP))
if (( WORLD_SIZE % ACTOR_MODEL_PARALLEL != 0 )); then
  echo "world_size=${WORLD_SIZE} must be divisible by actor model parallel=${ACTOR_MODEL_PARALLEL}" >&2
  exit 2
fi
if (( WORLD_SIZE % ROLLOUT_TP != 0 )); then
  echo "world_size=${WORLD_SIZE} must be divisible by rollout_tp=${ROLLOUT_TP}" >&2
  exit 2
fi
if (( RAY_ONLY == 1 && (SLURM_NNODES < 1 || SLURM_NNODES > 2 || GPUS_PER_NODE != 8) )); then
  echo "RAY_ONLY requires one or two nodes with eight GPUs per node" >&2
  exit 2
fi
if (( CONFIG_ONLY == 1 && (SLURM_NNODES != 1 || GPUS_PER_NODE != 0) )); then
  echo "CONFIG_ONLY requires one CPU-only node with GPUS_PER_NODE=0" >&2
  exit 2
fi
if (( VLLM_LOAD_ONLY == 1 && (SLURM_NNODES != 1 || GPUS_PER_NODE != 8) )); then
  echo "VLLM_LOAD_ONLY requires one node with eight GPUs" >&2
  exit 2
fi
if (( VLLM_CHECKPOINT_SYNC_PROBE == 1 && VLLM_LOAD_ONLY != 1 )); then
  echo "VLLM_CHECKPOINT_SYNC_PROBE requires VLLM_LOAD_ONLY=1" >&2
  exit 2
fi
if (( IMPORT_ONLY == 1 && SLURM_NNODES != 1 )); then
  echo "IMPORT_ONLY requires one CPU node" >&2
  exit 2
fi
if (( CONFIG_ONLY + IMPORT_ONLY + RAY_ONLY + VLLM_LOAD_ONLY > 1 )); then
  echo "CONFIG_ONLY, IMPORT_ONLY, RAY_ONLY, and VLLM_LOAD_ONLY are mutually exclusive" >&2
  exit 2
fi

if [[ "${IN_DS4_GRPO_CONTAINER:-0}" != "1" ]]; then
  [[ ! -e "${RUN_ROOT}" ]] || {
    echo "refusing to reuse existing DS4 GRPO run root: ${RUN_ROOT}" >&2
    exit 5
  }
  export BASE_IMAGE MLITE_SRC MLITE_COMMIT MEGATRON_ROOT VERL_ROOT
  export MLITE_SM90_SITE DS4_VLLM_SITE DS4_VLLM_SHIM CHECKPOINT_DIR RUN_ROOT
  export MASTER_ADDR RAY_PORT RAY_DASHBOARD_PORT GPUS_PER_NODE
  export CONFIG_TARGET_NNODES CONFIG_TARGET_GPUS_PER_NODE
  export PHASE1_STEPS TOTAL_STEPS CONFIG_ONLY IMPORT_ONLY RAY_ONLY VLLM_LOAD_ONLY
  export VLLM_CHECKPOINT_SYNC_PROBE
  export ACTOR_TP ACTOR_PP ACTOR_CP ACTOR_EP ROLLOUT_TP
  # Load-only diagnostics (C1 ray-executor arm + GPU-0 sibling residency probe); empty
  # backend => vLLM keeps its default multiproc executor (bayan 2026-07-12 22:05).
  export MLITE_VLLM_RESIDENCY_PROBE="${MLITE_VLLM_RESIDENCY_PROBE:-}"
  export MLITE_VLLM_DISTRIBUTED_EXECUTOR_BACKEND="${MLITE_VLLM_DISTRIBUTED_EXECUTOR_BACKEND:-}"
  exec srun --export=ALL \
    --nodes="${SLURM_NNODES}" \
    --ntasks="${SLURM_NNODES}" \
    --ntasks-per-node=1 \
    --container-image="${BASE_IMAGE}" \
    --container-mounts=/lustre:/lustre \
    --container-workdir="${MLITE_SRC}" \
    env IN_DS4_GRPO_CONTAINER=1 bash "${SCRIPT_PATH}"
fi

if [[ -n "${MLITE_WEIGHT_SYNC_PROBE:-}" || -n "${MLITE_WEIGHT_SYNC_PROBE_BACKEND:-}" ]]; then
  echo "DS4_WEIGHT_SYNC_PROFILE_DISABLED"
fi
unset MLITE_WEIGHT_SYNC_PROBE MLITE_WEIGHT_SYNC_PROBE_BACKEND

# Pin the CUDA toolkit to the container's coherent CUDA 13.0 install. Without
# this, TileLang's find_cuda_path() falls back to the pip-floating
# nvidia/cu13 package (a different nvcc minor, e.g. 13.2) whose compiler is
# incompatible with the cccl headers shipped alongside /usr/local/cuda, and the
# DS4 mhc_pre_tilelang runtime nvcc JIT compile fails with
# "CUDA compiler and CUDA toolkit headers are incompatible".
export CUDA_HOME=/usr/local/cuda
export CUDA_PATH=/usr/local/cuda
export PATH=/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin
hash -r
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export CPATH=/usr/include:/usr/include/x86_64-linux-gnu
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_NVLS_ENABLE=0
export PYTHONNOUSERSITE=1
export RAY_raylet_start_wait_time_s="${RAY_RAYLET_START_WAIT_TIME_S:-180}"
export VLLM_USE_V1=1
export VLLM_WORKER_MULTIPROC_METHOD="spawn"
export VLLM_DEEP_GEMM_WARMUP="skip"
export VLLM_NO_USAGE_STATS=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export VERL_VLLM_FP8_QUANT_ENABLED=0
export VERL_MLITE_HF_CONFIG_MODEL_TYPE=deepseek_v4
export VERL_MLITE_VLLM_SITE="${DS4_VLLM_SITE}"
export VERL_MLITE_VLLM_LD_PRELOAD="${DS4_VLLM_SHIM}"
export LD_PRELOAD="${DS4_VLLM_SHIM}"
export PYTHONPATH="${MLITE_SM90_SITE}/nvidia_cutlass_dsl/python_packages:${MLITE_SM90_SITE}:${DS4_VLLM_SITE}:${VERL_ROOT}:${MEGATRON_ROOT}:${MLITE_SRC}/experimental/lite/examples/verl:${MLITE_SRC}/experimental/lite:${MLITE_SRC}"
export TMPDIR="/tmp/ds4-grpo-${SLURM_JOB_ID}-${SLURM_PROCID}"
export PYTHONPYCACHEPREFIX="${TMPDIR}/pycache"
export TORCHINDUCTOR_CACHE_DIR="${TMPDIR}/torchinductor"
export TRITON_CACHE_DIR="${TMPDIR}/triton"
export HF_HOME="${RUN_ROOT}/hf-cache"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export VLLM_CACHE_ROOT="${RUN_ROOT}/vllm-cache"
export TILELANG_CACHE_DIR="${RUN_ROOT}/tilelang-cache"
export TILELANG_TMP_DIR="${TILELANG_CACHE_DIR}/tmp"
unset HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES
mkdir -p \
  "${PYTHONPYCACHEPREFIX}" \
  "${TORCHINDUCTOR_CACHE_DIR}" \
  "${TRITON_CACHE_DIR}" \
  "${HF_DATASETS_CACHE}" \
  "${VLLM_CACHE_ROOT}" \
  "${TILELANG_CACHE_DIR}" \
  "${TILELANG_TMP_DIR}"

if [[ "${IMPORT_ONLY}" == "1" ]]; then
  env \
    PYTHONPATH="${DS4_VLLM_SITE}:${PYTHONPATH}" \
    VERL_MLITE_SKIP_RUNTIME_PATCHES=0 \
    VERL_MLITE_RUNTIME_PATCH_TRACE=1 \
    python - <<'PY'
import os
import sys

import transformers
from transformers import AutoModelForImageTextToText

alias_name = "AutoModelForVision2Seq"
startup_sitecustomize = sys.modules.get("sitecustomize")
alias_before = (
    vars(transformers).get(alias_name) is AutoModelForImageTextToText
    or getattr(transformers, "_objects", {}).get(alias_name)
    is AutoModelForImageTextToText
)
from verl_mlite import compat

print(
    "DS4_SERVER_IMPORT_STATE "
    f"sitecustomize={getattr(startup_sitecustomize, '__file__', None)} "
    f"compat={compat.__file__} "
    f"transformers_id={id(transformers)} "
    f"transformers_version={transformers.__version__} "
    f"vllm_site={os.environ.get('VERL_MLITE_VLLM_SITE')} "
    f"weight_sync_probe={os.environ.get('MLITE_WEIGHT_SYNC_PROBE')} "
    f"skip_runtime_patches={os.environ.get('VERL_MLITE_SKIP_RUNTIME_PATCHES')} "
    f"alias_before={alias_before}",
    flush=True,
)
assert startup_sitecustomize is not None, "sitecustomize did not load at startup"
expected_sitecustomize = os.path.join(
    os.environ["MLITE_SRC"], "experimental/lite/examples/verl/sitecustomize.py"
)
assert os.path.realpath(startup_sitecustomize.__file__) == os.path.realpath(
    expected_sitecustomize
), (startup_sitecustomize.__file__, expected_sitecustomize)
assert alias_before, "sitecustomize did not preserve the Transformers v5 alias"

import vllm
from transformers import AutoModelForVision2Seq
from verl.utils.model import update_model_config
from verl.workers.config import HFModelConfig
from verl.workers.rollout.vllm_rollout.vllm_async_server import (
    run_headless,
    vLLMHttpServer,
    vLLMReplica,
)

rollout_site = os.path.realpath(os.environ["DS4_VLLM_SITE"])
headless_api_server_count_patch = bool(
    getattr(run_headless, "_verl_mlite_api_server_count_patch", False)
)
assert "verl_mlite.compat" in sys.modules
assert AutoModelForVision2Seq is AutoModelForImageTextToText
assert callable(update_model_config)
assert HFModelConfig is not None
assert vLLMHttpServer is not None
assert getattr(vLLMReplica, "_verl_mlite_server_profile_patch", False)
assert headless_api_server_count_patch
assert transformers.__version__ == "5.12.1"
assert vllm.__version__ == "0.20.2"
for module in (transformers, vllm):
    assert os.path.commonpath(
        [os.path.realpath(module.__file__), rollout_site]
    ) == rollout_site
print(
    "DS4_SERVER_IMPORT_PASSED "
    f"transformers={transformers.__version__} vllm={vllm.__version__} "
    f"headless_api_server_count_patch={headless_api_server_count_patch} "
    f"transformers_origin={transformers.__file__} vllm_origin={vllm.__file__}",
    flush=True,
)
PY
  exit 0
fi

if [[ "${VLLM_LOAD_ONLY}" == "1" ]]; then
  env -u VLLM_LOAD_ONLY \
    PYTHONPATH="${DS4_VLLM_SITE}:${PYTHONPATH}" \
    python experimental/lite/examples/verl/deepseek_v4_rollout_load_only.py
  exit 0
fi

RAY_CLI=(env VERL_MLITE_SKIP_RUNTIME_PATCHES=1 python -m ray.scripts.scripts)
"${RAY_CLI[@]}" --help >/dev/null

runtime_env_with_file_logger_path() {
  VERL_MLITE_SKIP_RUNTIME_PATCHES=1 python - "$1" "$2" <<'PY'
import json
import sys

runtime_env = json.loads(sys.argv[1])
env_vars = runtime_env.setdefault("env_vars", {})
env_vars["VERL_FILE_LOGGER_PATH"] = sys.argv[2]
print(json.dumps(runtime_env))
PY
}

NODE_RANK=${SLURM_PROCID}
DONE_FILE="${RUN_ROOT}/.ray_done"
RAY_TEMP_DIR="/tmp/ds4-grpo-${SLURM_JOB_ID}-ray"
RAY_LOG_DIR="${RUN_ROOT}/ray-logs/node-${NODE_RANK}"
if [[ "${CONFIG_ONLY}" == "1" ]]; then
  if [[ "${NODE_RANK}" == "0" ]]; then
    mkdir -p "${RUN_ROOT}"
    COMPOSE_ONLY=1 \
      DRY_RUN=0 \
      MODEL_PATH="${CHECKPOINT_DIR}" \
      TRAIN_FILES="${RUN_ROOT}/train.parquet" \
      VAL_FILES="${RUN_ROOT}/test.parquet" \
      OUTPUT_ROOT="${RUN_ROOT}/output" \
      CKPT_DIR="${RUN_ROOT}/checkpoints" \
      NNODES="${CONFIG_TARGET_NNODES}" \
      NGPUS_PER_NODE="${CONFIG_TARGET_GPUS_PER_NODE}" \
      ACTOR_TP="${ACTOR_TP}" \
      ACTOR_PP="${ACTOR_PP}" \
      ACTOR_CP="${ACTOR_CP}" \
      ACTOR_EP="${ACTOR_EP}" \
      ROLLOUT_TP="${ROLLOUT_TP}" \
      TOTAL_TRAINING_STEPS="${PHASE1_STEPS}" \
      bash "${MLITE_SRC}/experimental/lite/examples/verl/scripts/run_deepseek_v4_gsm8k_grpo.sh" \
      > "${RUN_ROOT}/resolved-config.yaml" 2>&1
    python - "${RUN_ROOT}/resolved-config.yaml" <<'PY'
import os
import sys

config = open(sys.argv[1], encoding="utf-8").read()
required = [
    "strategy: mlite",
    "model_name: deepseek_v4",
    "resync_format: vllm_checkpoint",
    "expert_dtype: fp8",
    "recompute: full",
    "use_legacy_worker_impl: disable",
    "custom_chat_template:",
    "name: ChatTemplateRLHFDataset",
    "verl_mlite/dataset.py",
    "gpu_memory_utilization: 0.6",
    "chat_template:",
    "<｜User｜>",
    "<｜Assistant｜>",
]
missing = [marker for marker in required if marker not in config]
if missing:
    raise SystemExit(f"resolved DS4 GRPO config is missing markers: {missing}")
if "custom_chat_template: null" in config:
    raise SystemExit("resolved DS4 GRPO config has no chat template")
cache_roots = {
    "vLLM": os.environ["VLLM_CACHE_ROOT"],
    "TileLang": os.environ["TILELANG_CACHE_DIR"],
    "TileLang tmp": os.environ["TILELANG_TMP_DIR"],
}
for cache_name, cache_root in cache_roots.items():
    if not os.path.isdir(cache_root):
        raise SystemExit(f"{cache_name} cache root was not created: {cache_root}")
for forbidden in (
    "Error in sitecustomize",
    "Failed to import Triton kernels",
    "Traceback",
):
    if forbidden in config:
        raise SystemExit(f"resolved DS4 GRPO config contains initialization error: {forbidden}")
PY
    env \
      PYTHONPATH="${DS4_VLLM_SITE}:${PYTHONPATH}" \
      VERL_MLITE_SKIP_RUNTIME_PATCHES=1 \
      python - <<'PY'
import os

from tilelang import env as tilelang_env

for name, resolved_path in {
    "TILELANG_CACHE_DIR": tilelang_env.TILELANG_CACHE_DIR,
    "TILELANG_TMP_DIR": tilelang_env.TILELANG_TMP_DIR,
}.items():
    expected_path = os.environ[name]
    if os.path.realpath(resolved_path) != os.path.realpath(expected_path):
        raise SystemExit(
            f"TileLang resolved {name}={resolved_path}, expected {expected_path}"
        )
print(
    "DS4_TILELANG_CACHE_PREFLIGHT_PASSED "
    f"cache={tilelang_env.TILELANG_CACHE_DIR} "
    f"tmp={tilelang_env.TILELANG_TMP_DIR}",
    flush=True,
)
PY
    env \
      VERL_FILE_LOGGER_PATH="${RUN_ROOT}/file-logger-preflight.jsonl" \
      VERL_MLITE_SKIP_RUNTIME_PATCHES=1 \
      python - <<'PY'
import json
import os

from verl.utils.tracking import FileLogger

logger = FileLogger("ds4-preflight", "file-logger")
logger.log({"writable": True}, step=0)
logger.finish()
with open(os.environ["VERL_FILE_LOGGER_PATH"], encoding="utf-8") as log_file:
    record = json.loads(log_file.read())
assert record == {"step": 0, "data": {"writable": True}}, record
print(
    "DS4_FILE_LOGGER_PREFLIGHT_PASSED "
    f"path={os.environ['VERL_FILE_LOGGER_PATH']}",
    flush=True,
)
PY
    FILE_LOGGER_PATH="${RUN_ROOT}/phase1.jsonl"
    FILE_LOGGER_RUNTIME_ENV_JSON=$(runtime_env_with_file_logger_path \
      '{"env_vars": {"sentinel": "preserved"}}' "${FILE_LOGGER_PATH}")
    VERL_MLITE_SKIP_RUNTIME_PATCHES=1 python - \
      "${FILE_LOGGER_RUNTIME_ENV_JSON}" "${FILE_LOGGER_PATH}" <<'PY'
import json
import sys

runtime_env = json.loads(sys.argv[1])
env_vars = runtime_env["env_vars"]
assert env_vars["sentinel"] == "preserved", env_vars
assert env_vars["VERL_FILE_LOGGER_PATH"] == sys.argv[2], env_vars
print(
    "DS4_FILE_LOGGER_RUNTIME_ENV_PREFLIGHT_PASSED "
    f"path={env_vars['VERL_FILE_LOGGER_PATH']}",
    flush=True,
)
PY
    echo "DS4_GRPO_CONFIG_COMPOSE_PASSED source=${MLITE_COMMIT} vllm_cache_root=${VLLM_CACHE_ROOT} tilelang_cache_dir=${TILELANG_CACHE_DIR}"
  fi
fi

mkdir -p "${RAY_TEMP_DIR}" "${RAY_LOG_DIR}"

archive_ray_logs() {
  if [[ -d "${RAY_TEMP_DIR}/session_latest/logs" ]]; then
    cp -a "${RAY_TEMP_DIR}/session_latest/logs/." "${RAY_LOG_DIR}/" || true
  fi
}

"${RAY_CLI[@]}" stop --force >/dev/null 2>&1 || true

cleanup() {
  if [[ "${NODE_RANK}" == "0" ]]; then
    touch "${DONE_FILE}" 2>/dev/null || true
  fi
  "${RAY_CLI[@]}" stop --force >/dev/null 2>&1 || true
  archive_ray_logs
}
trap cleanup EXIT

if [[ "${NODE_RANK}" != "0" ]]; then
  NODE_ADDR=$(hostname -i)
  NODE_ADDR=${NODE_ADDR%% *}
  for attempt in $(seq 1 150); do
    if "${RAY_CLI[@]}" start --address="${MASTER_ADDR}:${RAY_PORT}" \
      --num-gpus="${GPUS_PER_NODE}" \
      --node-ip-address="${NODE_ADDR}" \
      --disable-usage-stats; then
      break
    fi
    if [[ "${attempt}" == "150" ]]; then
      echo "failed to join Ray head at ${MASTER_ADDR}:${RAY_PORT}" >&2
      exit 1
    fi
    sleep 2
  done
  while [[ ! -f "${DONE_FILE}" ]]; do
    sleep 10
  done
  exit 0
fi

mkdir -p "${RUN_ROOT}"
"${RAY_CLI[@]}" start --head \
  --node-ip-address="${MASTER_ADDR}" \
  --port="${RAY_PORT}" \
  --num-gpus="${GPUS_PER_NODE}" \
  --temp-dir="${RAY_TEMP_DIR}" \
  --disable-usage-stats \
  --dashboard-host=0.0.0.0 \
  --dashboard-port="${RAY_DASHBOARD_PORT}"

python - <<'PY'
import os
import time

import ray

expected = int(os.environ["SLURM_NNODES"])
ray.init(address=f"{os.environ['MASTER_ADDR']}:{os.environ['RAY_PORT']}")
deadline = time.time() + 600
while True:
    alive = [node for node in ray.nodes() if node.get("Alive")]
    print(f"RAY_CLUSTER_NODES {len(alive)}/{expected}", flush=True)
    if len(alive) == expected:
        break
    if time.time() > deadline:
        raise SystemExit(f"timed out waiting for Ray nodes: {len(alive)}/{expected}")
    time.sleep(2)
ray.shutdown()
PY

RUNTIME_ENV_JSON=$(VERL_MLITE_SKIP_RUNTIME_PATCHES=1 python - <<'PY'
import json
import os

names = [
    "PYTHONPATH",
    "PATH",
    "CUDA_HOME",
    "CUDA_PATH",
    "CC",
    "CXX",
    "CPATH",
    "LD_PRELOAD",
    "CUDA_DEVICE_MAX_CONNECTIONS",
    "NCCL_NVLS_ENABLE",
    "PYTHONNOUSERSITE",
    "VLLM_USE_V1",
    "VLLM_WORKER_MULTIPROC_METHOD",
    "VLLM_DEEP_GEMM_WARMUP",
    "VLLM_NO_USAGE_STATS",
    "PYTORCH_CUDA_ALLOC_CONF",
    "VLLM_CACHE_ROOT",
    "TILELANG_CACHE_DIR",
    "TILELANG_TMP_DIR",
    "VERL_VLLM_FP8_QUANT_ENABLED",
    "VERL_MLITE_HF_CONFIG_MODEL_TYPE",
    "VERL_MLITE_VLLM_SITE",
    "VERL_MLITE_VLLM_LD_PRELOAD",
    "MLITE_SM90_SITE",
    "DS4_VLLM_SITE",
    "DS4_VLLM_SHIM",
    "MEGATRON_ROOT",
    "VERL_ROOT",
    "CHECKPOINT_DIR",
    "HF_HOME",
    "HF_DATASETS_CACHE",
]
env = {name: os.environ[name] for name in names}
# Opt-in resync smoke/memlog controls are read by the actor's engine
# (os.environ); propagate them into the Ray actor env only when set so the
# 8-GPU proxy gets its fail-fast verdict + JSONL curve, while the production
# run (vars unset) stays a no-op.
for name in ("MLITE_RESYNC_SMOKE_EXIT_AFTER", "MLITE_RESYNC_MEMLOG_PATH"):
    if os.environ.get(name):
        env[name] = os.environ[name]
env["VERL_MLITE_SKIP_RUNTIME_PATCHES"] = "0"
print(json.dumps({"env_vars": env}))
PY
)

if [[ "${RAY_ONLY}" == "1" ]]; then
  "${RAY_CLI[@]}" job submit \
    --address="http://127.0.0.1:${RAY_DASHBOARD_PORT}" \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- \
    python -c '
import json
import os
import sys

import ray
from transformers import AutoConfig
from verl_mlite.compat import _RayActorClassProfile, _vllm_server_profile_env
from verl.workers.config import HFModelConfig

assert os.environ["VERL_MLITE_SKIP_RUNTIME_PATCHES"] == "0"
assert "verl_mlite.compat" in sys.modules
hf_config = AutoConfig.from_pretrained(
    os.environ["CHECKPOINT_DIR"], trust_remote_code=True
)
assert hf_config.model_type == os.environ["VERL_MLITE_HF_CONFIG_MODEL_TYPE"]
print(f"DS4_RAY_HF_CONFIG_PASSED model_type={hf_config.model_type}", flush=True)
ray.init(address="auto")


class DeviceProbe:
    def __init__(self, model_config_cls):
        self.model_config_cls = model_config_cls

    def inspect(self):
        import compressed_tensors
        import numba
        import outlines_core
        import torch
        import transformers
        import tvm_ffi
        import vllm
        import xgrammar

        from importlib.metadata import distributions
        from transformers import AutoModelForImageTextToText, AutoModelForVision2Seq
        from verl.utils.device import get_device_id
        from verl.utils.model import update_model_config
        from verl.workers.rollout.vllm_rollout.vllm_async_server import (
            vLLMHttpServer,
            vLLMReplica,
        )
        from verl.workers.rollout.vllm_rollout.vllm_rollout import get_device_uuid

        assert AutoModelForVision2Seq is AutoModelForImageTextToText
        assert callable(update_model_config)
        assert self.model_config_cls.__name__ == "HFModelConfig"
        assert vLLMHttpServer is not None
        assert getattr(vLLMReplica, "_verl_mlite_server_profile_patch", False)

        rollout_site = os.path.realpath(os.environ["DS4_VLLM_SITE"])
        training_site = os.path.realpath(os.environ["MLITE_SM90_SITE"])

        def site_versions(site):
            return {
                distribution.metadata["Name"].lower().replace("_", "-"): distribution.version
                for distribution in distributions(path=[site])
                if distribution.metadata["Name"]
            }

        rollout_versions = site_versions(rollout_site)

        accelerator_ids = [
            str(value)
            for value in ray.get_runtime_context()
            .get_accelerator_ids()
            .get("GPU", [])
        ]
        if len(accelerator_ids) != 1 or not accelerator_ids[0].isdigit():
            raise AssertionError(
                "accelerator id must be a decimal CUDA device id: "
                f"{accelerator_ids}"
            )
        return {
            "accelerator_ids": accelerator_ids,
            "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "current_device": torch.cuda.current_device(),
            "device_count": torch.cuda.device_count(),
            "logical_device_uuid": get_device_uuid(get_device_id()),
            "physical_device_uuid": get_device_uuid(int(accelerator_ids[0])),
            "patch_applied": bool(
                getattr(get_device_uuid, "_verl_mlite_visible_device_patch", False)
            ),
            "server_profile_patch_applied": bool(
                getattr(vLLMReplica, "_verl_mlite_server_profile_patch", False)
            ),
            "dependency_origins": {
                "compressed_tensors": compressed_tensors.__file__,
                "numba": numba.__file__,
                "outlines_core": outlines_core.__file__,
                "tvm_ffi": tvm_ffi.__file__,
                "xgrammar": xgrammar.__file__,
            },
            "dependency_versions": {
                "apache-tvm-ffi": rollout_versions["apache-tvm-ffi"],
                "compressed-tensors": rollout_versions["compressed-tensors"],
                "numba": rollout_versions["numba"],
                "outlines-core": rollout_versions["outlines-core"],
                "transformers": rollout_versions["transformers"],
                "vllm": rollout_versions["vllm"],
                "xgrammar": rollout_versions["xgrammar"],
            },
            "pythonpath": os.environ.get("PYTHONPATH"),
            "transformers_origin": transformers.__file__,
            "vllm_origin": vllm.__file__,
            "training_site": training_site,
            "rollout_site": rollout_site,
            "node_id": ray.get_runtime_context().get_node_id(),
        }


ProbeActor = _RayActorClassProfile(
    ray.remote(num_gpus=1)(DeviceProbe),
    _vllm_server_profile_env(),
)
expected_nodes = int(os.environ["SLURM_NNODES"])
gpus_per_node = int(os.environ["GPUS_PER_NODE"])
actor_count = expected_nodes * gpus_per_node
actors = [ProbeActor.options().remote(HFModelConfig) for _ in range(actor_count)]
records = ray.get([actor.inspect.remote() for actor in actors])
assert len(records) == actor_count, records
node_ids = [record["node_id"] for record in records]
assert len(set(node_ids)) == expected_nodes, records
training_site = os.path.realpath(os.environ["MLITE_SM90_SITE"])
rollout_site = os.path.realpath(os.environ["DS4_VLLM_SITE"])
expected_dependency_versions = {
    "apache-tvm-ffi": "0.1.9",
    "compressed-tensors": "0.15.0.1",
    "numba": "0.65.0",
    "outlines-core": "0.2.14",
    "transformers": "5.12.1",
    "vllm": "0.20.2",
    "xgrammar": "0.2.2",
}
physical_ids = []
device_uuids = []
node_records = {}
for record in records:
    accelerator_ids = record["accelerator_ids"]
    assert len(accelerator_ids) == 1, records
    physical_id = accelerator_ids[0]
    visible_devices = record["visible_devices"]
    assert visible_devices, records
    visible_ids = {
        value.strip() for value in visible_devices.split(",") if value.strip()
    }
    assert physical_id in visible_ids, records
    assert 0 <= record["current_device"] < record["device_count"], records
    assert record["patch_applied"], records
    assert record["server_profile_patch_applied"], records
    assert record["logical_device_uuid"], records
    assert record["physical_device_uuid"], records
    assert record["logical_device_uuid"] == record["physical_device_uuid"], records
    pythonpath = record["pythonpath"]
    assert pythonpath, records
    pythonpath_entries = [
        os.path.realpath(entry)
        for entry in pythonpath.split(os.pathsep)
        if entry
    ]
    assert pythonpath_entries.index(rollout_site) < pythonpath_entries.index(
        training_site
    ), records
    assert os.path.commonpath(
        [os.path.realpath(record["transformers_origin"]), rollout_site]
    ) == rollout_site, records
    assert os.path.commonpath(
        [os.path.realpath(record["vllm_origin"]), rollout_site]
    ) == rollout_site, records
    for dependency_origin in record["dependency_origins"].values():
        assert os.path.commonpath(
            [os.path.realpath(dependency_origin), rollout_site]
        ) == rollout_site, records
    assert record["dependency_versions"] == expected_dependency_versions, records
    physical_ids.append(physical_id)
    device_uuids.append(record["physical_device_uuid"])
    node_records.setdefault(record["node_id"], []).append(physical_id)
assert len(node_records) == expected_nodes, records
for node_physical_ids in node_records.values():
    assert len(node_physical_ids) == gpus_per_node, records
    assert len(set(node_physical_ids)) == gpus_per_node, records
assert len(set(device_uuids)) == actor_count, records
transformers_origin = records[0]["transformers_origin"]
vllm_origin = records[0]["vllm_origin"]
print("DS4_RAY_JOB_RUNTIME_ENV_PASSED", flush=True)
print(
    "DS4_RAY_SERVER_PROFILE_PASSED "
    f"transformers_origin={transformers_origin} "
    f"vllm_origin={vllm_origin}",
    flush=True,
)
print(f"DS4_RAY_GPU_ENV_PASSED {json.dumps(records, sort_keys=True)}", flush=True)
print(
    "DS4_RAY_MULTINODE_PROXY_PASSED "
    f"nodes={expected_nodes} actors={actor_count} "
    f"node_physical_ids={json.dumps(node_records, sort_keys=True)} "
    f"device_uuids={device_uuids}",
    flush=True,
)
'
  echo "DS4_RAY_CLUSTER_PASSED nodes=${SLURM_NNODES} source=${MLITE_COMMIT}"
  exit 0
fi

if [[ "${CONFIG_ONLY}" == "1" ]]; then
  config_jsonl_file="${RUN_ROOT}/config-only-runtime-env.jsonl"
  config_runtime_env_json=$(runtime_env_with_file_logger_path "${RUNTIME_ENV_JSON}" "${config_jsonl_file}")
  "${RAY_CLI[@]}" job submit \
    --address="http://127.0.0.1:${RAY_DASHBOARD_PORT}" \
    --runtime-env-json="${config_runtime_env_json}" \
    -- \
    python -c '
import json
import os

from verl.utils.tracking import FileLogger

config_path = os.path.join(os.environ["RUN_ROOT"], "resolved-config.yaml")
config = open(config_path, encoding="utf-8").read()
required = ["strategy: mlite", "model_name: deepseek_v4", "recompute: full"]
missing = [marker for marker in required if marker not in config]
assert not missing, missing
assert os.environ["VERL_MLITE_SKIP_RUNTIME_PATCHES"] == "0"
logger = FileLogger("ds4-config-only", "file-logger")
logger.log({"config_runtime_env": True}, step=0)
logger.finish()
target_nodes = os.environ["CONFIG_TARGET_NNODES"]
target_gpus_per_node = os.environ["CONFIG_TARGET_GPUS_PER_NODE"]
print(
    "DS4_GRPO_CONFIG_RUNTIME_ENV_PASSED "
    f"target_nodes={target_nodes} "
    f"target_gpus_per_node={target_gpus_per_node}",
    flush=True,
)
'
  python - "${config_jsonl_file}" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as log_file:
    record = json.loads(log_file.read())
assert record == {"step": 0, "data": {"config_runtime_env": True}}, record
PY
  echo "DS4_GRPO_CONFIG_ONLY_COMPLETE target_nodes=${CONFIG_TARGET_NNODES} target_gpus_per_node=${CONFIG_TARGET_GPUS_PER_NODE}"
  exit 0
fi

DATA_DIR="${RUN_ROOT}/data"
python "${MLITE_SRC}/experimental/lite/examples/verl/prepare_math_smoke_data.py" \
  --output-dir "${DATA_DIR}" \
  --train-size "${TRAIN_SIZE:-64}" \
  --test-size "${TEST_SIZE:-16}"

CKPT_DIR="${RUN_ROOT}/checkpoints"
VERL_OUTPUT="${RUN_ROOT}/verl"
RUN_NAME="ds4_grpo_pp${ACTOR_PP}_ep${ACTOR_EP}_cp${ACTOR_CP}_rtp${ROLLOUT_TP}"

run_phase() {
  local phase=$1
  local total_steps=$2
  local resume_mode=$3
  local log_file="${RUN_ROOT}/${phase}.log"
  local jsonl_file="${RUN_ROOT}/${phase}.jsonl"
  local cmd_file="${RUN_ROOT}/${phase}.cmd.sh"
  local phase_runtime_env_json
  phase_runtime_env_json=$(runtime_env_with_file_logger_path "${RUNTIME_ENV_JSON}" "${jsonl_file}")
  "${RAY_CLI[@]}" job submit \
    --address="http://127.0.0.1:${RAY_DASHBOARD_PORT}" \
    --runtime-env-json="${phase_runtime_env_json}" \
    -- \
    env \
      MODEL_PATH="${CHECKPOINT_DIR}" \
      TRAIN_FILES="${DATA_DIR}/train.parquet" \
      VAL_FILES="${DATA_DIR}/test.parquet" \
      HYDRA_RUN_DIR="${RUN_ROOT}/hydra/${phase}" \
      OUTPUT_ROOT="${VERL_OUTPUT}" \
      CKPT_DIR="${CKPT_DIR}" \
      LOG_FILE="${log_file}" \
      JSONL_FILE="${jsonl_file}" \
      CMD_FILE="${cmd_file}" \
      RUN_NAME="${RUN_NAME}" \
      NNODES="${SLURM_NNODES}" \
      NGPUS_PER_NODE="${GPUS_PER_NODE}" \
      ACTOR_TP="${ACTOR_TP}" \
      ACTOR_PP="${ACTOR_PP}" \
      ACTOR_CP="${ACTOR_CP}" \
      ACTOR_EP="${ACTOR_EP}" \
      ROLLOUT_TP="${ROLLOUT_TP}" \
      TOTAL_TRAINING_STEPS="${total_steps}" \
      SAVE_FREQ="${PHASE1_STEPS}" \
      RESUME_MODE="${resume_mode}" \
      RESUME_FROM_PATH=null \
      DRY_RUN=0 \
      bash "${MLITE_SRC}/experimental/lite/examples/verl/scripts/run_deepseek_v4_gsm8k_grpo.sh"
}

echo "DS4_GRPO_PHASE1_START steps=${PHASE1_STEPS} world=${WORLD_SIZE} actor_mp=${ACTOR_MODEL_PARALLEL} rollout_tp=${ROLLOUT_TP}"
run_phase phase1 "${PHASE1_STEPS}" disable
PHASE1_CHECKPOINT=$(find "${CKPT_DIR}" -maxdepth 2 -type d -name "global_step_${PHASE1_STEPS}" -print -quit)
[[ -n "${PHASE1_CHECKPOINT}" ]] || {
  echo "phase1 did not create global_step_${PHASE1_STEPS}" >&2
  exit 6
}
echo "DS4_GRPO_PHASE1_COMPLETE step=${PHASE1_STEPS}"

echo "DS4_GRPO_RESUME_START from=${PHASE1_STEPS} target=${TOTAL_STEPS}"
run_phase resume "${TOTAL_STEPS}" auto
FINAL_CHECKPOINT=$(find "${CKPT_DIR}" -maxdepth 2 -type d -name "global_step_${TOTAL_STEPS}" -print -quit)
[[ -n "${FINAL_CHECKPOINT}" ]] || {
  echo "resume phase did not create global_step_${TOTAL_STEPS}" >&2
  exit 6
}
echo "DS4_GRPO_RESUME_COMPLETE step=${TOTAL_STEPS}"
python "${MLITE_SRC}/experimental/lite/examples/verl/validate_grpo_metrics.py" \
  --metrics "${RUN_ROOT}/phase1.jsonl" "${RUN_ROOT}/resume.jsonl" \
  --expected-steps "${TOTAL_STEPS}" \
  --output "${RUN_ROOT}/metrics-report.json"
echo "DS4_GRPO_RUN_COMPLETE steps=${TOTAL_STEPS} world=${WORLD_SIZE} actor_mp=${ACTOR_MODEL_PARALLEL} rollout_tp=${ROLLOUT_TP}"
