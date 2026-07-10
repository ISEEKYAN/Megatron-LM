#!/usr/bin/env bash
#SBATCH --job-name=ds4-grpo
#SBATCH --account=coreai_devtech_all
#SBATCH --partition=batch
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=14-00:00:00

set -euo pipefail

BASE_IMAGE=${BASE_IMAGE:?set BASE_IMAGE to the Hopper-qualified container}
MLITE_SRC=${MLITE_SRC:?set MLITE_SRC to a committed Megatron Lite snapshot}
MLITE_COMMIT=${MLITE_COMMIT:?set MLITE_COMMIT to the snapshot commit}
MEGATRON_ROOT=${MEGATRON_ROOT:?set MEGATRON_ROOT to a compatible MCore snapshot}
VERL_ROOT=${VERL_ROOT:?set VERL_ROOT to a committed VERL snapshot}
MLITE_SM90_SITE=${MLITE_SM90_SITE:?set MLITE_SM90_SITE to the SM90 training overlay}
DS4_VLLM_SITE=${DS4_VLLM_SITE:?set DS4_VLLM_SITE to the DS4 vLLM thin overlay}
DS4_VLLM_SHIM=${DS4_VLLM_SHIM:?set DS4_VLLM_SHIM to the Torch/vLLM ABI shim}
CHECKPOINT_DIR=${CHECKPOINT_DIR:?set CHECKPOINT_DIR to the official mixed checkpoint}
RUN_ROOT=${RUN_ROOT:?set RUN_ROOT to a fresh result directory}

for path in \
  "${BASE_IMAGE}" \
  "${MLITE_SRC}" \
  "${MEGATRON_ROOT}" \
  "${VERL_ROOT}" \
  "${MLITE_SM90_SITE}" \
  "${DS4_VLLM_SITE}" \
  "${DS4_VLLM_SHIM}" \
  "${CHECKPOINT_DIR}"; do
  [[ -e "${path}" ]] || {
    echo "required DS4 GRPO path does not exist: ${path}" >&2
    exit 2
  }
done
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
  readarray -t SLURM_HOSTS < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
  MASTER_ADDR=${SLURM_HOSTS[0]}
fi
RAY_PORT=${RAY_PORT:-6379}
RAY_DASHBOARD_PORT=${RAY_DASHBOARD_PORT:-8265}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
PHASE1_STEPS=${PHASE1_STEPS:-3}
TOTAL_STEPS=${TOTAL_STEPS:-6}
ACTOR_TP=${ACTOR_TP:-1}
ACTOR_PP=${ACTOR_PP:-4}
ACTOR_CP=${ACTOR_CP:-4}
ACTOR_EP=${ACTOR_EP:-8}
ROLLOUT_TP=${ROLLOUT_TP:-16}

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

if [[ "${IN_DS4_GRPO_CONTAINER:-0}" != "1" ]]; then
  [[ ! -e "${RUN_ROOT}" ]] || {
    echo "refusing to reuse existing DS4 GRPO run root: ${RUN_ROOT}" >&2
    exit 5
  }
  export BASE_IMAGE MLITE_SRC MLITE_COMMIT MEGATRON_ROOT VERL_ROOT
  export MLITE_SM90_SITE DS4_VLLM_SITE DS4_VLLM_SHIM CHECKPOINT_DIR RUN_ROOT
  export MASTER_ADDR RAY_PORT RAY_DASHBOARD_PORT GPUS_PER_NODE
  export PHASE1_STEPS TOTAL_STEPS ACTOR_TP ACTOR_PP ACTOR_CP ACTOR_EP ROLLOUT_TP
  exec srun --nodes="${SLURM_NNODES}" \
    --ntasks="${SLURM_NNODES}" \
    --ntasks-per-node=1 \
    --container-image="${BASE_IMAGE}" \
    --container-mounts=/lustre:/lustre \
    --container-workdir="${MLITE_SRC}" \
    env IN_DS4_GRPO_CONTAINER=1 bash "${SCRIPT_PATH}"
fi

export PATH=/usr/local/bin:/usr/bin:/bin
hash -r
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_NVLS_ENABLE=0
export PYTHONNOUSERSITE=1
export VLLM_USE_V1=1
export VERL_VLLM_FP8_QUANT_ENABLED=0
export LD_PRELOAD="${DS4_VLLM_SHIM}"
export PYTHONPATH="${DS4_VLLM_SITE}:${MLITE_SM90_SITE}/nvidia_cutlass_dsl/python_packages:${MLITE_SM90_SITE}:${VERL_ROOT}:${MEGATRON_ROOT}:${MLITE_SRC}/experimental/lite/examples/verl:${MLITE_SRC}/experimental/lite:${MLITE_SRC}"
export TMPDIR="/tmp/ds4-grpo-${SLURM_JOB_ID}-${SLURM_PROCID}"
export PYTHONPYCACHEPREFIX="${TMPDIR}/pycache"
export TORCHINDUCTOR_CACHE_DIR="${TMPDIR}/torchinductor"
export TRITON_CACHE_DIR="${TMPDIR}/triton"
mkdir -p "${PYTHONPYCACHEPREFIX}" "${TORCHINDUCTOR_CACHE_DIR}" "${TRITON_CACHE_DIR}"

NODE_RANK=${SLURM_PROCID}
DONE_FILE="${RUN_ROOT}/.ray_done"
ray stop --force >/dev/null 2>&1 || true

cleanup() {
  if [[ "${NODE_RANK}" == "0" ]]; then
    touch "${DONE_FILE}" 2>/dev/null || true
  fi
  ray stop --force >/dev/null 2>&1 || true
}
trap cleanup EXIT

if [[ "${NODE_RANK}" != "0" ]]; then
  NODE_ADDR=$(hostname -i)
  NODE_ADDR=${NODE_ADDR%% *}
  for attempt in $(seq 1 150); do
    if ray start --address="${MASTER_ADDR}:${RAY_PORT}" \
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
ray start --head \
  --node-ip-address="${MASTER_ADDR}" \
  --port="${RAY_PORT}" \
  --num-gpus="${GPUS_PER_NODE}" \
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

DATA_DIR="${RUN_ROOT}/data"
python "${MLITE_SRC}/experimental/lite/examples/verl/prepare_math_smoke_data.py" \
  --output-dir "${DATA_DIR}" \
  --train-size "${TRAIN_SIZE:-64}" \
  --test-size "${TEST_SIZE:-16}"

RUNTIME_ENV_JSON=$(python - <<'PY'
import json
import os

names = [
    "PYTHONPATH",
    "PATH",
    "LD_PRELOAD",
    "CUDA_DEVICE_MAX_CONNECTIONS",
    "NCCL_NVLS_ENABLE",
    "PYTHONNOUSERSITE",
    "VLLM_USE_V1",
    "VERL_VLLM_FP8_QUANT_ENABLED",
    "MLITE_SM90_SITE",
    "DS4_VLLM_SITE",
    "DS4_VLLM_SHIM",
    "MEGATRON_ROOT",
    "VERL_ROOT",
]
print(json.dumps({"env_vars": {name: os.environ[name] for name in names}}))
PY
)

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
  ray job submit \
    --address="http://127.0.0.1:${RAY_DASHBOARD_PORT}" \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- \
    env \
      MODEL_PATH="${CHECKPOINT_DIR}" \
      TRAIN_FILES="${DATA_DIR}/train.parquet" \
      VAL_FILES="${DATA_DIR}/test.parquet" \
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
