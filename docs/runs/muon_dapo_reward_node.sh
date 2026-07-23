#!/usr/bin/env bash
# Qwen3-30B-A3B DAPO reward A/B: the only experimental variable is the
# Megatron optimizer algorithm (Muon or AdamW).
set -euo pipefail

ROLE=${1:?role head|worker}
HEAD_ADDR=${2:?head ip:port}
NNODES=${3:-2}
HEAD_IP=${HEAD_ADDR%:*}
RAY_PORT=${HEAD_ADDR##*:}

B=${B:-/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/bayan}
CODE_ROOT=${CODE_ROOT:-${B}/code}
RUN_ROOT=${RUN_ROOT:-${CODE_ROOT}/runtime/muon-dapo-reward-ab-30b}
VERL_ROOT=${VERL_ROOT:-${CODE_ROOT}/verl_muon_sft}
SNAPSHOT_ROOT=${SNAPSHOT_ROOT:-${CODE_ROOT}/runtime/muon-mbridge-386bf7af6-r2}
MLITE_ROOT=${MLITE_ROOT:-${SNAPSHOT_ROOT}/mlite}
MEGATRON_ROOT=${MEGATRON_ROOT:-${SNAPSHOT_ROOT}/megatron-d64}
EMERGING_OPT_ROOT=${EMERGING_OPT_ROOT:-${SNAPSHOT_ROOT}/emerging-optimizers}
MBRIDGE_ROOT=${MBRIDGE_ROOT:-${SNAPSHOT_ROOT}/mbridge-f5d6e2e}
NVRX_SITE=${NVRX_SITE:-${CODE_ROOT}/runtime/muon-p0p1-4d2a5b1df-mb-f5d6e2e-v2/nvrx-only-venv/lib/python3.12/site-packages}

MODEL_PATH=${MODEL_PATH:-${CODE_ROOT}/models/Qwen3-30B-A3B}
TRAIN_FILE=${TRAIN_FILE:-${CODE_ROOT}/verl_update_mcore/data/dapo-math-17k.parquet}
VAL_FILE=${VAL_FILE:-${CODE_ROOT}/verl_update_mcore/data/aime-2024.parquet}
OFFICIAL_RUN=${OFFICIAL_RUN:-${VERL_ROOT}/examples/grpo_trainer/run_qwen3_30b_a3b_megatron.sh}

export PATH=/usr/local/bin:/usr/bin:/bin
unset CONDA_DEFAULT_ENV CONDA_PREFIX PYTHONHOME VIRTUAL_ENV ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES || true
export PYTHONNOUSERSITE=1
export PYTHONPATH="${VERL_ROOT}:${MLITE_ROOT}/experimental/lite:${MEGATRON_ROOT}:${EMERGING_OPT_ROOT}:${MBRIDGE_ROOT}:${NVRX_SITE}"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_NVLS_ENABLE=0
export VLLM_USE_V1=1
export VLLM_ALLREDUCE_USE_SYMM_MEM=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export RAY_memory_monitor_refresh_ms=0
export HYDRA_FULL_ERROR=1
export PYTHONHASHSEED=42
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME="${RUN_ROOT}/hf-home"
export HF_DATASETS_CACHE="${RUN_ROOT}/hf-datasets-cache"
export TRITON_CACHE_DIR="/tmp/triton-muon-dapo-${SLURM_JOB_ID:-x}-${SLURMD_NODENAME:-node}"
export TORCHINDUCTOR_CACHE_DIR="/tmp/torchinductor-muon-dapo-${SLURM_JOB_ID:-x}-${SLURMD_NODENAME:-node}"
mkdir -p "${RUN_ROOT}" "${HF_HOME}" "${HF_DATASETS_CACHE}" "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}"

for path in \
  "${VERL_ROOT}" "${MLITE_ROOT}/experimental/lite" "${MEGATRON_ROOT}" \
  "${EMERGING_OPT_ROOT}" "${MBRIDGE_ROOT}" "${NVRX_SITE}" \
  "${MODEL_PATH}/config.json" "${TRAIN_FILE}" "${VAL_FILE}" "${OFFICIAL_RUN}"; do
  [[ -e "${path}" ]] || { echo "FATAL missing ${path}" >&2; exit 2; }
done

python - <<'PY'
import importlib.metadata
import json
import os

import emerging_optimizers
import megatron.core
import mbridge
import ray
import torch
import verl
import vllm

with open(os.path.join(os.environ["MODEL_PATH"], "config.json")) as stream:
    model_type = json.load(stream)["model_type"]
assert model_type == "qwen3_moe", model_type
assert importlib.metadata.version("nvidia-resiliency-ext") == "0.6.0"
print(
    "MUON_DAPO_IMPORTS_OK",
    f"torch={torch.__version__}",
    f"vllm={vllm.__version__}",
    f"model_type={model_type}",
    f"verl={verl.__file__}",
    f"megatron={megatron.core.__file__}",
    f"emerging={emerging_optimizers.__file__}",
    f"mbridge={mbridge.__file__}",
    f"ray={ray.__version__}",
    flush=True,
)
PY

RAY_PORTS="--min-worker-port=21000 --max-worker-port=31999 --runtime-env-agent-port=19435 --metrics-export-port=34001 --dashboard-agent-grpc-port=34002"
if [[ "${ROLE}" == head ]]; then
  ray start --head --node-ip-address="${HEAD_IP}" --port="${RAY_PORT}" --num-gpus=8 ${RAY_PORTS} --disable-usage-stats
  export RAY_ADDRESS="${HEAD_IP}:${RAY_PORT}"
  python - "${NNODES}" <<'PY'
import sys
import time
import ray

expected = int(sys.argv[1]) * 8
ray.init(address="auto", ignore_reinit_error=True)
deadline = time.time() + 600
while time.time() < deadline:
    available = int(ray.cluster_resources().get("GPU", 0))
    if available >= expected:
        print(f"MUON_DAPO_RAY_READY gpus={available} expected={expected}", flush=True)
        break
    print(f"MUON_DAPO_RAY_WAIT gpus={available} expected={expected}", flush=True)
    time.sleep(5)
else:
    raise RuntimeError(f"Ray registered only {available}/{expected} GPUs")
ray.shutdown()
PY

  export DATA_DIR="${VERL_ROOT}"
  export MODEL_PATH TRAIN_FILE VAL_FILE
  export NNODES NGPUS_PER_NODE=8
  export TRAIN_FILES="${TRAIN_FILE}" VAL_FILES="${VAL_FILE}"
  export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-16}
  export PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-16}
  export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
  export MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-1024}
  export PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-4096}
  export ACTOR_LR=${ACTOR_LR:-1e-5}
  export ACTOR_TP=${ACTOR_TP:-2} ACTOR_PP=${ACTOR_PP:-1} ACTOR_VPP=1 ACTOR_EP=${ACTOR_EP:-8} ACTOR_CP=1
  export REF_TP=2 REF_PP=1 REF_VPP=1 REF_EP=8 REF_CP=1
  export ALL_OFFLOAD=True
  export ROLLOUT_TP=${ROLLOUT_TP:-4}
  export ROLLOUT_N=${ROLLOUT_N:-4}
  export ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.55}
  export ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-64}
  export ROLLOUT_MAX_NUM_BATCHED_TOKENS=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-4096}
  export ROLLOUT_MAX_MODEL_LEN=${ROLLOUT_MAX_MODEL_LEN:-2048}
  export REF_LOG_PROB_MAX_TOKEN_LEN_PER_GPU=4096
  export REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=1
  export ROLLOUT_LOG_PROB_MAX_TOKEN_LEN_PER_GPU=4096
  export ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=1
  export TOTAL_EPOCHS=1 SAVE_FREQ=-1 TEST_FREQ=-1

  for optimizer in muon adam; do
    arm_root="${RUN_ROOT}/job-${SLURM_JOB_ID}/${optimizer}"
    mkdir -p "${arm_root}"
    export VERL_FILE_LOGGER_ROOT="${arm_root}"
    export PROJECT_NAME=muon-dapo-reward-ab-30b
    export EXPERIMENT_NAME="qwen3-30b-dapo-${optimizer}"
    echo "MUON_DAPO_ARM_START optimizer=${optimizer} root=${arm_root}" | tee "${arm_root}/arm.log"

    optim_args=(
      "actor_rollout_ref.actor.optim.optimizer=${optimizer}"
      actor_rollout_ref.actor.optim.lr_warmup_steps=0
      actor_rollout_ref.actor.optim.lr_warmup_init=0
      actor_rollout_ref.actor.optim.lr_decay_style=constant
      actor_rollout_ref.actor.optim.min_lr=${ACTOR_LR}
      'actor_rollout_ref.actor.optim.betas=[0.9,0.95]'
      actor_rollout_ref.actor.optim.weight_decay=0.1
      actor_rollout_ref.actor.optim.clip_grad=1.0
      actor_rollout_ref.actor.optim.use_precision_aware_optimizer=False
      actor_rollout_ref.actor.optim.main_grads_dtype=fp32
      actor_rollout_ref.actor.optim.exp_avg_dtype=fp32
      actor_rollout_ref.actor.optim.exp_avg_sq_dtype=fp32
    )
    if [[ "${optimizer}" == muon ]]; then
      optim_args+=(
        actor_rollout_ref.actor.optim.use_layer_wise_distributed_optimizer=True
        actor_rollout_ref.actor.optim.use_layer_wise_param_layout=True
        actor_rollout_ref.actor.optim.muon_tp_mode=blockwise
        actor_rollout_ref.actor.optim.muon_scalar_optimizer=adam
        actor_rollout_ref.actor.optim.muon_num_ns_steps=5
      )
    fi

    bash "${OFFICIAL_RUN}" \
      "${optim_args[@]}" \
      trainer.total_training_steps=${TOTAL_TRAINING_STEPS:-5} \
      actor_rollout_ref.actor.megatron.seed=42 \
      actor_rollout_ref.rollout.seed=42 \
      trainer.resume_mode=disable \
      trainer.resume_from_path=null \
      trainer.val_before_train=False \
      trainer.logger='["console","file"]' \
      "trainer.default_local_dir=${arm_root}" \
      2>&1 | tee -a "${arm_root}/arm.log"
    echo "MUON_DAPO_ARM_DONE optimizer=${optimizer} rc=0" | tee -a "${arm_root}/arm.log"

    ray status
    rm -rf "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}"
    mkdir -p "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}"
  done
  echo "MUON_DAPO_AB_DONE job=${SLURM_JOB_ID} rc=0"
else
  ray start --address="${HEAD_IP}:${RAY_PORT}" --num-gpus=8 ${RAY_PORTS} --block
fi
