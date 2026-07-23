#!/usr/bin/env bash
#SBATCH --job-name=muon-dapo-config
#SBATCH --account=coreai_devtech_all
#SBATCH --partition=cpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=00:15:00
#SBATCH --output=/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/bayan/code/runtime/muon-dapo-reward-ab-30b/config-%j.out
set -euo pipefail

B=/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/bayan
CODE_ROOT=${CODE_ROOT:-${B}/code}
RUN_ROOT=${RUN_ROOT:-${CODE_ROOT}/runtime/muon-dapo-reward-ab-30b}
VERL_ROOT=${VERL_ROOT:-${CODE_ROOT}/verl_muon_sft}
SNAPSHOT_ROOT=${SNAPSHOT_ROOT:-${CODE_ROOT}/runtime/muon-mbridge-386bf7af6-r2}
CONTAINER_IMAGE=${CONTAINER_IMAGE:-/lustre/fsw/portfolios/coreai/users/bayan/code/megatron_lite/verl.complete.sqsh}
MODEL_PATH=${MODEL_PATH:-${CODE_ROOT}/models/Qwen3-30B-A3B}
TRAIN_FILE=${TRAIN_FILE:-${CODE_ROOT}/verl_update_mcore/data/dapo-math-17k.parquet}
VAL_FILE=${VAL_FILE:-${CODE_ROOT}/verl_update_mcore/data/aime-2024.parquet}
OFFICIAL_RUN=${OFFICIAL_RUN:-${VERL_ROOT}/examples/grpo_trainer/run_qwen3_30b_a3b_megatron.sh}

mkdir -p "${RUN_ROOT}"
srun --container-image="${CONTAINER_IMAGE}" --container-mounts=/lustre:/lustre --export=ALL \
  /bin/bash -c '
set -euo pipefail
export PATH=/usr/local/bin:/usr/bin:/bin
unset CONDA_DEFAULT_ENV CONDA_PREFIX PYTHONHOME VIRTUAL_ENV ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES || true
export PYTHONNOUSERSITE=1
export PYTHONPATH="'"${VERL_ROOT}"':'"${SNAPSHOT_ROOT}"'/mlite/experimental/lite:'"${SNAPSHOT_ROOT}"'/megatron-d64:'"${SNAPSHOT_ROOT}"'/emerging-optimizers:'"${SNAPSHOT_ROOT}"'/mbridge-f5d6e2e:'"${CODE_ROOT}"'/runtime/muon-p0p1-4d2a5b1df-mb-f5d6e2e-v2/nvrx-only-venv/lib/python3.12/site-packages"
export MODEL_PATH="'"${MODEL_PATH}"'"

python - <<"PY"
import json
import os
import ray
import vllm
import verl
import megatron.core
import emerging_optimizers
import mbridge

with open(os.path.join(os.environ["MODEL_PATH"], "config.json")) as stream:
    assert json.load(stream)["model_type"] == "qwen3_moe"
ray.init(
    include_dashboard=False,
    num_cpus=2,
    runtime_env={"env_vars": {"MUON_DAPO_CONFIG_GATE": "1"}},
)
@ray.remote
def env_probe():
    import os
    import verl.trainer.main_ppo
    return os.environ.get("MUON_DAPO_CONFIG_GATE"), verl.trainer.main_ppo.__file__
assert ray.get(env_probe.remote())[0] == "1"
ray.shutdown()
print("MUON_DAPO_RAY_RUNTIME_ENV_OK")
PY

common=(
  DATA_DIR="'"${VERL_ROOT}"'"
  MODEL_PATH="'"${MODEL_PATH}"'"
  TRAIN_FILES="'"${TRAIN_FILE}"'"
  VAL_FILES="'"${VAL_FILE}"'"
  NNODES=2 NGPUS_PER_NODE=8
  TRAIN_BATCH_SIZE=16 PPO_MINI_BATCH_SIZE=16
  MAX_PROMPT_LENGTH=1024 MAX_RESPONSE_LENGTH=1024 PPO_MAX_TOKEN_LEN_PER_GPU=4096
  ACTOR_LR=1e-5 ACTOR_TP=2 ACTOR_PP=1 ACTOR_VPP=1 ACTOR_EP=8 ACTOR_CP=1
  REF_TP=2 REF_PP=1 REF_VPP=1 REF_EP=8 REF_CP=1
  ROLLOUT_TP=4 ROLLOUT_N=4 ROLLOUT_GPU_MEM_UTIL=0.55
  ROLLOUT_MAX_NUM_SEQS=64 ROLLOUT_MAX_NUM_BATCHED_TOKENS=4096 ROLLOUT_MAX_MODEL_LEN=2048
  REF_LOG_PROB_MAX_TOKEN_LEN_PER_GPU=4096 REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=1
  ROLLOUT_LOG_PROB_MAX_TOKEN_LEN_PER_GPU=4096 ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=1
  TOTAL_EPOCHS=1 SAVE_FREQ=-1 TEST_FREQ=-1
)
for optimizer in muon adam; do
  args=(
    actor_rollout_ref.actor.optim.optimizer=${optimizer}
    actor_rollout_ref.actor.optim.lr_warmup_steps=0
    actor_rollout_ref.actor.optim.lr_decay_style=constant
    actor_rollout_ref.actor.optim.min_lr=1e-5
    actor_rollout_ref.actor.optim.use_precision_aware_optimizer=False
    actor_rollout_ref.actor.optim.main_grads_dtype=fp32
    actor_rollout_ref.actor.optim.exp_avg_dtype=fp32
    actor_rollout_ref.actor.optim.exp_avg_sq_dtype=fp32
    trainer.total_training_steps=5
    trainer.seed=42
    actor_rollout_ref.rollout.seed=42
    trainer.resume_mode=disable
    trainer.val_before_train=False
    "trainer.logger=[console,file]"
    --cfg job
    --resolve
  )
  if [[ ${optimizer} == muon ]]; then
    args+=(
      actor_rollout_ref.actor.optim.use_layer_wise_distributed_optimizer=True
      actor_rollout_ref.actor.optim.use_layer_wise_param_layout=True
      actor_rollout_ref.actor.optim.muon_tp_mode=blockwise
      actor_rollout_ref.actor.optim.muon_scalar_optimizer=adam
      actor_rollout_ref.actor.optim.muon_num_ns_steps=5
    )
  fi
  env "${common[@]}" bash "'"${OFFICIAL_RUN}"'" "${args[@]}" >"'"${RUN_ROOT}"'/resolved-${optimizer}.yaml"
  grep -q "optimizer: ${optimizer}" "'"${RUN_ROOT}"'/resolved-${optimizer}.yaml"
  grep -q "reward_manager: dapo" "'"${RUN_ROOT}"'/resolved-${optimizer}.yaml"
done
echo "MUON_DAPO_CONFIG_ONLY_OK job=${SLURM_JOB_ID}"
'
