#!/usr/bin/env bash
# One-GPU Qwen3.5 LoRA SFT smoke through MLite dist_opt.

#SBATCH --job-name=lora-q35-sft
#SBATCH --account=coreai_devtech_all
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --output=/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/bayan/code/runtime/lora-qwen35-sft/slurm-%j.out

set -euo pipefail

B=/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/bayan
RUN_ROOT=${B}/code/runtime/lora-qwen35-sft
BASE_IMAGE=/home/bayan/code/verl_optimize/verl.vllm023.sqsh
MLITE_SRC=${MLITE_SRC:?submit with MLITE_SRC pointing to the staged source tree}
MEGATRON_ROOT=${B}/code/runtime/ds4-csacp-parity-eaa5b486d/mcore
QWEN35_FLA_SITE=${B}/code/mlite-newenv-cache/qwen35-cp-overlay-20260613/site
SCRIPT=${MLITE_SRC}/experimental/lite/docs/runs/codex/lora-qwen35-sft/lora_qwen35_sft_smoke.py

for path in "${BASE_IMAGE}" "${MLITE_SRC}" "${MEGATRON_ROOT}" "${QWEN35_FLA_SITE}" "${SCRIPT}"; do
  [[ -e "${path}" ]] || { echo "MISSING ${path}" >&2; exit 2; }
done

mkdir -p "${RUN_ROOT}/wandb" "${RUN_ROOT}/cache"
export WANDB_DIR="${RUN_ROOT}/wandb"
export WANDB_RUN_NAME="qwen35-lora-dist-opt-${SLURM_JOB_ID}"
PYTHONPATH_IN="${QWEN35_FLA_SITE}:${MEGATRON_ROOT}:${MLITE_SRC}/experimental/lite:${MLITE_SRC}"

exec srun --unbuffered --export=ALL \
  --nodes=1 \
  --ntasks=1 \
  --gres=gpu:1 \
  --container-image="${BASE_IMAGE}" \
  --container-mounts=/lustre:/lustre \
  --container-workdir="${MLITE_SRC}" \
  env \
    PATH=/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin \
    CC=/usr/bin/gcc \
    CXX=/usr/bin/g++ \
    BASH_ENV=/dev/null \
    ENV=/dev/null \
    CUDA_HOME=/usr/local/cuda \
    CUDA_PATH=/usr/local/cuda \
    PYTHONNOUSERSITE=1 \
    PYTHONUNBUFFERED=1 \
    CUDA_VISIBLE_DEVICES=0 \
    CUDA_DEVICE_MAX_CONNECTIONS=1 \
    TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
    XDG_CACHE_HOME="${RUN_ROOT}/cache" \
    PYTHONPATH="${PYTHONPATH_IN}" \
    bash --noprofile --norc -c \
      "nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader; \
       export LD_LIBRARY_PATH=${QWEN35_FLA_SITE}/tvm_ffi/lib:\$LD_LIBRARY_PATH; \
       exec torchrun --standalone --nproc_per_node=1 ${SCRIPT}"
