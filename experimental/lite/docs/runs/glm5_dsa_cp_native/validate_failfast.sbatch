#!/bin/bash
#SBATCH -A coreai_devtech_all
#SBATCH -p batch
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --mem=0
#SBATCH --time=01:00:00
#SBATCH -J glm52_cpnative_validate
#SBATCH --output=/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/bayan/code/llmrl_runs/%x_%j.out
set -euxo pipefail
B=/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/bayan
RUN=$B/code/runtime/glm52-cpnative-validate
BASE_IMAGE=$B/code/env/pytorch_26.04-py3.sqsh
MLITE_SRC=$B/code/runtime/glm52-mlite-src
OUT=$B/code/llmrl_runs/glm52_validate_result_${SLURM_JOB_ID}.json
ROUNDTRIP_OUT=$B/code/llmrl_runs/glm52_weight_roundtrip_${SLURM_JOB_ID}.json
VALIDATE_ARGS=
if [[ ${ONLY_THD:-0} == 1 ]]; then
  VALIDATE_ARGS=--only-thd
fi
srun --nodes=1 --ntasks=1 --gpus-per-node=8 \
  --container-image="$BASE_IMAGE" \
  --container-mounts=/lustre:/lustre \
  --container-workdir="$MLITE_SRC/experimental/lite" \
  bash -lc "set -euo pipefail; source $RUN/env-common.sh; \
    echo '===== 精度/显存/CP+THD (8-rank CP8 validate) ====='; \
    torchrun --standalone --nproc_per_node=8 docs/runs/glm5_dsa_cp_native/validate_dsa_cp.py --seq-lens 4096,4104,8192,8200 --thd-seq-lens 4096,4104 --hf-seq-len 4096 --warmup 1 --steps 2 --output $OUT $VALIDATE_ARGS; \
    echo '===== GPU bf16 load/save/export HF round-trip (IndexShare off/on) ====='; \
    python docs/runs/glm5_dsa_cp_native/validate_weight_roundtrip.py --output $ROUNDTRIP_OUT; \
    echo '===== load/save/export HF (CPU static 套件) ====='; \
    python -m pytest -q tests/unit/model/test_glm5_lite_static.py"
