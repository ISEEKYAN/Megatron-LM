#!/bin/bash
#SBATCH -A coreai_devtech_all
#SBATCH -p batch
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --mem=0
#SBATCH -J gdn_ep8_depth
#SBATCH --time=01:00:00
#SBATCH --output=/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/bayan/code/qwen35_dapo_g2/logs/gdn_ep8_depth_%j.out
set -euo pipefail

BASE=/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/bayan/code
IMG=$BASE/verl_optimize/verl.vllm023.sqsh
REPO=$BASE/qwen35_dapo_g2/wt-gdn-backward
MCORE=$BASE/qwen35_dapo_g2/mcore-gdn-family
MBRIDGE=$BASE/qwen35_dapo_ab/mbridge_main
CP_SITE=$BASE/mlite-newenv-cache/qwen35-cp-overlay-20260613/site
HF=/lustre/fsw/portfolios/coreai/users/bayan/code/models/Qwen3.5-35B-A3B
OUT=$BASE/qwen35_dapo_g2/diagnosis/task_1_1_15_9/gdn_ep8_depth_${SLURM_JOB_ID}
LAYERS=${TRUNCATE_LAYERS:-40}

COMMON=(
  --hf-path "$HF" --model-name qwen3_5
  --tp 1 --etp 1 --ep 8 --pp 1 --cp 1
  --steps 1 --num-microbatches 1 --seq-len 256 --seed 42
  --truncate-layers "$LAYERS" --disable-mtp
  --same-data-across-dp --skip-weight-hash --use-thd
)
if [[ "${MLITE_CORRECTNESS_WITH_OPTIMIZER:-0}" != 1 ]]; then
  COMMON+=(--no-optimizer)
fi
MLITE_PROBES=
MBRIDGE_PROBES=
for ((i=0; i<LAYERS; i++)); do
  [[ -n "$MLITE_PROBES" ]] && MLITE_PROBES+=,
  [[ -n "$MBRIDGE_PROBES" ]] && MBRIDGE_PROBES+=,
  MLITE_PROBES+="\"module.layers.$i\""
  MBRIDGE_PROBES+="\"module.module.language_model.decoder.layers.$i\""
done
STAGE_PROBES="{\"mlite\":[$MLITE_PROBES],\"mbridge\":[$MBRIDGE_PROBES]}"
COMMON+=(--activation-probes-json "$STAGE_PROBES")

mkdir -p "$OUT"
SA=(--container-image="$IMG" --container-mounts=/lustre:/lustre --export=ALL)
ENV=(env PYTHONNOUSERSITE=1
  PYTHONPATH="$REPO/experimental/lite:$REPO:$MCORE:$MBRIDGE:$CP_SITE:/vllm"
  LD_LIBRARY_PATH="$CP_SITE/tvm_ffi/lib:${LD_LIBRARY_PATH:-}"
  MLITE_CORRECTNESS_GRAD_DETAILS=1
  MLITE_CORRECTNESS_NONDETERMINISTIC=1
  MLITE_CORRECTNESS_WEIGHTED_LOGPROB="${MLITE_CORRECTNESS_WEIGHTED_LOGPROB:-0}"
  MLITE_GRAD_NORM_DEBUG=1
  MLITE_GRAD_FAMILY_DEBUG=1
  MLITE_BENCH_PACKED_SEQ_LENS=17,31,5,64,23,47,69)

for backend in mlite mbridge; do
  BACKEND_ARGS=()
  if [[ "$backend" == mlite ]]; then
    if [[ "${MLITE_DISABLE_RECOMPUTE:-0}" == 1 ]]; then
      BACKEND_ARGS=(--impl-cfg-json '{"mount_vision_model":false,"optimizer":"fsdp2"}')
    else
      BACKEND_ARGS=(--impl-cfg-json '{"mount_vision_model":false,"recompute":["full"],"optimizer":"fsdp2"}')
    fi
  fi
  srun "${SA[@]}" "${ENV[@]}" torchrun --nproc_per_node 8 \
    "$REPO/experimental/lite/examples/bench/correctness.py" run \
    --backend "$backend" "${COMMON[@]}" "${BACKEND_ARGS[@]}" \
    --output-json "$OUT/${backend}.json" > "$OUT/${backend}.log" 2>&1
done

srun "${SA[@]}" "${ENV[@]}" python \
  "$REPO/experimental/lite/examples/bench/correctness.py" compare \
  "$OUT/mlite.json" "$OUT/mbridge.json" --output-json "$OUT/compare.json"
echo "GDN_EP8_DEPTH_OUTPUT=$OUT"
