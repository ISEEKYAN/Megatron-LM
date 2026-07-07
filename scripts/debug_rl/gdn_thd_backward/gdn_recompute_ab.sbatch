#!/bin/bash
#SBATCH -A coreai_devtech_all
#SBATCH -p batch
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=0
#SBATCH -J gdn_recompute_ab
#SBATCH --time=00:30:00
#SBATCH --output=/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/bayan/code/qwen35_dapo_g2/logs/gdn_recompute_ab_%j.out
set -euo pipefail

BASE=/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/bayan/code
IMG=$BASE/verl_optimize/verl.vllm023.sqsh
REPO=$BASE/qwen35_dapo_g2/wt-gdn-backward
CP_SITE=$BASE/mlite-newenv-cache/qwen35-cp-overlay-20260613/site
HF=/lustre/fsw/portfolios/coreai/users/bayan/code/models/Qwen3.5-35B-A3B
OUT=$BASE/qwen35_dapo_g2/diagnosis/task_1_1_15_9/gdn_recompute_ab_${SLURM_JOB_ID}

COMMON=(
  --hf-path "$HF" --model-name qwen3_5
  --tp 1 --etp 1 --ep 1 --pp 1 --cp 1
  --steps 1 --num-microbatches 1 --seq-len 256 --seed 42
  --truncate-layers 1 --keep-experts 2 --disable-mtp
  --same-data-across-dp --no-optimizer --skip-weight-hash --use-thd
)
STAGE_PROBES='{"mlite":["module.layers.0.linear_attn::_prepare_qkv","module.layers.0.linear_attn::_compute_g_and_beta","module.layers.0.linear_attn::_gated_delta_rule","module.layers.0.linear_attn.o_proj","module.layers.0"]}'
COMMON+=(--activation-probes-json "$STAGE_PROBES")

mkdir -p "$OUT"
SA=(--container-image="$IMG" --container-mounts=/lustre:/lustre --export=ALL)
ENV=(env PYTHONNOUSERSITE=1
  PYTHONPATH="$REPO/experimental/lite:$REPO:$CP_SITE:/vllm"
  MLITE_CORRECTNESS_GRAD_DETAILS=1
  MLITE_CORRECTNESS_NONDETERMINISTIC=1
  MLITE_BENCH_PACKED_SEQ_LENS=17,31,5,64,23,47,69)

for variant in plain full; do
  IMPL='{"mount_vision_model":false}'
  if [[ "$variant" == full ]]; then
    IMPL='{"mount_vision_model":false,"recompute":["full"]}'
  fi
  srun "${SA[@]}" "${ENV[@]}" torchrun --nproc_per_node 1 \
    "$REPO/experimental/lite/examples/bench/correctness.py" run \
    --backend mlite "${COMMON[@]}" --impl-cfg-json "$IMPL" \
    --output-json "$OUT/${variant}.json" > "$OUT/${variant}.log" 2>&1
done

srun "${SA[@]}" "${ENV[@]}" python \
  "$REPO/experimental/lite/examples/bench/correctness.py" compare \
  "$OUT/plain.json" "$OUT/full.json" --output-json "$OUT/compare.json"
echo "GDN_RECOMPUTE_AB_OUTPUT=$OUT"
