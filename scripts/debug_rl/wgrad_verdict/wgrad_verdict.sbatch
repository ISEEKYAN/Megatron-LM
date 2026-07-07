#!/bin/bash
#SBATCH -A coreai_devtech_all
#SBATCH -p batch
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --mem=0
#SBATCH -J wgrad_verdict
#SBATCH --time=03:00:00
#SBATCH --output=/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/bayan/code/qwen35_dapo_g2/logs/wgrad_verdict_%j.out
set -euo pipefail

# TASK-1.1.15.11 判决实验: raw autograd wgrad(无优化器/无fsdp2包裹) + fp64地面真值定罪.
# 三情景 × 两后端: long_w(8192,weighted PPO-like) / long_u(8192,uniform CE) / short_w(256 packed,weighted)

BASE=/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/bayan/code
IMG=$BASE/verl_optimize/verl.vllm023.sqsh
REPO=${WGRAD_REPO:-$BASE/qwen35_dapo_g2/wt-task11-wgrad}
MCORE=$BASE/qwen35_dapo_g2/mcore-gdn-family
MBRIDGE=$BASE/qwen35_dapo_ab/mbridge_main
CP_SITE=$BASE/mlite-newenv-cache/qwen35-cp-overlay-20260613/site
HF=/lustre/fsw/portfolios/coreai/users/bayan/code/models/Qwen3.5-35B-A3B
OUT=$BASE/qwen35_dapo_g2/diagnosis/task_1_1_15_11/wgrad_verdict_${SLURM_JOB_ID}
LAYERS=${TRUNCATE_LAYERS:-12}
mkdir -p "$OUT"

SA=(--container-image="$IMG" --container-mounts=/lustre:/lustre --export=ALL)
BASE_ENV=(env PYTHONNOUSERSITE=1
  PYTHONPATH="$REPO/experimental/lite:$REPO:$MCORE:$MBRIDGE:$CP_SITE:/vllm"
  LD_LIBRARY_PATH="$CP_SITE/tvm_ffi/lib:${LD_LIBRARY_PATH:-}"
  MLITE_CORRECTNESS_GRAD_DETAILS=1
  MLITE_CORRECTNESS_NONDETERMINISTIC=1)

COMMON=(
  --hf-path "$HF" --model-name qwen3_5
  --tp 1 --etp 1 --ep 8 --pp 1 --cp 1
  --steps 1 --num-microbatches 1 --seed 42
  --truncate-layers "$LAYERS" --disable-mtp
  --same-data-across-dp --skip-weight-hash --use-thd --no-optimizer
)

run_case () {
  local scenario=$1 backend=$2 seq_len=$3 weighted=$4 packed=$5
  local dump="$OUT/dump_${scenario}_${backend}"
  local patterns
  if [[ "$backend" == mlite ]]; then
    patterns="linear_attn.o_proj,full_attn.proj"
    extra=(--impl-cfg-json '{"mount_vision_model":false,"optimizer":null}')
  else
    patterns="out_proj,linear_proj"
    extra=()
  fi
  local env_extra=(MLITE_CORRECTNESS_WEIGHTED_LOGPROB="$weighted"
    MLITE_WGRAD_DUMP_DIR="$dump" MLITE_WGRAD_DUMP_PATTERNS="$patterns")
  if [[ -n "$packed" ]]; then
    env_extra+=(MLITE_BENCH_PACKED_SEQ_LENS="$packed")
  fi
  echo "=== CASE scenario=$scenario backend=$backend seq=$seq_len weighted=$weighted packed=${packed:-none} ==="
  srun "${SA[@]}" "${BASE_ENV[@]}" "${env_extra[@]}" torchrun --nproc_per_node 8 \
    "$REPO/experimental/lite/examples/bench/correctness.py" run \
    --backend "$backend" "${COMMON[@]}" --seq-len "$seq_len" "${extra[@]}" \
    --output-json "$OUT/${scenario}_${backend}.json" > "$OUT/${scenario}_${backend}.log" 2>&1
  echo "=== CASE $scenario $backend rc=$? ==="
}

for backend in mlite mbridge; do
  run_case long_w  "$backend" 8192 1 ""
  run_case long_u  "$backend" 8192 0 ""
  run_case short_w "$backend" 256  1 "17,31,5,64,23,47,69"
done

for scenario in long_w long_u short_w; do
  echo "=== ANALYZE $scenario ==="
  srun "${SA[@]}" "${BASE_ENV[@]}" python \
    "$REPO/scripts/debug_rl/wgrad_verdict/analyze_wgrad_dumps.py" \
    --mlite "$OUT/dump_${scenario}_mlite" --mcore "$OUT/dump_${scenario}_mbridge" \
    --output-json "$OUT/verdict_${scenario}.json" > "$OUT/verdict_${scenario}.log" 2>&1 || true
  tail -30 "$OUT/verdict_${scenario}.log"
done
echo "WGRAD_VERDICT_OUTPUT=$OUT"
