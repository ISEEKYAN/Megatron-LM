#!/bin/bash
#SBATCH -A coreai_devtech_all
#SBATCH -p batch
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --mem=0
#SBATCH -J aux_verdict
#SBATCH --time=01:00:00
#SBATCH --output=/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/bayan/code/qwen35_dapo_g2/logs/aux_verdict_%j.out
set -euo pipefail

# TASK-1.1.15.11 一发式定罪: mlite router_aux_loss_coef=0 后,weighted长序列wgrad是否塌回MCore水平.
# 对照组(mlite默认aux=0.001 与 mbridge)复用 wgrad_verdict_${REF_JOB} 的 long_w dump.

BASE=/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/bayan/code
IMG=$BASE/verl_optimize/verl.vllm023.sqsh
REPO=${WGRAD_REPO:-$BASE/qwen35_dapo_g2/wt-task11-wgrad}
MCORE=$BASE/qwen35_dapo_g2/mcore-gdn-family
MBRIDGE=$BASE/qwen35_dapo_ab/mbridge_main
CP_SITE=$BASE/mlite-newenv-cache/qwen35-cp-overlay-20260613/site
HF=/lustre/fsw/portfolios/coreai/users/bayan/code/models/Qwen3.5-35B-A3B
OUT=$BASE/qwen35_dapo_g2/diagnosis/task_1_1_15_11/aux_verdict_${SLURM_JOB_ID}
REF_JOB=${REF_JOB:-13521612}
REF=$BASE/qwen35_dapo_g2/diagnosis/task_1_1_15_11/wgrad_verdict_${REF_JOB}
mkdir -p "$OUT"

SA=(--container-image="$IMG" --container-mounts=/lustre:/lustre --export=ALL)
BASE_ENV=(env PYTHONNOUSERSITE=1
  PYTHONPATH="$REPO/experimental/lite:$REPO:$MCORE:$MBRIDGE:$CP_SITE:/vllm"
  LD_LIBRARY_PATH="$CP_SITE/tvm_ffi/lib:${LD_LIBRARY_PATH:-}"
  MLITE_CORRECTNESS_GRAD_DETAILS=1
  MLITE_CORRECTNESS_NONDETERMINISTIC=1
  MLITE_CORRECTNESS_WEIGHTED_LOGPROB=1)

srun "${SA[@]}" "${BASE_ENV[@]}" \
  MLITE_WGRAD_DUMP_DIR="$OUT/dump_long_w_mlite_aux0" \
  MLITE_WGRAD_DUMP_PATTERNS="linear_attn.o_proj,full_attn.proj" \
  torchrun --nproc_per_node 8 \
  "$REPO/experimental/lite/examples/bench/correctness.py" run \
  --backend mlite \
  --hf-path "$HF" --model-name qwen3_5 \
  --tp 1 --etp 1 --ep 8 --pp 1 --cp 1 \
  --steps 1 --num-microbatches 1 --seed 42 --seq-len 8192 \
  --truncate-layers 12 --disable-mtp \
  --same-data-across-dp --skip-weight-hash --use-thd --no-optimizer \
  --impl-cfg-json '{"mount_vision_model":false,"optimizer":null,"router_aux_loss_coef":0.0}' \
  --output-json "$OUT/long_w_mlite_aux0.json" > "$OUT/long_w_mlite_aux0.log" 2>&1
echo "=== CASE long_w mlite_aux0 rc=$? ==="

echo "=== ANALYZE aux0-vs-mbridge ==="
srun "${SA[@]}" "${BASE_ENV[@]}" python \
  "$REPO/scripts/debug_rl/wgrad_verdict/analyze_wgrad_dumps.py" \
  --mlite "$OUT/dump_long_w_mlite_aux0" --mcore "$REF/dump_long_w_mbridge" \
  --output-json "$OUT/verdict_aux0_vs_mbridge.json" > "$OUT/verdict_aux0_vs_mbridge.log" 2>&1 || true
tail -25 "$OUT/verdict_aux0_vs_mbridge.log"

echo "=== ANALYZE aux0-vs-mlite-default ==="
srun "${SA[@]}" "${BASE_ENV[@]}" python \
  "$REPO/scripts/debug_rl/wgrad_verdict/analyze_wgrad_dumps.py" \
  --mlite "$REF/dump_long_w_mlite" --mcore "$OUT/dump_long_w_mlite_aux0" \
  --output-json "$OUT/verdict_auxdefault_vs_aux0.json" > "$OUT/verdict_auxdefault_vs_aux0.log" 2>&1 || true
tail -25 "$OUT/verdict_auxdefault_vs_aux0.log"
echo "AUX_VERDICT_OUTPUT=$OUT"
