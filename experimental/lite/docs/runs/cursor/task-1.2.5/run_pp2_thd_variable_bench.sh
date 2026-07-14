#!/usr/bin/env bash
# PP2+THD bench with variable per-microbatch packed lengths (core boundary test).
set -euo pipefail

REPO_ROOT=${REPO_ROOT:?REPO_ROOT required}
DSA_SITE=${DSA_SITE:?DSA_SITE required}
WORK_DIR=${WORK_DIR:-$REPO_ROOT/experimental/lite/docs/runs/cursor/task-1.2.5}
OUTPUT_DIR=${OUTPUT_DIR:-$WORK_DIR/output}
mkdir -p "$OUTPUT_DIR"

export PYTHONPATH="$DSA_SITE:$REPO_ROOT/experimental/lite:$REPO_ROOT:${PYTHONPATH:-}"
export CUBLAS_WORKSPACE_CONFIG=${CUBLAS_WORKSPACE_CONFIG:-:4096:8}
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=${NVTE_ALLOW_NONDETERMINISTIC_ALGO:-0}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

MODEL_NAME=${MODEL_NAME:-qwen3_moe}
TP=${TP:-1}
PP=${PP:-2}
EP=${EP:-4}
CP=${CP:-1}
STEPS=${STEPS:-2}
WARMUP=${WARMUP:-0}
NUM_MB=${NUM_MB:-4}
# Replicate 1.13.20 dynamic-pack pressure: mismatched totals + variable sample counts.
MB_SCHEDULE=${MB_SCHEDULE:-'[[2206],[2213],[2145],[2148]]'}
TRUNCATE_LAYERS=${TRUNCATE_LAYERS:-4}
KEEP_EXPERTS=${KEEP_EXPERTS:-2}

product=$((TP * PP * CP * EP))
world=${WORLD_SIZE:-8}
if (( product != world )); then
  echo "parallel product mismatch: TP*PP*CP*EP=$product != WORLD_SIZE=$world" >&2
  exit 2
fi

echo "PP2_THD_VAR_BENCH_START model=$MODEL_NAME tp=$TP pp=$PP ep=$EP cp=$CP steps=$STEPS num_mb=$NUM_MB schedule=$MB_SCHEDULE"
echo "PP2_THD_VAR_BENCH_REPO commit=$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo missing)"

python3 - <<'PY'
import torch
import transformer_engine.pytorch as te
assert hasattr(te, "Linear")
print("PP2_THD_VAR_BENCH_PREFLIGHT_OK torch", torch.__version__, "cuda", torch.cuda.is_available())
PY

OUT_JSON="$OUTPUT_DIR/pp2_thd_variable_bench_${SLURM_JOB_ID:-local}.json"
torchrun --nproc_per_node="$world" \
  "$REPO_ROOT/experimental/lite/examples/bench/bench.py" \
  --backend mlite \
  --model-name "$MODEL_NAME" \
  --hf-path "" \
  --impl lite \
  --tp "$TP" --pp "$PP" --ep "$EP" --cp "$CP" \
  --steps "$STEPS" --warmup "$WARMUP" \
  --num-microbatches "$NUM_MB" \
  --use-thd \
  --skip-load-hf-weights \
  --truncate-layers "$TRUNCATE_LAYERS" \
  --keep-experts "$KEEP_EXPERTS" \
  --microbatch-seq-lens-json "$MB_SCHEDULE" \
  --output-json "$OUT_JSON"

echo "PP2_THD_VAR_BENCH_DONE rc=0 output=$OUT_JSON"
