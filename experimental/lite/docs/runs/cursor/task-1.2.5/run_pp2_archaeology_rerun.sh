#!/usr/bin/env bash
# Archaeological PP boundary rerun (bayan 2026-07-14): replay historically-green
# configs on today's HEAD before variable-length pressure tests.
set -euo pipefail

REPO_ROOT=${REPO_ROOT:?REPO_ROOT required}
STAGING=${STAGING:-}
DSA_SITE=${DSA_SITE:?DSA_SITE required}
WORK_DIR=${WORK_DIR:-${STAGING:-$REPO_ROOT}/experimental/lite/docs/runs/cursor/task-1.2.5}
OUTPUT_DIR=${OUTPUT_DIR:-$WORK_DIR/output}
mkdir -p "$OUTPUT_DIR"

if [[ -n "$STAGING" ]]; then
  cp -f "$STAGING/experimental/lite/examples/bench/bench.py" \
    "$REPO_ROOT/experimental/lite/examples/bench/bench.py"
  cp -f "$STAGING/experimental/lite/examples/bench/session.py" \
    "$REPO_ROOT/experimental/lite/examples/bench/session.py"
fi

# srun --export=ALL leaks the host login-node PATH into the container, which puts
# bayan's host miniforge python3 (broken/uncompiled Transformer Engine) ahead of
# the container's own python3.12. Pin PATH to the container toolchain so we use
# /usr/bin/python3.12 + /usr/local/bin/torchrun with the container's working TE
# (/usr/local/lib/python3.12/dist-packages/transformer_engine). The DSA overlay
# is a supplemental site-packages (custom kernels, no TE) layered via PYTHONPATH.
export PATH="/usr/local/bin:/usr/bin:/bin"
unset PYTHONHOME
export PYTHONPATH="$DSA_SITE:$REPO_ROOT/experimental/lite:$REPO_ROOT"
export CUBLAS_WORKSPACE_CONFIG=${CUBLAS_WORKSPACE_CONFIG:-:4096:8}
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=${NVTE_ALLOW_NONDETERMINISTIC_ALGO:-0}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export MEGATRON_LITE_DETERMINISTIC=${MEGATRON_LITE_DETERMINISTIC:-1}

HEAD=$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo missing)
echo "PP2_ARCHAEOLOGY_START repo=$REPO_ROOT commit=$HEAD job=${SLURM_JOB_ID:-local}"

python3 - <<'PY'
import torch
import transformer_engine.pytorch as te
assert hasattr(te, "Linear")
print("PP2_ARCHAEOLOGY_PREFLIGHT_OK torch", torch.__version__, "cuda", torch.cuda.is_available())
PY

# ── Arm A: historical PP2 train smoke (save/load/export matrix, commit 0a206bf0b+) ──
# Uses pp2 topology + PackedBatch train step; use_thd defaults False (BSHD forward).
# This is the documented green PP>1 training path predating RL PP2+THD.
MODEL=${ARCH_MODEL:-qwen3_moe}
BACKEND=${ARCH_BACKEND:-dist_opt}
echo "PP2_ARCHAEOLOGY_ARM_A model=$MODEL backend=$BACKEND test=save_load_roundtrip"
torchrun --nproc_per_node=8 -m pytest \
  "$REPO_ROOT/experimental/lite/tests/smoke/primitive/test_save_load_export_smoke.py::test_save_load_roundtrip[$MODEL-$BACKEND]" \
  -q --tb=short --mlite-smoke 2>&1 | tee "$OUTPUT_DIR/arch_arm_a_${MODEL}_${BACKEND}_${SLURM_JOB_ID:-local}.log"
echo "PP2_ARCHAEOLOGY_ARM_A_DONE rc=0 model=$MODEL backend=$BACKEND"

# ── Arm B: PP2 + explicit THD bench (fixed seq, single microbatch) ──
echo "PP2_ARCHAEOLOGY_ARM_B bench pp2+thd fixed-seq qwen3_moe tp1pp2ep4"
OUT_B="$OUTPUT_DIR/arch_arm_b_pp2_thd_fixed_${SLURM_JOB_ID:-local}.json"
torchrun --nproc_per_node=8 \
  "$REPO_ROOT/experimental/lite/examples/bench/bench.py" \
  --backend mlite \
  --model-name qwen3_moe \
  --hf-path "" \
  --impl lite \
  --tp 1 --pp 2 --ep 4 --cp 1 \
  --steps 2 --warmup 0 \
  --num-microbatches 1 \
  --seq-len 2048 \
  --use-thd \
  --skip-load-hf-weights \
  --truncate-layers 4 \
  --keep-experts 2 \
  --output-json "$OUT_B" \
  2>&1 | tee "$OUTPUT_DIR/arch_arm_b_pp2_thd_fixed_${SLURM_JOB_ID:-local}.log"
echo "PP2_ARCHAEOLOGY_ARM_B_DONE rc=0 output=$OUT_B"

# ── Arm C (step-2 pressure): variable per-microbatch lengths (2206/2213/…) ──
if [[ "${RUN_VAR_MB:-1}" == "1" ]]; then
  echo "PP2_ARCHAEOLOGY_ARM_C variable microbatch schedule"
  bash "$WORK_DIR/run_pp2_thd_variable_bench.sh" \
    2>&1 | tee "$OUTPUT_DIR/arch_arm_c_var_mb_${SLURM_JOB_ID:-local}.log"
  echo "PP2_ARCHAEOLOGY_ARM_C_DONE rc=0"
fi

echo "PP2_ARCHAEOLOGY_ALL_GREEN job=${SLURM_JOB_ID:-local} commit=$HEAD"
