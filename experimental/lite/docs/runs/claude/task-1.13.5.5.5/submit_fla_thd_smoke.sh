#!/usr/bin/env bash
# Orchestrator for the FLA-enabled Qwen3.5 GDN THD smoke (TASK-1.13.5.5.5).
# Run from the CW login node (cw-dfw-cs-001-login-02).
#
#   GATE_ONLY=1 bash submit_fla_thd_smoke.sh   # CPU import gate only, no GPU
#   bash submit_fla_thd_smoke.sh               # gate + submit the 8xH100 smoke
#
# Stages the mlite checkout under test (default: TASK-1.13.5.5.3 HEAD, so this
# directly proves "CP_SITE unblocks chunk3"), runs the CPU import gate, then
# sbatch-submits the GPU smoke.  The GPU submission is gated behind the pre-GPU
# moe review; do not fire until that gate passes.
set -euo pipefail

BASE="${BASE:-/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/bayan/code}"
RUN_ROOT="${RUN_ROOT:-$BASE/runtime/task-1-13-5-5-5-fla-thd-smoke}"
# mlite under test.  Default = the three-arm harness HEAD that hit the FLA block.
MLITE_COMMIT="${MLITE_COMMIT:-042ef7b09}"
MLITE_REPO="${MLITE_REPO:-$RUN_ROOT/mlite-${MLITE_COMMIT}}"
MLITE_REMOTE="${MLITE_REMOTE:-https://github.com/ISEEKYAN/Megatron-LM.git}"
MEGATRON_ROOT="${MEGATRON_ROOT:-$BASE/runtime/ds4-csacp-parity-eaa5b486d/mcore}"
IMG="${IMG:-$BASE/verl_optimize/verl.vllm023.sqsh}"
CP_SITE="${CP_SITE:-$BASE/mlite-newenv-cache/qwen35-cp-overlay-20260613/site}"
VERL="${VERL:-$BASE/verl-main-latest}"
MODEL_PATH="${MODEL_PATH:-/lustre/fsw/portfolios/coreai/users/bayan/code/models/Qwen3.5-35B-A3B}"
DATA_DIR="${DATA_DIR:-$RUN_ROOT/data/gsm8k_sft}"
TRAIN_FILES="${TRAIN_FILES:-$DATA_DIR/train.parquet}"
SBATCH="${SBATCH:-fla_thd_smoke.sbatch}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -L)"
mkdir -p "$RUN_ROOT/logs" "$DATA_DIR"

# ---- 1. stage the mlite checkout under test ---------------------------------
if [[ ! -d "$MLITE_REPO/.git" ]]; then
  echo "[stage] cloning mlite -> $MLITE_REPO"
  git clone --no-checkout "$MLITE_REMOTE" "$MLITE_REPO"
fi
git -C "$MLITE_REPO" fetch --depth 1 origin "$MLITE_COMMIT" 2>/dev/null || git -C "$MLITE_REPO" fetch origin
git -C "$MLITE_REPO" checkout -q "$MLITE_COMMIT"
echo "[stage] mlite at $(git -C "$MLITE_REPO" rev-parse HEAD)"

# The node script lives with the recipe under this task's docs/runs.  If the
# staged mlite predates it, ship this dir's copy alongside.
NODE_SCRIPT="$MLITE_REPO/experimental/lite/docs/runs/claude/task-1.13.5.5.5/fla_thd_smoke_node.sh"
if [[ ! -f "$NODE_SCRIPT" ]]; then
  mkdir -p "$RUN_ROOT/node"
  cp "$SCRIPT_DIR/fla_thd_smoke_node.sh" "$RUN_ROOT/node/fla_thd_smoke_node.sh"
  export NODE_SCRIPT_OVERRIDE="$RUN_ROOT/node/fla_thd_smoke_node.sh"
  echo "[stage] shipped node script -> $NODE_SCRIPT_OVERRIDE"
fi

# ---- 2. CPU import gate (container): the exact FLA imports the code needs ----
echo "===================== FLA CPU IMPORT GATE ====================="
srun -A coreai_devtech_all -p cpu_short --nodes=1 --ntasks=1 --time=00:12:00 \
  --container-image="$IMG" --container-mounts=/lustre:/lustre --export=ALL,CP_SITE="$CP_SITE" \
  bash -c '
    set -e
    export PATH="$(printf %s "$PATH" | tr : "\n" | grep -viE "miniforge|/conda|/anaconda" | paste -sd: -)"
    export PYTHONNOUSERSITE=1
    export PYTHONPATH="/vllm:$CP_SITE:${PYTHONPATH:-}"
    export LD_LIBRARY_PATH="$CP_SITE/tvm_ffi/lib:${LD_LIBRARY_PATH:-}"
    python3 - <<PY
import fla, tilelang
from fla.modules.convolution import causal_conv1d
from fla.ops.gated_delta_rule import chunk_gated_delta_rule
from fla.modules.l2norm import l2norm
print("FLA_GATE_OK fla=%s tilelang=%s causal_conv1d=%s chunk_gdr=%s l2norm=%s" % (
    fla.__version__, tilelang.__version__,
    callable(causal_conv1d), callable(chunk_gated_delta_rule), callable(l2norm)))
PY
  '
echo "==============================================================="

if [[ "${GATE_ONLY:-0}" == "1" ]]; then
  echo "[gate-only] stopping before GPU submission."
  exit 0
fi

# ---- 3. ensure verl messages-format gsm8k SFT parquet exists ----------------
if [[ ! -f "$TRAIN_FILES" ]]; then
  echo "[data] generating verl gsm8k SFT parquet in-container -> $DATA_DIR"
  srun -A coreai_devtech_all -p cpu_short --nodes=1 --ntasks=1 --time=00:20:00 \
    --container-image="$IMG" --container-mounts=/lustre:/lustre --export=ALL \
    python3 "$VERL/examples/data_preprocess/gsm8k_multiturn_sft.py" --local_save_dir "$DATA_DIR"
fi
[[ -f "$TRAIN_FILES" ]] || { echo "ERROR: TRAIN_FILES=$TRAIN_FILES missing after data prep" >&2; exit 1; }

# ---- 4. submit the GPU smoke ------------------------------------------------
out="$RUN_ROOT/logs/%x_%j.out"
jid=$(sbatch --parsable --output="$out" \
  --export=ALL,BASE="$BASE",IMG="$IMG",CP_SITE="$CP_SITE",MLITE_REPO="$MLITE_REPO",MEGATRON_ROOT="$MEGATRON_ROOT",VERL="$VERL",MODEL_PATH="$MODEL_PATH",RUN_ROOT="$RUN_ROOT",TRAIN_FILES="$TRAIN_FILES"${NODE_SCRIPT_OVERRIDE:+,NODE_SCRIPT_OVERRIDE="$NODE_SCRIPT_OVERRIDE"} \
  "$SCRIPT_DIR/$SBATCH")
echo "SUBMITTED fla_thd_smoke job=$jid -> $RUN_ROOT/logs (${out})"
