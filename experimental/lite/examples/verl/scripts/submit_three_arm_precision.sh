#!/usr/bin/env bash
# Orchestrator for the AC#3 three-arm Muon precision experiment on CW.
#
# Run from the CW login node (cw-dfw-cs-001-login-02).  Stages the task's mlite
# checkout on lustre, ensures a verl messages-format gsm8k SFT parquet exists,
# runs a config-only dry-run gate (command resolution, NOT a precision claim),
# then sbatch-submits the requested arms.
#
#   ARMS="adamw muon"  bash submit_three_arm_precision.sh          # launch A/B
#   GATE_ONLY=1        bash submit_three_arm_precision.sh          # dry-run gate only
#
set -euo pipefail

BASE="${BASE:-/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/bayan/code}"
RUN_ROOT="${RUN_ROOT:-$BASE/runtime/task-1-13-5-5-3-muon-precision}"
MLITE_COMMIT="${MLITE_COMMIT:-7b3dc95e40964e75eaa8615c2ee613c9feb3a836}"
MLITE_REPO="${MLITE_REPO:-$RUN_ROOT/mlite-${MLITE_COMMIT:0:9}}"
MLITE_REMOTE="${MLITE_REMOTE:-https://github.com/ISEEKYAN/Megatron-LM.git}"
# NVIDIA Megatron-Core that ships megatron/core/optimizer/muon.py (dist_opt Muon).
MEGATRON_ROOT="${MEGATRON_ROOT:-$BASE/runtime/ds4-csacp-parity-eaa5b486d/mcore}"
IMG="${IMG:-$BASE/verl_optimize/verl.vllm023.sqsh}"
VERL="${VERL:-$BASE/verl-main-latest}"
MODEL_PATH="${MODEL_PATH:-/lustre/fsw/portfolios/coreai/users/bayan/code/models/Qwen3.5-35B-A3B}"
DATA_DIR="${DATA_DIR:-$RUN_ROOT/data/gsm8k_sft}"
TRAIN_FILES="${TRAIN_FILES:-$DATA_DIR/train.parquet}"
ARMS="${ARMS:-adamw muon}"
SBATCH="${SBATCH:-three_arm_precision_sft.sbatch}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -L)"
mkdir -p "$RUN_ROOT/logs" "$DATA_DIR"

# ---- 1. stage the task's mlite checkout (branch @ MLITE_COMMIT) --------------
if [[ ! -d "$MLITE_REPO/.git" ]]; then
  echo "[stage] cloning mlite -> $MLITE_REPO"
  git clone --no-checkout "$MLITE_REMOTE" "$MLITE_REPO"
fi
git -C "$MLITE_REPO" fetch --depth 1 origin "$MLITE_COMMIT" 2>/dev/null || git -C "$MLITE_REPO" fetch origin
git -C "$MLITE_REPO" checkout -q "$MLITE_COMMIT"
echo "[stage] mlite at $(git -C "$MLITE_REPO" rev-parse HEAD)"

# ---- 2a. config dry-run gate (command resolution) ---------------------------
echo "===================== CONFIG DRY-RUN GATE ====================="
echo "NOTE: proves each arm resolves the intended optimizer wiring."
echo "      NOT a precision result; precision comes from GPU JSONL loss."
for arm in adamw muon muon_local; do
  case "$arm" in
    adamw) oa=adamw; tp=distributed;;
    muon) oa=muon; tp=distributed;;
    muon_local) oa=muon; tp=blockwise;;
  esac
  echo "----- arm=$arm -> optimizer_algorithm=$oa muon_tp_mode=$tp -----"
  OUTPUT_ROOT="$(mktemp -d)" MODEL_PATH="$MODEL_PATH" TRAIN_FILES="$TRAIN_FILES" \
    OPTIMIZER_ALGORITHM="$oa" MUON_TP_MODE="$tp" MLITE_OPTIMIZER_BACKEND=dist_opt \
    DRY_RUN=1 bash "$SCRIPT_DIR/run_qwen3moe_sft.sh" \
    | tr ' ' '\n' | grep -E 'optim\.optimizer=|muon_tp_mode=|impl_cfg\.optimizer=' || true
done

# ---- 2b. init-chain gate (container, CPU): imports + optimizer construction --
# This crosses the real import chain (megatron.core+megatron.lite namespace merge,
# NVIDIA Muon getter, verl_mlite engine) and constructs the OptimizerConfig per arm,
# so a broken env is caught here on 0 GPUs instead of on an 8xH100 allocation.
echo "===================== INIT-CHAIN GATE (container, CPU) ====================="
srun -A coreai_devtech_all -p cpu_short --nodes=1 --ntasks=1 --time=00:15:00 \
  --container-image="$IMG" --container-mounts=/lustre:/lustre --export=ALL,MLITE_REPO="$MLITE_REPO",MEGATRON_ROOT="$MEGATRON_ROOT",VERL="$VERL" \
  bash -c '
    set -e
    export PATH="$(printf %s "$PATH" | tr : "\n" | grep -viE "miniforge|/conda|/anaconda" | paste -sd: -)"
    export PYTHONNOUSERSITE=1
    MLITE_LITE="$MLITE_REPO/experimental/lite"
    export PYTHONPATH="/vllm:$MLITE_LITE/examples/verl:$MLITE_LITE:$VERL:$MEGATRON_ROOT"
    python3 - <<PY
import megatron.core, megatron.lite
from megatron.core.optimizer.muon import get_megatron_muon_optimizer
from megatron.lite.runtime.contracts.config import OptimizerConfig
import verl_mlite.engine.mlite_engine  # noqa
print("INIT_GATE core=", megatron.core.__file__)
for arm,(oa,tp) in {"adamw":("adamw","distributed"),"muon":("muon","distributed"),"muon_local":("muon","blockwise")}.items():
    c = OptimizerConfig(optimizer=oa, muon_tp_mode=tp)
    assert c.optimizer_algorithm==oa, (arm,c.optimizer_algorithm)
    assert c.muon_tp_mode==tp, (arm,c.muon_tp_mode)
    print(f"INIT_GATE_ARM_OK arm={arm} optimizer_algorithm={c.optimizer_algorithm} muon_tp_mode={c.muon_tp_mode}")
print("INIT_GATE_OK")
PY
  '
echo "===================================================================="

if [[ "${GATE_ONLY:-0}" == "1" ]]; then
  echo "[gate-only] stopping before GPU submission."
  exit 0
fi

# ---- 3. ensure verl messages-format gsm8k SFT parquet exists -----------------
if [[ ! -f "$TRAIN_FILES" ]]; then
  echo "[data] generating verl gsm8k SFT parquet in-container -> $DATA_DIR"
  srun -A coreai_devtech_all -p cpu_short --nodes=1 --ntasks=1 --time=00:20:00 \
    --container-image="$IMG" --container-mounts=/lustre:/lustre --export=ALL \
    python3 "$VERL/examples/data_preprocess/gsm8k_multiturn_sft.py" --local_save_dir "$DATA_DIR"
fi
[[ -f "$TRAIN_FILES" ]] || { echo "ERROR: TRAIN_FILES=$TRAIN_FILES missing after data prep" >&2; exit 1; }

# ---- 4. submit arms ---------------------------------------------------------
for arm in $ARMS; do
  out="$RUN_ROOT/logs/%x_%j_${arm}.out"
  jid=$(sbatch --parsable --output="$out" \
    --export=ALL,ARM="$arm",BASE="$BASE",IMG="$IMG",MLITE_REPO="$MLITE_REPO",MEGATRON_ROOT="$MEGATRON_ROOT",VERL="$VERL",MODEL_PATH="$MODEL_PATH",RUN_ROOT="$RUN_ROOT",TRAIN_FILES="$TRAIN_FILES" \
    "$SCRIPT_DIR/$SBATCH")
  echo "SUBMITTED arm=$arm job=$jid -> $RUN_ROOT/logs (${out})"
done
