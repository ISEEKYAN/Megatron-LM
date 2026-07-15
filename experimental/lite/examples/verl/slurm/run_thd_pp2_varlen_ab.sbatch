#!/usr/bin/env bash
#SBATCH --account=coreai_devtech_all
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=32
#SBATCH --time=00:30:00
#SBATCH --job-name=thd-pp2-varlen-ab
#
# Minimal decisive A/B for the THD pipeline-P2P per-micro-batch shape fix
# (approach A). Any small HF model, VERL SFT path, PP=2, THD dynamic-bsz packing
# over *variable-length* data on 2 GPUs. No rollout / reward / real weights
# needed: the bug is a P2P recv-buffer size mismatch, so "loss finite + no hang"
# is a decisive green and "hang / NCCL size mismatch" is a decisive red.
#
#   arm NEW : MLITE_NEW checkout (approach A HEAD)            expect GREEN (rc=0)
#   arm OLD : MLITE_OLD checkout (pre-approach-A, 76905da37^) expect RED (hang/crash)
#
# Both arms read the SAME variable-length parquet, generated once from the NEW
# checkout, so the two arms differ solely in pipeline.py/runtime.py.
#
# Required env (all cw lustre paths):
#   BASE_IMAGE   canonical single-torch2.12 container (PP1 job13983827 stack)
#   MLITE_NEW    checkout at approach-A HEAD (this branch)
#   MLITE_OLD    checkout at 76905da37^ (== f9604ea84, pre-approach-A)
#   MODEL_PATH   a small HF model dir (config+tokenizer; e.g. Qwen2.5-0.5B). Weights
#                optional — LOAD_HF_WEIGHTS defaults False (structural THD-PP test).
#   OUTPUT_DIR   fresh result directory
set -euo pipefail

BASE_IMAGE=${BASE_IMAGE:?set BASE_IMAGE to the canonical single-torch2.12 container}
MLITE_NEW=${MLITE_NEW:?set MLITE_NEW to the approach-A HEAD checkout}
MLITE_OLD=${MLITE_OLD:?set MLITE_OLD to the pre-approach-A (76905da37^) checkout}
MODEL_PATH=${MODEL_PATH:?set MODEL_PATH to a small HF model dir (config+tokenizer)}
OUTPUT_DIR=${OUTPUT_DIR:?set OUTPUT_DIR to a fresh result directory}

# Knobs (defaults tuned for a 2-GPU PP2 decisive run).
NPROC=${NPROC:-2}
PP_SIZE=${PP_SIZE:-2}
TP_SIZE=${TP_SIZE:-1}
EP_SIZE=${EP_SIZE:-1}
CP_SIZE=${CP_SIZE:-1}
TOTAL_STEPS=${TOTAL_STEPS:-3}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-32}
MAX_TOKENS_PER_GPU=${MAX_TOKENS_PER_GPU:-2048}
MAX_LENGTH=${MAX_LENGTH:-4096}
LOAD_HF_WEIGHTS=${LOAD_HF_WEIGHTS:-False}
ARM_TIMEOUT=${ARM_TIMEOUT:-600}   # a hung OLD arm is the red signal; bound it.

for p in "${BASE_IMAGE}" "${MLITE_NEW}" "${MLITE_OLD}" "${MODEL_PATH}"; do
  [[ -e "${p}" ]] || { echo "missing required path: ${p}" >&2; exit 2; }
done
mkdir -p "${OUTPUT_DIR}"
DATA_DIR="${OUTPUT_DIR}/varlen_data"

run() {  # <mlite_src> -- <cmd...>
  local src=$1; shift
  [[ "${1:-}" == "--" ]] && shift
  srun --mpi=pmix --export=ALL --no-container-mount-home \
    --container-image="${BASE_IMAGE}" \
    --container-mounts=/lustre:/lustre \
    --container-workdir="${src}/experimental/lite" \
    bash -lc "
      set -uo pipefail
      export PATH=/usr/local/bin:/usr/bin:/bin; hash -r
      unset CONDA_DEFAULT_ENV CONDA_PREFIX PYTHONHOME VIRTUAL_ENV ROCR_VISIBLE_DEVICES || true
      export PYTHONNOUSERSITE=1
      # experimental/lite of THIS arm's checkout must resolve first so
      # megatron.lite is that checkout's pipeline.py/runtime.py; megatron is a
      # PEP-420 namespace so an optional MCore (MROOT) merges without shadowing.
      export PYTHONPATH='${src}/experimental/lite':'${MROOT:-}':\${PYTHONPATH:-}
      export TMPDIR=/tmp/thdpp-\${SLURM_JOB_ID}
      export TORCHINDUCTOR_CACHE_DIR=\${TMPDIR}/ti TRITON_CACHE_DIR=\${TMPDIR}/triton
      export CUDA_DEVICE_MAX_CONNECTIONS=1 NCCL_NVLS_ENABLE=0
      mkdir -p \${TORCHINDUCTOR_CACHE_DIR} \${TRITON_CACHE_DIR}
      $*
    "
}

# --- step 0: generate the shared variable-length data once (NEW checkout) -----
echo "==== step 0: generate variable-length SFT data ===="
if [[ ! -e "${DATA_DIR}/train.parquet" ]]; then
  MROOT="${MEGATRON_CSA:-}" run "${MLITE_NEW}" -- \
    "python -m examples.verl.prepare_varlen_sft_data --output-dir '${DATA_DIR}' \
       --train-size ${VARLEN_TRAIN_SIZE:-128} --test-size ${VARLEN_TEST_SIZE:-16}" \
    || { echo "DATA_GEN rc=$?"; exit 3; }
fi

# Shared SFT launch knobs — the two arms differ only in the checkout they run in.
arm() {  # <mlite_src> <run_name>  -> runs the SFT launcher with the shared knobs
  local src=$1 run_name=$2
  MROOT="${MEGATRON_CSA:-}" \
  run "${src}" -- \
    "MODEL_PATH='${MODEL_PATH}' \
     TRAIN_FILES='${DATA_DIR}/train.parquet' \
     VAL_FILES='${DATA_DIR}/test.parquet' \
     OUTPUT_ROOT='${OUTPUT_DIR}/${run_name}' \
     RUN_NAME='${run_name}' \
     NUM_GPUS=${NPROC} NPROC_PER_NODE=${NPROC} \
     TP_SIZE=${TP_SIZE} PP_SIZE=${PP_SIZE} EP_SIZE=${EP_SIZE} ETP_SIZE=1 CP_SIZE=${CP_SIZE} \
     USE_DYNAMIC_BSZ=True \
     TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE} \
     MAX_TOKENS_PER_GPU=${MAX_TOKENS_PER_GPU} MAX_LENGTH=${MAX_LENGTH} \
     TOTAL_STEPS=${TOTAL_STEPS} SAVE_FREQ=-1 TEST_FREQ=-1 \
     LOAD_HF_WEIGHTS=${LOAD_HF_WEIGHTS} \
     MESSAGES_KEY=messages \
     timeout ${ARM_TIMEOUT} bash examples/verl/scripts/run_qwen3moe_sft.sh"
}

# `set -e` must not abort on the (expected) OLD failure before we capture rc.
echo "==== arm NEW (approach A HEAD) — expect GREEN (rc=0, loss finite) ===="
NEW_RC=0; arm "${MLITE_NEW}" new || NEW_RC=$?
echo "ARM_NEW_RC=${NEW_RC}"

echo "==== arm OLD (pre-approach-A) — expect RED (hang->timeout / NCCL size mismatch) ===="
OLD_RC=0; arm "${MLITE_OLD}" old || OLD_RC=$?
echo "ARM_OLD_RC=${OLD_RC}"

echo "==== verdict ===="
# GREEN when NEW succeeds; the A/B is DECISIVE only if OLD also fails (proving the
# data actually triggers the bug and the fix is what flipped it).
if [[ "${NEW_RC}" -eq 0 && "${OLD_RC}" -ne 0 ]]; then
  echo "THD_PP2_VARLEN_VERDICT=PASS_DECISIVE (NEW green, OLD red)"
elif [[ "${NEW_RC}" -eq 0 && "${OLD_RC}" -eq 0 ]]; then
  echo "THD_PP2_VARLEN_VERDICT=PASS_BUT_NONDECISIVE (OLD also green — data may not vary enough; inspect micro-batch token counts in the OLD log)"
else
  echo "THD_PP2_VARLEN_VERDICT=FAIL (NEW rc=${NEW_RC}) — inspect ${OUTPUT_DIR}/new"
fi
echo "DONE thd-pp2-varlen-ab job=${SLURM_JOB_ID} out=${OUTPUT_DIR}"
