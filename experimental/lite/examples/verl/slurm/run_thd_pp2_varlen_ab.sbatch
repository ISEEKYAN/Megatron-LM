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
#   MODEL_PATH   a small HF model dir (config+tokenizer; e.g. tiny qwen3_moe). Weights
#                optional — LOAD_HF_WEIGHTS defaults False (structural THD-PP test).
#   VERL_ROOT    committed VERL snapshot (provides verl.trainer.sft_trainer). Not baked
#                into the container; run_qwen3moe_sft.sh puts it on PYTHONPATH from env.
#   MEGATRON_ROOT compatible MCore snapshot (provides megatron.core; experimental/lite
#                imports it and does NOT bundle a real megatron/core).
#   MLITE_SM90_SITE canonical training overlay site-packages (provides transformers,
#                torchdata; the container has NO transformers, so the SFT tokenizer/
#                chat_template import dies with ModuleNotFoundError without it — job
#                13990351 died here at 1:44). torch is NOT in the overlay (container).
#   OUTPUT_DIR   fresh result directory
set -euo pipefail

BASE_IMAGE=${BASE_IMAGE:?set BASE_IMAGE to the canonical single-torch2.12 container}
MLITE_NEW=${MLITE_NEW:?set MLITE_NEW to the approach-A HEAD checkout}
MLITE_OLD=${MLITE_OLD:?set MLITE_OLD to the pre-approach-A (76905da37^) checkout}
MODEL_PATH=${MODEL_PATH:?set MODEL_PATH to a small HF model dir (config+tokenizer)}
VERL_ROOT=${VERL_ROOT:?set VERL_ROOT to a committed VERL snapshot (verl.trainer.sft_trainer)}
MEGATRON_ROOT=${MEGATRON_ROOT:?set MEGATRON_ROOT to a compatible MCore snapshot (megatron.core)}
MLITE_SM90_SITE=${MLITE_SM90_SITE:?set MLITE_SM90_SITE to the canonical training overlay site-packages (transformers)}
OUTPUT_DIR=${OUTPUT_DIR:?set OUTPUT_DIR to a fresh result directory}
# verl / mcore are not baked into the container; run_qwen3moe_sft.sh prepends them to
# PYTHONPATH from the environment, so they must reach the container via srun --export=ALL.
# MLITE_SM90_SITE (transformers) is threaded directly onto the container PYTHONPATH in run().
export VERL_ROOT MEGATRON_ROOT MLITE_SM90_SITE

# Knobs (defaults tuned for a 2-GPU PP2 decisive run).
NPROC=${NPROC:-2}
PP_SIZE=${PP_SIZE:-2}
TP_SIZE=${TP_SIZE:-1}
EP_SIZE=${EP_SIZE:-1}
CP_SIZE=${CP_SIZE:-1}
# bayan 21:06: run more steps so the PP1-vs-PP2 loss delta reveals whether it is
# converging bf16 noise (< tol, shrinking) or a persistent/diverging real residual.
TOTAL_STEPS=${TOTAL_STEPS:-16}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-32}
# bayan 21:06: PP degree must be the ONLY variable across arms. Dynamic-bsz packs a
# *different number of micro-batches* for PP1 vs PP2 (different token-budget slicing),
# so their grad-accumulation differs and the loss offset is a packing artifact, not a
# PP-fix signal. Turn dynamic-bsz OFF and fix micro_batch_size_per_gpu: with DP=1 in
# both arms, num_micro_batches = TRAIN_BATCH_SIZE / MICRO_BATCH_SIZE is then identical
# across PP1/PP2 (32/8 = 4 micro-batches: >= PP for a real 1F1B warmup->steady path).
# THD no_padding still packs variable-length sequences inside each static micro-batch,
# so per-micro-batch token counts still vary => the per-mb P2P shape bug is still armed.
# Sweep num-mb via MICRO_BATCH_SIZE (32/16=2, 32/8=4, 32/4=8) to watch offset vs mb count.
USE_DYNAMIC_BSZ=${USE_DYNAMIC_BSZ:-False}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-8}
MAX_TOKENS_PER_GPU=${MAX_TOKENS_PER_GPU:-2048}
MAX_LENGTH=${MAX_LENGTH:-4096}
LOAD_HF_WEIGHTS=${LOAD_HF_WEIGHTS:-False}
ARM_TIMEOUT=${ARM_TIMEOUT:-600}   # a hung OLD arm is the red signal; bound it.

for p in "${BASE_IMAGE}" "${MLITE_NEW}" "${MLITE_OLD}" "${MODEL_PATH}" \
         "${VERL_ROOT}" "${MEGATRON_ROOT}" "${MLITE_SM90_SITE}"; do
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
      # MLITE_SM90_SITE supplies transformers (tokenizer/chat_template); the
      # container has none, so it MUST be on PYTHONPATH or the SFT launcher dies
      # with ModuleNotFoundError: No module named 'transformers' (job 13990351).
      # It carries no torch (container torch 2.12 wins), no megatron/lite (no
      # shadow of the arm's pipeline.py), so tail placement is safe.
      export PYTHONPATH='${src}/experimental/lite':'${MLITE_SM90_SITE}':'${MROOT:-}':\${PYTHONPATH:-}
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
       --train-size ${VARLEN_TRAIN_SIZE:-640} --test-size ${VARLEN_TEST_SIZE:-32}" \
    || { echo "DATA_GEN rc=$?"; exit 3; }
fi

# Shared SFT launch knobs — arms differ only in the checkout + pipeline degree.
arm() {  # <mlite_src> <run_name> <pp> <nproc>  -> runs the SFT launcher
  local src=$1 run_name=$2 pp=$3 nproc=$4
  MROOT="${MEGATRON_CSA:-}" \
  run "${src}" -- \
    "MODEL_PATH='${MODEL_PATH}' \
     TRAIN_FILES='${DATA_DIR}/train.parquet' \
     VAL_FILES='${DATA_DIR}/test.parquet' \
     OUTPUT_ROOT='${OUTPUT_DIR}/${run_name}' \
     RUN_NAME='${run_name}' \
     NUM_GPUS=${nproc} NPROC_PER_NODE=${nproc} \
     TP_SIZE=${TP_SIZE} PP_SIZE=${pp} EP_SIZE=${EP_SIZE} ETP_SIZE=1 CP_SIZE=${CP_SIZE} \
     USE_DYNAMIC_BSZ=${USE_DYNAMIC_BSZ} \
     MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE} \
     TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE} \
     MAX_TOKENS_PER_GPU=${MAX_TOKENS_PER_GPU} MAX_LENGTH=${MAX_LENGTH} \
     TOTAL_STEPS=${TOTAL_STEPS} SAVE_FREQ=-1 TEST_FREQ=-1 \
     LOAD_HF_WEIGHTS=${LOAD_HF_WEIGHTS} \
     MESSAGES_KEY=messages \
     timeout ${ARM_TIMEOUT} bash examples/verl/scripts/run_qwen3moe_sft.sh"
}

# `set -e` must not abort on the (expected) OLD failure before we capture rc.
# arm 1 — NEW at PP2: the fix under test. Expect GREEN (rc=0, loss finite, no hang).
echo "==== arm NEW_PP2 (approach A HEAD, PP=2) — expect GREEN ===="
NEW_RC=0; arm "${MLITE_NEW}" new_pp2 "${PP_SIZE}" "${NPROC}" || NEW_RC=$?
echo "ARM_NEW_PP2_RC=${NEW_RC}"

# arm 2 — OLD at PP2: pre-fix. Expect RED (hang->timeout / NCCL size mismatch),
# proving the variable-length data actually triggers the P2P buffer bug.
echo "==== arm OLD_PP2 (pre-approach-A 76905da37^, PP=2) — expect RED ===="
OLD_RC=0; arm "${MLITE_OLD}" old_pp2 "${PP_SIZE}" "${NPROC}" || OLD_RC=$?
echo "ARM_OLD_PP2_RC=${OLD_RC}"

# arm 3 — NEW at PP1 on a single process: NO inter-stage P2P at all, so it is the
# trusted numerical reference. Same model, same data, DP=1 in both PP1 and PP2, and
# (bayan 21:06) static micro_batch_size => IDENTICAL num_micro_batches and grad-accum
# structure across arms => per-step loss must match within fp tolerance, with PP degree
# as the sole variable. A green-but-wrong-value NEW_PP2 (silent off-by-one in the
# recv<->mb mapping) diverges here even though rc=0. (bayan 20:15/21:06: 三臂数值一致性)
echo "==== arm NEW_PP1 (approach A HEAD, PP=1, single process) — numerical reference ===="
REF_RC=0; arm "${MLITE_NEW}" new_pp1 1 1 || REF_RC=$?
echo "ARM_NEW_PP1_RC=${REF_RC}"

echo "==== numerical parity: NEW_PP1 vs NEW_PP2 per-step train/loss ===="
PARITY=UNKNOWN
PP1_JSONL="${OUTPUT_DIR}/new_pp1/new_pp1.jsonl"
PP2_JSONL="${OUTPUT_DIR}/new_pp2/new_pp2.jsonl"
PARITY_TOL="${PARITY_TOL:-2e-3}"
if [[ -f "${PP1_JSONL}" && -f "${PP2_JSONL}" ]]; then
  PARITY=$(python3 - "${PP1_JSONL}" "${PP2_JSONL}" "${PARITY_TOL}" <<'PY'
import json, sys
def losses(p):
    out={}
    for line in open(p):
        line=line.strip()
        if not line: continue
        try: o=json.loads(line)
        except Exception: continue
        d=o.get("data",{}); s=o.get("step")
        for k in ("train/loss","loss"):
            if k in d: out[s]=float(d[k]); break
    return out
a=losses(sys.argv[1]); b=losses(sys.argv[2]); tol=float(sys.argv[3])
steps=sorted(set(a)&set(b))
if not steps:
    print("NO_COMMON_STEPS"); sys.exit(0)
mx=max(abs(a[s]-b[s]) for s in steps)
tag="MATCH" if mx<=tol else "DIVERGE"
print("%s max_abs_diff=%.3e tol=%.1e steps=%d pp1=%s pp2=%s"%(
    tag, mx, tol, len(steps),
    ["%.6f"%a[s] for s in steps], ["%.6f"%b[s] for s in steps]))
PY
) || PARITY="PARITY_SCRIPT_ERROR"
fi
echo "THD_PP2_VARLEN_PARITY=${PARITY}"

echo "==== verdict ===="
# PRIMARY (P2P fix): DECISIVE only if NEW_PP2 green AND OLD_PP2 red (proving the
# data triggers the bug and the fix is what flipped it). Numerical parity is a
# secondary guard against a green-but-wrong-value fix.
if [[ "${NEW_RC}" -eq 0 && "${OLD_RC}" -ne 0 ]]; then
  echo "THD_PP2_VARLEN_VERDICT=PASS_DECISIVE (NEW_PP2 green, OLD_PP2 red)"
elif [[ "${NEW_RC}" -eq 0 && "${OLD_RC}" -eq 0 ]]; then
  echo "THD_PP2_VARLEN_VERDICT=PASS_BUT_NONDECISIVE (OLD_PP2 also green — data may not vary enough; inspect micro-batch token counts in the OLD log)"
else
  echo "THD_PP2_VARLEN_VERDICT=FAIL (NEW_PP2 rc=${NEW_RC}) — inspect ${OUTPUT_DIR}/new_pp2"
fi
echo "DONE thd-pp2-varlen-ab job=${SLURM_JOB_ID} out=${OUTPUT_DIR}"
