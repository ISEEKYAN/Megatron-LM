#!/usr/bin/env bash
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Container-side body of the Branch B M-FSDP per-cycle retention probe
# (TASK-1.13.8.2). Runs the 2x2 matrix — {mfsdp, fsdp2} x expandable_segments
# {True, False} — as four torchrun invocations of mfsdp_cycle_probe.py against a
# tiny qwen3_moe proxy. Reuses the validated verl.vllm023 env recipe (conda-
# scrubbed PATH, CP overlay on PYTHONPATH/LD_LIBRARY_PATH, lustre cache redirect)
# from qwen35_dapo_mfsdp_62295f9b3/mfsdp_config_only_*.sh. Invoked via
# run_mfsdp_cycle_probe.sbatch inside the container; not run standalone.
set -euo pipefail

BASE=${BASE:-/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/bayan/code}
# ad1364f76: latest parent HEAD — export peak fix (aec21e63c) + host-tensor
# census helper (summarize_host_storages) the host-RAM probe reuses.
MLITE=${MLITE:-$BASE/qwen35_dapo_mfsdp_62295f9b3/mlite-ad1364f76}
CP_SITE=${CP_SITE:-$BASE/mlite-newenv-cache/qwen35-cp-overlay-20260613/site}
PROBE_DIR=${PROBE_DIR:-$BASE/branchb-mfsdp/mfsdp_cycle_probe}
MODEL=${MODEL:-$BASE/models/Qwen3-30B-A3B}          # qwen3_moe; shrunk below to <=1B
NGPUS=${NGPUS:-2}                                    # dp=NGPUS; FSDP mechanism is per-rank, dp=2 reproduces it
CYCLES=${CYCLES:-20}
SEQ_LEN=${SEQ_LEN:-512}
TRUNCATE_LAYERS=${TRUNCATE_LAYERS:-2}               # 2 layers + 4 experts + hidden 2048 vocab 151936 ~= 0.7B <= 1B
KEEP_EXPERTS=${KEEP_EXPERTS:-4}
WARMUP_STEPS=${WARMUP_STEPS:-2}
ARM_TIMEOUT=${ARM_TIMEOUT:-480}                     # per-arm hard cap (s); 4 arms << 30 min --time
EVID=${EVID:?EVID (evidence dir) must be set by the sbatch}
CACHE=${CACHE:?CACHE dir must be set by the sbatch}

# Reference-source freshness datum (bayan 2026-07-13 上游新鲜度铁律).
echo "REF mlite tree=$MLITE (commit ad1364f76: export-peak fix + host-tensor census)"
echo "REF image=verl.vllm023.sqsh  proxy=$MODEL (qwen3_moe, truncate=$TRUNCATE_LAYERS keep_experts=$KEEP_EXPERTS => ~0.7B)"

# ── validated env recipe (conda-scrubbed; CP overlay; lustre caches) ─────────
export PATH="$(printf %s "$PATH" | tr : '\n' | grep -viE 'miniforge|/conda|/anaconda' | paste -sd: -)"
unset PYTHONHOME CONDA_PREFIX CONDA_DEFAULT_ENV ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES || true
export PYTHONNOUSERSITE=1
export PYTHONPATH="/vllm:$CP_SITE:$MLITE/experimental/lite/examples/verl:$MLITE/experimental/lite:$MLITE:$PROBE_DIR:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$CP_SITE/tvm_ffi/lib:${LD_LIBRARY_PATH:-}"
export HYDRA_FULL_ERROR=1

# The container root FS is tiny; torch.compile / HF caches ENOSPC there. Redirect
# HOME + every cache onto lustre (same fix family as the sibling cumem probe).
export HOME="$CACHE/home"
export XDG_CACHE_HOME="$CACHE/xdg"
export HF_HOME="$CACHE/hf"
export HF_DATASETS_CACHE="$CACHE/hfds"
export TORCHINDUCTOR_CACHE_DIR="$CACHE/inductor"
export TRITON_CACHE_DIR="$CACHE/triton"
mkdir -p "$HOME" "$XDG_CACHE_HOME" "$HF_HOME" "$HF_DATASETS_CACHE" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$EVID"

test -f "$MODEL/config.json" || { echo "FATAL: model config missing: $MODEL" >&2; exit 2; }
test -f "$PROBE_DIR/mfsdp_cycle_probe.py" || { echo "FATAL: probe missing: $PROBE_DIR" >&2; exit 2; }

cd /tmp
COMMON=(
  --hf-path "$MODEL"
  --model-name qwen3_moe
  --cycles "$CYCLES"
  --seq-len "$SEQ_LEN"
  --truncate-layers "$TRUNCATE_LAYERS"
  --keep-experts "$KEEP_EXPERTS"
  --skip-load-hf-weights
  --warmup-steps "$WARMUP_STEPS"
  --tp 1 --ep 1 --pp 1 --cp 1
  --out-dir "$EVID"
)

rc_any=0
for OPT in mfsdp fsdp2; do
  for EXP in True False; do
    TAG="${OPT}-exp${EXP}"
    PORT=$((29500 + RANDOM % 2000))
    echo "==================== ARM $TAG (PYTORCH_CUDA_ALLOC_CONF=expandable_segments:$EXP) ===================="
    if PYTORCH_CUDA_ALLOC_CONF="expandable_segments:${EXP}" \
       timeout "$ARM_TIMEOUT" torchrun --nproc_per_node="$NGPUS" --master_port="$PORT" \
         "$PROBE_DIR/mfsdp_cycle_probe.py" --optimizer "$OPT" --tag "$TAG" "${COMMON[@]}"; then
      echo "ARM $TAG OK"
    else
      rc=$?
      rc_any=$rc
      echo "ARM $TAG FAILED rc=$rc"
    fi
  done
done

echo "=== per-arm summaries ==="
for f in "$EVID"/*-summary.json; do [ -f "$f" ] && echo "--- $f ---" && cat "$f"; done

# Gold-standard A/B: fold the per-arm summaries into mfsdp-fsdp2 per-cycle
# retention deltas + live-stack diff (zero-GPU, plain python — no torchrun).
echo "=== gold-standard mfsdp-fsdp2 combine ==="
if python "$PROBE_DIR/mfsdp_cycle_probe.py" --combine --out-dir "$EVID"; then
  echo "--- gold-standard-AB.json ---"; cat "$EVID/gold-standard-AB.json"
else
  rc=$?; rc_any=$rc; echo "COMBINE FAILED rc=$rc"
fi

echo "ALL_ARMS_DONE rc_any=$rc_any evidence=$EVID"
exit "$rc_any"
