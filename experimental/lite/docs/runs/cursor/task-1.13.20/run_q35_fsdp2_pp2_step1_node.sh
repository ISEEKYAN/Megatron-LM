#!/usr/bin/env bash
# Node launcher for Qwen3.5 FSDP2 PP2 one-step smoke (TASK-1.13.20).
set -euo pipefail

ROLE=${1:?role head|worker}; HEAD_ADDR=${2:?head ip:port}; BACKEND=${3:?backend}; NN=${4:-1}
HEAD_IP=${HEAD_ADDR%:*}; PORT=${HEAD_ADDR##*:}
BASE=${BASE:-/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/bayan/code}
RUN_DIR=${RUN_DIR:-$BASE/qwen35_dapo_mfsdp_62295f9b3}
SCRIPT_DIR=${SCRIPT_DIR:-$RUN_DIR/task-1.13.20}
VERL=${VERL:-$BASE/verl-main-latest}
MLITE_REPO=${MLITE_REPO:-$RUN_DIR/mlite-57a2064b3}
CP_SITE=${CP_SITE:-$BASE/mlite-newenv-cache/qwen35-cp-overlay-20260613/site}

export PATH=$(printf %s "$PATH" | tr : '\n' | grep -viE 'miniforge|/conda|/anaconda' | paste -sd: -)
unset PYTHONHOME CONDA_PREFIX CONDA_DEFAULT_ENV ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES || true
unset CUDA_DEVICE_MAX_CONNECTIONS
export MLITE_OPTIMIZER=${MLITE_OPTIMIZER:-fsdp2}
export PYTHONNOUSERSITE=1 VERL_ROOT=$VERL
export G0B_BACKEND="$BACKEND"
export PYTHONPATH="/vllm:$RUN_DIR:$CP_SITE:$MLITE_REPO/experimental/lite/examples/verl:$MLITE_REPO/experimental/lite:$MLITE_REPO:$VERL:${PYTHONPATH:-}"
export RUNTIME_PYTHONPATH="/vllm:$RUN_DIR:$CP_SITE:$MLITE_REPO/experimental/lite/examples/verl:$MLITE_REPO/experimental/lite:$MLITE_REPO:$VERL"
export LD_LIBRARY_PATH="$CP_SITE/tvm_ffi/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# TASK-1.13.20: job13945545 died ENOSPC — vLLM usage_stats (HOME/.config/vllm) and
# flashinfer/tilelang JIT compilation filled the node-local disk (/tmp + container HOME).
# Redirect ALL scratch caches to a per-job lustre dir (bayan 2026-07-14 guide; matches the
# established mfsdp cache-$SLURM_JOB_ID convention: run_dir/cache-<jobid>/{triton,inductor,tmp}).
CACHE="$RUN_DIR/cache-${SLURM_JOB_ID:-manual}"
export CACHE
export HOME="$CACHE/home"
export XDG_CACHE_HOME="$CACHE/xdg-cache"
# NB: do NOT redirect TMPDIR onto lustre — Ray places its plasma_store AF_UNIX
# socket under $TMPDIR/ray/session_*/sockets/ and the long lustre prefix blows the
# 107-byte unix-socket path limit (job13950787). TMPDIR was never the ENOSPC culprit
# (that was usage_stats + JIT under HOME); leave it at the short node-local /tmp default.
export TRITON_CACHE_DIR="$CACHE/triton"
export TORCHINDUCTOR_CACHE_DIR="$CACHE/inductor"
export PYTHONPYCACHEPREFIX="$CACHE/pycache"
export VLLM_CACHE_ROOT="$CACHE/vllm"
export FLASHINFER_WORKSPACE_BASE="$CACHE/flashinfer"
export TILELANG_CACHE_DIR="$CACHE/tilelang"
# The usage-stats write (usage_lib._write_to_file) was the exact ENOSPC crash site; disable it.
export VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
export HF_HOME="$RUN_DIR/hf-home"
export HF_DATASETS_CACHE="$RUN_DIR/hf-datasets-cache"
export NCCL_NVLS_ENABLE=0 VLLM_ALLREDUCE_USE_SYMM_MEM=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn RAY_memory_monitor_refresh_ms=0 HYDRA_FULL_ERROR=1
mkdir -p "$HOME" "$XDG_CACHE_HOME" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" \
  "$PYTHONPYCACHEPREFIX" "$VLLM_CACHE_ROOT" "$FLASHINFER_WORKSPACE_BASE" "$TILELANG_CACHE_DIR" \
  "$HF_HOME" "$HF_DATASETS_CACHE" "$RUN_DIR/logs" "$RUN_DIR/output" "$SCRIPT_DIR"

cache_available_kb=$(df -Pk "$CACHE" | awk 'END {print $4}')
echo "Q35_PP2_CACHE_REDIRECT node=${SLURMD_NODENAME:-unknown} cache=$CACHE lustre_available_kb=$cache_available_kb"

mlite_head=$(git -C "$MLITE_REPO" rev-parse HEAD 2>/dev/null || echo missing)
verl_head=$(git -C "$VERL" rev-parse HEAD 2>/dev/null || echo missing)
echo "Q35_PP2_COMMITS backend=$BACKEND mlite=$mlite_head verl=$verl_head optimizer=$MLITE_OPTIMIZER"

python3 - "$BACKEND" <<'PY'
import os
import sys
import fla
import tilelang
import vllm

backend = sys.argv[1]
if backend == "mlite":
    import verl_mlite.engine.mlite_engine
    from megatron.lite.primitive.optimizers import get_optimizer_backend

    selected = os.environ.get("MLITE_OPTIMIZER", "fsdp2")
    registered = get_optimizer_backend(selected)
    assert registered.name == selected, (selected, registered.name)
    from megatron.lite.primitive.ckpt import hf_weights

    assert hasattr(hf_weights.export_hf_weights, "__call__")
    print("Q35_PP2_PREFLIGHT_OK", backend, vllm.__version__, fla.__version__, tilelang.__version__, "optimizer", selected)
else:
    raise SystemExit("this smoke requires BACKEND=mlite")
PY

RAY_PORTS='--min-worker-port=21000 --max-worker-port=31999 --runtime-env-agent-port=19435 --metrics-export-port=34001 --dashboard-agent-grpc-port=34002'
if [[ $ROLE == head ]]; then
  ray start --head --node-ip-address="$HEAD_IP" --port="$PORT" --num-gpus=8 $RAY_PORTS --disable-usage-stats
  sleep 90
  export RAY_ADDRESS="$HEAD_IP:$PORT"
  python3 - "$((NN * 8))" <<'PY'
import sys
import time

import ray

expected = int(sys.argv[1])
ray.init(address="auto", ignore_reinit_error=True)
deadline = time.time() + 600
available = 0
while time.time() < deadline:
    available = int(ray.cluster_resources().get("GPU", 0))
    if available >= expected:
        print(f"RAY_CLUSTER_READY gpus={available} expected={expected}", flush=True)
        break
    print(f"RAY_CLUSTER_WAIT gpus={available} expected={expected}", flush=True)
    time.sleep(5)
else:
    raise RuntimeError(f"Ray registered only {available}/{expected} GPUs after 600s")
ray.shutdown()
PY
  export BASE NNODES=$NN NGPUS_PER_NODE=8 BACKEND
  export MODEL_PATH=/lustre/fsw/portfolios/coreai/users/bayan/code/models/Qwen3.5-35B-A3B
  export TRAIN_FILE=$BASE/verl_update_mcore/data/dapo-math-17k.parquet
  export TEST_FILE=$BASE/verl_update_mcore/data/aime-2024.parquet
  export OUTPUT_ROOT=$RUN_DIR/output/task-1.13.20
  echo "Q35_PP2_TRAIN_START job=${SLURM_JOB_ID:-none} nodes=$NN model=$MODEL_PATH"
  bash "$SCRIPT_DIR/run_q35_fsdp2_pp2_step1.sh"
  echo "Q35_PP2_TRAIN_DONE rc=0"
else
  sleep 20
  ray start --address="$HEAD_IP:$PORT" --num-gpus=8 $RAY_PORTS --block
fi
