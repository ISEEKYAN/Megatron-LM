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
export TRITON_CACHE_DIR="/tmp/triton-q35pp2-${SLURM_JOB_ID:-x}-${SLURMD_NODENAME:-0}"
export TORCHINDUCTOR_CACHE_DIR="/tmp/torchinductor-q35pp2-${SLURM_JOB_ID:-x}-${SLURMD_NODENAME:-0}"
export PYTHONPYCACHEPREFIX="/tmp/pycache-q35pp2-${SLURM_JOB_ID:-x}-${SLURMD_NODENAME:-0}"
export HF_HOME="$RUN_DIR/hf-home"
export HF_DATASETS_CACHE="$RUN_DIR/hf-datasets-cache"
export NCCL_NVLS_ENABLE=0 VLLM_ALLREDUCE_USE_SYMM_MEM=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn RAY_memory_monitor_refresh_ms=0 HYDRA_FULL_ERROR=1
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$PYTHONPYCACHEPREFIX" "$HF_HOME" "$HF_DATASETS_CACHE" "$RUN_DIR/logs" "$RUN_DIR/output" "$SCRIPT_DIR"

tmp_available_kb=$(df -Pk /tmp | awk 'END {print $4}')
tmp_required_kb=$((20 * 1024 * 1024))
echo "Q35_PP2_TMP_CAPACITY node=${SLURMD_NODENAME:-unknown} available_kb=$tmp_available_kb required_kb=$tmp_required_kb"
if (( tmp_available_kb < tmp_required_kb )); then
  echo "Q35_PP2_TMP_CAPACITY_FAILED node=${SLURMD_NODENAME:-unknown}" >&2
  exit 75
fi

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
