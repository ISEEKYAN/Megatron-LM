#!/usr/bin/env bash
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#SBATCH --job-name=cg-chunk-smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --account=coreai_devtech_all
#SBATCH --partition=batch
#SBATCH --time=00:25:00
#
# Chunk-wise CUDA Graph qualification smoke (TASK-1.21).
#
# Three staged qualifications on real H100 + Transformer Engine:
#   A. CG unit tests inside the real TE-bearing container (import + CPU logic).
#   B. capture/replay of a real te.pytorch.Linear via CudaGraphController
#      (_capture_slot -> make_graphed_callables -> get_graphed; replay==eager).
#   C. best-effort: 8-card qwen3.5 forward-backward via the proven correctness
#      harness, with MLITE_CG_QUALIFY_PROBE=1 running qualify() over the real
#      runtime-built chunks. Non-fatal: A+B gate the job rc.
#
# Env (defaults are the proven CW assets; override as needed):
#   MYSRC          staged branch experimental/lite root (has the CG spine)
#   HY3_ROOT       proven megatron_lite checkout (megatron.core + devlite-venv)
#   CONTAINER_IMAGE canonical qwen3.5 sqsh
#   HF_PATH        truncated qwen3.5 HF checkpoint for stage C
#   OUTPUT_ROOT    persistent artifact root

set -euo pipefail

BASE=${BASE:-/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/bayan}
MYSRC=${MYSRC:?set MYSRC to the staged branch experimental/lite root}
HY3_ROOT=${HY3_ROOT:-${BASE}/code/megatron_lite/hy3-validation-bcf0dc22b}
CONTAINER_IMAGE=${CONTAINER_IMAGE:-${BASE}/code/verl_optimize/verl.vllm016.dev.qwen3_5.sqsh}
HF_PATH=${HF_PATH:-${BASE}/code/qwen35_dapo_mfsdp_62295f9b3/Qwen3.5-35B-A3B-nine-layer}
OUTPUT_ROOT=${OUTPUT_ROOT:?set OUTPUT_ROOT to a persistent output root}
RUN_ID=${RUN_ID:-${SLURM_JOB_ID:-manual}}

for p in "${MYSRC}" "${HY3_ROOT}" "${CONTAINER_IMAGE}" "${OUTPUT_ROOT}"; do
  [[ "${p}" == /* ]] || { echo "path must be absolute: ${p}" >&2; exit 2; }
done
RUN_ROOT="${OUTPUT_ROOT}/${RUN_ID}"
mkdir -p "${RUN_ROOT}"

export MYSRC HY3_ROOT HF_PATH RUN_ROOT
export VENV_PY="${HY3_ROOT}/devlite-venv/bin/python"

srun \
  --container-image="${CONTAINER_IMAGE}" \
  --container-mounts=/lustre:/lustre \
  --container-workdir="${MYSRC}" \
  /bin/bash -c '
    set -uo pipefail
    : "${MYSRC:?}"; : "${HY3_ROOT:?}"; : "${RUN_ROOT:?}"; : "${VENV_PY:?}"

    unset CONDA_DEFAULT_ENV CONDA_PREFIX PYTHONHOME VIRTUAL_ENV
    export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
    # my megatron.lite (staged) first, then proven megatron.core from HY3_ROOT.
    export PYTHONPATH="${MYSRC}:${HY3_ROOT}"
    export CUBLAS_WORKSPACE_CONFIG=:4096:8
    export MEGATRON_LITE_DETERMINISTIC=1
    export CUDA_DEVICE_MAX_CONNECTIONS=1
    export NVTE_ALLOW_NONDETERMINISTIC_ALGO=0
    export NCCL_ALGO=Ring NCCL_NVLS_ENABLE=0

    PY="${VENV_PY}"
    [[ -x "${PY}" ]] || PY=python
    echo "=== python: ${PY} ==="; "${PY}" -c "import torch,transformer_engine; print('torch',torch.__version__,'gpus',torch.cuda.device_count())"

    rc_a=0; rc_b=0; rc_c=0

    echo "=== STAGE A: CG unit tests in-container ==="
    "${PY}" -m pytest -q \
      "${MYSRC}/tests/unit/primitive/test_chunk_cuda_graph_slots_unit.py" \
      "${MYSRC}/tests/unit/primitive/test_cuda_graph_controller_unit.py" \
      2>&1 | tee "${RUN_ROOT}/stage_a_unit.log" || rc_a=$?

    echo "=== STAGE B: capture/replay of real TE Linear (1 GPU) ==="
    CUDA_VISIBLE_DEVICES=0 "${PY}" \
      "${MYSRC}/examples/bench/cuda_graph_gpu_smoke.py" \
      --output-json "${RUN_ROOT}/stage_b_capture_replay.json" \
      --marker "${RUN_ROOT}/NON_SKIP_CUDA_GRAPH_CAPTURE_REPLAY_PASSED" \
      2>&1 | tee "${RUN_ROOT}/stage_b_capture_replay.log" || rc_b=$?

    echo "=== STAGE C (best-effort): 8-card qwen3.5 FB + qualify probe ==="
    if [[ -d "${HF_PATH:-}" ]]; then
      export MLITE_CG_QUALIFY_PROBE=1
      export MLITE_CG_QUALIFY_PROBE_OUT="${RUN_ROOT}/stage_c_qualify.json"
      torchrun --standalone --nproc_per_node=8 \
        "${MYSRC}/examples/bench/correctness.py" run \
        --backend mlite --hf-path "${HF_PATH}" --model-name qwen3_5 \
        --tp 1 --etp 1 --ep 8 --pp 1 --cp 1 \
        --steps 1 --num-microbatches 2 --seq-len 128 --seed 42 \
        --truncate-layers 2 --keep-experts 4 --disable-mtp --same-data-across-dp \
        --output-json "${RUN_ROOT}/stage_c_qwen35_fb.json" \
        2>&1 | tee "${RUN_ROOT}/stage_c_qwen35_fb.log" || rc_c=$?
    else
      echo "HF_PATH missing (${HF_PATH}); stage C skipped" | tee "${RUN_ROOT}/stage_c_qwen35_fb.log"
      rc_c=127
    fi

    echo "=== VERDICT: rc_a=${rc_a} rc_b=${rc_b} rc_c=${rc_c} (A+B gate) ==="
    printf "rc_a=%s rc_b=%s rc_c=%s\n" "${rc_a}" "${rc_b}" "${rc_c}" > "${RUN_ROOT}/verdict.txt"
    # Job rc reflects the low-skew primary evidence (unit tests + capture/replay).
    [[ ${rc_a} -eq 0 && ${rc_b} -eq 0 ]]
  '
