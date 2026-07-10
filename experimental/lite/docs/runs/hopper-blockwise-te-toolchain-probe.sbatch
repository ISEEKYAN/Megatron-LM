#!/usr/bin/env bash
#SBATCH --job-name=mlite-te-toolchain-probe
#SBATCH --account=coreai_devtech_all
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:10:00
#SBATCH --output=/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/bayan/work/codex/fp8-primitives-c9c00dfac-cw/te-toolchain-probe-%j.out

set -euo pipefail

RUN_ROOT=${RUN_ROOT:-/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/bayan/work/codex/fp8-primitives-c9c00dfac-cw}
BASE_IMAGE=${BASE_IMAGE:-/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/bayan/code/env/pytorch_26.04-py3.sqsh}

srun --ntasks=1 \
  --container-image="$BASE_IMAGE" \
  --container-mounts=/lustre:/lustre \
  --container-workdir="$RUN_ROOT" \
  bash -lc 'set -euo pipefail
    RUN_ROOT=${RUN_ROOT:-/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/bayan/work/codex/fp8-primitives-c9c00dfac-cw}
    PROBE_DIR="$RUN_ROOT/run-te-toolchain-probe-${SLURM_JOB_ID}"
    mkdir -p "$PROBE_DIR"
    printf "#include <cuda_runtime.h>\n#include <stdio.h>\nint main() { return 0; }\n" > "$PROBE_DIR/probe.cu"

    echo "== inherited compiler environment =="
    printenv | LC_ALL=C sort | grep -E "^(CONDA|CPATH|C_INCLUDE_PATH|CPLUS_INCLUDE_PATH|CUDA|CMAKE|GCC|COMPILER|NVCC|PATH|PYTHON)" || true
    echo "== compiler paths =="
    command -v gcc g++ /usr/bin/g++ /usr/local/cuda-13.2/bin/nvcc
    readlink -f /usr/bin/g++
    /usr/bin/g++ -v
    echo "== default NVCC dry run =="
    /usr/local/cuda-13.2/bin/nvcc -v --dryrun -c "$PROBE_DIR/probe.cu" -o "$PROBE_DIR/default.o" 2>&1
    echo "== explicit host compiler NVCC dry run =="
    /usr/local/cuda-13.2/bin/nvcc -v --dryrun -ccbin /usr/bin/g++ \
      -c "$PROBE_DIR/probe.cu" -o "$PROBE_DIR/explicit.o" 2>&1
    echo "== explicit host compiler compile =="
    /usr/local/cuda-13.2/bin/nvcc -v -ccbin /usr/bin/g++ \
      -c "$PROBE_DIR/probe.cu" -o "$PROBE_DIR/explicit.o" 2>&1
    echo "TE_TOOLCHAIN_PROBE_OK"'
