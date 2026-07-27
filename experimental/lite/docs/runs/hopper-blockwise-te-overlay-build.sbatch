#!/usr/bin/env bash
#SBATCH --job-name=mlite-te-blockwise-build
#SBATCH --account=coreai_devtech_all
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --time=02:00:00
#SBATCH --output=/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/bayan/work/codex/fp8-primitives-c9c00dfac-cw/te-overlay-build-%j.out

set -euo pipefail

RUN_ROOT=${RUN_ROOT:-/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/bayan/work/codex/fp8-primitives-c9c00dfac-cw}
BASE_IMAGE=${BASE_IMAGE:-/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/bayan/code/env/pytorch_26.04-py3.sqsh}
OVERLAY=${OVERLAY:-/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/bayan/code/te-8b9968255-cu132-py312-overlay-r7}

srun --ntasks=1 \
  --container-image="$BASE_IMAGE" \
  --container-mounts=/lustre:/lustre \
  --container-workdir="$RUN_ROOT" \
  bash -lc 'set -euo pipefail
    RUN_ROOT=${RUN_ROOT:-/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/bayan/work/codex/fp8-primitives-c9c00dfac-cw}
    OVERLAY=${OVERLAY:-/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/bayan/code/te-8b9968255-cu132-py312-overlay-r7}
    CACHE_ROOT=/lustre/fs1/portfolios/coreai/users/bayan
    SOURCE_COMMIT=8b9968255eb879e6e390f427836906b29aad64d2
    RUN_DIR="$RUN_ROOT/run-te-build-${SLURM_JOB_ID}"
    SRC_ROOT="$RUN_DIR/TransformerEngine"
    NCCL_INCLUDE_ROOT="$RUN_DIR/nccl-include"

    export PATH=/usr/local/bin:/usr/bin:/bin
    unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_EXE CONDA_PYTHON_EXE
    unset CONDA_BUILD_CROSS_COMPILATION CONDA_BUILD_SYSROOT CONDA_SHLVL
    unset CONDA_TOOLCHAIN_BUILD CONDA_TOOLCHAIN_HOST
    unset CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH LIBRARY_PATH
    unset GCC_EXEC_PREFIX COMPILER_PATH CFLAGS CXXFLAGS CPPFLAGS LDFLAGS
    unset GCC GCC_AR GCC_NM GCC_RANLIB
    unset CMAKE_ARGS CMAKE_PREFIX_PATH CUDA_PATH
    unset NVCC_APPEND_FLAGS NVCC_PREPEND_FLAGS NVCC_PREPEND_FLAGS_BACKUP
    export CC=/usr/bin/gcc
    export CXX=/usr/bin/g++
    export NVCC_CXX_COMPILER=/usr/bin/g++
    export TMPDIR="$CACHE_ROOT/tmp/te-8b9968255-${SLURM_JOB_ID}"
    export XDG_CACHE_HOME="$CACHE_ROOT/.cache"
    export CUDA_HOME=/usr/local/cuda-13.2
    export CUDACXX=/usr/local/cuda-13.2/bin/nvcc
    export CUDAToolkit_ROOT=/usr/local/cuda-13.2
    export NVTE_CMAKE_EXTRA_ARGS="-DCUDAToolkit_ROOT=/usr/local/cuda-13.2 -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.2/bin/nvcc -DNCCL_INCLUDE_DIR=$NCCL_INCLUDE_ROOT -DCMAKE_CXX_FLAGS=-I$NCCL_INCLUDE_ROOT -DCMAKE_CUDA_FLAGS=-I$NCCL_INCLUDE_ROOT"
    export CMAKE_BUILD_PARALLEL_LEVEL=${SLURM_CPUS_PER_TASK:-96}
    export MAX_JOBS=${SLURM_CPUS_PER_TASK:-96}
    export NVTE_BUILD_THREADS=${SLURM_CPUS_PER_TASK:-96}
    export NVTE_CUDA_ARCHS=90
    export NVTE_FRAMEWORK=pytorch
    export NVTE_WITH_NCCL_EP=0

    test ! -e "$OVERLAY"
    mkdir -p "$RUN_DIR" "$TMPDIR" "$XDG_CACHE_HOME" "$NCCL_INCLUDE_ROOT"
    ln -s /usr/include/nccl.h "$NCCL_INCLUDE_ROOT/nccl.h"
    command -v "$CC"
    command -v "$CXX"
    "$CXX" -v

    git init "$SRC_ROOT"
    git -C "$SRC_ROOT" remote add origin https://github.com/NVIDIA/TransformerEngine.git
    git -C "$SRC_ROOT" fetch --depth=1 origin "$SOURCE_COMMIT"
    SOURCE_SHA=$(git -C "$SRC_ROOT" rev-parse "FETCH_HEAD^{commit}")
    git -C "$SRC_ROOT" checkout --detach "$SOURCE_SHA"
    git -C "$SRC_ROOT" submodule update --init --recursive --depth=1
    test "$(git -C "$SRC_ROOT" rev-parse HEAD)" = "$SOURCE_SHA"
    case "$SOURCE_SHA" in
      "$SOURCE_COMMIT"*) ;;
      *) echo "unexpected Transformer Engine commit: $SOURCE_SHA" >&2; exit 1 ;;
    esac

    python -m venv --system-site-packages "$OVERLAY"
    . "$OVERLAY/bin/activate"
    python -m pip install --upgrade pip "setuptools<82" wheel ninja cmake pybind11
    python -m pip install --no-build-isolation --no-deps "$SRC_ROOT"

    python - "$SOURCE_SHA" "$OVERLAY" <<"PY"
import pathlib
import sys

import torch
import transformer_engine as te

source_sha, overlay = sys.argv[1:]
assert source_sha == "8b9968255eb879e6e390f427836906b29aad64d2", source_sha
assert te.__version__.startswith("2.18.0.dev0"), te.__version__
assert pathlib.Path(te.__file__).is_relative_to(pathlib.Path(overlay)), te.__file__
print(f"torch={torch.__version__} cuda={torch.version.cuda}")
print(f"te={te.__version__} path={te.__file__}")
print(f"source_sha={source_sha}")
print("TE_BLOCKWISE_OVERLAY_IMPORT_OK")
PY

    printf "SOURCE_SHA=%s\nBASE_IMAGE=%s\nOVERLAY=%s\n" \
      "$SOURCE_SHA" "/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/bayan/code/env/pytorch_26.04-py3.sqsh" "$OVERLAY" \
      > "$OVERLAY/BUILD_OK"
    printenv | LC_ALL=C sort > "$RUN_DIR/build-environment.txt"
    python -m pip freeze > "$RUN_DIR/pip-freeze.txt"
    echo "TE_BLOCKWISE_OVERLAY_BUILD_OK overlay=$OVERLAY source_sha=$SOURCE_SHA"'
