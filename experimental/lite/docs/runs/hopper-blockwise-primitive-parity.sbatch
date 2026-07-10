#!/usr/bin/env bash
#SBATCH --job-name=mlite-blockwise-parity
#SBATCH --account=coreai_devtech_all
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --time=01:00:00
#SBATCH --output=/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/bayan/work/codex/fp8-primitives-c9c00dfac-cw/primitive-parity-%j.out

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
    OVERLAY_SITE="$OVERLAY/lib/python3.12/site-packages"
    RUN_DIR="$RUN_ROOT/run-primitive-parity-${SLURM_JOB_ID}"
    REFERENCE_ROOT="$RUN_DIR/mcore-reference"
    OUTPUT="$RUN_ROOT/results/primitive-parity-${SLURM_JOB_ID}"

    export PATH="$OVERLAY/bin:/usr/local/bin:/usr/bin:/bin"
    export PYTHONPATH="$OVERLAY_SITE:$RUN_DIR/experimental/lite:$REFERENCE_ROOT"
    unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_EXE CONDA_PYTHON_EXE
    unset CONDA_BUILD_CROSS_COMPILATION CONDA_BUILD_SYSROOT CONDA_SHLVL
    unset CONDA_TOOLCHAIN_BUILD CONDA_TOOLCHAIN_HOST
    unset CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH LIBRARY_PATH
    unset GCC GCC_AR GCC_NM GCC_RANLIB GCC_EXEC_PREFIX COMPILER_PATH
    unset CFLAGS CXXFLAGS CPPFLAGS LDFLAGS CMAKE_ARGS CMAKE_PREFIX_PATH
    unset NVCC_APPEND_FLAGS NVCC_PREPEND_FLAGS NVCC_PREPEND_FLAGS_BACKUP
    export CC=/usr/bin/gcc
    export CXX=/usr/bin/g++
    export XDG_CACHE_HOME="$RUN_DIR/cache/xdg"
    export TRITON_CACHE_DIR="$RUN_DIR/cache/triton"
    export TORCHINDUCTOR_CACHE_DIR="$RUN_DIR/cache/torchinductor"
    export TMPDIR="$RUN_DIR/cache/tmp"
    export CUBLAS_WORKSPACE_CONFIG=:4096:8
    export NVTE_ALLOW_NONDETERMINISTIC_ALGO=0
    export NVTE_DEBUG=1
    export NVTE_DEBUG_LEVEL=1
    export OMP_NUM_THREADS=1
    export CUDA_DEVICE_MAX_CONNECTIONS=1
    unset NVTE_FP8_BLOCK_SCALING_FP32_SCALES NVTE_BACKWARD_OVERRIDE

    test -f "$OVERLAY/BUILD_OK"
    test -f "$RUN_ROOT/hopper_blockwise_primitive_parity.py"
    test "$(sha256sum "$RUN_ROOT/mcore-cf2f07d7.tar.gz" | cut -d " " -f 1)" = "c08272b18d171553f2dcd04937d27e77d3a6be223860726be7c56fcc90c558b1"
    mkdir -p "$RUN_DIR" "$REFERENCE_ROOT" "$OUTPUT" \
      "$XDG_CACHE_HOME" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$TMPDIR"
    tar -xf "$RUN_ROOT/mlite-fp8-c9c00dfac.tar" -C "$RUN_DIR"
    tar -xf "$RUN_ROOT/mcore-cf2f07d7.tar.gz" -C "$REFERENCE_ROOT"
    test "$(cat "$REFERENCE_ROOT/REFERENCE_COMMIT")" = "cf2f07d7b1315c96c05554c670c43207c6783e5e"
    cd "$RUN_DIR"
    sha256sum "$RUN_ROOT/hopper_blockwise_primitive_parity.py"
    python -c "from megatron.lite.primitive.precision import validate_hopper_environment; print(validate_hopper_environment())"
    torchrun --standalone --nproc_per_node=4 \
      "$RUN_ROOT/hopper_blockwise_primitive_parity.py" \
      --output "$OUTPUT"
    test -s "$OUTPUT/manifest.json"
    sha256sum "$OUTPUT/manifest.json"
    echo "HOPPER_BLOCKWISE_PRIMITIVE_PARITY_JOB_OK output=$OUTPUT"'
