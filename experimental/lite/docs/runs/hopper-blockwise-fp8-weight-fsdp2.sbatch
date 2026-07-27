#!/usr/bin/env bash
#SBATCH --job-name=mlite-fp8w-fsdp2
#SBATCH --account=coreai_devtech_all
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=00:40:00
#SBATCH --output=/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/bayan/work/codex/fp8-primitives-c9c00dfac-cw/fp8-weight-fsdp2-%j.out

set -euo pipefail

RUN_ROOT=/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/bayan/work/codex/fp8-primitives-c9c00dfac-cw
BASE_IMAGE=/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/bayan/code/env/pytorch_26.04-py3.sqsh
TARBALL=$RUN_ROOT/fp8-weight-fsdp2/mlite-fp8-fsdp2-727fd8ed5.tar
TARBALL_SHA=832b979df00156d4a6fdd09ac1d539d4a050f1c64028b88a6024ac95e6622116

srun --ntasks=1 \
  --container-image="$BASE_IMAGE" \
  --container-mounts=/lustre:/lustre \
  --container-workdir="$RUN_ROOT" \
  bash -lc "set -euo pipefail
    RUN_ROOT=$RUN_ROOT
    TARBALL=$TARBALL
    RUN_DIR=\$RUN_ROOT/fp8-weight-fsdp2/run-\${SLURM_JOB_ID}
    OUTPUT=\$RUN_ROOT/results/fp8-weight-fsdp2-\${SLURM_JOB_ID}

    export PATH=/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin
    unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_EXE CONDA_PYTHON_EXE || true
    unset CONDA_SHLVL CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH LIBRARY_PATH || true
    export PYTHONNOUSERSITE=1
    export CC=/usr/bin/gcc CXX=/usr/bin/g++
    export XDG_CACHE_HOME=\$RUN_DIR/cache/xdg
    export TRITON_CACHE_DIR=\$RUN_DIR/cache/triton
    export TORCHINDUCTOR_CACHE_DIR=\$RUN_DIR/cache/torchinductor
    export TMPDIR=\$RUN_DIR/cache/tmp
    export HOME=\$RUN_DIR/home
    export CUBLAS_WORKSPACE_CONFIG=:4096:8
    export CUDA_DEVICE_MAX_CONNECTIONS=1
    export OMP_NUM_THREADS=1

    test -f \$TARBALL
    test \"\$(sha256sum \$TARBALL | cut -d' ' -f1)\" = $TARBALL_SHA
    mkdir -p \$RUN_DIR \$OUTPUT \$XDG_CACHE_HOME \$TRITON_CACHE_DIR \$TORCHINDUCTOR_CACHE_DIR \$TMPDIR \$HOME
    tar -xf \$TARBALL -C \$RUN_DIR
    export PYTHONPATH=\$RUN_DIR/experimental/lite:/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/bayan/work/codex/fp8-primitives-c9c00dfac-cw/pytools:/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/bayan/code/megatron_lite/Megatron-LM

    cd \$RUN_DIR
    python -c 'import torch, transformer_engine; print(\"torch\", torch.__version__, \"TE\", transformer_engine.__version__, \"cuda\", torch.cuda.is_available(), torch.cuda.get_device_name(0))'
    python -c 'from megatron.lite.primitive.precision import validate_hopper_environment; print(validate_hopper_environment())'

    python -m pytest -v -p no:cacheprovider \
      experimental/lite/tests/unit/primitive/test_fsdp2_offload_gpu.py \
      -k fp8_weight \
      --junitxml=\$OUTPUT/fp8-weight-fsdp2.xml \
      | tee \$OUTPUT/fp8-weight-fsdp2.log

    echo MLITE_FP8_WEIGHT_FSDP2_JOB_OK output=\$OUTPUT"
