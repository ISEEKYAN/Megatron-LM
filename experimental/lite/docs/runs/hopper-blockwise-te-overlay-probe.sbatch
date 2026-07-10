#!/usr/bin/env bash
#SBATCH --job-name=mlite-te-blockwise-probe
#SBATCH --account=coreai_devtech_all
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --output=/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/bayan/work/codex/fp8-primitives-c9c00dfac-cw/te-overlay-probe-%j.out

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
    RUN_DIR="$RUN_ROOT/run-te-probe-${SLURM_JOB_ID}"
    REFERENCE_ROOT="$RUN_DIR/mcore-reference"

    export PATH="$OVERLAY/bin:/usr/local/bin:/usr/bin:/bin"
    export PYTHONPATH="$OVERLAY_SITE:$RUN_DIR/experimental/lite:$REFERENCE_ROOT"
    export CUBLAS_WORKSPACE_CONFIG=:4096:8
    export NVTE_ALLOW_NONDETERMINISTIC_ALGO=0
    export OMP_NUM_THREADS=1
    unset NVTE_FP8_BLOCK_SCALING_FP32_SCALES NVTE_BACKWARD_OVERRIDE

    test -f "$OVERLAY/BUILD_OK"
    test "$(sha256sum "$RUN_ROOT/mcore-cf2f07d7.tar.gz" | cut -d " " -f 1)" = "c08272b18d171553f2dcd04937d27e77d3a6be223860726be7c56fcc90c558b1"
    mkdir -p "$RUN_DIR" "$REFERENCE_ROOT"
    tar -xf "$RUN_ROOT/mlite-fp8-c9c00dfac.tar" -C "$RUN_DIR"
    tar -xf "$RUN_ROOT/mcore-cf2f07d7.tar.gz" -C "$REFERENCE_ROOT"
    test "$(cat "$REFERENCE_ROOT/REFERENCE_COMMIT")" = "cf2f07d7b1315c96c05554c670c43207c6783e5e"
    cd "$RUN_DIR"
    nvidia-smi -L
    python - <<"PY"
import pathlib

import torch
import transformer_engine as te
import transformer_engine.pytorch as tex
import transformer_engine_torch as tex_cpp
from transformer_engine.pytorch.quantization import check_fp8_block_scaling_support

from megatron.lite.primitive.precision import (
    build_hopper_blockwise_recipe,
    validate_hopper_environment,
)

torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)
torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32 = False

overlay = pathlib.Path("/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/bayan/code/te-8b9968255-cu132-py312-overlay-r7")
assert pathlib.Path(te.__file__).is_relative_to(overlay), te.__file__
environment = validate_hopper_environment()
supported, reason = check_fp8_block_scaling_support()
assert supported, reason
recipe = build_hopper_blockwise_recipe()

linear = tex.Linear(1024, 4096, bias=False, params_dtype=torch.bfloat16).cuda()
x = torch.randn(256, 2, 1024, device="cuda", dtype=torch.bfloat16, requires_grad=True)
with tex.fp8_autocast(enabled=True, fp8_recipe=recipe):
    y = linear(x)
y.float().square().mean().backward()
assert torch.isfinite(y).all() and torch.isfinite(x.grad).all()
assert torch.isfinite(linear.weight.grad).all()

grouped = tex.GroupedLinear(
    2, 1024, 8192, bias=False, params_dtype=torch.bfloat16
).cuda()
gx = torch.randn(32, 1024, device="cuda", dtype=torch.bfloat16, requires_grad=True)
with tex.fp8_autocast(enabled=True, fp8_recipe=recipe):
    gy = grouped(gx, [16, 16])
gy.float().square().mean().backward()
assert torch.isfinite(gy).all() and torch.isfinite(gx.grad).all()
assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in grouped.parameters())

print(f"python_te={te.__version__} path={te.__file__}")
print(f"torch={torch.__version__} cuda={torch.version.cuda} capability={torch.cuda.get_device_capability()}")
print(f"cublaslt={tex_cpp.get_cublasLt_version()} blockwise={(supported, reason)}")
print(f"environment={environment}")
print(f"linear_forward={tuple(y.shape)} dx={tuple(x.grad.shape)} dw={tuple(linear.weight.grad.shape)}")
print(f"grouped_forward={tuple(gy.shape)} dx={tuple(gx.grad.shape)} params={len(list(grouped.parameters()))}")
print("TE_BLOCKWISE_OVERLAY_GPU_PROBE_OK")
PY'
