#!/usr/bin/env bash
# Read-only runtime probe for the fixed MCore used by MLite's verl.vllm023 jobs.
# Submit from a Slurm login node: sbatch validation/probe_mcore_hdo_runtime.sbatch
# The paths are reused from the existing M-FSDP environment-probe receipt.
#SBATCH --job-name=mfsdp-hdo-runtime-probe
#SBATCH --account=coreai_devtech_all
#SBATCH --partition=cpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:03:00
#SBATCH --output=/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/bayan/code/runtime/mfsdp-hdo-probe-%j.log

set -euo pipefail

image=/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/bayan/code/verl_optimize/verl.vllm023.sqsh
mcore=/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/bayan/code/megatron_lite/mcore-pinned-6204b925

test -f "${image}"
test -f "${mcore}/megatron/core/__init__.py"

srun --container-image="${image}" --container-mounts=/lustre:/lustre \
  --container-workdir="${mcore}" /bin/bash --noprofile --norc -s <<'CONTAINER'
set -euo pipefail
unset PYTHONHOME VIRTUAL_ENV CONDA_PREFIX PYTHONUSERBASE
export PATH=/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin
export PYTHONNOUSERSITE=1
export PYTHONPATH=/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/bayan/code/megatron_lite/mcore-pinned-6204b925

echo "[HDO_PROBE] which_python=$(command -v python)"
python --version
git rev-parse HEAD
python - <<'PY'
import importlib
import inspect
from pathlib import Path

import megatron.core
import transformer_engine

print(f"[HDO_PROBE] megatron_core_file={megatron.core.__file__}")
print(f"[HDO_PROBE] transformer_engine_file={transformer_engine.__file__}")
candidates = (
    "megatron.core.optimizer.cpu_offloading.hybrid_device_optimizer",
    "megatron.core.optimizer.hybrid_device_optimizer",
)
for name in candidates:
    try:
        module = importlib.import_module(name)
    except Exception as exc:
        print(f"[HDO_PROBE] import_failed module={name} error={type(exc).__name__}:{exc}")
        continue
    cls = getattr(module, "HybridDeviceOptimizer", None)
    print(f"[HDO_PROBE] imported module={name} file={module.__file__} class_found={cls is not None}")
    if cls is None:
        continue
    source_file = Path(inspect.getsourcefile(cls)).resolve()
    start_line = inspect.getsourcelines(cls)[1]
    end_line = start_line + len(inspect.getsource(cls).splitlines()) - 1
    print(f"[HDO_PROBE] class_file={source_file} lines={start_line}-{end_line}")
    print(f"[HDO_PROBE] class_signature={inspect.signature(cls)}")
    print(f"[HDO_PROBE] init_signature={inspect.signature(cls.__init__)}")
PY
CONTAINER
