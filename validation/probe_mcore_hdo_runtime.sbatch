#!/usr/bin/env bash
# Read-only runtime probe for the fixed MCore used by MLite's verl.vllm023 jobs.
# Submit from a Slurm login node after copying this file into the task-specific
# remote directory.  The MCore checkout is the fixed runtime dependency, not
# the shared MLite checkout.
#SBATCH --job-name=mfsdp-hdo-runtime-probe
#SBATCH --account=coreai_devtech_all
#SBATCH --partition=interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=8
#SBATCH --time=00:03:00
#SBATCH --output=/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/bayan/code/runtime/task_1_13_24_21/mfsdp-hdo-probe-%j.log

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

echo "[HDO_PROBE] python3_command=$(command -v python3)"
echo "[HDO_PROBE] python3_resolved=$(readlink -f /usr/bin/python3)"
/usr/bin/python3 --version
git rev-parse HEAD
/usr/bin/python3 - <<'PY'
import importlib
import inspect
from pathlib import Path
import traceback

import megatron.core
import transformer_engine
import torch

print(f"[HDO_PROBE] megatron_core_file={megatron.core.__file__}")
print(f"[HDO_PROBE] transformer_engine_file={transformer_engine.__file__}")
candidates = (
    "megatron.core.optimizer.cpu_offloading.hybrid_optimizer",
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
    if name == "megatron.core.optimizer.cpu_offloading.hybrid_optimizer":
        from transformer_engine.pytorch.optimizers import FusedAdam

        print(
            "[HDO_PROBE] construction_dependencies="
            f"cpu_optimizer_cls={torch.optim.AdamW.__module__}.{torch.optim.AdamW.__name__} "
            f"gpu_optimizer_cls={FusedAdam.__module__}.{FusedAdam.__name__}"
        )
        try:
            parameter = torch.nn.Parameter(torch.ones(8, device="cuda", dtype=torch.bfloat16))
            optimizer = cls(
                [{"params": [parameter], "lr": 1.0e-3}],
                offload_fraction=1.0,
                cpu_optimizer_cls=torch.optim.AdamW,
                gpu_optimizer_cls=FusedAdam,
                param_update_in_fp32=True,
            )
            print(
                "[HDO_PROBE] construction_succeeded "
                f"cpu_optimizers={len(optimizer.cpu_optimizers)} "
                f"gpu_optimizer_present={optimizer.gpu_optimizer is not None} "
                f"cpu_copy_dtype={next(iter(optimizer.cpu_copys_map_gpu_param)).dtype} "
                f"master_dtype={next(iter(optimizer.param_to_fp32_param.values())).dtype} "
                "gradient_input=getattr(param, 'decoupled_grad', param.grad)"
            )
        except Exception:
            print("[HDO_PROBE] construction_failed_traceback")
            traceback.print_exc()
            raise
PY
CONTAINER
