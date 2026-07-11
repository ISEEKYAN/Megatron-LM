#!/usr/bin/env bash
# Read-only capability inventory for the canonical image's TE 2.15 package.
# This is intentionally not a profile validation: it records whether each
# mandatory blockwise primitive can run before an overlay is considered.
#SBATCH --job-name=mlite-te215-blockwise-inventory
#SBATCH --account=coreai_devtech_all
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --output=/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/bayan/work/codex/fp8-primitives-c9c00dfac-cw/te215-blockwise-inventory-%j.out

set -euo pipefail

RUN_ROOT=${RUN_ROOT:-/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/bayan/work/codex/fp8-primitives-c9c00dfac-cw}
BASE_IMAGE=${BASE_IMAGE:-/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/bayan/code/env/pytorch_26.04-py3.sqsh}

srun --ntasks=1 \
  --container-image="$BASE_IMAGE" \
  --container-mounts=/lustre:/lustre \
  --container-workdir="$RUN_ROOT" \
  bash -lc 'set -euo pipefail
    export PATH=/usr/local/bin:/usr/bin:/bin
    export PYTHONPATH=""
    export CUBLAS_WORKSPACE_CONFIG=:4096:8
    export NVTE_ALLOW_NONDETERMINISTIC_ALGO=0
    export OMP_NUM_THREADS=1
    unset NVTE_FP8_BLOCK_SCALING_FP32_SCALES NVTE_BACKWARD_OVERRIDE

    python - <<"PY"
import json
import traceback

import torch

result = {"operations": {}}

try:
    import transformer_engine as te
    import transformer_engine.pytorch as tex
    import transformer_engine_torch as tex_cpp
    from transformer_engine.common.recipe import Format
    from transformer_engine.common.recipe import Float8BlockScaling
    from transformer_engine.common.recipe import MMParams
except Exception as error:
    result["import_error"] = repr(error)
    print(json.dumps(result, sort_keys=True))
    print("TE215_BLOCKWISE_INVENTORY_COMPLETE status=package-unavailable")
    raise SystemExit(0)

result.update(
    te_version=getattr(te, "__version__", "unknown"),
    te_path=te.__file__,
    torch=torch.__version__,
    cuda=torch.version.cuda,
    capability=torch.cuda.get_device_capability(),
    cublaslt=(getattr(tex_cpp, "get_cublasLt_version", lambda: "unavailable")()),
)

try:
    from transformer_engine.pytorch.quantization import check_fp8_block_scaling_support
    result["block_scaling_support"] = list(check_fp8_block_scaling_support())
except Exception as error:
    result["block_scaling_support"] = [False, repr(error)]

try:
    recipe = Float8BlockScaling(
        fp8_format=Format.E4M3,
        x_block_scaling_dim=1,
        w_block_scaling_dim=2,
        grad_block_scaling_dim=1,
        fp8_gemm_fprop=MMParams(use_split_accumulator=True),
        fp8_gemm_dgrad=MMParams(use_split_accumulator=True),
        fp8_gemm_wgrad=MMParams(use_split_accumulator=True),
        fp8_dpa=False,
        fp8_mha=False,
    )
except Exception as error:
    result["recipe_error"] = repr(error)
    print(json.dumps(result, sort_keys=True))
    print("TE215_BLOCKWISE_INVENTORY_COMPLETE status=recipe-unavailable")
    raise SystemExit(0)

torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)
torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32 = False

def probe(name, factory, args):
    try:
        module = factory().cuda()
        x = torch.randn(*args, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        with tex.fp8_autocast(enabled=True, fp8_recipe=recipe):
            y = module(x) if name != "grouped_linear" else module(x, [16, 16])
        y.float().square().mean().backward()
        assert torch.isfinite(y).all() and torch.isfinite(x.grad).all()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters())
        result["operations"][name] = {"status": "ran", "output": list(y.shape)}
    except Exception as error:
        result["operations"][name] = {
            "status": "unavailable",
            "error": repr(error),
            "traceback": traceback.format_exc(limit=2),
        }

probe("linear", lambda: tex.Linear(1024, 4096, bias=False, params_dtype=torch.bfloat16), (256, 2, 1024))
probe("layernorm_linear", lambda: tex.LayerNormLinear(1024, 4096, bias=False, normalization="RMSNorm", params_dtype=torch.bfloat16), (256, 2, 1024))
probe("grouped_linear", lambda: tex.GroupedLinear(2, 1024, 8192, bias=False, params_dtype=torch.bfloat16), (32, 1024))

all_ran = all(value["status"] == "ran" for value in result["operations"].values())
result["all_mandatory_ops_ran"] = all_ran
print(json.dumps(result, sort_keys=True))
status = "all-ran" if all_ran else "missing-op"
print(f"TE215_BLOCKWISE_INVENTORY_COMPLETE status={status}")
PY'
