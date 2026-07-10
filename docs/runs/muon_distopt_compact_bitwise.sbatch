#!/usr/bin/env bash
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#SBATCH --job-name=muon-distopt-bitwise
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --account=coreai_devtech_all
#SBATCH --partition=batch
#SBATCH --time=00:40:00

# SNAPSHOT_ROOT must be a read-only `git archive` of MLITE_HEAD.  After extraction,
# write MLITE_HEAD to .mlite-source-head and `git rev-parse HEAD^{tree}` to
# .mlite-source-tree, then create .mlite-source-sha256 with:
#   find . -type f ! -name .mlite-source-sha256 -print0 | sort -z |
#     xargs -0 sha256sum >.mlite-source-sha256
# The job verifies all three markers before entering the container.

set -euo pipefail

readonly PINNED_MEGATRON_REVISION=d64ba4ccb1e3e878c15171c9cc58d5d3b46bf4d5
readonly PINNED_EMERGING_REVISION=b309e2f01cda75dc96a6dc1a2355a7b3b64b5e16
readonly PINNED_MBRIDGE_REVISION=a61943d7fcb34a190471cfeb0a0eb8bbda621ddf

MODE=${1:-${MODE:-}}
if [[ -z "${MODE}" ]]; then
  echo "Set MODE or pass one of: upstream, mlite, compare, all, adam" >&2
  exit 2
fi
case "${MODE}" in
  upstream|mlite|compare|all|adam) ;;
  *)
    echo "Unsupported MODE=${MODE}; expected upstream, mlite, compare, all, or adam" >&2
    exit 2
    ;;
esac

SNAPSHOT_ROOT=${SNAPSHOT_ROOT:?set SNAPSHOT_ROOT to the MLite source snapshot}
CONTAINER_IMAGE=${CONTAINER_IMAGE:?set CONTAINER_IMAGE to the pinned container image}
OUTPUT_ROOT=${OUTPUT_ROOT:?set OUTPUT_ROOT to a persistent output root}
MEGATRON_PIN_ROOT=${MEGATRON_PIN_ROOT:?set MEGATRON_PIN_ROOT to pinned Megatron d64ba4ccb}
EMERGING_PIN_ROOT=${EMERGING_PIN_ROOT:?set EMERGING_PIN_ROOT to pinned Emerging-Optimizers}
NVRX_SITE=${NVRX_SITE:?set NVRX_SITE to the proven nvidia-resiliency-ext 0.6 site-packages}
MLITE_HEAD=${MLITE_HEAD:?set MLITE_HEAD to the committed MLite snapshot revision}
MLITE_TREE=${MLITE_TREE:?set MLITE_TREE to the committed MLite snapshot tree}
RUN_ID=${RUN_ID:-${SLURM_JOB_ID:-}}
if [[ ! "${RUN_ID}" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  echo "RUN_ID must contain only letters, digits, dot, underscore, and dash" >&2
  exit 2
fi
for object_id in "${MLITE_HEAD}" "${MLITE_TREE}"; do
  if [[ ! "${object_id}" =~ ^[0-9a-f]{40}$ ]]; then
    echo "MLITE_HEAD and MLITE_TREE must be full 40-character Git object ids" >&2
    exit 2
  fi
done

for root in \
  "${SNAPSHOT_ROOT}" \
  "${OUTPUT_ROOT}" \
  "${MEGATRON_PIN_ROOT}" \
  "${EMERGING_PIN_ROOT}" \
  "${NVRX_SITE}"; do
  if [[ "${root}" != /* ]]; then
    echo "All snapshot/output/pin roots must be absolute: ${root}" >&2
    exit 2
  fi
done
if [[ ! -d "${NVRX_SITE}/nvidia_resiliency_ext-0.6.0.dist-info" ]]; then
  echo "NVRX_SITE does not contain the proven nvidia-resiliency-ext 0.6.0" >&2
  exit 2
fi
if [[ "${CONTAINER_IMAGE}" != /* && "${CONTAINER_IMAGE}" != docker://* ]]; then
  echo "CONTAINER_IMAGE must be an absolute snapshot path or docker:// reference" >&2
  exit 2
fi

actual_megatron_revision=$(git -C "${MEGATRON_PIN_ROOT}" rev-parse HEAD)
actual_emerging_revision=$(git -C "${EMERGING_PIN_ROOT}" rev-parse HEAD)
if [[ "${actual_megatron_revision}" != "${PINNED_MEGATRON_REVISION}" ]]; then
  echo "Megatron pin mismatch: ${actual_megatron_revision}" >&2
  exit 2
fi
if [[ "${actual_emerging_revision}" != "${PINNED_EMERGING_REVISION}" ]]; then
  echo "Emerging-Optimizers pin mismatch: ${actual_emerging_revision}" >&2
  exit 2
fi
for root in "${MEGATRON_PIN_ROOT}" "${EMERGING_PIN_ROOT}"; do
  if [[ -n "$(git -C "${root}" status --porcelain --untracked-files=all)" ]]; then
    echo "Pinned source checkout is dirty: ${root}" >&2
    exit 2
  fi
done
MEGATRON_TREE=$(git -C "${MEGATRON_PIN_ROOT}" rev-parse 'HEAD^{tree}')
EMERGING_TREE=$(git -C "${EMERGING_PIN_ROOT}" rev-parse 'HEAD^{tree}')

# The immutable snapshot staging recipe writes both object ids and a content manifest.
for marker in .mlite-source-head .mlite-source-tree .mlite-source-sha256; do
  if [[ ! -f "${SNAPSHOT_ROOT}/${marker}" ]]; then
    echo "MLite snapshot marker is missing: ${SNAPSHOT_ROOT}/${marker}" >&2
    exit 2
  fi
done
if [[ "$(<"${SNAPSHOT_ROOT}/.mlite-source-head")" != "${MLITE_HEAD}" ]]; then
  echo "MLite snapshot HEAD marker mismatch" >&2
  exit 2
fi
if [[ "$(<"${SNAPSHOT_ROOT}/.mlite-source-tree")" != "${MLITE_TREE}" ]]; then
  echo "MLite snapshot tree marker mismatch" >&2
  exit 2
fi
(cd "${SNAPSHOT_ROOT}" && sha256sum --check --strict --quiet .mlite-source-sha256)

mkdir -p "${OUTPUT_ROOT}"
RUN_ROOT="${OUTPUT_ROOT}/${RUN_ID}"
case "${MODE}" in
  all)
    if [[ -e "${RUN_ROOT}" ]]; then
      echo "Refusing to reuse existing all-mode run root: ${RUN_ROOT}" >&2
      exit 2
    fi
    ;;
  upstream)
    if [[ -e "${RUN_ROOT}/upstream" || -e "${RUN_ROOT}/checkpoints/upstream" ]]; then
      echo "Refusing to reuse existing upstream artifacts: ${RUN_ROOT}" >&2
      exit 2
    fi
    ;;
  mlite)
    if [[ -e "${RUN_ROOT}/mlite" || -e "${RUN_ROOT}/checkpoints/mlite" ]]; then
      echo "Refusing to reuse existing MLite artifacts: ${RUN_ROOT}" >&2
      exit 2
    fi
    ;;
  compare)
    if [[ ! -d "${RUN_ROOT}/upstream" || ! -d "${RUN_ROOT}/mlite" ]]; then
      echo "Compare requires both implementations under ${RUN_ROOT}" >&2
      exit 2
    fi
    ;;
  adam)
    if [[ -e "${RUN_ROOT}/adam" ]]; then
      echo "Refusing to reuse existing Adam artifacts: ${RUN_ROOT}/adam" >&2
      exit 2
    fi
    ;;
esac
mkdir -p "${RUN_ROOT}"

# Match the proven CW/Pyxis recipe. Source immutability is enforced above by
# clean Git pins plus the committed snapshot's HEAD/tree/content manifests.
mounts=/lustre:/lustre
if [[ "${MODE}" == adam ]]; then
  HF_PATH=${HF_PATH:?set HF_PATH for MODE=adam}
  MBRIDGE_PIN_ROOT=${MBRIDGE_PIN_ROOT:?set MBRIDGE_PIN_ROOT for MODE=adam}
  if [[ "${HF_PATH}" != /* || "${MBRIDGE_PIN_ROOT}" != /* ]]; then
    echo "HF_PATH and MBRIDGE_PIN_ROOT must be absolute" >&2
    exit 2
  fi
  actual_mbridge_revision=$(git -C "${MBRIDGE_PIN_ROOT}" rev-parse HEAD)
  if [[ "${actual_mbridge_revision}" != "${PINNED_MBRIDGE_REVISION}" ]]; then
    echo "MBridge pin mismatch: ${actual_mbridge_revision}" >&2
    exit 2
  fi
  if [[ -n "$(git -C "${MBRIDGE_PIN_ROOT}" status --porcelain --untracked-files=all)" ]]; then
    echo "Pinned MBridge checkout is dirty: ${MBRIDGE_PIN_ROOT}" >&2
    exit 2
  fi
  MBRIDGE_TREE=$(git -C "${MBRIDGE_PIN_ROOT}" rev-parse 'HEAD^{tree}')
  export HF_PATH MBRIDGE_PIN_ROOT MBRIDGE_TREE
fi

export MODE SNAPSHOT_ROOT CONTAINER_IMAGE OUTPUT_ROOT RUN_ROOT RUN_ID
export MEGATRON_PIN_ROOT EMERGING_PIN_ROOT NVRX_SITE
export MLITE_HEAD MLITE_TREE MEGATRON_TREE EMERGING_TREE
export PINNED_MEGATRON_REVISION PINNED_EMERGING_REVISION PINNED_MBRIDGE_REVISION

srun \
  --container-image="${CONTAINER_IMAGE}" \
  --container-mounts="${mounts}" \
  --container-workdir="${SNAPSHOT_ROOT}" \
  /bin/bash -c '
    set -euo pipefail
    : "${MODE:?}"
    : "${SNAPSHOT_ROOT:?}"
    : "${OUTPUT_ROOT:?}"
    : "${MEGATRON_PIN_ROOT:?}"
    : "${EMERGING_PIN_ROOT:?}"
    : "${RUN_ROOT:?}"
    : "${RUN_ID:?}"
    : "${NVRX_SITE:?}"

    unset CONDA_DEFAULT_ENV CONDA_PREFIX PYTHONHOME VIRTUAL_ENV
    unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES
    export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
    export PYTHONNOUSERSITE=1
    export PYTHONDONTWRITEBYTECODE=1
    export PYTHONPATH="${SNAPSHOT_ROOT}/experimental/lite:${MEGATRON_PIN_ROOT}:${EMERGING_PIN_ROOT}:${NVRX_SITE}"
    if [[ "${MODE}" == adam ]]; then
      export PYTHONPATH="${PYTHONPATH}:${MBRIDGE_PIN_ROOT}"
    fi
    export CUBLAS_WORKSPACE_CONFIG=:4096:8
    export MEGATRON_LITE_DETERMINISTIC=1
    export CUDA_DEVICE_MAX_CONNECTIONS=1
    export NVTE_ALLOW_NONDETERMINISTIC_ALGO=0
    export NCCL_ALGO=Ring
    export NCCL_NVLS_ENABLE=0

    harness="${SNAPSHOT_ROOT}/experimental/lite/examples/bench/muon_distopt_correctness.py"
    upstream_dir="${RUN_ROOT}/upstream"
    mlite_dir="${RUN_ROOT}/mlite"
    upstream_checkpoint_dir="${RUN_ROOT}/checkpoints/upstream"
    mlite_checkpoint_dir="${RUN_ROOT}/checkpoints/mlite"

    python - <<"PY"
import inspect
import importlib.metadata
import hashlib
import json
import os
from pathlib import Path

import emerging_optimizers
import megatron.core
import megatron.lite
import nvidia_resiliency_ext
import torch
from megatron.core.dist_checkpointing.strategies.nvrx import has_nvrx_async_support


def beneath(path: str, root: str) -> bool:
    return Path(path).resolve().is_relative_to(Path(root).resolve())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


core_source = inspect.getsourcefile(megatron.core)
lite_source = inspect.getsourcefile(megatron.lite)
emerging_source = inspect.getsourcefile(emerging_optimizers)
nvrx_source = inspect.getsourcefile(nvidia_resiliency_ext)
assert core_source and beneath(core_source, os.environ["MEGATRON_PIN_ROOT"]), core_source
assert lite_source and beneath(lite_source, os.environ["SNAPSHOT_ROOT"]), lite_source
assert emerging_source and beneath(emerging_source, os.environ["EMERGING_PIN_ROOT"]), emerging_source
assert nvrx_source and beneath(nvrx_source, os.environ["NVRX_SITE"]), nvrx_source
assert importlib.metadata.version("nvidia-resiliency-ext") == "0.6.0"
assert has_nvrx_async_support()
if os.environ["MODE"] == "adam":
    import mbridge

    mbridge_source = inspect.getsourcefile(mbridge)
    assert mbridge_source and beneath(mbridge_source, os.environ["MBRIDGE_PIN_ROOT"]), mbridge_source
print("PINNED_IMPORT_ROOTS_VERIFIED")
mode = os.environ["MODE"]
manifest = {
    "run_id": os.environ["RUN_ID"],
    "mode": os.environ["MODE"],
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    "container_image": os.environ["CONTAINER_IMAGE"],
    "mlite_head": os.environ["MLITE_HEAD"],
    "mlite_tree": os.environ["MLITE_TREE"],
    "megatron_revision": os.environ["PINNED_MEGATRON_REVISION"],
    "megatron_tree": os.environ["MEGATRON_TREE"],
    "emerging_revision": os.environ["PINNED_EMERGING_REVISION"],
    "emerging_tree": os.environ["EMERGING_TREE"],
    "nvidia_resiliency_ext": importlib.metadata.version("nvidia-resiliency-ext"),
    "allocated_gpus": os.environ.get("SLURM_GPUS_ON_NODE"),
    "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "topology": {
        "world": 1 if os.environ["MODE"] == "adam" else 2,
        "dp": 1 if os.environ["MODE"] == "adam" else 2,
        "tp": 1,
        "etp": 1,
        "pp": 1,
        "vpp": 1,
        "cp": 1,
        "ep": 1,
    },
}
if os.environ["MODE"] == "adam":
    manifest["mbridge_revision"] = os.environ["PINNED_MBRIDGE_REVISION"]
    manifest["mbridge_tree"] = os.environ["MBRIDGE_TREE"]
    checkpoint = Path(os.environ["HF_PATH"]).resolve()
    config_path = checkpoint / "config.json"
    index_path = checkpoint / "model.safetensors.index.json"
    assert config_path.is_file(), config_path
    manifest["checkpoint"] = str(checkpoint)
    manifest["checkpoint_fingerprints"] = {"config.json": sha256(config_path)}
    if index_path.is_file():
        manifest["checkpoint_fingerprints"][index_path.name] = sha256(index_path)
    manifest["adam_golden_contract"] = "slurm-12630675"
    manifest["mount_vision_model"] = True
job_id = os.environ.get("SLURM_JOB_ID", "no_slurm_job")
manifest_path = Path(os.environ["RUN_ROOT"]) / f"runtime_manifest.{mode}.{job_id}.json"
assert not manifest_path.exists(), f"refusing to overwrite runtime manifest: {manifest_path}"
manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
print(json.dumps(manifest, sort_keys=True))
PY

    run_implementation() {
      implementation=$1
      output_dir=$2
      checkpoint_dir=$3
      torchrun --standalone --nproc_per_node=2 "${harness}" run \
        --implementation "${implementation}" \
        --trajectory continuous \
        --output-dir "${output_dir}" \
        --checkpoint-dir "${checkpoint_dir}"
      torchrun --standalone --nproc_per_node=2 "${harness}" run \
        --implementation "${implementation}" \
        --trajectory save \
        --output-dir "${output_dir}" \
        --checkpoint-dir "${checkpoint_dir}"
      # A third torchrun is intentional: resume must build a fresh process/optimizer.
      torchrun --standalone --nproc_per_node=2 "${harness}" run \
        --implementation "${implementation}" \
        --trajectory resume \
        --output-dir "${output_dir}" \
        --checkpoint-dir "${checkpoint_dir}"
    }

    compare_implementations() {
      python "${harness}" compare \
        --upstream-dir "${upstream_dir}" \
        --mlite-dir "${mlite_dir}" \
        --output-json "${RUN_ROOT}/comparison.json"
      test -f "${RUN_ROOT}/NON_SKIP_MUON_DISTOPT_COMPACT_BITWISE_PASSED"
    }

    case "${MODE}" in
      upstream)
        run_implementation upstream "${upstream_dir}" "${upstream_checkpoint_dir}"
        ;;
      mlite)
        run_implementation mlite "${mlite_dir}" "${mlite_checkpoint_dir}"
        ;;
      compare)
        compare_implementations
        ;;
      all)
        run_implementation upstream "${upstream_dir}" "${upstream_checkpoint_dir}"
        run_implementation mlite "${mlite_dir}" "${mlite_checkpoint_dir}"
        compare_implementations
        ;;
      adam)
        export REPO_ROOT="${SNAPSHOT_ROOT}"
        export MEGATRON_ROOT="${MEGATRON_PIN_ROOT}"
        export EMERGING_ROOT="${EMERGING_PIN_ROOT}"
        export OUTPUT_DIR="${RUN_ROOT}/adam"
        export MLITE_CONTAINER_IMAGE="${CONTAINER_IMAGE}"
        bash "${SNAPSHOT_ROOT}/experimental/lite/examples/bench/scripts/run_pinned_adam_distopt_correctness.sh"

        python - <<"PY"
import json
import os
from pathlib import Path


output_dir = Path(os.environ["OUTPUT_DIR"])
comparison = json.loads((output_dir / "qwen35_correctness_compare.json").read_text())
assert comparison["passed"] is True, comparison
assert comparison["mismatches"] == [], comparison
assert comparison["samples"] == 2, comparison
assert comparison["max_loss_abs"] == 0.0, comparison
assert comparison["max_grad_norm_abs"] == 0.0, comparison

expected_steps = [
    {
        "loss": 13.027458190917969,
        "grad_norm": 120.75512734973202,
        "weight_sha256": "1e3176a8cb18d68c5da9bfa5f31a507fa8d51a3a5ddc10fbcd821260a4c6c980",
    },
    {
        "loss": 14.698704719543457,
        "grad_norm": 96.15656991334498,
        "weight_sha256": "e6d034f7e05ee5ee6a42ceddda8970874c6742baf59cade38f53550abd7aec29",
    },
]
eval_bf16_sha256 = "2f805802633927852c5ec87455b1afa3a68597e1e411b979f2678eaedfb1c710"
for backend in ("mlite", "mbridge"):
    result = json.loads((output_dir / f"qwen35_{backend}_correctness.json").read_text())
    assert result["backend"] == backend, result["backend"]
    assert len(result["steps"]) == 2, len(result["steps"])
    for step, expected in zip(result["steps"], expected_steps, strict=True):
        assert step["loss"]["value"] == expected["loss"], (backend, step)
        assert step["grad_norm"]["value"] == expected["grad_norm"], (backend, step)
        assert step["grad_fingerprint"]["tensor_count"] == 352, (backend, step)
        assert step["post_step_weights"]["tensor_count"] == 354, (backend, step)
        assert step["post_step_weights"]["sha256"] == expected["weight_sha256"], (
            backend,
            step,
        )
        assert step["update_successful"] is True, (backend, step)
    assert result["eval_logits"]["shape"] == [1, 8, 248320], backend
    assert result["eval_logits"]["sha256_as_bf16"] == eval_bf16_sha256, backend

verdict = {
    "golden_contract": "slurm-12630675",
    "mismatches": [],
    "non_skip": True,
    "passed": True,
    "post_step_tensor_count": 354,
    "samples": 2,
}
(output_dir / "golden_verdict.json").write_text(
    json.dumps(verdict, indent=2, sort_keys=True) + "\n"
)
print(json.dumps(verdict, sort_keys=True))
PY

        sha256sum \
          "${RUN_ROOT}/runtime_manifest.adam.${SLURM_JOB_ID}.json" \
          "${OUTPUT_DIR}/runtime_manifest.json" \
          "${OUTPUT_DIR}/qwen35_mlite_correctness.json" \
          "${OUTPUT_DIR}/qwen35_mbridge_correctness.json" \
          "${OUTPUT_DIR}/qwen35_correctness_compare.json" \
          "${OUTPUT_DIR}/golden_verdict.json" \
          >"${OUTPUT_DIR}/canonical_artifact_sha256.txt"
        echo NON_SKIP_PINNED_ADAM_DISTOPT_GATE_PASSED \
          | tee "${OUTPUT_DIR}/NON_SKIP_PINNED_ADAM_DISTOPT_GATE_PASSED"
        ;;
    esac
  '
