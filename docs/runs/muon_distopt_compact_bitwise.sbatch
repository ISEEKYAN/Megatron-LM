#!/usr/bin/env bash
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#SBATCH --job-name=muon-distopt-layout-overlap
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --account=coreai_devtech_all
#SBATCH --partition=batch
#SBATCH --time=00:40:00

# This recipe retains its historical filename so the validated compact setup stays
# discoverable; MODE=all now covers compact/padded x overlap off/on.
# SNAPSHOT_ROOT must be a read-only `git archive` of MLITE_HEAD.  After extraction,
# write MLITE_HEAD to .mlite-source-head and `git rev-parse HEAD^{tree}` to
# .mlite-source-tree, then create .mlite-source-sha256 with:
#   find . -type f ! -name .mlite-source-sha256 -print0 | sort -z |
#     xargs -0 sha256sum >.mlite-source-sha256
# The job verifies all three markers before entering the container.

set -euo pipefail

readonly PINNED_MEGATRON_REVISION=d64ba4ccb1e3e878c15171c9cc58d5d3b46bf4d5
readonly PINNED_EMERGING_REVISION=b309e2f01cda75dc96a6dc1a2355a7b3b64b5e16

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
  if [[ "${HF_PATH}" != /* ]]; then
    echo "HF_PATH must be absolute" >&2
    exit 2
  fi
  export HF_PATH
fi

export MODE SNAPSHOT_ROOT CONTAINER_IMAGE OUTPUT_ROOT RUN_ROOT RUN_ID
export MEGATRON_PIN_ROOT EMERGING_PIN_ROOT NVRX_SITE
export MLITE_HEAD MLITE_TREE MEGATRON_TREE EMERGING_TREE
export PINNED_MEGATRON_REVISION PINNED_EMERGING_REVISION

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
if os.environ["MODE"] == "all":
    manifest["muon_configurations"] = [
        "compact_off",
        "compact_on",
        "padded_off",
        "padded_on",
    ]
if os.environ["MODE"] == "adam":
    checkpoint = Path(os.environ["HF_PATH"]).resolve()
    config_path = checkpoint / "config.json"
    index_path = checkpoint / "model.safetensors.index.json"
    assert config_path.is_file(), config_path
    manifest["checkpoint"] = str(checkpoint)
    manifest["checkpoint_fingerprints"] = {"config.json": sha256(config_path)}
    if index_path.is_file():
        manifest["checkpoint_fingerprints"][index_path.name] = sha256(index_path)
    manifest["adam_model_contract"] = "compact_text_only_qwen35"
    manifest["adam_model_contract_evidence"] = "slurm-13635323"
    manifest["adam_numeric_contract_evidence"] = "slurm-13694402-mlite-subrun"
    manifest["mount_vision_model"] = False
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
      test -f "${RUN_ROOT}/NON_SKIP_MUON_DISTOPT_BITWISE_PASSED"
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
        for config in compact_off compact_on padded_off padded_on; do
          export MUON_LAYOUT=${config%_*}
          export MUON_OVERLAP=${config#*_}
          config_root="${RUN_ROOT}/${config}"
          upstream_dir="${config_root}/upstream"
          mlite_dir="${config_root}/mlite"
          upstream_checkpoint_dir="${config_root}/checkpoints/upstream"
          mlite_checkpoint_dir="${config_root}/checkpoints/mlite"
          mkdir -p "${config_root}"
          run_implementation upstream "${upstream_dir}" "${upstream_checkpoint_dir}"
          run_implementation mlite "${mlite_dir}" "${mlite_checkpoint_dir}"
          RUN_ROOT="${config_root}" compare_implementations
        done
        ;;
      adam)
        output_dir="${RUN_ROOT}/adam"
        mkdir -p "${output_dir}"
        impl_cfg_json="{\"mount_vision_model\": false}"
        torchrun --nproc_per_node 1 \
          "${SNAPSHOT_ROOT}/experimental/lite/examples/bench/correctness.py" run \
          --backend mlite \
          --hf-path "${HF_PATH}" \
          --model-name qwen3_5 \
          --tp 1 \
          --etp 1 \
          --ep 1 \
          --pp 1 \
          --cp 1 \
          --steps 2 \
          --num-microbatches 1 \
          --seq-len 8 \
          --seed 42 \
          --truncate-layers 1 \
          --keep-experts 2 \
          --disable-mtp \
          --same-data-across-dp \
          --impl-cfg-json "${impl_cfg_json}" \
          --output-json "${output_dir}/qwen35_mlite_correctness.json" \
          2>&1 | tee "${output_dir}/qwen35_mlite_correctness.log"

        python "${harness}" validate-adam \
          --input-json "${output_dir}/qwen35_mlite_correctness.json" \
          --output-json "${output_dir}/text_only_verdict.json"
        test -f "${output_dir}/NON_SKIP_PINNED_ADAM_DISTOPT_GATE_PASSED"

        sha256sum \
          "${RUN_ROOT}/runtime_manifest.adam.${SLURM_JOB_ID}.json" \
          "${output_dir}/qwen35_mlite_correctness.json" \
          "${output_dir}/text_only_verdict.json" \
          "${output_dir}/NON_SKIP_PINNED_ADAM_DISTOPT_GATE_PASSED" \
          >"${output_dir}/canonical_artifact_sha256.txt"
        ;;
    esac
  '
