#!/usr/bin/env bash
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
set -euo pipefail

readonly PINNED_MEGATRON_REVISION=d64ba4ccb1e3e878c15171c9cc58d5d3b46bf4d5
readonly PINNED_EMERGING_REVISION=b309e2f01cda75dc96a6dc1a2355a7b3b64b5e16

REPO_ROOT=${REPO_ROOT:-$(git rev-parse --show-toplevel)}
MEGATRON_ROOT=${MEGATRON_ROOT:?set MEGATRON_ROOT to the pinned Megatron checkout}
EMERGING_ROOT=${EMERGING_ROOT:?set EMERGING_ROOT to the pinned Emerging-Optimizers checkout}
HF_PATH=${HF_PATH:?set HF_PATH to the fixed HuggingFace Qwen3.5 checkpoint}
OUTPUT_DIR=${OUTPUT_DIR:?set OUTPUT_DIR to a persistent artifact directory}
MLITE_CONTAINER_IMAGE=${MLITE_CONTAINER_IMAGE:?record the Slurm container image path}

actual_megatron_revision=$(git -C "${MEGATRON_ROOT}" rev-parse HEAD)
actual_emerging_revision=$(git -C "${EMERGING_ROOT}" rev-parse HEAD)
if [[ "${actual_megatron_revision}" != "${PINNED_MEGATRON_REVISION}" ]]; then
  echo "Megatron revision mismatch: ${actual_megatron_revision}" >&2
  exit 2
fi
if [[ "${actual_emerging_revision}" != "${PINNED_EMERGING_REVISION}" ]]; then
  echo "Emerging-Optimizers revision mismatch: ${actual_emerging_revision}" >&2
  exit 2
fi
if [[ ! -f "${HF_PATH}/config.json" ]]; then
  echo "Qwen3.5 checkpoint is missing config.json: ${HF_PATH}" >&2
  exit 2
fi

mkdir -p "${OUTPUT_DIR}"
export REPO_ROOT MEGATRON_ROOT EMERGING_ROOT HF_PATH OUTPUT_DIR MLITE_CONTAINER_IMAGE
export PINNED_MEGATRON_REVISION PINNED_EMERGING_REVISION
export PYTHONPATH="${REPO_ROOT}/experimental/lite:${MEGATRON_ROOT}:${EMERGING_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

# This is the previously validated Adam×DistOpt bitwise topology. Keep these values fixed;
# Muon topology/layout variants are validated by subsequent implementation slices.
export REFERENCE_BACKEND=mbridge
export NPROC=1 TP=1 ETP=1 EP=1 PP=1 CP=1
export STEPS=2 NUM_MICROBATCHES=1 SEQ_LEN=8 SEED=42
export TRUNCATE_LAYERS=1 KEEP_EXPERTS=2
export MEGATRON_LITE_DETERMINISTIC=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

python - <<'PY'
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path

import torch
import transformer_engine
import emerging_optimizers
from megatron.core.optimizer.optimizer_config import OptimizerConfig as CoreOptimizerConfig


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


checkpoint = Path(os.environ["HF_PATH"]).resolve()
fingerprints = {"config.json": sha256(checkpoint / "config.json")}
index = checkpoint / "model.safetensors.index.json"
if index.exists():
    fingerprints[index.name] = sha256(index)
try:
    emerging_version = importlib.metadata.version("emerging-optimizers")
except importlib.metadata.PackageNotFoundError:
    emerging_version = "source-checkout"

manifest = {
    "container_image": os.environ["MLITE_CONTAINER_IMAGE"],
    "megatron_revision": os.environ["PINNED_MEGATRON_REVISION"],
    "emerging_optimizers_revision": os.environ["PINNED_EMERGING_REVISION"],
    "megatron_optimizer_config_source": inspect.getsourcefile(CoreOptimizerConfig),
    "torch_version": torch.__version__,
    "transformer_engine_version": transformer_engine.__version__,
    "emerging_optimizers_version": emerging_version,
    "emerging_optimizers_source": emerging_optimizers.__file__,
    "checkpoint": str(checkpoint),
    "checkpoint_fingerprints": fingerprints,
    "topology": {"world": 1, "tp": 1, "etp": 1, "ep": 1, "pp": 1, "cp": 1},
    "seed": 42,
    "steps": 2,
    "num_microbatches": 1,
    "seq_len": 8,
    "truncate_layers": 1,
    "keep_experts": 2,
    "optimizer": "adam",
    "optimizer_backend": "dist_opt",
    "reference_backend": "mbridge",
}
output = Path(os.environ["OUTPUT_DIR"]) / "runtime_manifest.json"
output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
print(json.dumps(manifest, sort_keys=True))
PY

bash "${REPO_ROOT}/experimental/lite/examples/bench/scripts/run_qwen35_correctness_pair.sh"

sha256sum \
  "${OUTPUT_DIR}/runtime_manifest.json" \
  "${OUTPUT_DIR}/qwen35_mlite_correctness.json" \
  "${OUTPUT_DIR}/qwen35_mbridge_correctness.json" \
  "${OUTPUT_DIR}/qwen35_correctness_compare.json" \
  >"${OUTPUT_DIR}/artifact_sha256.txt"
