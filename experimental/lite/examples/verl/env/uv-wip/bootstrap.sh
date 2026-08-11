#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -L)"
EXCLUDES="${SCRIPT_DIR}/uv-excludes.txt"
SOURCE_REQUIREMENTS="${SCRIPT_DIR}/source-requirements.txt"
EXPECTED_TORCH_VERSION="2.12.0a0+5aff3928d8.nv26.05"

fail() {
  printf 'uv-wip bootstrap: %s\n' "$*" >&2
  exit 64
}

: "${UV_WIP_SITE:?UV_WIP_SITE is required}"
: "${VLLM_WHEEL_URL:?VLLM_WHEEL_URL is required}"
UV_BIN="${UV_BIN:-uv}"

command -v "${UV_BIN}" >/dev/null || fail "UV_BIN=${UV_BIN} is not executable"
[[ -f "${EXCLUDES}" ]] || fail "missing exclusion contract: ${EXCLUDES}"
[[ -f "${SOURCE_REQUIREMENTS}" ]] || fail "missing source requirements: ${SOURCE_REQUIREMENTS}"
[[ ! -e "${UV_WIP_SITE}" ]] || fail "UV_WIP_SITE must not already exist: ${UV_WIP_SITE}"

base_torch_version="$(/usr/bin/python3 -c 'import torch; print(torch.__version__)')"
[[ "${base_torch_version}" == "${EXPECTED_TORCH_VERSION}" ]] || \
  fail "base /usr/bin/python3 has torch ${base_torch_version}, expected ${EXPECTED_TORCH_VERSION}"

uv() {
  "${UV_BIN}" "$@"
}

# Resolve first, then reject any forbidden package before uv writes the site.
# This is intentionally a full dependency resolution, not a --no-deps install.
PLAN_PARENT="$(dirname "${UV_WIP_SITE}")"
mkdir -p "${PLAN_PARENT}"
PLAN_TARGET="$(mktemp -d "${PLAN_PARENT}/.uv-wip-plan.XXXXXX")"
cleanup_plan() {
  rm -rf -- "${PLAN_TARGET}"
}
trap cleanup_plan EXIT

set +e
plan="$({
  uv pip install \
    --python /usr/bin/python3 \
    --target "${PLAN_TARGET}" \
    --excludes "${EXCLUDES}" \
    --requirements "${SOURCE_REQUIREMENTS}" \
    --dry-run \
    "${VLLM_WHEEL_URL}"
} 2>&1)"
plan_status="$?"
set -e
printf '%s\n' "${plan}"
(( plan_status == 0 )) || fail "resolution failed with status ${plan_status}"

if grep -Eiq '(^|[[:space:]+])(torch|torchvision|torchaudio|torchcodec|torch-c-dlpack-ext|functorch|triton|nvidia[-_][[:alnum:]._-]*)([=[:space:]]|$)' <<<"${plan}"; then
  fail "dependency plan contains a container-owned torch/CUDA distribution"
fi

cleanup_plan
trap - EXIT

BUILD_TARGET="$(mktemp -d "${PLAN_PARENT}/.uv-wip-build.XXXXXX")"
cleanup_build() {
  rm -rf -- "${BUILD_TARGET}"
}
trap cleanup_build EXIT

uv pip install \
  --python /usr/bin/python3 \
  --target "${BUILD_TARGET}" \
  --excludes "${EXCLUDES}" \
  --requirements "${SOURCE_REQUIREMENTS}" \
  "${VLLM_WHEEL_URL}"

# vLLM main imports its bundled kernels through the historical top-level
# package name before VERL's runtime compatibility hook can install an alias.
if [[ -d "${BUILD_TARGET}/vllm/third_party/triton_kernels" && ! -e "${BUILD_TARGET}/triton_kernels" ]]; then
  (
    cd "${BUILD_TARGET}"
    ln -s vllm/third_party/triton_kernels triton_kernels
  )
fi

UV_WIP_SITE="${BUILD_TARGET}" EXPECTED_TORCH_VERSION="${EXPECTED_TORCH_VERSION}" \
PYTHONPATH="${BUILD_TARGET}" /usr/bin/python3 - <<'PY'
import importlib.metadata
import json
import os
import re
import torch

site = os.environ["UV_WIP_SITE"]
expected_torch = os.environ["EXPECTED_TORCH_VERSION"]
if torch.__version__ != expected_torch:
    raise SystemExit(
        f"site replaced base torch: got {torch.__version__}, expected {expected_torch}"
    )

distributions = []
for distribution in importlib.metadata.distributions(path=[site]):
    name = re.sub(r"[-_.]+", "-", distribution.metadata["Name"].lower())
    version = distribution.version
    distributions.append({"name": name, "version": version})

for distribution in distributions:
    name = distribution["name"]
    if name in {
        "torch",
        "torchvision",
        "torchaudio",
        "torchcodec",
        "torch-c-dlpack-ext",
        "functorch",
        "triton",
    } or name.startswith("nvidia-"):
        raise SystemExit(f"forbidden distribution in uv site: {name}")

print("UV_WIP_DIST_INFO=" + json.dumps(sorted(distributions, key=lambda item: item["name"])))
print(f"UV_WIP_BASE_TORCH={torch.__version__} {torch.__file__}")
PY

# Publish only a completely installed and audited site. The staging directory
# is on the target filesystem, so this rename is atomic to env.sh readers.
touch "${BUILD_TARGET}/.uv-wip-ready"
mv -T -- "${BUILD_TARGET}" "${UV_WIP_SITE}"
trap - EXIT
