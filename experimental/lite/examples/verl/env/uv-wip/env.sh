#!/usr/bin/env bash
set -euo pipefail

fail() {
  printf 'uv-wip env: %s\n' "$*" >&2
  exit 64
}

require_root() {
  local variable_name="$1"
  local marker="$2"
  local value="${!variable_name:-}"

  [[ -n "${value}" ]] || fail "${variable_name} is required"
  [[ -d "${value}/${marker}" ]] || fail "${variable_name}=${value} does not contain ${marker}"
}

require_root UV_WIP_SITE vllm
require_root VERL_ROOT verl
require_root MEGATRON_ROOT megatron/core
require_root MLITE_ROOT experimental/lite/megatron/lite
[[ -d "${MLITE_ROOT}/experimental/lite/examples/verl/verl_mlite" ]] || \
  fail "MLITE_ROOT=${MLITE_ROOT} does not contain experimental/lite/examples/verl/verl_mlite"
(( $# > 0 )) || fail "a command is required"

python_roots=(
  "${UV_WIP_SITE}"
  "${VERL_ROOT}"
  "${MEGATRON_ROOT}"
  "${MLITE_ROOT}/experimental/lite"
  "${MLITE_ROOT}/experimental/lite/examples/verl"
)

if [[ -n "${UV_WIP_KERNEL_SITE:-}" ]]; then
  cutlass_python="${UV_WIP_KERNEL_SITE}/nvidia_cutlass_dsl/python_packages"
  [[ -d "${cutlass_python}" ]] || \
    fail "UV_WIP_KERNEL_SITE=${UV_WIP_KERNEL_SITE} does not contain nvidia_cutlass_dsl/python_packages"
  python_roots+=("${cutlass_python}" "${UV_WIP_KERNEL_SITE}")
fi

export PYTHONPATH="$(IFS=:; printf '%s' "${python_roots[*]}")${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export PATH="/usr/local/cuda/bin:/usr/bin${PATH:+:${PATH}}"

exec "$@"
