#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH=/opt/runtime/cpu-driver-stub:${LD_LIBRARY_PATH:-}
export DS41_PYTHON=/opt/runtime/.venv/bin/python
export DS41_RELEASE=/release DS41_MODEL=/run-artifacts/inputs/2l/model
export DS41_OUTPUT=/run-artifacts/cpu-config DS41_LAYERS=2
export DS41_DATA=/data/${DS41_DATA_FILE:-dapo-math-17k-cleaned.parquet}
export PYTHONPATH=/opt/src/mlite/experimental/lite/examples/verl:${PYTHONPATH:-}
bash /opt/src/mlite/experimental/lite/examples/verl/scripts/ds41_w4a8_proxy/run.sh --config-only
