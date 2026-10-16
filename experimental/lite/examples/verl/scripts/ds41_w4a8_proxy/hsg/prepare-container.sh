#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH=/opt/runtime/cpu-driver-stub:${LD_LIBRARY_PATH:-}
export DS41_PYTHON=/opt/runtime/.venv/bin/python
RECIPE_LAYERS=${RECIPE_LAYERS:-2}
export DS41_RELEASE=/release DS41_MODEL=/run-artifacts/inputs/${RECIPE_LAYERS}l/model
export DS41_OUTPUT=/run-artifacts/cpu-config DS41_LAYERS=$RECIPE_LAYERS
if [[ "$RECIPE_LAYERS" == 6 ]]; then
 # Renumbered release subset instead of a closed prefix (see README).
 export DS41_VARIANT_LAYERS=1,14,15,20,21,24
fi
export DS41_DATA=/data/${DS41_DATA_FILE:-dapo-math-17k-cleaned.parquet}
export PYTHONPATH=/opt/src/mlite/experimental/lite/examples/verl:${PYTHONPATH:-}
bash /opt/src/mlite/experimental/lite/examples/verl/scripts/ds41_w4a8_proxy/run.sh --config-only
if [[ "$RECIPE_LAYERS" == 6 ]]; then
 # Past the 2048x8 candidate pool, so the two-level Top-K drops blocks.
 "$DS41_PYTHON" /opt/src/mlite/experimental/lite/examples/verl/scripts/ds41_w4a8_proxy/prepare_long_prompts.py \
  --data "$DS41_DATA" --model "$DS41_MODEL" --output /run-artifacts/inputs/c8-candidate-long-prompts.parquet \
  --min-tokens 17408 --limit 16 --repeat-to-limit --repeat-content-to-min
fi
