#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH=/opt/runtime/cpu-driver-stub:${LD_LIBRARY_PATH:-}
export DS41_PYTHON=/opt/runtime/.venv/bin/python
RECIPE_LAYERS=${RECIPE_LAYERS:-2}
export DS41_RELEASE=/release DS41_MODEL=/run-artifacts/inputs/${RECIPE_LAYERS}l/model
export DS41_OUTPUT=/run-artifacts/cpu-config DS41_LAYERS=$RECIPE_LAYERS
mkdir -p "$DS41_OUTPUT"
if [[ "$RECIPE_LAYERS" == 6 ]]; then
 # Renumbered release subset instead of a closed prefix (see README).
 export DS41_VARIANT_LAYERS=1,14,15,20,21,24
elif [[ "$RECIPE_LAYERS" == 9 || "$RECIPE_LAYERS" == 8 ]]; then
 # Long-sequence proxy: every 40L layer role incl. the ratio-2 CED layer 19,
 # trained at 17408+128 under EP8xCP2 with full activation recompute.
 # 8L is the fallback without layer 15.
 export DS41_VARIANT_LAYERS=0,1,2,14,15,19,20,21,24 DS41_CP=2 DS41_WORLD=8 DS41_RECOMPUTE=full
 if [[ "$RECIPE_LAYERS" == 8 ]]; then export DS41_VARIANT_LAYERS=0,1,2,14,19,20,21,24; fi
elif [[ "$RECIPE_LAYERS" == 40 ]]; then
 export DS41_EP=64 DS41_WORLD=64 DS41_TRAIN_BATCH=16
fi
export DS41_DATA=/data/${DS41_DATA_FILE:-dapo-math-17k-cleaned.parquet}
export PYTHONPATH=/opt/src/mlite/experimental/lite/examples/verl:${PYTHONPATH:-}
if [[ -n "${DS41_VARIANT_LAYERS:-}" ]]; then
 "$DS41_PYTHON" /opt/src/mlite/experimental/lite/examples/verl/scripts/ds41_w4a8_proxy/prepare_variant.py
else
 "$DS41_PYTHON" /opt/src/mlite/experimental/lite/examples/verl/scripts/ds41_w4a8_proxy/prepare_prefix.py
fi
unset DS41_RELEASE
if [[ "$RECIPE_LAYERS" == 6 || "$RECIPE_LAYERS" == 8 || "$RECIPE_LAYERS" == 9 ]]; then
 # Past the 2048x8 candidate pool, so the two-level Top-K drops blocks.
 "$DS41_PYTHON" /opt/src/mlite/experimental/lite/examples/verl/scripts/ds41_w4a8_proxy/prepare_long_prompts.py \
  --data "$DS41_DATA" --model "$DS41_MODEL" --output /run-artifacts/inputs/c8-candidate-long-prompts.parquet \
  --min-tokens 17408 --limit 16 --repeat-to-limit --repeat-content-to-min
 export DS41_DATA=/run-artifacts/inputs/c8-candidate-long-prompts.parquet DS41_PROMPT_LENGTH=17408
elif [[ "$RECIPE_LAYERS" == 4 ]]; then
 # The 4L C8 crosses the 12288-row indexer boundary with long prompts.
 "$DS41_PYTHON" /opt/src/mlite/experimental/lite/examples/verl/scripts/ds41_w4a8_proxy/prepare_long_prompts.py \
  --data "$DS41_DATA" --model "$DS41_MODEL" --output /run-artifacts/inputs/c8-boundary-long-prompts.parquet \
  --min-tokens 12800 --limit 16 --repeat-to-limit --repeat-content-to-min
 export DS41_DATA=/run-artifacts/inputs/c8-boundary-long-prompts.parquet DS41_PROMPT_LENGTH=12800
else
 export DS41_PROMPT_LENGTH=256
fi
bash /opt/src/mlite/experimental/lite/examples/verl/scripts/ds41_w4a8_proxy/run.sh --config-only
if [[ "${DS41_PP:-1}" != 1 ]]; then
 "$DS41_PYTHON" /recipe/verify-parallel-config.py "$DS41_OUTPUT/resolved.yaml"
fi
