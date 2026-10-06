#!/bin/bash
set -euo pipefail
: "${DS41_DATA:?DAPO parquet required}"
: "${DS41_OUTPUT:?output directory required}"
: "${WANDB_API_KEY:?W&B credentials required}"
DS41_RECIPE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$DS41_OUTPUT"
if [[ -n "${DS41_RELEASE:-}" ]]; then
 export DS41_MODEL="${DS41_MODEL:-$DS41_OUTPUT/model}"
 python "$DS41_RECIPE/prepare_prefix.py"
fi
: "${DS41_MODEL:?provide DS41_RELEASE or a prepared release 2-layer prefix directory}"
export WANDB_ENTITY=megatron-core-moe-dev VLLM_BATCH_INVARIANT=1
export VLLM_DEEP_GEMM_PAGED_MQA_USE_VENDORED=1 VLLM_DEEP_GEMM_MEGA_MHC_USE_VENDORED=1
export VERL_MLITE_HF_CONFIG_MODEL_TYPE=deepseek_v41 VERL_ROLLOUT_DISABLE_DEBUG_FILL=1
export MEGATRON_LITE_MOE_PERMUTE_FUSION=0 VERL_FULL_DETERMINISM=1 VERL_DISABLE_FLASH_ATTN_CE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OMEGACONF_MAX_YAML_EXPANDED_NODES=600000
export W4_GRPO_AUDIT=1 W4_GRPO_OUT="$DS41_OUTPUT"
if [[ "${1:-}" == --config-only ]]; then export W4_GRPO_AUDIT=0; fi
export PYTHONPATH="$DS41_RECIPE/audit:$DS41_RECIPE:${PYTHONPATH:-}"
exec python -u "$DS41_RECIPE/command.py" "$@"
