#!/usr/bin/env bash
set -euo pipefail
if [[ -n "${DS41_OVERLAY_SETUP:-}" ]]; then source "$DS41_OVERLAY_SETUP"; fi
RECIPE_ROOT=${1:?root}
RECIPE_OUTPUT=${2:?shared output}
RECIPE_ROLE=${3:?head or worker}
RECIPE_HEAD_IP=${4:?head IP}
printf 'RECIPE_NODE_FIRST_DIAG_UTC='
date -u +%FT%TZ
hostname
if [[ "${RECIPE_CPU_COLLECTION:-0}" == 1 ]]; then
 echo RECIPE_CPU_COLLECTION_NO_GPU
else
 nvidia-smi --query-gpu=uuid,name,driver_version --format=csv,noheader
fi
export ENROOT_DATA_PATH="${DS41_ENROOT_DATA_PATH:-$RECIPE_ROOT/enroot}"
export ENROOT_TEMP_PATH="/var/tmp/${USER}-ds41-enroot/$SLURM_JOB_ID/$(hostname)"
RECIPE_LOCAL_CACHE=/var/tmp/${USER}-ds41-fp4-v1
RECIPE_WARM_CACHE="$RECIPE_ROOT/warm-cache"
mkdir -p "$ENROOT_TEMP_PATH" "$RECIPE_LOCAL_CACHE/tmp" "$RECIPE_WARM_CACHE"
case "$(findmnt -T "$RECIPE_LOCAL_CACHE" -n -o FSTYPE)" in
 lustre|nfs|nfs4|cifs|gpfs) echo RECIPE_RUNTIME_CACHE_NOT_LOCAL; exit 1 ;;
esac
findmnt -T "$RECIPE_LOCAL_CACHE" -o TARGET,SOURCE,FSTYPE
rsync -a --ignore-existing --exclude=locks/ --exclude='*.lock' --exclude=lock \
 --exclude='*.tmp' --exclude=tmp/ --exclude='ray*/' --exclude='wandb*/' \
 "$RECIPE_WARM_CACHE/" "$RECIPE_LOCAL_CACHE/"
export PYTHONPATH=/opt/runtime/.venv/lib/python3.12/site-packages:/opt/src/vllm
export PYTHONDONTWRITEBYTECODE=1
export WANDB_MODE=online VLLM_BATCH_INVARIANT=1
export WANDB_ENTITY=${WANDB_ENTITY:-megatron-core-moe-dev}
export CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST=10.0 MAX_JOBS=8
export RECIPE_ALLOCATED_GPU_COUNT=$((4 * ${RECIPE_NNODES:-2}))
if [[ "${RECIPE_CPU_COLLECTION:-0}" == 1 ]]; then
 export CUDA_VISIBLE_DEVICES=''
 export RECIPE_ALLOCATED_GPU_COUNT=0
else
 export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:?Slurm mask required}"
fi
export NVIDIA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES"
test "${RAY_AUTH_MODE:?Ray auth mode required}" = token
: "${RAY_AUTH_TOKEN:?per-job Ray token required}"
export RECIPE_ROLE RECIPE_HEAD_IP
export RECIPE_NODE_IP=$(getent ahostsv4 "$(hostname)" | awk 'NR==1 {print $1}')
# A reused 6L/8L/9L subset links to shards at its own absolute path.
RECIPE_REUSE=()
if [[ -n "${DS41_VARIANT_REUSE:-}" ]]; then
 RECIPE_REUSE=(--mount "$DS41_VARIANT_REUSE:$DS41_VARIANT_REUSE:none:bind,ro")
fi
set +e
enroot start --root --rw "${RECIPE_REUSE[@]}" \
 --env CUDA_VISIBLE_DEVICES --env NVIDIA_VISIBLE_DEVICES --env PYTHONPATH \
 --env PYTHONDONTWRITEBYTECODE --env DS41_DATA_FILE --env WANDB_API_KEY --env WANDB_MODE --env WANDB_ENTITY \
 --env RAY_AUTH_MODE --env RAY_AUTH_TOKEN \
 --env DS41_PRECISION --env DS41_CUDAGRAPH --env DS41_CUDAGRAPH_MAX_CAPTURE \
 --env DS41_PP --env DS41_PIPELINE_SPLIT_LAYER --env DS41_EP --env DS41_CP --env DS41_WORLD \
 --env VLLM_USE_BREAKABLE_CUDAGRAPH --env VLLM_BATCH_INVARIANT --env CUDA_HOME --env TORCH_CUDA_ARCH_LIST --env MAX_JOBS \
 --env SLURM_JOB_ID --env DS41_PARAMETER_COUNTS --env RECIPE_C8_MODE --env RECIPE_NNODES --env RECIPE_LAYERS --env RECIPE_CPU_COLLECTION --env RECIPE_ALLOCATED_GPU_COUNT --env RECIPE_ROLE --env RECIPE_HEAD_IP --env RECIPE_NODE_IP \
 --mount "$RECIPE_OUTPUT:/test-output:none:bind" \
 --mount "$RECIPE_ROOT:/run-artifacts:none:bind" \
 --mount "$RECIPE_LOCAL_CACHE:/runtime-cache:none:bind" \
 --mount "$RECIPE_LOCAL_CACHE/tmp:/tmp:none:bind" \
 --mount "$DS41_BUNDLE/recipe/hsg:/recipe:none:bind,ro" \
 --mount "${DS41_RELEASE:?release path}:/release:none:bind,ro" \
 --mount "${DS41_DATA_DIR:?data directory}:/data:none:bind,ro" \
 ds41-runtime bash /recipe/container-node.sh
RECIPE_RESULT=$?
set -e
# The container node entry waits for its own Ray service shutdown before returning.
if ! rsync -a --ignore-existing --exclude=locks/ --exclude='*.lock' --exclude=lock \
 --exclude='*.tmp' --exclude=tmp/ --exclude='ray*/' --exclude='wandb*/' \
 "$RECIPE_LOCAL_CACHE/" "$RECIPE_WARM_CACHE/"; then
 echo RECIPE_WARM_CACHE_PERSIST_FAILED
fi
if test -d "$RECIPE_LOCAL_CACHE/wandb"; then
 rsync -a "$RECIPE_LOCAL_CACHE/wandb/" "$RECIPE_OUTPUT/wandb-$(hostname)/" || true
fi
exit "$RECIPE_RESULT"
