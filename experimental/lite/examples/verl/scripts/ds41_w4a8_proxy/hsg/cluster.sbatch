#!/usr/bin/env bash
#SBATCH --export=ALL
set -euo pipefail
RECIPE_ROOT=${1:?run directory}
RECIPE_STAGE=${2:?cpu or gpu}
: "${DS41_BUNDLE:?shared release directory}"
export RECIPE_LAYERS=${DS41_LAYERS:-2}
export RECIPE_CPU_COLLECTION=0 RECIPE_C8_MODE=--c8-${RECIPE_LAYERS}l
if [[ "$RECIPE_STAGE" == cpu ]]; then
 export RECIPE_CPU_COLLECTION=1 CUDA_VISIBLE_DEVICES=''
 (cd "$DS41_BUNDLE" && sha256sum -c SHA256SUMS)
 export ENROOT_DATA_PATH="$RECIPE_ROOT/enroot" ENROOT_TEMP_PATH="$RECIPE_ROOT/enroot-tmp"
 mkdir -p "$ENROOT_DATA_PATH" "$ENROOT_TEMP_PATH" "$RECIPE_ROOT/inputs/${RECIPE_LAYERS}l/model"
 mkdir -p "$RECIPE_ROOT/warm-cache"
 if test -d "$DS41_BUNDLE/warm-cache"; then
  rsync -a --ignore-existing "$DS41_BUNDLE/warm-cache/" "$RECIPE_ROOT/warm-cache/"
 fi
 enroot create --name ds41-runtime "$DS41_BUNDLE/runtime.sqsh"
 # CPU-only preparation creates faithful prefix links and resolves the exact config.
 export PYTHONPATH=/opt/runtime/.venv/lib/python3.12/site-packages
 enroot start --root --rw --env DS41_DATA_FILE --env WANDB_API_KEY --env PYTHONPATH --env RECIPE_LAYERS \
  --env DS41_PRECISION --env DS41_CUDAGRAPH --env DS41_CUDAGRAPH_MAX_CAPTURE --env VLLM_USE_BREAKABLE_CUDAGRAPH \
  --mount "$RECIPE_ROOT:/run-artifacts:none:bind" \
  --mount "$DS41_RELEASE:/release:none:bind,ro" \
  --mount "$DS41_DATA_DIR:/data:none:bind,ro" \
  --mount "$DS41_BUNDLE/recipe/hsg:/recipe:none:bind,ro" \
  ds41-runtime bash /recipe/prepare-container.sh
else
 test "$RECIPE_STAGE" = gpu
 grep -q '"passed": true' "$RECIPE_ROOT/logs/cpu/result.json"
 grep -q '"collect_only": true' "$RECIPE_ROOT/logs/cpu/result.json"
fi
export RAY_AUTH_MODE=token
export RAY_AUTH_TOKEN=$(openssl rand -hex 32)
unset RAY_AUTH_TOKEN_PATH
mapfile -t RECIPE_NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
export RECIPE_NNODES=${#RECIPE_NODES[@]}
test "$RECIPE_NNODES" -ge 2
RECIPE_HEAD_IP=$(getent ahostsv4 "${RECIPE_NODES[0]}" | awk 'NR==1 {print $1}')
RECIPE_OUTPUT="$RECIPE_ROOT/logs/$RECIPE_STAGE"
mkdir -p "$RECIPE_OUTPUT"
srun --exact -N1 -n1 -w "${RECIPE_NODES[0]}" \
 bash "$DS41_BUNDLE/recipe/hsg/node.sh" "$RECIPE_ROOT" "$RECIPE_OUTPUT" head "$RECIPE_HEAD_IP" \
 >"$RECIPE_OUTPUT/head.log" 2>&1 &
RECIPE_HEAD_PID=$!
RECIPE_WORKER_PIDS=()
for RECIPE_WORKER in "${RECIPE_NODES[@]:1}"; do
 srun --exact -N1 -n1 -w "$RECIPE_WORKER" \
  bash "$DS41_BUNDLE/recipe/hsg/node.sh" "$RECIPE_ROOT" "$RECIPE_OUTPUT" worker "$RECIPE_HEAD_IP" \
  >"$RECIPE_OUTPUT/worker-$RECIPE_WORKER.log" 2>&1 &
 RECIPE_WORKER_PIDS+=($!)
done
cleanup() {
 touch "$RECIPE_OUTPUT/stop-workers"
 kill -TERM "$RECIPE_HEAD_PID" "${RECIPE_WORKER_PIDS[@]}" 2>/dev/null || true
}
trap cleanup TERM INT
RECIPE_RESULT=0
wait "$RECIPE_HEAD_PID" || RECIPE_RESULT=$?
touch "$RECIPE_OUTPUT/stop-workers"
for RECIPE_WORKER_PID in "${RECIPE_WORKER_PIDS[@]}"; do
 wait "$RECIPE_WORKER_PID" || RECIPE_RESULT=$?
done
exit "$RECIPE_RESULT"
