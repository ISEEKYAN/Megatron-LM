#!/usr/bin/env bash
set -euo pipefail
source /recipe/runtime-cache-env.sh
source /recipe/job-state.sh
: "${SLURM_JOB_ID:?job ID required}"
export RECIPE_STOP_FILE="/test-output/stop-workers-$SLURM_JOB_ID"
export RECIPE_RAY_TEMP_DIR="/var/tmp/${USER}-ray-${SLURM_JOB_ID}"
if [[ "$RECIPE_ROLE" == head ]]; then
 recipe_check_output /test-output "$SLURM_JOB_ID"
fi
mkdir -p "$RECIPE_RAY_TEMP_DIR"
export RAY_worker_register_timeout_seconds=180
RECIPE_C8_MODE=${RECIPE_C8_MODE:---c8-2l}
case "$RECIPE_C8_MODE" in --c8-2l|--c8-4l|--c8-6l|--c8-8l|--c8-9l|--c8-40l) ;; *) exit 2 ;; esac
printf 'RECIPE_C8_MODE=%s\n' "$RECIPE_C8_MODE"
sha256sum /recipe/node.sh /recipe/container-node.sh \
 /recipe/run-c8.py /recipe/ray_ports.py /recipe/runtime-cache-env.sh \
 /recipe/preflight-optimizer-workers.py
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export MLITE_TEST_HARNESS=1 MEGATRON_LITE_REQUIRE_CUDA_TESTS=1
export PYTHONPATH=/opt/src/mlite/experimental/lite/examples/verl:${PYTHONPATH:-}
export VERL_MLITE_HF_CONFIG_MODEL_TYPE=deepseek_v41
RECIPE_PYTHON=/opt/runtime/.venv/bin/python
RECIPE_RAY=/opt/runtime/.venv/bin/ray
export RAY_HEAD_IP="$RECIPE_HEAD_IP"
eval "$("$RECIPE_PYTHON" -S /recipe/ray_ports.py)"
export DS41_RAY_ADDRESS="$RAY_ADDRESS"
RECIPE_RAY_PID=''
cleanup() {
 if [[ "$RECIPE_ROLE" == head ]]; then touch "$RECIPE_STOP_FILE"; fi
 if [[ -n "$RECIPE_RAY_PID" ]]; then
  # Only this invocation's blocking service is terminated; never global ray stop.
  kill -TERM "$RECIPE_RAY_PID" 2>/dev/null || true
  wait "$RECIPE_RAY_PID" 2>/dev/null || true
  mkdir -p "/test-output/ray-internal-$RECIPE_ROLE"
  if test -d "$RECIPE_RAY_TEMP_DIR/session_latest/logs"; then
   cp -aL "$RECIPE_RAY_TEMP_DIR/session_latest/logs/." "/test-output/ray-internal-$RECIPE_ROLE/" || {
    echo RECIPE_RAY_LOG_ARCHIVE_FAILED >&2
    return 1
   }
  fi
 fi
}
trap cleanup EXIT
RECIPE_GPUS=4
RECIPE_CPUS=28
RECIPE_OBJECT_BYTES=8589934592
if [[ "${RECIPE_CPU_COLLECTION:-0}" == 1 ]]; then
 RECIPE_GPUS=0
 RECIPE_CPUS=12
 RECIPE_OBJECT_BYTES=1073741824
fi
RECIPE_RAY_ARGS=(start --node-ip-address="$RECIPE_NODE_IP" --num-gpus="$RECIPE_GPUS" --num-cpus="$RECIPE_CPUS"
 --ray-client-server-port="$RECIPE_RAY_CLIENT_PORT" --object-store-memory="$RECIPE_OBJECT_BYTES" --disable-usage-stats --block)
if [[ "$RECIPE_ROLE" == head ]]; then
 "$RECIPE_PYTHON" -S /recipe/ray_ports.py --check
 RECIPE_RAY_ARGS+=(--head --port="$RECIPE_RAY_GCS_PORT" --temp-dir="$RECIPE_RAY_TEMP_DIR" --include-dashboard=false)
else
 # The head may still be warming its node-local cache.
 "$RECIPE_PYTHON" -S - <<'PY'
import os,socket,time,pathlib
deadline=time.monotonic()+180
heartbeat=time.monotonic()
while True:
 if time.monotonic()>=heartbeat:
  print('RECIPE_WORKER_WAIT_HEAD',os.environ['SLURM_JOB_ID'],time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),flush=True)
  heartbeat=time.monotonic()+30
 if pathlib.Path(os.environ['RECIPE_STOP_FILE']).exists():
  raise SystemExit('Head stopped before worker connection')
 try:
  with socket.create_connection((os.environ['RECIPE_HEAD_IP'],int(os.environ['RECIPE_RAY_GCS_PORT'])),timeout=2): break
 except OSError:
  if time.monotonic()>=deadline: raise
  time.sleep(2)
PY
 RECIPE_RAY_ARGS+=(--address="$DS41_RAY_ADDRESS")
fi
env VERL_MLITE_SKIP_RUNTIME_PATCHES=1 "$RECIPE_PYTHON" -c \
 'import sys; assert "torch" not in sys.modules and "vllm" not in sys.modules; print("RECIPE_RAY_SERVICE_STARTUP_LIGHTWEIGHT_OK", flush=True)'
env -u RAY_ADDRESS VERL_MLITE_SKIP_RUNTIME_PATCHES=1 \
 "$RECIPE_RAY" "${RECIPE_RAY_ARGS[@]}" >"/test-output/ray-$RECIPE_ROLE.log" 2>&1 &
RECIPE_RAY_PID=$!
if [[ "$RECIPE_ROLE" != head ]]; then
 RECIPE_HEARTBEAT=$SECONDS
 while ! test -f "$RECIPE_STOP_FILE"; do
  if (( SECONDS >= RECIPE_HEARTBEAT )); then
   printf 'RECIPE_WORKER_WAIT_HEAD_EXIT job=%s time=' "$SLURM_JOB_ID"
   date -u +%FT%TZ
   RECIPE_HEARTBEAT=$((SECONDS + 30))
  fi
  kill -0 "$RECIPE_RAY_PID"
  sleep 2
 done
 exit 0
fi
RECIPE_READY_DEADLINE=$((SECONDS + 180))
until grep -qF 'Ray runtime started.' /test-output/ray-head.log; do
 if ! kill -0 "$RECIPE_RAY_PID" 2>/dev/null; then
  tail -60 /test-output/ray-head.log
  exit 1
 fi
 if (( SECONDS >= RECIPE_READY_DEADLINE )); then
  echo RECIPE_RAY_HEAD_READINESS_TIMEOUT
  tail -60 /test-output/ray-head.log
  exit 1
 fi
 sleep 2
done
"$RECIPE_PYTHON" - <<'PY'
import os,time,ray
assert os.environ['RAY_AUTH_MODE']=='token' and os.environ['RAY_AUTH_TOKEN']
print('RECIPE_RAY_EXPLICIT_PER_JOB_AUTH_TOKEN_PRESENT',flush=True)
print('RECIPE_RAY_NODE_REGISTRATION_WAIT_UNMODIFIED',30,flush=True)
assert ray._config.worker_register_timeout_seconds()==180
print('RECIPE_RAY_NATIVE_WORKER_REGISTRATION_WAIT',180,flush=True)
ray.init(address=os.environ['DS41_RAY_ADDRESS'])
deadline=time.monotonic()+180
nodes=int(os.environ.get('RECIPE_NNODES','2'))
expected=0 if os.environ.get('RECIPE_CPU_COLLECTION')=='1' else 4*nodes
while ray.cluster_resources().get('GPU',0)!=expected or len([n for n in ray.nodes() if n['Alive']])!=nodes:
 assert time.monotonic()<deadline,ray.cluster_resources()
 time.sleep(2)
print('RECIPE_RAY_CLUSTER_READY',ray.cluster_resources(),flush=True)
if os.environ.get('RECIPE_CPU_COLLECTION')=='1':
 import runpy
 runpy.run_path('/recipe/preflight-optimizer-workers.py',run_name='__main__')
ray.shutdown()
PY
export DS41_C8_ENABLED=1 DS41_C8_INPUTS=/run-artifacts/inputs DS41_C8_GPUS=$((4 * ${RECIPE_NNODES:-2}))
export DS41_C8_OUTPUT=/test-output/c8
export DS41_C8_SHORT_DATA=/data/${DS41_DATA_FILE:-dapo-math-17k-cleaned.parquet}
mkdir -p "$DS41_C8_OUTPUT"
exec_stage=0
RECIPE_ENTRY_ARGS=("$RECIPE_C8_MODE")
if [[ "${RECIPE_CPU_COLLECTION:-0}" == 1 ]]; then
 RECIPE_ENTRY_ARGS+=(--collect-only --runtime-preflight)
fi
"$RECIPE_PYTHON" -u /recipe/run-c8.py "${RECIPE_ENTRY_ARGS[@]}" || exec_stage=$?
if [[ "$exec_stage" == 0 && "${RECIPE_CPU_COLLECTION:-0}" == 1 ]]; then
 "$RECIPE_PYTHON" /recipe/check-cpu-receipts.py --config /run-artifacts/cpu-config \
  --output /test-output --world "${DS41_WORLD:-8}" --pp "${DS41_PP:-1}" || exec_stage=$?
fi
exit "$exec_stage"
