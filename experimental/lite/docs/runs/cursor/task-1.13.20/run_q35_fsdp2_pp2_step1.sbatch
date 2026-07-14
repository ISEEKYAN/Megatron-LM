#!/bin/bash
#SBATCH -A coreai_devtech_all
#SBATCH -p batch
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --mem=0
#SBATCH --time=01:00:00
#SBATCH -J q35_fsdp2_pp2_s1
#SBATCH --output=/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/bayan/code/qwen35_dapo_mfsdp_62295f9b3/logs/%x_%j.out
set -euo pipefail

( sleep "${WATCHDOG_SECS:-2700}"; echo "Q35_PP2_WATCHDOG_SELF_KILL job=$SLURM_JOB_ID after ${WATCHDOG_SECS:-2700}s"; scancel "$SLURM_JOB_ID" ) &
WD_PID=$!

: "${BACKEND:?submit with --export=ALL,BACKEND=mlite}"
BASE=${BASE:-/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/bayan/code}
RUN_DIR=${RUN_DIR:-$BASE/qwen35_dapo_mfsdp_62295f9b3}
SCRIPT_DIR=${SCRIPT_DIR:-$RUN_DIR/task-1.13.20}
IMG=${IMG:-$BASE/verl_optimize/verl.vllm023.sqsh}
NODE=$SCRIPT_DIR/run_q35_fsdp2_pp2_step1_node.sh
nodes=($(scontrol show hostnames "$SLURM_JOB_NODELIST")); NN=${#nodes[@]}
head_node=${nodes[0]}; head_ip=$(getent hosts "$head_node" | awk '{print $1}'); port=$((33000 + SLURM_JOB_ID % 1000))
submit_ts=$(date -Is)
echo "Q35_PP2_JOB job=$SLURM_JOB_ID submit=$submit_ts backend=$BACKEND nodes=${nodes[*]} image=$IMG mlite=${MLITE_REPO:-$RUN_DIR/mlite-57a2064b3}"
SA=(--container-image="$IMG" --container-mounts=/lustre:/lustre --export=ALL)
srun --nodes=1 --ntasks=1 -w "$head_node" "${SA[@]}" bash "$NODE" head "$head_ip:$port" "$BACKEND" "$NN" &
HEAD_PID=$!
MEMORY_LOG="$RUN_DIR/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.memory.csv"
(
  echo "timestamp,index,utilization_gpu,memory_used_mib,memory_free_mib"
  while kill -0 "$HEAD_PID" 2>/dev/null; do
    ts=$(date -Is)
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.free \
      --format=csv,noheader,nounits | sed "s/^/$ts,/"
    sleep 5
  done
) >"$MEMORY_LOG" 2>&1 &
MONITOR_PID=$!
sleep 8
running_ts=$(date -Is)
echo "Q35_PP2_RUNNING job=$SLURM_JOB_ID running=$running_ts first_diag=$running_ts"
set +e
wait "$HEAD_PID"; rc=$?
set -e
kill "$MONITOR_PID" 2>/dev/null || true
wait "$MONITOR_PID" 2>/dev/null || true
kill "$WD_PID" 2>/dev/null || true
echo "Q35_PP2_JOB_DONE job=$SLURM_JOB_ID rc=$rc"
exit "$rc"
