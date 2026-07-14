#!/bin/bash
#SBATCH -A coreai_devtech_all
#SBATCH -p batch
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --mem=0
#SBATCH --time=00:25:00
#SBATCH -J gdn_cp_mode_parity
#SBATCH --output=%x_%j.out
set -euo pipefail

# 8-GPU proxy: Qwen3.5 GatedDeltaNet gdn_cp_mode parity (replicated vs sharded,
# CP off vs CP2/CP4). <=25 min wallclock keeps well under the 4 GPU-h AC cap.
( sleep "${WATCHDOG_SECS:-1200}"; echo "GDN_CP_PARITY_WATCHDOG_SELF_KILL job=$SLURM_JOB_ID"; scancel "$SLURM_JOB_ID" ) &
WD_PID=$!

BASE=${BASE:-/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/bayan/code}
IMG=${IMG:-$BASE/verl_optimize/verl.vllm023.sqsh}
MLITE_REPO=${MLITE_REPO:?submit with --export=ALL,MLITE_REPO=/abs/path/to/worktree}
NODE=${NODE:-$MLITE_REPO/experimental/lite/docs/runs/gdn_cp_mode_parity/run_gdn_cp_mode_parity_node.sh}
NPROC=${NPROC:-8}

submit_ts=$(date -Is)
echo "GDN_CP_PARITY_JOB job=$SLURM_JOB_ID submit=$submit_ts image=$IMG mlite=$MLITE_REPO nproc=$NPROC"

SA=(--container-image="$IMG" --container-mounts=/lustre:/lustre --export=ALL)
srun --nodes=1 --ntasks=1 "${SA[@]}" \
  env MLITE_REPO="$MLITE_REPO" NPROC="$NPROC" bash "$NODE" &
JOB_PID=$!

# 5-min-rule GPU accounting (bayan GPU 铁律).
MEMLOG="gdn_cp_parity_${SLURM_JOB_ID}.memory.csv"
(
  echo "timestamp,index,utilization_gpu,memory_used_mib,memory_free_mib"
  while kill -0 "$JOB_PID" 2>/dev/null; do
    ts=$(date -Is)
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.free \
      --format=csv,noheader,nounits 2>/dev/null | sed "s/^/$ts,/" || true
    sleep 5
  done
) >"$MEMLOG" 2>&1 &
MON_PID=$!

sleep 8
echo "GDN_CP_PARITY_RUNNING job=$SLURM_JOB_ID running=$(date -Is)"
set +e
wait "$JOB_PID"; rc=$?
set -e
kill "$MON_PID" 2>/dev/null || true
kill "$WD_PID" 2>/dev/null || true
echo "GDN_CP_PARITY_JOB_DONE job=$SLURM_JOB_ID rc=$rc"
exit "$rc"
