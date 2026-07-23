#!/usr/bin/env bash
#SBATCH --job-name=muon-dapo-reward-ab
#SBATCH --account=coreai_devtech_all
#SBATCH --partition=batch
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=01:30:00
#SBATCH --output=/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/bayan/code/runtime/muon-dapo-reward-ab-30b/slurm-%j.out
set -euo pipefail

B=/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/bayan
CODE_ROOT=${CODE_ROOT:-${B}/code}
RUN_ROOT=${RUN_ROOT:-${CODE_ROOT}/runtime/muon-dapo-reward-ab-30b}
CONTAINER_IMAGE=${CONTAINER_IMAGE:-/lustre/fsw/portfolios/coreai/users/bayan/code/megatron_lite/verl.complete.sqsh}
NODE_SCRIPT=${NODE_SCRIPT:-${RUN_ROOT}/harness/muon_dapo_reward_node.sh}
mkdir -p "${RUN_ROOT}"

( sleep "${WATCHDOG_SECS:-5100}"; echo "MUON_DAPO_WATCHDOG job=${SLURM_JOB_ID}"; scancel "${SLURM_JOB_ID}" ) &
watchdog_pid=$!

nodes=($(scontrol show hostnames "${SLURM_JOB_NODELIST}"))
head_node=${nodes[0]}
head_ip=$(getent hosts "${head_node}" | awk '{print $1}')
ray_port=${RAY_PORT:-6379}
container_args=(--container-image="${CONTAINER_IMAGE}" --container-mounts=/lustre:/lustre --export=ALL)

echo "MUON_DAPO_JOB job=${SLURM_JOB_ID} submit_start=$(date -Is) nodes=${nodes[*]} image=${CONTAINER_IMAGE}"
srun --nodes=1 --ntasks=1 -w "${head_node}" "${container_args[@]}" \
  env MODEL_PATH=/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/bayan/code/models/Qwen3-30B-A3B \
  bash "${NODE_SCRIPT}" head "${head_ip}:${ray_port}" "${#nodes[@]}" &
head_pid=$!

worker_pids=()
sleep 8
for node in "${nodes[@]:1}"; do
  srun --nodes=1 --ntasks=1 -w "${node}" "${container_args[@]}" \
    env MODEL_PATH=/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/bayan/code/models/Qwen3-30B-A3B \
    bash "${NODE_SCRIPT}" worker "${head_ip}:${ray_port}" "${#nodes[@]}" &
  worker_pids+=("$!")
done

diag_log="${RUN_ROOT}/job-${SLURM_JOB_ID}-first-diagnostic.log"
(
  sleep 180
  echo "diagnostic_time=$(date -Is)"
  squeue -j "${SLURM_JOB_ID}" -o "%.18i %.9P %.32j %.8T %.10M %.6D %R"
  for node in "${nodes[@]}"; do
    echo "node=${node}"
    srun --overlap --nodes=1 --ntasks=1 -w "${node}" "${container_args[@]}" \
      bash -c '
        nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
        pgrep -af "ray|python.*main_ppo" | head -40
        if command -v py-spy >/dev/null; then
          for pid in $(pgrep -f "python.*main_ppo" | head -4); do
            echo "py-spy pid=${pid}"
            timeout 15s py-spy dump --pid "${pid}" || true
          done
        else
          echo "py-spy unavailable; /proc stacks follow"
          for pid in $(pgrep -f "python.*main_ppo" | head -4); do
            echo "proc-stack pid=${pid}"
            cat "/proc/${pid}/stack" || true
          done
        fi
      ' || true
  done
) >"${diag_log}" 2>&1 &
diag_pid=$!

set +e
wait "${head_pid}"
rc=$?
set -e
kill "${diag_pid}" 2>/dev/null || true
wait "${diag_pid}" 2>/dev/null || true
kill "${watchdog_pid}" 2>/dev/null || true
for pid in "${worker_pids[@]}"; do kill "${pid}" 2>/dev/null || true; done
for pid in "${worker_pids[@]}"; do wait "${pid}" 2>/dev/null || true; done
echo "MUON_DAPO_JOB_DONE job=${SLURM_JOB_ID} rc=${rc}"
exit "${rc}"
