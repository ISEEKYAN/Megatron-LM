#!/bin/bash
#SBATCH -A coreai_devtech_all
#SBATCH -p batch
#SBATCH --nodes=4
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --mem=0
#SBATCH -J q35_gdn_family
#SBATCH --time=02:00:00
#SBATCH --output=/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/bayan/code/qwen35_dapo_g2/logs/mlite_gdn_family_%j.out
set -euo pipefail

BASE=/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/bayan/code
IMG=$BASE/verl_optimize/verl.vllm023.sqsh
NODE=$BASE/qwen35_dapo_g2/diagnosis/task_1_1_15_9/mlite_family_node.sh
CP="${CP:-1}"
EP="${EP:-8}"
export USE_FUSED_KERNELS=True

nodes=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
NN=${#nodes[@]}
head_node=${nodes[0]}
head_ip=$(getent hosts "$head_node" | awk '{print $1}')
port=6379
echo "JOB $SLURM_JOB_ID nodes=${nodes[*]} head=$head_node CP=$CP EP=$EP fused=$USE_FUSED_KERNELS"
SA=(--container-image="$IMG" --container-mounts=/lustre:/lustre --export=ALL)
srun --nodes=1 --ntasks=1 -w "$head_node" "${SA[@]}" bash "$NODE" head "$head_ip:$port" "$CP" "$EP" "$NN" &
HEAD_PID=$!
sleep 8
for node in "${nodes[@]:1}"; do
  srun --nodes=1 --ntasks=1 -w "$node" "${SA[@]}" bash "$NODE" worker "$head_ip:$port" "$CP" "$EP" "$NN" &
done
wait "$HEAD_PID"
echo "head(training) finished; cancelling job to release ray workers"
scancel "$SLURM_JOB_ID"
