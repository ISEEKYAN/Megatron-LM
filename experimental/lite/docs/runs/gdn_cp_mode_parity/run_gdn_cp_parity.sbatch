#!/bin/bash
#SBATCH --account=coreai_devtech_all
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=00:40:00
#SBATCH --job-name=gdn-cpmode-parity-1171

# Qwen3.5 GatedDeltaNet CP-mode precision parity proxy.
# Reuses the proven verl.vllm023 + mlite env from the DS4 CSA-CP parity workspace
# (ds4-csacp-parity-eaa5b486d/parity_v4b.sbatch, job 13905677 COMPLETED 0:0),
# stripped of the CSA-only cutedsl/cudnn/mcore overlays which GDN does not need.

set -x
B=/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/bayan
R=$B/code/runtime/gdn-cpmode-parity-1171
BASE_IMAGE=/home/bayan/code/verl_optimize/verl.vllm023.sqsh

MLITE_SRC=$R/mlite
SCRIPT=$R/gdn_cp_mode_parity.py

# fla + tilelang + tvm_ffi are NOT in verl.vllm023 base; they ship in the proven
# qwen3.5 CP overlay (K-0123; docs/runs/cursor/task-1.13.20 launcher). Add its site
# to PYTHONPATH and its tvm_ffi/lib to LD_LIBRARY_PATH inside the container.
CP_SITE=$B/code/mlite-newenv-cache/qwen35-cp-overlay-20260613/site

# megatron/ is a namespace package: experimental/lite provides megatron.lite while
# the container image provides megatron.core -> both resolve under one `megatron`.
PYP=$CP_SITE:$MLITE_SRC/experimental/lite:$MLITE_SRC

mkdir -p $R/home $R/caches/triton $R/caches/inductor $R/caches/xdg $R/caches/tilelang

srun --container-image=$BASE_IMAGE --container-mounts=/lustre:/lustre \
     --export=ALL,\
CUDA_HOME=/usr/local/cuda,\
CUDA_PATH=/usr/local/cuda,\
HOME=$R/home,\
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,\
CUDA_DEVICE_MAX_CONNECTIONS=1,\
PYTHONNOUSERSITE=1,\
XDG_CACHE_HOME=$R/caches/xdg,\
TRITON_CACHE_DIR=$R/caches/triton,\
TORCHINDUCTOR_CACHE_DIR=$R/caches/inductor,\
TILELANG_CACHE_DIR=$R/caches/tilelang,\
CP_SITE=$CP_SITE,\
PYTHONPATH=$PYP,\
TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
     bash -c "export LD_LIBRARY_PATH=$CP_SITE/tvm_ffi/lib:\$LD_LIBRARY_PATH && cd $R && \
       python -c 'import torch, fla; from megatron.lite.primitive.modules.gated_delta_net import GatedDeltaNet; print(\"IMPORTS_OK\")' && \
       echo '=== CP2 ===' && \
       torchrun --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=29531 $SCRIPT && \
       echo '=== CP4 ===' && \
       torchrun --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=29532 $SCRIPT && \
       echo '=== CP8 128k (variable-length packed, tot=131072) ===' && \
       GDN_PARITY_128K=1 torchrun --nproc_per_node=8 --master_addr=127.0.0.1 --master_port=29533 $SCRIPT"

echo "SBATCH_BODY_DONE rc=$?"
