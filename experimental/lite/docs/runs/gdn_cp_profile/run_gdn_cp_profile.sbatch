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
#SBATCH --job-name=gdn-cp-profile-1174

# Qwen3.5 GatedDeltaNet CP-mode SEGMENTED profiler (chunkwise vs headwise).
# Same env as the parity/peakmem harnesses (run_gdn_cp_peakmem_bench.sbatch):
# verl.vllm023 base + qwen3.5 CP overlay (fla/tilelang/tvm_ffi, K-0123).
#
# Times the GDN forward segment-by-segment (a2a cp2hp/hp2cp, chunkwise ctx_build
# ring, reshuffle, chunk_gated_delta_rule recurrence, conv) over two regimes:
#   - packed_multi: THD packed batch of many short variable-length sequences
#     (the DAPO packing regime where per-layer plumbing is suspected to dominate).
#   - single_long : one long sequence (the 128k-style regime the peakmem bench used).
# Also emits a torch.profiler fwd+bwd device-time breakdown (comm/recurrence/conv/other).
# Answer target: locate WHICH segment makes chunkwise slower than replicated/headwise
# in the DAPO regime (update_actor chunkwise 159 vs replicated 111) vs the 128k regime.

set -x
B=/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/bayan
R=$B/code/runtime/gdn-cp-profile-1174
BASE_IMAGE=/home/bayan/code/verl_optimize/verl.vllm023.sqsh

MLITE_SRC=$R/mlite
# gdn_cp_profile.py imports its sibling gdn_cp_mode_parity.py (helpers), so BOTH
# must be staged flat next to each other in $R (the profiler inserts its own dir
# onto sys.path). Stage step (run once before sbatch):
#   mkdir -p $R && rsync -a --delete <worktree>/ $R/mlite/
#   cp $R/mlite/experimental/lite/tests/smoke/primitive/gdn_cp_mode_parity.py $R/
#   cp $R/mlite/experimental/lite/tests/smoke/primitive/gdn_cp_profile.py     $R/
SCRIPT=$R/gdn_cp_profile.py

# fla + tilelang + tvm_ffi are NOT in verl.vllm023 base; they ship in the proven
# qwen3.5 CP overlay (K-0123). Add its site to PYTHONPATH, its tvm_ffi/lib to
# LD_LIBRARY_PATH inside the container.
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
       echo '=== CP8 segmented profile: modes x regimes ===' && \
       torchrun --nproc_per_node=8 --master_addr=127.0.0.1 --master_port=29542 $SCRIPT"

echo "SBATCH_BODY_DONE rc=$?"
