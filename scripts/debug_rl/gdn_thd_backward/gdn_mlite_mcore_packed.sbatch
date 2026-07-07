#!/bin/bash
#SBATCH -A coreai_devtech_all
#SBATCH -p batch
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=0
#SBATCH -J gdn_packed_bwd
#SBATCH --time=00:30:00
#SBATCH --output=/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/bayan/code/qwen35_dapo_g2/logs/gdn_packed_bwd_%j.out
set -euo pipefail

BASE=/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/bayan/code
IMG=$BASE/verl_optimize/verl.vllm023.sqsh
REPO=$BASE/qwen35_dapo_g2/wt-gdn-backward
MCORE=$BASE/qwen35_dapo_g2/mcore-gdn-family
MBRIDGE=$BASE/qwen35_dapo_ab/mbridge_main
CP_SITE=$BASE/mlite-newenv-cache/qwen35-cp-overlay-20260613/site
HF=/lustre/fsw/portfolios/coreai/users/bayan/code/models/Qwen3.5-35B-A3B
OUT=$BASE/qwen35_dapo_g2/diagnosis/task_1_1_15_9/gdn_packed_${SLURM_JOB_ID}

COMMON=(
  --hf-path "$HF" --model-name qwen3_5
  --tp 1 --etp 1 --ep 1 --pp 1 --cp 1
  --steps 1 --num-microbatches 1 --seq-len 256 --seed 42
  --truncate-layers 1 --keep-experts 2 --disable-mtp
  --same-data-across-dp --no-optimizer --skip-weight-hash --use-thd
)
STAGE_PROBES='{"mlite":["linear_attn.in_proj","linear_attn::_prepare_qkv","linear_attn::_compute_g_and_beta","linear_attn::_gated_delta_rule","linear_attn::_apply_gated_norm","linear_attn.o_proj"],"mbridge":["self_attention.in_proj","self_attention::_prepare_qkv_for_gated_delta_rule","self_attention::_compute_g_and_beta","self_attention::gated_delta_rule","self_attention::_apply_gated_norm","self_attention.out_proj"]}'
COMMON+=(--activation-probes-json "$STAGE_PROBES")

mkdir -p "$OUT"
SA=(--container-image="$IMG" --container-mounts=/lustre:/lustre --export=ALL)
ENV=(env PYTHONNOUSERSITE=1
  PYTHONPATH="$REPO/experimental/lite:$REPO:$MCORE:$MBRIDGE:$CP_SITE:/vllm"
  LD_LIBRARY_PATH="$CP_SITE/tvm_ffi/lib:${LD_LIBRARY_PATH:-}"
  MLITE_CORRECTNESS_GRAD_DETAILS=1
  MLITE_CORRECTNESS_PARAM_PROBES=conv1d.weight
  MLITE_CORRECTNESS_NONDETERMINISTIC=1
  MLITE_BENCH_PACKED_SEQ_LENS=17,31,5,64,23,47,69)

for backend in mlite mbridge; do
  BACKEND_ARGS=()
  if [[ "$backend" == mlite ]]; then
    BACKEND_ARGS=(--impl-cfg-json '{"mount_vision_model":false}')
  fi
  srun "${SA[@]}" "${ENV[@]}" torchrun --nproc_per_node 1 \
    "$REPO/experimental/lite/examples/bench/correctness.py" run \
    --backend "$backend" "${COMMON[@]}" "${BACKEND_ARGS[@]}" \
    --output-json "$OUT/${backend}.json" > "$OUT/${backend}.log" 2>&1
done

srun "${SA[@]}" "${ENV[@]}" python \
  "$REPO/experimental/lite/examples/bench/correctness.py" compare \
  "$OUT/mlite.json" "$OUT/mbridge.json" --output-json "$OUT/compare.json"
echo "GDN_PACKED_OUTPUT=$OUT"
