#!/bin/bash
#SBATCH -A coreai_devtech_all
#SBATCH -p batch
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --mem=0
#SBATCH -J gdn_ep8_layer0
#SBATCH --time=00:30:00
#SBATCH --output=/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/bayan/code/qwen35_dapo_g2/logs/gdn_ep8_layer0_%j.out
set -euo pipefail

BASE=/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/bayan/code
IMG=$BASE/verl_optimize/verl.vllm023.sqsh
REPO=$BASE/qwen35_dapo_g2/wt-gdn-backward
MCORE=$BASE/qwen35_dapo_g2/mcore-gdn-family
MBRIDGE=$BASE/qwen35_dapo_ab/mbridge_main
CP_SITE=$BASE/mlite-newenv-cache/qwen35-cp-overlay-20260613/site
HF=/lustre/fsw/portfolios/coreai/users/bayan/code/models/Qwen3.5-35B-A3B
OUT=$BASE/qwen35_dapo_g2/diagnosis/task_1_1_15_9/gdn_ep8_layer0_${SLURM_JOB_ID}

COMMON=(
  --hf-path "$HF" --model-name qwen3_5
  --tp 1 --etp 1 --ep 8 --pp 1 --cp 1
  --steps 1 --num-microbatches 1 --seq-len 256 --seed 42
  --truncate-layers 1 --disable-mtp
  --same-data-across-dp --no-optimizer --skip-weight-hash --use-thd
)
STAGE_PROBES='{"mlite":["module.layers.0.linear_attn.o_proj","module.layers.0.mlp_norm","module.layers.0.moe.router","module.layers.0.moe.experts","module.layers.0.moe.shared_expert","module.layers.0.moe","module.layers.0"],"mbridge":["module.module.language_model.decoder.layers.0.self_attention.out_proj","module.module.language_model.decoder.layers.0.pre_mlp_layernorm","module.module.language_model.decoder.layers.0.mlp.router","module.module.language_model.decoder.layers.0.mlp.experts","module.module.language_model.decoder.layers.0.mlp.shared_experts::get_output","module.module.language_model.decoder.layers.0.mlp::routed_experts_compute","module.module.language_model.decoder.layers.0.mlp::combine","module.module.language_model.decoder.layers.0.mlp::postprocess","module.module.language_model.decoder.layers.0.mlp","module.module.language_model.decoder.layers.0"]}'
COMMON+=(--activation-probes-json "$STAGE_PROBES")

mkdir -p "$OUT"
SA=(--container-image="$IMG" --container-mounts=/lustre:/lustre --export=ALL)
ENV=(env PYTHONNOUSERSITE=1
  PYTHONPATH="$REPO/experimental/lite:$REPO:$MCORE:$MBRIDGE:$CP_SITE:/vllm"
  LD_LIBRARY_PATH="$CP_SITE/tvm_ffi/lib:${LD_LIBRARY_PATH:-}"
  MLITE_CORRECTNESS_GRAD_DETAILS=1
  MLITE_CORRECTNESS_NONDETERMINISTIC=1
  MEGATRON_LITE_MOE_PERMUTE_FUSION=0
  MLITE_BENCH_PACKED_SEQ_LENS=17,31,5,64,23,47,69)

for backend in mlite mbridge; do
  BACKEND_ARGS=()
  if [[ "$backend" == mlite ]]; then
    BACKEND_ARGS=(--impl-cfg-json '{"mount_vision_model":false,"recompute":["full"]}')
  fi
  ENTRY="$REPO/../diagnosis/task_1_1_15_9/dispatcher_combine_probe.py"
  srun "${SA[@]}" "${ENV[@]}" env \
    MLITE_CORRECTNESS_ENTRY="$REPO/experimental/lite/examples/bench/correctness.py" \
    torchrun --nproc_per_node 8 "$ENTRY" run \
    --backend "$backend" "${COMMON[@]}" "${BACKEND_ARGS[@]}" \
    --output-json "$OUT/${backend}.json" > "$OUT/${backend}.log" 2>&1
done

srun "${SA[@]}" "${ENV[@]}" python \
  "$REPO/experimental/lite/examples/bench/correctness.py" compare \
  "$OUT/mlite.json" "$OUT/mbridge.json" --output-json "$OUT/compare.json"
echo "GDN_EP8_LAYER0_OUTPUT=$OUT"
