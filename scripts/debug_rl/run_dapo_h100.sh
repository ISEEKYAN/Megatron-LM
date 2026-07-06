#!/usr/bin/env bash
# TASK-1.1.15 G0b: DAPO/GRPO Qwen3.5-35B-A3B 复现脚本,cw H100 移植版。
# 源: /home/bayan/code/llmrl/debug_rl/script (H20/ByteDance Collo 环境)。
# 与源脚本的刻意差异(其余超参逐字段保持一致):
#   1. BACKEND=mlite|megatron 双变体开关(源脚本 Megatron 基线数组已被删,此处恢复)。
#   2. calculate_log_probs / rollout_correction 无条件透传——源脚本把它藏在
#      using_rs=False 死分支里,导致实际 run 中 rollout.calculate_log_probs 未生效
#      (与历史 F3 坑 old_log_prob=None 同型,见 dead_ends/mlite-env-setup)。
#   3. 路径全部改为 cw lustre,由环境变量注入;字节路径(/mnt/bn,/opt/tiger)移除。
# 用法(在 ray head 节点、vllm023 容器内):
#   BACKEND=mlite  bash run_dapo_h100.sh
#   BACKEND=megatron bash run_dapo_h100.sh
# DRY_RUN=1 只打印 resolved 命令,用于 G0b 的 config diff。

set -xeuo pipefail

export CUDA_DEVICE_MAX_CONNECTIONS=1
export VLLM_USE_V1=1
export VLLM_ALLREDUCE_USE_SYMM_MEM=0
export NCCL_NVLS_ENABLE=${NCCL_NVLS_ENABLE:-0}
export VLLM_WORKER_MULTIPROC_METHOD=${VLLM_WORKER_MULTIPROC_METHOD:-spawn}
export RAY_memory_monitor_refresh_ms=0
export PYTHONNOUSERSITE=1
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES || true

########################### cw paths (G0a 确认后填死) ###########################
LUSTRE_USER=${LUSTRE_USER:-/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/bayan}
MLITE_ROOT=${MLITE_ROOT:-${LUSTRE_USER}/code/megatron_lite/Megatron-LM}
VERL_SRC=${VERL_SRC:-${LUSTRE_USER}/code/megatron_lite/verl}
MODEL_PATH=${MODEL_PATH:-${LUSTRE_USER}/models/Qwen3.5-35B-A3B}   # TODO G0a: cw 实际路径
TRAIN_FILE=${TRAIN_FILE:-${LUSTRE_USER}/data/dapo-math/dapo-math-17k.parquet} # TODO G0a
TEST_FILE=${TEST_FILE:-${LUSTRE_USER}/data/dapo-math/aime-2024.parquet}       # TODO G0a
CKPTS_ROOT=${CKPTS_ROOT:-${LUSTRE_USER}/ckpts}

########################### Quick Config (与源脚本一致) ###########################
BACKEND=${BACKEND:?set BACKEND=mlite|megatron}
project_name='GRPO_Qwen3.5-35B-A3B'
exp_name="H100_cw_${BACKEND}"
adv_estimator=grpo
RAY_ADDRESS=${RAY_ADDRESS:-"http://localhost:8265"}
WORKING_DIR=${WORKING_DIR:-"${PWD}"}
RUNTIME_ENV=${RUNTIME_ENV:-"${VERL_SRC}/verl/trainer/runtime_env.yaml"}
NNODES=${NNODES:-4}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}

train_batch_size=${train_batch_size:-256}
ppo_mini_batch_size=${ppo_mini_batch_size:-32}
train_prompt_bsz=${train_batch_size}

response_len=${response_len:-20}
max_prompt_length=$((1024 * 2))
max_response_length=$((1024 * response_len))
enable_overlong_buffer=True
actor_ppo_max_token_len=$((max_prompt_length + max_response_length))
infer_ppo_max_token_len=$((max_prompt_length + max_response_length))
filter_overlong_prompts_workers=64

DATA=(
    data.train_files=${TRAIN_FILE}
    data.val_files=${TEST_FILE}
    data.train_batch_size=${train_prompt_bsz}
    data.max_prompt_length=${max_prompt_length}
    data.max_response_length=${max_response_length}
    data.truncation='error'
    data.filter_overlong_prompts=${enable_overlong_buffer}
    data.filter_overlong_prompts_workers=${filter_overlong_prompts_workers}
)

use_remove_padding=${use_remove_padding:-True}
MODEL=(
    actor_rollout_ref.model.path=${MODEL_PATH}
    actor_rollout_ref.model.trust_remote_code=True
    actor_rollout_ref.model.use_remove_padding=${use_remove_padding}
    actor_rollout_ref.model.use_fused_kernels=True
)
use_dynamic_bsz=True
COMMON_ACTOR=(
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size}
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len}
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz}
    actor_rollout_ref.actor.use_kl_loss=False
    actor_rollout_ref.actor.kl_loss_coef=0.01
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.entropy_coeff=0
)

rollout_name="vllm"
gen_tp=${gen_tp:-2}
gpu_memory_utilization=${gpu_memory_utilization:-0.7}
n_resp_per_prompt=${n_resp_per_prompt:-8}
enforce_eager=${enforce_eager:-False}
rollout_dtype=${rollout_dtype:-bfloat16}
dcp_size=${dcp_size:-1}
max_num_seqs=${max_num_seqs:-1024}
update_weights_bucket_megabytes=${update_weights_bucket_megabytes:-1024}
ROLLOUT=(
    actor_rollout_ref.rollout.name=${rollout_name}
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp}
    actor_rollout_ref.rollout.gpu_memory_utilization=${gpu_memory_utilization}
    actor_rollout_ref.rollout.n=${n_resp_per_prompt}
    actor_rollout_ref.rollout.mode=async
    actor_rollout_ref.rollout.enforce_eager=${enforce_eager}
    actor_rollout_ref.rollout.dtype=${rollout_dtype}
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz}
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len}
    actor_rollout_ref.rollout.max_num_seqs=${max_num_seqs}
    +actor_rollout_ref.rollout.engine_kwargs.vllm.decode_context_parallel_size=${dcp_size}
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=${update_weights_bucket_megabytes}
)

COMMON_REF=(
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz}
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len}
)

ALGORITHM=(
    algorithm.adv_estimator=${adv_estimator}
    algorithm.use_kl_in_reward=False
)

########################### 并行布局(两后端同 mesh) ###########################
train_tp=${train_tp:-1}
train_pp=${train_pp:-1}
train_ep=${train_ep:-8}
train_cp=${train_cp:-4}
train_etp=${train_etp:-1}
OPTIMIZER=${OPTIMIZER:-fsdp2}
ALL_OFFLOAD=${ALL_OFFLOAD:-True}

MLITE=(
    actor@actor_rollout_ref.actor=mlite_actor
    actor_rollout_ref.model.external_lib=verl_mlite.engine.mlite_engine
    actor_rollout_ref.actor.engine.tp=${train_tp}
    actor_rollout_ref.actor.engine.pp=${train_pp}
    actor_rollout_ref.actor.engine.vpp=1
    actor_rollout_ref.actor.engine.ep=${train_ep}
    actor_rollout_ref.actor.engine.cp=${train_cp}
    actor_rollout_ref.actor.engine.etp=${train_etp}
    actor_rollout_ref.actor.engine.param_offload=${ALL_OFFLOAD}
    actor_rollout_ref.actor.engine.optimizer_offload=${ALL_OFFLOAD}
    actor_rollout_ref.actor.engine.grad_offload=${ALL_OFFLOAD}
    actor_rollout_ref.actor.engine.attention_backend_override=flash
    actor_rollout_ref.actor.engine.impl_cfg.use_thd=True
    +actor_rollout_ref.actor.engine.impl_cfg.optimizer=${OPTIMIZER}
    +actor_rollout_ref.actor.engine.impl_cfg.recompute=[full]
    +actor_rollout_ref.actor.optim.override_optimizer_config.offload_fraction=1.0
)

# Megatron 基线(源脚本已删,按 verl ppo_megatron_trainer.yaml 字段恢复,mesh 与 mlite 一致)
MEGATRON_ACTOR=(
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${train_tp}
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${train_pp}
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=${train_ep}
    actor_rollout_ref.actor.megatron.context_parallel_size=${train_cp}
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=${train_etp}
    actor_rollout_ref.actor.megatron.param_offload=${ALL_OFFLOAD}
    actor_rollout_ref.actor.megatron.optimizer_offload=${ALL_OFFLOAD}
    actor_rollout_ref.actor.megatron.grad_offload=${ALL_OFFLOAD}
    actor_rollout_ref.actor.megatron.use_dist_checkpointing=False
)
MEGATRON_REF=(
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${train_tp}
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=${train_pp}
    actor_rollout_ref.ref.megatron.expert_model_parallel_size=${train_ep}
    actor_rollout_ref.ref.megatron.context_parallel_size=${train_cp}
    actor_rollout_ref.ref.megatron.expert_tensor_parallel_size=${train_etp}
    actor_rollout_ref.ref.megatron.param_offload=True
)

if [ "${BACKEND}" = mlite ]; then
    ACTOR=( "${MLITE[@]}" "${COMMON_ACTOR[@]}" )
    REF=( "${COMMON_REF[@]}" )
    EXTRA=( hydra.searchpath=[pkg://verl_mlite.config] )
    exp_name=${exp_name}_tp${train_tp}pp${train_pp}cp${train_cp}ep${train_ep}etp${train_etp}_opt-${OPTIMIZER}
else
    ACTOR=( "${MEGATRON_ACTOR[@]}" "${COMMON_ACTOR[@]}" )
    REF=( "${MEGATRON_REF[@]}" "${COMMON_REF[@]}" )
    EXTRA=( )
    exp_name=${exp_name}_tp${train_tp}pp${train_pp}cp${train_cp}ep${train_ep}etp${train_etp}
fi
config_name=ppo_megatron_trainer.yaml
exp_name=${exp_name}_${rollout_name}_tp${gen_tp}_bs${train_batch_size}_${response_len}K

CKPTS_DIR=${CKPTS_DIR:-"${CKPTS_ROOT}/${project_name}/${exp_name}"}
save_freq=${save_freq:--1}
test_freq=${test_freq:-5}
total_steps=${total_steps:-}   # G2 短步验证时设 10

VLLM_ENGINE_KWARGS=(
    +actor_rollout_ref.rollout.engine_kwargs.vllm.enable_expert_parallel=False
    +actor_rollout_ref.rollout.engine_kwargs.vllm.enable_eplb=False
    +actor_rollout_ref.rollout.engine_kwargs.vllm.enable_elastic_ep=False
    +actor_rollout_ref.rollout.engine_kwargs.vllm.enable_ep_weight_filter=False
    +actor_rollout_ref.rollout.engine_kwargs.vllm.reasoning_parser=qwen3
    +actor_rollout_ref.rollout.engine_kwargs.vllm.gdn_prefill_backend=flashinfer
)

# 无条件透传(修复点2)。bypass_mode=True: rollout logprob 直接作 old_log_prob。
RS_CONFIG=(
    algorithm.rollout_correction.rollout_is=null
    algorithm.rollout_correction.rollout_rs=null
    algorithm.rollout_correction.rollout_is_threshold=2.0
    algorithm.rollout_correction.rollout_rs_threshold="0.999_1.001"
    actor_rollout_ref.rollout.calculate_log_probs=True
    algorithm.rollout_correction.bypass_mode=True
)

TRAINER=(
    trainer.critic_warmup=0
    trainer.logger='["console","wandb"]'
    trainer.project_name=${project_name}
    trainer.experiment_name=${exp_name}
    trainer.n_gpus_per_node=${NGPUS_PER_NODE}
    trainer.nnodes=${NNODES}
    trainer.save_freq=${save_freq}
    trainer.val_before_train=True
    trainer.test_freq=${test_freq}
    trainer.total_epochs=15
)
if [ -n "${total_steps}" ]; then
    TRAINER+=( trainer.total_training_steps=${total_steps} )
fi

CMD=( python3 -m verl.trainer.main_ppo
    "${EXTRA[@]}"
    "${DATA[@]}"
    "${ALGORITHM[@]}"
    "${MODEL[@]}"
    "${ROLLOUT[@]}"
    "${ACTOR[@]}"
    "${REF[@]}"
    "${RS_CONFIG[@]}"
    "${TRAINER[@]}"
    "${VLLM_ENGINE_KWARGS[@]}"
    trainer.default_local_dir="${CKPTS_DIR}"
)

if [ "${DRY_RUN:-0}" = 1 ]; then
    printf '%s\n' "${CMD[@]}" "$@"
    exit 0
fi

ray job submit --no-wait --runtime-env="${RUNTIME_ENV}" \
    --address "${RAY_ADDRESS}" \
    --working-dir "${WORKING_DIR}" \
    -- "${CMD[@]}" "$@"
