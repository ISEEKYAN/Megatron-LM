# VERL Megatron Lite Example

This directory contains a runnable VERL external engine integration for
Megatron Lite plus Qwen3.5-35B-A3B SFT and GRPO launch scripts.

The Python package is `verl_mlite`. It registers VERL's language-model engine
backend as `mlite`, while Megatron Lite model implementations still use
`impl=lite`.

## Layout

- `verl_mlite/engine/mlite_engine.py`: VERL `BaseEngine` implementation backed
  by `megatron.lite.runtime`.
- `verl_mlite/config/engine/mlite.yaml`: Hydra engine config for
  `engine=mlite`.
- `scripts/run_qwen3moe_sft.sh`: Qwen MoE SFT launcher using
  `verl.trainer.sft_trainer`.
- `scripts/run_qwen3moe_gsm8k_sft.sh`: GSM8K wrapper around the SFT launcher.
- `scripts/run_qwen3moe_gsm8k_grpo.sh`: GSM8K GRPO launcher with MLite actor
  training and a standard VERL rollout backend.

## Prerequisites

Install or expose these packages before running:

- VERL with the new engine worker path.
  See [`REQUIRED_VERL.txt`](REQUIRED_VERL.txt) for the reference upstream
  source pin (commit).
- Megatron-LM from this repository, or another source tree via
  `MEGATRON_ROOT=/path/to/Megatron-LM`.
- Megatron Lite from this repository. The script automatically adds
  `experimental/lite` to `PYTHONPATH`.
- The examples directory is also added to `PYTHONPATH` and loads a local
  compatibility hook for known VERL/vLLM/Transformers dependency gaps.

Optional source-tree override:

```bash
export VERL_ROOT=/path/to/verl
export MEGATRON_ROOT=/path/to/Megatron-LM
```

## SFT

The SFT script expects VERL messages-format parquet input.

```bash
export MODEL_PATH=/path/to/qwen3.5-35b-a3b-hf
export TRAIN_FILES=/path/to/train.parquet
export VAL_FILES=/path/to/val.parquet

bash experimental/lite/examples/verl/scripts/run_qwen3moe_sft.sh
```

Useful knobs:

- `TP_SIZE`, `PP_SIZE`, `VPP_SIZE`, `CP_SIZE`, `EP_SIZE`, `ETP_SIZE`
- `TOTAL_STEPS`, `TOTAL_EPOCHS`, `TRAIN_BATCH_SIZE`, `MICRO_BATCH_SIZE`
- `MAX_TOKENS_PER_GPU`, `MAX_LENGTH`, `MESSAGES_KEY`
- `PARAM_OFFLOAD`, `OPTIMIZER_OFFLOAD`, `GRAD_OFFLOAD`
- `MLITE_MODEL_NAME=auto`, `MLITE_IMPL=lite`
- `ATTENTION_BACKEND=flash`
- `DRY_RUN=1` to print the resolved `torchrun` command without launching

FSDP2 supports two offload modes. `PARAM_OFFLOAD=True` and
`OPTIMIZER_OFFLOAD=True` move model parameters and optimizer state between CPU
and GPU when VERL switches execution contexts. `OPTIMIZER_OFFLOAD=True` also
sets `optim.override_optimizer_config.offload_fraction=1.0` by default, which
keeps FSDP2 optimizer update state on CPU during forward/backward to reduce GPU
memory pressure.

Example dry run:

```bash
MODEL_PATH=/path/to/qwen3.5-35b-a3b-hf \
TRAIN_FILES=/path/to/train.parquet \
DRY_RUN=1 \
bash experimental/lite/examples/verl/scripts/run_qwen3moe_sft.sh
```

By default, logs, command snapshots, JSONL logger output, and checkpoints are
written under `experimental/lite/examples/verl/outputs/qwen3moe_sft`. Override
`OUTPUT_ROOT`, `LOG_FILE`, `JSONL_FILE`, `CMD_FILE`, or `CKPT_DIR` to redirect
artifacts.

For local dry runs, prefer a temporary output directory if you do not want
command snapshots under the source tree:

```bash
OUTPUT_ROOT="$(mktemp -d)" \
MODEL_PATH=/path/to/qwen3.5-35b-a3b-hf \
TRAIN_FILES=/path/to/train.parquet \
DRY_RUN=1 \
bash experimental/lite/examples/verl/scripts/run_qwen3moe_sft.sh
```

## GSM8K SFT

Build messages-format GSM8K parquet files with VERL's SFT preprocessor:

```bash
python3 /path/to/verl/examples/data_preprocess/gsm8k_multiturn_sft.py \
  --local_save_dir ~/data/gsm8k_sft
```

Run the MLite GSM8K SFT wrapper:

```bash
MODEL_PATH=Qwen/Qwen3.5-35B-A3B \
DRY_RUN=1 \
bash experimental/lite/examples/verl/scripts/run_qwen3moe_gsm8k_sft.sh
```

The wrapper defaults to `Qwen/Qwen3.5-35B-A3B`,
`~/data/gsm8k_sft/train.parquet`, and
`~/data/gsm8k_sft/test.parquet`, then delegates to
`scripts/run_qwen3moe_sft.sh`. Override `DATASET_DIR`, `TRAIN_FILES`, or
`VAL_FILES` to use another location.

By default, GSM8K SFT artifacts are written under
`experimental/lite/examples/verl/outputs/qwen35_gsm8k_sft`.

## GSM8K GRPO

Build RL-format GSM8K parquet files with VERL's GRPO/PPO preprocessor:

```bash
python3 /path/to/verl/examples/data_preprocess/gsm8k.py \
  --local_save_dir ~/data/gsm8k
```

Run GRPO with the MLite actor and vLLM rollout:

```bash
MODEL_PATH=Qwen/Qwen3.5-35B-A3B \
DRY_RUN=1 \
bash experimental/lite/examples/verl/scripts/run_qwen3moe_gsm8k_grpo.sh
```

Useful GRPO knobs:

- `TRAIN_BATCH_SIZE`, `PPO_MINI_BATCH_SIZE`,
  `ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU`
- `MAX_PROMPT_LENGTH`, `MAX_RESPONSE_LENGTH`, `PPO_MAX_TOKEN_LEN_PER_GPU`
- `ROLLOUT_N`, `ROLLOUT_TP`, `ROLLOUT_GPU_MEMORY_UTILIZATION`
- `ROLLOUT_MODE=async`, `ROLLOUT_MAX_MODEL_LEN`, `ROLLOUT_MAX_NUM_BATCHED_TOKENS`
- `ROLLOUT_LIMIT_IMAGES=0`, `ROLLOUT_LIMIT_VIDEOS=0` keep the vLLM rollout
  backend in text-only mode for GSM8K by default.
- `ACTOR_TP`, `ACTOR_PP`, `ACTOR_VPP`, `ACTOR_CP`, `ACTOR_EP`, `ACTOR_ETP`
- `PARAM_OFFLOAD`, `OPTIMIZER_OFFLOAD`, `GRAD_OFFLOAD`
- `INFER_BACKEND=vllm`
- `POLICY_LOSS_MODE=vanilla` and `LOSS_AGG_MODE=seq-mean-token-sum-norm`
  select the pure GRPO baseline policy loss and aggregation mode.

The GRPO launcher keeps the reference policy disabled by default
(`algorithm.use_kl_in_reward=False`, `actor_rollout_ref.actor.use_kl_loss=False`)
so the example exercises the current MLite actor path without expanding scope
to a separate reference model. On latest verl, both the v0 and V1 trainer paths
route `actor@actor_rollout_ref.actor=mlite_actor` to the unified engine workers,
so the MLite actor is wired up correctly without any extra worker-path knob.

By default, GSM8K GRPO artifacts are written under
`experimental/lite/examples/verl/outputs/qwen35_gsm8k_grpo`.

## DeepSeek V4 GRPO on Slurm

`scripts/run_deepseek_v4_gsm8k_grpo.sh` specializes the generic GRPO launcher
for the native DeepSeek V4 actor. It fixes the actor to FSDP2, fused DSA, THD,
MTP training, full activation recomputation, and checkpoint-format block-FP8
resync. The colocated vLLM workers use a dummy cold load with pure-FP8 Hugging
Face overrides, then receive the actor's serialized block-FP8 stream through
`VllmCheckpointWorkerExtension`; veRL's online FP8 quantizer remains disabled.

`slurm/run_ds4_gsm8k_grpo.sbatch` builds a multi-node Ray cluster and runs two
phases against one checkpoint directory. Phase one saves a bounded checkpoint;
phase two uses `resume_mode=auto` and must reach the requested final step. The
job then validates continuous JSONL steps, finite reward and policy loss,
positive actor gradient norm, reward variation, and positive per-GPU token
throughput before printing `DS4_GRPO_RUN_COMPLETE`.

The Slurm script creates deterministic arithmetic prompts in the canonical
GSM8K parquet schema. This is a network-independent mechanism smoke dataset,
not a claim to reproduce the public GSM8K benchmark distribution. Supply
separate `TRAIN_FILES` and `VAL_FILES` when using the model wrapper directly
for a benchmark run.

The checked-in Slurm defaults describe the 128-GPU PP4/EP8/CP4 target. A
64-GPU staircase run can override the allocation and topology at submission:

```bash
sbatch --nodes=8 --time=06:00:00 \
  --export=ALL,ACTOR_PP=2,ACTOR_EP=8,ACTOR_CP=2,ROLLOUT_TP=16,PHASE1_STEPS=3,TOTAL_STEPS=6 \
  experimental/lite/examples/verl/slurm/run_ds4_gsm8k_grpo.sbatch
```

The remaining required environment variables are fail-closed path contracts:
`BASE_IMAGE`, `MLITE_SRC`, `MLITE_COMMIT`, `MEGATRON_ROOT`, `VERL_ROOT`,
`MLITE_SM90_SITE`, `DS4_VLLM_SITE`, `DS4_VLLM_SHIM`, `CHECKPOINT_DIR`, and a
fresh `RUN_ROOT`.

### Serialized checkpoint weight resync

Quantized inference models must receive weights in the serialized format
declared by their rollout checkpoint configuration. Set the MLite actor engine's
`resync_format=vllm_checkpoint` and select
`verl_mlite.rollout.verl_worker.VllmCheckpointWorkerExtension` through vLLM's
`worker_extension_cls` engine argument. The extension streams all IPC buckets
through one vLLM layerwise reload lifecycle. It does not call veRL's online FP8
quantizer, so `VERL_VLLM_FP8_QUANT_ENABLED` must be unset or `0`.

The generic engine only forwards the format contract. Model adapters own
checkpoint naming and serialization. The DeepSeek-V4 adapter emits 128x128
block-FP8 linear weights and keeps routed experts in the checkpoint's declared
format: MXFP4 E2M1 with UE8M0 scales for `expert_dtype=fp4`, or block-FP8 with
FP32 scales for `expert_dtype=fp8`. Router, normalization, compressor, and other
unscaled checkpoint families remain unquantized.

To load the mixed DeepSeek-V4 Flash checkpoint into the BF16 training master
but resync every quantized rollout matrix, including routed experts, as block
FP8, set `resync_config.expert_dtype=fp8` alongside
`resync_format=vllm_checkpoint`. The generic engine passes this model-owned
option through without interpreting DeepSeek-V4 tensor families.

For the H100/SM90 environment, short BF16 training, and the official-checkpoint
tensor resync proxy, follow [the Hopper runbook](../../docs/deepseek-v4-hopper.md).

## Smoke / Dry-Run Checks

Checked on this branch on 2026-06-07. These checks cover shell syntax,
Python import compilation, and resolved command construction only; they do not
cover end-to-end SFT or GRPO training.

- Shell syntax:
  - `bash -n experimental/lite/examples/verl/scripts/run_qwen3moe_sft.sh`
  - `bash -n experimental/lite/examples/verl/scripts/run_qwen3moe_gsm8k_sft.sh`
  - `bash -n experimental/lite/examples/verl/scripts/run_qwen3moe_gsm8k_grpo.sh`
- Python import compilation:
  - `PYTHONPYCACHEPREFIX="$(mktemp -d)" python3 -m compileall -q experimental/lite/examples/verl/verl_mlite`
- GSM8K SFT dry run:
  - `OUTPUT_ROOT="$(mktemp -d)" MODEL_PATH=Qwen/Qwen3.5-35B-A3B DRY_RUN=1 bash experimental/lite/examples/verl/scripts/run_qwen3moe_gsm8k_sft.sh`
  - Dry-run output shows `torchrun -m verl.trainer.sft_trainer`,
    `engine=mlite`, `model.path=Qwen/Qwen3.5-35B-A3B`,
    `data.train_files=${HOME}/data/gsm8k_sft/train.parquet`, and
    `data.val_files=${HOME}/data/gsm8k_sft/test.parquet`.
- GSM8K GRPO dry run:
  - `OUTPUT_ROOT="$(mktemp -d)" MODEL_PATH=Qwen/Qwen3.5-35B-A3B DRY_RUN=1 bash experimental/lite/examples/verl/scripts/run_qwen3moe_gsm8k_grpo.sh`
  - Dry-run output shows `python3 -m verl.trainer.main_ppo`,
    `actor@actor_rollout_ref.actor=mlite_actor`,
    `actor_rollout_ref.rollout.name=vllm`,
    `actor_rollout_ref.actor.engine.impl=lite`,
    `actor_rollout_ref.actor.engine.ep=8`,
    `algorithm.adv_estimator=grpo`, `actor_rollout_ref.actor.policy_loss.loss_mode=vanilla`,
    and `critic.enable=False`.
