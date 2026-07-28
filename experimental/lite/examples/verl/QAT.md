# Weight-only QAT in Megatron Lite

Megatron Lite applies weight-only fake quantization inside each model protocol.
The configuration surface is `ImplConfig.qat`; Qwen3 MoE, Qwen3.5,
DeepSeek V4, GLM-5, and Kimi K2 accept either a `QATSpec`, a matching
dictionary, or `None`.

For example, Qwen3 MoE training can opt into MXFP4 QAT with:

```python
from megatron.lite.model.qwen3_moe.lite.protocol import ImplConfig
from megatron.lite.primitive.quantization import QATSpec

impl_cfg = ImplConfig(
    qat=QATSpec(
        enabled=True,
        format="mxfp4",
        group_size=32,
        symmetric=True,
        ste_clip=True,
        ignore_patterns=("lm_head", "head", "gate", "router", "embedding", "embed"),
        export_mode="fake",
    )
)
```

The same fields can be supplied as YAML/Hydra data under `impl_cfg.qat`.

Rollout export has a separate, verl-owned configuration under the engine's
top-level `qat` key:

```yaml
qat:
  enable: true
  apply_modelopt_fake_quant: false
  mode: mxfp4
  group_size: 32
  ignore_patterns:
    - lm_head
    - embed_tokens
    - "re:.*mlp.gate$"
```

These three export ignore patterns are required for this MLite path. In the
verl exporter used by this integration, a missing or empty `ignore_patterns`
value becomes an empty list; there is no built-in export ignore set.
Consequently, the HF-stream exporter otherwise quantizes every eligible
`.weight` tensor other than norms, including the output head, embeddings, and
router gate. Literal export patterns use substring/glob matching, and a
`"re:"` prefix selects regular-expression matching.

Do not confuse this with the training-side `QATSpec.ignore_patterns`.
`QATSpec` has its own built-in defaults for embedding, output-head, and router
path components and matches exact, case-insensitive dotted path components.
Changing the training list does not change verl's export list, or vice versa;
configure both owners explicitly as shown above.

## `QATSpec` fields

| Field | Meaning |
|---|---|
| `enabled` | Explicit opt-in. `False` registers no parametrizations and leaves the model unchanged. |
| `format` | Quantization contract: `int8`, `int4`, `fp8_e4m3`, or `mxfp4`. The alias `fp8` canonicalizes to `fp8_e4m3`. |
| `group_size` | `0` is per-tensor, `-1` is per-output-channel, and a positive value groups the input-feature dimension. MXFP4 requires the OCP block size `32`. |
| `symmetric` | Selects symmetric rather than affine integer quantization. Floating-point formats do not use an integer zero point. |
| `ste_clip` | `True` zeros gradients outside the representable range; `False` uses a pure pass-through straight-through estimator. |
| `ignore_patterns` | Exact, case-insensitive dotted path components to skip. Defaults exclude embeddings, output heads, and router gates. |
| `export_mode` | `"fake"` identifies the training representation; `"packed"` identifies a deployment snapshot request. |

The supported and deferred formats are:

| Format | Status | Notes |
|---|---|---|
| `int8` | supported | Weight-only integer fake quantization and packed integer export. |
| `int4` | supported | Weight-only integer fake quantization and packed 4-bit export. |
| `fp8_e4m3` | supported | E4M3 fake quantization and FP8-code export. |
| `mxfp4` | supported | OCP E2M1 values with one E8M0 scale per 32 weights. |
| `nvfp4_w4a16`, `nvfp4_w4a4` | deferred | Both names are in `qat.py::_DEFERRED_FORMATS`; selecting either raises `ValueError`. NVFP4 needs a separate scale and serializer contract and must not reuse the deleted recipe table. |

## Construction and checkpoint ordering

Every protocol calls `apply_qat_to_chunks` after model construction and before
optimizer construction. This order is mandatory: parametrization moves the
trainable BF16 parameter from `...weight` to
`...parametrizations.weight.original`, and the optimizer must capture that
master parameter. Applying QAT after optimizer construction can leave the
optimizer holding the wrong parameter set.

For the same reason, checkpoint readers canonicalize a logical
`...weight` key onto `...parametrizations.weight.original` when QAT is active.
The fake-quantized view is derived during forward; it is not a checkpoint
master weight.

## Training and packed deployment snapshots

Use `export_mode="fake"` while training. The parametrized forward computes a
dequantized fake weight and the optimizer updates the original BF16 weight.

Use `export_mode="packed"` when requesting deployment material, then call
`quantize_weight` on the master weight. For MXFP4 the result is:

```python
{
    "qweight": uint8_e2m1_nibbles,
    "scale": uint8_e8m0_exponents,
    "format": "mxfp4",
}
```

Packed weights never enter the optimizer or training forward. The VERL engine
hands the exported stream to its rollout exporter, while the vLLM compatibility
and refit boundary is installed from
`experimental/lite/examples/verl/verl_mlite/compat.py`.

## End-to-end QAT training launch

The following recipe launches the same Qwen3 MoE MXFP4-QAT arm shape used by
the four-arm DAPO experiment: 32 GPUs (four nodes with eight GPUs each), 30
steps, eight responses per prompt, 2,048 prompt tokens, 14,336 response tokens,
and a 16,384-token actor/rollout limit.

The user must provide:

- a local snapshot of the public `Qwen/Qwen3-30B-A3B` model;
- a verl-compatible training parquet derived from the public
  `BytedTsinghua-SIA/DAPO-Math-17k` dataset;
- a verl-compatible AIME 2024 validation parquet;
- local checkouts of verl and Megatron-LM on `VERL_ROOT` and `MEGATRON_ROOT`;
- a running four-node Ray allocation with eight suitable GPUs per node.

Run this from the Megatron-LM repository root on the Ray head node. Replace
the six `/path/to/...` placeholders:

```bash
export VERL_ROOT=/path/to/verl
export MEGATRON_ROOT=/path/to/Megatron-LM
export MODEL_PATH=/path/to/Qwen3-30B-A3B
export TRAIN_FILES=/path/to/dapo-math-17k.parquet
export VAL_FILES=/path/to/aime-2024.parquet
export OUTPUT_ROOT=/path/to/output

NNODES=4 \
NGPUS_PER_NODE=8 \
TRAIN_BATCH_SIZE=32 \
PPO_MINI_BATCH_SIZE=32 \
ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU=1 \
MAX_PROMPT_LENGTH=2048 \
MAX_RESPONSE_LENGTH=14336 \
PPO_MAX_TOKEN_LEN_PER_GPU=16384 \
ROLLOUT_LOG_PROB_MAX_TOKEN_LEN_PER_GPU=16384 \
ROLLOUT_MAX_MODEL_LEN=16384 \
ROLLOUT_MAX_NUM_BATCHED_TOKENS=16384 \
ROLLOUT_MAX_NUM_SEQS=32 \
ROLLOUT_N=8 \
ROLLOUT_TP=8 \
ROLLOUT_GPU_MEMORY_UTILIZATION=0.6 \
ACTOR_TP=2 \
ACTOR_EP=8 \
ACTOR_CP=1 \
ACTOR_LR=1e-5 \
USE_FUSED_KERNELS=True \
PARAM_OFFLOAD=True \
OPTIMIZER_OFFLOAD=True \
GRAD_OFFLOAD=True \
LOSS_AGG_MODE=token-mean \
TOTAL_TRAINING_STEPS=30 \
SAVE_FREQ=5 \
TEST_FREQ=5 \
bash experimental/lite/examples/verl/scripts/run_qwen3moe_gsm8k_grpo.sh \
  'trainer.use_v1=True' \
  'algorithm.rollout_correction.bypass_mode=False' \
  '+algorithm.filter_groups.enable=True' \
  '+algorithm.filter_groups.metric=acc' \
  '+algorithm.filter_groups.max_inflight_gen_batches=1' \
  'actor_rollout_ref.actor.clip_ratio_low=0.2' \
  'actor_rollout_ref.actor.clip_ratio_high=0.28' \
  'actor_rollout_ref.actor.engine.router_replay_mode=disabled' \
  '+actor_rollout_ref.actor.engine.impl_cfg.recompute=full' \
  'actor_rollout_ref.actor.engine.qat.enable=true' \
  'actor_rollout_ref.actor.engine.qat.apply_modelopt_fake_quant=false' \
  'actor_rollout_ref.actor.engine.qat.mode=mxfp4' \
  'actor_rollout_ref.actor.engine.qat.group_size=32' \
  'actor_rollout_ref.actor.engine.qat.ignore_patterns=[lm_head,embed_tokens,"re:.*mlp.gate$"]' \
  'actor_rollout_ref.actor.engine.impl_cfg.qat.enabled=true' \
  'actor_rollout_ref.actor.engine.impl_cfg.qat.format=mxfp4' \
  'actor_rollout_ref.actor.engine.impl_cfg.qat.group_size=32' \
  'actor_rollout_ref.actor.engine.impl_cfg.qat.symmetric=true' \
  'actor_rollout_ref.actor.engine.impl_cfg.qat.ste_clip=true' \
  'actor_rollout_ref.actor.engine.impl_cfg.qat.ignore_patterns=[lm_head,head,output_layer,gate,router,embedding,embed,word_embeddings]' \
  'actor_rollout_ref.actor.engine.impl_cfg.qat.export_mode=fake' \
  'actor_rollout_ref.rollout.quantization=mxfp4'
```

`engine.qat` controls the verl-owned online deployment export;
`engine.impl_cfg.qat` controls MLite's training parametrizations. Both must be
configured. Do not replace either ignore list with the other: their owners and
matching semantics differ as described above.

This command shape and its resolved Hydra argument list can be checked without
allocating GPUs by adding `DRY_RUN=1`. A dry run validates configuration
construction only; it is not evidence that training ran.
