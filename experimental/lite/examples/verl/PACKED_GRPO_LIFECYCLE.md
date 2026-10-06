# Packed deployment head and owned optimizer residency

Dependencies: DS4.1 model consumer and opt-in deployment providers, plus the
VERL engine/receiver integration for an actual GRPO job. Use the model's
original Muon configuration; do not replace native F32 masters with decoded
BF16 parameters. The ordinary non-deployment head path remains unchanged.

A deployment model returns materialized FP32 logits. The packed protocol
passes them directly to text_output instead of projecting a head_hidden
value again. Packed labels/masks shift within each document, including its
terminal token; temperature, entropy, loss scale and normalization denominator
come from LossContext. A supplied logits path requires CP1 and a full-vocab
TP1 head. Gradients reach the original FP32 head and norm leaves.

For the existing DS4.1 VERL GRPO configuration, select:

```text
actor_rollout_ref.actor.engine.impl_cfg.w4a8_experts=true
actor_rollout_ref.actor.engine.impl_cfg.deployment_math=true
actor_rollout_ref.actor.engine.impl_cfg.optimizer=muon
actor_rollout_ref.actor.engine.param_offload=true
actor_rollout_ref.actor.engine.optimizer_offload=false
```

Use the configured engine's supported Hydra add/override syntax for these
impl fields. Ref/rollout numerics, resync target and rewards still belong to
their normal configurations, rather than a diagnostic hook in this feature.
The online include_archival=False DS4.1 exporter setting is already in #231;
full HF archives retain their DSpark bytes.

During model transfer, the runtime moves the owned gradient buffers and
invokes the bundle's post_model_device_transfer_hook. Offload releases the
old public DDP wrapper; load binds a new wrapper around the same model owners.
This avoids stale AccumulateGrad/reducer connections after parameter storage
migration. The owned mixed optimizer moves each published state tensor beside
its owner; non-capturable/non-fused AdamW step counters remain CPU. A prepared
transaction cannot be migrated. No optimizer or parameter bank is replaced.

Run a small head/master and state-residency example:

```bash
PYTHONPATH=experimental/lite python \
  experimental/lite/examples/deployment/owned_head_update.py --device cpu
```

This needs the usual emerging_optimizers dependency. It performs two actual
owned Muon updates and state roundtrips; it is not a serving/GRPO parity test.
Use --device cuda in the supported environment to exercise device residency.
Packed probabilities/gradients, mixed-backend state equality and CPU Gloo DDP
rebinds are tested separately; actual CUDA offload cases are hardware gated.

The prior real-prefix integration completed five VERL GRPO steps, 20480 token
rawLP differences/K3KL exactly0, 40 FP32 updates and48 drained receivers. It
used an explicit deterministic digit/nonwhitespace proxy reward with nonzero
advantage; DAPO scoring was recorded separately and all160 scores were-1.
These results do not establish mathematical quality, full-release parity or
successful checkpoint restore after GRPO.
