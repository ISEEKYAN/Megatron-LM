# Runtime

The public runtime entrypoint is `megatron.lite.runtime`.

```python
from megatron.lite.runtime import MegatronLiteConfig, ParallelConfig, RuntimeConfig, create_runtime

cfg = RuntimeConfig(
    backend="mlite",
    hf_path="/path/to/hf-model",
    backend_cfg=MegatronLiteConfig(
        model_name="qwen3_moe",
        impl="lite",
        parallel=ParallelConfig(tp=1, pp=1, cp=1, ep=1),
    ),
)
runtime = create_runtime(cfg)
handle = runtime.build_model()
```

## API Tiers

All runtime backends implement the pretraining tier:

- `build_model`
- `save_checkpoint`
- `load_checkpoint`
- `train_mode`
- `eval_mode`
- `forward_backward`
- `zero_grad`
- `optimizer_step`
- `lr_scheduler_step`

The lite runtime also implements `export_weights` and `to` when the underlying
model and optimizer support those operations.

The `mbridge` runtime implements the same runtime contract through the legacy
`mbridge` package and Megatron-Core optimizer/checkpoint helpers. The benchmark
example currently uses this backend for validated reference runs.

The `bridge` runtime is the real Megatron-Bridge path. It imports
`megatron.bridge` lazily from `build_model()`, so config construction and dry-run
examples can execute without Megatron-Bridge installed.

## Config Types

`RuntimeConfig` selects the backend and carries the Hugging Face model path.

`MegatronLiteConfig` carries `mlite` backend settings:

- `model_name`: `qwen3_moe` or `qwen3_5` for new configs. `qwen3` remains
  accepted as a legacy alias for `qwen3_moe` only; dense Qwen3 is not included.
- `impl`: currently only `lite`.
- `parallel`: tensor, expert, pipeline, virtual pipeline, and context sizes.
- `optimizer`: Megatron-Core optimizer settings.
- `impl_cfg`: model-specific options consumed by each model protocol.

`BridgeConfig` carries shared `mbridge` and `bridge` backend settings:

- `model_name`: optional model identifier used for benchmark metadata.
- `parallel`: tensor, expert, pipeline, virtual pipeline, and context sizes.
- `optimizer`: Megatron-Core optimizer settings.
- `override_ddp_config`, `override_transformer_config`, and
  `override_optimizer_config`: explicit reference-backend/Core override maps.
- `param_offload` and `optimizer_offload`: offload model/optimizer state between
  train/eval contexts.

## Backend Registry

The built-in backend keys are `mlite`, `mbridge`, and `bridge`. Model
implementations for the native runtime remain selected through
`MegatronLiteConfig.impl`, which currently supports `impl="lite"`.

Custom runtime backends can be registered with:

```python
from megatron.lite.runtime import register_runtime

register_runtime("my_backend", "my_package.my_runtime")
```

The target module must expose `create(hf_path, cfg)`.

## Checkpoint tensor kwargs: gradient behavior change

`wrap_checkpoint` now flattens tensor leaves in positional arguments and keyword
arguments (including registered pytree containers) into explicit autograd Function
inputs. Repeated references to the same tensor share one replay leaf; distinct
views keep their input edges and storage aliases. Non-tensor leaves and the
original callable signature are retained. Do not mutate argument containers or
non-tensor state between forward and replay. Output-container support is unchanged.

**This fixes missing gradients and can change existing training results.**
Previously a keyword-only call could produce an output with no checkpoint gradient
edge even when its inputs and module weights required gradients. Tensor kwargs in
mixed calls could also bypass the detached replay inputs. The corrected gradients
are now enabled by default; historical runs depending on the old omission will
not be numerically equivalent. No compatibility mode silently preserves that bug.
If none of the explicit tensor leaves require gradients, the wrapper runs ordinary
forward with the caller's grad mode, so captured trainable parameters still get
gradients. That case retains activations instead of checkpointing them.

RNG preservation covers CPU and the already-initialized current CUDA device.
CPU-only checkpointing does not initialize CUDA. Moving to a new CUDA device or
initializing CUDA inside the wrapped function is outside this RNG contract.
`preserve_rng_state=False` still allows different random draws during replay.

```python
from megatron.lite.primitive.recompute import wrap_checkpoint

# Works for forward(self, *, hidden_states, position_ids, ...).
wrap_checkpoint(layer, preserve_rng_state=True)
out = layer(hidden_states=hidden_states, position_ids=position_ids)
```

## Optional EP backward contract check

Set `MEGATRON_LITE_VALIDATE_EP_BACKWARD=1` **on every rank of the job** to enable
an opt-in check in native MLite training. The runtime wires it into the ordinary
microbatch loop and the PP/virtual-PP schedules. Forward-only schedules bypass it;
EP=1 bypasses it. Direct users of `run_microbatch_loop` must pass `ps=parallel_state`.
Enabling it on only some ranks is unsupported and can itself leave peers waiting.

```python
from megatron.lite.primitive.train_step import run_microbatch_loop

# Set MEGATRON_LITE_VALIDATE_EP_BACKWARD=1 uniformly before launching the job.
run_microbatch_loop(model, data_iter, num_microbatches, forward_fn,
                    loss_fn=loss_fn, ps=parallel_state)
```

The check runs after local forward and external loss adaptation, **before loss
scaling, P2P output transfer (PP), backward, or checkpoint replay**. It validates a
scalar differentiable training loss on the last stage, or a differentiable hidden
output on a non-last stage. A non-last stage does not need a local loss. It then
compares exact integer metadata across the EP group: stage/microbatch identity and
counts of reachable native all-to-all, DeepEP dispatch/combine, reentrant checkpoint,
and activation-checkpoint nodes. Loss `requires_grad` alone is not the contract:
a loss connected only to a different branch, or a partial visible MoE graph, can
still have `requires_grad=True` but a different communication census.

All ranks at this boundary receive the same reduced bounds and raise on a mismatch
or invalid training root, before any of them enter the missing backward/replay.
Missing loss keys, non-tensor/None loss values, and detached roots are diagnosed
before loss scaling. Exceptions *inside* forward or the external callback are not
coordinated by this check. It assumes ranks reach the same training boundary and
use the same enable setting/group. Jobs intentionally omitting a last-stage
training objective or using frozen non-last-stage outputs should leave this
strict diagnostic disabled until their usage contract is defined.

Cost per enabled boundary: one graph traversal, one 18-element int64 MAX
all-reduce (144-byte logical payload), and a host synchronization to inspect the
result. There is one boundary per microbatch per local PP/VPP chunk, not one per
optimizer step. Disabled checks perform no graph traversal or collective. CPU/Gloo
latency is measured by the contract test; GPU/NCCL latency and throughput have not
been measured, so this remains opt-in.

Limitations: this is a graph census, not complete graph isomorphism or transport
validation. It cannot inspect an opaque checkpoint's *future replay*, distinguish
all same-count internal graph changes, validate token/split ordering, or rescue a
rank that never reaches the boundary. A partial detach occurring only inside
replay remains outside coverage. No zero gradients are fabricated, no absent
backward is forced, and no indexer loss or other training objective is introduced.
These fixes address CPU-reproduced mechanisms and contract gaps; they do **not**
establish the root cause of any production incident.

### Reference mechanisms

Reviewed `NVIDIA/Megatron-LM` `nv/dev@0cd11658f44350a141656751259cfe1f72398e9f`,
fetched 2026-09-11 UTC. Core's `transformer/moe/token_dispatcher.py` uses gathered
per-expert counts for variable splits (562–615), or capacity padding (536–559),
then calls token and probability all-to-all without a local-zero-token skip
(735–754). `moe_layer.py:837–871` routes, dispatches, computes experts, and combines;
`fused_a2a.py:135–204` uses layout/handle-based dispatch and backward combine.
`num_local_tokens` in `fused_a2a.py:998–1012` specifies output length, not an opt-out
from communication. These wrappers provide no external-loss graph agreement check
to reuse; their external kernels are not validated by our CPU tests.

The rollout-side HybridEP fix `99e7d7e85e3eba71d85b828c18117d623190369b` uses bitwise
route fingerprints to align independent dispatch streams before combine. Here
rank-local routes may legitimately differ, so matching route fingerprints across
ranks would be wrong. We reuse the explicit-metadata-before-consumption principle,
comparing the small contract vector exactly instead of hashing routing payloads.
