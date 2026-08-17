# FSDP2 parameter offload and copy/compute overlap

## Decision

Parameter offload is technically feasible for MLite, but the existing
`engine.param_offload` switch is not an implementation of FSDP2 parameter
offload and does not satisfy the overlap requirement.

This revision is research-only. It does not enable the feature and does not
add a zero-GPU implementation.

## What exists today

There are two different mechanisms in the current tree:

1. `experimental/lite/examples/verl/verl_mlite/engine/mlite_engine.py`
   handles `engine.param_offload` at a training/rollout context boundary. It
   calls the runtime transfer path for the complete model. The MLite runtime
   then calls `offload_model_to_cpu()` or `load_model_to_gpu()` in
   `experimental/lite/megatron/lite/runtime/megatron_utils.py`. The FSDP2
   branch first reshares modules and then calls `model_chunk.to(device)`. This
   is a whole-model residency transition; it is not a per-unit prefetch
   pipeline, and the boundary deliberately synchronizes CUDA before releasing
   memory.

2. `experimental/lite/megatron/lite/primitive/optimizers/fsdp2/wrap.py`
   already accepts an `offload_policy` and forwards it to PyTorch
   `fully_shard`. However, `build_fsdp2_training_optimizer()` currently does
   not expose or construct that policy. The default MLite path therefore uses
   the normal FSDP2 residency policy. Its existing `forward_prefetch_depth`
   and `backward_prefetch_depth` settings prefetch FSDP all-gathers, but do not
   turn CPU parameter residency on by themselves.

The existing offload tests validate device residency, round trips, checkpoint
behavior, and optimizer updates. They do not measure copy/compute overlap.

## Why the native FSDP2 route is the viable route

The supported PyTorch API is `torch.distributed.fsdp.CPUOffloadPolicy`, passed
as `offload_policy` to `fully_shard`:

```python
CPUOffloadPolicy(pin_memory=True)
```

Its contract is the required one: sharded parameters are copied host-to-device
before all-gather, sharded gradients are copied device-to-host during
backward, and pinned host memory enables those transfers to overlap with
compute. `reshard_after_forward` controls when the all-gathered parameter is
released. This is materially different from calling `Module.to("cpu")` around
the entire training loop.

The intended MLite integration is therefore:

```text
engine/optimizer config
        |
        v
build_fsdp2_training_optimizer(..., param_offload=...)
        |
        v
CPUOffloadPolicy(pin_memory=True)
        |
        v
wrap_fsdp2(..., offload_policy=policy,
           reshard_after_forward=..., forward_prefetch_depth=...)
```

The policy must be attached to every FSDP2 unit, including expert units that
use a separate expert-DP mesh. The existing prefetch wiring should remain
enabled and be evaluated together with `reshard_after_forward`; otherwise a
policy can be present while the schedule still has no useful next-unit work
to overlap.

## Constraints and implementation hazards

- `pin_memory=True` is necessary for the overlap claim, but increases pinned
  host-memory pressure. The implementation needs an explicit memory budget and
  a configuration validation/error path; silently falling back to pageable
  memory would violate the hard requirement.
- `engine.param_offload` should not be reused as the per-unit policy without
  changing its semantics. It currently means “move the model out of GPU at a
  context boundary,” which is useful for colocated rollout but is not a
  training-time overlap control.
- The optimizer state offload path is separate from parameter offload. Its
  CPU↔GPU moves in `fsdp2/state.py` happen at optimizer/context boundaries and
  must not be cited as evidence of parameter/compute overlap.
- FSDP2 resharding and offload must preserve the real global shape/stride
  metadata for non-divisible shards. Local shard shape is insufficient when
  reconstructing a DTensor after a CPU round trip.
- The existing FP32-shard choice should remain explicit. CPU parameter
  residency must not be implemented by introducing an accidental second FP32
  master copy or by changing the gradient contract.

## Required GPU validation before implementation is accepted

No CPU-only test can establish the hard requirement. A future implementation
must run on the target Slurm GPU environment and provide all of the following:

1. A baseline with the same model, mesh, microbatching, precision, and
   `reshard_after_forward`, but without CPU parameter offload.
2. The same run with `CPUOffloadPolicy(pin_memory=True)` and the intended
   prefetch depths.
3. Nsight Systems (or an equivalent CUDA timeline) showing an H2D/D2H copy
   interval overlapping a non-copy compute kernel for at least one steady-state
   FSDP unit, rather than merely showing asynchronous API calls.
4. A residency check proving that the sharded parameter and gradient storage
   are CPU-resident between uses, plus numerical parity against the baseline.
5. Throughput and peak GPU-memory measurements. The report must distinguish
   overlap from a slower but memory-saving execution.

The GPU evidence must be a real Slurm job with `sacct` return code 0; a skipped
GPU test or a CPU mock is not evidence for this acceptance criterion.

## Recommendation for a follow-up implementation

Add a narrowly scoped FSDP2 parameter-offload option to the FSDP2 optimizer
configuration, construct `CPUOffloadPolicy(pin_memory=...)` in the FSDP2
builder, and pass the same policy to dense and expert wrappers. Keep runtime
context-boundary offload unchanged for rollout memory reclamation. Add a GPU
timeline test/benchmark and reject configurations that cannot provide pinned
host memory. Do not implement this follow-up in the current zero-GPU
revision.
