# MLite parameter offload and copy/compute overlap

## Decision

Training-time parameter offload is feasible through two distinct MLite
implementations, but neither may claim copy/compute overlap without a GPU
timeline satisfying the quantitative gate below.

* PyTorch FSDP2 has the required mechanism: `CPUOffloadPolicy(pin_memory=True)`
  plus a multi-unit prefetch schedule.
* Standalone M-FSDP has a working, bounded CPU-shard path, but its default full
  training-offload policy is explicitly memory-first and disables parameter
  gather prefetch. It is therefore **not** an overlap implementation today.
* `engine.param_offload` remains a whole-model train/rollout residency switch.
  It is not either of the above and cannot be used as overlap evidence.

This is research only. It neither enables a new option nor changes the current
zero-GPU behavior.

## Existing MLite paths

`experimental/lite/examples/verl/verl_mlite/engine/mlite_engine.py` passes
`engine.param_offload` to the runtime transfer boundary. The FSDP2 branch in
`experimental/lite/megatron/lite/runtime/megatron_utils.py` reshares modules
and calls `model_chunk.to(device)`. The boundary synchronizes before releasing
memory. This is useful for colocated rollout, but it has no next-unit pipeline.

FSDP2 wrapping already accepts `offload_policy` in
`primitive/optimizers/fsdp2/wrap.py`; the training optimizer builder does not
construct one. Its `forward_prefetch_depth` and `backward_prefetch_depth`
configure FSDP all-gather scheduling only. They do not make CPU residency
happen without an offload policy.

Standalone M-FSDP lowers `offload_fraction=1.0` to
`MFSDPConfig(full_optimizer_offload=True)` in the historical implementation
commit [`0d14590c0`](https://github.com/NVIDIA/Megatron-LM/blob/0d14590c0/experimental/lite/megatron/lite/primitive/optimizers/mfsdp/optimizer.py#L494-L519).
It keeps the local compute-dtype shard and FP32 optimizer shard in pinned CPU
memory, stages the current local compute shard to a GPU lease with
`copy_(..., non_blocking=source.is_pinned())`, and all-gathers it. See
[`buffer.py`](https://github.com/NVIDIA/Megatron-LM/blob/0d14590c0/experimental/lite/megatron/lite/primitive/optimizers/mfsdp/buffer.py#L516-L575).

That code proves a bounded, asynchronous-capable H2D staging operation, not
overlap: with `offload_fraction=1.0`, the same builder sets
`overlap_param_gather=False` unless a caller explicitly overrides it. With no
next unit enqueued, `wait_param_gather()` launches (or waits for) the current
unit before its compute can start. An explicit override is only a hypothesis to
benchmark, since it may increase concurrent communication/scratch memory and
has not been validated for this path.

M-FSDP's CPU AdamW transfer ring is separate optimizer-state traffic. It owns
two pinned gradient slots and dedicated D2H/H2D streams, but its
`wait_grad_slice()` synchronizes an event before CPU Adam consumes a slice;
that must not be reported as parameter-prefetch overlap. See
[`cpu_offload.py`](https://github.com/NVIDIA/Megatron-LM/blob/0d14590c0/experimental/lite/megatron/lite/primitive/optimizers/mfsdp/cpu_offload.py#L18-L146).

## PyTorch source-level evidence and limits

The following was inspected against PyTorch `main` on 2026-08-17. It is
mechanistic evidence, not a performance result for an MLite build.

1. [`CPUOffloadPolicy`](https://github.com/pytorch/pytorch/blob/main/torch/distributed/fsdp/_fully_shard/_fsdp_api.py#L881-L909)
   specifies CPU-resident parameter, gradient, and optimizer state; its
   `pin_memory` documentation explicitly says pinned H2D/D2H copies may overlap
   compute.
2. [`FSDPParam._init_sharded_param`](https://github.com/pytorch/pytorch/blob/main/torch/distributed/fsdp/_fully_shard/_fsdp_param.py#L2749-L2771)
   moves the local shard to CPU and pins it when that policy is selected.
3. [`foreach_all_gather`](https://github.com/pytorch/pytorch/blob/main/torch/distributed/fsdp/_fully_shard/_fsdp_collectives.py#L2477-L2581)
   runs the CPU-to-GPU copy-in on `all_gather_copy_in_stream`, makes the
   all-gather stream wait for that stream, then records an all-gather event.
   The copy-in itself is `torch._foreach_copy_`, so a CPU-resident shard is not
   silently treated as a normal GPU input.
4. [`foreach_reduce`](https://github.com/pytorch/pytorch/blob/main/torch/distributed/fsdp/_fully_shard/_fsdp_collectives.py#L3163-L3225)
   only makes the gradient D2H copy non-blocking when the shard is pinned,
   gradients are not being accumulated, and no post-accumulate-grad hook will
   synchronously read the CPU gradient. It records an event which must complete
   before the optimizer can consume the gradient.

Thus `pin_memory=True` is necessary but insufficient. A single root FSDP unit
has one all-gather before all its compute and cannot overlap that H2D copy with
another unit's compute. The FSDP2 implementation must wrap a useful sequence
of units, retain a nonzero prefetch depth, and use a reshard choice that leaves
next-unit work eligible. MLite must pass the same policy to dense units and
expert-DP units; otherwise only part of an MoE model follows the claimed
residency contract.

## Required implementation shape

For FSDP2, add a separate training option (do not overload
`engine.param_offload`) and lower it in `build_fsdp2_training_optimizer()`:

```text
optimizer config: fsdp2_param_cpu_offload + pin_memory + prefetch depths
    -> CPUOffloadPolicy(pin_memory=True)
    -> wrap_fsdp2(..., offload_policy=policy)             [dense DP/CP mesh]
    -> wrap_fsdp2_module(..., offload_policy=policy)      [expert-DP mesh]
```

Reject the option if pinned host allocation is unavailable. A pageable fallback
is a valid memory-saving mode only if it has a different name and does not make
the overlap claim. Preserve real DTensor global shape and stride on CPU state
round trips, especially for uneven shards.

For M-FSDP, no option should be called `overlap` until it both (a) launches a
future unit's CPU-shard copy/all-gather before current-unit compute completes
and (b) bounds the simultaneous staging/full-buffer leases. The minimum design
needs separate copy/communication ordering, event ownership, and a specified
maximum number of live CPU/GPU staging buffers. Simply changing
`overlap_param_gather=True` is not sufficient evidence and may be invalid for
the MCore communication schedule.

Parameter offload must remain independent of optimizer-state offload and of
the FP32-gradient contract. It must not add a hidden FP32 parameter replica or
turn a BF16-produced gradient into a nominal FP32 gradient by casting.

## Quantitative GPU acceptance gate

CPU tests can validate configuration plumbing and buffer bounds but cannot
establish overlap. A future implementation must run a real Slurm GPU job with
the target topology and record its job ID, exact commit, clean worktree status,
and `sacct` exit code `0`.

Use Nsight Systems (or a trace exporter with CUDA kernel and memcpy start/end
timestamps) after at least five warm-up iterations and across at least ten
steady-state iterations. Compare an otherwise identical no-parameter-offload
baseline: model, mesh, unit wrapping, microbatching, precision,
`reshard_after_forward`, and prefetch depths must match.

For every eligible staged shard copy `C=[c_start,c_end]`, pair it with a
non-memcpy GPU compute kernel `K=[k_start,k_end]` from a different FSDP unit on
the same iteration. Define:

```text
overlap_us(C, K) = max(0, min(c_end, k_end) - max(c_start, k_start))
copy_coverage(C) = max_K overlap_us(C, K) / (c_end - c_start)
```

The overlap claim passes only when all conditions hold:

* the trace shows pinned H2D parameter copies (and, if claimed, pinned D2H
  gradient copies), with a distinct copy/communication stream and the required
  event dependency;
* at least 90% of eligible H2D copies have
  `copy_coverage >= 0.10` and overlap of at least 10 microseconds;
* report median and p10 coverage, total copied bytes, copies considered and
  excluded (including final/root units); never replace this with host API timing
  or the mere presence of `non_blocking=True`;
* local shards are CPU-resident between uses and GPU full/staging storage is
  released at the documented reshard boundary; and
* loss/gradient/update parity against the baseline is within the pre-declared
  precision tolerance, while reporting steady-state tokens/s and peak allocated
  GPU memory separately. Memory reduction without the coverage gate is not an
  overlap success.

The report must call out cases that intentionally suppress overlap: FSDP2
gradient accumulation or post-accumulate hooks disable its asynchronous D2H
path, and current M-FSDP full offload disables parameter-gather prefetch by
default. A skipped GPU test, CPU mock, or an asynchronous Python call without
device-timeline interval intersection does not satisfy this gate.
