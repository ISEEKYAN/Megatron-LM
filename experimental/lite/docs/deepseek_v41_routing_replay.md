# V4.1 routing replay integration

V4.1 uses the shared `RouterReplayDriver` and sigmoid router hooks. The protocol
packs routes and causal replay masks with `contiguous=True` and
`contiguous_padding=True`: `thd_pack_meta(..., contiguous=True)` padding
(alignment `TP * CP`), then a contiguous CP slice, then the TP/SP
slice. `PackedBatch.seq_lens` always describes the original, unpadded sequences.
The model's token buffer must use this same layout; do not slice it twice.

For the existing single-rank text protocol, use the runtime's
`forward_backward(..., router_replay="record")`. Its forward result contains
`model_output.routed_experts`. Supply these routes and an explicit causal
`PackedBatch.r3_replay_mask` to `router_replay="replay"`. Scores are recomputed
from the live router, including for externally selected experts. Unmasked rows
keep native routing. Supplying routed inputs directly to the model protocol
without an active replay driver fails explicitly.

## Packed samples and activation checkpoints

`packed_forward` preserves every sample's recorded routes in token order and
slices replay targets and masks at the same boundaries. The shared adapter is
also available to a pipeline caller that already owns its token layout:

```python
from megatron.lite.primitive.modules.router_replay import PackedRouterReplay

scope = PackedRouterReplay(total_local_tokens)
for begin, end in local_sequence_ranges:
    with scope.sequence(begin, end):
        # Invoke the existing model range entry on this sequence.
        ...
scope.finish()
```

Ranges must partition the local router buffer without gaps or overlap, and each
active local router must be visited once per range. This adapter does not
construct, distribute, or schedule a model. Whole-packed recomputation consumes
one existing FIFO entry per router and microbatch. For activation checkpoints
inside a range, bind the checkpoint's participating module explicitly:

```python
from functools import partial
from torch.utils.checkpoint import checkpoint
from megatron.lite.primitive.modules.router_replay import router_replay_checkpoint_contexts

output = checkpoint(
    block_forward,
    hidden,
    use_reentrant=False,
    context_fn=partial(router_replay_checkpoint_contexts, model=block),
)
```

Create that checkpoint inside the sequence scope. The contexts capture its
forward routes and mask; reverse-order segment recomputation after another
microbatch therefore cannot consume that microbatch's mutable targets. A
checkpoint covering the entire registered router set can omit `model`.

## Pipeline boundary

Pipeline stages retain global layer slots, with `None` outside
`local_layer_range=(start, end)`. The protocol selects only those live routers;
archival DSpark modules are excluded. Shared PP route selection operates on the
global layer axis, before token packing. Only one local pipeline chunk is
supported by this integration; VPP fails explicitly.

Record collection must run after all participating stages have completed their
forward work. A scheduler can retain each microbatch's
`RouterReplay.get_recorded_data()` list under its existing pipeline generation
identity, then call, in matching collective order on all participating ranks:

```python
routes = protocol.unpack_recorded_routed_experts(
    model, batch, recorded, pipeline_drained=True
)
```

This gathers TP/SP rows, contiguous CP rows, and variable-width PP layer columns,
then removes THD padding. The caller must retain separate lists per microbatch
and generation until collection. The current runtime driver rejects PP record
before entering stage execution: its post-drain generation callback is not yet
connected. There is no PP collective hidden inside a stage forward. The existing
V4.1 model's parallel-construction guards remain in force; distributed model
execution and its scheduler are supplied by the parallel-model integration.

The tests cover packed record and replay, live-score gradients, checkpoint
recomputation, protocol fallback detection, and four-rank PP2/CP2 and TP2/CP2
route collectives. EP groups are present in those topologies, but these are router
and packing tests, not expert-dispatch or full distributed-model acceptance.
