# V4.1 pipeline integration

Depends on the standalone Engram distribution primitives and C4.

## Real model range interface

On the C4 assembly base, `DeepseekV41Model.forward_pipeline_range` executes an
explicit `[start,end)` range of the actual layers. It carries the shifted HC
pair, the saved Block(19) pair, latent/main KV, index keys, selection/candidate
metadata and the canonical KV/index owners. Layer 20 consumes the saved pair;
subsequent ranges preserve the source-20 KV while reindexing updates selection.
`finish_pipeline` uses the actual final HC contraction, normalization and head.

`protocol.pipeline_forward_step` exposes the range result and produces the same
text logits, shifted-label loss and log-probabilities as the ordinary protocol on
the final range. It currently requires one sequence per microbatch; multi-sequence
packed and CP inputs fail explicitly until separate state routing is integrated.
`build_model` now accepts PP-only topology through the existing `init_parallel`
and constructs equal contiguous local layer intervals. Non-owner layer slots are
`None`, preserving global indices; only the first stage allocates the embedding,
and only the last allocates normalization/head parameters. Local actual bindings
remain exact and unique. Standalone stage export is rejected until distributed
checkpoint assembly is implemented. The common scheduler's single-tensor contract
has not yet been extended to this paired payload.

The four-rank integration fixture constructs the real registry/protocol model,
compares two microbatches against monolithic execution, transfers every boundary
through typed NCCL P2P, reverses backward order, and compares each range's actual
parameter bindings and one owner update. Before PP construction it releases the
monolithic reference model and retains only local reference weights/gradients
and output values. The real parallel protocol allocates the local stage and
strictly loads those local weights. This reduced fixture uses FP32 diagnostic
projections, not native quantized full-scale qualification. Common scheduler,
VPP, mixed optimizers, TP/EP/CP composition and complete restart/export gates remain
open.
