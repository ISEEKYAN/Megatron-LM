# DeepSeek-V4 packed document execution

`ImplConfig(packed_documents=True)` runs each packed document through the existing
V4 model, scopes router replay to that document, and computes a masked token
objective for the main head. It is a correctness path: it materializes vocabulary
logits and visits documents serially. The default remains `False`, with the
original model class, loss, MTP hooks and imports.

## Using the V4 protocol

Run in an initialized CUDA/distributed MLite environment with the existing V4
CSA kernels installed. Supply a model config/checkpoint that fits your environment.
The following is the forward/backward portion of a single-stage training program;
process-group initialization and optimizer stepping remain with the caller.

```python
import torch

from megatron.lite.model.deepseek_v4.lite import protocol
from megatron.lite.runtime.contracts import PackedBatch

model_cfg = protocol.build_model_config("path/to/hf-checkpoint")
impl_cfg = protocol.ImplConfig(
    optimizer=None,                 # caller owns the optimizer
    mtp_enable=False,
    packed_documents=True,
    recompute=["full"],              # optional: checkpoint each complete document
)
bundle = protocol.build_model(model_cfg, impl_cfg=impl_cfg)
model = bundle.chunks[0]
# Load weights for training an existing checkpoint:
protocol.load_hf_weights(model, "path/to/hf-checkpoint", model_cfg, bundle.parallel_state)

# Original, unshifted token labels and mask. The protocol shifts each document
# separately and zeros its last target; do not shift them again here.
ids = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8], device="cuda")
batch = PackedBatch(
    input_ids=ids,
    labels=ids.clone(),
    seq_lens=torch.tensor([3, 5], device="cuda"),
    loss_mask=torch.ones_like(ids, dtype=torch.float32),
)
output = bundle.forward_step(model, batch)
output["loss"].backward()
```

The loss is `-sum(token_log_probs * shifted_mask) / max(sum(shifted_mask), 1)`.
For these two documents the denominator is six, not eight. An all-zero mask
produces zero loss. `LossContext` supplies temperature, entropy/output policy
and loss scaling through the usual protocol. Low-level callers passing
`packed_seq_params` directly must supply already-shifted labels and an aligned
`loss_mask`, including zeros for padding and terminal positions.

R3 uses the existing runtime `router_replay="record"` / `"replay"` lifecycle.
Replay inputs must include the hash-router and learned-router columns in model
order and a causal `r3_replay_mask`; `protocol.pack_routed_experts` and
`protocol.pack_r3_replay_mask` retain the existing physical token layout.
The driver owns router attachment and cleanup. Do not attach MTP routers to the
main decoder replay list. A whole packed backward invocation consumes one FIFO
entry per router, regardless of document count. Internal document checkpoints
hold their own target snapshots; pending driver queues are cleared by driver end.

## Primitive contracts

- `packed_objective(logits, labels, mask, temperature=..., denominator=...)`
  accepts **already-aligned** targets and an optional external token denominator.
  `tp_group` refers to vocabulary partitioning. Inference logits retain that
  partition; callers decide whether to gather. There is no target shifting,
  model policy, or runtime-context import inside the primitive.
- `packed_forward(sequence_forward, state, cu_seqlens, axes=..., output_axes=...)`
  slices tensor trees on explicit token axes, calls the sequence function once
  per document, and concatenates its output tree. A scalar axis applies to all
  leaves; otherwise axes mirror the tree. V4 uses input axis 1, hidden-output
  axis 0 and logits-output axis 1. Paired BSHD states can use `(1, 1)`.
- Offsets describe **physical storage**. V4 explicitly uses
  `cu_seqlens_q_padded`; an unpadded adapter supplies its own unpadded offsets.
  The callable creates fresh document-local state. An optional `cp_context`
  provides `total_length` and `document(begin, end).local_length`; empty local
  intersections still invoke the callable.
- `routers` is an explicit collection of `RouterReplay` instances, not a scan
  of global models. Each active router must visit every document token exactly
  once. Missing routers/tokens or incomplete partitions raise. Exceptions
  restore action/targets/recorded state and leave FIFO entries unconsumed.

## Supported scope and validation

This V4 adapter requires TP=ETP=CP=PP=EP=VPP=1 and no activation offload, and either no recompute or `recompute=["full"]`.
Full recompute means **whole-document** checkpointing; it requires packed
metadata. Submodule recompute selections are rejected. EP needs a shared cross-rank
document schedule, which this local executor does not provide. Packed execution disables
MTP, as the existing packed protocol already does. Non-packed calls with no
recompute retain the ordinary model path. The default model/MTP path is unchanged.
When no attention override is supplied, the opt-in adapter selects the existing
`flash` sparse backend; the default adapter retains its previous selection.

CPU tests construct the real V4 modules and compare packed execution against
ordinary independent-document execution, including outputs, parameter gradients,
forced routes, recompute, the real packer and R3 driver cleanup. TE RMSNorm,
GroupedLinear and the sparse CUDA attention kernel boundary are replaced with
Torch CPU implementations in those tests. Default `build_model` weights,
outputs and gradients are compared byte-for-byte against pinned main, including
MTP training. An import guard verifies default execution loads no new packed
modules. No GPU, EP, distributed optimizer, or compressed-CSA kernel validation
is claimed by these CPU tests.
