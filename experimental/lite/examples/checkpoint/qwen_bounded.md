# Bounded Qwen3-MoE checkpoints

The existing checkpoint API is unchanged by default. Opt in on an already
constructed Qwen3MoEModel (including a QAT-parametrized model):

```python
from megatron.lite.model.qwen3_moe.lite import protocol

# `model` is the real Qwen3MoEModel in your model bundle.
# All data-parallel replicas must call save/export and drain the iterator.
owners = model.checkpoint_bindings()
assert len(owners) == len(list(model.parameters()))
options = dict(bounded=True, buffer_max_size_bytes=2 * 1024**3)
protocol.save_hf_weights(
    [model], "checkpoint", model.config, model.ps, target="mxfp4", **options
)
protocol.load_hf_weights(model, "checkpoint", model.config, model.ps, **options)
```

The root contains ordinary HF safetensors, with Qwen's compressed-tensors MXFP4
weight/scale names when `target="mxfp4"`. `target="hf"` and its `target="bf16"` alias preserve parameter
dtypes. Keep the source HF configuration/tokenizer assets alongside these weight
files; this API writes weights, not a complete Transformers model package.
`mlite_masters/` contains every original parameter in its exact native dtype,
including FP32 QAT masters. Loading a directory with that sidecar requires a
complete matching master inventory and dtypes. Loading HF weights without it
decodes the release representation and cannot recover rounding information.

To export without a training sidecar, consume the iterator immediately:

```python
from megatron.lite.primitive.ckpt.hf_weights import stream_export_to_shards

stream_export_to_shards(
    protocol.export_hf_weights(
        [model], model.config, model.ps, target="mxfp4", **options
    ),
    "hf_weights",
)
```

The bounded iterator contains borrowed `RowChunk` objects as well as ordinary
named tensors. Never collect it with `list()` or `dict()`. A row chunk expires
when the iterator advances. Embedding/head and exact-master matrices are streamed
through fixed buffers; QKV/expert layout conversions must fit the configured
buffer or fail explicitly. Resident parameters and caller-owned outputs are
excluded from the scratch limit.

This opt-in Qwen adapter accepts one complete model chunk, replicated DP, and
TP=EP=ETP=PP=CP=1. It rejects MTP, unmapped parameters (including LoRA), meta
parameters and DTensors. Row streaming uses local complete tables; no distributed
row ownership or Qwen tensor/expert-parallel resharding is provided.

CPU checkpoint tests instantiate the production Qwen model with the existing
test-only TE parameter containers. They cover checkpoint/codec/master behavior;
they do not emulate TE kernels or establish forward/backward numerical parity.

Limits of the memory evidence: non-row loads retain complete decoded sources
(e.g. QKV) together before assembly. Per-source allowances are conservative, not
a measured aggregate peak guarantee. RSS is sampled every 2 ms; short-lived
allocations can be missed, and lifetime ru_maxrss is recorded but not asserted.
Save-side allocation peaks and borrowed-view retention across iterator advances
are not separately tested. MXFP4 bounded tests establish decoded round trips,
not packed-byte identity against the unbounded exporter. DTensor/LoRA rejection
and an independent primitive-spec fixture remain coverage gaps.

Training saves always include the complete native-master sidecar, and loads
prefer it over release values. Save reserves half the budget for export; export
uses another factor of 64 for row scratch (approximately caller budget / 128).
The budget therefore is not the available row payload capacity.
