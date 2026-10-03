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
weight/scale names when `target="mxfp4"`. `target="hf"` preserves parameter
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
parameters and DTensors. The row primitive separately supports explicit uneven
row ownership and loading a different local row span. No Qwen tensor-parallel
or expert-parallel resharding is claimed for the opt-in adapter.

CPU checkpoint tests instantiate the production Qwen model with the existing
test-only TE parameter containers. They cover checkpoint/codec/master behavior;
they do not emulate TE kernels or establish forward/backward numerical parity.
