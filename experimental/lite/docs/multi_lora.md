# Multi-LoRA for Qwen3-MoE

Multi-LoRA provides a dense bank of LoRA adapters selected by one `int64`
adapter index per token. The primitive keeps the prototype API: a bank owns
`A[slots, rank, in_features]` and `B[slots, out_features, rank]`, and its
`delta(x, lora_indices, scale=...)` returns one delta row for every input row.

## MoE data flow

The Qwen3-MoE integration uses a `MoELoraSidecar` containing separate `fc1`
and `fc2` banks plus the original-token adapter indices.

1. `fc1` deltas are calculated once for the original tokens, before routing.
2. The dispatcher carries those deltas and indices alongside routed tokens.
3. Each expert adds the routed `fc1` delta after its first grouped linear
   operation and before activation.
4. Each expert calculates the `fc2` delta from the routed activation and adds
   it after its second grouped linear operation.

The grouped linear operations remain unchanged. Empty routes remain valid
because sidecars use the same dispatch and combine ordering as their tokens.
Activation recomputation consumes the already routed `fc1` sidecar, preserving
the one-computation-per-original-token rule.

Standard expert parallelism is supported: model-owned replicated banks use
the dist_opt dense-DP/finalize lifecycle as their sole gradient reduction
owner, while legacy external sidecars retain explicit EP synchronization.
Routing preserves sidecar row ordering. Expert tensor parallelism and DeepEP
fail loudly when a sidecar is requested, rather than silently selecting a
different execution path.

## Model-owned named adapters

Enable named banks through the runtime model configuration; do not inject a
registry or `MoELoraSidecar` from a test or caller. The model builder owns the
parameters, exposes the same registry through `ModelHandle._extras`, and uses
the native Qwen3-MoE expert-weight names for checkpoint and HF export mapping.

```python
impl_cfg = {
    "multi_lora": {"names": ["customer-a", "customer-b"], "rank": 16, "alpha": 32},
}
# Per batch: one resident adapter slot for each original token, per layer.
batch.extras["multi_lora_slots"] = {0: torch.tensor([0, 0, 1], dtype=torch.int64)}
```

The training forward step creates sidecars from those slots and the model-owned
banks. A named export selects `multi_lora_name="customer-a"` from that same
registry and always uses the HF TP/EP gather, native-name mapping, and scaling
path. The banks are shared across experts within a layer; export therefore
emits the corresponding factors for every expert rather than pretending they
are independently trained expert-specific adapters.

## Minimal bank usage

```python
import torch
from megatron.lite.primitive.modules.multi_lora_bank import DenseLoraBank

a = torch.randn(2, 4, 16, device="cuda", dtype=torch.bfloat16)
b = torch.randn(2, 32, 4, device="cuda", dtype=torch.bfloat16)
x = torch.randn(8, 16, device="cuda", dtype=torch.bfloat16)
adapter_for_token = torch.tensor([0, 0, 0, 1, 1, 1, 1, 1], device="cuda")

bank = DenseLoraBank(a, b)
delta = bank.delta(x, adapter_for_token, scale=0.25)
assert delta.shape == (8, 32)
```

Adapter indices must be `torch.int64`, device-resident, and non-decreasing for
the primitive interface. The MoE composition sorts and restores sidecar
indices when routing produces another order.

## Validation matrix

- Dense PyTorch reference and accelerated forward/backward parity.
- Model-owned registry/config construction, named HF export, and invalid-index checks.
- MoE composition, recomputation, empty-route, and unsupported-mode checks.
- Two-rank standard expert-parallel smoke coverage.
## Supported optimizer lifecycle

Model-owned multi-LoRA currently supports the `dist_opt` lifecycle only.
`fsdp2` is rejected at model construction: its parameter replacement/flattening
would invalidate the registry's bank object identity until it has a dedicated
ownership integration.  This is a fail-loud configuration boundary, not a
fallback to external sidecars.
