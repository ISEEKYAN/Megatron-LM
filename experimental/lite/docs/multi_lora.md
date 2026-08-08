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

Standard expert parallelism is supported: replicated banks synchronize their
gradients over the expert-parallel group while routing preserves sidecar row
ordering. Expert tensor parallelism and DeepEP fail loudly when a sidecar is
requested, rather than silently selecting a different execution path.

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
- Registry save/load round trip and invalid-index checks.
- MoE composition, recomputation, empty-route, and unsupported-mode checks.
- Two-rank standard expert-parallel smoke coverage.
