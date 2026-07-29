# LoRA in Megatron Lite

Megatron Lite accepts a `LoraSpec`, a matching dictionary, or `None` through
each model protocol's `ImplConfig.lora`.

```python
from megatron.lite.model.qwen3_5.lite.protocol import ImplConfig
from megatron.lite.primitive.modules.lora import LoraSpec

impl_cfg = ImplConfig(
    lora=LoraSpec(
        enabled=True,
        rank=128,
        alpha=256,
        dropout=0.1,
        ignore_patterns=("router", "output_layer"),
    )
)
```

`enabled=True` is the only opt-in. Supplying a positive `rank` without
`enabled=True` leaves LoRA disabled and emits a warning. `ignore_patterns`
matches exact, case-insensitive dotted module-path components; a matching
module is not wrapped.

## Parallel scope

LoRA supports TP and EP with `etp=1`. ETP is deliberately unsupported:
applying enabled LoRA with `etp>1` raises `NotImplementedError` before changing
the model. PEFT import/export also rejects `etp>1`; it never emits a partial or
silently incorrect adapter.

GLM-5 and DeepSeek V4 currently reject TP as a model-level limitation of their
DSA/CSA attention implementations. This is independent of LoRA.

## Model capability matrix

The matrix is maintained independently of the implementation diff so absent
model integrations remain visible. “Primitive-tested” means the shared
implementation has a direct test, but no model-specific distributed test.

| Model | Attach and trainable count | TP>1 gradients | Merged export | PEFT round-trip | `ignore_patterns` |
|---|---|---|---|---|---|
| Qwen3 MoE | Real tiny model tested | GQA and grouped-expert paths tested on CPU/Gloo TP=2 | Dense and grouped deltas unit tested; rollout merge path tested | Implemented for `pp=1, etp=1`; unit tested | Primitive behavior tested |
| Qwen3.5 | Real tiny model tested | GQA/grouped primitive-tested; GatedDeltaNet has no TP=2 LoRA gradient test | Primitive-tested; no model-specific merge test | Out of scope | Primitive behavior tested |
| Kimi K2 | Real tiny model tested | Grouped primitive-tested; MLA declarations have no TP=2 LoRA gradient test | Primitive-tested; no model-specific merge test | Out of scope | Primitive behavior tested |
| GLM-5 | Real tiny model tested | Out of scope: model rejects TP>1 | Primitive-tested; no model-specific merge test | Out of scope | Primitive behavior tested |
| DeepSeek V4 | Real tiny model tested | Out of scope: model rejects TP>1 | Primitive-tested; no model-specific merge test | Out of scope | Primitive behavior tested |

All five real tiny-model tests construct the actual model classes with CPU
Transformer Engine stand-ins, attach adapters, verify that only LoRA
parameters remain trainable, check the reported adapter count, and execute
each adapter forward. These are construction/adapter proxy tests rather than
full model-forward or GPU throughput tests.
