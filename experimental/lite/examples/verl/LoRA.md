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

The post-build lifecycle is symmetric:

```python
from megatron.lite.primitive.modules.lora_apply import (
    apply_lora_to_chunks,
    load_lora_adapter_state,
    merge_lora_in_chunks,
    remove_lora_from_chunks,
    save_lora_adapter_state,
    unmerge_lora_in_chunks,
)

apply_lora_to_chunks(chunks, spec, ps=ps, model_targets=LORA_TARGETS)
save_lora_adapter_state(chunks, "adapter.pt")
load_lora_adapter_state(chunks, "adapter.pt")
merge_lora_in_chunks(chunks)
unmerge_lora_in_chunks(chunks)
remove_lora_from_chunks(chunks)
```

Applying twice, removing an absent or still-merged adapter, merging twice,
unmerging an unmerged adapter, saving without adapters, or loading a
missing/incompatible adapter checkpoint raises immediately. An enabled
configuration that matches no declared target also raises and restores the
original trainability state instead of reporting a successful no-op.

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
| Qwen3 MoE | Real tiny model tested | GQA and grouped-expert paths tested on CPU/Gloo TP=2 | Real tiny model merge/unmerge tested; rollout merge path tested | Native PEFT import/export for `pp=1, etp=1`; generic save/load also tested on the real tiny model | Primitive behavior tested |
| Qwen3.5 | Real tiny model tested | GQA/grouped primitive-tested; GatedDeltaNet has no TP=2 LoRA gradient test | Real tiny model merge/unmerge tested | Native PEFT format out of scope; generic save/load tested on the real tiny model | Primitive behavior tested |
| Kimi K2 | Real tiny model tested | Grouped primitive-tested; MLA declarations have no TP=2 LoRA gradient test | Real tiny model merge/unmerge tested | Native PEFT format out of scope; generic save/load tested on the real tiny model | Primitive behavior tested |
| GLM-5 | Real tiny model tested | Out of scope: model rejects TP>1 | Real tiny model merge/unmerge tested | Native PEFT format out of scope; generic save/load tested on the real tiny model | Primitive behavior tested |
| DeepSeek V4 | Real tiny model tested | Out of scope: model rejects TP>1 | Real tiny model merge/unmerge tested | Native PEFT format out of scope; generic save/load tested on the real tiny model | Primitive behavior tested |

All five real tiny-model tests construct the actual model classes and traverse
their declared production target paths. CPU implementations replace only the
unavailable Transformer Engine kernels; no identity model, generic model spec,
or foreign model spec substitutes for a model under test. Each case proves a
LoRA-off versus LoRA-on trainable-parameter-count difference, executes every
attached adapter before and after a nonzero update, and exercises save/load,
merge/unmerge, and apply/remove. These are construction and adapter-forward
tests rather than full model-forward or GPU throughput tests.
