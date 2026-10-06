# DS4.1 W4A8 model consumption

This consumer is stacked on deployment math (#244) and includes the
explicit #242 W4A8 and #231/#238 export prerequisites in its tree. Online
export/reload remains owned by #231; it is not a generic QAT export path.
Default construction keeps the existing model arithmetic.

```python
import json
import torch
from megatron.lite.model.deepseek_v41.config import DeepseekV41Config
from megatron.lite.model.deepseek_v41.lite.protocol import ImplConfig, build_model
from megatron.lite.runtime.contracts import ParallelConfig
from megatron.lite.model.deepseek_v41.vision_config import OptimizerConfig

# Use a local HF config and the ordinary model-owned HF weight loader.
with open("config.json") as f:
    config = DeepseekV41Config(json.load(f))
impl = ImplConfig(
    device="cuda", dtype=torch.bfloat16, quantized=True,
    w4a8_experts=True, deployment_math=True, use_deepep=False,
    parallel=ParallelConfig(ep=8), optimizer="muon",
    optimizer_config=OptimizerConfig(lr=1e-6, ns_steps=2, coefficient_type="quintic"),
)
bundle = build_model(config, impl_cfg=impl)
```

Run with the normal distributed launcher, TP/CP/PP=1, EP=4 or 8 and
`MEGATRON_LITE_MOE_PERMUTE_FUSION=0`. Residuals remain BF16; optimizer-owned
routed expert leaves remain FP32 on this explicit Muon path before
deployment-codec quantization. The w4a8_experts flag alone selects arithmetic,
not parameter dtype: an optimizer-free model retains its constructed BF16
leaves. A8
uses group128, and expert K dimensions must be divisible by128. Keep the
model's SwiGLU limit (10 for the accepted recipe) and FP32 top-k slot order.
EP ownership is contiguous with global expert IDs; returned rows are combined
after transport. No expert is frozen or removed.

For a closed text prefix, retain the chosen prefix's original hidden width,
HC copies and expert bank. Set `candidate_source_layer_id=-1`, keep only
KV/index owners inside the prefix, and retain the topology's three appended
DSpark entries. A disabled publisher cannot serve ratio1 reuse. The final
text layer owns the real head, rather than returning an unfinished pair.
Use the original release index_topk for a real prefix; tiny tests may choose
a smaller value explicitly. Deployment math is text-only and rejects an
external fused head, DeepEP and TP/CP/PP>1.

Tests cover F32 leaves/gradients, two real Muon steps, independent FP4/A8
references, global EP ownership, and closed-prefix checkpoint mirrors.
A separate stacked packed-head/lifecycle PR is required for actual GRPO.
Prior GB200 validation covered EP4/EP8 and a real 2-layer/hidden5120/hc4/
384-expert prefix. Full release depth/MTP, TP>1 parity, long-sequence
performance and checkpoint restore after GRPO are not established.

Library/model added-line counts are reported without netting deletions in
the PR body; line-count compression is deferred to a later change.
