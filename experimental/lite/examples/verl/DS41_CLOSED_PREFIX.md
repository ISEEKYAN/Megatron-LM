# Closed DS4.1 text prefix

The ordinary DS4.1 model can close a text prefix before the candidate
publisher: set `candidate_source_layer_id=-1`, retain the prefix width,
expert bank and three appended topology entries, and keep KV/index owners
inside the prefix. Ratio-1 reuse cannot be used without a publisher. The
last text layer executes its real output head; checkpoint load/export remain
mirrored.

```python
import json
import torch
from megatron.lite.model.deepseek_v41.config import DeepseekV41Config
from megatron.lite.model.deepseek_v41.lite.protocol import ImplConfig, build_model
from megatron.lite.model.deepseek_v41.vision_config import OptimizerConfig

with open("config.json") as f:
    config = DeepseekV41Config(json.load(f))
impl = ImplConfig(
    device="cuda", dtype=torch.bfloat16, quantized=True, optimizer="muon",
    optimizer_config=OptimizerConfig(lr=1e-6, ns_steps=2, coefficient_type="quintic"),
)
bundle = build_model(config, impl_cfg=impl)
```

Use the ordinary model-owned HF loader before training. The config must
already describe the desired closed prefix. Optimizer settings are explicit;
`optimizer="muon"` requires `OptimizerConfig`. This PR does not provide
W4A8/deployment-math model consumers; those belong to a dependent PR after
their generic providers.
