# DS4.1 index-rope

Set `deployment_math=True` on an existing DS4.1 W4A8 configuration. CSA index queries use the deployment cosine/sine multiply order before BF16 and FP4 rounding, matching the native serving indexer. Ordinary math keeps its complex rotation; the owned analytic VJP remains differentiable.

```python
from dataclasses import replace

config = replace(config, deployment_math=True, quantized=True, w4a8_experts=True)
```
