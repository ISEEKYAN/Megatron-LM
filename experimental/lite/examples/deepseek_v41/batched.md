# Batched DS4.1 deployment math

Enable `deployment_math=True` on an existing DS4.1 W4A8 configuration. Dense/grouped MXFP8 and sparse metadata use batched grids without host per-token launches. Ascending block-32 FP32 accumulation matches the serving BI provider; packed scale bytes are read directly. SM100 and native serving extensions are required.

```python
from dataclasses import replace

config = replace(config, deployment_math=True, quantized=True, w4a8_experts=True)
```
