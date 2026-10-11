# DS4.1 index-ranking

For an existing DS4.1 W4A8 configuration, set `deployment_math=True`. Long-context CSA index scores use the native FP32 MXFP4 packet scorer and selector; selected KV positions are read in ascending order. Short contexts and ordinary math retain their existing paths. The original quantized packets are required; reconstructing them from dequantized values changes ranking.

```python
from dataclasses import replace

config = replace(config, deployment_math=True, quantized=True, w4a8_experts=True)
```
