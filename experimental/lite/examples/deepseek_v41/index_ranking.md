# DS4.1 index-ranking

For an existing DS4.1 W4A8 configuration, set `deployment_math=True`. Long-context CSA index scores use the native FP32 MXFP4 packet scorer and selector; selected KV positions are read in ascending order. Short contexts and ordinary math retain their existing paths. Use the native MXFP4 packet-score API rather than FP8-reconstructed scoring operands.

```python
from dataclasses import replace

config = replace(config, deployment_math=True, quantized=True, w4a8_experts=True)
```

The CSA state retains decoded STE Q/K values, not the original activation
packet bytes. The scorer encodes those values to MXFP4 and invokes the native
FP32 packet-score API; it does not reconstruct FP8 scoring operands. Tests
compare scores and selected positions against native packets for their stated
fixtures. Combined-prefix strict0 receipts do not prove arbitrary codec
round-trip byte identity or every untested input. Native selection requires
CUDA, index QAT, context longer than top-k, and top-k 512/1024/2048.
