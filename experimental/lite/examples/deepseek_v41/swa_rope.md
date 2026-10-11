# Native SWA KV RoPE in the deployment recipe

Enable `deployment_math=True` on the DS4.1 implementation configuration with
quantized W4A8 experts. The attention provider applies the same ordered RoPE
multiply/adds to queries and window KV. KV rounds to BF16 before the FP8 codec,
matching the receiver. The ordinary recipe retains complex multiplication.

For an existing implementation config:

```python
from dataclasses import replace

config = replace(config, deployment_math=True, quantized=True, w4a8_experts=True)
```

The CUDA production-boundary regression injects a BF16 midpoint after KV
projection and normalization, then compares the actual encoder code and scale bytes with
the native serving encoder. The optional decoder is not used. It also checks the ordinary path. CPU
coverage checks the analytic RoPE VJP and unchanged ordinary arithmetic. CUDA
coverage requires the native DS4.1 serving operators on Blackwell.
