# Batched DS4.1 deployment math

Enable `deployment_math=True` on an existing DS4.1 W4A8 configuration. Dense/grouped MXFP8 and sparse metadata use batched grids without host per-token launches. Ascending block-32 FP32 accumulation matches the serving BI provider; the current MLite callers read unpacked E8M0 scales directly. SM100 and native serving extensions are required.

```python
from dataclasses import replace

config = replace(config, deployment_math=True, quantized=True, w4a8_experts=True)
```

Evidence and compatibility limits: the packed INT32 scale branch of the MLite
kernel is not exercised by its current callers; packed/strided scale tests
cover the serving provider. Index scores encode MXFP4 from decoded STE Q/K
state; original activation packet bytes are not retained across that boundary.

`attention_codecs()` provides `linear_grouped`. For compatibility with older
three-key attention maps, an omitted grouped key retains the original
CUDA-only operator; custom grouped behavior must supply this key explicitly.
The CPU CSA integration test supplies a grouped codec and uses CPU reference
linears elsewhere. Native arithmetic evidence is separate. Sparse backward
is checked for finiteness, not an independent bitwise VJP oracle; the grouped
serial comparison shares the batched kernel and also checks the serving
provider. These limits remain review caveats, not a full-model claim.
