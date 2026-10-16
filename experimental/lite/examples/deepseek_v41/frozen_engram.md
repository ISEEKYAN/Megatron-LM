# Frozen Engram deployment

Use `trainable_engram=False` to keep Engram tables out of optimizer ownership. Set `resync_config={"freeze_engram": True}` for rollout transfers: the first generation sends tables and later generations retain them. Tables must have no FP32 masters. The receiver validates census and rejects changes or inconsistent freeze mode; sleep/offload retains the frozen allocation. Saved HF checkpoints remain standalone and include tables. Ordinary resync still sends all tensors.

```python
from dataclasses import replace

config = replace(config, trainable_engram=False)
resync_config = {"freeze_engram": True}
```
