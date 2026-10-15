# Segmented host-backed optimizer

For an existing DS4.1 OptimizerConfig set `segmented_host=True`. FP32 authoritative masters and Muon/AdamW moments remain on CPU; one owner is staged at a time. A complete validation pass and global vote precede publication, including dense and pipeline replicas. A rejected candidate changes no weight or moment; device failure during publication poisons the optimizer. Native DCP checkpoints use host parameters and restore CPU state. This trades extra copies and a second update pass for bounded GPU temporary storage. Frozen Engram tables have no optimizer owner.

```python
from dataclasses import replace

optimizer_config = replace(optimizer_config, segmented_host=True)
```
