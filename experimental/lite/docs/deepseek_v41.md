# DeepSeek V4.1 Flash

The MLite model composes shared primitives for CSA2, shifted mHC, Engram,
vision and object-based optimizer routing. Single-rank full-sequence training
is supported; distributed execution is being consolidated separately.

Use the released nested config with the `deepseek_v41` / `lite` registry entry.
Select post-training trainability explicitly. Indexers remain frozen without
an auxiliary loss. Engram uses frozen FP8 storage or a persistent FP32 master
with regenerated FP8 scales; tables are resident, not offloaded.

Muon handles logical linear matrices; AdamW handles norms, biases and vectors.
External vision uses model-owned parameters, weight copies before forward,
then LLM backward followed by visual backward and gradient publication.

Reference tests read official sources only from `DS41_REFERENCE_DIR` and verify
SHA-256. Missing references are errors. No official source or fixture payload
is included in this repository. Run GPU tests through the Slurm test harness.
