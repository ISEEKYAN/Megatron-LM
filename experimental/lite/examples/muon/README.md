# Muon optimizer in Megatron Lite

Megatron Lite adds production Muon support for both **DistOpt** (Megatron-Core
distributed optimizer) and **FSDP2** backends. Matrix weights are optimized with
Muon (Newton–Schulz orthogonalized updates); embeddings, output layers, biases,
and norms fall back to AdamW under the same facade.

Upstream Megatron-Core currently routes Muon only through the distributed
optimizer path. This PR additionally lowers Muon onto FSDP2 sharded parameters
via distributed Newton–Schulz (`emerging_optimizers.newton_schulz_tp`).

## Quick start (VERL + Megatron Lite)

Set the optimizer name to `muon` in your VERL actor config. Megatron Lite maps
native Muon fields from `OptimizerConfig`:

```yaml
actor_rollout_ref:
  actor:
    optim:
      optimizer: muon
      lr: 1.0e-5
      muon_momentum: 0.95
      muon_num_ns_steps: 5
      muon_coefficient_type: quintic
      muon_fp32_matmul_prec: medium
```

Supported backends:

| Backend | Entry point | Notes |
|---------|-------------|-------|
| `dist_opt` | `build_dist_opt_stack()` | Compact LayerWise layout; bitwise parity vs Megatron-Core TensorParallelMuon |
| `fsdp2` | `build_fsdp2_training_optimizer()` | Distributed NS on DTensor shards; AdamW fallback for non-matrix params |

## Python API (direct)

```python
from megatron.lite.runtime.contracts.config import OptimizerConfig

opt = OptimizerConfig(
    optimizer="muon",
    lr=1e-4,
    muon_momentum=0.95,
    muon_split_qkv=True,
    muon_num_ns_steps=5,
)
```

DistOpt training uses `megatron.lite.primitive.optimizers.megatron_wrap.build_dist_opt_stack`.
FSDP2 training uses `megatron.lite.primitive.optimizers.fsdp2.optimizer.build_fsdp2_training_optimizer`.

Parameter routing metadata is tagged automatically before wrapping:

- `VocabParallelEmbedding` / `VocabParallelOutput` mark embedding/output weights.
- `GQAttention` marks fused QKV weights for per-head Muon splits.
- `tag_muon_parameter_metadata()` tags expert parameters.

## FSDP2 dependency

FSDP2 Muon imports `emerging_optimizers` (same package as Megatron-Core Muon).
For unit tests, set `EMERGING_OPT_SITE` to a site-packages tree containing
`emerging_optimizers`, or install the pinned Megatron emerging-optimizers wheel.

## Current limitations

Muon compact DistOpt lowering does **not** yet support:

- Padded LayerWise layout (`use_layer_wise_param_layout=True`)
- Overlap grad reduce / param gather
- FP8/FP4 param gather
- Precision-aware optimizer
- Optimizer CPU offload (use the dedicated offload lowering when available)

FSDP2 Muon does not support optimizer-state CPU offload in this release.

## Validation summary

| Check | Result |
|-------|--------|
| DistOpt Muon vs Megatron-Core | Bitwise (`torch.equal`, 2000 tensor checks, DP=2) |
| FSDP2 Muon vs reference | Within round-off (~1e-6 fp32 highest; ~1e-2 medium production precision) |
| Peak memory (30B, 16 GPUs) | Muon **39.66 GiB** < Adam **46.63 GiB** |
| GSM8K RL reward (GRPO) | Muon 0.297→0.8125 (Δ+0.516) ≥ Adam 0.266→0.750 |

## Tests

CPU unit tests (no GPU required):

```bash
PYTHONPATH="$(pwd):$(pwd)/experimental/lite" \
  pytest \
    experimental/lite/tests/unit/primitive/test_muon_routing.py \
    experimental/lite/tests/unit/primitive/test_muon_fsdp2_unit.py \
    experimental/lite/tests/unit/runtime/test_optimizer_config_contract.py
```

GPU lifecycle test (single CUDA device):

```bash
PYTHONPATH="$(pwd):$(pwd)/experimental/lite" \
  pytest experimental/lite/tests/unit/primitive/test_muon_fsdp2_offload_gpu.py
```
