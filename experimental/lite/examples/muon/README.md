# Muon optimizer in Megatron Lite

Megatron Lite supports Muon through the **DistOpt** (Megatron-Core distributed
optimizer) path. Matrix weights use Muon (Newton–Schulz orthogonalized updates);
embeddings, output layers, biases, and norms use the upstream scalar-optimizer
fallback.

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
      muon_match_adamw_update_rms: true
```

### Learning rate: Muon's effective step is `lr x muon_extra_scale_factor`

An AdamW learning rate is **not** directly reusable. The Megatron-Core default
`muon_extra_scale_factor = 1.0` is not AdamW-comparable: carrying an AdamW `lr`
over unchanged yields roughly a **4.4x** larger effective step.

The closed form for the factor that matches AdamW's update RMS norm is:

```
muon_extra_scale_factor = sqrt((1 - beta1) / (1 + beta1))
```

where `beta1` is AdamW's first-moment coefficient—`0.229416` at
`beta1 = 0.9`. Setting `muon_match_adamw_update_rms: true` derives the factor
from `adam_beta1` and logs the resolved value on rank 0. Setting it together
with an explicit `muon_extra_scale_factor` raises.

Supported backends:

| Backend | Entry point | Notes |
|---------|-------------|-------|
| `dist_opt` | `build_dist_opt_stack()` | Compact LayerWise layout; bitwise parity vs Megatron-Core TensorParallelMuon |

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

Parameter routing metadata is tagged automatically before wrapping:

- `VocabParallelEmbedding` / `VocabParallelOutput` mark embedding/output weights.
- `GQAttention` marks fused QKV weights for per-head Muon splits.
- `tag_muon_parameter_metadata()` tags expert parameters.

## Current limitations

Muon compact DistOpt lowering does **not** yet support:

- Padded LayerWise layout (`use_layer_wise_param_layout=True`)
- Overlap grad reduce / param gather
- FP8/FP4 param gather
- Precision-aware optimizer
- Optimizer CPU offload (use the dedicated offload lowering when available)

## Validation summary

| Check | Result |
|-------|--------|
| DistOpt Muon vs Megatron-Core | Bitwise (`torch.equal`, 2000 tensor checks, DP=2) |
| GSM8K RL reward (GRPO) | Muon and AdamW are within the repeat-run spread |

This table summarizes a mix of automated primitive checks and manual
end-to-end validation. The DistOpt result is from a one-off offline DP=2 parity
run; this repository does not currently contain an automated assertion for the
2000-check TensorParallelMuon comparison. The RL observation is not asserted by
an automated test in this repository.

## Tests

CPU unit tests (no GPU required):

```bash
PYTHONPATH="$(pwd):$(pwd)/experimental/lite" \
  pytest \
    experimental/lite/tests/unit/primitive/test_muon_routing.py \
    experimental/lite/tests/unit/runtime/test_optimizer_config_contract.py
```

## Backend support

Muon is currently supported only through the MLite DistOpt path.
