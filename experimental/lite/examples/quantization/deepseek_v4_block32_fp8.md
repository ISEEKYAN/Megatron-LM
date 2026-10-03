# Block32 FP8 attention projections with native FP32 wgrad (DeepSeek-V4)

`megatron.lite.primitive.modules.native_fp32_linear` provides opt-in projection
providers with a persistent FP32 master weight:

| provider | forward | weight gradient |
|---|---|---|
| `"default"` | `torch.nn.Linear` itself (unchanged BF16 path) | autograd default |
| `"native_fp32"` | activation-dtype GEMM on the FP32 master cast to the activation dtype | FP32 GEMM of FP32 operands, returned as FP32 |
| `"block32_fp8"` | `dynamic_fp8_linear`: E4M3 operands, UE8M0 scales on activation rows `(1, 32)` and weight tiles `(32, 32)`, one `torch._scaled_mm` per 32-wide K block, FP32 scale correction and accumulation | FP32 GEMM of the decoded activation and FP32 grad |

The codec is `quantize_block_fp8(..., scale_format="e8m0")` from
`primitive/quantization/block_fp8.py`; no new rounding rule is introduced.

## DeepSeek-V4

The provider replaces CSA's `wq_a`, `wq_b`, `wkv` and `wo_b` in every decoder
and MTP layer. `wo_a`, the compressor and the indexer are unchanged.

```python
from megatron.lite.model.deepseek_v4.lite.protocol import ImplConfig, build_model

impl_cfg = ImplConfig(
    hf_path=hf_path,
    optimizer=optimizer_config,          # dist_opt
    attention_linear="block32_fp8",      # default: "default"
)
bundle = build_model(model_cfg, impl_cfg=impl_cfg)
```

`build_model` casts the model to BF16 as before and then restores the provider
weights to FP32 masters (`restore_fp32_masters`). Parameter names are
unchanged, so HF load and export use the same keys.

Direct use in another model:

```python
from megatron.lite.primitive.modules.native_fp32_linear import (
    linear_provider,
    restore_fp32_masters,
)

linear = linear_provider("block32_fp8")
proj = linear(4096, 1024, bias=False)        # in/out divisible by 32
model = restore_fp32_masters(model.to(torch.bfloat16).cuda())
```

To separate codec error from FP8 GEMM error, call
`dynamic_fp8_linear(x, weight, diagnostic=True)`: the same block32 codec,
followed by one FP32 GEMM on the decoded operands. It also runs on CPU.

## Rejected combinations

- unknown provider names; bias; in/out features not divisible by 32 for `block32_fp8`
- non-FP32 masters for native wgrad; activation/weight dtype mismatch unless the weight is the FP32 master
- `block32_fp8` forward on CPU (no CPU GEMM fallback)
- DeepSeek-V4 `attention_linear != "default"` with weight QAT enabled (double quantization) or with `optimizer="fsdp2"` (mixed-dtype parameters)

## Limitations

- The FP8 GEMM is a correctness provider: K/32 separate `_scaled_mm` calls, not a fused blockwise kernel.
- With FP32 masters, HF export emits these four weights as FP32 unless the caller passes `export_dtype="bf16"`. The block-FP8 resync quantizes from FP32 directly.
- Random initialization still passes through the module-wide BF16 cast before the FP32 restore; weights loaded from HF are unaffected.
- Real CUDA `_scaled_mm` has GPU numerical coverage. The `dist_opt` mixed BF16/FP32 parameter path has not been validated on GPU.
