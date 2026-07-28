# Weight-only QAT in Megatron Lite

Megatron Lite applies weight-only fake quantization inside each model protocol.
The configuration surface is `ImplConfig.qat`; Qwen3 MoE, Qwen3.5,
DeepSeek V4, GLM-5, and Kimi K2 accept either a `QATSpec`, a matching
dictionary, or `None`.

For example, Qwen3 MoE training can opt into MXFP4 QAT with:

```python
from megatron.lite.model.qwen3_moe.lite.protocol import ImplConfig
from megatron.lite.primitive.quantization import QATSpec

impl_cfg = ImplConfig(
    qat=QATSpec(
        enabled=True,
        format="mxfp4",
        group_size=32,
        symmetric=True,
        ste_clip=True,
        ignore_patterns=("lm_head", "head", "gate", "router", "embedding", "embed"),
        export_mode="fake",
    )
)
```

The same fields can be supplied as YAML/Hydra data under `impl_cfg.qat`.

Rollout export has a separate, verl-owned configuration under the engine's
top-level `qat` key:

```yaml
qat:
  enable: true
  apply_modelopt_fake_quant: false
  mode: mxfp4
  group_size: 32
  ignore_patterns:
    - lm_head
    - embed_tokens
    - "re:.*mlp.gate$"
```

These three export ignore patterns are required for this MLite path. In the
verl exporter used by this integration, a missing or empty `ignore_patterns`
value becomes an empty list; there is no built-in export ignore set.
Consequently, the HF-stream exporter otherwise quantizes every eligible
`.weight` tensor other than norms, including the output head, embeddings, and
router gate. Literal export patterns use substring/glob matching, and a
`"re:"` prefix selects regular-expression matching.

Do not confuse this with the training-side `QATSpec.ignore_patterns`.
`QATSpec` has its own built-in defaults for embedding, output-head, and router
path components and matches exact, case-insensitive dotted path components.
Changing the training list does not change verl's export list, or vice versa;
configure both owners explicitly as shown above.

## `QATSpec` fields

| Field | Meaning |
|---|---|
| `enabled` | Explicit opt-in. `False` registers no parametrizations and leaves the model unchanged. |
| `format` | Quantization contract: `int8`, `int4`, `fp8_e4m3`, or `mxfp4`. The alias `fp8` canonicalizes to `fp8_e4m3`. |
| `group_size` | `0` is per-tensor, `-1` is per-output-channel, and a positive value groups the input-feature dimension. MXFP4 requires the OCP block size `32`. |
| `symmetric` | Selects symmetric rather than affine integer quantization. Floating-point formats do not use an integer zero point. |
| `ste_clip` | `True` zeros gradients outside the representable range; `False` uses a pure pass-through straight-through estimator. |
| `ignore_patterns` | Exact, case-insensitive dotted path components to skip. Defaults exclude embeddings, output heads, and router gates. |
| `export_mode` | `"fake"` identifies the training representation; `"packed"` identifies a deployment snapshot request. |

The supported and deferred formats are:

| Format | Status | Notes |
|---|---|---|
| `int8` | supported | Weight-only integer fake quantization and packed integer export. |
| `int4` | supported | Weight-only integer fake quantization and packed 4-bit export. |
| `fp8_e4m3` | supported | E4M3 fake quantization and FP8-code export. |
| `mxfp4` | supported | OCP E2M1 values with one E8M0 scale per 32 weights. |
| `nvfp4_w4a16`, `nvfp4_w4a4` | deferred | Both names are in `qat.py::_DEFERRED_FORMATS`; selecting either raises `ValueError`. NVFP4 needs a separate scale and serializer contract and must not reuse the deleted recipe table. |

## Construction and checkpoint ordering

Every protocol calls `apply_qat_to_chunks` after model construction and before
optimizer construction. This order is mandatory: parametrization moves the
trainable BF16 parameter from `...weight` to
`...parametrizations.weight.original`, and the optimizer must capture that
master parameter. Applying QAT after optimizer construction can leave the
optimizer holding the wrong parameter set.

For the same reason, checkpoint readers canonicalize a logical
`...weight` key onto `...parametrizations.weight.original` when QAT is active.
The fake-quantized view is derived during forward; it is not a checkpoint
master weight.

## Training and packed deployment snapshots

Use `export_mode="fake"` while training. The parametrized forward computes a
dequantized fake weight and the optimizer updates the original BF16 weight.

Use `export_mode="packed"` when requesting deployment material, then call
`quantize_weight` on the master weight. For MXFP4 the result is:

```python
{
    "qweight": uint8_e2m1_nibbles,
    "scale": uint8_e8m0_exponents,
    "format": "mxfp4",
}
```

Packed weights never enter the optimizer or training forward. The VERL engine
hands the exported stream to its rollout exporter, while the vLLM compatibility
and refit boundary is installed from
`experimental/lite/examples/verl/verl_mlite/compat.py`.

## Runnable CPU example

The example below runs one MXFP4 fake-quantized optimizer step on a tiny
Qwen3-MoE-shaped block, verifies that the optimizer owns `weight.original`,
and creates a packed MXFP4 snapshot:

```bash
PYTHONPATH=experimental/lite \
python experimental/lite/examples/verl/scripts/qat_mxfp4_minimal.py
```

This is a CPU contract example, not a throughput or convergence benchmark.
Production model construction passes the same `QATSpec` through
`ImplConfig.qat`.
