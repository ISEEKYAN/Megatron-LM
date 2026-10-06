# Weight-only QAT in Megatron Lite

Megatron Lite applies weight-only fake quantization inside each model protocol.
Qwen3 MoE, Qwen3.5, DeepSeek V4, GLM-5, and Kimi K2 accept a `QATSpec`, a
matching dictionary, or `None` through `ImplConfig.qat`.

```python
from megatron.lite.model.qwen3_moe.lite.protocol import ImplConfig
from megatron.lite.primitive.quantization import QATSpec

impl_cfg = ImplConfig(
    qat=QATSpec(
        enabled=True,
        format="mxfp4",
    )
)
```

`QATSpec` defaults skip the numerically fragile output head, embedding, and
router-gate path components. The default engine YAML exposes only the two
decisions required for MXFP4: whether QAT is enabled and which format to use.

## `QATSpec` fields

| Field | Meaning |
|---|---|
| `enabled` | Explicit opt-in. `False` registers no parametrizations and leaves the model unchanged. |
| `format` | `int8`, `int4`, `fp8_e4m3`, or `mxfp4`; `fp8` canonicalizes to `fp8_e4m3`. |
| `group_size` | Integer and FP8 formats accept `0` (per-tensor), `-1` (per-output-channel), or a positive input-feature group. MXFP4 derives its fixed OCP block size `32` when omitted; an explicit conflicting value is rejected. |
| `symmetric` | Integer formats only: selects symmetric rather than affine quantization. FP8 and MXFP4 do not consume it. |
| `ste_clip` | Integer and FP8 formats: controls the saturation gradient mask. MXFP4 does not consume it because its saturation mask is always true. |
| `ignore_patterns` | Exact, case-insensitive dotted Megatron path components to skip. |
| `activation_bits` | `None` (weight-only) or `8` with `format="mxfp4"`: Qwen3 MoE routed experts run the W4A8 forward described below. Other values, and other models, raise. |

The format evidence is intentionally separated:

| Format | Implementation status | Validation status |
|---|---|---|
| `int8` | Fake quantization and packed integer conversion implemented. | CPU unit tested. |
| `int4` | Fake quantization and packed 4-bit conversion implemented. | CPU unit tested. |
| `fp8_e4m3` | E4M3 fake quantization and code conversion implemented. | CPU unit tested. |
| `mxfp4` | OCP E2M1 with one E8M0 scale per 32 weights implemented. | CPU unit tested, bitwise-checked against ModelOpt over 99,090,432 real weight elements, and exercised by end-to-end RL. |
| `nvfp4_w4a16`, `nvfp4_w4a4` | Deferred; selecting either raises `ValueError`. | Not supported by MLite QAT. |

The 99,090,432-element parity result is a one-off offline validation whose environment and run identifiers are recorded in the PR description; the repository currently has no automated assertion for it, and a small real-weight subset could be preserved as a rerunnable test in the future.

MXFP4 is the validated delivery in this change: it has ModelOpt parity over real
weights and end-to-end RL evidence. The other format code paths exist but have
not received end-to-end validation.

## W4A8 routed experts (opt-in)

`activation_bits=8` with `format="mxfp4"` makes Qwen3 MoE run its routed
experts with the operand contract of the rollout's DeepGEMM FP8 x MXFP4 MoE
path (vLLM `DeepGemmFP4Experts`). The routed-expert forward itself becomes the
quantized computation; it is not a fake-quantized BF16 GEMM:

| Step | Contract |
|---|---|
| FC1/FC2 input | Dynamic per-token FP8 E4M3, one UE8M0 scale per 128 K values: `scale = ceil_ue8m0(max(amax / 448, 1e-10))`. |
| FC1/FC2 weight | MXFP4 bytes from `quantize_mxfp4` directly on the BF16 or FP32 master (without a BF16 cast), the same bytes the exporter streams to the rollout. |
| GEMM | DeepGEMM `m_grouped_fp8_fp4_gemm_nt_contiguous` on CUDA (an error if unavailable); a CPU reference on the same dequantized operands. |
| Activation | FC1 output BF16, SwiGLU in FP32 rounded once to BF16, then requantized to FP8. With the model's `swiglu_limit` `L` (vLLM `gemm1_clamp_limit`), `gate = min(gate, L)` and `up = clamp(up, -L, L)` in FP32 before SwiGLU, so the FP8 requantization sees the clamped result. |
| Top-k combine | The unweighted BF16 FC2 rows of each token are accumulated in FP32 in router top-k slot order, `acc = fma(row, weight, acc)`, then rounded once to BF16, as vLLM's `ep_gather` does. |
| Backward | Straight-through estimator on both operands into BF16 or FP32 master weights; Decoded A8 activations first take the BF16 output-gradient dtype. For FP32 masters, those activations and upstream gradients are promoted to FP32 for the wgrad GEMM with autocast disabled; its result is not rounded to BF16. |

`Experts` constructs its grouped parameters in BF16, so the Qwen3-MoE recipe
below uses BF16 masters. `enable_w4a8_experts` selects the arithmetic path;
it does not convert parameters or create FP32 masters. The FP32 contract
applies when the caller already provides live FP32 weights to the primitive,
or explicitly converts the expert module to FP32 before enabling W4A8 and
constructing its optimizer (for example, `experts.float()`). Activations and
expert outputs remain BF16. An optimizer's separate FP32 state does not make
the live BF16 expert parameters FP32.

```python
impl_cfg = ImplConfig(
    qat=QATSpec(
        enabled=True,
        format="mxfp4",
        activation_bits=8,
        # Quantize routed experts only; keep attention linears in BF16.
        ignore_patterns=(
            "lm_head", "head", "output_layer", "gate", "router",
            "embedding", "embed", "word_embeddings", "attn",
        ),
    )
)
```

The same switch from a verl launcher:

```bash
++actor_rollout_ref.actor.engine.impl_cfg.qat.enabled=true
++actor_rollout_ref.actor.engine.impl_cfg.qat.format=mxfp4
++actor_rollout_ref.actor.engine.impl_cfg.qat.activation_bits=8
```

Limits: the activation scale is recomputed from the live tensor on every call,
so there is no observer or cross-rank amax state. The routed experts must
not use FP8 padding, `moe_act` recompute or expert LoRA; those raise. Hidden and
per-rank intermediate sizes must be multiples of 128; `build_model` rejects
other shapes (including an ETP split whose per-rank intermediate size is not a
multiple of 128) before training starts. Expert parallelism (EP>1) is also
rejected at enable time: the all-to-all return of the unreduced top-k rows has
not been validated on multiple GPUs yet. Only the Qwen3 MoE protocol wires the
W4A8 experts; the other protocols reject `activation_bits`.

**Parity with the rollout.** Measured on GB200 against vLLM's own
`DeepGemmFP4Experts` path, the routed-expert output is bit-identical to the
rollout. That covers A1/A2 codes and scales, FC1, the SwiGLU requantization,
FC2, and the top-k sum, at Qwen3-30B-A3B expert shapes with top-k 8. The
Qwen3 MoE layer, including dispatch and combine, is also checked bit for bit
on CPU against an exact transcription of `ep_gather`. Conditions:

- The combine reads the unfused dispatch layout. `build_model` rejects W4A8
  experts with `MEGATRON_LITE_MOE_PERMUTE_FUSION=1` or DeepEP.
- The router weights are taken as the training router returns them. Router
  parity (dtype and slot order) is not part of this contract.
- The rollout must use vLLM's default `VLLM_USE_DEEP_GEMM_E8M0=1`. With it
  off, vLLM quantizes activations with float32 scales, which is a different
  contract.

The default (non-W4A8) path keeps the BF16 per-row weighting and BF16 sum.

The training side only matches a rollout that serves the routed experts with
the same contract: MXFP4 expert weights and dynamic FP8 activations on vLLM's
DeepGEMM MoE backend. The four-arm script below is a W4A16 rollout
(`mxfp4-pack-quantized`, BF16 activations) and does **not** match W4A8; this
change does not provide a W4A8 rollout configuration.

## Safe training exclusions

MLite matches exact dotted path components. Its defaults are `lm_head`, `head`,
`output_layer`, `gate`, `router`, `embedding`, `embed`, and
`word_embeddings`. An empty explicit list disables every default and would
fake-quantize output heads, embeddings, and MoE router gates. Router-gate
quantization can change expert selection, so do not replace the defaults with
an empty list.

These names cannot be copied from verl's export schema. The exporter matches
HF names and accepts `"re:"` regular expressions; MLite matches Megatron module
names and does not interpret regular expressions. In particular,
`embed_tokens` is not a component of MLite's `embed.embedding` module path, and
`"re:.*mlp.gate$"` can never equal a dotted path component. Copying those
export patterns into `QATSpec` would silently fail to protect the training-side
head, embedding, and router gate.

## Construction and checkpoint ordering

Every protocol calls `apply_qat_to_chunks` after model construction and before
optimizer construction. This order is mandatory:
`torch.nn.utils.parametrize` moves the BF16 master from `...weight` to
`...parametrizations.weight.original`, and the optimizer must capture that
surviving master parameter.

HF checkpoints continue to use logical `...weight` names. `canonical_state_key`
maps those names onto the parametrized master during loading. Without the
mapping, a tensor can be silently dropped and training can proceed from random
initialization. Quantizer buffers such as
`...parametrizations.weight.0.amax` remain unchanged.

## Training and packed snapshots

There is no mode switch between fake and packed representations:

- training registers `WeightFakeQuant`; its forward calls
  `fake_quantize_weight`, returning a dequantized tensor while STE gradients
  update the original BF16 parameter;
- deployment code explicitly calls `quantize_weight`. The training step never
  calls this function.

For MXFP4, `quantize_weight` returns:

```python
{
    "qweight": uint8_e2m1_nibbles,
    "scale": uint8_e8m0_exponents,
    "format": "mxfp4",
}
```

The vLLM compatibility/refit boundary is installed from
`experimental/lite/examples/verl/verl_mlite/compat.py`.

## MLite QAT versus verl ModelOpt QAT

This comparison uses the exact verl revision from the four-arm run,
`9356c26c6cd1475bd515fa758e52cdae2c7e3613`.

| Concern | verl at `9356c26c` | MLite |
|---|---|---|
| Training fake quantization | `verl/utils/modelopt/qat_utils.py:26-37` calls `apply_qat`; `verl/utils/modelopt/quantize.py:84-92` calls `mtq.quantize`. This path requires ModelOpt at runtime. | `primitive/quantization/qat.py:650-713` implements and registers the fake-quant parametrization without importing ModelOpt. MXFP4 math is locked to ModelOpt by parity tests. |
| Formats | `quantize.py:24-42,60-81` accepts `w4a16` (NVFP4) and `mxfp4`; other names raise. | `int8`, `int4`, `fp8_e4m3`, and `mxfp4` are implemented; NVFP4 is explicitly deferred. |
| Parameters | ModelOpt converts the model in place. `qat_weight_exporter.py:161-184` reads its injected quantizer modules. | `torch.nn.utils.parametrize` preserves a BF16 `.original`, requiring `canonical_state_key` on load. |
| Export | `qat_weight_exporter.py:250-330` emits NVFP4 or MXFP4 weights and scales for its vLLM integration. | `quantize_weight` returns an explicit `qweight`/`scale`/`format` dictionary. |
| Coexistence | `apply_modelopt_fake_quant=false` tells the exporter not to expect ModelOpt training metadata. | This setting is required when MLite supplies training fake quantization. |

Some older verl revisions accept only `w4a16`; they do not describe the exact
four-arm pin above.

### Which path produced the four-arm MXFP4 rollout?

The quantized arms enabled both verl-owned QAT dictionaries:

- `actor.engine.qat` exported MXFP4 weights at synchronization time, with
  `apply_modelopt_fake_quant=false` because MLite owned training fake quant.
- `rollout.qat` configured vLLM compressed-tensors with the matching
  `mxfp4-pack-quantized` JSON and installed `verl.utils.qat.vllm_patch`.

They did **not** set `rollout.quantization=mxfp4`. That field is not the
four-arm MXFP4 route. Runtime logs selected
`CompressedTensorsW4A4Mxfp4`/Marlin through the paired exporter and rollout QAT
configuration.

## verl exporter and rollout QAT dictionaries

The engine accepts an optional, verl-owned passthrough dictionary:

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

This is not the MLite `QATSpec` schema. The exact four-arm verl pin accepts both
`mxfp4` and `w4a16`/NVFP4 here; check the selected verl revision because older
revisions are NVFP4-only.

Exporter exclusions use HF names and support `"re:"` regular expressions;
MLite training exclusions use exact Megatron dotted path components. Do not
copy either list into the other. When combining this optional exporter with
MLite training QAT, keep exporter `mode` and `group_size` consistent with the
block size implied by the training `format` (32 for MXFP4), or training and
export will use different quantization contracts. Keep the two enable switches
independent, and leave `apply_modelopt_fake_quant=false`.

## Four-arm QAT launch

[`scripts/run_qwen3moe_mxfp4_qat.sh`](scripts/run_qwen3moe_mxfp4_qat.sh)
wraps the repository's Qwen3-MoE GRPO launcher and exposes four modes:

| Mode | Training | Rollout | Purpose |
|---|---|---|---|
| `baseline` | BF16 | BF16 | Unquantized reference. |
| `qat_off` | BF16 | MXFP4 | Measures train/rollout mismatch without fake quantization. |
| `qat_on` | MXFP4 fake quant | MXFP4 | Makes training aware of rollout quantization. |
| `r3` | MXFP4 fake quant + router replay | MXFP4 | Adds router replay to `qat_on`. |

`qat_off` and `qat_on` have identical rollout arguments. Their only QAT
difference is `impl_cfg.qat.enabled`. The measured final
`rollout_probs_diff_mean` was `0.00598` for baseline, `0.0267` for `qat_off`,
and `0.0166` for `qat_on`: about 38% below `qat_off`, but about 2.8 times the
BF16 baseline.

Prepare verl-compatible parquet files from the public
`BytedTsinghua-SIA/DAPO-Math-17k` and AIME 2024 datasets, then provide their
paths. Quantized modes also require a vLLM compressed-tensors JSON. Prepare it
from the schema below and pass its path through
`MXFP4_QUANTIZATION_CONFIG`:

```json
{
  "quant_method": "compressed-tensors",
  "format": "mxfp4-pack-quantized",
  "quantization_status": "compressed",
  "config_groups": {
    "group_0": {
      "format": "mxfp4-pack-quantized",
      "targets": ["Linear"],
      "weights": {
        "actorder": null,
        "block_structure": null,
        "dynamic": false,
        "group_size": 32,
        "num_bits": 4,
        "observer": "minmax",
        "observer_kwargs": {},
        "strategy": "group",
        "symmetric": true,
        "type": "float"
      },
      "input_activations": null,
      "output_activations": null
    }
  },
  "ignore": ["lm_head", "re:.*mlp\\.gate$"],
  "kv_cache_scheme": null,
  "sparsity_config": {},
  "transform_config": {},
  "global_compression_ratio": null
}
```

This is the JSON shape used by the measured run. Validate it against the
selected compressed-tensors revision before launching. Then run:

```bash
MODEL_PATH=Qwen/Qwen3-30B-A3B \
TRAIN_FILES=/path/to/dapo-math-17k.parquet \
VAL_FILES=/path/to/aime-2024.parquet \
MXFP4_QUANTIZATION_CONFIG=/path/to/mxfp4_w4a16.json \
NNODES=4 \
NGPUS_PER_NODE=8 \
bash experimental/lite/examples/verl/scripts/run_qwen3moe_mxfp4_qat.sh \
  --mode qat_on
```

`TRAIN_FILES` and `VAL_FILES` are required.
`MXFP4_QUANTIZATION_CONFIG` is additionally required for `qat_off`, `qat_on`,
and `r3`. `MODEL_PATH`, node/GPU counts, parallelism, batch sizes, sequence
lengths, response count, step count, and output root are overridable
environment variables. Use a local model snapshot for `MODEL_PATH` if the
runtime cannot resolve the public Hub name.

The Hydra prefix forms intentionally match the measured four-arm commands:
`impl_cfg.recompute` and `impl_cfg.qat.*` use `+`, while the exporter,
rollout-QAT, cross-entropy-fusion, and R3 engine fields use `++`. Do not
normalize these prefixes without revalidating the full target config.

Use `DRY_RUN=1` to inspect the resolved command without allocating GPUs. This
checks argument construction only and is not evidence of a training run.
