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
        group_size=32,
        symmetric=True,
        ste_clip=True,
    )
)
```

`QATSpec` defaults skip the numerically fragile output head, embedding, and
router-gate path components. The default engine YAML repeats those defaults
explicitly so a user does not accidentally replace them with an empty list.

## `QATSpec` fields

| Field | Meaning |
|---|---|
| `enabled` | Explicit opt-in. `False` registers no parametrizations and leaves the model unchanged. |
| `format` | `int8`, `int4`, `fp8_e4m3`, or `mxfp4`; `fp8` canonicalizes to `fp8_e4m3`. |
| `group_size` | `0` is per-tensor, `-1` is per-output-channel, and a positive value groups the input-feature dimension. MXFP4 requires `32`. |
| `symmetric` | Selects symmetric rather than affine integer quantization. |
| `ste_clip` | `True` zeros gradients outside the representable range; `False` uses a pass-through STE. |
| `ignore_patterns` | Exact, case-insensitive dotted Megatron path components to skip. |

The format evidence is intentionally separated:

| Format | Implementation status | Validation status |
|---|---|---|
| `int8` | Fake quantization and packed integer conversion implemented. | CPU unit tested. |
| `int4` | Fake quantization and packed 4-bit conversion implemented. | CPU unit tested. |
| `fp8_e4m3` | E4M3 fake quantization and code conversion implemented. | CPU unit tested. |
| `mxfp4` | OCP E2M1 with one E8M0 scale per 32 weights implemented. | CPU unit tested, bitwise-checked against ModelOpt over 99,090,432 real weight elements, and exercised by end-to-end RL. |
| `nvfp4_w4a16`, `nvfp4_w4a4` | Deferred; selecting either raises `ValueError`. | Not supported by MLite QAT. |

The first three rows are implemented and unit tested, but have not received the
ModelOpt parity and end-to-end RL validation completed for MXFP4.

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

The four-arm experiment did **not** use `export_qat_weights`. Its frozen
Megatron-LM revision had `qat: {}`, and `mlite_engine.py:401-408` called the
exporter only when `qat.enable` was true. The launcher instead selected
`rollout.quantization=mxfp4`. Runtime logs selected vLLM
`CompressedTensorsW4A4Mxfp4`/Marlin and did not contain the verl exporter
patch-injection markers.

Therefore MLite supplied training fake quantization, while vLLM
compressed-tensors plus `verl.utils.qat.vllm_patch` supplied rollout-side
MXFP4. The exact verl revision also contains an optional MXFP4 online exporter,
but the experiment did not enable or exercise it.

## Alternative path: verl `export_qat_weights`

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

This is not the MLite `QATSpec` schema and is absent from the shipped defaults
apart from an inert `qat: {}` placeholder. The exact four-arm verl pin accepts
both `mxfp4` and `w4a16`/NVFP4 here; check the selected verl revision because
older revisions are NVFP4-only. The four-arm MXFP4 experiment did not use this
path.

Exporter exclusions use HF names and support `"re:"` regular expressions;
MLite training exclusions use exact Megatron dotted path components. Do not
copy either list into the other. When combining this optional exporter with
MLite training QAT, keep exporter `mode`/`group_size` consistent with training
`format`/`group_size`, keep the two enable switches independent, and leave
`apply_modelopt_fake_quant=false`.

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
paths:

```bash
MODEL_PATH=Qwen/Qwen3-30B-A3B \
TRAIN_FILES=/path/to/dapo-math-17k.parquet \
VAL_FILES=/path/to/aime-2024.parquet \
NNODES=4 \
NGPUS_PER_NODE=8 \
bash experimental/lite/examples/verl/scripts/run_qwen3moe_mxfp4_qat.sh \
  --mode qat_on
```

`TRAIN_FILES` and `VAL_FILES` are required. `MODEL_PATH`, node/GPU counts,
parallelism, batch sizes, response count, step count, and output root are
overridable environment variables. Use a local model snapshot for
`MODEL_PATH` if the runtime cannot resolve the public Hub name.

Use `DRY_RUN=1` to inspect the resolved command without allocating GPUs. This
checks argument construction only and is not evidence of a training run.
