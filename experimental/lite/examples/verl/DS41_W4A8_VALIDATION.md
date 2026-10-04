# DS4.1 W4A8 validation on #227

This validation branch extends #242 and its SwiGLU-clamp follow-up. It is not
an online rollout integration or a GPU parity acceptance result.

Enable the routed-expert path through the DS4.1 implementation config:

```python
ImplConfig(
    dtype=torch.bfloat16,
    quantized=True,
    w4a8_experts=True,
    use_deepep=False,
    optimizer="muon",
    optimizer_config=OptimizerConfig(
        lr=1e-4, ns_steps=2, coefficient_type="quintic"
    ),
)
```

Keep EP=1 and `MEGATRON_LITE_MOE_PERMUTE_FUSION=0`. Both the hidden size and
expert intermediate size must be divisible by 128. The sample optimizer
settings are illustrative, not an official training recipe. Shared experts,
attention, routing, checkpoint bindings and optimizer policy retain their
existing DS4.1 paths. `w4a8_experts=False` is the default.

Unlike the original #242 BF16-master consumer, DS4.1 Muon keeps FP32 masters.
The primitive quantizes each live master directly using deployment MXFP4
(group32, ties-down, zero floor); it never casts W to BF16 before quantizing.
The native exporter reads the same master. FC1 concatenates w1/gate and w3/up
by rows, preserving every group32 operand and the original parameter owners.
A1/A2 use dynamic E4M3 group128 scales; activation uses `config.swiglu_limit`;
top-k weights and slot-ordered FMA accumulation remain FP32 until the final
BF16 rounding. The CUDA backend is DeepGEMM, without a numerical fallback.

For FP32 masters, the STE weight-gradient GEMM multiplies FP32 operands with
autocast disabled and returns FP32 directly to the leaf. It does not widen an
already-rounded BF16 gradient. Existing BF16-master backward behavior remains
unchanged. This primitive extension needs to be carried back to the eventual
primitive PR; HeadwiseMuon/MixedOptimizer checks are unchanged.

`quantized=False, w4a8_experts=True` is useful for CPU diagnostics: routed
experts still execute W4A8, while non-expert projections use the existing
floating diagnostic path. It is not full native mixed-format model parity.
The pinned Core CPU mHC operation needs a test-only FP32 operand adapter for
BF16 full-model diagnostics; the product CPU path has not been repaired here.

The #227 checkpoint API does not yet provide #231's online resync entrypoint.
CPU tests compare the actual HF exporter, save/load masters, and the pinned
#231 transport implementation, including two real optimizer updates. Integrate
the approved resync delivery before launching a full actor/rollout GPU proxy.
See `evidence/1.5-W1ds41/report.md` at the repository root for exact evidence,
reference revisions, CPU boundaries and the planned 8-card GB200 gates.
