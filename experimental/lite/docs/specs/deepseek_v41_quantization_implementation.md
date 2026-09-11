# V4.1 quantization interfaces and independent cards

The input to these cache codecs is the full vector **after RoPE**. Production
call-site ordering belongs to D2; isolated codec tests cannot prove that wiring.
The main cache uses E2M1 with 16-value groups and E4M3 scales, without a second
global scale. The indexer uses E2M1 with 32-value groups and E8M0 scales.

`quantize_main_kv(post_rope)` and `quantize_index(post_rope)` return packed I8
codes, encoded scales, and floating decoded values. Neither mutates the input.
`fake_quant_main_kv(x, enabled=True, phase="post-training")` and
`fake_quant_index(x, enabled=True)` have separate switches. Main QAT rejects
pretraining. Under resolved O14, enabled forward returns the decoded value and
backward returns the incoming floating gradient unchanged; scales are detached.
Disabled operator diagnostics preserve input dtype and values under O15. No
indexer objective or Engram scale regeneration is selected (O12/O08/O09 inactive).

For scale 1, positive midpoints `[.25,.75,1.25,1.75,2.5,3.5,5]` round to
codes `[0,2,2,4,4,6,6]` (ties to even). Negative values add sign bit 8,
including negative zero. Main scale 1 is encoded E4M3 byte 56; index scale 1
is E8M0 byte 127. Zero main groups have scale 2^-9, encoded byte 1; zero
index groups have scale 2^-126, encoded byte 1. Include a maximum 6 in each
midpoint group to fix its scale independently of the tested scale selection.

Discriminating format card: constant 6.25 has main scale 1 and decoded 6;
the index scale is 2 and decoded 6. Identical decoded values alone do not
prove the formats: assert the distinct scale codes and group shapes.

Validation steps: first run missing-module tests in
`tests/unit/deepseek_v41/test_{kv,index}_quantization.py`, implement each codec,
then run exact codes/scales/values plus nonuniform input gradients and independent
switch combinations. Execute the pinned official kernels via Slurm separately;
CPU cards and identity-gradient policy do not certify GPU parity.

SWA applies FP8 to the full post-RoPE vector in groups of 32 with E8M0 scales.
Linear independently quantizes each activation row in groups of 32 and each
weight matrix in 32x32 blocks. `dynamic_fp8_linear(x, weight)` accepts floating
training tensors, performs actual native FP8 `_scaled_mm` per K block, applies
detached per-row/per-block scales, and accumulates in FP32 before casting to the
input dtype. This correctness implementation is unfused across K blocks.
Its O14 backward uses decoded floating operands for input and weight derivatives.
It requires CUDA and never silently replaces the forward with BF16 GEMM.

Independent FP8 card: a block containing `.265625`, `.296875` and maximum `448`
has scale 1 (E8M0 byte 127), rounding those first values to `.25` and `.3125`.
All-zero groups use scale 2^-22 (E8M0 byte 105). SWA and Linear policies have
separate entry points even though their present activation codecs agree.
