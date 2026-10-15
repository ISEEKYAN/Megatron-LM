# DS4.1 W4A8 deployment math

Use the explicit `quantized=True, w4a8_experts=True, deployment_math=True`
implementation with TP/CP/PP1 and contiguous EP ownership. Routed expert
masters remain FP32 on the Muon path; forward consumes deployment-codec
views without overwriting those masters. Activations use dynamic FP8 group128,
expert weights use MXFP4 group32 with ties-down rounding, and the release
SwiGLU limit is 10. Combine top-k slots in their declared order with FP32 FMA
before the final BF16 store.

Deployment math owns the materialized vocabulary head and packed-token
objective. Its CUDA forward shares native vLLM probability arithmetic; its
reference VJP keeps the live training leaves. External fused heads and DeepEP
are rejected on this path. Freeze Engram tables explicitly for host-state
training; retain their trainable projections and normalization weights.

On SM100, index scoring follows current DeepGEMM's BF16 head weights and
BF16-rounded scores. Invalid native output tails are masked before selection.
The FP32 tensor passed to Top-K stores those rounded values. This changes the
older FP32 index-scoring contract, so evaluate selected slots and token
logprobs in the four-layer RL sequence as well as primitive tests.

See [the GRPO recipe](scripts/ds41_w4a8_proxy/README.md) for closed two/four-layer
release prefixes, CPU configuration checks, paired BI modes and independent
update/checkpoint verification. Keep all original widths and the full expert
bank in that recipe; small primitive tests do not certify the complete model.
Current-main C8 and throughput gates are pending. Historical validations on
other revisions do not establish current-main correctness or performance.
