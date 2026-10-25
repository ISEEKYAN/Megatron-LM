
### Fused MegaMoE LoRA hooks

`ImplConfig(fused_moe_lora=True)` selects DeepGEMM's native LoRA epilogue rather
than adding a delta to an already rounded BF16 base. It requires an immutable
release-native base, the routed and all three shared-expert adapter targets,
and equal ranks divisible by eight, up to 64. The default remains `False` while
the new path is being validated. The serving reference is
`ISEEKYAN/vllm@dcc4f82cc484104bd2dee4f218edfa4dc58bb1be` with
`ISEEKYAN/DeepGEMM@9f9f78ff7e638c6c365b82c0549d0dfd1e41ff28`.

The native hook adds the gate/up delta to the FP32 L1 accumulator before BF16
rounding; the exported post-activation rows feed the serving down-delta kernels
and one final addition per token. Release weights are packed temporarily per
layer, without a model-wide decoded or second packed cache. Serving kernels are
imported lazily, not copied.

Backward uses the existing grouped decoded-operand replay. Because the native
hook does not export its pre-SwiGLU accumulator, replay uses BF16 base-plus-delta
and is an **approximate adjoint**, not an exact VJP of the fused accumulator.
CPU layout and wiring tests do not establish native GPU forward parity; the
new stack still requires forward bit-string and finite-gradient gates.
