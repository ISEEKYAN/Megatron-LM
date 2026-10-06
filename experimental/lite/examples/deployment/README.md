# Opt-in deployment arithmetic with owned reference VJPs

The optional providers expose RMS/QKV normalization, router slot selection,
RoPE/sparse attention, grouped output projection, compressor, Engram and
Mega-mHC visible arithmetic. The default model/primitive path is unchanged;
set `deployment_math=True` only for a supported consumer. DS4.1 model wiring
is supplied separately; generic W4A8 experts are supplied by #242.

The CUDA forward consumes live parameters and MLite-owned buffers. It may
load generic vLLM kernel providers, but never constructs a rollout engine or
imports the legacy layers.batch_invariant initialization. Paired Q/KV
reduction width is explicit and opt-in. Owned compressor/Engram/RMS kernels
are in the primitive layer, not in a model-specific replacement.

The backward computes a declared Torch reference VJP under disabled autocast.
FP32 parameter leaves and their gradient dtype survive visible BF16 decoding;
this is an STE/reference contract, not a derivative of the quantization kernel.
Higher derivatives are unsupported. Shape/dtype/provider violations fail
rather than silently changing the CUDA arithmetic. The accepted deployment
consumer is text-only, BF16 residuals, TP/CP/PP=1, native unfused EP transport;
ordinary non-deployment consumers retain their previous configurations.

Run a small adjacent normalization/master example:

```bash
PYTHONPATH=experimental/lite python \
  experimental/lite/examples/deployment/rms_vjp.py --device cpu
```

Use `--device cuda` only with the compatible SM100 provider build. The CPU
example validates the reference/master contract, not CUDA bitwise parity.
The deployment tests also compose normalization, projection, RoPE, attention,
compressor, Engram and hyper-connection providers and compare their VJPs to
independent references. Default-off tests compare forward and every gradient
with a pinned prior CSA implementation; missing reference source is an error.

The prior GB200 integration passed an 85-case combined kernel gate and
EP4/EP8 end-to-end comparisons. Those are integration results, not 85 tests
unique to this PR. New CPU skips do not represent CUDA coverage. Full-depth
release, image deployment, TP>1 deployment parity and performance are outside
this evidence.
