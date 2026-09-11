# V4.1 logical optimizer routing

The single-rank text protocol constructs an actual Muon/Sinkhorn/AdamW stack.
It consumes module objects and validates their checkpoint bindings. Parameter
names are audit labels only. An unknown owner, alias, active indexer, unresolved
head count, unsupported backend, or missing NS configuration raises an error.
Archival vision and DSpark tensors are not trainable parameters. A future active
vision implementation must register its explicit routing and trainability mask;
there is no matrix-shaped catch-all route.

## Backend and numerical contract

Muon reuses NVIDIA `emerging-optimizers==0.3.0`, tag
[`b309e2f01cda75dc96a6dc1a2355a7b3b64b5e16`](https://github.com/NVIDIA-NeMo/Emerging-Optimizers/tree/b309e2f01cda75dc96a6dc1a2355a7b3b64b5e16).
The consumed API is `muon_utils.newton_schulz(matrix, steps, coefficient_type)`.
The audited `muon_utils.py` SHA256 is
`81319fbb67d1ec0663a9d954388696f933a6b0500e1d2cc8841469e6548ea75f`.
No NS polynomial is duplicated in MLite. NS steps and coefficient family must
be supplied explicitly; the example below is a port recipe, not a claim about
unpublished official NS settings. Each head uses the same unbatched backend
call as its independent reference, with highest FP32 matmul precision.

For each logical matrix, `M = .95 M + .05 G`, `N = .95 M + .05 G`.
The NS direction is normalized to RMS .18 (zero stays zero), followed by
decoupled decay and the group's LR. These are separate operations: the RMS
correction does not multiply weight decay. FP32 momentum is owned by the
original physical parameter; splitting creates no new parameter owners.

| Actual owner | Algorithm / logical partition | LR | Decay |
|---|---|---|---|
| Text embedding and prediction head | Sinkhorn, each complete matrix | base | 0 |
| Trainable Engram table master | Sinkhorn, complete logical table | 5x | 0 |
| Attention `wq_b` | Muon, `[heads, head_dim, q_rank]` | base | .1 |
| Attention `wq_a`, shared latent `wkv` | Muon, one matrix each | base | .1 |
| `wo_a`, `wo_b`, compressor projections, each expert projection, router, HC linear | Muon, each existing physical matrix; no new `wo_a` grouping | base | .1 |
| Engram projection | Muon, one matrix | 5x | .1 |
| Norms, including matrix-shaped Engram gains | AdamW, elementwise | base / Engram 5x | .1 |
| Learned HC bias/scale and attention sink | AdamW, elementwise | base | 0 |
| Indexer and router correction buffers | No optimizer group | — | — |

AdamW uses betas (.9, .95) and epsilon 1e-20. Sinkhorn reuses the shared
Algorithm-1 primitive: K=11, tau=1e-3, epsilon=1e-20, momentum=.95, gamma=.18,
fresh normalization on each step, no weight decay. Engram's construction-time
switch controls its persistent FP32 table master. Frozen tables retain FP8
storage and scales without allocating table optimizer state.

## Gradient and publication contract

Optimizer-enabled construction preserves FP32 parameter masters while retaining
BF16 residual computation. Floating linear and FP4 STE weight gradients use
an FP32 GEMM directly; the FP8 provider returns its FP32 weight GEMM to the
master without an intervening BF16 cast. This is a numerical correctness
provider, not a claim of TE fused accumulation. Autocast is disabled around
weight-gradient GEMMs. The post-accumulation hook exposes the native `.grad`
buffer as `main_grad`; it does not widen or copy a BF16 gradient.

The coordinator computes one global norm and clips all active groups together.
Muon/Sinkhorn stage weight and momentum candidates. AdamW uses the actual Torch
backend on private candidate parameters/state. Engram candidate storage/scales
are encoded before any backend publishes. Nonfinite inputs/candidates or a
candidate-encoding exception leave live weights, optimizer states, and table storage
unchanged. An accepted step publishes all three algorithms and Engram scales.

This local coordinator uses extra candidate storage, including an AdamW copy.
It is not a distributed/state-sharded performance implementation. TP/EP/CP/PP,
offload, and logical reassembly remain explicitly unsupported by this protocol.
Checkpoint model state together with `bundle.optimizer.state_dict()`; optimizer
restore validates owner ordering and logical matrix layouts. HF export is not
a substitute for restoring high-precision training masters.

## Use and validation

Install the pinned optional Muon dependency in the execution environment:

```sh
python -m pip install emerging-optimizers==0.3.0
```

```python
import torch
from megatron.lite.model.deepseek_v41.lite import protocol
from megatron.lite.model.deepseek_v41.lite.optimizer_groups import OptimizerConfig

bundle = protocol.build_model(
    model_config,
    impl_cfg=protocol.ImplConfig(
        device='cuda', dtype=torch.bfloat16, quantized=True,
        token_map=token_map, trainable_engram=True,
        optimizer='muon',
        optimizer_config=OptimizerConfig(
            lr=1e-3, ns_steps=5, coefficient_type='quintic', clip_grad=1.0,
        ),
    ),
)
bundle.optimizer.zero_grad()
bundle.forward_step(bundle.chunks[0], packed_batch)['loss'].backward()
successful, grad_norm, _ = bundle.optimizer.step()
```

Tests are integrated in `tests/unit/model/test_deepseek_v41_semantics.py`.
The distinct-head test compares three updates and momentum against separate
per-head NVIDIA calls, bitwise, and confirms that a whole-matrix update differs.
The native-gradient test checks two microbatches against direct FP32 products,
including a BF16 autocast context, and rejects BF16-roundtrippable gradients.
Actual-bundle tests check both Engram modes, owner coverage, live backend types,
FP32 momentum, update reachability, atomic failure, and exact resumed updates.
CUDA variants exercise the quantized model and require the repository's GPU
harness; a CPU skip does not validate them.
