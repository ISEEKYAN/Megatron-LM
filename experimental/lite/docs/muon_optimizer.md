# Muon optimizer integration study

## Executive summary

Muon is an optimizer algorithm, while `dist_opt`, PyTorch FSDP2, and
Megatron-FSDP are parameter-placement and communication strategies. They should
be selectable independently in the user-facing configuration. They are not,
however, implementation-independent: Muon couples every element of a logical 2-D
update during orthogonalization, so a sharding backend must provide a correct way
to execute that whole-matrix operation.

The main findings are:

- NVIDIA Megatron-LM currently supports Muon with regular DDP and with a
  specialized layer-wise distributed optimizer. It does **not** currently
  support Muon with PyTorch FSDP2 or Megatron-FSDP.
- Megatron routes non-embedding 2-D parameters to Muon and routes embeddings,
  output weights, biases, normalization weights, and other non-2-D parameters to
  Adam. Muon and Adam therefore coexist in one chained optimizer.
- Megatron's distributed solution does not run Newton--Schulz on arbitrary
  byte-level ZeRO-1 shards. It assigns each complete Muon matrix to one data
  parallel rank, performs Newton--Schulz on that owner, and then synchronizes the
  updated parameters. Non-Muon parameters still use the standard byte-level
  distributed optimizer.
- Megatron Lite already has two conceptual configuration axes:
  `ImplConfig.optimizer` chooses `dist_opt` or `fsdp2`, while
  `OptimizerConfig.optimizer` currently chooses `adam`. The first field should
  be renamed to `optimizer_backend` (with a compatibility alias), and both
  backends should consume a common algorithm/parameter-routing specification.
- For a first FSDP2 implementation, the safest lowering is to keep Muon momentum
  sharded, all-gather the pre-orthogonalization update for one matrix, run the
  same Newton--Schulz calculation on every FSDP rank, and slice the result back to
  the local shard. A distributed Newton--Schulz implementation can replace this
  later without changing the public optimizer configuration.

This document is a source study and design proposal. No implementation or GPU
validation was performed.

## Source baseline and terminology

This study uses:

- NVIDIA Megatron-LM `dev` at
  [`d64ba4ccb1e3e878c15171c9cc58d5d3b46bf4d5`](https://github.com/NVIDIA/Megatron-LM/tree/d64ba4ccb1e3e878c15171c9cc58d5d3b46bf4d5),
  fetched on 2026-07-09.
- NVIDIA-NeMo Emerging-Optimizers v0.3.0 at
  [`b309e2f01cda75dc96a6dc1a2355a7b3b64b5e16`](https://github.com/NVIDIA-NeMo/Emerging-Optimizers/tree/b309e2f01cda75dc96a6dc1a2355a7b3b64b5e16).
  Megatron pins this tag in
  [`pyproject.toml:203`](https://github.com/NVIDIA/Megatron-LM/blob/d64ba4ccb1e3e878c15171c9cc58d5d3b46bf4d5/pyproject.toml#L203).
- The Megatron Lite checkout at `69ea18d07`. It has no Muon implementation.

In this document:

- **DistOpt / ZeRO-1** means optimizer state and main-parameter/gradient update
  shards across the data-parallel group, followed by parameter all-gather. Model
  parameters are logically replicated when used by forward/backward.
- **FSDP2 / ZeRO-3** means parameters, gradients, and optimizer state remain
  sharded between computations. Parameters are all-gathered just in time for a
  module and gradients are reduce-scattered after backward.
- **TP sharding** is independent of DP/FSDP sharding. A logical matrix can be
  sharded by TP and by the data-parallel strategy at the same time.

The current upstream support matrix is:

| Algorithm | DDP, no optimizer sharding | MCore DistOpt | PyTorch FSDP2 | Megatron-FSDP |
| --- | --- | --- | --- | --- |
| AdamW | Supported | Supported | Supported | Supported |
| Muon | Supported | Supported through LayerWise + DistOpt | Rejected | Rejected |

The rejection is explicit in
[`arguments.py:1872-1894`](https://github.com/NVIDIA/Megatron-LM/blob/d64ba4ccb1e3e878c15171c9cc58d5d3b46bf4d5/megatron/training/arguments.py#L1872-L1894).
This matters because it prevents us from describing an upstream FSDP2 path that
does not exist.

## What Megatron's Muon implementation does

### Source map

The implementation is split across several layers instead of living in one
`muon.py` file:

| Responsibility | Source |
| --- | --- |
| Momentum, weight decay, and optimizer step | Emerging-Optimizers [`orthogonalized_optimizer.py:126-201`](https://github.com/NVIDIA-NeMo/Emerging-Optimizers/blob/b309e2f01cda75dc96a6dc1a2355a7b3b64b5e16/emerging_optimizers/orthogonalized_optimizers/orthogonalized_optimizer.py#L126-L201) |
| Newton--Schulz and TP-aware variants | Emerging-Optimizers [`muon_utils.py:120-320`](https://github.com/NVIDIA-NeMo/Emerging-Optimizers/blob/b309e2f01cda75dc96a6dc1a2355a7b3b64b5e16/emerging_optimizers/orthogonalized_optimizers/muon_utils.py#L120-L320) |
| Update scaling | Emerging-Optimizers [`muon.py:136-162`](https://github.com/NVIDIA-NeMo/Emerging-Optimizers/blob/b309e2f01cda75dc96a6dc1a2355a7b3b64b5e16/emerging_optimizers/orthogonalized_optimizers/muon.py#L136-L162) |
| Megatron TP and fused-QKV adapter | Megatron [`emerging_optimizers.py:160-291`](https://github.com/NVIDIA/Megatron-LM/blob/d64ba4ccb1e3e878c15171c9cc58d5d3b46bf4d5/megatron/core/optimizer/emerging_optimizers.py#L160-L291) |
| Parameter routing and optimizer construction | Megatron [`emerging_optimizers.py:81-130`](https://github.com/NVIDIA/Megatron-LM/blob/d64ba4ccb1e3e878c15171c9cc58d5d3b46bf4d5/megatron/core/optimizer/emerging_optimizers.py#L81-L130) and [`optimizer/__init__.py:725-972`](https://github.com/NVIDIA/Megatron-LM/blob/d64ba4ccb1e3e878c15171c9cc58d5d3b46bf4d5/megatron/core/optimizer/__init__.py#L725-L972) |
| Whole-matrix ZeRO-1 ownership | Megatron [`layer_wise_optimizer.py:37-103`](https://github.com/NVIDIA/Megatron-LM/blob/d64ba4ccb1e3e878c15171c9cc58d5d3b46bf4d5/megatron/core/optimizer/layer_wise_optimizer.py#L37-L103) and [`layer_wise_optimizer.py:293-543`](https://github.com/NVIDIA/Megatron-LM/blob/d64ba4ccb1e3e878c15171c9cc58d5d3b46bf4d5/megatron/core/optimizer/layer_wise_optimizer.py#L293-L543) |
| CLI validation and defaults | Megatron [`arguments.py:3213-3280`](https://github.com/NVIDIA/Megatron-LM/blob/d64ba4ccb1e3e878c15171c9cc58d5d3b46bf4d5/megatron/training/arguments.py#L3213-L3280) |

`megatron/core/optimizer/muon.py` is only a backward-compatible getter; it is not
the algorithm implementation.

### Algorithm, step by step

For a parameter matrix `P`, gradient `G`, momentum `M`, learning rate `lr`, and
momentum coefficient `beta`, the non-adaptive Muon step is:

1. Apply weight decay. Megatron selects decoupled weight decay by default.
2. Update EMA momentum: `M <- beta * M + (1 - beta) * G`.
3. If Nesterov is enabled, use `(1 - beta) * G + beta * M`; otherwise use `M`.
4. Normalize the matrix and apply Newton--Schulz iterations to approximate its
   zeroth power, i.e. an orthogonal/polar factor.
5. Scale the orthogonalized matrix according to the logical matrix shape and an
   optional extra multiplier.
6. Update `P <- P - lr * orthogonalized_update`.

The momentum and Nesterov calculation is visible in
[`orthogonalized_optimizer.py:173-199`](https://github.com/NVIDIA-NeMo/Emerging-Optimizers/blob/b309e2f01cda75dc96a6dc1a2355a7b3b64b5e16/emerging_optimizers/orthogonalized_optimizers/orthogonalized_optimizer.py#L173-L199).

For one Newton--Schulz iteration with coefficients `(a, b, c)`, the implementation
computes:

```text
A  = X X^T
B  = b A + c A^2
X' = a X + B X
```

The input is first Frobenius-normalized, transposed when beneficial, and is
converted to BF16 for the matrix multiplications when FP32 matmul precision is
`medium`; the result is returned in FP32. The default quintic coefficient
sequence and loop are in
[`muon_utils.py:30-44`](https://github.com/NVIDIA-NeMo/Emerging-Optimizers/blob/b309e2f01cda75dc96a6dc1a2355a7b3b64b5e16/emerging_optimizers/orthogonalized_optimizers/muon_utils.py#L30-L44)
and
[`muon_utils.py:176-225`](https://github.com/NVIDIA-NeMo/Emerging-Optimizers/blob/b309e2f01cda75dc96a6dc1a2355a7b3b64b5e16/emerging_optimizers/orthogonalized_optimizers/muon_utils.py#L176-L225).

This differs from AdamW in the important ways below:

| Property | AdamW | Muon |
| --- | --- | --- |
| First-order state | Per-element first moment | Per-element matrix-shaped momentum |
| Second-order state | Per-element second moment | None in base Muon |
| Update coupling | Elementwise after reductions | Every element depends on the logical 2-D matrix through `X X^T` |
| Valid local-shard step | Yes | Only if the shard is the intended orthogonalization domain, or communication reconstructs the logical domain |
| Typical parameters | All trainable params, with WD groups | Hidden 2-D weights only; scalar/vector/embedding/output params use AdamW |

Adaptive Muon is a separate option that adds AdamW- or NorMuon-style second
moment processing after orthogonalization; it should not be conflated with base
Muon.

### Which parameters use Muon

Megatron's effective default predicate is:

```text
Muon:  param.ndim == 2 and not embedding/output
Adam:  everything else
```

The predicate is defined at
[`emerging_optimizers.py:128-130`](https://github.com/NVIDIA/Megatron-LM/blob/d64ba4ccb1e3e878c15171c9cc58d5d3b46bf4d5/megatron/core/optimizer/emerging_optimizers.py#L128-L130)
and registered at
[`emerging_optimizers.py:429-455`](https://github.com/NVIDIA/Megatron-LM/blob/d64ba4ccb1e3e878c15171c9cc58d5d3b46bf4d5/megatron/core/optimizer/emerging_optimizers.py#L429-L455).
Consequently:

- dense linear and expert linear weights normally use Muon;
- embeddings and output projections use Adam even though they are 2-D;
- biases, LayerNorm/RMSNorm weights, and other vectors use Adam;
- fused QKV weights are optionally split into Q/K/V matrices before
  orthogonalization;
- expert matrices use the expert TP process group, while dense matrices use the
  regular TP group.

The declared `muon_scalar_optimizer` option accepts `adam` or `lion`, but at the
fixed upstream commit it is not consumed by optimizer construction: the
registered fallback is hard-coded to `adam`. The config field is declared at
[`optimizer_config.py:287-289`](https://github.com/NVIDIA/Megatron-LM/blob/d64ba4ccb1e3e878c15171c9cc58d5d3b46bf4d5/megatron/core/optimizer/optimizer_config.py#L287-L289).
An MLite design should not copy this inactive knob without either wiring it or
rejecting non-Adam values.

### Hyperparameter conventions

The effective training CLI defaults are:

| Setting | CLI default | Meaning |
| --- | --- | --- |
| `lr` | General optimizer setting | Applied after orthogonalization |
| `weight_decay` | General optimizer setting | Decoupled by default |
| `muon_momentum` | `0.9` | EMA coefficient for the internal SGD momentum |
| `muon_nesterov` | `False` | Use Nesterov-style momentum before NS |
| `muon_split_qkv` | `True` | Orthogonalize fused Q, K, and V blocks separately |
| `muon_scale_mode` | `spectral` | Multiply by `sqrt(max(fan_out, fan_in))` |
| `muon_fp32_matmul_prec` | `medium` | BF16 tensor-core work inside NS, FP32 result |
| `muon_coefficient_type` | `quintic` | Polynomial coefficient sequence |
| `muon_num_ns_steps` | `5` | Newton--Schulz iterations |
| `muon_tp_mode` | `blockwise` | Orthogonalize each local TP block independently |
| `muon_extra_scale_factor` | `1.0` | Additional update multiplier |

There is a default mismatch worth making explicit: the standalone MCore
`OptimizerConfig` declares `muon_momentum=0.95` at
[`optimizer_config.py:259-285`](https://github.com/NVIDIA/Megatron-LM/blob/d64ba4ccb1e3e878c15171c9cc58d5d3b46bf4d5/megatron/core/optimizer/optimizer_config.py#L259-L285),
while the training CLI declares `0.9`. MLite should use one explicit value in its
own config instead of relying on entry-point-dependent defaults.

The TP mode controls a second, independent matrix boundary:

- `blockwise`: treat the local TP shard as the Muon matrix; no TP collective is
  introduced by NS.
- `duplicated`: all-gather TP shards, run full-matrix NS redundantly on all TP
  ranks, then keep the local output chunk.
- `distributed`: keep the matrix sharded, all-reduce the normalization scalar and
  the Gram matrix once per NS iteration. This avoids a full-matrix materialization
  but can compute a larger Gram matrix when the inconvenient dimension is sharded.

These paths are implemented at
[`muon_utils.py:228-290`](https://github.com/NVIDIA-NeMo/Emerging-Optimizers/blob/b309e2f01cda75dc96a6dc1a2355a7b3b64b5e16/emerging_optimizers/orthogonalized_optimizers/muon_utils.py#L228-L290).

## Muon with MCore DistOpt (ZeRO-1)

### Why ordinary byte sharding is invalid

Standard MCore DistOpt places flattened parameter ranges into equal DP shards,
reduce-scatters gradients to those shards, updates a local main-parameter shard,
and all-gathers updated parameters. This is correct for AdamW because its update
is elementwise after DP gradient reduction.

An arbitrary flat shard can contain half of one matrix and half of another.
Running Newton--Schulz independently on those fragments changes both the Gram
matrix and the update scaling, so it is not Muon on either logical weight.

### Upstream solution: whole-matrix ownership

When an emerging optimizer is combined with `--use-distributed-optimizer`, the
training arguments convert the request into
`use_layer_wise_distributed_optimizer=True` rather than sending Muon through the
standard DistOpt class. The conversion is in
[`arguments.py:1872-1886`](https://github.com/NVIDIA/Megatron-LM/blob/d64ba4ccb1e3e878c15171c9cc58d5d3b46bf4d5/megatron/training/arguments.py#L1872-L1886).

The model is then split into two buffer/optimizer domains:

| Parameter domain | Ownership granularity | Gradient synchronization | Optimizer step | Parameter synchronization |
| --- | --- | --- | --- | --- |
| Muon 2-D matrices | Complete matrices assigned across DP ranks | Default compact path: all-reduce; padded path: reduce-scatter with each matrix wholly inside one shard | Only the owner rank runs momentum + NS on the complete local matrix | All-gather updated complete matrices or the padded parameter buffer |
| Adam fallback params | Byte-level flat ranges | Reduce-scatter | Standard local DistOpt shard | Standard parameter-buffer all-gather |

The factory separation is visible in
[`optimizer/__init__.py:800-864`](https://github.com/NVIDIA/Megatron-LM/blob/d64ba4ccb1e3e878c15171c9cc58d5d3b46bf4d5/megatron/core/optimizer/__init__.py#L800-L864),
and the two optimizers are chained at
[`optimizer/__init__.py:952-970`](https://github.com/NVIDIA/Megatron-LM/blob/d64ba4ccb1e3e878c15171c9cc58d5d3b46bf4d5/megatron/core/optimizer/__init__.py#L952-L970).

There are two LayerWise layouts:

1. **Compact decoupled layout (default).** Muon buffers locally disable DistOpt
   semantics, so DDP all-reduces complete gradients. Each DP rank retains
   optimizer state only for its assigned complete matrices. After the owner
   updates them, a variable-size all-gather reconstructs the replicas. Adam
   fallback buffers still use reduce-scatter and standard DistOpt.
2. **Padded shard-aligned layout (opt-in).** A bin-packing layout guarantees that
   every matrix lies fully within one equal-sized DP buffer shard. DDP can then
   reduce-scatter a complete matrix gradient to its owner. After the update, the
   existing parameter-buffer all-gather restores replicas. Padding can be large,
   which is why this is no longer the default.

The layout guarantee is stated and enforced in
[`layer_wise_optimizer.py:293-362`](https://github.com/NVIDIA/Megatron-LM/blob/d64ba4ccb1e3e878c15171c9cc58d5d3b46bf4d5/megatron/core/optimizer/layer_wise_optimizer.py#L293-L362)
and
[`layer_wise_optimizer.py:482-543`](https://github.com/NVIDIA/Megatron-LM/blob/d64ba4ccb1e3e878c15171c9cc58d5d3b46bf4d5/megatron/core/optimizer/layer_wise_optimizer.py#L482-L543).
The compact/padded CLI contract is documented at
[`arguments.py:4222-4231`](https://github.com/NVIDIA/Megatron-LM/blob/d64ba4ccb1e3e878c15171c9cc58d5d3b46bf4d5/megatron/training/arguments.py#L4222-L4231).

For both layouts, Newton--Schulz runs only on the DP owner of a given matrix.
The subsequent sync is at
[`layer_wise_optimizer.py:649-701`](https://github.com/NVIDIA/Megatron-LM/blob/d64ba4ccb1e3e878c15171c9cc58d5d3b46bf4d5/megatron/core/optimizer/layer_wise_optimizer.py#L649-L701)
and
[`layer_wise_optimizer.py:795-821`](https://github.com/NVIDIA/Megatron-LM/blob/d64ba4ccb1e3e878c15171c9cc58d5d3b46bf4d5/megatron/core/optimizer/layer_wise_optimizer.py#L795-L821).

This is still ZeRO-1 in purpose--optimizer state is distributed--but its Muon
shard unit is a whole parameter rather than an arbitrary byte range.

### Current limitations

The current path also constrains the design MLite should initially match:

- the default compact layout requires one distributed-optimizer instance;
- FP8/FP4 parameter gather is rejected;
- overlap of the first parameter gather with the optimizer step is rejected;
- the split LayerWise + DistOpt path does not yet support non-emerging
  expert-parallel fallback groups;
- only `torch` and `torch_dist` checkpoint formats are accepted.

These checks are adjacent to the FSDP rejection in
[`arguments.py:1891-1909`](https://github.com/NVIDIA/Megatron-LM/blob/d64ba4ccb1e3e878c15171c9cc58d5d3b46bf4d5/megatron/training/arguments.py#L1891-L1909)
and in
[`optimizer/__init__.py:897-923`](https://github.com/NVIDIA/Megatron-LM/blob/d64ba4ccb1e3e878c15171c9cc58d5d3b46bf4d5/megatron/core/optimizer/__init__.py#L897-L923).

## Muon with FSDP2 (ZeRO-3)

### Current upstream and MLite behavior

Upstream PyTorch FSDP2 wraps transformer units with `fully_shard`, all-gathers
parameters just in time, and reshards them after use; see
[`torch_fully_sharded_data_parallel.py:28-53`](https://github.com/NVIDIA/Megatron-LM/blob/d64ba4ccb1e3e878c15171c9cc58d5d3b46bf4d5/megatron/core/distributed/torch_fully_sharded_data_parallel.py#L28-L53)
and
[`torch_fully_sharded_data_parallel.py:76-146`](https://github.com/NVIDIA/Megatron-LM/blob/d64ba4ccb1e3e878c15171c9cc58d5d3b46bf4d5/megatron/core/distributed/torch_fully_sharded_data_parallel.py#L76-L146).
Muon is rejected before this path is built.

MLite's FSDP2 path similarly calls `fully_shard` before constructing the
optimizer ([`wrap.py:117-194`](../megatron/lite/primitive/optimizers/fsdp2/wrap.py#L117-L194)).
Its placement function shards the first divisible tensor dimension and falls back
to dimension zero. The optimizer therefore sees DTensor parameter and gradient
shards, not complete 2-D matrices. The current builder explicitly accepts only
Adam/AdamW ([`optimizer.py:276-340`](../megatron/lite/primitive/optimizers/fsdp2/optimizer.py#L276-L340)).

For AdamW, local FP32 master parameters and both moments have the same local
shard shape, so the elementwise step is valid
([`adamw.py:117-225`](../megatron/lite/primitive/optimizers/fsdp2/adamw.py#L117-L225)).
The same local-shard assumption is not valid for Newton--Schulz.

### Required data flow

FSDP2 has already reduce-scattered the gradient by the time `optimizer.step()`
runs. A correct Muon lowering must therefore add a matrix-domain operation
between gradient synchronization and the local shard update.

The recommended first implementation is:

```text
FSDP2 backward
  -> local gradient shard
  -> update local momentum shard
  -> all-gather pre-NS momentum into the logical 2-D matrix
  -> run identical Newton--Schulz on every FSDP rank
  -> slice the orthogonalized update back to the local DTensor placement
  -> update the local parameter shard
  -> next FSDP2 forward performs its normal just-in-time parameter all-gather
```

Communication consequences:

- Existing backward communication remains one FSDP2 reduce-scatter per FSDP
  unit/bucket.
- Muon adds an all-gather of the pre-NS momentum/update for each logical matrix
  (preferably bucketed without mixing matrix boundaries).
- No new post-NS reduce-scatter is required: every rank slices the same result.
- No immediate parameter all-gather is required after the step: FSDP2's next
  forward already gathers the updated parameter shards.
- All ranks must execute collectives in a stable, globally agreed parameter
  order. Gather phases should be separated from variable-duration NS compute to
  avoid one rank entering the next collective while another is still computing.

This preserves ZeRO-3 storage for parameters and optimizer state, but temporarily
materializes one complete matrix and duplicates NS compute across the FSDP group.
The implementation should cap peak memory with one or a small bounded number of
in-flight matrices.

### Longer-term distributed Newton--Schulz

The full-matrix all-gather can later be replaced by a distributed NS plan:

1. Keep the momentum/update sharded along one matrix dimension.
2. All-reduce the scalar used for global normalization.
3. Form the local contribution to the Gram matrix and all-reduce that Gram matrix
   once per NS iteration.
4. Produce the local update shard directly.

This is the same communication pattern already used by the TP `distributed`
mode at
[`muon_utils.py:279-320`](https://github.com/NVIDIA-NeMo/Emerging-Optimizers/blob/b309e2f01cda75dc96a6dc1a2355a7b3b64b5e16/emerging_optimizers/orthogonalized_optimizers/muon_utils.py#L279-L320).
It removes the full update materialization but introduces five Gram-matrix
all-reduces with the default NS step count. The sharding dimension determines the
Gram shape, so FSDP placement and Muon cost are coupled even though their public
configuration axes remain separate.

A third option--excluding Muon matrices from FSDP2 and applying LayerWise
ZeRO-1 only to them--reuses the upstream design but replicates the model's largest
parameters. It defeats much of the reason to choose ZeRO-3 and should be a
fallback, not the default.

There is an unmerged upstream experiment named `FSDPTensorParallelMuon` at
[`c7d1aff65090fc04a50db67ddb83a51ea9615606`](https://github.com/NVIDIA/Megatron-LM/commit/c7d1aff65090fc04a50db67ddb83a51ea9615606).
It follows the gather-full-update/reshard approach and separates gather and NS
phases. It is useful prior art, but it is not part of the current `dev` support
surface and should not be treated as production behavior.

## Why algorithm and parallelism are orthogonal

The precise claim should be:

> Optimizer algorithm and parallel placement are independent configuration axes,
> provided the placement backend can lower the algorithm's update-domain
> requirements into correct local computation and communication.

AdamW's update domain is one element, so almost any shard is directly valid.
Muon's update domain is a logical matrix. That requirement can be realized in
multiple ways:

| Placement | Valid Muon lowering |
| --- | --- |
| Replicated parameter | Local NS on every rank |
| ZeRO-1 byte shard | Invalid directly; change to whole-matrix owner or gather/distributed NS |
| ZeRO-1 whole-matrix owner | Owner runs NS, then all-gather updated matrices |
| ZeRO-3 DTensor shard | Gather + duplicated NS + reshard, or distributed NS |
| TP shard | Blockwise, gather + duplicated NS, or distributed NS |

The abstraction boundary must therefore sit above a raw `torch.optim.Optimizer`
constructor. A useful split is:

1. **Parameter routing policy** maps a named logical parameter to `muon` or the
   scalar fallback algorithm.
2. **Algorithm specification** owns momentum, NS coefficients, scale mode, weight
   decay, and the required logical update domain.
3. **Placement descriptor** describes logical shape, local slice/DTensor
   placement, TP/DP groups, and expert status.
4. **Update transport/lowering** chooses local, whole-owner,
   gather-orthogonalize-reshard, or distributed-NS execution.
5. **Backend lifecycle** owns model wrapping, gradient synchronization, parameter
   synchronization, offload, and checkpoint I/O.

With this split, `adamw + fsdp2`, `muon + fsdp2`, `adamw + dist_opt`, and
`muon + dist_opt` are compositions. A backend must reject a composition when it
cannot satisfy the algorithm requirement; it must never silently run Muon on an
arbitrary local fragment.

## Megatron Lite design space

### Current abstraction

MLite already stores the two choices separately, but under confusing names:

- model `ImplConfig.optimizer` is `dist_opt`, `fsdp2`, or `None`, for example
  [`qwen3_5/lite/protocol.py:48-62`](../megatron/lite/model/qwen3_5/lite/protocol.py#L48-L62);
- `OptimizerConfig.optimizer` is the algorithm name and currently defaults to
  `adam` ([`runtime/contracts/config.py:38-69`](../megatron/lite/runtime/contracts/config.py#L38-L69));
- the backend registry contains only `dist_opt` and `fsdp2`
  ([`primitive/optimizers/__init__.py:8-17`](../megatron/lite/primitive/optimizers/__init__.py#L8-L17));
- each model protocol branches manually on the backend, for example
  [`qwen3_5/lite/protocol.py:226-273`](../megatron/lite/model/qwen3_5/lite/protocol.py#L226-L273).

The DistOpt adapter passes the algorithm name into MCore but always builds the
standard distributed layout
([`megatron_wrap.py:63-195`](../megatron/lite/primitive/optimizers/megatron_wrap.py#L63-L195)).
The FSDP2 adapter hard-codes an AdamW builder
([`optimizer.py:343-474`](../megatron/lite/primitive/optimizers/fsdp2/optimizer.py#L343-L474)).
Thus the data model is nearly orthogonal, but construction and capability
lowering are not.

There is no `mfsdp` backend in the current MLite registry. If one is added later,
it should consume the same algorithm specification and provide its own placement
lowering; it should not introduce another Muon implementation. Upstream currently
rejects emerging optimizers with Megatron-FSDP, so this remains design space, not
an existing integration.

### Options

| Option | Advantages | Problems | Recommendation |
| --- | --- | --- | --- |
| Add Muon directly to each backend builder | Small initial diff | Duplicates routing/config/state logic; encourages incorrect shard-local NS | Reject |
| Treat `muon` as another backend beside `dist_opt`/`fsdp2` | Simple dispatch | Makes algorithm and placement mutually exclusive; cannot express Muon + FSDP2 | Reject |
| Common algorithm spec + backend-specific update lowering | Preserves independent axes; central routing and checkpoint schema | Requires a small capability interface | Recommended |
| Exclude Muon weights from FSDP2 and use LayerWise ZeRO-1 | Reuses upstream semantics | Replicates the largest weights; mixed lifecycle is complex | Fallback/prototype only |

### Recommended API direction

1. Rename `ImplConfig.optimizer` to `optimizer_backend`. Accept the old field as a
   compatibility alias for one deprecation window.
2. Keep `OptimizerConfig.optimizer` as the algorithm field initially because it
   matches the VERL config, but document it as `algorithm`; a future typed alias
   may expose `algorithm` without breaking callers.
3. Extend the shared config with explicit Muon fields. Do not inherit the
   upstream `0.9` versus `0.95` momentum ambiguity.
4. Centralize the named-parameter routing policy. The policy should receive model
   metadata instead of relying only on substrings and should emit stable logical
   parameter IDs for checkpointing.
5. Have each backend advertise/lower matrix-update capabilities. The initial
   capability set can be small:

   ```text
   local_matrix
   whole_parameter_owner
   gather_orthogonalize_reshard
   distributed_orthogonalize
   ```

6. Build one mixed optimizer facade for step, grad norm, zeroing, offload, and
   state dict, with Muon and Adam child optimizers behind it.
7. Keep checkpoint state keyed by stable logical parameter name. Momentum may be
   whole-owner state under LayerWise DistOpt and DTensor-local state under FSDP2;
   resharding belongs to the backend checkpoint adapter.

### Backend-specific recommendation

For `dist_opt`:

- Do not pass Muon into the current byte-sharded MCore DistOpt path.
- Rebase/port the cohesive upstream stack: parameter tagging before DDP buffer
  construction, split layouts, `LayerWiseDistributedOptimizer`, and the chained
  Adam DistOpt fallback. Porting only `TensorParallelMuon` is insufficient.
- Start with the compact decoupled layout because it avoids persistent matrix
  padding. Preserve the padded layout only if parity or performance evidence
  justifies it.

For `fsdp2`:

- Generalize `build_fsdp2_adamw` into an algorithm-dispatching builder while
  retaining the existing AdamW implementation unchanged.
- Add a Muon child that stores only local momentum shards.
- Build a stable per-matrix execution plan after `fully_shard`, including logical
  shape, placement dimension, FSDP mesh/group, TP metadata, and local slice.
- Implement bounded gather + duplicated NS + local reshard first. Use the dense
  DP/CP mesh for dense weights and the expert-DP mesh for expert weights.
- Keep TP policy explicit. The simplest first supported combination is TP
  `blockwise` plus FSDP gather/reshard; full TP + FSDP distributed NS should be a
  later capability, not an accidental interaction.
- Preserve FSDP2's normal next-forward parameter all-gather instead of adding a
  redundant post-step gather.

For a future `mfsdp` backend:

- Reuse the FSDP2 algorithm/routing specification.
- Supply an uneven-DTensor-aware gather/reshard or distributed-NS transport.
- Treat the historical `FSDPTensorParallelMuon` commit as a prototype to test,
  not code to copy without current M-FSDP integration and checkpoint validation.

## Proposed delivery phases

1. **Configuration and routing only.** Add explicit backend/algorithm names,
   Muon config, routing-unit tests, capability rejection, and no distributed
   behavior change.
2. **DistOpt integration.** Port the complete upstream LayerWise split and verify
   Muon matrices are never byte-split; compare one-step updates against the
   pinned Emerging-Optimizers reference.
3. **FSDP2 correctness path.** Add bounded gather/reshard and compare every local
   shard with a single-rank full-matrix reference after momentum, NS, weight
   decay, and update.
4. **Checkpoint and mixed-model coverage.** Verify Muon + Adam chained state,
   DP-size reshard where supported, expert meshes, QKV split, offload, and
   continued training after load.
5. **Performance path.** Measure full-gather peak memory and communication; add
   distributed NS only when evidence shows it is necessary.

Future integration validation must use real distributed runs. At minimum it
should cover `dist_opt` and `fsdp2`, dense and expert parameters, checkpoint
continuation, and an independent full-matrix numerical reference. A test that
only checks that `optimizer.step()` returns is not evidence that the logical
matrix boundary is correct.

## Decision

Proceed with a common algorithm specification and backend-specific lowering.
For DistOpt, use whole-matrix owner sharding as upstream does. For FSDP2, start
with gather-orthogonalize-reshard while preserving local optimizer state, then
optimize to distributed Newton--Schulz behind the same interface if profiling
requires it. Do not encode Muon as a parallel backend, and do not run
Newton--Schulz directly on arbitrary ZeRO shards.
