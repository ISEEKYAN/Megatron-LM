# Post-training decision evidence

## Approved port policy

Maintainer ruling, 2026-09-11: inherit the DS4 parameter policy outside Engram;
use the report's Engram 5x LR for e_proj and e_norm; use post-training scope,
not the pretraining visual freeze/unfreeze schedule. FP32 main_grad, FP32
momentum, DS4 router reduction and STE are explicit port choices. These are
approved project policies, not claims that official inference proves backward.
The 17-entry JSON preserves the original questions and blocking gates.

Baseline inspected: `26e9bf64faf06be04686f199f8eecbfd07861a9c`.
Paths below are under `experimental/lite/megatron/lite/`:

- `primitive/optimizers/megatron_wrap.py:78–113`: optimizer/LR/decay are
  runtime-configured and passed into CoreOptimizerConfig; no new model-specific
  Muon splitting should be invented for O03/O04/O07.
- `model/deepseek_v4/lite/checkpoint.py:193`: fused gate/up is a storage mapping,
  not evidence of a separate Muon optimizer group.
- DS4 in this checkout has no V4.1 vision-vector equivalent or native headwise
  Muon implementation. O03/O04/O07 resolve **inheritance policy**, not an already
  tested numeric grouping. F1/F2 must enumerate actual owners and the selected
  DS4 optimizer config (algorithm, logical shape, LR, decay); unsupported mapping
  must fail explicitly. No 1x LR/Adam/Muon numeric fallback is inferred here.
- `primitive/optimizers/megatron_wrap.py:155–157` configures FP32 reduction
  buffers. That alone does not establish native FP32 gradient production.
  O10 implementation must use FP32 main_grad production/accumulation and reject
  BF16-to-FP32 widening using a low-mantissa test; the TE fused wgrad route
  requires `grad_added_to_main_grad` and `fuse_wgrad_accumulation`.
- `primitive/modules/router.py:189,224–242`: DS4 SigmoidTopKRouter uses TP
  count reduction and adjusts total tokens by TP size; it is not evidence for a
  new DP/CP-global modality objective. O13 inherits that scope, while O16
  remains OPEN for loss weighting/scaling. D5 must handle an empty modality.

## Report audit

Source: supplied `/tmp/v41_tech.pdf`, SHA-256 `ba68e2e40408125ae6d2f63a9a241b61c73910691c74ec1a2a7023c851eac08d`.
Reproduce extraction with `pdftotext -layout /tmp/v41_tech.pdf /tmp/a8-report.txt`.
Printed page numbers are report page numbers, not text-extraction line numbers.

Read §5 in full, printed pages 25–36: §5.1 and §5.1.1–5.1.4 describe the
SFT/RL/OPD pipeline, synthesized tasks, RL scaling, sandbox infrastructure and
effort conditioning. §5.2.1–5.2.4 describe colocated time-sharing, replay,
off-policy masking, interruption and distillation. §5.3.1–5.3.5 cover evaluation,
scaffolds and agent teams. None specifies an Engram parameter-group freeze or
update mask, master weight representation or post-training optimizer selection.
This is an evidence gap, **not proof of frozen or updated tables**.

§3.1.3 (p18), the passage supplied in the task and checked against the PDF:

> During RL rollouts, Engram embedding tables remain resident in GPU memory.
> This placement reduces host memory pressure and helps avoid out-of-memory
> failures caused by host memory fragmentation.

§3.1.3 (p18) describes row sharding, optimizer-state sharding, buffered gradient
return, FP8 lookup and Sinkhorn updates in the general training system. It also
explicitly requires GPU residency during RL rollouts to avoid host-memory
pressure and fragmentation. §3.2's inference host/RDMA path must not be imported
into this post-training design. The approved scope keeps E1 table shards on GPU;
GPU residency does not determine trainability. Rollout implementation is excluded.

§4.2 (p22) specifies Engram LR 5x in the pretraining discussion. Applying that
multiplier to e_proj/e_norm in post-training is the maintainer's explicit policy,
not a separate statement in §5. O05/O06 pretraining unfreeze choices are N/A.

O08/O09 and E1e/E2 remain conditional on a cited post-training trainability
mask plus approved master/quantization policy. If tables train, qualify master,
scale publication, Sinkhorn updates and restart. If tables freeze, table-update
work may be excluded only by a new approved mask; token embedding/prediction-head
optimizer routing must be assessed separately before removing all Sinkhorn work.
An estimated ~1200-line E2 implementation is planning input, not authorization
or evidence that it is needed. O12 remains OPEN: integer Top-K cannot determine
objective, coefficient, detach boundaries or contributing consumers. O15's
FP4-off diagnostic, O16 normalization and O17 atomic skip have no explicit
resolution in the ruling; they remain OPEN instead of guessing DS4 equivalence.

## Delivery boundary

This change delivers decision records and 64 task-ready work items, not 64
implementations or claims that scheduler nodes have been created. It restores
four prerequisite planning files from the prior planning revision; referenced
A1–A4 specification files remain named prerequisites, not certified mainline
contents. The graph retains full training acceptance dependencies; partial
forward progress does not close a blocked whole work item.
