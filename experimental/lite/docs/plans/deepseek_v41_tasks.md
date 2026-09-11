# Post-training work items

64 task-ready records (32 S/I pairs), in topological order. Estimates are engineer hours, not elapsed-time promises. OPEN lists include transitive prerequisites; see each activation condition in the ledger. No scheduler nodes or implementations are certified by this list.

## B3-S: fixture dimensions, weights, position/margin/tie rules from A1–A4

- Dependencies: none within B–G
- Direct OPEN: none
- Full-gate OPEN (including prerequisites): none
- Estimated hours: 2–6
- External prerequisites: Corrected A1/A3 and A2/A4 specifications named in plan section 0; verify content/ancestry before implementation
- Deferred acceptance: G1 actual dimensions and weights; never label a reduced fixture full-model parity

Acceptance:

- [ ] fixture dimensions, weights, position/margin/tie rules from A1–A4
- [ ] Publish reviewed interfaces, precision and independent expected fixtures; consume the named A prerequisites and resolved policy without reopening them.
- [ ] Enumerate active decision IDs and run validate_deepseek_v41_plan.py with each --active-decision ID; unresolved active choices block acceptance.

Constraints:

- Post-training scope only; no DSpark forward or rollout generation.
- Full-gate dependencies below are mandatory for acceptance; forward-only partial progress is not completion.

## B1-S: official wrapper input/output and instrumentation contract

- Dependencies: none within B–G
- Direct OPEN: none
- Full-gate OPEN (including prerequisites): none
- Estimated hours: 2–6
- External prerequisites: Corrected A1/A3 and A2/A4 specifications named in plan section 0; verify content/ancestry before implementation
- Deferred acceptance: G1 full-size official generation/forward comparison

Acceptance:

- [ ] official wrapper input/output and instrumentation contract
- [ ] Publish reviewed interfaces, precision and independent expected fixtures; consume the named A prerequisites and resolved policy without reopening them.
- [ ] Enumerate active decision IDs and run validate_deepseek_v41_plan.py with each --active-decision ID; unresolved active choices block acceptance.

Constraints:

- Post-training scope only; no DSpark forward or rollout generation.
- Full-gate dependencies below are mandatory for acceptance; forward-only partial progress is not completion.

## B2-S: independent report-derived differentiable equations, objectives and update vectors; list unknown training choices

- Dependencies: none within B–G
- Direct OPEN: O16
- Full-gate OPEN (including prerequisites): O16
- Estimated hours: 2–6
- External prerequisites: Corrected A1/A3 and A2/A4 specifications named in plan section 0; verify content/ancestry before implementation
- Deferred acceptance: D/F native backward/update; unknown official objectives remain blocked

Acceptance:

- [ ] independent report-derived differentiable equations, objectives and update vectors; list unknown training choices
- [ ] Publish reviewed interfaces, precision and independent expected fixtures; consume the named A prerequisites and resolved policy without reopening them.
- [ ] Enumerate active decision IDs and run validate_deepseek_v41_plan.py with each --active-decision ID; unresolved active choices block acceptance.

Constraints:

- Post-training scope only; no DSpark forward or rollout generation.
- Full-gate dependencies below are mandatory for acceptance; forward-only partial progress is not completion.

## C1-S: A2 header/payload/decode contract

- Dependencies: none within B–G
- Direct OPEN: none
- Full-gate OPEN (including prerequisites): none
- Estimated hours: 2–6
- External prerequisites: Corrected A1/A3 and A2/A4 specifications named in plan section 0; verify content/ancestry before implementation
- Deferred acceptance: C4 full binding; G1 actual shards

Acceptance:

- [ ] A2 header/payload/decode contract
- [ ] Publish reviewed interfaces, precision and independent expected fixtures; consume the named A prerequisites and resolved policy without reopening them.
- [ ] Enumerate active decision IDs and run validate_deepseek_v41_plan.py with each --active-decision ID; unresolved active choices block acceptance.

Constraints:

- Post-training scope only; no DSpark forward or rollout generation.
- Full-gate dependencies below are mandatory for acceptance; forward-only partial progress is not completion.

## C2a-S: A3 main KV E2M1, group16/E4M3, no second global scale, post-RoPE, post-training only

- Dependencies: none within B–G
- Direct OPEN: O15
- Full-gate OPEN (including prerequisites): O15
- Estimated hours: 2–6
- External prerequisites: Corrected A1/A3 and A2/A4 specifications named in plan section 0; verify content/ancestry before implementation
- Deferred acceptance: D2 sparse ABI and G1 phase transition

Acceptance:

- [ ] A3 main KV E2M1, group16/E4M3, no second global scale, post-RoPE, post-training only
- [ ] Publish reviewed interfaces, precision and independent expected fixtures; consume the named A prerequisites and resolved policy without reopening them.
- [ ] Enumerate active decision IDs and run validate_deepseek_v41_plan.py with each --active-decision ID; unresolved active choices block acceptance.

Constraints:

- Post-training scope only; no DSpark forward or rollout generation.
- Full-gate dependencies below are mandatory for acceptance; forward-only partial progress is not completion.

## C2b-S: A3 index Q/K E2M1 group32/E8M0, independent QAT switch

- Dependencies: none within B–G
- Direct OPEN: none
- Full-gate OPEN (including prerequisites): none
- Estimated hours: 2–6
- External prerequisites: Corrected A1/A3 and A2/A4 specifications named in plan section 0; verify content/ancestry before implementation
- Deferred acceptance: D2/D6

Acceptance:

- [ ] A3 index Q/K E2M1 group32/E8M0, independent QAT switch
- [ ] Publish reviewed interfaces, precision and independent expected fixtures; consume the named A prerequisites and resolved policy without reopening them.
- [ ] Enumerate active decision IDs and run validate_deepseek_v41_plan.py with each --active-decision ID; unresolved active choices block acceptance.

Constraints:

- Post-training scope only; no DSpark forward or rollout generation.
- Full-gate dependencies below are mandatory for acceptance; forward-only partial progress is not completion.

## C2c-S: A3 SWA full-vector post-RoPE FP8 and Linear dynamic activation FP8, separate policies

- Dependencies: none within B–G
- Direct OPEN: none
- Full-gate OPEN (including prerequisites): none
- Estimated hours: 2–6
- External prerequisites: Corrected A1/A3 and A2/A4 specifications named in plan section 0; verify content/ancestry before implementation
- Deferred acceptance: D2/D4 and G1; BF16 decode+GEMM alone is insufficient

Acceptance:

- [ ] A3 SWA full-vector post-RoPE FP8 and Linear dynamic activation FP8, separate policies
- [ ] Publish reviewed interfaces, precision and independent expected fixtures; consume the named A prerequisites and resolved policy without reopening them.
- [ ] Enumerate active decision IDs and run validate_deepseek_v41_plan.py with each --active-decision ID; unresolved active choices block acceptance.

Constraints:

- Post-training scope only; no DSpark forward or rollout generation.
- Full-gate dependencies below are mandatory for acceptance; forward-only partial progress is not completion.

## C3-S: A2 inactive MTP carrier, original config, independent execution flag

- Dependencies: none within B–G
- Direct OPEN: none
- Full-gate OPEN (including prerequisites): none
- Estimated hours: 2–6
- External prerequisites: Corrected A1/A3 and A2/A4 specifications named in plan section 0; verify content/ancestry before implementation
- Deferred acceptance: C4/G1; MTP excluded from trainable coverage but included in checkpoint coverage

Acceptance:

- [ ] A2 inactive MTP carrier, original config, independent execution flag
- [ ] Publish reviewed interfaces, precision and independent expected fixtures; consume the named A prerequisites and resolved policy without reopening them.
- [ ] Enumerate active decision IDs and run validate_deepseek_v41_plan.py with each --active-decision ID; unresolved active choices block acceptance.

Constraints:

- Post-training scope only; no DSpark forward or rollout generation.
- Full-gate dependencies below are mandatory for acceptance; forward-only partial progress is not completion.

## C4-S: consume A1–A4, all 96085 keys and 87 config leaves; declare canonical owner/alias, header→logical weight binding and protocol contract

- Dependencies: none within B–G
- Direct OPEN: none
- Full-gate OPEN (including prerequisites): none
- Estimated hours: 2–6
- External prerequisites: Corrected A1/A3 and A2/A4 specifications named in plan section 0; verify content/ancestry before implementation
- Deferred acceptance: F2 actual optimizer enumeration; E distributed binding and G1 real-size all-key load

Acceptance:

- [ ] consume A1–A4, all 96085 keys and 87 config leaves; declare canonical owner/alias, header→logical weight binding and protocol contract
- [ ] Publish reviewed interfaces, precision and independent expected fixtures; consume the named A prerequisites and resolved policy without reopening them.
- [ ] Enumerate active decision IDs and run validate_deepseek_v41_plan.py with each --active-decision ID; unresolved active choices block acceptance.

Constraints:

- Post-training scope only; no DSpark forward or rollout generation.
- Full-gate dependencies below are mandatory for acceptance; forward-only partial progress is not completion.

## E1e-S: repeated/unvisited rows and FP8 update publication contract below

- Dependencies: none within B–G
- Direct OPEN: O08, O09, O17
- Full-gate OPEN (including prerequisites): O08, O09, O17
- Estimated hours: 2–6
- External prerequisites: Corrected A1/A3 and A2/A4 specifications named in plan section 0; verify content/ancestry before implementation
- Deferred acceptance: E2/E5 distributed updates

Acceptance:

- [ ] repeated/unvisited rows and FP8 update publication contract below
- [ ] Publish reviewed interfaces, precision and independent expected fixtures; consume the named A prerequisites and resolved policy without reopening them.
- [ ] Enumerate active decision IDs and run validate_deepseek_v41_plan.py with each --active-decision ID; unresolved active choices block acceptance.
- [ ] Obtain evidence or approved policy for post-training Engram trainability before update implementation; O08/O09 remain unresolved. If frozen, obtain explicit scope adjustment and separately audit token embedding/head optimizer needs.
- [ ] Verify native FP32 main_grad rather than widened BF16 gradients, and FP32 momentum; state dtype policy does not resolve table trainability.

Constraints:

- Post-training scope only; no DSpark forward or rollout generation.
- Full-gate dependencies below are mandatory for acceptance; forward-only partial progress is not completion.
- Engram tables/scales remain GPU-resident; no CPU table offload or host/RDMA prefetch; account for approved state/workspace in memory budget.

## E2-S: A4 Algorithm1, logical whole-matrix statistics and training-state precision decisions

- Dependencies: none within B–G
- Direct OPEN: O08, O09
- Full-gate OPEN (including prerequisites): O08, O09
- Estimated hours: 2–6
- External prerequisites: Corrected A1/A3 and A2/A4 specifications named in plan section 0; verify content/ancestry before implementation
- Deferred acceptance: E4 mixed backend/E5 restore

Acceptance:

- [ ] A4 Algorithm1, logical whole-matrix statistics and training-state precision decisions
- [ ] Publish reviewed interfaces, precision and independent expected fixtures; consume the named A prerequisites and resolved policy without reopening them.
- [ ] Enumerate active decision IDs and run validate_deepseek_v41_plan.py with each --active-decision ID; unresolved active choices block acceptance.
- [ ] Obtain evidence or approved policy for post-training Engram trainability before update implementation; O08/O09 remain unresolved. If frozen, obtain explicit scope adjustment and separately audit token embedding/head optimizer needs.
- [ ] Verify native FP32 main_grad rather than widened BF16 gradients, and FP32 momentum; state dtype policy does not resolve table trainability.

Constraints:

- Post-training scope only; no DSpark forward or rollout generation.
- Full-gate dependencies below are mandatory for acceptance; forward-only partial progress is not completion.

## D1-S: corrected A1 shifted mHC and paired CED state

- Dependencies: none within B–G
- Direct OPEN: none
- Full-gate OPEN (including prerequisites): none
- Estimated hours: 2–6
- External prerequisites: Corrected A1/A3 and A2/A4 specifications named in plan section 0; verify content/ancestry before implementation
- Deferred acceptance: E3 paired payload, recompute, PP

Acceptance:

- [ ] corrected A1 shifted mHC and paired CED state
- [ ] Publish reviewed interfaces, precision and independent expected fixtures; consume the named A prerequisites and resolved policy without reopening them.
- [ ] Enumerate active decision IDs and run validate_deepseek_v41_plan.py with each --active-decision ID; unresolved active choices block acceptance.

Constraints:

- Post-training scope only; no DSpark forward or rollout generation.
- Full-gate dependencies below are mandatory for acceptance; forward-only partial progress is not completion.

## D2-S: corrected A1/A3 owner/mode/operator decisions and C formats

- Dependencies: none within B–G
- Direct OPEN: O15
- Full-gate OPEN (including prerequisites): O15
- Estimated hours: 2–6
- External prerequisites: Corrected A1/A3 and A2/A4 specifications named in plan section 0; verify content/ancestry before implementation
- Deferred acceptance: E3/E4 state transport and kernel ABI; G1

Acceptance:

- [ ] corrected A1/A3 owner/mode/operator decisions and C formats
- [ ] Publish reviewed interfaces, precision and independent expected fixtures; consume the named A prerequisites and resolved policy without reopening them.
- [ ] Enumerate active decision IDs and run validate_deepseek_v41_plan.py with each --active-decision ID; unresolved active choices block acceptance.

Constraints:

- Post-training scope only; no DSpark forward or rollout generation.
- Full-gate dependencies below are mandatory for acceptance; forward-only partial progress is not completion.

## D3-S: A3 two-level candidate builder, post-training switch

- Dependencies: none within B–G
- Direct OPEN: none
- Full-gate OPEN (including prerequisites): none
- Estimated hours: 2–6
- External prerequisites: Corrected A1/A3 and A2/A4 specifications named in plan section 0; verify content/ancestry before implementation
- Deferred acceptance: E4 CP positions

Acceptance:

- [ ] A3 two-level candidate builder, post-training switch
- [ ] Publish reviewed interfaces, precision and independent expected fixtures; consume the named A prerequisites and resolved policy without reopening them.
- [ ] Enumerate active decision IDs and run validate_deepseek_v41_plan.py with each --active-decision ID; unresolved active choices block acceptance.

Constraints:

- Post-training scope only; no DSpark forward or rollout generation.
- Full-gate dependencies below are mandatory for acceptance; forward-only partial progress is not completion.

## D4-S: official Engram hash and computation

- Dependencies: none within B–G
- Direct OPEN: none
- Full-gate OPEN (including prerequisites): none
- Estimated hours: 2–6
- External prerequisites: Corrected A1/A3 and A2/A4 specifications named in plan section 0; verify content/ancestry before implementation
- Deferred acceptance: E1/E2 table training

Acceptance:

- [ ] official Engram hash and computation
- [ ] Publish reviewed interfaces, precision and independent expected fixtures; consume the named A prerequisites and resolved policy without reopening them.
- [ ] Enumerate active decision IDs and run validate_deepseek_v41_plan.py with each --active-decision ID; unresolved active choices block acceptance.

Constraints:

- Post-training scope only; no DSpark forward or rollout generation.
- Full-gate dependencies below are mandatory for acceptance; forward-only partial progress is not completion.

## D5-S: modality-specific load update reference, rate 0.001 and sequence auxiliary coefficient 0.0001; unresolved reduction scope stays explicit

- Dependencies: none within B–G
- Direct OPEN: O16
- Full-gate OPEN (including prerequisites): O16
- Estimated hours: 2–6
- External prerequisites: Corrected A1/A3 and A2/A4 specifications named in plan section 0; verify content/ancestry before implementation
- Deferred acceptance: E4 cross-rank reduction/atomic step; E5 restart; G2 replay

Acceptance:

- [ ] modality-specific load update reference, rate 0.001 and sequence auxiliary coefficient 0.0001; unresolved reduction scope stays explicit
- [ ] Publish reviewed interfaces, precision and independent expected fixtures; consume the named A prerequisites and resolved policy without reopening them.
- [ ] Enumerate active decision IDs and run validate_deepseek_v41_plan.py with each --active-decision ID; unresolved active choices block acceptance.

Constraints:

- Post-training scope only; no DSpark forward or rollout generation.
- Full-gate dependencies below are mandatory for acceptance; forward-only partial progress is not completion.

## D6-S: exact indexer objective, coefficient, normalization, detach, consumer contribution table; unknown choices block official training acceptance

- Dependencies: none within B–G
- Direct OPEN: O12, O16
- Full-gate OPEN (including prerequisites): O12, O16
- Estimated hours: 2–6
- External prerequisites: Corrected A1/A3 and A2/A4 specifications named in plan section 0; verify content/ancestry before implementation
- Deferred acceptance: E3/E4 shared owner gradient delivery; G1 training

Acceptance:

- [ ] exact indexer objective, coefficient, normalization, detach, consumer contribution table; unknown choices block official training acceptance
- [ ] Publish reviewed interfaces, precision and independent expected fixtures; consume the named A prerequisites and resolved policy without reopening them.
- [ ] Enumerate active decision IDs and run validate_deepseek_v41_plan.py with each --active-decision ID; unresolved active choices block acceptance.

Constraints:

- Post-training scope only; no DSpark forward or rollout generation.
- Full-gate dependencies below are mandatory for acceptance; forward-only partial progress is not completion.

## D7-S: A1/A3 packed sequence-local history and visibility

- Dependencies: none within B–G
- Direct OPEN: none
- Full-gate OPEN (including prerequisites): none
- Estimated hours: 2–6
- External prerequisites: Corrected A1/A3 and A2/A4 specifications named in plan section 0; verify content/ancestry before implementation
- Deferred acceptance: E4 real runtime full-THD→CP-local and non-skip gradient parity

Acceptance:

- [ ] A1/A3 packed sequence-local history and visibility
- [ ] Publish reviewed interfaces, precision and independent expected fixtures; consume the named A prerequisites and resolved policy without reopening them.
- [ ] Enumerate active decision IDs and run validate_deepseek_v41_plan.py with each --active-decision ID; unresolved active choices block acceptance.

Constraints:

- Post-training scope only; no DSpark forward or rollout generation.
- Full-gate dependencies below are mandatory for acceptance; forward-only partial progress is not completion.

## E0-S: owner/replica/optimizer-shard layout and relation to dense/expert rank decompositions

- Dependencies: none within B–G
- Direct OPEN: none
- Full-gate OPEN (including prerequisites): none
- Estimated hours: 2–6
- External prerequisites: Corrected A1/A3 and A2/A4 specifications named in plan section 0; verify content/ancestry before implementation
- Deferred acceptance: E1–E4 collective execution

Acceptance:

- [ ] owner/replica/optimizer-shard layout and relation to dense/expert rank decompositions
- [ ] Publish reviewed interfaces, precision and independent expected fixtures; consume the named A prerequisites and resolved policy without reopening them.
- [ ] Enumerate active decision IDs and run validate_deepseek_v41_plan.py with each --active-decision ID; unresolved active choices block acceptance.

Constraints:

- Post-training scope only; no DSpark forward or rollout generation.
- Full-gate dependencies below are mandatory for acceptance; forward-only partial progress is not completion.

## E1a-S: 24 rows/token ×256=6144 features, routing IDs/scales

- Dependencies: none within B–G
- Direct OPEN: none
- Full-gate OPEN (including prerequisites): none
- Estimated hours: 2–6
- External prerequisites: Corrected A1/A3 and A2/A4 specifications named in plan section 0; verify content/ancestry before implementation
- Deferred acceptance: G1 size

Acceptance:

- [ ] 24 rows/token ×256=6144 features, routing IDs/scales
- [ ] Publish reviewed interfaces, precision and independent expected fixtures; consume the named A prerequisites and resolved policy without reopening them.
- [ ] Enumerate active decision IDs and run validate_deepseek_v41_plan.py with each --active-decision ID; unresolved active choices block acceptance.

Constraints:

- Post-training scope only; no DSpark forward or rollout generation.
- Full-gate dependencies below are mandatory for acceptance; forward-only partial progress is not completion.
- Engram tables/scales remain GPU-resident; no CPU table offload or host/RDMA prefetch; account for approved state/workspace in memory budget.

## E1b-S: A2 streaming intervals/header identity

- Dependencies: none within B–G
- Direct OPEN: none
- Full-gate OPEN (including prerequisites): none
- Estimated hours: 2–6
- External prerequisites: Corrected A1/A3 and A2/A4 specifications named in plan section 0; verify content/ancestry before implementation
- Deferred acceptance: G1 actual table load

Acceptance:

- [ ] A2 streaming intervals/header identity
- [ ] Publish reviewed interfaces, precision and independent expected fixtures; consume the named A prerequisites and resolved policy without reopening them.
- [ ] Enumerate active decision IDs and run validate_deepseek_v41_plan.py with each --active-decision ID; unresolved active choices block acceptance.

Constraints:

- Post-training scope only; no DSpark forward or rollout generation.
- Full-gate dependencies below are mandatory for acceptance; forward-only partial progress is not completion.
- Engram tables/scales remain GPU-resident; no CPU table offload or host/RDMA prefetch; account for approved state/workspace in memory budget.

## E1c-S: FP8 values/scales ABI, distinct numerical and performance contracts

- Dependencies: none within B–G
- Direct OPEN: none
- Full-gate OPEN (including prerequisites): none
- Estimated hours: 2–6
- External prerequisites: Corrected A1/A3 and A2/A4 specifications named in plan section 0; verify content/ancestry before implementation
- Deferred acceptance: G1 representative performance

Acceptance:

- [ ] FP8 values/scales ABI, distinct numerical and performance contracts
- [ ] Publish reviewed interfaces, precision and independent expected fixtures; consume the named A prerequisites and resolved policy without reopening them.
- [ ] Enumerate active decision IDs and run validate_deepseek_v41_plan.py with each --active-decision ID; unresolved active choices block acceptance.

Constraints:

- Post-training scope only; no DSpark forward or rollout generation.
- Full-gate dependencies below are mandatory for acceptance; forward-only partial progress is not completion.
- Engram tables/scales remain GPU-resident; no CPU table offload or host/RDMA prefetch; account for approved state/workspace in memory budget.

## E1d-S: local-batch prefetch before stage microbatches; buffered gradient return after backbone backward, dependency events

- Dependencies: none within B–G
- Direct OPEN: none
- Full-gate OPEN (including prerequisites): none
- Estimated hours: 2–6
- External prerequisites: Corrected A1/A3 and A2/A4 specifications named in plan section 0; verify content/ancestry before implementation
- Deferred acceptance: F3 overlap/G1 performance

Acceptance:

- [ ] local-batch prefetch before stage microbatches; buffered gradient return after backbone backward, dependency events
- [ ] Publish reviewed interfaces, precision and independent expected fixtures; consume the named A prerequisites and resolved policy without reopening them.
- [ ] Enumerate active decision IDs and run validate_deepseek_v41_plan.py with each --active-decision ID; unresolved active choices block acceptance.

Constraints:

- Post-training scope only; no DSpark forward or rollout generation.
- Full-gate dependencies below are mandatory for acceptance; forward-only partial progress is not completion.
- Engram tables/scales remain GPU-resident; no CPU table offload or host/RDMA prefetch; account for approved state/workspace in memory budget.

## E3-S: paired HC/CED payload, shadow indexers, per-microbatch state lifetime

- Dependencies: none within B–G
- Direct OPEN: none
- Full-gate OPEN (including prerequisites): none
- Estimated hours: 2–6
- External prerequisites: Corrected A1/A3 and A2/A4 specifications named in plan section 0; verify content/ancestry before implementation
- Deferred acceptance: E5/G1

Acceptance:

- [ ] paired HC/CED payload, shadow indexers, per-microbatch state lifetime
- [ ] Publish reviewed interfaces, precision and independent expected fixtures; consume the named A prerequisites and resolved policy without reopening them.
- [ ] Enumerate active decision IDs and run validate_deepseek_v41_plan.py with each --active-decision ID; unresolved active choices block acceptance.

Constraints:

- Post-training scope only; no DSpark forward or rollout generation.
- Full-gate dependencies below are mandatory for acceptance; forward-only partial progress is not completion.

## E4-S: TP score reduction, EP modality loads, CP global/local positions and mixed backend state contract

- Dependencies: none within B–G
- Direct OPEN: O17
- Full-gate OPEN (including prerequisites): O17
- Estimated hours: 2–6
- External prerequisites: Corrected A1/A3 and A2/A4 specifications named in plan section 0; verify content/ancestry before implementation
- Deferred acceptance: E5/G1

Acceptance:

- [ ] TP score reduction, EP modality loads, CP global/local positions and mixed backend state contract
- [ ] Publish reviewed interfaces, precision and independent expected fixtures; consume the named A prerequisites and resolved policy without reopening them.
- [ ] Enumerate active decision IDs and run validate_deepseek_v41_plan.py with each --active-decision ID; unresolved active choices block acceptance.
- [ ] Verify native FP32 main_grad rather than widened BF16 gradients, and FP32 momentum; state dtype policy does not resolve table trainability.

Constraints:

- Post-training scope only; no DSpark forward or rollout generation.
- Full-gate dependencies below are mandatory for acceptance; forward-only partial progress is not completion.

## E5t-S: text training restart trajectory

- Dependencies: none within B–G
- Direct OPEN: O17
- Full-gate OPEN (including prerequisites): O17
- Estimated hours: 2–6
- External prerequisites: Corrected A1/A3 and A2/A4 specifications named in plan section 0; verify content/ancestry before implementation
- Deferred acceptance: G1

Acceptance:

- [ ] text training restart trajectory
- [ ] Publish reviewed interfaces, precision and independent expected fixtures; consume the named A prerequisites and resolved policy without reopening them.
- [ ] Enumerate active decision IDs and run validate_deepseek_v41_plan.py with each --active-decision ID; unresolved active choices block acceptance.

Constraints:

- Post-training scope only; no DSpark forward or rollout generation.
- Full-gate dependencies below are mandatory for acceptance; forward-only partial progress is not completion.

## E5v-S: F3 multimodal stage transition restart

- Dependencies: none within B–G
- Direct OPEN: O17
- Full-gate OPEN (including prerequisites): O17
- Estimated hours: 2–6
- External prerequisites: Corrected A1/A3 and A2/A4 specifications named in plan section 0; verify content/ancestry before implementation
- Deferred acceptance: G1

Acceptance:

- [ ] F3 multimodal stage transition restart
- [ ] Publish reviewed interfaces, precision and independent expected fixtures; consume the named A prerequisites and resolved policy without reopening them.
- [ ] Enumerate active decision IDs and run validate_deepseek_v41_plan.py with each --active-decision ID; unresolved active choices block acceptance.
- [ ] Use explicit post-training trainability mask; O05/O06 pretraining unfreeze/LR transitions are N/A and do not establish that vision is frozen.

Constraints:

- Post-training scope only; no DSpark forward or rollout generation.
- Full-gate dependencies below are mandatory for acceptance; forward-only partial progress is not completion.

## F1-S: A4 logical head/matrix split and audited Muon backend revision/API

- Dependencies: none within B–G
- Direct OPEN: none
- Full-gate OPEN (including prerequisites): none
- Estimated hours: 2–6
- External prerequisites: Corrected A1/A3 and A2/A4 specifications named in plan section 0; verify content/ancestry before implementation
- Deferred acceptance: E4 TP/FSDP logical reassembly

Acceptance:

- [ ] A4 logical head/matrix split and audited Muon backend revision/API
- [ ] Publish reviewed interfaces, precision and independent expected fixtures; consume the named A prerequisites and resolved policy without reopening them.
- [ ] Enumerate active decision IDs and run validate_deepseek_v41_plan.py with each --active-decision ID; unresolved active choices block acceptance.
- [ ] Audit actual DS4 algorithm/group/shape/LR/decay for inherited families; reject unsupported vision mapping or silent numeric defaults. Apply Engram e_proj/e_norm 5x only if trainable.

Constraints:

- Post-training scope only; no DSpark forward or rollout generation.
- Full-gate dependencies below are mandatory for acceptance; forward-only partial progress is not completion.

## F2-S: A4 all-column OPEN ledger below, active-phase resolution records

- Dependencies: none within B–G
- Direct OPEN: none
- Full-gate OPEN (including prerequisites): none
- Estimated hours: 2–6
- External prerequisites: Corrected A1/A3 and A2/A4 specifications named in plan section 0; verify content/ancestry before implementation
- Deferred acceptance: E4 mixed backend/E5/G1

Acceptance:

- [ ] A4 all-column OPEN ledger below, active-phase resolution records
- [ ] Publish reviewed interfaces, precision and independent expected fixtures; consume the named A prerequisites and resolved policy without reopening them.
- [ ] Enumerate active decision IDs and run validate_deepseek_v41_plan.py with each --active-decision ID; unresolved active choices block acceptance.
- [ ] Use explicit post-training trainability mask; O05/O06 pretraining unfreeze/LR transitions are N/A and do not establish that vision is frozen.
- [ ] Audit actual DS4 algorithm/group/shape/LR/decay for inherited families; reject unsupported vision mapping or silent numeric defaults. Apply Engram e_proj/e_norm 5x only if trainable.
- [ ] Verify native FP32 main_grad rather than widened BF16 gradients, and FP32 momentum; state dtype policy does not resolve table trainability.

Constraints:

- Post-training scope only; no DSpark forward or rollout generation.
- Full-gate dependencies below are mandatory for acceptance; forward-only partial progress is not completion.

## F3-S: A4 visual schedule, preprocessing, spans, external-encoder sync and gradient contract

- Dependencies: none within B–G
- Direct OPEN: none
- Full-gate OPEN (including prerequisites): none
- Estimated hours: 2–6
- External prerequisites: Corrected A1/A3 and A2/A4 specifications named in plan section 0; verify content/ancestry before implementation
- Deferred acceptance: E5v/G1

Acceptance:

- [ ] A4 visual schedule, preprocessing, spans, external-encoder sync and gradient contract
- [ ] Publish reviewed interfaces, precision and independent expected fixtures; consume the named A prerequisites and resolved policy without reopening them.
- [ ] Enumerate active decision IDs and run validate_deepseek_v41_plan.py with each --active-decision ID; unresolved active choices block acceptance.
- [ ] Use explicit post-training trainability mask; O05/O06 pretraining unfreeze/LR transitions are N/A and do not establish that vision is frozen.

Constraints:

- Post-training scope only; no DSpark forward or rollout generation.
- Full-gate dependencies below are mandatory for acceptance; forward-only partial progress is not completion.

## G2-S: external routes payload coordinates and segment contract

- Dependencies: none within B–G
- Direct OPEN: none
- Full-gate OPEN (including prerequisites): none
- Estimated hours: 2–6
- External prerequisites: Corrected A1/A3 and A2/A4 specifications named in plan section 0; verify content/ancestry before implementation
- Deferred acceptance: G1 combined training

Acceptance:

- [ ] external routes payload coordinates and segment contract
- [ ] Publish reviewed interfaces, precision and independent expected fixtures; consume the named A prerequisites and resolved policy without reopening them.
- [ ] Enumerate active decision IDs and run validate_deepseek_v41_plan.py with each --active-decision ID; unresolved active choices block acceptance.

Constraints:

- Post-training scope only; no DSpark forward or rollout generation.
- Full-gate dependencies below are mandatory for acceptance; forward-only partial progress is not completion.

## G1-S: actual config/weights, memory placement, dtype/phase matrix, numeric thresholds and runtime fixture manifest

- Dependencies: none within B–G
- Direct OPEN: O15
- Full-gate OPEN (including prerequisites): O15
- Estimated hours: 2–6
- External prerequisites: Corrected A1/A3 and A2/A4 specifications named in plan section 0; verify content/ancestry before implementation
- Deferred acceptance: Final acceptance; cannot close on resource readiness alone

Acceptance:

- [ ] actual config/weights, memory placement, dtype/phase matrix, numeric thresholds and runtime fixture manifest
- [ ] Publish reviewed interfaces, precision and independent expected fixtures; consume the named A prerequisites and resolved policy without reopening them.
- [ ] Enumerate active decision IDs and run validate_deepseek_v41_plan.py with each --active-decision ID; unresolved active choices block acceptance.

Constraints:

- Post-training scope only; no DSpark forward or rollout generation.
- Full-gate dependencies below are mandatory for acceptance; forward-only partial progress is not completion.

## B3-I: `tests/fixtures/deepseek_v41/manifest.json` and generator; retain 40-layer ownership, all three CSA2 modes, both Engrams, CED boundary, nonaligned packed/image spans

- Dependencies: B3-S
- Direct OPEN: none
- Full-gate OPEN (including prerequisites): none
- Estimated hours: 8–24
- External prerequisites: Corrected A1/A3 and A2/A4 specifications named in plan section 0; verify content/ancestry before implementation
- Deferred acceptance: G1 actual dimensions and weights; never label a reduced fixture full-model parity

Acceptance:

- [ ] `tests/fixtures/deepseek_v41/manifest.json` and generator; retain 40-layer ownership, all three CSA2 modes, both Engrams, CED boundary, nonaligned packed/image spans
- [ ] `test_fixtures.py`: check generated shapes, owner identities, reproducibility and hand-known sentinels
- [ ] Run actual implementation against the S fixtures; record commands, exit status, numerical thresholds and applicable discriminating mutations. GPU tests require Slurm job ID, sacct rc=0 and non-skip execution.

Constraints:

- Post-training scope only; no DSpark forward or rollout generation.
- Full-gate dependencies below are mandatory for acceptance; forward-only partial progress is not completion.

## B1-I: `tools/deepseek_v41/oracle.py`; build official ModelArgs and converted shards from B3, execute original methods, expose all-token head output and named CED intermediates

- Dependencies: B1-S, B3-I
- Direct OPEN: none
- Full-gate OPEN (including prerequisites): none
- Estimated hours: 24–64
- External prerequisites: Corrected A1/A3 and A2/A4 specifications named in plan section 0; verify content/ancestry before implementation
- Deferred acceptance: G1 full-size official generation/forward comparison

Acceptance:

- [ ] `tools/deepseek_v41/oracle.py`; build official ModelArgs and converted shards from B3, execute original methods, expose all-token head output and named CED intermediates
- [ ] `test_oracle.py`: independent official execution comparison, complete weight/token coverage, deterministic seeds; use A5 AST-extracted executable dataflow probes, including their eleven mutations
- [ ] Run actual implementation against the S fixtures; record commands, exit status, numerical thresholds and applicable discriminating mutations. GPU tests require Slurm job ID, sacct rc=0 and non-skip execution.

Constraints:

- Post-training scope only; no DSpark forward or rollout generation.
- Full-gate dependencies below are mandatory for acceptance; forward-only partial progress is not completion.

## B2-I: `tests/reference/deepseek_v41/training.py`; scalar/float64 reference independent of native kernels

- Dependencies: B2-S, B3-I
- Direct OPEN: O12
- Full-gate OPEN (including prerequisites): O12, O16
- Estimated hours: 8–24
- External prerequisites: Corrected A1/A3 and A2/A4 specifications named in plan section 0; verify content/ancestry before implementation
- Deferred acceptance: D/F native backward/update; unknown official objectives remain blocked

Acceptance:

- [ ] `tests/reference/deepseek_v41/training.py`; scalar/float64 reference independent of native kernels
- [ ] `test_training_reference.py`: analytical versus finite-difference checks where smooth, detach and sum-of-consumer checks; discrete selections held fixed
- [ ] Run actual implementation against the S fixtures; record commands, exit status, numerical thresholds and applicable discriminating mutations. GPU tests require Slurm job ID, sacct rc=0 and non-skip execution.

Constraints:

- Post-training scope only; no DSpark forward or rollout generation.
- Full-gate dependencies below are mandatory for acceptance; forward-only partial progress is not completion.

## E0-I: `model/deepseek_v41/lite/parallel.py`

- Dependencies: B3-I, E0-S
- Direct OPEN: none
- Full-gate OPEN (including prerequisites): none
- Estimated hours: 8–24
- External prerequisites: Corrected A1/A3 and A2/A4 specifications named in plan section 0; verify content/ancestry before implementation
- Deferred acceptance: E1–E4 collective execution

Acceptance:

- [ ] `model/deepseek_v41/lite/parallel.py`
- [ ] `test_ownership.py`: logical coverage/no overlap, aliases/shadows one owner; dedicated groups do not imply a new orthogonal axis
- [ ] Run actual implementation against the S fixtures; record commands, exit status, numerical thresholds and applicable discriminating mutations. GPU tests require Slurm job ID, sacct rc=0 and non-skip execution.

Constraints:

- Post-training scope only; no DSpark forward or rollout generation.
- Full-gate dependencies below are mandatory for acceptance; forward-only partial progress is not completion.

## C1-I: `model/deepseek_v41/lite/checkpoint.py` FP4 packed I8 and FP8 block [32,32] decoding

- Dependencies: B1-I, C1-S
- Direct OPEN: none
- Full-gate OPEN (including prerequisites): none
- Estimated hours: 8–24
- External prerequisites: Corrected A1/A3 and A2/A4 specifications named in plan section 0; verify content/ancestry before implementation
- Deferred acceptance: C4 full binding; G1 actual shards

Acceptance:

- [ ] `model/deepseek_v41/lite/checkpoint.py` FP4 packed I8 and FP8 block [32,32] decoding
- [ ] `test_checkpoint_decode.py`: independent official decode, known code values and loaded GEMM; separate raw-byte roundtrip including header dtype/shape/digest
- [ ] Run actual implementation against the S fixtures; record commands, exit status, numerical thresholds and applicable discriminating mutations. GPU tests require Slurm job ID, sacct rc=0 and non-skip execution.

Constraints:

- Post-training scope only; no DSpark forward or rollout generation.
- Full-gate dependencies below are mandatory for acceptance; forward-only partial progress is not completion.

## C2a-I: `primitive/quantization/ds41_kv.py`

- Dependencies: B1-I, C2a-S
- Direct OPEN: none
- Full-gate OPEN (including prerequisites): O15
- Estimated hours: 8–24
- External prerequisites: Corrected A1/A3 and A2/A4 specifications named in plan section 0; verify content/ancestry before implementation
- Deferred acceptance: D2 sparse ABI and G1 phase transition

Acceptance:

- [ ] `primitive/quantization/ds41_kv.py`
- [ ] `test_kv_quantization.py`: codes/scales, rounding/ties, zero, saturation, dequantized values and separately approved fake-quant derivative
- [ ] Run actual implementation against the S fixtures; record commands, exit status, numerical thresholds and applicable discriminating mutations. GPU tests require Slurm job ID, sacct rc=0 and non-skip execution.

Constraints:

- Post-training scope only; no DSpark forward or rollout generation.
- Full-gate dependencies below are mandatory for acceptance; forward-only partial progress is not completion.

## C2b-I: `primitive/quantization/ds41_index.py`

- Dependencies: B1-I, C2b-S
- Direct OPEN: none
- Full-gate OPEN (including prerequisites): none
- Estimated hours: 8–24
- External prerequisites: Corrected A1/A3 and A2/A4 specifications named in plan section 0; verify content/ancestry before implementation
- Deferred acceptance: D2/D6

Acceptance:

- [ ] `primitive/quantization/ds41_index.py`
- [ ] `test_index_quantization.py`: the same distinct format-level checks plus input gradients; main-KV off must not disable index QAT
- [ ] Run actual implementation against the S fixtures; record commands, exit status, numerical thresholds and applicable discriminating mutations. GPU tests require Slurm job ID, sacct rc=0 and non-skip execution.

Constraints:

- Post-training scope only; no DSpark forward or rollout generation.
- Full-gate dependencies below are mandatory for acceptance; forward-only partial progress is not completion.

## C2c-I: `primitive/quantization/ds41_fp8.py`

- Dependencies: B1-I, C2c-S
- Direct OPEN: none
- Full-gate OPEN (including prerequisites): none
- Estimated hours: 8–24
- External prerequisites: Corrected A1/A3 and A2/A4 specifications named in plan section 0; verify content/ancestry before implementation
- Deferred acceptance: D2/D4 and G1; BF16 decode+GEMM alone is insufficient

Acceptance:

- [ ] `primitive/quantization/ds41_fp8.py`
- [ ] `test_fp8_quantization.py`: official rounding/scale/saturation and actual Linear GEMM, input/weight derivatives under declared training policy
- [ ] Run actual implementation against the S fixtures; record commands, exit status, numerical thresholds and applicable discriminating mutations. GPU tests require Slurm job ID, sacct rc=0 and non-skip execution.

Constraints:

- Post-training scope only; no DSpark forward or rollout generation.
- Full-gate dependencies below are mandatory for acceptance; forward-only partial progress is not completion.

## D1-I: `model/deepseek_v41/lite/block.py`; attention consumes pre_mix, FFN consumes attn_pre, return ffn_pre, initial/final contraction

- Dependencies: B1-I, B2-I, D1-S
- Direct OPEN: none
- Full-gate OPEN (including prerequisites): O12, O16
- Estimated hours: 8–24
- External prerequisites: Corrected A1/A3 and A2/A4 specifications named in plan section 0; verify content/ancestry before implementation
- Deferred acceptance: E3 paired payload, recompute, PP

Acceptance:

- [ ] `model/deepseek_v41/lite/block.py`; attention consumes pre_mix, FFN consumes attn_pre, return ffn_pre, initial/final contraction
- [ ] `test_hc_boundary.py`: B1 real sublayer/CED values and B2 gradients; distinct residual copies and old/new coefficients make wrong contraction observable
- [ ] Run actual implementation against the S fixtures; record commands, exit status, numerical thresholds and applicable discriminating mutations. GPU tests require Slurm job ID, sacct rc=0 and non-skip execution.

Constraints:

- Post-training scope only; no DSpark forward or rollout generation.
- Full-gate dependencies below are mandatory for acceptance; forward-only partial progress is not completion.

## D5-I: `model/deepseek_v41/lite/moe.py` and generic router capability extension in `primitive/modules/router.py`

- Dependencies: B1-I, B2-I, D5-S
- Direct OPEN: none
- Full-gate OPEN (including prerequisites): O12, O16
- Estimated hours: 8–24
- External prerequisites: Corrected A1/A3 and A2/A4 specifications named in plan section 0; verify content/ancestry before implementation
- Deferred acceptance: E4 cross-rank reduction/atomic step; E5 restart; G2 replay

Acceptance:

- [ ] `model/deepseek_v41/lite/moe.py` and generic router capability extension in `primitive/modules/router.py`
- [ ] `test_moe.py`: bias crosses selected/unselected boundary, unbiased gate weights stay correct; compare two bias updates' signs/values and empty-modality behavior
- [ ] Run actual implementation against the S fixtures; record commands, exit status, numerical thresholds and applicable discriminating mutations. GPU tests require Slurm job ID, sacct rc=0 and non-skip execution.

Constraints:

- Post-training scope only; no DSpark forward or rollout generation.
- Full-gate dependencies below are mandatory for acceptance; forward-only partial progress is not completion.

## F1-I: `primitive/optimizers/headwise_muon.py`

- Dependencies: B2-I, F1-S
- Direct OPEN: none
- Full-gate OPEN (including prerequisites): O12, O16
- Estimated hours: 8–24
- External prerequisites: Corrected A1/A3 and A2/A4 specifications named in plan section 0; verify content/ancestry before implementation
- Deferred acceptance: E4 TP/FSDP logical reassembly

Acceptance:

- [ ] `primitive/optimizers/headwise_muon.py`
- [ ] `test_muon.py`: at least two distinct heads with independently different whole-matrix/head-wise updates; reject always-vanilla; actual backend type and state checked
- [ ] Run actual implementation against the S fixtures; record commands, exit status, numerical thresholds and applicable discriminating mutations. GPU tests require Slurm job ID, sacct rc=0 and non-skip execution.
- [ ] Audit actual DS4 algorithm/group/shape/LR/decay for inherited families; reject unsupported vision mapping or silent numeric defaults. Apply Engram e_proj/e_norm 5x only if trainable.

Constraints:

- Post-training scope only; no DSpark forward or rollout generation.
- Full-gate dependencies below are mandatory for acceptance; forward-only partial progress is not completion.

## C3-I: `model/deepseek_v41/lite/checkpoint_store.py`; 2401 MTP keys in file/CPU carrier and independent storage shards

- Dependencies: C1-I, C3-S
- Direct OPEN: none
- Full-gate OPEN (including prerequisites): none
- Estimated hours: 8–24
- External prerequisites: Corrected A1/A3 and A2/A4 specifications named in plan section 0; verify content/ancestry before implementation
- Deferred acceptance: C4/G1; MTP excluded from trainable coverage but included in checkpoint coverage

Acceptance:

- [ ] `model/deepseek_v41/lite/checkpoint_store.py`; 2401 MTP keys in file/CPU carrier and independent storage shards
- [ ] `test_mtp_store.py`: load/save/repartition bytes identical; original dspark_block_size=5 accepted with enable_dspark_execution=False; execution request raises NotImplementedError
- [ ] Run actual implementation against the S fixtures; record commands, exit status, numerical thresholds and applicable discriminating mutations. GPU tests require Slurm job ID, sacct rc=0 and non-skip execution.

Constraints:

- Post-training scope only; no DSpark forward or rollout generation.
- Full-gate dependencies below are mandatory for acceptance; forward-only partial progress is not completion.

## E1b-I: checkpoint loader row streaming

- Dependencies: C1-I, E0-I, E1b-S
- Direct OPEN: none
- Full-gate OPEN (including prerequisites): none
- Estimated hours: 8–24
- External prerequisites: Corrected A1/A3 and A2/A4 specifications named in plan section 0; verify content/ancestry before implementation
- Deferred acceptance: G1 actual table load

Acceptance:

- [ ] checkpoint loader row streaming
- [ ] `test_table_load.py`: reconstruct source bytes, reject gaps/overlaps without materializing full table per rank
- [ ] Run actual implementation against the S fixtures; record commands, exit status, numerical thresholds and applicable discriminating mutations. GPU tests require Slurm job ID, sacct rc=0 and non-skip execution.

Constraints:

- Post-training scope only; no DSpark forward or rollout generation.
- Full-gate dependencies below are mandatory for acceptance; forward-only partial progress is not completion.
- Engram tables/scales remain GPU-resident; no CPU table offload or host/RDMA prefetch; account for approved state/workspace in memory budget.

## D4-I: `model/deepseek_v41/lite/engram.py`; NFKC/accent removal/lowercase, odd multiplier with 10007*layer_id, prime buckets, token_mask excludes images, no short causal convolution; wkv and per-HC-copy RMS, signed-sqrt sigmoid gate

- Dependencies: B1-I, B2-I, C2c-I, D4-S
- Direct OPEN: none
- Full-gate OPEN (including prerequisites): O12, O16
- Estimated hours: 8–24
- External prerequisites: Corrected A1/A3 and A2/A4 specifications named in plan section 0; verify content/ancestry before implementation
- Deferred acceptance: E1/E2 table training

Acceptance:

- [ ] `model/deepseek_v41/lite/engram.py`; NFKC/accent removal/lowercase, odd multiplier with 10007*layer_id, prime buckets, token_mask excludes images, no short causal convolution; wkv and per-HC-copy RMS, signed-sqrt sigmoid gate
- [ ] `test_engram.py`: known integer hashes plus B1 forward/B2 derivatives, perturb seed/offset/reset and reject specific wrong values
- [ ] Run actual implementation against the S fixtures; record commands, exit status, numerical thresholds and applicable discriminating mutations. GPU tests require Slurm job ID, sacct rc=0 and non-skip execution.

Constraints:

- Post-training scope only; no DSpark forward or rollout generation.
- Full-gate dependencies below are mandatory for acceptance; forward-only partial progress is not completion.

## D2-I: `model/deepseek_v41/lite/attention.py`; explicit source state and CSA2 modes without extra decoder projection

- Dependencies: C2a-I, C2b-I, C2c-I, D1-I, D2-S
- Direct OPEN: none
- Full-gate OPEN (including prerequisites): O12, O15, O16
- Estimated hours: 24–64
- External prerequisites: Corrected A1/A3 and A2/A4 specifications named in plan section 0; verify content/ancestry before implementation
- Deferred acceptance: E3/E4 state transport and kernel ABI; G1

Acceptance:

- [ ] `model/deepseek_v41/lite/attention.py`; explicit source state and CSA2 modes without extra decoder projection
- [ ] `test_attention.py`: B1 CED intermediates and all consumer outputs using injected official candidate masks (no native builder claim), B2 floating KV gradient sums; ranking-reversal fixture crosses Top-K boundary; same-source replay is positive equivalence
- [ ] Run actual implementation against the S fixtures; record commands, exit status, numerical thresholds and applicable discriminating mutations. GPU tests require Slurm job ID, sacct rc=0 and non-skip execution.

Constraints:

- Post-training scope only; no DSpark forward or rollout generation.
- Full-gate dependencies below are mandatory for acceptance; forward-only partial progress is not completion.

## E1a-I: `primitive/modules/engram_lookup.py`

- Dependencies: D4-I, E0-I, E1a-S
- Direct OPEN: none
- Full-gate OPEN (including prerequisites): O12, O16
- Estimated hours: 8–24
- External prerequisites: Corrected A1/A3 and A2/A4 specifications named in plan section 0; verify content/ancestry before implementation
- Deferred acceptance: G1 size

Acceptance:

- [ ] `primitive/modules/engram_lookup.py`
- [ ] Slurm `tests/distributed/deepseek_v41/test_lookup.py`: sharded versus unsharded raw values/scales bitwise, uneven shards/repeated IDs
- [ ] Run actual implementation against the S fixtures; record commands, exit status, numerical thresholds and applicable discriminating mutations. GPU tests require Slurm job ID, sacct rc=0 and non-skip execution.

Constraints:

- Post-training scope only; no DSpark forward or rollout generation.
- Full-gate dependencies below are mandatory for acceptance; forward-only partial progress is not completion.
- Engram tables/scales remain GPU-resident; no CPU table offload or host/RDMA prefetch; account for approved state/workspace in memory budget.

## D3-I: `model/deepseek_v41/lite/candidates.py`

- Dependencies: D2-I, D3-S
- Direct OPEN: none
- Full-gate OPEN (including prerequisites): O12, O15, O16
- Estimated hours: 8–24
- External prerequisites: Corrected A1/A3 and A2/A4 specifications named in plan section 0; verify content/ancestry before implementation
- Deferred acceptance: E4 CP positions

Acceptance:

- [ ] `model/deepseek_v41/lite/candidates.py`
- [ ] `test_candidates.py`: independently compare layer20 pool membership then later Reindex selection; newest reachable block pinned, -inf blocks excluded, incomplete/empty tails handled, pool-outside winner from different later scores; mandatory native D2+D3 consumer integration parity after D3
- [ ] Run actual implementation against the S fixtures; record commands, exit status, numerical thresholds and applicable discriminating mutations. GPU tests require Slurm job ID, sacct rc=0 and non-skip execution.

Constraints:

- Post-training scope only; no DSpark forward or rollout generation.
- Full-gate dependencies below are mandatory for acceptance; forward-only partial progress is not completion.

## D6-I: `model/deepseek_v41/lite/indexer_loss.py`; consume loss autoscaler through protocol

- Dependencies: B2-I, D2-I, D6-S
- Direct OPEN: O12
- Full-gate OPEN (including prerequisites): O12, O15, O16
- Estimated hours: 8–24
- External prerequisites: Corrected A1/A3 and A2/A4 specifications named in plan section 0; verify content/ancestry before implementation
- Deferred acceptance: E3/E4 shared owner gradient delivery; G1 training

Acceptance:

- [ ] `model/deepseek_v41/lite/indexer_loss.py`; consume loss autoscaler through protocol
- [ ] `test_indexer_loss.py`: independent objective value, input/parameter gradients and distinct multiple-consumer sums; reject wrong sign, coefficient, normalization, detach and omitted consumer
- [ ] Run actual implementation against the S fixtures; record commands, exit status, numerical thresholds and applicable discriminating mutations. GPU tests require Slurm job ID, sacct rc=0 and non-skip execution.

Constraints:

- Post-training scope only; no DSpark forward or rollout generation.
- Full-gate dependencies below are mandatory for acceptance; forward-only partial progress is not completion.

## E1c-I: lookup→FP8 GEMM adapter

- Dependencies: C2c-I, E1a-I, E1b-I, E1c-S
- Direct OPEN: none
- Full-gate OPEN (including prerequisites): O12, O16
- Estimated hours: 8–24
- External prerequisites: Corrected A1/A3 and A2/A4 specifications named in plan section 0; verify content/ancestry before implementation
- Deferred acceptance: G1 representative performance

Acceptance:

- [ ] lookup→FP8 GEMM adapter
- [ ] Slurm `test_fp8_lookup.py`: actual values/scales passed to GEMM, independently matched arithmetic with approved thresholds
- [ ] Run actual implementation against the S fixtures; record commands, exit status, numerical thresholds and applicable discriminating mutations. GPU tests require Slurm job ID, sacct rc=0 and non-skip execution.

Constraints:

- Post-training scope only; no DSpark forward or rollout generation.
- Full-gate dependencies below are mandatory for acceptance; forward-only partial progress is not completion.
- Engram tables/scales remain GPU-resident; no CPU table offload or host/RDMA prefetch; account for approved state/workspace in memory budget.

## D7-I: `model/deepseek_v41/lite/packing.py`, protocol shared THD split helper

- Dependencies: D3-I, D4-I, D5-I, D7-S
- Direct OPEN: none
- Full-gate OPEN (including prerequisites): O12, O15, O16
- Estimated hours: 8–24
- External prerequisites: Corrected A1/A3 and A2/A4 specifications named in plan section 0; verify content/ancestry before implementation
- Deferred acceptance: E4 real runtime full-THD→CP-local and non-skip gradient parity

Acceptance:

- [ ] `model/deepseek_v41/lite/packing.py`, protocol shared THD split helper
- [ ] `test_packing.py`: independent versus packed with fixed weights/bias/RNG/loss normalization and isolated auxiliary statistics; backprop only B loss, perturb A, compare B output/input gradients/parameter contribution and integer hash/visibility
- [ ] Run actual implementation against the S fixtures; record commands, exit status, numerical thresholds and applicable discriminating mutations. GPU tests require Slurm job ID, sacct rc=0 and non-skip execution.

Constraints:

- Post-training scope only; no DSpark forward or rollout generation.
- Full-gate dependencies below are mandatory for acceptance; forward-only partial progress is not completion.

## E1d-I: `model/deepseek_v41/lite/prefetch.py`

- Dependencies: E1c-I, E1d-S
- Direct OPEN: none
- Full-gate OPEN (including prerequisites): O12, O16
- Estimated hours: 8–24
- External prerequisites: Corrected A1/A3 and A2/A4 specifications named in plan section 0; verify content/ancestry before implementation
- Deferred acceptance: F3 overlap/G1 performance

Acceptance:

- [ ] `model/deepseek_v41/lite/prefetch.py`
- [ ] Slurm `test_prefetch.py`: nonzero distinct step/microbatch gradient tags; reject lost, duplicate, stale returns; compare actual accumulated vectors
- [ ] Run actual implementation against the S fixtures; record commands, exit status, numerical thresholds and applicable discriminating mutations. GPU tests require Slurm job ID, sacct rc=0 and non-skip execution.

Constraints:

- Post-training scope only; no DSpark forward or rollout generation.
- Full-gate dependencies below are mandatory for acceptance; forward-only partial progress is not completion.
- Engram tables/scales remain GPU-resident; no CPU table offload or host/RDMA prefetch; account for approved state/workspace in memory budget.

## C4-I: `model/deepseek_v41/{config.py,__init__.py,lite/model.py,lite/protocol.py,lite/checkpoint.py,lite/vision.py}` plus `model/registry.py`; compose D modules, backbone, vision and aligner, preserve full module tree in text-only mode

- Dependencies: C3-I, C4-S, D7-I, E1e-S, E2-S
- Direct OPEN: O09
- Full-gate OPEN (including prerequisites): O08, O09, O12, O15, O16, O17
- Estimated hours: 24–64
- External prerequisites: Corrected A1/A3 and A2/A4 specifications named in plan section 0; verify content/ancestry before implementation
- Deferred acceptance: F2 actual optimizer enumeration; E distributed binding and G1 real-size all-key load

Acceptance:

- [ ] `model/deepseek_v41/{config.py,__init__.py,lite/model.py,lite/protocol.py,lite/checkpoint.py,lite/vision.py}` plus `model/registry.py`; compose D modules, backbone, vision and aligner, preserve full module tree in text-only mode
- [ ] `test_model_binding.py`: nested config reaches actual behavior, every key has a header/store/owner mapping, every active parameter is bound once; construct via registry and run protocol; forward/binding first; training export subgate after active decisions close: backbone changed, MTP bytes unchanged
- [ ] Run actual implementation against the S fixtures; record commands, exit status, numerical thresholds and applicable discriminating mutations. GPU tests require Slurm job ID, sacct rc=0 and non-skip execution.

Constraints:

- Post-training scope only; no DSpark forward or rollout generation.
- Full-gate dependencies below are mandatory for acceptance; forward-only partial progress is not completion.

## E1e-I: `model/deepseek_v41/lite/table_state.py`

- Dependencies: B2-I, E1d-I, E1e-S
- Direct OPEN: none
- Full-gate OPEN (including prerequisites): O08, O09, O12, O16, O17
- Estimated hours: 8–24
- External prerequisites: Corrected A1/A3 and A2/A4 specifications named in plan section 0; verify content/ancestry before implementation
- Deferred acceptance: E2/E5 distributed updates

Acceptance:

- [ ] `model/deepseek_v41/lite/table_state.py`
- [ ] `test_table_update.py`: sub-FP8-step repeated updates accumulate, repeated row coalescing, momentum updates unvisited rows, next-step version and restore
- [ ] Run actual implementation against the S fixtures; record commands, exit status, numerical thresholds and applicable discriminating mutations. GPU tests require Slurm job ID, sacct rc=0 and non-skip execution.
- [ ] Obtain evidence or approved policy for post-training Engram trainability before update implementation; O08/O09 remain unresolved. If frozen, obtain explicit scope adjustment and separately audit token embedding/head optimizer needs.
- [ ] Verify native FP32 main_grad rather than widened BF16 gradients, and FP32 momentum; state dtype policy does not resolve table trainability.

Constraints:

- Post-training scope only; no DSpark forward or rollout generation.
- Full-gate dependencies below are mandatory for acceptance; forward-only partial progress is not completion.
- Engram tables/scales remain GPU-resident; no CPU table offload or host/RDMA prefetch; account for approved state/workspace in memory budget.

## E3-I: protocol/pipeline payload extension

- Dependencies: C4-I, E0-I, E3-S
- Direct OPEN: none
- Full-gate OPEN (including prerequisites): O08, O09, O12, O15, O16, O17
- Estimated hours: 24–64
- External prerequisites: Corrected A1/A3 and A2/A4 specifications named in plan section 0; verify content/ancestry before implementation
- Deferred acceptance: E5/G1

Acceptance:

- [ ] protocol/pipeline payload extension
- [ ] Slurm `test_pipeline.py`: different consumer gradient vectors match exact reference sum, single owner update; interleaving/recompute retain p20 and generation tags; stale reads rejected
- [ ] Run actual implementation against the S fixtures; record commands, exit status, numerical thresholds and applicable discriminating mutations. GPU tests require Slurm job ID, sacct rc=0 and non-skip execution.

Constraints:

- Post-training scope only; no DSpark forward or rollout generation.
- Full-gate dependencies below are mandatory for acceptance; forward-only partial progress is not completion.

## F2-I: `model/deepseek_v41/lite/optimizer_groups.py`

- Dependencies: C4-I, E1e-S, E2-S, F1-I, F2-S
- Direct OPEN: O08, O09
- Full-gate OPEN (including prerequisites): O08, O09, O12, O15, O16, O17
- Estimated hours: 8–24
- External prerequisites: Corrected A1/A3 and A2/A4 specifications named in plan section 0; verify content/ancestry before implementation
- Deferred acceptance: E4 mixed backend/E5/G1

Acceptance:

- [ ] `model/deepseek_v41/lite/optimizer_groups.py`
- [ ] `test_optimizer_groups.py`: enumerate C4 actual objects, exactly one justified group per owner, reject unknown/alias duplicate/unresolved active field; no catch-all defaults
- [ ] Run actual implementation against the S fixtures; record commands, exit status, numerical thresholds and applicable discriminating mutations. GPU tests require Slurm job ID, sacct rc=0 and non-skip execution.
- [ ] Use explicit post-training trainability mask; O05/O06 pretraining unfreeze/LR transitions are N/A and do not establish that vision is frozen.
- [ ] Audit actual DS4 algorithm/group/shape/LR/decay for inherited families; reject unsupported vision mapping or silent numeric defaults. Apply Engram e_proj/e_norm 5x only if trainable.
- [ ] Verify native FP32 main_grad rather than widened BF16 gradients, and FP32 momentum; state dtype policy does not resolve table trainability.

Constraints:

- Post-training scope only; no DSpark forward or rollout generation.
- Full-gate dependencies below are mandatory for acceptance; forward-only partial progress is not completion.

## E2-I: `primitive/optimizers/sinkhorn.py`

- Dependencies: E0-I, E1e-I, E2-S
- Direct OPEN: none
- Full-gate OPEN (including prerequisites): O08, O09, O12, O16, O17
- Estimated hours: 24–48
- External prerequisites: Corrected A1/A3 and A2/A4 specifications named in plan section 0; verify content/ancestry before implementation
- Deferred acceptance: E4 mixed backend/E5 restore

Acceptance:

- [ ] `primitive/optimizers/sinkhorn.py`
- [ ] `test_sinkhorn.py` and Slurm `test_sinkhorn_shards.py`: actual W/M after several steps versus independent A4 scalar reference across row/column cuts/replicas
- [ ] Run actual implementation against the S fixtures; record commands, exit status, numerical thresholds and applicable discriminating mutations. GPU tests require Slurm job ID, sacct rc=0 and non-skip execution.
- [ ] Obtain evidence or approved policy for post-training Engram trainability before update implementation; O08/O09 remain unresolved. If frozen, obtain explicit scope adjustment and separately audit token embedding/head optimizer needs.
- [ ] Verify native FP32 main_grad rather than widened BF16 gradients, and FP32 momentum; state dtype policy does not resolve table trainability.

Constraints:

- Post-training scope only; no DSpark forward or rollout generation.
- Full-gate dependencies below are mandatory for acceptance; forward-only partial progress is not completion.

## F3-I: protocol vision schedule consuming C4-owned `model/deepseek_v41/lite/vision.py`; F3 does not define the visual model classes

- Dependencies: B1-I, B2-I, C4-I, F2-I, F3-S
- Direct OPEN: none
- Full-gate OPEN (including prerequisites): O08, O09, O12, O15, O16, O17
- Estimated hours: 24–64
- External prerequisites: Corrected A1/A3 and A2/A4 specifications named in plan section 0; verify content/ancestry before implementation
- Deferred acceptance: E5v/G1

Acceptance:

- [ ] protocol vision schedule consuming C4-owned `model/deepseek_v41/lite/vision.py`; F3 does not define the visual model classes
- [ ] `test_vision.py` then Slurm `test_vision_schedule.py`: official processor forward, independent differentiable serial baseline, real external copies/sync, trainable exceptions and transitions
- [ ] Run actual implementation against the S fixtures; record commands, exit status, numerical thresholds and applicable discriminating mutations. GPU tests require Slurm job ID, sacct rc=0 and non-skip execution.
- [ ] Use explicit post-training trainability mask; O05/O06 pretraining unfreeze/LR transitions are N/A and do not establish that vision is frozen.

Constraints:

- Post-training scope only; no DSpark forward or rollout generation.
- Full-gate dependencies below are mandatory for acceptance; forward-only partial progress is not completion.

## E4-I: generic transport/optimizer adapters plus V4.1 protocol wiring

- Dependencies: E2-I, E3-I, E4-S, F2-I
- Direct OPEN: none
- Full-gate OPEN (including prerequisites): O08, O09, O12, O15, O16, O17
- Estimated hours: 24–64
- External prerequisites: Corrected A1/A3 and A2/A4 specifications named in plan section 0; verify content/ancestry before implementation
- Deferred acceptance: E5/G1

Acceptance:

- [ ] generic transport/optimizer adapters plus V4.1 protocol wiring
- [ ] Slurm `test_parallel_training.py`: real full packed THD split before local CP, forward/backward/update parity; actual mixed optimizer state placement, GPU-resident Engram tables, clip/atomic skip and bias updates; no Engram CPU offload
- [ ] Run actual implementation against the S fixtures; record commands, exit status, numerical thresholds and applicable discriminating mutations. GPU tests require Slurm job ID, sacct rc=0 and non-skip execution.
- [ ] Verify native FP32 main_grad rather than widened BF16 gradients, and FP32 momentum; state dtype policy does not resolve table trainability.

Constraints:

- Post-training scope only; no DSpark forward or rollout generation.
- Full-gate dependencies below are mandatory for acceptance; forward-only partial progress is not completion.

## E5t-I: checkpoint integration

- Dependencies: E4-I, E5t-S
- Direct OPEN: none
- Full-gate OPEN (including prerequisites): O08, O09, O12, O15, O16, O17
- Estimated hours: 8–24
- External prerequisites: Corrected A1/A3 and A2/A4 specifications named in plan section 0; verify content/ancestry before implementation
- Deferred acceptance: G1

Acceptance:

- [ ] checkpoint integration
- [ ] Slurm `test_restart_text.py`: N uninterrupted versus k+save/load+(N-k), same data/RNG/scheduler; momentum/shards/bias/phase/owner state restored, pending prefetch drained/rebuilt
- [ ] Run actual implementation against the S fixtures; record commands, exit status, numerical thresholds and applicable discriminating mutations. GPU tests require Slurm job ID, sacct rc=0 and non-skip execution.

Constraints:

- Post-training scope only; no DSpark forward or rollout generation.
- Full-gate dependencies below are mandatory for acceptance; forward-only partial progress is not completion.

## E5v-I: visual/external encoder optimizer and scheduler restore

- Dependencies: E4-I, E5v-S, F3-I
- Direct OPEN: none
- Full-gate OPEN (including prerequisites): O08, O09, O12, O15, O16, O17
- Estimated hours: 8–24
- External prerequisites: Corrected A1/A3 and A2/A4 specifications named in plan section 0; verify content/ancestry before implementation
- Deferred acceptance: G1

Acceptance:

- [ ] visual/external encoder optimizer and scheduler restore
- [ ] Slurm `test_restart_vision.py`: same-total-step trajectory across save/load with the explicit post-training trainability mask; replicas and LR/state restored
- [ ] Run actual implementation against the S fixtures; record commands, exit status, numerical thresholds and applicable discriminating mutations. GPU tests require Slurm job ID, sacct rc=0 and non-skip execution.
- [ ] Use explicit post-training trainability mask; O05/O06 pretraining unfreeze/LR transitions are N/A and do not establish that vision is frozen.

Constraints:

- Post-training scope only; no DSpark forward or rollout generation.
- Full-gate dependencies below are mandatory for acceptance; forward-only partial progress is not completion.

## G2-I: extend `primitive/modules/router_replay.py` and V4.1 protocol adapters

- Dependencies: C4-I, D5-I, E4-I, G2-S
- Direct OPEN: none
- Full-gate OPEN (including prerequisites): O08, O09, O12, O15, O16, O17
- Estimated hours: 8–24
- External prerequisites: Corrected A1/A3 and A2/A4 specifications named in plan section 0; verify content/ancestry before implementation
- Deferred acceptance: G1 combined training

Acceptance:

- [ ] extend `primitive/modules/router_replay.py` and V4.1 protocol adapters
- [ ] `test_replay.py` and Slurm `test_replay_parallel.py`: actual replay routing map, weights and gradients; D5 auxiliary/phase compatibility
- [ ] Run actual implementation against the S fixtures; record commands, exit status, numerical thresholds and applicable discriminating mutations. GPU tests require Slurm job ID, sacct rc=0 and non-skip execution.

Constraints:

- Post-training scope only; no DSpark forward or rollout generation.
- Full-gate dependencies below are mandatory for acceptance; forward-only partial progress is not completion.

## G1-I: `tests/integration/deepseek_v41/run_acceptance.py` and Slurm recipe

- Dependencies: D6-I, E5t-I, E5v-I, F3-I, G1-S, G2-I
- Direct OPEN: O12, O16
- Full-gate OPEN (including prerequisites): O08, O09, O12, O15, O16, O17
- Estimated hours: 24–64
- External prerequisites: Corrected A1/A3 and A2/A4 specifications named in plan section 0; verify content/ancestry before implementation
- Deferred acceptance: Final acceptance; cannot close on resource readiness alone

Acceptance:

- [ ] `tests/integration/deepseek_v41/run_acceptance.py` and Slurm recipe
- [ ] Real 40-layer/all-key model protocol: all-token text and multi-image forward against B1, loss/input+parameter gradients against declared references, real update/export, long-boundary cases, restart and replay
- [ ] Run actual implementation against the S fixtures; record commands, exit status, numerical thresholds and applicable discriminating mutations. GPU tests require Slurm job ID, sacct rc=0 and non-skip execution.

Constraints:

- Post-training scope only; no DSpark forward or rollout generation.
- Full-gate dependencies below are mandatory for acceptance; forward-only partial progress is not completion.
