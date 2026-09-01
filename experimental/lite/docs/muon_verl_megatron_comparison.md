# Muon in verl + Megatron vs. Megatron-Lite: a comparison study

> Zero-GPU source study. No new training run was performed for this document;
> the Megatron-Lite numerical evidence it cites comes from the prior task runs
> referenced inline. It compares three things: (1) how the public **verl** RL
> framework drives Muon on its Megatron backend, (2) the current state of
> **NVIDIA Megatron-LM** Muon support relative to the 2026-07-09 pin our earlier
> study was written against, and (3) our own **Megatron-Lite (MLite)** native
> Muon contract, DistOpt lowering, FSDP2 lowering, post-training recipe, and
> validation depth. The goal is an explicit gap-and-borrow list.

## Baselines and terminology

This study builds on two earlier MLite studies and the shipped MLite Muon
implementation:

- **Megatron interface study** — `experimental/lite/docs/muon_optimizer.md`,
  pinned to NVIDIA/Megatron-LM `dev`
  `d64ba4ccb1e3e878c15171c9cc58d5d3b46bf4d5` (fetched 2026-07-09) and
  NVIDIA-NeMo/Emerging-Optimizers v0.3.0 `b309e2f01cda75dc96a6dc1a2355a7b3b64b5e16`.
- **Post-training recipe study** — `experimental/lite/docs/muon_post_training.md`.
- **Shipped MLite Muon** — branch
  `feature/muon-adam-distopt-fsdp2` (integration HEAD `a872cd34d`; native
  contract commit `e4e96814f`). Its numerical acceptance evidence is in
  `docs/runs/muon_distopt_compact_bitwise_evidence.md`.

Conventions (`ZeRO-1`/`DistOpt`, `ZeRO-3`/`FSDP2`, `TP sharding`) are used as
defined in `muon_optimizer.md`.

## 1. verl + Megatron: does it use Muon? — No.

**Public verl has no Muon support on its Megatron backend.** This is the single
most important finding for the "verl + Megatron Muon" comparison the task asks
for: the thing being compared against does not exist upstream today.

Read at `verl-project/verl` HEAD `04bac4ee35e155841b3586b66e91350cfdbe1c76`
(note: `volcengine/verl` was migrated/renamed to the `verl-project` org in
2026/01 and now redirects to the same repo — **same codebase, not a divergent
fork**; `README.md:57`).

**Config surface (Megatron).** verl's Megatron optimizer config is a thin
subclass of Megatron-core's own `OptimizerConfig`:

- `verl/workers/config/optimizer.py:128` `class McoreOptimizerConfig(OptimizerConfig)`
  adds only scheduler / precision-aware / `override_optimizer_config` fields;
  `optimizer: str = "adam"`. The YAML surface is
  `verl/trainer/config/optim/megatron.yaml` (`optimizer: adam`, betas, clip_grad,
  precision-aware dtypes, `override_optimizer_config: {}`). There are **no**
  `muon_*` fields and **no** emerging-/layer-wise-optimizer args.

**Construction.** verl does **not** build optimizers itself for Megatron — it
forwards to Megatron-core:

- `verl/utils/megatron/optimizer.py:31-38` copies `optim_config.optimizer` into
  `optim_args`, `:97` builds `OptimizerConfig(**optim_args)`, `:101-109` calls
  the native `get_megatron_optimizer`. `init_megatron_optim_config(...,
  use_distributed_optimizer=True)` (`:27,37`) uses Megatron's native ZeRO-1
  DistOpt. Construction site: `verl/workers/engine/megatron/transformer_impl.py:371-386`.
- At verl's pinned Megatron-core (`core_r0.13.0`), the core factory
  `get_megatron_optimizer` accepts only `'adam'`/`'sgd'` and raises
  `"{} optimizer is not supported."` otherwise. So setting `optimizer: muon` in
  verl today would raise at construction; there is no code path routing it to a
  Muon implementation. (This release tag also predates the dev
  emerging-optimizer/LayerWise routing our contract pins, so verl's Megatron is
  behind on the Muon code itself, not just the config surface.)

**RL-scenario handling.** There is **no Muon-specific handling anywhere**. The
train↔rollout resharding offloads optimizer state generically and Adam-shaped
(`verl/utils/megatron_utils.py:663` `offload_megatron_copy_params`, ChainedOptimizer-aware);
nothing keys off optimizer type. Because the Megatron actor optimizer is always
Adam/SGD, there is no Muon–resharding interaction in verl.

**Status.** Muon is an **open, unimplemented feature request**
(`verl-project/verl#3246`, opened 2025-08-28, no linked PR). The only `muon`
string in the entire repo is a comment
(`verl/workers/engine_workers_tinker.py:70`) about the **external** VeOmni
package's `MultiOptimizer` on verl's *non-Megatron* (torch/FSDP) VeOmni engine —
verl forwards to `veomni.optim.build_optimizer`, it does not build Muon itself,
and it is unrelated to Megatron.

**Consequence for us.** The in-repo `examples/verl/verl_mlite` engine (§3.4) is,
as far as this study found, the *only* place where a verl-style optimizer config
drives **Megatron-native Muon** — MLite is strictly ahead of public verl on this
axis, and there is no upstream verl Muon-on-Megatron design to borrow from.


## 2. Megatron-LM upstream: delta vs. the pinned `d64ba4ccb` baseline

**Headline: the delta is essentially zero.** The interface study
(`muon_optimizer.md`) pinned `dev` at
`d64ba4ccb` on 2026-07-09. The current `dev` HEAD is
`fd1121b8ff7e3a4f83a28d35aed172d7bc0260e1` (also dated 2026-07-09). Every
Muon-relevant source file is **byte-for-byte identical** between the two SHAs
(verified by fetching raw source at both explicit SHAs and diffing):
`megatron/training/arguments.py`, `core/optimizer/emerging_optimizers.py`,
`core/optimizer/layer_wise_optimizer.py`, `core/optimizer/optimizer_config.py`,
`core/optimizer/__init__.py`, and `pyproject.toml`. So the pinned `d64ba4ccb`
baseline sits at (or within a day of) the current dev tip for Muon purposes, and
nothing below has moved.

| Baseline (`d64ba4ccb`) finding | Current dev tip (`fd1121b8`) |
| --- | --- |
| Emerging-Optimizers pinned at v0.3.0 (`b309e2f`) via `pyproject.toml:203` | **Unchanged** — still `rev = "v0.3.0"` |
| `muon_scalar_optimizer` declared (`adam`/`lion`) but hard-coded to adam | **Unchanged** — `arguments.py:3274-3279` still `choices=['adam','lion']`; Lion is an unused import (`emerging_optimizers.py:34`); routing hard-codes `{'optimizer':'adam'}` (`emerging_optimizers.py:86,440,452`). Still **not wired**. |
| Momentum default mismatch: CLI `0.9` vs `OptimizerConfig` `0.95` | **Unchanged** — `arguments.py:3215` (0.9) vs `optimizer_config.py:259` (0.95). Still unfixed. |
| Muon+DistOpt = whole-matrix owner LayerWise; compact default, padded opt-in | **Unchanged** — `--optimizer dist_muon` still the deprecated alias auto-converting to `muon` + `use_layer_wise_distributed_optimizer=True` (`arguments.py:1875-1880,3686-3688`); compact default, padded via `--use-layer-wise-param-layout` (`arguments.py:1906-1909,4227`). |
| Limitations: FP8/FP4 gather rejected, one DistOpt instance, torch/torch_dist ckpt only | **Unchanged** — `arguments.py:1898-1901` (fp8/fp4), `1906-1908` (single instance), `1894-1897` (ckpt formats). |
| Muon **rejected** with PyTorch FSDP2 and Megatron-FSDP | **Unchanged** — the two `assert not args.use_torch_fsdp2 / use_megatron_fsdp` guards are still at `arguments.py:1890-1893`. No support landed. |
| Unmerged experiment `FSDPTensorParallelMuon` (`c7d1aff65090`) | **Did not merge** — the identifier appears in none of the dev Muon files, which are identical to baseline. |
| Default recipe: spectral / 5 NS / split-qkv / extra-scale 1.0 / blockwise / quintic / no-nesterov / medium matmul | **Unchanged** — `optimizer_config.py:262-285`. |

Implication: our pinned-`d64ba4ccb` contract is not stale, and there is no new
upstream Muon capability we need to catch up to as of this study. (SHA confirmed
from the deterministic `dev.atom` commit feed; `gh api` was SAML-blocked, so all
comparisons used pinned-SHA raw source, reproducible.)


## 3. MLite native Muon: interface, lowering, recipe, validation

MLite's Muon is not a re-implementation of the algorithm bolted onto a backend.
It reuses Megatron-Core's own Muon numerics and, for DistOpt, Megatron-Core's own
`LayerWiseDistributedOptimizer`; it adds a **native configuration contract** and
two **backend-specific lowerings** (DistOpt compact, FSDP2 bounded gather).

### 3.1 Native configuration + routing contract (commit `e4e96814f`)

The algorithm is a first-class field on the shared `OptimizerConfig`, not a
separate backend, exactly as `muon_optimizer.md` recommended:

- `OptimizerConfig.optimizer: str = "adam"` selects the algorithm; `"muon"` is
  now accepted
  (`megatron/lite/runtime/contracts/config.py:47`).
- Explicit Muon fields with **one** authoritative default each — no reliance on
  entry-point-dependent defaults:
  `muon_momentum=0.95`, `muon_split_qkv=True`, `muon_nesterov=False`,
  `muon_scale_mode="spectral"`, `muon_fp32_matmul_prec="medium"`,
  `muon_coefficient_type="quintic"`, `muon_num_ns_steps=5`,
  `muon_tp_mode="blockwise"`, `muon_extra_scale_factor=1.0`,
  `muon_scalar_optimizer="adam"`
  (`config.py:64-73`).
- The "inactive knob" hazard flagged in `muon_optimizer.md` is closed: MLite
  **rejects** any `muon_scalar_optimizer != "adam"` at config-validation time
  (`config.py:110-111`) rather than silently ignoring it as upstream did.
- Note MLite deliberately picks `muon_momentum=0.95` (the standalone MCore
  `OptimizerConfig` value), not the training-CLI `0.9`. This is a single explicit
  choice, but callers running post-training should override per the recipe below
  (NeMo post-training uses `0.9`).

Parameter routing is intentionally thin. `tag_muon_parameter_metadata`
(`megatron/lite/primitive/optimizers/muon_routing.py:13-32`) tags **only** the
caller-provided expert classification (`param.expert_tp = True`) and otherwise
defers to module-owned semantic metadata (vocab primitives mark their own
embedding/output weight; attention primitives mark their own fused-QKV weight)
and to Megatron-Core's own buffer-routing pass. This respects the MLite
`primitive` skill's layering rule: the optimizer primitive does **not** hardcode
model-name lists or re-derive which tensor is an embedding — it consumes metadata
the model modules already own.

### 3.2 DistOpt (ZeRO-1) lowering — compact, upstream-reused

`build_dist_opt_stack`
(`megatron/lite/primitive/optimizers/megatron_wrap.py:242+`) is a three-step
lowering that **reuses the upstream cohesive stack** rather than porting only
`TensorParallelMuon`:

1. **Tag** — `tag_muon_parameter_metadata` marks expert params, then
   `_mark_dist_opt_parallel_attrs` + upstream
   `layer_wise_optimizer.tag_params_for_buffer_routing(model_chunks)` split the
   model into the Muon-matrix buffer domain and the Adam-fallback byte domain
   (`megatron_wrap.py:322-329`).
2. **Layout** — upstream
   `LayerWiseDistributedOptimizer.compute_full_param_layout(...)` computes the
   whole-matrix DP ownership layout per chunk, using the `dp_cp` mesh for dense
   and the `expt_dp` mesh for expert matrices (`megatron_wrap.py:331-345`).
3. **Build** — DDP wrap (`use_distributed_optimizer=True`) with the layout, then
   `get_megatron_optimizer` produces the chained `LayerWiseDistributedOptimizer`
   (Muon owner-runs-NS) + standard DistOpt (Adam fallback).

Scope is deliberately bounded to the **compact decoupled layout only**, matching
the `muon_optimizer.md` recommendation. `validate_dist_opt_config`
(`megatron_wrap.py:32-67`) fails loud on everything not yet lowered: padded
LayerWise layout, `overlap_grad_reduce`, `overlap_param_gather`,
gather-overlap-with-step, fp8/fp4 param gather, precision-aware optimizer, and
optimizer offload (deferred to the dedicated offload lowering). `build_dist_opt_optimizer_config`
also asserts the pinned-`d64ba4ccb` Muon field set is present before it will
construct the config (`megatron_wrap.py:200-221`) — a fail-loud pin guard.

### 3.3 FSDP2 (ZeRO-3) lowering — bounded gather / NS / reshard

Upstream **rejects** Muon on FSDP2 entirely. MLite implements the
gather-orthogonalize-reshard path `muon_optimizer.md` proposed, in
`megatron/lite/primitive/optimizers/fsdp2/muon.py`:

- Momentum EMA, decoupled weight decay, and the final parameter update are
  element-wise on the **local DTensor shard** (sharding commutes → no comms).
- Orthogonalization is not element-wise, so the momentum is all-gathered into the
  full matrix **one parameter at a time** (bounded peak memory), the identical
  Newton-Schulz runs on the full matrix on every FSDP rank, and the
  orthogonalized update is resharded to the local placement — reusing the
  parameter's own DTensor mesh/placement (dense over dense DP/CP mesh, expert over
  expert-DP mesh), no new transport API, and never gathering the parameter itself
  (FSDP2's next forward does that).
- Non-matrix / embedding / output params fall back to `FP32AdamW`; the two
  children are composed under one `FSDP2Optimizer` facade so the runtime sees one
  `step`/`grad-norm`/`zero`/`state` surface.

The Newton-Schulz coefficient sets and iteration mode are mirrored verbatim from
the pinned Emerging-Optimizers reference (`fsdp2/muon.py` `_NS_COEFFICIENT_SETS`),
including `simple`/`quintic`/`polar_express`/`cans`/`aol`/`deepseekv4`.

### 3.4 verl → MLite → Megatron-native Muon (in-repo engine)

The in-repo verl adapter `examples/verl/verl_mlite` is the actual RL entry point.
`MegatronLiteEngine._build_optimizer_config` (`engine/mlite_engine.py`, contract
diff in `e4e96814f`):

- `_normalize_optimizer_name` now accepts `"muon"` in addition to Adam-family
  (previously Adam-only).
- verl's `optim.override_optimizer_config` dict is filtered through an explicit
  `native_override_fields` allow-list (all `muon_*` fields plus overlap/offload
  fields) and forwarded to `MegatronLiteOptimizerConfig` as Megatron-native
  fields — i.e. verl config flows straight into the Megatron-native Muon contract
  with no re-implementation, and the legacy `offload_fraction` alias is
  reconciled against the canonical `optimizer_offload_fraction` with a conflict
  guard.

This means: for MLite, "verl + Megatron Muon" is a real, wired path today (verl
optim config → native Muon contract → LayerWise DistOpt / FSDP2 lowering).

### 3.5 Validation depth

The DistOpt lowering is validated **bitwise** against upstream, not merely
"step() returns":

- Slurm job **13875018** (`COMPLETED 0:0`, H100×2, DP=2) compares
  `build_dist_opt_stack` against upstream `TensorParallelMuon` across
  `continuous`/`save`/`resume` trajectories: `tensor_checks=2000`,
  `torch_equal_checks=2000`, `assert_close_checks=2000` at **atol=0, rtol=0**,
  `mismatches=0`, marker `NON_SKIP_MUON_DISTOPT_COMPACT_BITWISE_PASSED`. Mixed
  routing is exercised (4 weight matrices → Muon; embedding/output/bias/norm →
  Adam, with their `exp_avg`/`exp_avg_sq` moments in the comparison).
  (`docs/runs/muon_distopt_compact_bitwise_evidence.md`.)
- A real-Qwen3.5 Adam DistOpt gate passed on the compact parent
  (job **13697707**, marker `NON_SKIP_PINNED_ADAM_DISTOPT_GATE_PASSED`); a re-run
  on the integrated tree is blocked only by a missing HF checkpoint path
  (environment, not code).
- FSDP2 Muon has CPU-offload lifecycle GPU unit tests and bf16/checkpoint/offload
  roundtrip tests (commits `9696b99fc`, `b858c0525`, `84f02c7c7`), but its
  numerical parity is against a single-rank full-matrix reference at unit scale,
  not a large real-model E2E.

## 4. Item-by-item comparison

| Axis | Public verl (Megatron) | Megatron-LM upstream `dev` | MLite (`feature/muon-adam-distopt-fsdp2`) |
| --- | --- | --- | --- |
| Muon supported at all? | **No** — Adam/SGD only; FR #3246 open | **Yes** — DDP + LayerWise DistOpt | **Yes** — DistOpt (compact) + FSDP2 |
| Config surface | Passthrough `OptimizerConfig`, no `muon_*` | CLI `--muon-*` + `OptimizerConfig.muon_*` | `OptimizerConfig.optimizer="muon"` + explicit `muon_*` fields (`config.py:64-73`) |
| Algorithm vs backend | N/A | Coupled through CLI flags | **Orthogonal axes**: algorithm field + per-backend lowering (the `muon_optimizer.md` design) |
| Parameter routing | N/A | ndim==2, non-emb/out → Muon; module/heuristic tagging | Module-owned metadata + thin expert tag (`muon_routing.py`); no model-name hardcoding |
| DistOpt (ZeRO-1) lowering | Native Adam DistOpt only | Whole-matrix owner LayerWise; compact default + padded opt-in | **Reuses** upstream `LayerWiseDistributedOptimizer` + `tag_params_for_buffer_routing` + `compute_full_param_layout`; compact-only, padded/offload/overlap fail-loud (`megatron_wrap.py:32-67,242+`) |
| FSDP2 (ZeRO-3) lowering | — | **Rejected** (assert, `arguments.py:1890-1893`) | **Implemented** — bounded gather / duplicated NS / reshard, one matrix at a time, Adam fallback under one facade (`fsdp2/muon.py`) |
| M-FSDP lowering | — | **Rejected** | Not yet (design space; `mfsdp` backend TBD) |
| `muon_scalar_optimizer` | — | Declared `adam`/`lion` but **hard-coded adam** (dead knob) | **Rejects** non-`adam` at validation (`config.py:110-111`) — no dead knob |
| Momentum default | — | **Ambiguous**: CLI 0.9 vs OptimizerConfig 0.95 | Single explicit `0.95`; recipe overrides to `0.9` for post-training |
| Post-training recipe | — | Pre-training-centric defaults | Dedicated SFT/DAPO recipe with named LR convention (`muon_post_training.md`) |
| RL / verl integration | Adam only; no Muon in reshard/offload | N/A (pre-training framework) | verl config → native Muon contract wired (`mlite_engine.py`, `e4e96814f`) |
| Validation depth | N/A | Upstream tests | **Bitwise** DistOpt A/B vs `TensorParallelMuon` (job 13875018, 2000 checks atol=rtol=0); FSDP2 unit + offload lifecycle |



## 5. Gap-and-borrow list (for bayan)

### Where MLite is ahead — keep, and treat as our differentiators

1. **Muon on FSDP2 exists in MLite and nowhere upstream.** Both Megatron `dev`
   and verl reject or lack it. This is our clearest lead; the bounded
   gather/NS/reshard design is the thing to defend and productionize.
2. **verl-style RL config → Megatron-native Muon is wired only in MLite**
   (`verl_mlite` engine). Public verl has no Muon on Megatron at all (FR #3246).
   If Muon-in-RL is a goal, MLite is the vehicle; there is no upstream verl path
   to wait on or copy.
3. **Two design bugs upstream still ships, MLite fixed:** the dead
   `muon_scalar_optimizer=lion` knob (MLite rejects it) and the 0.9/0.95 momentum
   default ambiguity (MLite is explicit). Do not regress these when syncing.
4. **Algorithm/backend orthogonality + module-owned routing** (no model-name
   hardcoding in the optimizer primitive) is cleaner than upstream's CLI-flag
   coupling and satisfies the MLite `primitive` layering rule.
5. **Bitwise DistOpt parity evidence** (atol=rtol=0 vs `TensorParallelMuon`) is a
   stronger acceptance bar than "step() returns"; this is the standard to hold
   the FSDP2 path to as well.

### What to borrow / stay aligned with upstream — we already do, keep pinned

6. **DistOpt = upstream `LayerWiseDistributedOptimizer`, not a re-port.** MLite
   correctly reuses `tag_params_for_buffer_routing` + `compute_full_param_layout`
   + the compact layout rather than copying only `TensorParallelMuon`. Keep this
   coupling; it is why the bitwise A/B passes. Re-run the pin guard whenever the
   upstream pin moves.
7. **Numerics pinned to Emerging-Optimizers `b309e2f` (v0.3.0).** Upstream has
   **not** bumped this pin (delta = 0 as of `fd1121b8`), so our contract is
   current. The FSDP2 NS coefficient tables are mirrored verbatim from it — audit
   them against the pin on any Emerging-Optimizers bump.

### What we have not caught up on / open risks

8. **Padded LayerWise layout is deferred** (compact-only). Upstream keeps padded
   as an opt-in; if a real workload needs the shard-aligned reduce-scatter path,
   this is the first thing to add — the validator already fails loud on it.
9. **Muon optimizer offload on DistOpt is deferred** to a dedicated lowering
   (`megatron_wrap.py:67`). For RL train↔rollout resharding (the verl_mlite use
   case), offload matters; this is a concrete follow-up.
10. **FSDP2 Muon lacks large-model E2E numerical parity.** Its evidence is
    single-rank full-matrix unit reference + offload lifecycle, not a real-model
    distributed A/B like the DistOpt path has. Close this before treating FSDP2
    Muon as production, especially given the RLVR-collapse risk in the recipe.
11. **M-FSDP Muon is unbuilt** and upstream rejects it — pure design space. Any
    future `mfsdp` backend must reuse the same algorithm/routing contract and add
    an uneven-DTensor gather/reshard, not a third Muon implementation.

### Post-training / RL recipe alignment (cross-check with the DistOpt validation)

12. The recipe (`muon_post_training.md`) is deliberately conservative: **AdamW is
    the production recommendation for RL**, and the Muon RL row is a bounded
    experiment with stop-gates (RLVR collapse and Adam→Muon mismatch papers
    postdate the pre-training recipes). This is consistent with the DistOpt
    bitwise-validation study (`muon_distopt_compact_bitwise_evidence.md`), whose
    acceptance is *numerical parity* ("not worse than Megatron Muon = bitwise"),
    **not** an efficacy claim that Muon improves RL. The two artifacts agree:
    MLite has proven the Muon *implementation* is correct, not that Muon is the
    right optimizer for RL. Any decision to default RL to Muon needs the efficacy
    experiment the recipe scopes, not the parity evidence we already have.

## Bottom line

- **AC#1 (verl):** verl has no Muon on Megatron — Adam/SGD passthrough, FR #3246
  open. Our `verl_mlite` engine is the only verl→Megatron-native-Muon path.
- **AC#2 (Megatron delta):** zero — `d64ba4ccb` ≈ current dev tip `fd1121b8` for
  Muon; our pinned contract is not stale.
- **AC#3 (MLite compare):** MLite reuses upstream DistOpt LayerWise, adds the
  FSDP2 path upstream rejects, fixes two upstream design bugs, and proves DistOpt
  parity bitwise.
- **AC#4 (borrow list):** items 1–12 above. Highest-value follow-ups: FSDP2
  large-model parity (#10) and DistOpt Muon offload for RL resharding (#9).
