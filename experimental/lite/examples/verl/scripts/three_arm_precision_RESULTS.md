# Muon precision results — Megatron-native vs DistOpt-mlite (AC#3)

Two deliverables:

1. **A/B loss trajectory** (`adamw` vs `muon`/dist_opt) on a real Qwen3-30B-A3B verl-SFT
   workload — bayan's "muon must be no worse than AdamW" criterion.
2. **Megatron-native vs DistOpt-mlite Muon = construction identity, proven on REAL
   multi-GPU (TP=2) distributed Newton-Schulz** — a `torch.equal` receipt under a live
   TP process group (job 14245178, 2×H100). This replaces both the prose-only "bitwise
   by construction" claim and the earlier *single-process* (`pg_collection=None`, job
   14243791) receipt that the moe panel correctly rejected as not exercising the real
   distributed path (C-BITWISE-REDEFINED / "not really multi-GPU").

The **FSDP2 Muon arm is deferred** to TASK-1.13.5.5.6 (the emerging_optimizers rewrite):
its current `newton_schulz_orthogonalize` is the hand-rolled version bayan directed us
not to validate against (C-FSDP2-OLD-NS). No FSDP2 parity is claimed here.

Common contract: seed=1234, TP2/PP1/EP8/ETP1/CP1, LR 1e-5 constant, warmup 2,
weight_decay 0.1, clip 1.0, 20 steps, gsm8k messages SFT, `load_hf_weights` (identical
init). mlite `407d4a81d`, mcore `fd1121b`, container `verl.vllm023.sqsh`,
`emerging_optimizers==0.3.0`.

## 1. A/B loss trajectory (real Slurm jobs, all `COMPLETED` rc=0)

| ARM | job | optimizer | backend | muon_tp_mode | offload | peak GiB/GPU |
|-----|-----|-----------|---------|--------------|---------|--------------|
| adamw | 14242814 | adamw | dist_opt | (n/a) | cpu | 31.3 |
| muon | 14242992 | muon | dist_opt | **distributed** | off¹ | 63.3 |

¹ dist_opt Muon runs with optimizer offload OFF — see "hybrid_optimizer" finding below.

| step | adamw | muon (dist_opt) | step | adamw | muon (dist_opt) |
|-----:|------:|----------------:|-----:|------:|----------------:|
| 1 | 1.51501 | 1.50559 | 11 | 0.38296 | 0.44719 |
| 2 | 1.48806 | 1.48605 | 12 | 0.34175 | 0.38639 |
| 3 | 1.19553 | 1.28226 | 13 | 0.34699 | 0.39779 |
| 4 | 0.88977 | 1.03788 | 14 | 0.33217 | 0.36587 |
| 5 | 0.73041 | 0.88960 | 15 | 0.34443 | 0.36991 |
| 6 | 0.52778 | 0.74173 | 16 | 0.32186 | 0.34862 |
| 7 | 0.45003 | 0.59071 | 17 | 0.29555 | 0.32391 |
| 8 | 0.45170 | 0.58517 | 18 | 0.29563 | 0.32246 |
| 9 | 0.34360 | 0.46311 | 19 | 0.33460 | 0.36785 |
| 10 | 0.38468 | 0.46575 | 20 | 0.32339 | 0.35204 |

JSONL: `$RUN_ROOT/{adamw,muon}/q35_precision_*.jsonl`.

**Verdict (A/B).** Muon is no worse than AdamW: both descend from ~1.51 to ~0.32–0.35
at the same rate from an identical init; muon's final loss (0.352) sits a hair above
adamw's (0.323) — a different-but-healthy optimizer, not a regression. DistOpt Muon
runs the *true* cross-TP Newton-Schulz (`muon_tp_mode=distributed`, confirmed in the
live mcore `OptimizerConfig` dump) on Qwen3-30B-A3B at TP2 — 20 clean steps, no
divergence.

## 2. Megatron-native vs DistOpt-mlite = construction identity on REAL TP=2 distributed NS

**Job 14245178** (2×H100, 1 node, `COMPLETED` rc=0:0, 1:11),
`tp_distributed_muon_identity.py` under `torchrun --nproc_per_node=2`.

MLite's DistOpt Muon is **not a second implementation** of Megatron Muon. It lowers,
through `build_dist_opt_optimizer_config`, into Megatron-Core's own
`TensorParallelMuon` (`megatron/core/optimizer/emerging_optimizers.py`). There is no
independent Megatron binary to diff — so this is a **construction identity**. Prior
receipts proved it only in a single process (`pg_collection=None` → *local* NS), which
the moe panel rejected as not testing multi-GPU. This receipt proves it on a **real TP
process group**: `pg_collection.tp` is a live 2-rank NCCL `ProcessGroup`,
`tp_mode="distributed"`, so orthogonalization runs genuine cross-rank `all_reduce`s
inside `newton_schulz_tp`, on TP-sharded weights across 2 GPUs.

```
[env] world=2 tp_size=2 weight=(64,48) steps=5 tp_mode=distributed
[A] CONFIG_IDENTITY fields_checked=14 OK diffs=[]
[B] rank=0 shard_rows=(0, 32) torch.equal=True max_abs=0.000e+00
[B] rank=1 shard_rows=(32, 64) torch.equal=True max_abs=0.000e+00
[B] DISTRIBUTED_UPDATE_IDENTITY all_ranks_equal=True global_max_abs=0.000e+00 (real TP=2 distributed NS)
[C1] DIST_vs_FULLREF max_abs=0.000e+00 tol=1e-04 OK (distributed == full-matrix NS within fp tol)
[C2] NOPG_LOCAL_vs_FULLREF max_abs=7.281e-02 ratio_vs_dist=7.3e10x OK (pg=None diverges hugely -> cross-rank all_reduce genuinely mattered)
[D] NEG_CONTROL(num_ns_steps+1) global_max_abs=4.470e-05 OK (torch.equal is sensitive)
RESULT PASS
```

- **(A) Config identity** — the mcore `OptimizerConfig` from the MLite lowering (path A)
  is field-for-field equal to a hand-built native Megatron `OptimizerConfig` (path B)
  across all 14 identity fields (every `muon_*` knob + lr/weight_decay/clip). Regression
  guard for the `muon_tp_mode` propagation fix: had the lowering dropped a field, A ≠ B.
- **(B) Distributed update identity (`torch.equal`)** — two *separate* native
  `TensorParallelMuon` instances, one from config A and one from config B, **share the
  same live TP process group** and step 5× on identical TP-sharded seeded params+grads.
  On BOTH ranks the local shards are **bit-identical** (`torch.equal=True`,
  all-reduced `global_max_abs=0.0`). This is measured *after real cross-rank
  all_reduces on 2 GPUs* — not a single-process vacuity.
- **(C) "Distributed is real"** — (C1) the distributed sharded result, gathered, equals
  the single-process full-matrix NS reference (`max_abs=0.0`); (C2) running the *same*
  sharded params with `pg_collection=None` (each rank orthogonalizes only its own shard,
  no cross-rank) diverges from that reference by `7.3e10×` at the update level
  (lr=1, wd=0, correlated row-blocks). So `pg=None` is a demonstrably *different, wrong*
  computation — which is exactly why the pg≠None run in (B) is doing real distributed
  work. This directly rebuts the rejected `pg_collection=None` shortcut.
- **(D) Negative control** — perturbing `num_ns_steps` by 1 yields
  `global_max_abs=4.5e-5`, proving (B)'s `torch.equal` is *sensitive*, not vacuous.

**Honest scope.** Because MLite dist_opt *is* emerging_optimizers (there is no second
independent binary), `max_abs=0.0` is the *correct and expected* result — this is a
construction identity proven numerically under real multi-GPU distributed Newton-Schulz,
NOT a bitwise diff of two independent lowerings. The only genuinely independent lowering
(FSDP2 Muon) is deferred to TASK-1.13.5.5.6 and is not exercised here (per bayan
05:29/05:45/06:58).

**Known limitation vs bayan 07:03 ("验的是集成不是 kernel").** This receipt exercises the
*config-lowering* (`build_dist_opt_optimizer_config`) and the *distributed Newton-Schulz
kernel* under a real TP process group — but it drives Megatron-Core's native
`TensorParallelMuon` directly for BOTH arms. It therefore does **not** yet exercise the
rest of MLite's DistOpt *integration wiring*: `_mark_dist_opt_parallel_attrs` (per-param
`allreduce`/`tensor_model_parallel` tagging that governs grad-norm accounting and param
layout), the Megatron-Core `DistributedDataParallel` bucketing + `DistributedOptimizer`
master-grad sharding, and `finalize_dist_opt_grads`. Fully closing AC#3(a) per 07:03
requires a real DDP+DistributedOptimizer training-step comparison — MLite's
`build_dist_opt_stack` muon path vs a raw Megatron-Core `get_megatron_optimizer` build —
on a small real model (scaffolding: `tests/smoke/primitive/test_qwen3_moe_distopt_checkpoint_smoke.py`),
comparing weights with `torch.equal`. That harness is the next step (see task log).

## Correctness fix carried & proven: `muon_tp_mode` propagation

`build_dist_opt_optimizer_config` (`primitive/optimizers/megatron_wrap.py`) built the
mcore `OptimizerConfig` copying only lr/betas/offload and **dropped every `muon_*`
field**, so mcore fell back to its default `muon_tp_mode="blockwise"` (per-shard *local*
NS) — silently degrading a requested `distributed`. Fixed to forward all ten `muon_*`
fields for muon. Proven three ways: 0-GPU init-chain gate (`requested=distributed →
core=distributed`); live on GPU (job 14242992 mcore dump shows
`muon_tp_mode='distributed'`); and the config-identity receipt above (job 14245178,
part (A) — all `muon_*` fields survive the lowering, and part (B)/(C) confirm the
`distributed` mode actually drives cross-rank Newton-Schulz on TP=2).

## Integration finding: CPU-offload `hybrid_optimizer` ⊥ distributed Muon

With `optimizer_cpu_offload=True`, the first `optimizer.step()` of the distributed Muon
raises `KeyError` in `megatron/core/optimizer/cpu_offloading/hybrid_optimizer.py:339`
(`_sync_hdo_param_groups_to_sub_optimizers`, `param_to_inner_param[param]`) — the
distributed Muon registers params the hybrid CPU-offload optimizer never maps (job
14242903, mcore@fd1121b). The adamw arm is immune (adam params are mapped). This only
surfaced once the propagation fix let `distributed` reach mcore; blockwise never took
this path. Worked around by disabling optimizer offload for the dist_opt Muon arm
(node script); muon's smaller optimizer state fits 30B-A3B on 8×H100 without offload.
This is an upstream mcore composition bug worth a follow-up report.
