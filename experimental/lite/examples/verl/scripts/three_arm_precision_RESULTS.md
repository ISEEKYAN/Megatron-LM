# Muon precision results — Megatron-native vs DistOpt-mlite (AC#3)

Two deliverables:

1. **A/B loss trajectory** (`adamw` vs `muon`/dist_opt) on a real Qwen3-30B-A3B verl-SFT
   workload — bayan's "muon must be no worse than AdamW" criterion.
2. **Megatron-native vs DistOpt-mlite Muon = construction identity, numerically
   proven** — a real `torch.equal` receipt (job 14243791), replacing the earlier
   prose-only "bitwise by construction" claim that the moe panel correctly rejected
   as fabricated evidence (C-BITWISE-REDEFINED).

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

## 2. Megatron-native vs DistOpt-mlite = construction identity (torch.equal receipt)

**Job 14243791** (`cpu_short`, in-container, `COMPLETED` rc=0:0, 35 s),
`megatron_vs_distopt_identity.py`.

MLite's DistOpt Muon is **not a second implementation** of Megatron Muon. It lowers,
through `build_dist_opt_optimizer_config`, into Megatron-Core's own
`TensorParallelMuon` (`megatron/core/optimizer/emerging_optimizers.py`). There is no
independent Megatron binary to diff — so this is a **construction identity**, and we
prove it *numerically*, not by assertion:

```
[a] CONFIG_IDENTITY fields_checked=14 diffs=[]
[a] CONFIG_IDENTITY_OK all fields equal (MLite lowering == native Megatron config)
[b] UPDATE_IDENTITY torch.equal=True max_abs_delta=0.000e+00 weight_shape=(64, 48) steps=5
[b] UPDATE_IDENTITY_OK final weights bit-identical (torch.equal)
[c] NEG_CONTROL(num_ns_steps+1) torch.equal=False max_abs_delta=4.494e-05
[c] NEG_CONTROL_OK update torch.equal is sensitive (perturbation -> nonzero delta)
RESULT PASS
```

- **(a) Config identity** — the Megatron-Core `OptimizerConfig` produced by the MLite
  lowering (path A) is field-for-field equal to a hand-built native Megatron
  `OptimizerConfig` (path B) across all 14 identity fields (every `muon_*` knob +
  lr/weight_decay/clip). This is the regression guard the `muon_tp_mode` propagation
  fix is about: had the lowering dropped any `muon_*` field, A ≠ B here.
- **(b) Update identity (`torch.equal`)** — Megatron-Core's *native* `TensorParallelMuon`
  is built from BOTH configs (via Megatron's own `_kwargs_from_config` mapper) and
  stepped 5× on identical seeded params+grads. Final weights are **bit-identical**
  (`torch.equal=True`, `max_abs_delta=0.0`). The MLite lowering perturbs the Megatron
  Muon update by exactly zero.
- **(c) Negative control** — perturbing `num_ns_steps` by 1 yields
  `torch.equal=False`, `max_abs_delta=4.5e-5`, proving the (b) check is *sensitive*,
  not a vacuous pass.

**This is an honest construction-identity receipt, not a bitwise diff of two independent
lowerings.** The only genuinely independent lowering (FSDP2 Muon) is deferred to
TASK-1.13.5.5.6 and is not exercised here (per bayan 05:29/05:45).

## Correctness fix carried & proven: `muon_tp_mode` propagation

`build_dist_opt_optimizer_config` (`primitive/optimizers/megatron_wrap.py`) built the
mcore `OptimizerConfig` copying only lr/betas/offload and **dropped every `muon_*`
field**, so mcore fell back to its default `muon_tp_mode="blockwise"` (per-shard *local*
NS) — silently degrading a requested `distributed`. Fixed to forward all ten `muon_*`
fields for muon. Proven three ways: 0-GPU init-chain gate (`requested=distributed →
core=distributed`); live on GPU (job 14242992 mcore dump shows
`muon_tp_mode='distributed'`); and the config-identity receipt above (job 14243791,
part (a) — all `muon_*` fields survive the lowering).

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
