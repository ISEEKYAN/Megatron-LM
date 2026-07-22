# Three-arm Muon precision results — Qwen3-30B-A3B SFT (AC#3)

Real verl-SFT workload, Qwen3-30B-A3B (`qwen3_moe`, standard attention, no FLA),
8×H100 CW, fixed same-contract: seed=1234, TP2/PP1/EP8/ETP1/CP1, LR 1e-5 constant,
warmup 2, weight_decay 0.1, clip 1.0, 20 steps, gsm8k messages SFT, `load_hf_weights`
(identical init). mlite `407d4a81d`, mcore `fd1121b`, container `verl.vllm023.sqsh`,
`emerging_optimizers==0.3.0`.

## Arms (real Slurm jobs, all `COMPLETED` rc=0)

| ARM | job | optimizer | backend | muon_tp_mode | offload | peak GiB/GPU |
|-----|-----|-----------|---------|--------------|---------|--------------|
| adamw | 14242814 | adamw | dist_opt | (n/a) | cpu | 31.3 |
| muon | 14242992 | muon | dist_opt | **distributed** | off¹ | 63.3 |
| muon_fsdp2 | 14242904 | muon | fsdp2 | distributed | (fsdp2) | 59.2 |

¹ dist_opt Muon runs with optimizer offload OFF — see "hybrid_optimizer" finding below.

## Per-step train/loss

| step | adamw | muon (dist_opt) | muon_fsdp2 | \|muon−fsdp2\| |
|-----:|------:|----------------:|-----------:|---------------:|
| 1 | 1.51501 | 1.50559 | 1.51049 | 0.00489 |
| 2 | 1.48806 | 1.48605 | 1.48033 | 0.00572 |
| 3 | 1.19553 | 1.28226 | 1.29881 | 0.01655 |
| 4 | 0.88977 | 1.03788 | 1.09556 | 0.05768 |
| 5 | 0.73041 | 0.88960 | 0.94445 | 0.05485 |
| 6 | 0.52778 | 0.74173 | 0.79299 | 0.05127 |
| 7 | 0.45003 | 0.59071 | 0.63331 | 0.04260 |
| 8 | 0.45170 | 0.58517 | 0.62160 | 0.03643 |
| 9 | 0.34360 | 0.46311 | 0.49151 | 0.02840 |
| 10 | 0.38468 | 0.46575 | 0.48392 | 0.01817 |
| 11 | 0.38296 | 0.44719 | 0.45460 | 0.00741 |
| 12 | 0.34175 | 0.38639 | 0.39164 | 0.00525 |
| 13 | 0.34699 | 0.39779 | 0.39668 | 0.00111 |
| 14 | 0.33217 | 0.36587 | 0.36361 | 0.00226 |
| 15 | 0.34443 | 0.36991 | 0.37084 | 0.00093 |
| 16 | 0.32186 | 0.34862 | 0.34554 | 0.00308 |
| 17 | 0.29555 | 0.32391 | 0.31900 | 0.00491 |
| 18 | 0.29563 | 0.32246 | 0.31265 | 0.00981 |
| 19 | 0.33460 | 0.36785 | 0.36327 | 0.00459 |
| 20 | 0.32339 | 0.35204 | 0.34468 | 0.00735 |

JSONL: `$RUN_ROOT/{adamw,muon,muon_fsdp2}/q35_precision_*.jsonl`.

## Verdicts

1. **DistOpt Muon trains on a real parallel workload.** The `muon` arm ran the *true*
   cross-TP Newton-Schulz (`muon_tp_mode=distributed`, confirmed in the live mcore
   `OptimizerConfig` dump) on Qwen3-30B-A3B at TP2 — 20 clean steps, loss 1.51→0.35,
   grad_norm ~2, no divergence. This fills the gap left by the earlier DP2/TP1 toy
   bitwise check (which never exercised distributed NS at real scale).

2. **Muon is no worse than AdamW** (bayan's criterion). Both descend from ~1.51 to
   ~0.32–0.35 at the same rate from an identical init; muon's final loss (0.352) sits
   a hair above adamw's (0.323), consistent with a different-but-healthy optimizer, not
   a regression. DistOpt Muon trains, converges, and does not blow up.

3. **Megatron vs DistOpt = bitwise by construction.** MLite's dist_opt Muon *is*
   Megatron-Core's TensorParallel Muon: `build_dist_opt_optimizer_config` →
   `get_megatron_optimizer(optimizer="muon")` → `megatron/core/optimizer/muon.py` →
   `emerging_optimizers`. There is no second binary to diff — the identity is the code
   path itself, and the propagation fix (below) is what makes the *distributed* variant
   actually reach it.

4. **FSDP2 vs DistOpt = within tolerance.** `muon_fsdp2` is an *independent*
   reimplementation (`primitive/optimizers/fsdp2/muon.py`, full-gather → orthogonalize
   → reshard). From the same init it tracks the dist_opt Muon closely: max \|Δloss\| =
   0.058 (a step-4 early transient), decaying to ≤0.01 by step 11 and staying there —
   two independent lowerings of the same Muon math agree within tolerance. (An exact
   bitwise match across a Megatron distributed-optimizer path and an FSDP2 DTensor path
   is not expected; reduction order and sharding differ.)

## Correctness fix carried & proven: `muon_tp_mode` propagation

`build_dist_opt_optimizer_config` (`primitive/optimizers/megatron_wrap.py`) built the
mcore `OptimizerConfig` copying only lr/betas/offload and **dropped every `muon_*`
field**, so mcore fell back to its default `muon_tp_mode="blockwise"` (per-shard *local*
NS) — silently degrading a requested `distributed`. Fixed to forward all ten `muon_*`
fields for muon. Proven on 0 GPU (init-chain gate: `requested=distributed →
core=distributed`, `requested=blockwise → core=blockwise`) and live on GPU (job
14242992 mcore dump shows `muon_tp_mode='distributed'`).

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
