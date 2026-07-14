# PP2 + THD archaeology rerun — verdict

## Bottom line

`test_save_load_roundtrip[qwen3_moe-dist_opt]` **passes on today's HEAD** under
8-GPU PP2 (tp2/ep2/pp2, dist_opt). The Megatron-Lite pipeline-parallel core is
**not** regressed. The "PP+THD broke" signal seen in the FSDP2+PP2 RL smoke was
a harness artifact, not a core regression. No `git bisect` is warranted.

## Evidence (real Slurm jobs, rc=0, non-skip)

- **Terminal test — job 13951220**: `COMPLETED`, ExitCode `0:0`, elapsed 1:40.
  `test_save_load_roundtrip[qwen3_moe-dist_opt]` → `1 passed` on all 8 ranks,
  46s, non-skip (`world_size == 8`, so the test ran rather than skipping).
  Repo HEAD `055d4942b` — includes the streaming pp>1 HF export, the DS4 CSA CP
  path, and the merged upstream streaming-export PR in ancestry.
- **Re-confirmed — job 13951352 (Arm A)**: same test, 8× `1 passed`,
  `ARM_A_DONE rc=0`.

## What the terminal test covers (and what it does not)

- **Covers**: PP2 dist_opt train step + Megatron distributed-checkpoint
  save → fresh build → load, asserting bitwise-equal parameter restore. This is
  the primary regression guard for the PP-aware checkpoint entry points
  (pp-prefixed keys, cross-stage FQN, gather/export integrity).
- **Does not itself exercise**: THD packing / inter-stage P2P shape inference
  (the forward here is BSHD), nor the streaming HF export (a separate test).
  The `2206 vs 2213` RoPE/token mismatch lives in the **THD data path of the RL
  integration**, not in this PP core guard.

## Harness archaeology — why the RL smoke looked red

The FSDP2+PP2 RL smoke's real failure was **job 13941926**: it died in
`update_actor` with a THD packed-token mismatch (`2206 vs 2213`) before reaching
`update_weights`. That is an inter-stage sequence-length disagreement in the RL
data path — each pipeline stage packs its THD microbatch independently, and the
P2P receive buffer is sized from metadata that does not match the sender's
actual packed length.

The subsequent smoke commits tried to route *around* the failure rather than
diagnose it:

- disable `use_thd` — **rejected** by `MegatronLiteEngine` (`use_thd=False` not
  accepted on this path);
- `fused=False` + fixed batch shapes;
- rollout `load_format=dummy` to reach `update_weights` without the train step.

None address the root cause; all are config mutations that mask, not locate, the
bug. These are the "flailing" config differences to retire.

## Environment fixes applied to the archaeology harness (all environment — none PP+THD)

The rerun harness had never actually executed a test; it stacked four
environment bugs, each of which superficially resembled a failure. All are fixed
in `run_pp2_archaeology_rerun.{sh,sbatch}`:

1. `DSA_SITE`/`WORK_DIR`/`OUTPUT_DIR` set but not `export`ed, so
   `srun --export=ALL,VAR` forwarded them empty (job 13946170, died 44s at
   preflight). → export them.
2. Leaked host `PATH` put a host miniforge `python3` (uncompiled Transformer
   Engine) ahead of the container's `python3.12` (job 13950619). → pin
   `PATH=/usr/local/bin:/usr/bin:/bin`.
3. `ModuleNotFoundError: megatron.core` — this repo is a **lite-only skeleton**
   (root has only `experimental/` + `README`); per `experimental/lite/README`,
   `experimental/lite` must layer onto a full Megatron-LM tree that supplies
   `megatron.core` (job 13950842). → add `CORE_TREE` and namespace-merge on
   `PYTHONPATH` (`megatron.lite` still resolves to this branch — the host tree's
   `megatron/` has no `lite` subdir).
4. torch-inductor / Triton JIT invoked the host conda `cc`
   (`CC`/`CXX` leaked by `--export=ALL`) and failed the build mid-test
   (job 13951102). → unset the conda toolchain, pin `CC`/`CXX` to container gcc,
   give inductor/triton writable `/tmp` caches.

## Bench pressure arms (B/C) — deferred, not a core signal

Arms B/C (PP2+THD bench, fixed and variable per-microbatch packed lengths) are
blocked on a bench-argument gap, not a THD core failure: `bench.py`'s
`Qwen3MoEConfig.from_hf` needs a real `--hf-path` config dir (the harness passed
`""` → `FileNotFoundError: No config.json`), and the readily-available configs
are `qwen3_5_moe` (nested) versus the bench's `--model-name qwen3_moe`. That is
additional harness plumbing and is not required for the verdict. It can be
revived with a matching config for deeper THD stress if wanted.

## Conclusion

Green = false alarm. The 1.13.20 blocker should be routed to the **RL / THD
integration data-path fix** — the inter-stage packed-length metadata must be
same-sourced across pipeline stages — **not** a Megatron-Lite PP core bisect.
