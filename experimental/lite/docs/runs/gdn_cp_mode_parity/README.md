# Qwen3.5 GatedDeltaNet `gdn_cp_mode` parity proxy

Answers TASK-1.1.7.1: provide a context-parallel GatedDeltaNet linear-attention
that is **numerically exact** vs CP-off (so RL train/inference log-probs match),
while still sharding per-head memory across CP ranks.

## Modes (post-port)
| `gdn_cp_mode` | strategy | numerics vs CP-off | memory |
| --- | --- | --- | --- |
| `headwise` (default) | head-parallel all-to-all: each rank holds the full seq for `1/cp` heads (+ matching conv1d/A_log/dt_bias slices), runs the ordinary full-seq recurrence, a2a back | **bitwise-exact** (heads independent) | per-head state/activations sharded across ranks |
| `replicated` | all-gather full seq → compute every head on every rank → zigzag-slice | bitwise-exact | full seq for all heads on every rank (worst) |

The former `sharded` mode (FLA `cp_context` chunkwise ring) was **removed**: its
cross-rank chunk-state accumulation order differs from the reference, and under RL
that bf16-reassociation gap amplified into a ~220× step-1 `ppo_kl` (train/inference
log-prob mismatch). `headwise` is the upstream Megatron answer for exact + sharded.

Ported from **NVIDIA/Megatron-LM@d1384c2d9** `megatron/core/ssm/gated_delta_net.py`
(`linear_cp_mode='headwise'`: `tensor_a2a_cp2hp`/`hp2cp`, `_build_head_perm_for_split_sections`,
`get_parameter_local_cp_headwise`). Fetched 2026-07-14.

## What the harness measures
`tests/smoke/primitive/gdn_cp_mode_parity.py` builds a tiny **proxy** GDN
(REAL head dims `dk=dv=128`, `conv_kernel=4`; only head COUNT + seq truncated,
because dim=4 probes are non-representative per K-0125). Input is **SBHD**
`x = [seq, batch, hidden]`, zigzag-sharded along the seq dim. Per CP size in `{2,4}`:

1. CP-off reference: identical-weight `cp_size==1` module on the FULL sequence.
2. `headwise` and `replicated` at that CP size, each rank fed its zigzag shard.
3. Per-tensor diff (`max_abs`/`max_rel`) of forward output, input-activation grad,
   and CP-all-reduced weight grads, vs the reference (zigzag-sliced). Both modes are
   expected bitwise-exact in forward (`max_abs == 0`); grads carry ordinary bf16
   backward reassociation noise.

## CPU plumbing check (no GPU)
`headwise`'s only new distributed code is the cp2hp/hp2cp redistribution. A CPU
gloo CP4 round-trip test confirms it is **lossless / bitwise** (`cp2hp_max_abs=0`,
`hp2cp_max_abs=0`) against the ground-truth full-sequence layout. Since the per-head
conv/recurrence arithmetic is unchanged from the verified `replicated` path and
heads are independent, a correct redistribution ⇒ `headwise == CP-off` bitwise.

## Results (GPU proxy A/B)
_Pending the post-port GPU rerun (pre-GPU review gate first)._ Prior `replicated`
baseline (job `13950929`) was bitwise-exact in forward at CP2/CP4; the rerun adds
the `headwise` column and is expected bitwise-exact there too. Old `sharded` numbers
(~7.96e-3 forward) are retired with the mode.

## Environment (reuse, do not reinvent — see durable `mlite_env_setup`, K-0123)
- image: `verl.vllm023.sqsh` (qwen3.5 line canonical).
- overlay: `qwen35-cp-overlay-20260613/site` on `PYTHONPATH` + `tvm_ffi/lib` on
  `LD_LIBRARY_PATH` — provides FLA/tilelang kernels used by the non-deterministic
  compute path. Without it FLA reports "Please install tilelang" (env gap, not a hang).

## Reproduce
`run_gdn_cp_parity.sbatch` is the as-run recipe: rsync the mlite worktree into
`$R/mlite`, drop the harness at `$R/gdn_cp_mode_parity.py`, run CP2 then CP4 under
`torchrun` in one `srun`. Grep the log for `GDN_CP_PARITY` and `GDN_CP_PARITY_DONE`.
