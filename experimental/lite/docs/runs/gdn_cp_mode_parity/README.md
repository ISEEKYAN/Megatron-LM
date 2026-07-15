# Qwen3.5 GatedDeltaNet `gdn_cp_mode` parity proxy

Answers TASK-1.1.7.1: provide a context-parallel GatedDeltaNet linear-attention
that is **numerically exact** vs CP-off (so RL train/inference log-probs match),
while still sharding per-head memory across CP ranks.

## Modes
| `gdn_cp_mode` | strategy | numerics vs CP-off | memory |
| --- | --- | --- | --- |
| `headwise` (default) | head-parallel all-to-all: each rank holds the full seq for `1/cp` heads (+ matching conv1d/A_log/dt_bias slices), runs the ordinary full-seq recurrence, a2a back | **bitwise-exact** (heads independent) | per-head state/activations sharded across ranks |
| `replicated` | all-gather full seq → compute every head on every rank → zigzag-slice | bitwise-exact | full seq for all heads on every rank (worst) |
| `chunkwise` | FLA `cp_context` ring: keep the `1/cp` seq shard *and* all heads, reshuffle zigzag→contiguous chunks, pass per-chunk state around the CP ring, reshuffle back | **bf16 floor** (ring reassociates) | seq shard + all heads (best) |

`chunkwise` is a **faithful packing-aware mirror** of upstream Megatron
`linear_cp_mode='chunkwise'` (**NVIDIA/Megatron-LM@d1384c2d9**
`megatron/core/ssm/gated_delta_net.py` + `megatron/core/context_parallel_layout.py`,
fetched 2026-07-15). The packed/THD reshuffle routes on the **global** `cu_seqlens`
(`get_thd_context_parallel_rank_indices` + single packed-token all-to-all), and
`_resolve_cu_seqlens` validates padded lengths + per-seq `% cp_size` divisibility.

An earlier local chunkwise (`sharded`) copy sliced `cu_seqlens // cp_size` and swapped
each sequence independently — corrupting any packed batch whose contiguous CP boundary
falls inside a sequence, which under RL amplified into a ~220× step-1 `ppo_kl`
(train/inference log-prob mismatch). That miscopy — **not** the chunkwise ring itself —
was the root cause; this run restores chunkwise as the faithful, memory-optimal mode.

`headwise` was ported from the same upstream (`tensor_a2a_cp2hp`/`hp2cp`,
`_build_head_perm_for_split_sections`, `get_parameter_local_cp_headwise`).

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
gloo CP4 round-trip test (`hw_a2a_cpu_check.py`) confirms it is **lossless / bitwise**
(`cp2hp_max_abs=0`, `hp2cp_max_abs=0`) against the ground-truth full-sequence layout.
Since the per-head conv/recurrence arithmetic is unchanged from the verified
`replicated` path and heads are independent, a correct redistribution ⇒
`headwise == CP-off` bitwise.

`chunkwise`'s only new packed-path code is the packing-aware zigzag↔contiguous
reshuffle. `thd_reshuffle_cpu_check.py` (also `tests/unit/primitive/
test_gdn_chunkwise_thd_reshuffle.py`) runs the real primitive under gloo on
multi-sequence packed batches whose contiguous CP boundary falls *inside* a sequence
(the case the old copy corrupted) and asserts the zigzag→contiguous reshuffle equals
the ground-truth contiguous span **bitwise**, and round-trips back **bitwise**. The
recurrence itself is unchanged FLA GPU code, so a correct reshuffle ⇒ the GPU
chunkwise path is fed the same contiguous-time tokens upstream feeds its FLA kernel.
Verified run: `THD_RESHUFFLE_DONE ALL_PASS` (cp2/cp4, `fwd_max=0 rt_max=0`).

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
