# Qwen3.5 dense-backward responsibility

## Result

The 3.45x gap is now bounded to the FSDP2 parameter-wgrad/sharding boundary,
not the model backward. Independent packed GDN, long-sequence, recompute,
full-depth, full-expert EP8, and PPO-weighted proxies all exclude the earlier
kernel hypotheses. In the decisive PPO-weighted proxy, layer-output backward
norm ratios remain between 1.0007 and 1.0197, while enabling the real FSDP2
optimizer changes the reported gradient-norm comparison to 12.6699 versus
2.7470 (4.612x). The same-cache RL replay remains 0.139983 versus MCore
0.041327 (3.387x), despite aligned loss.

Several genuine forward call-surface gaps were fixed (compiled GDN helpers,
FP32 router, FP32 MRoPE, and MoE permute fusion), but none changed the replay
gradient gate. A dead `grad_sync_enabled` flag was also wired to the existing
FSDP2 no-sync helper and tested; same-cache replay components were unchanged,
so that candidate was reverted. No production-safe FSDP2 wgrad fix has yet
been demonstrated, and the five-step gate remains closed.

## Final responsibility boundary

- Same-cache job `13500928` loaded the original serialized batch and produced
  mLite grad norm `0.139983` versus MCore `0.041327`; pg_loss differed by less
  than `4e-8`.
- PPO-weighted 40-layer EP8 job `13502096` showed model activation-gradient
  ratios of only `1.0007` at layer 39 to `1.0197` at layer 0.
- The 12-layer real-optimizer proxy `13503009` reproduced the failure only
  after enabling FSDP2: `12.669855 / 2.746979 = 4.612286`.
- Disabling recompute in `13503610` left the mLite norm unchanged at
  `12.670125`, excluding checkpoint recomputation.
- The FSDP2 no-sync candidate passed 34 focused tests (`13503919`) but failed
  its same-cache gate (`13504003`); its three global squared norms remained
  `0.020939`, `0.017856`, and `0.020345`. It was reverted.

The next repair must instrument/compare per-parameter wgrad immediately before
and after FSDP2 reduce-scatter on a 4-DP x EP8 topology, then fix the offending
fully-shard mesh/reduction contract. A one-node EP8 fingerprint is not a valid
substitute because it overcounts replicated families differently from MCore.

## Same-batch parameter families

Both runs loaded the exact serialized cache from the preceding runtime-fix
experiment. mLite job `13490269` and MCore job `13491111` completed their
training head with `DAPO_*_RC=0`; their batch, extern, and head Slurm steps all
completed with `0:0`. The top-level jobs were cancelled only by the proven
wrapper after the head returned, to release Ray workers.

All family squared sums close to their corresponding reported total. MCore's
relative closure errors are `8.62e-8`, `1.18e-7`, and `1.33e-8` for the three
optimizer mini-steps.

| family | mini-step 1 norm ratio | mini-step 2 | mini-step 3 |
| --- | ---: | ---: | ---: |
| head | 0.9981 | 1.0017 | 0.9966 |
| MoE expert | 1.3563 | 1.8844 | 1.2786 |
| MoE dense | 2.0756 | 3.4105 | 1.6804 |
| embedding | 2.5010 | 4.9214 | 2.7196 |
| attention | 3.4566 | 6.2008 | 2.8208 |
| GDN | 4.0542 | 7.8898 | 3.5336 |

The head gradient is already aligned. The discrepancy grows while propagating
backward through deeper dense layers and is largest in GDN; this shape rules
out a global loss multiplier and a generic optimizer norm reduction factor.

## Independent backward isolation

Slurm job `13491486` ran one Qwen3.5 linear-attention layer with two experts on
one H100. Both backends loaded the same HF checkpoint and random-token stream.
The packed case used seven sequences of lengths `17,31,5,64,23,47,69` and the
real non-deterministic FLA kernels. Every Slurm step completed with `0:0`.

| case | mLite loss | MCore loss | GDN grad norm ratio | MoE | head | embedding |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| dense single sequence | 13.9225769 | 13.9225683 | 1.000405 | 0.999975 | 1.000004 | 1.000021 |
| packed seven sequences | 14.2813711 | 14.2808800 | 1.000102 | 0.999944 | 0.999990 | 1.000026 |

The lower-level FLA probe in job `13489830` also completed all Slurm steps with
`0:0`. Packed versus per-sequence delta-rule output and Q/K/V/beta gradients
were exact; `g` gradient relative L2 was `7.68e-7`. Causal-convolution output
and input gradient were exact. Padding the packed tail to 4096 was exact. Only
the shared convolution weight's accumulation order differed, with relative L2
`0.003716`, far below and structurally unable to explain the runtime-wide
3.45x norm gap.

Together these results exclude GDN chunk backward, cross-sequence THD
boundaries, 4096-token padding, and generic DTensor reduction.

## Stage localization and root cause

The matched packed stage probe (`13492062`, all steps `0:0`) found identical
BF16 input-projection output on both backends
(`a04a7b7124088e61...`). The first different observed activation was the Q
output of the prepare step: mLite `82f49f6f283b1515...` versus MCore
`a74134413dc4d438...`.

Final parameter-fingerprint job `13492279` completed the main job and all
three Slurm steps with `0:0`. Both backends loaded the same convolution tensor:
shape `[8192,1,4]`, BF16 SHA256
`1a9f2d454872c9d3d00a83ef59c695f146363e72d57371df4e009dc859b67711`,
and identical statistics. This excludes checkpoint conversion and TP mapping.

Call-level probe `13493047` corrected the initial causal-convolution
attribution. Both backends passed identical x, weight, layout, and cu-seqlens
to FLA, and FLA returned the same output SHA
`d22d6e53ccdf72b8...`. The difference was therefore inside prepare, after
convolution. The rejected direct-qkv-view candidate (`13492532`) had not
changed that execution mode.

Probe `13493616` dumped the exact tensors and call stacks. For identical Q
contents, mLite entered L2Norm with `torch.compiler.is_compiling() == False`,
while MCore entered from its `@jit_fuser` prepare helper with the value true.
An exhaustive replay of the real Q tensor in `13493643` tried all 25 FLA
`BT x num_warps` eager configs. Every config reproduced the mLite SHA
`66ac44232eacf5cb...`; none reproduced the MCore SHA
`5f683d4e06e4bda4...`. This excludes autotune configuration as the numerical
cause and assigns the difference to compiled versus eager execution.

The implementation aligns the two compiled contracts: joint Q/K prepare and
`g`/beta calculation use `jit_fuser`. Red/green jobs `13493708`/`13493742` and
`13493905`/`13493929` cover the two missing contracts. Packed stage job
`13493948` completed all steps with `0:0`; in-projection, prepare, `g`/beta,
gated-delta, gated-norm, and output-projection forward SHA values are all
bitwise identical across backends. The 17-test focused regression
`13493949` also completed with `0:0`.

The first full replay (`13493983`) exposed a multi-layer-only failure:
`_prepare_qkv` hit Dynamo's recompile limit and some layers fell back to eager,
so its three mini-step norms remained far from MCore. Raising the global limit
was rejected as a workaround. Instead, L2Norm itself is a static compiled
helper, preserving the compiled numerical contract when the outer per-instance
prepare wrapper falls back. Red/green jobs `13494152`/`13494159` cover this
fallback contract. Final same-cache replay remains the gate before five-step
RL.

## Rejected FP32-router hypothesis

The real MCore replay resolves `moe_router_dtype='fp32'`. MCore selects the
router GEMM dtype in `megatron/core/transformer/moe/router.py` before calling
`router_gating_linear`. The mLite Qwen3.5 model previously passed
`router_dtype=torch.float32 if deterministic else None` from
`experimental/lite/megatron/lite/model/qwen3_5/lite/model.py`; `None` makes the
primitive use the BF16 input dtype.

A clean candidate was tested on top of the PR #80 G2 port. Its regression test
first failed with `None is torch.float32`; the focused selection passed after
the one-line change. Fused same-cache Slurm job `13491663` then loaded the
exact original cache and completed its training head with `DAPO_MLITE_RC=0`:

| metric | MCore `13487566` | FP32-router mLite `13491663` | ratio/difference |
| --- | ---: | ---: | ---: |
| actor loss | 0.0172498560 | 0.0172483015 | -0.0090% |
| signed PPO KL | 0.0006347225 | 0.0006406197 | +0.93% |
| grad norm | 0.0421564206 | 0.1417501867 | 3.363x |

The candidate was therefore not pushed and the gated five-step regression was
not started. This rejected route is recorded as `qwen35-router-fp32-grad`.
