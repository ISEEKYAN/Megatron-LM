# DS4.1 W4A8 frozen-Engram GRPO proxy

Requires the matching MLite/vLLM experiment revisions, the canonical CUDA/VERL environment, an eight-GPU Ray cluster (two four-GPU nodes), W&B credentials, the DAPO parquet and a prepared release-revision 2-layer prefix. Preserve hidden5120/hc4/384 experts/top6 and the release tokenizer. Visual and archival MTP execution are excluded by the text recipe.

The tested runtime uses NVIDIA PyTorch 26.07, VERL `0c849d86175340b8d1141acda101d91c551493cb`, MegatronCore base `327a238239448de6771beea185dc6902a8de3eaa`, MLite frozen-table/sleep implementation `b062ebb9cefe9358f3680c9b450fa3902b323dd0` plus SWA KV deployment RoPE correction `dcfe8722b7b97e8d42b043315f2f1e77a3727f09`, and vLLM fork `bbddb5f5d0cf127182ff5a321b0670b730d73060`. Use this experiment branch for the recipe and implementation. The parent example's generic `REQUIRED_VERL.txt` is not the tested pin for this recipe. The runtime must include the pinned fork's native DS4.1 W4A8 batch-invariant kernels and MLite optimizer dependencies; this script does not install the CUDA environment. See the [VERL integration prerequisites](../../README.md) for source-tree setup.

Set `DS41_RELEASE` (official weights plus verified LFS receipts) or `DS41_MODEL` (prepared prefix), `DS41_DATA`, `DS41_OUTPUT`, and `PYTHONPATH` for MLite plus `examples/verl`, then run:

```bash
DS41_STEPS=2 DS41_SAVE_FREQ=2 bash run.sh
DS41_RESUME=/shared/run1/checkpoints/global_step_2 DS41_OUTPUT=/shared/run2 DS41_STEPS=22 DS41_SAVE_FREQ=20 bash run.sh
```

Use `DS41_RAY_ADDRESS` if the Ray cluster is not discoverable as `auto`; the default topology is EP8, TP/CP/PP1. `--config-only` resolves Hydra without launching training. Strict audit rejects any unequal response raw log probability, nonzero K3KL, zero advantage, failed/unmodified optimizer update, missing frozen-table census, or inconsistent checkpoint receipt. Recipe uses the explicit numeric-character proxy reward; DAPO correctness is logged separately and does not train the model. The second command starts a new process and performs twenty steps after restoring the first checkpoint; choose a new output directory.

After completion, independently verify the saved audit tensors and checkpoint owner/state receipts:

```bash
python verify.py /shared/run1 --steps 2
python verify.py /shared/run2 --steps 20 --start-step 3
python verify_checkpoint.py /shared/run1 /shared/run2
```

The frozen tables use native level-2 sleep buffer save/restore; this requires host headroom for one local frozen table copy per rollout rank. The aliases share GPU storage and do not enter model checkpoints or optimizers. Actual CUDA allocator/memory samples and host RSS must be reviewed for stability alongside strict metrics.

The corrected recipe has completed two strict-zero steps, an eight-rank checkpoint save, and a new-process resume for twenty consecutive strict-zero GRPO steps. Every resumed step has nonzero advantage and actual parameter updates; saved/restored master, optimizer, scheduler and frozen-table digests match on all eight ranks. Native CUDA regression passes through `tests/run_tests.sh` with six passed and zero skipped.

Post-update GPU allocation stays near 78.602 GiB per training rank. The first DCP save raises training-worker RSS by approximately 18–22 GiB; allow for this host-memory step. An additional four-step resume with a checkpoint saved every step shows subsequent RSS near 50–54 GiB, without repeated accumulation of that first-save increase. These observations cover the tested two-layer, eight-GPU run windows; review host and physical-device samples for your run as well.

For a two-stage deployment proxy, retain the eight-GPU rollout EP8 layout and set the actor/reference layout explicitly:

```bash
DS41_PP=2 DS41_EP=4 DS41_STEPS=2 DS41_SAVE_FREQ=-1 bash run.sh
python verify.py /shared/pp2-run --steps 2 --parameter-counts 309,313
```

Use a fresh `DS41_OUTPUT` for this command. PP2 requires a closed CSA boundary; the two-layer release prefix splits at layer 1, and the full forty-layer configuration splits at layer 20. TP, CP and VPP remain 1. Stage transport preserves materialized hidden states and pending mHC post operands in FP32; the receiving stage retains their reference VJP edges and applies the original Engram reset. Stage-local encoded export is assembled over PP peers before the rollout generation ends, including first-generation frozen-table census and later table reuse checks. These options cover PP2 forward/backward and online resync. The checkpoint/resume and twenty-step stability results above cover PP1.

PP2 training defers DDP gradient reduction during pipeline forwards and averages dense owners after the complete microbatch schedule. Expert and row-sharded gradients retain their separate reductions. The verifier requires all training ranks to agree on the global clipping norm and checks parameter-digest continuity between updates; strict-zero logprobs alone do not establish correct data-parallel training.
