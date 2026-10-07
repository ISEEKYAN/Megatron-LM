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

For segmented CPU-master/momentum updates, opt in explicitly on either layout:

```bash
DS41_SEGMENTED_HOST=1 DS41_STEPS=5 DS41_SAVE_FREQ=5 bash run.sh
DS41_SEGMENTED_HOST=1 DS41_OUTPUT=/shared/resumed DS41_RESUME=/shared/fresh/checkpoints/global_step_5 DS41_STEPS=7 DS41_SAVE_FREQ=-1 bash run.sh
python verify.py /shared/fresh --steps 5 --segmented-host
python verify.py /shared/resumed --steps 2 --start-step 6 --segmented-host
python verify_checkpoint.py /shared/fresh /shared/resumed
```

For PP2, add `DS41_PP=2 DS41_EP=4` to both commands and `--parameter-counts 309,313` to both strict verifiers. Keep the same optimizer mode across resume. Numerical FP32 masters and all published optimizer moments remain on CPU; native FP32 execution caches and gradients remain on GPU during training. CPU model offload aliases the authoritative master storage and drops completed-window gradients. Dense initialization uses broadcasts and dense gradients are averaged once after the full schedule, avoiding an additional full DDP gradient-buffer bank.

The optimizer first recomputes and checks every owner candidate, then takes one global finite vote. No numerical rejection publishes any owner. It recomputes the same candidates and publishes one owner at a time in a second pass, retaining the existing Muon/Sinkhorn/torch AdamW operations, global clipping and immutable original gradients. This doubles candidate-update computation and adds host transfers. At most one owner's candidate workspace is live; it does not require all model candidates/moments on GPU. Unexpected device failure during publication is fatal and poisons the optimizer; restart from the last complete checkpoint. This is the same process-failure limit as the original transaction's publication copies, not a promise of recovery from a mid-copy hardware failure. Optimizer checkpoints reuse the existing state format; the model checkpoint contains numerical masters, so they are not duplicated in the optimizer file. The strict verifier checks actual backend types, CPU moments, host/master equality, all-rank gradient norms and cross-step continuity.

This mode also saves and loads DCP directly through complete CPU parameter owners, without CUDA reload or a second CPU master bank. PP keys remain stage-specific; frozen buffers and per-rank optimizer state retain their existing save/load paths. The complete-owner path requires frozen Engram tables and TP/CP1; row-sharded trainable owners are rejected. Resume keeps the tested parallel layout and stage split.

W&B stays online and writes its local run files beneath `DS41_OUTPUT`. Its finish wait defaults to 60 seconds (`WANDB_FINISH_TIMEOUT`) so an upload outage does not hold training GPUs indefinitely. A timeout prints a warning and can leave the remote run marked crashed or incomplete; upload the retained run directory with `wandb sync` from a CPU environment and verify the remote history before treating telemetry as complete. Model, optimizer and strict-zero receipts remain independently available.

Per-owner dense reductions can change FP32 addition order relative to legacy DDP buckets. Bitwise optimizer comparisons use identical gradient tensors; they do not promise identical legacy training trajectories.
