# DS4.1 W4A8 frozen-Engram GRPO proxy

Requires the matching MLite/vLLM experiment revisions, the canonical CUDA/VERL environment, an eight-GPU Ray cluster (two four-GPU nodes), W&B credentials, the DAPO parquet and a prepared release-revision 2-layer prefix. Preserve hidden5120/hc4/384 experts/top6 and the release tokenizer. Visual and archival MTP execution are excluded by the text recipe.

The tested runtime uses NVIDIA PyTorch 26.07, VERL `0c849d86175340b8d1141acda101d91c551493cb`, MegatronCore base `327a238239448de6771beea185dc6902a8de3eaa`, MLite frozen-table implementation `b062ebb9cefe9358f3680c9b450fa3902b323dd0`, and vLLM fork `bbddb5f5d0cf127182ff5a321b0670b730d73060`. Use this experiment branch for the recipe and implementation. The parent example's generic `REQUIRED_VERL.txt` is not the tested pin for this recipe. The runtime must include the pinned fork's native DS4.1 W4A8 batch-invariant kernels and MLite optimizer dependencies; this script does not install the CUDA environment. See the [VERL integration prerequisites](../../README.md) for source-tree setup.

Set `DS41_RELEASE` (official weights plus verified LFS receipts) or `DS41_MODEL` (prepared prefix), `DS41_DATA`, `DS41_OUTPUT`, and `PYTHONPATH` for MLite plus `examples/verl`, then run:

```bash
DS41_STEPS=10 DS41_SAVE_FREQ=10 bash run.sh
DS41_RESUME=/shared/run1/checkpoints/global_step_10 DS41_OUTPUT=/shared/run2 DS41_STEPS=30 bash run.sh
```

Use `DS41_RAY_ADDRESS` if the Ray cluster is not discoverable as `auto`; the default topology is EP8, TP/CP/PP1. `--config-only` resolves Hydra without launching training. Strict audit rejects any unequal response raw log probability, nonzero K3KL, zero advantage, failed/unmodified optimizer update, missing frozen-table census, or inconsistent checkpoint receipt. Recipe uses the explicit numeric-character proxy reward; DAPO correctness is logged separately and does not train the model. The second command starts a new process and performs twenty steps after restoring the first checkpoint; choose a new output directory.

After completion, independently verify the saved audit tensors and checkpoint owner/state receipts:

```bash
python verify.py /shared/run1 --steps 10
python verify.py /shared/run2 --steps 20 --start-step 11
python verify_checkpoint.py /shared/run1 /shared/run2
```

The frozen tables use native level-2 sleep buffer save/restore; this requires host headroom for one local frozen table copy per rollout rank. The aliases share GPU storage and do not enter model checkpoints or optimizers. Actual CUDA allocator/memory samples and host RSS must be reviewed for stability alongside strict metrics.

GPU acceptance and memory stability results are recorded separately; this draft recipe has not yet completed the W7 GPU/resume validation.
