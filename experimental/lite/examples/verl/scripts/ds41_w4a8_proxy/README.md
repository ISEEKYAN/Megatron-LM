# DS4.1 FP4 train/inference consistency on HSG

This recipe runs a faithful two-layer release prefix through two strict GRPO updates, saves a checkpoint, starts a new training process, restores it and runs two more updates. Training uses MLite deployment math; inference uses vLLM with `VLLM_BATCH_INVARIANT=1`. Expert weights are MXFP4 with dynamic FP8 activations (W4A8), EP8 and TP/CP/PP1. Optimizer state is on the host and Engram tables remain frozen.

Both proxy depths explicitly disable prefix caching. The runtime compatibility adapter preserves `enable_prefix_caching=false` as vLLM's `--no-enable-prefix-caching`; a resolved Hydra value alone does not verify the engine setting. Check that the vLLM engine startup configuration reports `enable_prefix_caching=False`. The v1.1 release lost the false value during CLI conversion and enabled caching; this configuration correction belongs to v1.2 and requires new C8 validation.

Disabling prefix caching makes repeated-prompt prefill slower. The two-layer C8 test allows 1500 seconds for the fresh stage and 2400 seconds for the resumed stage, including model construction, checkpoint loading, two updates, saving and final resync. The HSG GPU job requests 75 minutes. These time limits do not change the strict equality criteria.

## Environment and shared release

Use HSG GB200 (SM100), Enroot, Slurm, and an online W&B account. The release bundle supplies one runtime image with matched PyTorch, Transformer Engine, vLLM, MLite, VERL and native CUDA kernels; no source build or package install is needed during a GPU allocation. Set `DS41_BUNDLE` to the shared project release directory supplied with the handoff. Check its `SHA256SUMS` and `manifests/` before running. `paths.env` gives the existing weight directory and parquet directory; edit a copy if your input locations differ.

The required release revision is **`dba1be0a40aa45a94ad051997016db3960a90277`**. Reuse the cluster's existing verified weights. `prepare_prefix.py` checks revision/SHA sidecars and links the original shards; it preserves all release matrix and Engram dimensions. A replacement download must provide the same verified sidecars, tokenizer and shard index. Do not substitute a differently quantized checkpoint.

### Release layer subset (6L)

The 2L/4L prefixes reach neither ratio-1 compression nor the two-level candidate Top-K, the second Engram table, or vLLM's decoder SWA bounded replay. `prepare_variant.py` builds a proxy from release layers `DS41_VARIANT_LAYERS=1,14,15,20,21,24`: the first and second Engram tables, a ratio-2 KV/index owner and consumer, the ratio-1 owner and candidate publisher, a ratio-1 consumer and a candidate-reusing index layer. Layers are renumbered 0-5; each keeps its release KV/index owner, candidate mode and Engram slot, and `engram_hash_layer_ids` keeps the release hash seeds. Tensors are copied byte for byte into new shards under the new names (about 0.27 TB), so no release layer can be read under a proxy index.

```bash
DS41_RELEASE=<release dir> DS41_MODEL=<out>/6l/model DS41_OUTPUT=<out> \
DS41_DATA=<DAPO parquet> DS41_VARIANT_LAYERS=1,14,15,20,21,24 \
python prepare_variant.py
python prepare_long_prompts.py --model <out>/6l/model --min-tokens 17408 \
  --repeat-content-to-min --data <DAPO parquet> \
  --output <inputs>/c8-candidate-long-prompts.parquet
```

Run it with `DS41_LAYERS=6` (prompt 17408 + response 128, so the candidate pool of 2048 blocks of 8 actually drops blocks). The 6L C8 additionally requires the vLLM first-local-layer Engram fix, `DS41_PARAMETER_COUNTS` from the 6L CPU init receipt, two frozen tables per resync, training and inference candidate calls that dropped blocks, and a used bounded-replay batch.

## One command

Export `WANDB_API_KEY` without putting it in a script or command-line argument. Choose a new absolute output directory on shared Lustre, writable by your account:

```bash
export DS41_BUNDLE=/path/to/shared/ds41-fp4/v1
export DS41_SLURM_ACCOUNT='<ACCOUNT>'
bash "$DS41_BUNDLE/recipe/hsg/submit.sh" /path/to/your/new-c8-output
```

For the 6L release subset, pass `--layers 6` and the trainable count from its CPU init receipt; the CPU gate then writes the renumbered shards and 17408-token prompts (2 h CPU, 2.5 h GPU limits):

```bash
DS41_PARAMETER_COUNTS=<count> bash "$DS41_BUNDLE/recipe/hsg/submit.sh" --layers 6 /path/to/your/new-c8-output
```

The same launcher accepts `--layers 2`, `--layers 4`, `--layers 6`, and
`--layers 9`. The 6L and 9L variants require `DS41_PARAMETER_COUNTS` from
that variant's CPU construction receipt. Set `DS41_PRECISION=official`,
`DS41_CUDAGRAPH=full_and_piecewise`, and `DS41_CUDAGRAPH_MAX_CAPTURE=16`
before submission; both preparation and GPU workers receive these settings.

`DS41_CPU_ONLY=1` submits only the zero-GPU stage, including for 40L; it
does not require or grant GPU authorization. The default submits the CPU gate
and its dependent GPU job.

For a source/native overlay runtime, set `DS41_OVERLAY_SETUP` to its published
shell setup file. Both CPU and GPU stages source that same file before Enroot;
it must pin the runtime mounts, not replace the recipe workflow.
`DS41_ENROOT_DATA_PATH` can select an already expanded, verified runtime.

Resource overrides use `DS41_CPU_TIME`, `DS41_GPU_TIME` (Slurm time format),
`DS41_CPU_CPUS`, `DS41_GPU_CPUS`, `DS41_CPU_MEM`, `DS41_GPU_MEM`,
`DS41_CPU_PARTITION`, `DS41_GPU_PARTITION`, and `DS41_CPU_QOS`.
`DS41_GPU_QOS` defaults to `interactive`; set it to an empty string to use
the partition default. These options do not change the variant's node count
or authorize additional resources. For example:

```bash
DS41_CPU_QOS=cpu-short DS41_CPU_MEM=0 DS41_GPU_CPUS=64 DS41_GPU_TIME=01:30:00 \
  bash "$DS41_BUNDLE/recipe/hsg/submit.sh" --layers 4 /path/to/new-4l-output

```

The launcher first submits a **zero-GPU, two-node CPU gate**. It checks image/files, expands the image before any GPU allocation, prepares the 2L prefix, resolves the production config, exercises real optimizer workers and collects the exact C8 test. Its dependent GPU job runs only after success and is automatically cancelled if the dependency fails.

The account comes from `DS41_SLURM_ACCOUNT`; it is not embedded in the recipe. The GPU submission uses `sbatch --account=<ACCOUNT> -p batch --qos interactive`, **2 nodes × 4 GPUs, `--mem=0`, 60 minutes**. It does not pin nodes or increase scale to bypass a queue. Ray ports derive from the job ID, are checked before startup, and the dashboard is disabled. During GPU execution, JIT caches and container Python/Torch temporary files use a fixed, versioned node-local `/var/tmp` path without a job ID. Enroot stores the CPU-expanded image at `<OUTPUT>/enroot`, image-preparation staging at node-local `/var/tmp/<USER>-ds41-enroot-cpu`, and GPU container startup staging at node-local `/var/tmp/<USER>-ds41-enroot/<JOBID>/<HOST>`; these are distinct from runtime JIT caches. If your cluster administrator requires excluding CPU nodes, pass `DS41_CPU_EXCLUDE=<OPTIONAL_NODE_LIST>`; no host is embedded in the launcher. GPU/model workers share the eight-card allocation; the proxy reward is CPU code, with no GPU sandbox service.

## Results and strict acceptance

The launcher prints both job IDs and the output directory. Inspect `logs/cpu/result.json` for the CPU gate, `logs/gpu/head.log` for the W&B URL and first diagnostic, and `logs/gpu/pytest.log` for failures. The final `logs/gpu/result.json` must contain `passed: true`, `collect_only: false`, and `pytest_exit: 0`; the XML must show **one passed test and zero skips**.

Under `logs/gpu/c8/2l-bi1/`:

- `fresh/verified.json`: steps 1–2, raw rollout/training token logprobs exactly equal, checked GRPO/optimizer/resync receipts.
- `resumed/verified.json`: steps 3–4 in a fresh process, the same strict numerical checks.
- `checkpoint-verified.json`: all eight ranks restore the exact saved parameters and optimizer state before their next update.
- `fresh/audit/` and `resumed/audit/`: raw/advantage tensors, parameter digests, frozen-table census, resync acknowledgements and per-cycle host RSS peaks.

The production verifier requires zero raw mismatches and zero maximum absolute error; there is no tolerance fallback. A timeout or failed teardown is not a clean pass. Preserve raw tensors and events so the supplied `verify.py` and `verify_checkpoint.py` can independently distinguish a complete numerical/state sequence from an incomplete run. Resume checks establish checkpoint continuity; they do not claim equality to a separate uninterrupted random trajectory.

## Verified scope and limits

The prior 2L sequence passed via independent CPU verification of frozen production tensors after Slurm timed out during its overall job: 16,384 response tokens, zero differences, 32 rank updates, 48 resync acknowledgements, 16 saves and 8 exact loads. Consult the bundle's evidence manifest for its exact source/binary versions; a new base or wrapper does not inherit that end-to-end result automatically. The v1.1 segmented Engram census contract requires its own new 2L C8 receipt; CPU collection alone does not execute initial resync. The native quantization int32-size fix additionally has a single-GPU old-red/new-green regression across the large EP8 profile shape.

**4L/CSA/indexer C8 remains under validation.** The 2L prefix has no indexer. Indexer acceptance requires actual training and inference production selector calls with `query_rows > 12288`; long KV length alone is insufficient. Current indexer math follows native DeepGEMM BF16 weights/scores, differing from the older FP32 recipe; Top-K/logprob impact must be measured in 4L.

The published HSG launcher selects only the 2L C8 test; the 4L parametrization is not reachable through this launcher. C8 audit receipts and `verify.py` count frozen tables and reuse generations and fingerprint whole tables; they do not independently certify each `{start, end, sha256}` census segment. Segment-level rejection is covered by exporter/receiver CPU tests. Reuse generations check resident whole-table byte digests rather than rerunning the per-segment census, so they detect corruption without locating its segment. These are separate evidence limits, not relaxed strict logprob acceptance.

BI backend declarations are scoped to the tested **SM100 DS41** classes; shared guards remain closed for other hardware/models. FP4 experts use the verified DeepGEMM route. FlashInfer MoE and paired BI0/BI1 performance, the TP4 throughput floor, full 40-layer training and model-quality evaluation are separate pending gates. The numeric-character reward is only a proxy.
