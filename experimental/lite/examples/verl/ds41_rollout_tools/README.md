# DS4.1 rollout/training logprob tools

Standalone tools to check that a DS4.1 rollout engine reproduces the training
forward bit for bit: generate with vLLM alone (no Ray, no verl), score the same
tokens with the MLite training forward, and compare logprobs as FP32 bits.

| File | Role |
|---|---|
| `make_prompts.py` | Builds the fixed prompt set (token ids + sha256) from the DAPO parquet |
| `rollout_logprobs.py` | vLLM-only rollout with the DS4.1 proxy recipe's engine config; per-token logprobs |
| `ref_logprobs.py` | MLite training-side reference: teacher-forces rollout records, recomputes each token's logprob as the actor does |
| `compare.py` | Fail-closed bitwise comparison of two record files (exit 0 only on zero difference) |
| `common.py` | Record format and I/O (JSONL or `.pt`) |
| `test_compare.py` | CPU self-test of `compare.py` (positive case and malformed/different inputs) |
| `prompts/ds41-v14-prompts.jsonl` | Fixed prompt set: a 12800-token prompt (crosses the 12288 indexer threshold) and three DAPO prompts (87, 182, 1047 tokens) |
| `reference/rollout-4l.jsonl` | Reference rollout on the 4-layer prefix: 4 prompts x 4 samples x 128 tokens |
| `versions.json` | Runtime the reference was produced with |

sha256:
- `prompts/ds41-v14-prompts.jsonl`: `d1b9d94700485a09ee301fb00db90568c9c5bc82955eeb4deb34342949d0c325`
- `reference/rollout-4l.jsonl`: `37120de184bfe454b4b7ecf6046ebaf9408067840735369a2bd7738057fe3086`

## Check a rollout change (one command)

Run your modified rollout with the same model, prompts and settings, then:

```bash
python compare.py my-rollout.jsonl reference/rollout-4l.jsonl
```

Exit code 0 (`ZERO_DIFF`) means every token id and every logprob is
bit-identical. Exit code 1 means a token or logprob differs (the report names
the first one). Exit code 2 means an input is malformed: a missing field, a
duplicate `(id, sample)`, a length mismatch, or a non-finite logprob.

## Reproduce the reference

The 4-layer official-precision prefix is the release checkpoint's first four
layers prepared by `examples/verl/scripts/ds41_w4a8_proxy/prepare_prefix.py`
with `DS41_LAYERS=4 DS41_PRECISION=official`. Inside the runtime in
`versions.json`, with `VLLM_BATCH_INVARIANT=1`:

```bash
# Rollout: one node, DP = EP = 4, CUDA graph FULL_AND_PIECEWISE (capture <= 16).
python rollout_logprobs.py --model <4L prefix dir> \  --prompts prompts/ds41-v14-prompts.jsonl --out rollout.jsonl --gpus-per-node 4
# Training-side reference on the same tokens (EP4).
torchrun --standalone --nproc_per_node 4 ref_logprobs.py \  --model <4L prefix dir> --records rollout.jsonl --out reference.jsonl
# Rollout vs. training, and rollout vs. the shipped reference.
python compare.py rollout.jsonl reference.jsonl
python compare.py rollout.jsonl reference/rollout-4l.jsonl
```

For the shipped reference, the rollout and the MLite training-side reference
were bit-identical on all 16 records (2048 tokens). The training-side file is
not shipped. Regenerate it with `ref_logprobs.py` as shown above; it is
deterministic given the runtime.

Multi-node rollout (for example the 40-layer model on two 4-GPU nodes) uses
vLLM's native offline data parallel. Run the same command on every node:

```bash
python rollout_logprobs.py --model <40L dir> --prompts prompts/ds41-v14-prompts.jsonl \  --out rollout-40l.jsonl --nnodes 2 --gpus-per-node 4 --node-rank <0|1> \  --master-addr <node 0 address> --master-port 29551
```

Use `--dry-run` to print each rank's environment, request count and engine
arguments without a GPU. `--precision official` (the default) refuses a
model whose config lacks `expert_activation_group_size=32` and
`wo_a_dtype=bf16`.

## Record format

One JSON object per line: `id`, `sample`, `prompt_ids`, `response_ids`,
`logprobs` (FP32 values widened to Python floats, so JSONL round-trips bit for
bit) and `meta`. `logprobs[i]` is the log-probability of `response_ids[i]`.
