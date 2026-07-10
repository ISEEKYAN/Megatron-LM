# Hopper blockwise BF16-weight primitive parity

This run compares the production Megatron Lite primitives with the frozen
Megatron-Core Transformer Engine wrappers. Reference repeats are completed and
their thresholds are sealed before the corresponding target modules are
constructed.

## Frozen inputs

- Megatron-Core: `cf2f07d7b1315c96c05554c670c43207c6783e5e`
- Megatron source archive SHA-256:
  `c08272b18d171553f2dcd04937d27e77d3a6be223860726be7c56fcc90c558b1`
- Transformer Engine: `2.18.0.dev0+8b99682`, source commit
  `8b9968255eb879e6e390f427836906b29aad64d2`
- Transformer Engine overlay:
  `/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/bayan/code/te-8b9968255-cu132-py312-overlay-r7`
- driver SHA-256:
  `4d02d222a8455f70c988018e696f979d5618c24e618c882aace7981706c43c1f`
- launcher: `hopper-blockwise-primitive-parity.sbatch`

The cell uses four H100 GPUs with TP=2 and EP=2. Both `bf16` and
`hopper_blockwise_bf16_weight` use seed 1234, three repeats, hidden size 1024,
sequence length 256, micro-batch size 2, 16 query heads, eight KV heads, four
experts, top-k two, and routing-map padding to 16. The one-step update is AdamW
with an FP32 master, learning rate `1e-4`, betas `(0.9, 0.95)`, epsilon `1e-8`,
and weight decay `0.1`.

## Attempts

Failed jobs remain in the immutable Slurm logs. No failed threshold or target
artifact was reused by a later job.

| Job | Result | Finding |
| --- | --- | --- |
| `13713809` | `FAILED 1:0` | An installed checkpoint-only NVRx package was older than the frozen Megatron import requirement; the primitive driver now makes that unused optional integration absent. |
| `13713952` | `FAILED 1:0` | Corrected the frozen-source `AttnMaskType` import. |
| `13714092` | `FAILED 1:0` | Adapted the MCore DPA wrapper's explicit mask arguments. |
| `13714182` | `FAILED 1:0` | Normalized already-flattened MCore and raw TE core-attention outputs to `[S, B, H/tp]`. |
| `13714272` | `FAILED 1:0` | Explicitly selected SwiGLU in the reference configuration. |
| `13714386` | `FAILED 1:0` | Cleared an inherited Miniforge host compiler before Triton compilation. |
| `13714455` | `FAILED 1:0` | Moved Triton, TorchInductor, XDG, and temporary caches off the container root filesystem. |
| `13714533` | `FAILED 1:0` | Fixed a comparison-harness metric lookup after three BF16 reference and target repeats completed. |
| `13714605` | `FAILED 1:0` | BF16 passed; the blockwise reference was missing MCore's production `get_fp8_context` wrapper. |
| `13714801` | `COMPLETED 0:0` | Accepted BF16 and blockwise primitive parity. |

## Accepted result

Slurm job `13714801` completed every step with exit code `0:0` on node
`pool0-01422`. It emitted the non-skipped markers:

```text
HOPPER_BLOCKWISE_BF16_BASELINE_PARITY_OK
HOPPER_BLOCKWISE_PRIMITIVE_PARITY_OK
HOPPER_BLOCKWISE_PRIMITIVE_PARITY_JOB_OK
```

The comparison covers TP column linear, GQA QKV/core/output split, and the
reduced grouped-expert path. It checks public forward tensors, dX, every dW,
the FP32 AdamW result, BF16 attention-core/norm/router boundaries, and an
11-entry typed coverage manifest. Reference noise was zero, so both profiles
passed bitwise gates; target repeat noise was also zero.

The manifest SHA-256 is
`fcfd21e29db21af104a5437c2a77a9702579a1a9efb962e7df1fd6ee85231761`.
There are eight threshold files and eight comparison files, one for each
profile and rank. Artifacts and immutable stdout are stored under:

```text
/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/bayan/work/codex/fp8-primitives-c9c00dfac-cw/results/primitive-parity-13714801
/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/bayan/work/codex/fp8-primitives-c9c00dfac-cw/primitive-parity-13714801.out
```
