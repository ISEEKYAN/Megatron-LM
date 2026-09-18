# Native Nemotron-H alignment adapter

This correctness-first adapter implements BF16 Nemotron-H with native mlite
pipeline/context/expert parallelism and the distributed optimizer. It is not
the FSDP adapter. Expert dispatch uses all-to-all, not DeepEP.

Forward uses the companion vLLM alignment arithmetic; backward recomputes native
PyTorch/Transformers VJPs. In particular, this requires the non-stock vLLM
`nemotron_h_alignment` module, batch-invariant linear/logprob kernels, and aligned
Mamba kernels. Installing this change with an arbitrary stock vLLM is insufficient.

## Validated environment

- Torch 2.13.0+cu130, Transformers 5.16.1, Transformer Engine 2.19.0.
- Megatron-Core `15c83d2fcd00e283bb59ff26dce40266a445c615` on a separate source path.
- Companion vLLM 0.28.1rc1.dev580+g385dce36b plus Nemotron alignment patches.
- Full 52-layer BF16 historical `Nemotron-52L-aligned-cp2-sgd-step1` weights,
  not unmodified initial HF weights.
- GB200: two nodes, four GPUs each; TP1/PP2/CP2/EP4, dense DP2, ETP1.
- Rollout DP8/EP8, CUDA Graph FULL_AND_PIECEWISE, prefix cache disabled.

The example entry is `examples/verl/scripts/run_nemotron_alignment.sh` relative
to `experimental/lite`. Supply MODEL_PATH, TRAIN_FILES and OUTPUT_ROOT on an
already configured two-node Ray cluster. Both actor and rollout must enable
`nemotron_shared_norms`; the wrapper supplies both overrides. The tested verl
integration also has the raw-logprob equality gate, centralized-DP launch fix,
and per-session deterministic sampling seeds; those changes are not in this PR.

For the final two-step workload, set TRAIN_BATCH_SIZE=32, leave ROLLOUT_N=2,
and pass these overrides to the example:

```bash
actor_rollout_ref.actor.use_dynamic_bsz=False \
actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False
```

This uses one sequence per micro-batch. Dynamic packing previously exceeded
GPU memory in the second backward. Input/output limits are 2048/8192; observed
prompt maxima were only 278/268, while outputs reached 8192 in both steps.

## Evidence and limits

Both steps completed rollout, old-logprob recomputation, backward, optimizer,
and weight synchronization. Raw response logprobs were byte-identical across
126085 and 119482 tokens; probability-diff max, log-ppl diff and k3-KL were zero.
Rewards were all -1 and advantages/gradients zero in both steps. This does not
demonstrate consecutive alignment after a nonzero learning update, model
quality, production throughput, or full52 distributed checkpoint resume.

[Two-step metrics](https://wandb.ai/vime/nemotron-mlite-alignment/runs/native-two-step-15340-seedfix-micro1)

CPU correctness suites live in `tests/unit/model/test_nemotron_*_unit.py`.
CUDA/real-weight cases require the companion environment and NEMOTRON_TEST_MODEL;
skipped CUDA cases are not evidence of GPU correctness.
