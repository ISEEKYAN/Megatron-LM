# Muon for post-training: SFT and RL configuration

## Decision in one page

Do not reuse a pre-training Muon schedule for post-training. In particular, a
3,200-step warmup is longer than many complete SFT or RL jobs and turns a short
comparison into a test of the scheduler rather than the optimizer.

For Megatron's RMS-matched Muon convention, the practical starting points are:

| Workload | Muon peak LR | Warmup | Momentum | Nesterov | Weight decay | Status |
| --- | ---: | ---: | ---: | --- | ---: | --- |
| SFT, Adam-pretrained checkpoint | `2e-5`; screen `1e-5, 2e-5, 5e-5` | `0`; at most 10 stabilization steps | `0.9` | `false` | `0.01` | Experimental; always retain a tuned AdamW arm |
| SFT, Muon-pretrained checkpoint | `5e-5`; screen `2e-5, 5e-5` | `0` | `0.9` first, `0.95` only as a later sensitivity | `false` | `0.01` | Best-supported SFT use case |
| DAPO/GRPO canary | `1e-6`; screen `1e-7, 3e-7, 1e-6` | 10 steps, from `0.1 * peak_lr` | `0.9` | `false` | `0.1` | Research-only; positive and collapse results both exist |

These numbers assume all of the following are frozen:

- Megatron `muon_scale_mode=spectral`;
- `muon_extra_scale_factor=0.2` for the SFT starting point and `0.5` for the
  cited NeMo RL DAPO starting point;
- five Newton--Schulz steps, split QKV, and the normal Adam fallback for
  embeddings, output weights, biases, and normalization parameters;
- decoupled weight decay, with Megatron's normal no-decay overrides;
- the same checkpoint, data order, global token batch, clipping, and total
  optimized tokens across comparison arms.

Changing the scale mode or extra scale factor changes the meaning of the LR.
The table is not valid under Keller Jordan's original update-scaling convention
or under a DeepSpeed configuration with separate `muon_lr` and `adam_lr`.

The evidence supports a production recommendation for AdamW, not for Muon, in
RL. The RL row above is a bounded experiment configuration with explicit stop
gates. It must not be installed as an unconditional default.

## How the evidence was classified

This document uses three labels:

- **Observed**: a public paper, official configuration, or completed controlled
  run directly reports the value or behavior.
- **Inferred**: a conclusion follows from more than one observed result, but the
  exact configuration was not tested here.
- **Proposed**: an experiment design intended to resolve remaining uncertainty.

No new GPU experiment was run for this study.

## The LR convention must be named

There are two common Muon LR conventions, and their numeric values are not
interchangeable.

### Original/Keller convention

The [Keller Jordan reference implementation](https://github.com/KellerJordan/Muon)
shows a Muon LR much larger than the auxiliary Adam LR. Its example uses
`muon_lr=0.02` beside `adam_lr=3e-4`, `momentum=0.95`, Nesterov enabled, and
`weight_decay=0.01`. Its tuning guidance says to keep momentum, Nesterov, and
five NS steps fixed first, then tune LR and weight decay.

This is useful evidence for the algorithm's defaults. It is not a numeric LR
reference for Megatron's Moonlight-style RMS matching.

### Moonlight/Megatron RMS-matched convention

The [Moonlight paper](https://arxiv.org/abs/2502.16982) scales each matrix update
by shape and then matches its RMS to AdamW. Under that convention, the paper
states that Muon can reuse an LR and weight decay tuned for AdamW. Megatron adds
an explicit `muon_extra_scale_factor`; the post-training examples discussed
below use `0.2` or `0.5`.

This is the convention assumed by the recommendation tables. A fair run must
record the scale mode and extra factor alongside the LR. Recording `lr=2e-5`
without them is incomplete.

The pinned implementation baseline used while designing MLite is:

- [Megatron-LM `d64ba4ccb`](https://github.com/NVIDIA/Megatron-LM/tree/d64ba4ccb1e3e878c15171c9cc58d5d3b46bf4d5);
- [Emerging-Optimizers `b309e2f`](https://github.com/NVIDIA-NeMo/Emerging-Optimizers/tree/b309e2f01cda75dc96a6dc1a2355a7b3b64b5e16).

The entry points disagree on defaults: the pinned Megatron training CLI uses
momentum `0.9` and no Nesterov, while the standalone Emerging-Optimizers/Keller
family commonly uses momentum `0.95` with Nesterov. Therefore post-training
configs must set both fields explicitly. The direct NeMo post-training examples
use `0.9` and no Nesterov, so those are the first MLite A/B values.

## What the public post-training results actually show

| Source | Observed result | What it does not establish |
| --- | --- | --- |
| [Moonlight, Section 3.5](https://arxiv.org/abs/2502.16982) | Two-epoch Tulu-3 SFT used a linear decay from `5e-5` to zero. Muon-pretrained + Muon-SFT was the strongest optimizer pairing. On Adam-pretrained Qwen2.5-7B, Muon SFT used cosine `2e-5 -> 2e-6` and was roughly comparable but worse on all four reported tasks. | It does not provide a universal LR ratio, nor evidence that Adam-pretrained checkpoints should switch optimizers. |
| [Kimi K2](https://arxiv.org/abs/2507.20534) | K2 says it uses Muon during post-training and recommends it for fine-tuning a Muon-pretrained K2 checkpoint. | It does not disclose an SFT LR, warmup, momentum, or weight decay, and it does not disclose a reproducible Muon RL optimizer recipe. K2's `500`-step warmup and `2e-4` LR are explicitly pre-training values. |
| [NeMo RL Muon guide](https://docs.nvidia.com/nemo/rl/latest/guides/muon-optimizer.html) | The official Qwen3-235B SFT command sets LR `2e-5`, momentum `0.9`, no Nesterov, spectral scaling, and extra scale `0.2`. Its inherited SFT config uses weight decay `0.01` and effectively no warmup. The guide labels support experimental. | A plotted comparison is not a broad cross-model default. The guide also contains generic YAML with different values, so the exact command and inherited config must be resolved together. |
| [PyTorch/DeepSpeed fine-tuning report](https://pytorch.org/blog/using-muon-optimizer-with-deepspeed/) | One-epoch Moonlight-16B-A3B fine-tuning used a separate Muon LR `1e-4` and Adam fallback LR `2e-6`, batch 16, clipping 1.0, and reported small mixed gains. | DeepSpeed's separate-LR configuration cannot be copied into Megatron's shared-LR RMS-matched configuration. |
| [Can Muon Fine-tune Adam-Pretrained Models?](https://arxiv.org/abs/2605.10468) | Naively switching Adam-pretrained models to full Muon fine-tuning can degrade performance; the disruption grows with update strength. LoRA reduces the gap. Full-Muon selected LRs were commonly `2e-5` to `5e-5` after a sweep, not a fixed Adam-to-Muon multiplier. | It does not validate MLite's exact Megatron scale mode or make full Muon fine-tuning safe by lowering LR alone. |
| [NeMo RL DAPO recipe and Muon guide](https://docs.nvidia.com/nemo/rl/latest/guides/muon-optimizer.html) | The Muon DAPO command inherits Qwen2.5-7B DAPO `lr=1e-6`, constant LR after a 10-step warmup from `1e-7`, and weight decay `0.1`; it sets momentum `0.9`, no Nesterov, and extra scale `0.5`. The guide reports minor post-training improvements in its tested cases. | It is one experimental stack, model, reward, and data contract. It is not evidence that Muon is stable for RLVR generally. |
| [Rethinking Muon Beyond Pretraining](https://arxiv.org/abs/2605.19282) | On reported Qwen3 GRPO/GMPO RLVR experiments, vanilla Muon collapsed toward zero accuracy; the paper attributes the failure to whitening low-SNR tail directions and loss of per-head specialization. | One negative study does not prove every Muon RL run fails, but it rules out an unconditional recommendation. |

The 2026 paper on Adam-to-Muon mismatch and the RLVR collapse paper postdate the
original pre-training-centric recipes. They are why the tables above are more
conservative than a direct copy of Moonlight pre-training settings.

## Why the previous 100-step A/B was not informative

The previous protocol gave Muon:

```text
peak_lr = 1.5e-4
linear_warmup_steps = 3200
observed_steps = 100       # steps 0 through 99
```

During linear warmup,

```text
lr(t) = 1.5e-4 * t / 3200
```

Therefore:

- step 99 LR was `4.640625e-6`, only `3.09%` of Muon's intended peak;
- the run observed only `3.125%` of the warmup duration;
- mean Muon LR over steps 0--99 was `2.3203125e-6`;
- against a constant Adam LR of `1e-4`, cumulative LR exposure was only
  `2.32%` of Adam's exposure over the same 100 steps.

The observed loss difference is real for that frozen scheduler, but its correct
interpretation is:

> A 3,200-step pre-training warmup does not let Muon reach a useful LR inside a
> 100-step window.

It is not evidence that Muon is ineffective. Extending that exact run past
3,200 steps would test a pre-training recipe, not answer the post-training
question. The post-training fix is to use a post-training LR and short warmup.

## SFT configuration

### Recommended first MLite arm

Use this for a full-parameter SFT canary from an Adam-pretrained checkpoint:

```yaml
optimizer: muon
lr: 2.0e-5
min_lr: 2.0e-6
weight_decay: 0.01
clip_grad: 1.0
lr_warmup_steps: 0
lr_decay_style: cosine       # linear is also valid if frozen across arms
muon_momentum: 0.9
muon_nesterov: false
muon_scale_mode: spectral
muon_extra_scale_factor: 0.2
muon_num_ns_steps: 5
muon_split_qkv: true
```

This is a **proposed synthesis**, not a published MLite result. It combines the
direct NeMo SFT command with the Moonlight public-checkpoint LR range and uses a
nonzero LR floor only to avoid making end-of-run decay dominate a short canary.
For the final two-epoch confirmation, either decay to zero as Moonlight did or
use the established Adam baseline's end-LR fraction; choose before seeing the
results.

Screen `1e-5`, `2e-5`, and `5e-5` without changing momentum, weight decay, scale
mode, or extra factor. If the checkpoint was itself Muon-pretrained, start with
`5e-5` and include `2e-5` as the conservative arm.

### SFT warmup

For a pretrained checkpoint, use zero warmup by default. If the first few steps
show non-finite gradients or a reproducible clipping burst, allow exactly 10
linear warmup steps from `0.1 * peak_lr`; do not introduce a percentage that can
silently become hundreds or thousands of steps when the dataset changes.

A 100-step run with zero warmup can screen obviously bad LRs. It cannot decide
quality or forgetting. The efficacy comparison must cover at least one full
epoch; the closest public Moonlight reference uses two epochs.

### SFT momentum and weight decay

Keep momentum at `0.9` and Nesterov disabled for the first comparison because
that matches the direct Megatron/NeMo post-training path. Only after choosing an
LR may a separate `0.95` sensitivity be useful. Changing LR and momentum in the
same arm makes the result uninterpretable.

Use decoupled weight decay `0.01` for SFT and retain the normal zero-decay groups
for biases and normalization parameters. Do not use `weight_decay=0` merely
because the horizon is short: the Moonlight analysis identified weight growth
as a Muon failure mode, and direct SFT examples retain weight decay.

### Adam-pretrained checkpoints need an extra guardrail

For an Adam-pretrained checkpoint, full Muon SFT is experimental even when its
training loss looks good. Report held-out general capability and distance from
the initial checkpoint, not only SFT validation loss. If a parameter-efficient
path is in scope, LoRA-Muon is a more defensible follow-up than silently raising
the full-fine-tune LR, but it is a separate experiment and not covered by the
configuration above.

## RL configuration

### Recommended canary, not a default

Resolve the current NeMo DAPO reference into an explicit MLite arm:

```yaml
optimizer: muon
lr: 1.0e-6
min_lr: 1.0e-6
weight_decay: 0.1
clip_grad: 1.0
lr_warmup_steps: 10
lr_warmup_init: 1.0e-7
lr_decay_style: constant
muon_momentum: 0.9
muon_nesterov: false
muon_scale_mode: spectral
muon_extra_scale_factor: 0.5
muon_num_ns_steps: 5
muon_split_qkv: true
```

Screen peak LRs `1e-7`, `3e-7`, and `1e-6`. For each arm, set
`lr_warmup_init=0.1 * peak_lr` and keep the 10-step warmup. Do not change the
extra scale factor during this screen.

The weight decay value `0.1` is retained to keep the Muon arm on the same DAPO
regularization contract as the official Adam reference. It is not claimed to be
independently optimal for RL. Likewise, momentum `0.9` is frozen to the direct
post-training example rather than tuned jointly with LR.

### RL must fail fast

The canary must log at every validation interval:

- train reward and held-out accuracy with the complete denominator;
- KL to the starting/reference policy, entropy, response length, and clip
  fraction;
- gradient norm, update RMS by parameter family, skipped/non-finite steps, and
  maximum attention logits when available;
- identical metrics for the tuned Adam baseline.

Stop the Muon arm on non-finite values, near-zero validation collapse, or a
pre-frozen capability-retention breach. Do not wait for train reward to recover
after held-out capability has collapsed. A positive 100-step canary authorizes a
longer comparison; it is not an efficacy result.

## A/B protocol ready for a GPU budget

### 1. Freeze the contract

Before submitting any job, write one machine-readable manifest containing:

- MLite, Megatron, Emerging-Optimizers, container, and checkpoint revisions;
- checkpoint optimizer provenance: Adam-pretrained or Muon-pretrained;
- exact named-parameter routing and Adam fallback groups;
- scale mode, extra factor, NS coefficients/steps, QKV splitting, momentum,
  Nesterov, LR schedule, weight decay groups, and clipping;
- dataset revision, shuffle seed/order, packed token counts, global token batch,
  total optimizer steps, and validation cadence;
- rollout sampler, reward, KL, loss aggregation, and reference-policy settings
  for RL.

Do not combine Adam-pretrained and Muon-pretrained checkpoints in one pooled
claim. Optimizer provenance is an experimental variable.

### 2. Tune without giving either optimizer a scheduler handicap

Use the established tuned AdamW arm unchanged. Tune only Muon's peak LR in the
first screen. Both optimizers must see the same optimized tokens, data order,
batch, and validation calls; their numeric LRs need not be equal.

For SFT:

1. Run AdamW plus Muon at `1e-5`, `2e-5`, and `5e-5` for 200 steps, one seed.
   Use zero warmup, or the same pre-declared 10-step stabilization rule for all
   Muon LR arms. This is an LR elimination screen only.
2. Select by frozen SFT validation plus capability retention, not training loss.
3. Run tuned AdamW and the selected Muon arm for two full epochs, three seeds.

For DAPO/GRPO:

1. Run AdamW plus Muon at `1e-7`, `3e-7`, and `1e-6` for 100 actor updates, one
   seed. Muon uses the 10-step warmup, so 90% of the canary is at peak LR.
2. Eliminate any arm that trips a stability or retention gate.
3. Run tuned AdamW and the surviving Muon arm for at least 1,000 actor updates,
   three seeds. Proceed to the full 10,000-step recipe only if the pre-frozen
   validation gate is positive.

The 200/100-step screens choose a region. The two-epoch SFT and 1,000-step RL
runs are the minimum comparisons on which to discuss effectiveness.

### 3. Compare outcomes, not just successful steps

The primary report must include:

- validation loss/accuracy or RL held-out accuracy at equal optimized tokens;
- area under the validation curve and time/tokens to a pre-declared target;
- mean and per-seed results, with failed seeds in the denominator;
- capability retention versus the starting checkpoint;
- wall time, optimizer-step time, peak memory, and communication volume;
- update RMS by parameter family to verify the intended scaling convention.

"The optimizer stepped" and "training loss decreased" are correctness smoke
results, not evidence that Muon is more effective than AdamW.

### 4. Interpretation rules

- If Muon wins only at a larger LR, report a tuned-optimizer win, not an
  algorithm-only effect.
- If Muon loses during warmup but catches up later, report time-to-target and
  full-horizon results; do not crop the curve.
- If SFT improves the target task but harms retained capabilities, report the
  tradeoff rather than declaring a win.
- If RL reward rises while held-out accuracy, entropy, or response diversity
  collapses, fail the arm.
- If the result changes between Adam-pretrained and Muon-pretrained checkpoints,
  report optimizer-provenance interaction as the conclusion.

## Source and support notes

The strongest configuration references are the official post-training guide and
its inherited configs:

- [NeMo RL Muon guide](https://docs.nvidia.com/nemo/rl/latest/guides/muon-optimizer.html);
- [Qwen2.5-7B DAPO config](https://github.com/NVIDIA-NeMo/RL/blob/main/examples/configs/recipes/llm/dapo-qwen2.5-7b.yaml);
- [Megatron SFT config](https://github.com/NVIDIA-NeMo/RL/blob/main/examples/configs/sft_openmathinstruct2_megatron.yaml).

Megatron and Emerging-Optimizers source define implementation behavior and
defaults, but their functional/pre-training recipes do not override the direct
post-training references. Community reports are useful for discovering sharp
edges, not for selecting an LR. For example, [Megatron issue #2164](https://github.com/NVIDIA/Megatron-LM/issues/2164)
documents that `dist_muon` and the ordinary distributed-optimizer flag select
different paths; it provides no post-training hyperparameter evidence.

Finally, all linked projects are active. A delivered experiment must replace
moving `main` links with the exact revisions recorded in its manifest.
