# Muon × DistOpt same-contract bitwise A/B — run evidence

Harness: `docs/runs/muon_distopt_compact_bitwise.sbatch` (MODE=all) driving
`experimental/lite/examples/bench/muon_distopt_correctness.py`. It compares the
MLite Muon distributed-optimizer lowering (`build_dist_opt_stack`) against the
upstream Megatron-Core TensorParallelMuon lowering under one fixed synthetic
BF16 model, DP=2, and pinned process groups, across the `continuous`, `save`,
and `resume` trajectories (4 steps, save at step 2).

## Integrated tree (Muon compact DistOpt + FSDP2 arm merged)

- Slurm job: **13875018** — `COMPLETED`, `ExitCode 0:0`, elapsed `00:01:55`,
  1× node, `gpu:2`, `NVIDIA H100 80GB HBM3`.
- MLite HEAD: `a872cd34d8a615eda27b0019c66914ed40558f47`
  (tree `6d57a05e1ada167bf2ac5315d8394d6d98653e61`) — the integration merge of
  the compact DistOpt arm (`62404d4ab`) and the FSDP2 arm (`84f02c7c7`).
- Pins: Megatron-Core `d64ba4ccb1e3e878c15171c9cc58d5d3b46bf4d5`,
  Emerging-Optimizers `b309e2f01cda75dc96a6dc1a2355a7b3b64b5e16`,
  nvidia-resiliency-ext `0.6.0`.
- Container: `pytorch_26.04-py3.sqsh`.

### Verdict (`comparison.json`, sha256 `ae7f722a1fc31d254c85b1a31c9ec1a98d90d0f658a67c3365eedfb6a36591bd`)

- `passed = True`, `mismatches = 0`.
- `tensor_checks = 2000`, `torch_equal_checks = 2000`,
  `assert_close_checks = 2000` (atol=0, rtol=0 — bitwise).
- 36 named checks all pass (upstream/mlite × rank0/rank1 ×
  resume-vs-continuous / resume-metadata / cross-implementation).
- `NON_SKIP_MUON_DISTOPT_COMPACT_BITWISE_PASSED` marker emitted
  (`world=2 steps=4 tensor_checks=2000`).

The mixed routing is exercised: the four weight matrices route to Muon; the
embedding/output/bias/norm tensors route to Adam under the distributed
optimizer, and their `exp_avg` / `exp_avg_sq` moments are part of the bitwise
comparison. This is the "not worse than Megatron Muon = bitwise/tolerance
parity" acceptance for the parent task.

Artifacts on cw:
`.../code/runtime/muon-final-ab-a872cd34d/artifacts/13875018/`.

## Real-Qwen3.5 Adam DistOpt gate (MODE=adam)

Passed on the compact parent `62404d4ab` as Slurm job **13697707**
(`COMPLETED 0:0`), text-only Qwen3.5, DP=1, marker
`NON_SKIP_PINNED_ADAM_DISTOPT_GATE_PASSED`. The FSDP2 merge delta
(`62404d4ab..a872cd34d`) touches only
`experimental/lite/megatron/lite/primitive/optimizers/fsdp2/{__init__,muon,optimizer}.py`
plus two FSDP2 unit tests — it does not touch `correctness.py`,
`muon_distopt_correctness.py`, `megatron_wrap.py`, or any DistOpt lowering
path, so that gate is unaffected by the integration merge. A fresh re-run on
the integrated tree is currently blocked by the referenced HF checkpoint path
(`.../coreai_dlalgo_mcore/checkpoints/hf/Qwen3.5-35B-A3B`) no longer being
present on the cluster (environment, not code).
