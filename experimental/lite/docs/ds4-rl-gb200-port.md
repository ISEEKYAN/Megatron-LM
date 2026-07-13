# DS4 colocated RL harness — cw H100 → GB200/oci-hsg port

Executes the 2026-07-12 23:41 top-pinned order: run the 32-card GB200 resync
fail-fast smoke on the GPU-verified GB200 stack (cd0de48 vLLM overlay + q35
colocated recipe), reusing — not rebuilding — the existing colocated harness.

## Why a port, not a rebuild

The colocated harness is a single parameterized sbatch already in this repo:
`experimental/lite/examples/verl/slurm/run_ds4_gsm8k_grpo.sbatch`. It is driven
entirely by env pointers (base image, overlays, mcore/verl/mlite roots,
checkpoint, topology). "别重复建 harness" = reuse this sbatch and swap the
pointers from the cw H100/x86/NGC26.04/vLLM-0.20.2-thin stack to the GB200
NGC26.06/cd0de48 stack. All GB200 artifacts already exist (built by 1.1.22.1 /
1.1.22.3); nothing new is compiled.

## Env-pointer map (cw → GB200/oci-hsg)

HSG project scratch root:
`/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/bayan`
(abbrev `$B` below; `$G=$B/llmrl/gb200-vllm-ds4`).

| var | cw (H100/x86, deleted 0.20.2 stack) | GB200/oci-hsg (this port) |
|---|---|---|
| BASE_IMAGE | env/pytorch_26.04-py3.sqsh | `$G/containers/pytorch-26.06-py3.sqsh` |
| MLITE_SRC | runtime/.../source-<commit> | HSG snapshot of this worktree HEAD (git checkout at MLITE_COMMIT) |
| MLITE_COMMIT | <cw HEAD> | this worktree HEAD |
| MEGATRON_ROOT | ds4-hopper-15d86c49d/mcore | `$G/src/Megatron-LM-ds4-sm100` (or a resync-* checkout matching MLITE_SRC's mcore contract) |
| VERL_ROOT | ds4-hopper-15d86c49d/verl | `$B/rl-mlite/verl` (inner pkg `verl/verl` is on PYTHONPATH via VERL_ROOT) |
| MLITE_SM90_SITE | mlite-2604-verl-dsa-sm90-overlay/.../site-packages | `$G/overlays/mlite-dsa-sm100-a6ec2ba7bd0a7dff98b3f4d3e6b52b159c48d78b-ngc2606/lib/python3.12/site-packages` |
| DS4_VLLM_SITE | mlite-2604-ds4-vllm020-thin/.../site-packages | `$G/overlays/vllm-cd0de48d0883ecb8e1ef350a99baa0c158f58e82-ngc2606/lib/python3.12/site-packages` |
| DS4_VLLM_SHIM | .../libvllm_torch212_abi_shim.so | **empty** (native cd0de48 build — no ABI gap; sbatch now treats it optional) |
| CHECKPOINT_DIR | code/models/DeepSeek-V4-Flash | `$B/models/DeepSeek-V4-Flash-Base` (verify tokenizer_mode=deepseek_v4 auto) |
| partition/SLURM_CONF | cw-dfw-cs-001 | HSG `batch`/`batch_long`; no SLURM_CONF override |

Note: `MLITE_SM90_SITE` is a variable name, not a hardware assertion — on GB200
it points at the SM100 DSA overlay. The `nvidia_cutlass_dsl/python_packages`
subpath prepended in PYTHONPATH (sbatch line ~160) may be absent in the SM100
overlay; a missing PYTHONPATH entry is silently ignored by Python, so it is
harmless. The CONFIG_ONLY gate will confirm the import chain regardless.

## Topology: gpu:8/node (cw) → gpu:4/node (GB200 NVL4)

GB200 = 4 GPU/node. 32 cards = 8 nodes. The default actor parallel
PP4·EP8·CP4·TP1 = 128-way does NOT divide 32 (sbatch divisibility guard).
The 32-card smoke config must reduce to a 32-way actor parallel, e.g.
**PP2·EP8·CP2·TP1 = 32** (cw already exercised `l1-...-pp2ep8cp2`). ROLLOUT_TP
must divide 32 and fit within a node's memory: ROLLOUT_TP=4 (single-node TP)
is the conservative smoke choice.

Two sbatch gates hardcode `GPUS_PER_NODE==8` and must be relaxed for GB200
before the 8-GPU proxy / load-only arms run there:
- `VLLM_LOAD_ONLY` guard (requires 1 node × 8 GPU) → on GB200 the 8-GPU proxy
  is 2 nodes × 4 GPU; the guard needs a gpu:4 branch.
- `RAY_ONLY` guard (1–2 nodes × 8 GPU) → same.
`CONFIG_ONLY` requires GPUS_PER_NODE=0 (CPU-only) and is topology-agnostic — it
needs **no** topology edit and is the first gate to run.

## Gate sequence (CLAUDE.md pre-GPU 铁律)

1. **Zero-GPU CONFIG_ONLY** (1 CPU node, GPUS_PER_NODE=0): validates the full
   init chain — Ray runtime-env snapshot, PYTHONPATH import of
   dsa-sm100 + cd0de48-vllm + mcore + verl + mlite together in the NGC26.06
   image, config parse. This is the untested integration (training overlay +
   rollout overlay coexisting in ONE process was never done on GB200; q35 smoke
   was training-only, 1.1.22.1 smoke was rollout-only). CONFIG_ONLY is where a
   coexistence break surfaces cheaply.
2. **8-GPU (2×4) load-only + residency probe** on GB200 (needs the gpu:4
   guard edit): confirms per-worker residency under the cd0de48 stack.
3. **8-GPU (2×4) VLLM_CHECKPOINT_SYNC_PROBE / resync smoke** (MLITE_RESYNC_
   SMOKE_EXIT_AFTER=1 + MEMLOG): the actual resync-OOM question at proxy scale.
4. **32-GPU (8×4) smoke**, PP2·EP8·CP2, resync fail-fast + memlog: the ordered
   deliverable. ~20–35 min.

## Gate that does NOT apply here

1.1.22.4's "短步未过不发32卡" / 22.2-consistency-verdict gate governs the
*accuracy* claim of a DAPO training run. This is a resync **memory / residency
bring-up smoke** (load + one weight resync + fail-fast), asserting nothing about
quantization numerical correctness, so that gate is orthogonal. bayan/总秘书
explicitly pre-authorized this smoke as immediately executable.

## Status

- [x] Feasibility proven: bounded env-pointer port; all GB200 artifacts present.
- [x] sbatch shim made optional (native cd0de48 overlay, no LD_PRELOAD).
- [ ] Stage this worktree HEAD → HSG as MLITE_SRC.
- [ ] GB200 CONFIG_ONLY fire script + launch (gate 1).
- [ ] gpu:4 guard edit for 8-GPU proxy (gates 2–3).
- [ ] 32-card smoke (gate 4).
