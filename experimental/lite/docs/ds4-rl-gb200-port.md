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

---

# cw/H100 native-vLLM-0.23 route (2026-07-13, bayan 拆墙令 → cw = 主战场)

bayan (2026-07-12 22:40 → 07-13 00:24) reversed the GB200-first plan: the
"architecture wall" is void — q35 colocated RL has always run on cw/H100 x86
(1.13.8 mfsdp DAPO jobs). Directive: base the DS4 rollout on the **proven q35
image** and add only the DS4 training pieces as a thin overlay; cw is the main
battlefield, α/β (128-card cluster) closed onto cw.

## Base image = the q35 `verl.vllm023.sqsh` (native vLLM 0.23)

`/home/bayan/code/verl_optimize/verl.vllm023.sqsh` (cw host path, x86, 28.9 GB,
Jun 26; TASK-1.1.15.2 locked). Zero-GPU `unsquashfs` audit — it already provides
the **entire** DS4 stack except one kernel:

- python 3.12, **torch 2.11.0+cu130**, **vllm 0.23.1.dev0** (registry lists
  `DeepseekV4ForCausalLM` — native DS4, empirically confirmed).
- transformer_engine 2.15.0, megatron_core 0.16.1, flash_attn 2.8.3, apex,
  nvidia_cutlass_dsl 4.5.2 (cu13), tilelang 0.1.9.
- `cudnn 1.25.0` with `deepseek_sparse_attention` (DSA) incl. `dsa_*_sm90`
  (Hopper) + `dsa_*_sm100`.
- **Missing: `flash_mla`** — the one DS4 training/rollout kernel not in base.

## Thin sm90 overlay = flash_mla only (ABI-verified, NO rebuild)

The old fat cw overlay `mlite-2604-verl-dsa-sm90-overlay` is UNUSABLE as-is: it
is a full venv that bundles its own **vllm 0.12.0**, which — because the sbatch
*prepends* `MLITE_SM90_SITE` to PYTHONPATH — would shadow the base's native
0.23. Its `flash_mla/cuda*.so` links `libcudart.so.13` (CUDA 13, same major as
base) and it carries no torch (no torch shadow).

Extracted **only** `flash_mla/` + dist-info into a thin overlay:
`/lustre/.../users/bayan/llmrl/ds4-cw-thin/mlite-ds4-flashmla-sm90-cu13/lib/python3.12/site-packages`
(33 MB). Zero-GPU import probe inside the base container (Slurm cpu_short job
**13882421, rc=0**): `import flash_mla` + all symbols OK under torch 2.11+cu130;
`vllm` resolves to base `/vllm/vllm` (unshadowed); DSA imports from base. →
**flash_mla is ABI-compatible; no recompile needed.**

## cw pointer map (native route)

| var | value |
|---|---|
| BASE_IMAGE | `/home/bayan/code/verl_optimize/verl.vllm023.sqsh` |
| MLITE_SM90_SITE | `$B/llmrl/ds4-cw-thin/mlite-ds4-flashmla-sm90-cu13/lib/python3.12/site-packages` (thin: flash_mla only) |
| DS4_VLLM_SITE | **empty** (base is native vLLM 0.23; no rollout overlay) |
| DS4_VLLM_SHIM | empty (native) |
| MEGATRON_ROOT | `$B/code/runtime/ds4-hopper-15d86c49d/mcore` (or a resync-* checkout matching MLITE_SRC's mcore contract) |
| VERL_ROOT | `$B/code/runtime/ds4-hopper-15d86c49d/verl` |
| CHECKPOINT_DIR | official DeepSeek-V4-Flash mixed checkpoint (verify tokenizer_mode=deepseek_v4) |
| account / partition | `coreai_devtech_all` / gpu `batch*`, cpu `cpu_short` |

## Remaining work: harness gate-suite re-baselining (NEXT FRAME, pre-GPU moe门)

The sbatch `run_ds4_gsm8k_grpo.sbatch` is **hardcoded to the deleted 0.20.2
overlay** in three embedded gate blocks — this is NOT a pointer swap:

1. `DS4_VLLM_SITE` is mandatory (`:?`) + existence-checked + PYTHONPATH-prepended
   (lines 20/36/166/169/191/271/357). Native route needs it **optional/empty**:
   when empty, don't inject it into PYTHONPATH and treat rollout site = base.
2. **Server-import block** (~240-260, IMPORT_ONLY): asserts
   `transformers==5.12.1`, `vllm==0.20.2`, and `commonpath(module, rollout_site)
   == rollout_site` (i.e. vllm/transformers loaded *from* the overlay). Must
   accept base 0.23 loaded from `/vllm` / base dist-packages.
3. **Device-probe block** (~597-719, GPU-gated `ray.remote(num_gpus=1)`):
   `expected_dependency_versions` pins the whole 0.20.2 set and asserts every
   dep + PYTHONPATH ordering resolves under `rollout_site`. **GPU-only
   validation.**

Base pinset to re-baseline against (zero-GPU audited): apache-tvm-ffi 0.1.9,
**compressed-tensors 0.17.0** (was 0.15.0.1), numba 0.65.0, outlines-core
0.2.14, **transformers 5.3.0** (was 5.12.1 — a *downgrade*; must confirm
DeepseekV4 config/tokenizer support in 5.3.0), **vllm 0.23.1.dev0** (was
0.20.2), xgrammar 0.2.2.

Gate sequence after re-baselining: zero-GPU CONFIG_ONLY → 8-GPU (1×8) load-only
+ residency + resync-smoke → 32/128-card. The re-baselined sbatch gates GPU runs
→ route it through the pre-GPU moe门 before firing GPU jobs.
