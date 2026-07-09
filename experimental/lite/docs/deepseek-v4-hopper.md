# DeepSeek V4 on Hopper

This runbook defines the minimal H100 (SM90) environment for validating the
DeepSeek V4 MLite training path and the vLLM rollout binary. It deliberately
uses separate Python profiles: the vLLM thin overlay upgrades packages such as
`transformers`, so putting it on the MLite training `PYTHONPATH` can invalidate
the training environment.

## Pinned environment

Use an NGC PyTorch 26.04 container (Python 3.12, Torch 2.12, CUDA 13.2) and
assemble the following directories on a CPU or login node. Environment assembly
must not request a GPU. Do not import Torch, Transformer Engine, FlashMLA, or
vLLM while assembling the directories; those imports belong in the Slurm smoke.

The training profile is the SM90 VERL/DSA overlay:

- `nvidia-cudnn-frontend==1.25.0`
- `flash-mla==1.0.0+b7643bd`
- `nvidia-cutlass-dsl==4.5.2`
- `flashinfer-python==0.5.3`
- `ray==2.54.0`
- `vllm==0.12.0` is present but is not used for DeepSeek V4 rollout

The rollout profile prepends a thin overlay to that training profile:

- `vllm==0.20.2`
- `transformers==5.12.1`
- `triton==3.6.0`
- `tilelang==0.1.9`
- `apache-tvm-ffi==0.1.9`
- an ABI library that forwards Torch 2.11's
  `at::cuda::getCurrentCUDABlasHandle()` call to the Torch 2.12 signature

The thin overlay must not replace Torch or CUDA packages from the container.
The ABI library must link against `libtorch_cuda`; a library that merely leaves
the Torch 2.12 symbol unresolved can pass import and fail on the first GEMM.

Set these paths to the local installation:

```bash
export MLITE_SM90_SITE=/path/to/sm90-overlay/lib/python3.12/site-packages
export DS4_VLLM_SITE=/path/to/ds4-vllm-thin/lib/python3.12/site-packages
export DS4_VLLM_SHIM=/path/to/ds4-vllm-thin/abi_shim/libvllm_torch212_abi_shim.so
export MEGATRON_ROOT=/path/to/compatible/Megatron-LM
export VERL_ROOT=/path/to/verl
```

`MEGATRON_ROOT` is required even when the MLite delivery branch contains only
`experimental/lite`: the distributed optimizer and checkpoint path import
`megatron.core`. Use a committed checkout compatible with the MLite baseline;
do not point a validation job at a mutable shared worktree.

Check only file layout and package metadata on the login node. These commands
do not import a GPU library:

```bash
CHECK_ONLY=1 bash experimental/lite/examples/verl/scripts/run_deepseek_v4_hopper_smoke.sh training
CHECK_ONLY=1 bash experimental/lite/examples/verl/scripts/run_deepseek_v4_hopper_smoke.sh rollout-probe
```

The expected markers are `DS4_HOPPER_ENV_CHECK_PASSED mode=training` and
`DS4_HOPPER_ENV_CHECK_PASSED mode=rollout-probe`.

## Minimal Slurm smoke

Run all CUDA validation in a single-node Slurm allocation. Two H100s are enough
for the tiny MLite proxy: CP=2 exercises the SM90 DSA route, then a distributed
optimizer step must change parameters, and checkpoint save/load must restore
the updated parameters bitwise.

```bash
srun --nodes=1 --gres=gpu:2 \
  --container-image="$BASE_IMAGE" \
  --container-mounts=/lustre:/lustre \
  --container-workdir="$REPO_ROOT" \
  bash -lc '
    set -euo pipefail
    export PATH=/usr/local/bin:/usr/bin:/bin
    hash -r
    export TMPDIR=/tmp/ds4-${SLURM_JOB_ID:-local}
    mkdir -p "$TMPDIR"
    export MLITE_SM90_SITE=/path/to/sm90-overlay/lib/python3.12/site-packages
    export DS4_VLLM_SITE=/path/to/ds4-vllm-thin/lib/python3.12/site-packages
    export DS4_VLLM_SHIM=/path/to/ds4-vllm-thin/abi_shim/libvllm_torch212_abi_shim.so
    export MEGATRON_ROOT=/path/to/compatible/Megatron-LM
    export VERL_ROOT=/path/to/verl
    NPROC_PER_NODE=2 bash experimental/lite/examples/verl/scripts/run_deepseek_v4_hopper_smoke.sh training
    bash experimental/lite/examples/verl/scripts/run_deepseek_v4_hopper_smoke.sh rollout-probe
  '
```

Reset `PATH` after entering the container. Slurm can preserve a login-node
Miniforge path; if that path wins, Torch may appear importable while the
container's Transformer Engine extension cannot be found. Record
`command -v python` and `command -v torchrun` in the job log when diagnosing an
environment failure.

Keep `TMPDIR` short and node-local. Megatron distributed checkpointing creates
a multiprocessing manager socket below that directory; a deeply nested Lustre
path can exceed the AF_UNIX path limit before checkpoint I/O starts.

The training command must print
`NON_SKIP_VERL_MLITE_RUNTIME_THD_CP_SMOKE_PASSED` with finite `loss`, positive
`optimizer_grad_norm`, positive `param_delta_sum`, and
`checkpoint_roundtrip=bitwise`. The rollout probe forces `vllm._C` ABI
resolution and checks the `DeepseekV4ForCausalLM` registry entry; it must print
`DS4_HOPPER_VLLM_ENV_PROBE_PASSED`. The probe does not replace a full-checkpoint
rollout test when changing vLLM, Torch, CUDA, or the ABI library.

## BF16 training to pure-FP8 resync proxy

The official DeepSeek-V4 Flash checkpoint is a mixed artifact: scaled dense
matrices use block FP8 and routed experts use MXFP4. The training master must
load those values into BF16, while rollout resync must export every quantized
matrix, including routed experts, as block FP8 with FP32 scales. Do not use an
FP4 or MXFP4 rollout arm.

For a bounded H100 validation, use
`examples/verl/slurm/run_ds4_hopper_resync.sbatch`. It composes three checks in
one two-GPU allocation:

1. the two-rank MLite DS4 CP smoke performs a BF16 forward/backward,
   distributed-optimizer step, and bitwise checkpoint reload;
2. `ds4_hopper_resync_proxy.py` reads one scaled dense matrix and one routed
   expert matrix directly from the official mixed checkpoint, dequantizes a
   block-aligned crop to BF16, runs four deterministic optimizer steps, and
   sends both updated matrices through the production DS4
   `export_resync_weights(..., expert_dtype="fp8")` path;
3. the vLLM thin profile resolves the CUDA extension ABI and DS4 registry.

Prepare immutable MLite, MCore, and VERL snapshots on the login node, then
submit with paths exported explicitly:

```bash
export BASE_IMAGE=/path/to/pytorch_26.04-py3.sqsh
export MLITE_SRC=/path/to/committed/Megatron-LM
export MLITE_COMMIT=$(git -C "$MLITE_SRC" rev-parse HEAD)
export MEGATRON_ROOT=/path/to/committed/Megatron-LM-core
export VERL_ROOT=/path/to/committed/verl
export MLITE_SM90_SITE=/path/to/sm90-overlay/lib/python3.12/site-packages
export DS4_VLLM_SITE=/path/to/ds4-vllm-thin/lib/python3.12/site-packages
export DS4_VLLM_SHIM=/path/to/ds4-vllm-thin/abi_shim/libvllm_torch212_abi_shim.so
export CHECKPOINT_DIR=/path/to/official/DeepSeek-V4-Flash
export OUTPUT_DIR=/path/to/fresh/results
sbatch experimental/lite/examples/verl/slurm/run_ds4_hopper_resync.sbatch
```

The proxy stores both arms' full distributions as FP32 tensors. Its JSON report
also recomputes metrics after BF16 rounding and records per-weight relative L2
and maximum error, KL, selected-token log-probability delta, importance-ratio
deviation, clipping crossings, training loss/gradient/update evidence, and
component timings. A successful run ends with
`DS4_HOPPER_TRAIN_RESYNC_PASSED` and retains `report.json`,
`bf16-trained.pt`, and `fp8-export.pt`.

This is intentionally a mechanism proxy. It preserves official checkpoint
encodings and the production resync adapter, but it does not execute the full
model or vLLM's full-model FP8 kernels. Do not use it to override a failed or
missing full-model comparison on Blackwell. Performance numbers are useful
only for repeated runs of this exact proxy; compare neither its four-step
matrix loop nor the CP smoke wall time with a differently shaped GB200 run.

### Observed SM90 validation envelope

A two-H100 run of the frozen recipe completed in 121 seconds. The CP=2 MLite
smoke produced a finite loss of 5.373483, optimizer gradient norm 1.926429,
parameter delta 34338.04, and a bitwise checkpoint round trip. The official
checkpoint tensor proxy then produced the following bounded measurements:

- BF16 four-step loss: 4.852026 to 4.850984; parameter delta: 886.185;
- BF16-to-block-FP8 export: 19.1 milliseconds; BF16 training loop: 619.4
  milliseconds;
- dense/expert dequantized relative L2: 2.574% / 2.464%;
- FP32 p99 KL: 3.99e-7; p99 and maximum selected-token importance-ratio
  deviation: 1.57e-5; clipping crossings: zero;
- BF16-rounded log-probability deltas: zero for this proxy.

The runtime reported an H100 80GB at compute capability 9.0 and Torch
`2.12.0a0+5aff3928d8.nv26.05`. Both exported matrices, including the routed
expert that started as MXFP4, were `float8_e4m3fn` with FP32 scales.

There is no same-shape SM100 result for this proxy, so the cross-architecture
performance ratio is **unmeasured**. The 19.1 ms export number and 121 s total
job time are an H100 scheduling/debug envelope, not evidence that Hopper is
faster or slower than GB200. Obtain a ratio only by running this exact proxy,
source revision, and container contract on SM100; do not compare against the
pending full-model Blackwell workflow.

## Hopper and Blackwell differences

| Surface | H100 / SM90 | GB200 / SM100 | Reuse rule |
| --- | --- | --- | --- |
| Base container | NGC PyTorch 26.04, Torch 2.12, CUDA 13.2 | Use the Blackwell-qualified container and its Torch/CUDA pair | Do not copy binary extensions between the containers. |
| MLite model, Megatron Core, optimizer, checkpoint format | Shared source contracts | Shared source contracts | Pin both MLite and Megatron Core commits for every run. |
| FlashMLA | SM90 build (`1.0.0+b7643bd`) | SM100 build with the Blackwell kernels | Rebuild or install the architecture-qualified wheel. |
| cuDNN DSA indexer | SM90 interface from the 1.25.0 stack | SM100 indexer/top-k backend | Do not satisfy an SM100 import with the SM90 module. |
| CUTLASS DSL | 4.5.2 SM90 package path precedes system packages | Use the version pinned by the Blackwell stack | Preserve package ordering, but not the compiled artifacts. |
| vLLM | 0.20.2 CUDA-13 wheel plus the Torch 2.12 ABI library | Build the selected vLLM revision against the container Torch | The Hopper ABI library is specific to Torch 2.12 and must not be carried to another Torch ABI. |
| DeepSeek V4 Python model and tokenizer routing | Shared vLLM capability | Shared vLLM capability | Registry support is portable; kernels and extensions are not. |
| TileLang / TVM FFI | Matched `0.1.9` pair | Keep a matched pair required by the selected vLLM | Never mix independently built TileLang and TVM FFI packages. |
| Kernel caches and compile targets | SM90 only | SM100 only | Use separate Triton, TorchInductor, and DeepGEMM cache roots. |
| Exact tensor-proxy performance | 619.4 ms BF16 loop; 19.1 ms pure-FP8 export | Not measured with the same proxy | Record `unmeasured`; never derive a cross-architecture ratio from different workloads. |

When switching architectures, retain the model/checkpoint contracts and rerun
both profiles. Rebuild every CUDA extension for the target container and GPU;
an import-only success is insufficient evidence for a fused DSA or rollout path.
