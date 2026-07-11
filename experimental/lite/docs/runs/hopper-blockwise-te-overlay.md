# Hopper blockwise Transformer Engine overlay (SUPERSEDED)

> **Superseded.** This overlay is no longer used. A read-only capability
> inventory (`hopper-blockwise-te215-inventory`) showed the canonical training
> image's native Transformer Engine 2.15 already runs every mandatory blockwise
> primitive forward and backward under `Float8BlockScaling`. Blockwise FP8 now
> runs on the same image as BF16 with no FP8-only overlay, and the primitive
> parity re-anchored to native TE 2.15 (job `13756286`) reproduces the overlay's
> manifest byte-for-byte. This record is retained only for build-recipe
> provenance (CUDA 13.2 TE-from-source on `pytorch_26.04-py3.sqsh`); nothing in
> the shipped profiles or the parity harness depends on it.

This run record builds a pinned Transformer Engine source that an earlier
iteration used for the Hopper blockwise precision profiles. The build is
intentionally separate from GPU validation: compilation runs on a CPU Slurm
node, while import and kernel validation run on an H100 Slurm node.

## Frozen inputs

- Transformer Engine: `8b9968255eb879e6e390f427836906b29aad64d2`
  (`2.18.0.dev0`)
- base image:
  `/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/bayan/code/env/pytorch_26.04-py3.sqsh`
- CUDA toolkit selected inside the image: `/usr/local/cuda-13.2`
- target architecture: `SM90`
- Megatron reference: `cf2f07d7b1315c96c05554c670c43207c6783e5e`
- reference source archive SHA-256:
  `c08272b18d171553f2dcd04937d27e77d3a6be223860726be7c56fcc90c558b1`
- output overlay:
  `/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/bayan/code/te-8b9968255-cu132-py312-overlay-r7`
- build launcher: `hopper-blockwise-te-overlay-build.sbatch`
- GPU probe launcher: `hopper-blockwise-te-overlay-probe.sbatch`

The optional TE NCCL-EP transport extension is disabled with
`NVTE_WITH_NCCL_EP=0`. The frozen source documents this as the supported way to
skip NCCL-EP while building the rest of TE. The Hopper blockwise profiles use
TE Linear, LayerNormLinear, and GroupedLinear; they do not use NCCL-EP.

## Build attempts

All attempts ran in the CW `cpu` partition. Failed directories were preserved;
no failed overlay was reused as validation input.

| Job | Slurm result | Finding |
| --- | --- | --- |
| `13711582` | `FAILED` | A short commit ID is not accepted as a remote Git ref; compilation did not start. |
| `13711645` | `FAILED` | Inherited Miniforge GCC 14 headers mixed with the image system headers during the optional NCCL-EP build. |
| `13711870` | `FAILED` | System GCC 13 was selected correctly; optional NCCL-EP was incompatible with the image's NCCL device headers. |
| `13711996` | `FAILED` | NCCL-EP was skipped correctly, but CMake selected a Lustre Miniforge CUDA 12.8 toolkit and therefore missed the image's CUDA 13.2 target headers. |
| `13712580` | `FAILED` | CUDA 13.2 compiled 92 of 93 TE common objects; the NCCL-EP-off CMake branch did not add the existing `/usr/include/nccl.h` include root. |
| `13712832` | `FAILED` | Adding raw `/usr/include` to CUDA flags made NVCC resolve host C library headers before its CUDA sysroot during compiler detection. |
| `13712941` | `FAILED` | A private `nccl.h` include avoided raw `/usr/include`, but NVCC still inherited a hidden Lustre Miniforge sysroot not present in the emitted CMake compile command. |
| `13713346` | `COMPLETED 0:0` | Built and imported `2.18.0.dev0+8b99682` from the r7 overlay at the full frozen source SHA. |

A read-only header probe, job `13712183`, completed with exit code `0:0` and
located the required NVTX and NVML headers under
`/usr/local/cuda-13.2/targets/x86_64-linux/include`, plus `nccl.h` under
`/usr/include`. The final launcher consequently clears inherited CMake/CUDA
prefixes and pins both `CUDACXX` and `CUDAToolkit_ROOT` to CUDA 13.2.
The accepted launcher exposes only `nccl.h` through a private include directory;
it does not add raw `/usr/include` to host or CUDA compilation flags.
The read-only `hopper-blockwise-te-toolchain-probe.sbatch` launcher records
inherited compiler variables and NVCC dry-run commands before another build is
submitted.

Toolchain probe job `13713113` completed with exit code `0:0`. It showed that
the container inherited `NVCC_PREPEND_FLAGS` with a Miniforge `-ccbin`, while
the same CUDA source compiled successfully with `/usr/bin/g++`. The build
launcher therefore clears inherited NVCC, Conda toolchain, GCC wrapper, and
`CMAKE_ARGS` variables before selecting the system compilers.

The immutable stdout logs are in:

```text
/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/bayan/work/codex/fp8-primitives-c9c00dfac-cw/te-overlay-build-<job>.out
```

## Accepted build and GPU probe

Build job `13713346` satisfied the build gate: every Slurm step completed with
exit code `0:0`, the source SHA was
`8b9968255eb879e6e390f427836906b29aad64d2`, and the imported package was
`2.18.0.dev0+8b99682` under the r7 overlay. The built wheel SHA-256 was
`edd66a6e9912800e9ec4e0d65c458603fa6f79838804abdee3d91718fc69b80f`.

The GPU probe is accepted only with SM90, CUDA at least 12.9, cuBLASLt at least
13.4, block-scaling support, and non-skipped Linear and GroupedLinear
forward/backward markers.

GPU probe job `13713715` satisfied that gate with all Slurm steps
`COMPLETED 0:0`. It reported compute capability `(9, 0)`, CUDA `13.2`,
cuBLASLt `130401`, and block-scaling support. The non-skipped outputs were
Linear `(256, 2, 4096)` with dX `(256, 2, 1024)` and dW `(4096, 1024)`, plus
GroupedLinear `(32, 8192)` with dX `(32, 1024)` and two parameter gradients.
The terminal marker was `TE_BLOCKWISE_OVERLAY_GPU_PROBE_OK`.
