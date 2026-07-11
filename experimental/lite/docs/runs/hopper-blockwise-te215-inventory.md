# Canonical TE 2.15 blockwise capability inventory

This read-only inventory answers one question before any overlay is considered:
does the canonical training image's native Transformer Engine already run every
mandatory Hopper blockwise primitive? Blockwise FP8 must run on the same image
that runs BF16, so the parity baseline may only add a bespoke Transformer Engine
build if the canonical image genuinely lacks a kernel.

## Inputs

- canonical image:
  `/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/bayan/code/env/pytorch_26.04-py3.sqsh`
- launcher: `hopper-blockwise-te215-inventory.sbatch`
- probe: constructs the frozen `Float8BlockScaling` recipe (E4M3, x/w/grad block
  dims `1/2/1`, split accumulator, `fp8_dpa=False`, `fp8_mha=False`) and runs a
  forward + backward for each mandatory primitive under `fp8_autocast`.

The probe is intentionally not a profile validation. It records whether each op
can run and where it fails, nothing more.

## Result

Slurm job `13755858` completed every step with exit code `0:0` (elapsed 41s) and
emitted `TE215_BLOCKWISE_INVENTORY_COMPLETE status=all-ran`. The JSON payload:

```json
{
  "all_mandatory_ops_ran": true,
  "block_scaling_support": [true, ""],
  "capability": [9, 0],
  "cublaslt": 130401,
  "cuda": "13.2",
  "operations": {
    "grouped_linear": {"output": [32, 8192], "status": "ran"},
    "layernorm_linear": {"output": [256, 2, 4096], "status": "ran"},
    "linear": {"output": [256, 2, 4096], "status": "ran"}
  },
  "te_path": "/usr/local/lib/python3.12/dist-packages/transformer_engine/__init__.py",
  "te_version": "2.15.0+42b84005",
  "torch": "2.12.0a0+5aff3928d8.nv26.05"
}
```

Immutable stdout:

```text
/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/bayan/work/codex/fp8-primitives-c9c00dfac-cw/te215-blockwise-inventory-13755858.out
```

## Conclusion

The canonical image ships released Transformer Engine `2.15.0+42b84005` on SM90
with CUDA 13.2 and cuBLAS 130401. `check_fp8_block_scaling_support()` succeeds,
and all three mandatory blockwise primitives -- `Linear`, `LayerNormLinear`, and
`GroupedLinear` -- run forward and backward under `Float8BlockScaling` with
finite gradients. No mandatory kernel is missing.

Therefore the Hopper blockwise parity baseline is anchored to the canonical
image's native TE 2.15, and the earlier TE 2.18-dev build overlay is superseded
(see `hopper-blockwise-te-overlay.md`). The runtime gate pins the released
version `2.15.0` and keeps the capability preflight (SM90, block-scaling
support, cuBLAS >= 13.4, CUDA >= 12.9) as the fail-loud safety net; the build tag
is recorded for provenance only, because requiring an exact build SHA would
recreate a special-environment requirement.
