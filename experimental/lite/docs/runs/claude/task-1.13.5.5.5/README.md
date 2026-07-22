# FLA-enabled env for Qwen3.5 GatedDeltaNet packed-THD

Qwen3.5-35B-A3B uses a Gated DeltaNet linear-attention path. When the engine runs
with `impl_cfg.use_thd=True` (packed variable-length THD batches, the default of
`run_qwen3moe_sft.sh`), `GatedDeltaNet._causal_conv1d` takes the packed branch
(`cu_seqlens is not None`) and requires the FLA causal-conv kernel:

```
experimental/lite/megatron/lite/primitive/modules/gated_delta_net.py
  from fla.modules.convolution import causal_conv1d
  from fla.ops.gated_delta_rule import chunk_gated_delta_rule
experimental/lite/megatron/lite/primitive/ops/gated_delta_rule.py
  from fla.modules.l2norm import l2norm
```

If `fla` is not importable the packed branch raises:

```
NotImplementedError: GatedDeltaNet packed THD requires FLA causal conv.
```

The stock `verl.vllm023.sqsh` container does **not** ship `fla`. Neither does the
`mlite-2604-verl-dsa-sm90-overlay` (verified: no `fla` in its site-packages).

## The FLA-enabled overlay (reuse — already built & proven)

Do **not** build a fresh overlay: a container-matched FLA + TileLang overlay
already exists on lustre and has run full-model Qwen3.5 GatedDeltaNet THD
end-to-end. Reuse it (per the upstream-freshness rule: reuse a vk-verified
artifact before self-building).

```
IMG     = $BASE/verl_optimize/verl.vllm023.sqsh
CP_SITE = $BASE/mlite-newenv-cache/qwen35-cp-overlay-20260613/site
  where BASE = /lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/bayan/code
```

`CP_SITE` contains (cpython-3.12, matching the container's Python 3.12.3):

- `flash_linear_attention` / `fla_core` 0.5.0  (the `fla` package)
- `tilelang` 0.1.10  (GDN chunk backward backend)
- `tvm_ffi` + `apache_tvm_ffi` 0.1.11  (needs `LD_LIBRARY_PATH=$CP_SITE/tvm_ffi/lib`)

### Activation — two lines in the in-container node script

```sh
export PYTHONPATH="/vllm:$CP_SITE:<mlite paths>:$VERL:$MEGATRON_ROOT:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$CP_SITE/tvm_ffi/lib:${LD_LIBRARY_PATH:-}"
```

`CP_SITE` is prepended just after `/vllm` so the container's own `vllm` still
wins, but `fla`/`tilelang`/`tvm_ffi` become importable. This is exactly what the
known-good DAPO node script does (`qwen35_dapo_mfsdp_62295f9b3/
run_dapo_h100_node_c17a05eff.sh`, lines 21-27: `CP_SITE` on `PYTHONPATH` +
`LD_LIBRARY_PATH="$CP_SITE/tvm_ffi/lib"`).

## Provenance / known-good evidence

- `qwen35_dapo_mfsdp_62295f9b3/run_dapo_h100_node_c17a05eff.sh` — same container
  `verl.vllm023.sqsh`, same `CP_SITE`, preflight `import fla; import tilelang`
  (`G0B_PREFLIGHT_OK ... fla.__version__ tilelang.__version__`) and ran full
  Qwen3.5 GDN THD DAPO end-to-end (this is the config `three_arm_precision_sft.sbatch`
  cites as its env source). Verified present on CW lustre 2026-07-22; the
  `CP_SITE` wiring is lines 12/21-27 of that script.
- K-0123 — `qwen35-cp-overlay-20260613/site` is the FLA GDN CP/THD overlay;
  without it FLA GDN falls back to an unavailable backend / "Please install
  tilelang". Slurm `13062707` COMPLETED with the overlay loaded.
- CPU import gate this task (job `14240016`, `cpu_short`, in `verl.vllm023.sqsh`
  with `CP_SITE`): `PY 3.12.3`, `FLA_VER 0.5.0`, `TILELANG_VER 0.1.10`, and
  `FLA_GDN_IMPORTS_OK causal_conv1d=True chunk_gdr=True l2norm=True`.
  (The "Triton not supported on current platform, roll back to CPU" warning is
  expected on a CPU node — the triton kernels JIT on GPU at runtime.)

## Why the three-arm precision harness (TASK-1.13.5.5.3) hit the block

`three_arm_precision_node.sh` reused the DAPO env recipe but **dropped `$CP_SITE`
from `PYTHONPATH`** (and the `LD_LIBRARY_PATH` for `tvm_ffi`). Its `PYTHONPATH`
is `/vllm:$MLITE_LITE/examples/verl:$MLITE_LITE:$VERL:$MEGATRON_ROOT` — no FLA —
so every arm (not just muon) fails Qwen3.5 GDN model forward with the
`NotImplementedError` above. The fix is the two lines above; see
`fla_thd_smoke_node.sh` in this directory for the corrected node script.

## GPU verification harness (in this directory)

- `submit_fla_thd_smoke.sh` — orchestrator: stage the mlite checkout under test,
  run the CPU import gate, then submit the GPU smoke. Marks the GPU step behind
  the pre-GPU moe gate; does not self-fire.
- `fla_thd_smoke.sbatch` — 1 node × 8×H100, container `verl.vllm023.sqsh`.
- `fla_thd_smoke_node.sh` — in-container driver: DAPO env recipe **with** the
  `CP_SITE` wiring, `adamw` / `dist_opt` (simplest arm — no muon /
  emerging_optimizers), `LOAD_HF_WEIGHTS=True`, 2 steps.

Pass criterion: the run gets past the `NotImplementedError` and emits a real
per-step SFT loss for Qwen3.5-35B-A3B GatedDeltaNet with `use_thd=True`
(a finite `loss=` line / JSONL entry). That is the "GDN THD forward produces
loss" bar; the adamw arm isolates the FLA fix from the separate muon issues.
