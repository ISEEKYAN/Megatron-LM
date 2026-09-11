# Existing-suite regression validation

## Corrected baseline and historical result

The acceptance baseline is merge-base `a50244d7b`, staged as `cparent`, job
**18348391**: **16 failed, 294 passed, 21 skipped, 5 errors**. The previously
supplied `dev7064` (job 18342583) is from a different lineage and is not the
acceptance baseline. The earlier commit subject “Record the empty dev-baseline
regression difference and original test evidence” and its acceptance claim are
superseded by this report. Original logs and dev comparisons remain historical
artifacts, not acceptance evidence.

The **7 failed / 459 passed** log is job **18347358**, source **735e89e70**.
Against the correct parent it has **3 new failures** and **17 removed failures
or errors**, not an empty difference. Commit `15cba8e9b` records later evidence
and includes the intervening CSA repairs; its source must not be conflated with
that earlier run.

| Job | Source | Failed | Passed | Skipped | Errors | New vs parent | Removed vs parent |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 18348391 | a50244d7b (parent) | 16 | 294 | 21 | 5 | — | — |
| 18347086 | 405e0eb80 | 9 | 447 | 42 | 3 | 0 | 9 |
| 18347358 | 735e89e70 | 7 | 459 | 42 | 0 | 3 | 17 |
| 18347621 | bd1984827 | 4 | 463 | 42 | 0 | 1 | 18 |
| 18347896 | 9f72c3257 | 3 | 464 | 42 | 0 | 0 | 18 |

The historical candidate rows each also report one xfail. Counts above are
recomputed from the preserved raw logs with the correct parent, using
`regression-parent-vs-<job>.json`. The parent log is preserved as
`regression-parent-18348391.txt.gz`; its uncompressed SHA-256 is
`797892069f8d0500dfe96843f09c3066e3fa36899e979b84dff8eba4b3ce45c4`.

## CSA causal diagnosis and repair

The parent skipped all four original CSA tests because Core was unavailable.
Restoring Core and adapting its imports exposed these test/runtime incompatibilities:

- `test_forward_thd_packed_builds_layout_and_is_differentiable` and
  `test_project_boundary_kv_shape` sent CPU tensors into real Transformer Engine
  RMSNorm, failing with `CUDAGuardImpl initialized with non-CUDA DeviceType: cpu`.
  The projection bodies were unchanged. Commit `bd1984827` puts the modules and
  inputs on CUDA and retains their layout, shape and differentiability assertions.
- `test_cp1_thd_equals_bshd_fused` used FP32 indexer weights with a BF16-only
  fused kernel (`w must be bfloat16, got torch.float32`), unsupported tiny fused
  dimensions and a mock process group. Commit `bd1984827` uses BF16, supported
  kernel dimensions and a real single-rank NCCL group, preserving the THD/BSHD
  numerical equivalence check. Tiny dimensions remain in the projection tests.
- Once that path executed, the THD call unpacked two indexer outputs while the
  current Core returns three. Commit `9f72c3257` consumes the third return value.
  The additional interface test checks the current compressor/layout contract;
  it is separate from actual fused-kernel numerical execution.

These existing repairs are included in `15cba8e9b` and have been merged into the
current worktree. No CPU substitute for the CUDA kernel was introduced.

## Fresh reproduction

Fresh full-suite job **18349573** and CSA-specific job **18349574** use exact
published source `15cba8e9bb0a0f41bfdc23cc466d1445d3e7a8ae`. They reuse
`.clone-7064`, `precbase`, `harness/ds4_patch_nvrx_fallback.py` and
`tests7064.sbatch` in the existing runtime. Staging follows GitHub fetch,
checkout, `rsync -a --delete --exclude=.git`, then the NVRx harness patch.
Both jobs request one Slurm GPU using the existing PyTorch 26.04 container and
dependency overlays. The full run uses discovered `tests/unit`; the focused run
sets `MLITE_TEST_SELECTION=tests/unit/primitive/test_csa_thd_cp.py`.

CSA-specific job **18349574** completed with **5 passed, zero skipped** in
20.15 seconds, Slurm exit **0:0**. This includes all three reported failures.
The runtime warned that compact indexer forward plus Top-K was unavailable;
it used dense indexer forward plus standalone Top-K. This result therefore
covers that deployed path, not compact-indexer execution.
Full-suite job **18349573** completed with **3 failed, 464 passed, 42 skipped,
1 xfailed, zero errors** in 116.03 seconds. Slurm exited **1:0**, because the
three parent failures remain. The correct-parent comparator exited **0** with
**zero new failures/errors** and **18 removed failure/error outcomes**.
Remaining failures are:

- `test_dynamic_shape_variable_len_recv_gloo`
- `test_pp_export_never_materializes_the_whole_stage`
- `test_pp_export_streams_over_nccl_and_matches_materialized`

Raw evidence is in `regression-18349573.txt.gz`,
`regression-csa-18349574.txt.gz` and `regression-correct-parent-slurm.txt`.
`regression-parent-vs-18349573.json` records the comparison and raw-log hashes.
Reproduce the final comparison from repository root (expected exit **0**):

```bash
python experimental/lite/tools/deepseek_v41/compare_pytest_failures.py \
  experimental/lite/docs/validation/deepseek_v41/regression-parent-18348391.txt.gz \
  experimental/lite/docs/validation/deepseek_v41/regression-18349573.txt.gz
```

Using `regression-18347358.txt.gz` as the candidate instead exits **1**, with
exactly the three historical CSA regressions. Thus the corrected report preserves
the failing result as well as the repaired result.

After merging the published CSA fixes, the current D worktree CPU suite ran:

```bash
PYTHONPATH=experimental/lite OMP_NUM_THREADS=1 python -m pytest -c /dev/null \
  --confcutdir=experimental/lite --rootdir=experimental/lite -q \
  experimental/lite/tests/unit/deepseek_v41
```

Result: **85 passed in 4.31s**, exit **0**. The GPU jobs above validate exact
published `15cba8e9b`; they do not establish GPU coverage of later D additions.

## Coverage boundary

An empty candidate-minus-parent failure/error set permits pre-existing failures;
it is not an all-tests-passed claim. Parent dependency errors and skipped tests
mean test inventories and executed coverage differ. A removed collection error
is counted as one removed outcome, not as a passing numerical test. Skips and
xfails remain separate from passes. Raw logs are compressed losslessly and JSON
SHA-256 values identify their original bytes. Historical dev comparisons are
retained only to make the incorrect earlier baseline auditable.
