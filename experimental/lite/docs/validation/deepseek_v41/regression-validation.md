# Existing-suite regression validation

The acceptance condition is an empty candidate-minus-baseline failure/error
set against the supplied `dev7064` run, job 18342583. This condition permits
baseline failures; it is not an all-tests-passed claim. Parameterized cases
remain distinct. The comparator validates that every failure and collection,
setup or teardown error in the pytest summary was extracted exactly once.
Collection paths are compared by file name because the baseline reorganized
its test directories.

## Source and environment

The missing `megatron/` dependency tree was restored from ISEEKYAN/Megatron-LM
`dev` commit `917e8e940cd7e32b9a91fd265834f3850a23c484`. All 665 tracked files were compared by SHA-256 with the
supplied runtime baseline. The only content difference was the baseline's NVRx
fallback patch, which is reapplied by the existing staging harness. The two
additional baseline files were its NVRx backup and a built dataset extension.
The source license is preserved at repository root.

The existing `.clone-7064` and `precbase` staging directories were reused.
Each candidate was fetched from the pushed branch, fast-forwarded, synchronized
with `rsync -a --delete --exclude=.git`, and patched with the existing
`harness/ds4_patch_nvrx_fallback.py` before running `tests7064.sbatch` with
`ARM_SRC=precbase`. The script uses the existing PyTorch 26.04 container,
dependency overlays, one Slurm GPU and the full discovered `tests/unit` list.

## Fixes and evidence

- Restore the baseline per-expert `GroupedLinearLoRA` implementation. Its
  existing test first failed collection locally, then all four LoRA tests
  passed (exit 0).
- Defer optional VERL imports to fixture setup, following the baseline policy.
  Missing VERL cases are reported as skips, never as passes.
- Adapt DS4 CSA to the current Core compressor/layout and sparse-attention
  interfaces. The final three-value indexer unpack matches the supplied
  runtime baseline.
- Make single-process checkpoint tests independent of earlier distributed
  initialization; retain the separate rank-specific checkpoint test. The
  checkpoint CPU suite passed all nine cases (exit 0).
- Reuse the baseline CSA device/dtype setup and its current-interface test,
  preserving all four existing CSA cases and adding that interface case.
  The router-buffer test explicitly requests CPU output before comparison.
- The V4.1 CPU suite passed all 41 cases (exit 0).

| Run | Source | Failed | Passed | Skipped | Errors | New failures/errors |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Baseline 18342583 | supplied dev7064 | 8 | 792 | 19 | 2 | — |
| 18347086 | 405e0eb80 | 9 | 447 | 42 | 3 | 9 |
| 18347358 | 735e89e70 | 7 | 459 | 42 | 0 | 4 |
| 18347621 | bd1984827 | 4 | 463 | 42 | 0 | 1 |
| 18347896 | 9f72c3257 | 3 | 464 | 42 | 0 | 0 |

Each row also has one xfailed case. The failed intermediate runs are preserved
as raw logs and JSON comparisons in this directory; their comparator exit
codes are 1. The final comparator exit code is 0: its candidate-minus-baseline set is empty.
All Slurm test jobs, including the baseline and final run, exited `1:0`; see
`regression-slurm.txt`. Raw logs are losslessly compressed as `.txt.gz`; JSON
SHA-256 values identify the original uncompressed bytes.

## Coverage boundary

The supplied baseline contains 75 unit-test files; the candidate contains 61.
Their inventories predate this repair and include different model features,
optional integration cases and directory names. No tests were deleted in this
repair. Thus an empty failure difference establishes the requested acceptance
condition for these discovered suites; it does not establish execution of every
baseline-only test on the candidate. Skips and xfails remain separate from
passing cases.


## Final result and reproduction

Job **18347896**, source **9f72c3257**, satisfies the requested empty-difference
condition. The remaining three failures are all in the baseline set:

- `test_dynamic_shape_variable_len_recv_gloo`
- `test_pp_export_never_materializes_the_whole_stage`
- `test_pp_export_streams_over_nccl_and_matches_materialized`

The DS4 local-to-global loading, every-stage router-buffer export and MTP replay
root tests all ran without failures or skips. The fused CSA THD/BSHD equivalence
test also ran successfully on the Slurm GPU. There were zero collection errors.

Reproduce the comparison from repository root (expected exit code 0):

```bash
python experimental/lite/tools/deepseek_v41/compare_pytest_failures.py \
  experimental/lite/docs/validation/deepseek_v41/regression-baseline-18342583.txt.gz \
  experimental/lite/docs/validation/deepseek_v41/regression-18347896.txt.gz
```

Using any of the three earlier candidate logs returns exit code 1. A missing,
incomplete or ambiguously parsed summary raises an error instead of passing.
