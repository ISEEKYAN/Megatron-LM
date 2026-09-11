# V4.1 oracle, fixture and quantized payload validation

The implementation uses the pinned release revision
`df42c109f1defefcbfcedbe7d905718a12266e40`. Source hashes are checked before
official execution. The implementation and its reviewer should distinguish the
following independently executed checks.

| Check | Observed result | Scope |
| --- | --- | --- |
| CPU unit suite | 40 passed, exit 0 | Storage corruption, exact code/scale cards, derivatives, separate switches, fixture shapes and wrapper contracts |
| Corrected CED validator | Eleven mutations rejected, exit 0 | Executable AST dataflow probes with labeled doubles |
| Owner validator | 40 layers, 96,085 index keys, 43 owner keys, exit 0 | Actual pinned release index and source |
| CSA2 semantic validator | Rotary max error 2.384185791015625e-7; grouped projection error 0, exit 0 | FP32 CPU semantics at the stated actual positions |
| Real MTP storage | 2,401 keys, 7,932,874,632 bytes, seven storage ranks, exit 0 | All published MTP payloads; dtype/shape/bytes/digests preserved |
| Real checkpoint decoding | Nine cases, decode and BF16 reload errors 0, exit 0 | FP4, FP8 and BF16 samples across all three MTP stages; CPU loaded GEMM |
| GPU codecs, job 7079615 | Eight FP4 cases, four FP8 activation cases, FP8 GEMM error 0, two derivatives | Original TileLang kernels versus separate native codecs and actual native FP8 GEMM; COMPLETED 0:0 |
| Fixture generation, job 7079838 | 3,204 release keys, 3,164 converted keys | Original constructor, all 40 layers, three CSA2 modes, both Engrams and two vision blocks; COMPLETED 0:0 |
| Fixture byte validation | 3,204 release, 3,164 converted, 19 inputs, exit 0 | File and tensor SHA-256, identity transforms and independently decoded wo_a transforms |
| Official oracle, job 7080167 | Four sequences, 178 tokens total, 5,682 captures | All 40 layers, two images, all 320 experts probed per sequence; original/instrumented and sequence isolation exact; COMPLETED 0:0 |
| Final oracle, job 7080297 | 7,089 captures with tensor digests; four sequences and 178 tokens | Source `d63018057`; shortest sequence uses 1+1+1 chunks; COMPLETED 0:0 |
| Empty-prefix capture audit | Layers 2–19 record KV `[1,0,64]` and indices `[1,1,0]`, both zero bytes | Actual exported captures, SHA-256 verified against the oracle summary |

Raw Slurm results, codec logs, oracle summaries and the empty-prefix audit are
checked in under `docs/validation/deepseek_v41/`. The final complete capture
manifest has SHA-256
`2143102742aef2f9dd894ca42fc64414f357f27ccb4aa4e876c05322625f4310`.

Job 7080142 failed because the wrapper scoped the default CUDA device only to
construction. The published forward also creates SWA indices using the default
device. The fix scopes the full original execution to that device and restores
the caller context. The failed job is not counted as validation.

## Reproduction

Run the CPU suite from the repository root:

```bash
PYTHONPATH=experimental/lite python -m pytest -c /dev/null --confcutdir=experimental/lite --rootdir=experimental/lite -q experimental/lite/tests/unit/deepseek_v41
python experimental/lite/tools/validate_deepseek_v41_ced.py --official-model /tmp/ds41-review/model.py
python experimental/lite/tools/validate_deepseek_v41_csa2.py --official-model /tmp/ds41-review/model.py
python experimental/lite/tools/validate_deepseek_v41_owner_consumer.py --model /tmp/ds41-review/model.py --config /tmp/ds41-review/config.json --index /tmp/v41_index.json
PYTHONPATH=experimental/lite python experimental/lite/tools/deepseek_v41/validate_fixture.py --fixture-dir /tmp/ds41-b3-fixture
```

The real storage command uses `validate_mtp_store.py --checkpoint
/tmp/ds41-c-payload-release --output <new-directory> --storage-ranks 7` with
`PYTHONPATH=experimental/lite`. `deepseek_v41_mtp_provenance.json` records the
three source shards and their release-verified SHA-256 values. The sampled decode
command uses `validate_checkpoint_decode.py --checkpoint
/tmp/ds41-c-payload-release --official-convert /tmp/ds41-review/convert.py
--output <new-directory>`; per-case results and the independent converter hash
are in `deepseek_v41_decode_evidence.json`.

GPU entry points are checked in under `tools/deepseek_v41/slurm/`. They require
Slurm, the established GB200 container and dependency overlay, immutable source
snapshots and the pinned official files. No CPU fallback or skipped GPU result
is accepted. Oracle execution writes a capture manifest with per-tensor shapes,
dtypes, byte lengths and digests, plus a summary that hashes that manifest.

## Boundaries

The fixture manifest is checked in at `tests/fixtures/deepseek_v41/manifest.json`;
large generated tensors and downloaded release shards are not source artifacts.
The reduced fixture retains ownership topology, but does not establish full-size
196B Engram memory behavior or full-model training parity. Runtime integration
of the native operators into the backbone belongs to the subsequent model work.

Engram FP8 scale regeneration after an update remains an explicit evidence gap
(O08); neither regeneration nor master-weight policy (O09) is implemented here.
The indexer training objective (O12) and differentiable training oracle are outside
this delivery. O14 supplies the tested quantized-operator derivative policy, and
O15 permits explicit disabled-quantization diagnostics. Published
`dspark_block_size=5` remains unchanged in archival config; only the separate
execution flag rejects DSpark execution. MTP storage is not a model factory.

## Existing-suite regression follow-up

The acceptance baseline is parent `a50244d7b`, job 18348391. The previous
`dev7064` acceptance claim is superseded by
[the corrected regression report](../validation/deepseek_v41/regression-validation.md).
Historical job 18347358 (source 735e89e70, 7 failed / 459 passed) has three new
CSA failures and 17 removed failure/error outcomes against that parent.
The report distinguishes subsequent repairs and fresh exact-source verification
from that earlier failed run, preserving raw logs and actual exit codes.

Fresh exact-source `15cba8e9b` job 18349573 reports 3 failed / 464 passed,
with zero new failures/errors against the correct parent (comparator exit 0;
Slurm exit 1:0). CSA-specific job 18349574 reports 5 passed, zero skipped,
Slurm exit 0:0. These results do not claim GPU validation of later D additions.
