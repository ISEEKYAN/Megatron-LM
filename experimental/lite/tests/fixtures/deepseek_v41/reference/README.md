# Pinned oracle metadata

The JSON metadata comes from DeepSeek-V4.1-Flash revision
`df42c109f1defefcbfcedbe7d905718a12266e40`. Official Python source is not
redistributed here. Set `DS41_REFERENCE_DIR` to an external directory containing
the pinned reference files listed in `tools/deepseek_v41/fixtures.py`.
The default development location is `/tmp/ds41-fixture-reference`.

Oracle and semantic tests validate SHA-256 before parsing or executing external
source. Missing or changed reference files fail the tests; they are not skipped.
