# Pinned oracle test inputs

These unmodified reference files come from DeepSeek-V4.1-Flash revision
`df42c109f1defefcbfcedbe7d905718a12266e40`, audited in
`docs/specs/deepseek_v41_oracle.md`. The unit test checks their SHA-256 against
`tools/deepseek_v41/fixtures.py` before using the official method and defaults.
They are parsed as source by CPU tests; importing the full CUDA model is not
required. Keeping them here makes the tests independent of `/tmp` preparation.
