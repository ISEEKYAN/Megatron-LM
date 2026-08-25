from __future__ import annotations

import pytest
import torch

from vllm_forward_parity import (
    CANDIDATE_ROLE,
    OFFICIAL_ROLE,
    ParityTrace,
    REQUIRED_FORWARD_OPS,
    SYNTHETIC_CASES,
    assert_trace_equal,
    cases_named,
    compare_trace_artifacts,
    input_digest,
    load_trace,
    require_forward_ops,
    save_trace,
)


def test_synthetic_matrix_covers_batch_ragged_and_boundaries() -> None:
    assert {case.batch_size for case in SYNTHETIC_CASES} >= {1, 4, 32}
    lengths = {
        length for case in SYNTHETIC_CASES for length in case.sequence_lengths
    }
    assert {2047, 2048, 2049, 6143, 6144, 6145} <= lengths
    assert any(len(set(case.sequence_lengths)) > 1 for case in SYNTHETIC_CASES)
    for case in SYNTHETIC_CASES:
        assert case.cu_seqlens[0] == 0
        assert case.cu_seqlens[-1] == case.token_count


def test_trace_accepts_only_ordered_bitwise_equal_tensors() -> None:
    reference = ParityTrace()
    candidate = ParityTrace()
    value = torch.tensor([[1.0, -0.0], [float("inf"), 3.0]])
    reference.add("moe.quant", q=value.to(torch.float8_e4m3fn))
    candidate.add("moe.quant", q=value.to(torch.float8_e4m3fn))
    assert_trace_equal(reference, candidate)


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        ("value", r"op=moe.output tensor=hidden.*first_mismatch=\(1, 0\)"),
        ("dtype", r"op=moe.output tensor=hidden.*dtype=torch.float32"),
        ("shape", r"op=moe.output tensor=hidden.*shape=\(4,\)"),
        ("order", r"trace order"),
    ],
)
def test_first_mismatch_report_is_actionable(mutation: str, expected: str) -> None:
    reference = ParityTrace()
    candidate = ParityTrace()
    reference.add("moe.output", hidden=torch.arange(4, dtype=torch.int32).view(2, 2))
    if mutation == "value":
        value = torch.tensor([[0, 1], [9, 3]], dtype=torch.int32)
        candidate.add("moe.output", hidden=value)
    elif mutation == "dtype":
        candidate.add("moe.output", hidden=torch.arange(4).view(2, 2).float())
    elif mutation == "shape":
        candidate.add("moe.output", hidden=torch.arange(4, dtype=torch.int32))
    else:
        candidate.add("moe.intermediate", hidden=torch.arange(4).view(2, 2))
    with pytest.raises(AssertionError, match=expected):
        assert_trace_equal(reference, candidate)


def test_dump_pair_requires_official_role_and_identical_input(tmp_path) -> None:
    inputs = {"input_ids": torch.tensor([[1, 2, 3]], dtype=torch.int64)}
    digest = input_digest(inputs)
    official = ParityTrace()
    candidate = ParityTrace()
    for op in sorted(REQUIRED_FORWARD_OPS):
        official.add(op, output=torch.ones(1, 4, dtype=torch.bfloat16))
        candidate.add(op, output=torch.ones(1, 4, dtype=torch.bfloat16))
    official_path = tmp_path / "official.pt"
    candidate_path = tmp_path / "candidate.pt"
    save_trace(
        official_path,
        official,
        role=OFFICIAL_ROLE,
        case="bs1-2k",
        inputs_sha256=digest,
    )
    save_trace(
        candidate_path,
        candidate,
        role=CANDIDATE_ROLE,
        case="bs1-2k",
        inputs_sha256=digest,
    )
    compare_trace_artifacts(official_path, candidate_path)

    with pytest.raises(ValueError, match="expected 'official-vllm'"):
        load_trace(candidate_path, expected_role=OFFICIAL_ROLE)

    save_trace(
        candidate_path,
        candidate,
        role=CANDIDATE_ROLE,
        case="bs1-2k",
        inputs_sha256="different-input",
    )
    with pytest.raises(AssertionError, match="inputs_sha256 differs"):
        compare_trace_artifacts(official_path, candidate_path)


def test_case_selection_rejects_unknown_names() -> None:
    assert cases_named(["bs1-2k"])[0].sequence_lengths == (2048,)
    with pytest.raises(KeyError, match="not-a-case"):
        cases_named(["not-a-case"])


def test_rollout_trace_cannot_omit_an_operator() -> None:
    incomplete = ParityTrace()
    incomplete.add("mhc.pre", output=torch.zeros(1))
    with pytest.raises(AssertionError, match=r"missing ops:.*indexer.attention"):
        require_forward_ops(incomplete)
