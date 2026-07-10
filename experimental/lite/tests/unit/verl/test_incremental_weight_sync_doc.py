# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from pathlib import Path


DOC = Path(__file__).parents[3] / "docs" / "rl-incremental-weight-sync.md"


def test_two_slot_transport_is_documented_as_future_work() -> None:
    text = DOC.read_text()
    section = text.split("### Interaction with proposed two-slot transport", 1)[1]
    section = section.split("### BF16 to block-FP8 resync", 1)[0]
    normalized = " ".join(section.split()).lower()

    assert "is not implemented in the current tree" in normalized
    assert "compat.py" not in normalized
    assert "at most two in-flight slots" in normalized


def test_adjacent_step_fp8_evidence_keeps_algorithm_context() -> None:
    text = DOC.read_text()
    section = text.split("### Adjacent-step and block-FP8 follow-up", 1)[1]
    section = section.split("### Limitations", 1)[0]
    normalized = " ".join(section.split()).lower()

    assert "actual serialized block-fp8 target" in normalized
    assert "fp32 scales" in normalized
    assert "packed moe expert groups" in normalized
    assert "convolution weights remain bf16" in normalized
    assert "zero-advantage" in normalized
    assert "sft" in normalized
    assert "does not establish" in normalized


def test_layer_distribution_evidence_is_not_a_runtime_policy() -> None:
    text = DOC.read_text()
    section = text.split("#### Layer and depth distribution", 1)[1]
    section = section.split("### Limitations", 1)[0]
    normalized = " ".join(section.split()).lower()

    assert "shallow" in normalized
    assert "middle" in normalized
    assert "deep" in normalized
    assert "80%" in normalized
    assert "not a layer-selection policy" in normalized
