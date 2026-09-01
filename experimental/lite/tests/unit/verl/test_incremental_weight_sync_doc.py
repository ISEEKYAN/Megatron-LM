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
