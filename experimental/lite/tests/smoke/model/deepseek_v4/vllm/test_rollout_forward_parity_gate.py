"""Container gates over paired official-vLLM and mLite rollout traces.

These tests consume forward-only dumps.  They never launch an RL driver.  A
producer should record the same input capsule on both sides with
``vllm_forward_parity.save_trace`` and include every required operator output.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from vllm_forward_parity import compare_trace_artifacts


def _compare_from_environment(prefix: str) -> None:
    official = os.getenv(f"VLLM_FORWARD_PARITY_{prefix}_OFFICIAL")
    candidate = os.getenv(f"VLLM_FORWARD_PARITY_{prefix}_CANDIDATE")
    if not official or not candidate:
        pytest.skip(
            f"set VLLM_FORWARD_PARITY_{prefix}_OFFICIAL and "
            f"VLLM_FORWARD_PARITY_{prefix}_CANDIDATE to paired forward dumps"
        )
    for path in (official, candidate):
        if not Path(path).is_file():
            pytest.fail(f"parity dump does not exist: {path}")
    compare_trace_artifacts(official, candidate)


@pytest.mark.optional
@pytest.mark.smoke
@pytest.mark.vllm_parity_ep
@pytest.mark.gpus(8, min_architecture="blackwell")
def test_ep8_official_rollout_dump_matches_mlite_forward_bitwise() -> None:
    _compare_from_environment("EP")


@pytest.mark.optional
@pytest.mark.smoke
@pytest.mark.vllm_parity_cp
@pytest.mark.gpus(2, min_architecture="blackwell")
def test_cp2_official_rollout_dump_matches_mlite_forward_bitwise() -> None:
    _compare_from_environment("CP")


@pytest.mark.optional
@pytest.mark.smoke
@pytest.mark.vllm_parity_graph
@pytest.mark.gpus(1, min_architecture="blackwell")
def test_cuda_graph_official_rollout_dump_matches_mlite_forward_bitwise() -> None:
    _compare_from_environment("GRAPH")
