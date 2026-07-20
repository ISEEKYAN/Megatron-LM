# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path

import pytest


_SCRIPT = (
    Path(__file__).parents[3]
    / "docs"
    / "runs"
    / "glm5_dsa_cp_native"
    / "validate_dsa_cp.py"
)
_SPEC = importlib.util.spec_from_file_location("_glm5_dsa_cp_validation", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_VALIDATION = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_VALIDATION)


def _rank_result(*, cp1_memory: float = 100.0, cp2_memory: float = 60.0, hf_cosine: float = 0.99995):
    return {
        "hf_reference": {"logits_cosine": hf_cosine},
        "thd": {"output_cosine": 1.0, "input_grad_cosine": 1.0},
        "dense": {
            "512": {
                "cp1": {"activation_peak_mb": cp1_memory},
                "native": {
                    "activation_peak_mb": cp2_memory,
                    "output_cosine": 1.0,
                    "input_grad_cosine": 1.0,
                },
            }
        },
    }


def test_validation_accepts_independent_hf_parity_and_cp_memory_reduction():
    _VALIDATION._assert_validation([_rank_result()])


def test_hf_parity_uses_glm52_dsa_reference_and_indexer_config():
    source = inspect.getsource(_VALIDATION._hf_reference_logits)

    assert "GlmMoeDsaConfig" in source
    assert "GlmMoeDsaForCausalLM" in source
    assert "DeepseekV3" not in source
    for field in (
        "index_topk",
        "index_head_dim",
        "index_n_heads",
        "index_topk_freq",
        "index_skip_topk_offset",
        "indexer_types",
    ):
        assert f"{field}=" in source
    assert 'indexer_types=["full", "full", "full", "shared"]' in source


@pytest.mark.parametrize("cp2_memory", [100.0, 101.0])
def test_validation_rejects_cp_memory_that_does_not_shrink(cp2_memory):
    with pytest.raises(AssertionError, match="CP-native activation peak did not shrink"):
        _VALIDATION._assert_validation([_rank_result(cp2_memory=cp2_memory)])


def test_validation_rejects_hf_logits_cosine_below_contract():
    with pytest.raises(AssertionError, match="HF logits cosine below 0.9999"):
        _VALIDATION._assert_validation([_rank_result(hf_cosine=0.9998)])
