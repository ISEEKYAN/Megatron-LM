# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from pathlib import Path

import pytest

LITE_ROOT = Path(__file__).resolve().parents[3]
MODEL_ROOT = LITE_ROOT / "megatron/lite/model"
SUPPORTED_MODELS = {"qwen3_moe", "qwen3_5", "kimi_k2", "glm5", "deepseek_v4"}


@pytest.mark.parametrize(
    "model_name, implementation_file",
    [
        ("qwen3_moe", "model.py"),
        ("qwen3_5", "model.py"),
        ("kimi_k2", "model.py"),
        ("glm5", "model.py"),
        ("deepseek_v4", "moe.py"),
    ],
)
def test_every_lite_moe_model_consumes_the_shared_chunked_ep_primitive(
    model_name, implementation_file
):
    protocol = (MODEL_ROOT / model_name / "lite/protocol.py").read_text()
    implementation = (
        MODEL_ROOT / model_name / "lite" / implementation_file
    ).read_text()

    assert "num_chunks_ep_a2a_overlap: int = 1" in protocol
    assert "validate_ep_chunk_overlap_config(" in protocol
    assert "recompute_modules_for_ep_chunk_overlap(" in protocol
    assert "EPChunkOverlapOperator" in implementation
    assert "self.dispatcher = TokenDispatcher(" in implementation
    assert "self.ep_chunk_overlap = None" in implementation
    assert "if num_chunks_ep_a2a_overlap > 1:" in implementation
    assert "EPChunkOverlapOperator(" in implementation
    assert "if self.ep_chunk_overlap is not None:" in implementation
    assert "self.dispatcher.dispatch(" in implementation
    assert "dispatcher_factory" not in implementation
    assert "class MoELayer(EPChunkOverlap" not in implementation
    assert "class DeepseekV4MoE(EPChunkOverlap" not in implementation
    assert "num_chunks_ep_a2a_overlap" in implementation
    assert 'buffer_slot=("ep_chunk_overlap", "forward", idx)' in implementation
    assert "layer_slot" not in implementation


def test_backward_specific_chunk_parameter_is_removed_from_product_code():
    product = LITE_ROOT / "megatron/lite"

    matches = [
        path
        for path in product.rglob("*.py")
        if "ep_chunk_bwd_num_chunks" in path.read_text()
    ]

    assert matches == []


def test_coverage_list_matches_every_registered_lite_model():
    registered = {
        path.parents[1].name for path in MODEL_ROOT.glob("*/lite/protocol.py")
    }

    assert registered == SUPPORTED_MODELS


def test_chunk_policy_lives_with_the_moe_module_primitive():
    policy = (
        LITE_ROOT / "megatron/lite/primitive/modules/moe_ep_chunk_overlap_policy.py"
    )
    old_policy = LITE_ROOT / "megatron/lite/primitive/moe_ep_chunk_overlap_policy.py"

    assert policy.is_file()
    assert not old_policy.exists()


def test_chunked_ep_operator_does_not_construct_or_factory_dispatchers():
    implementation = (
        LITE_ROOT / "megatron/lite/primitive/modules/moe_ep_chunk_overlap.py"
    ).read_text()

    assert "dispatcher_factory" not in implementation
    assert "_new_dispatcher" not in implementation
    assert "TokenDispatcher(" not in implementation
    assert "config: Any" not in implementation


@pytest.mark.parametrize("model_name", sorted(SUPPORTED_MODELS))
def test_chunked_ep_reuses_moe_norm_output_storage(model_name):
    implementation = (MODEL_ROOT / model_name / "lite/model.py").read_text()

    assert "discard_and_recompute_output_storage" in implementation


def test_deepseek_hash_router_keeps_checkpoint_gate_name():
    implementation = (MODEL_ROOT / "deepseek_v4/lite/moe.py").read_text()

    assert "self.gate = SigmoidTopKRouter(" in implementation
    assert 'self._modules["gate"] = self._modules.pop("router")' not in implementation
