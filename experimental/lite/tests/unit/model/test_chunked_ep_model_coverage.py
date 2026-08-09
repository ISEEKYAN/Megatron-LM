# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

from pathlib import Path


# Static scope guards only: public bench + Slurm GPU validation own end-to-end
# DeepEP evidence.


LITE_ROOT = Path(__file__).resolve().parents[3]
MODEL_ROOT = LITE_ROOT / "megatron/lite/model"
SKILL_ROOT = LITE_ROOT / "skills"


def test_only_qwen3_moe_composes_the_chunked_ep_primitive():
    qwen_protocol = (MODEL_ROOT / "qwen3_moe/lite/protocol.py").read_text()
    qwen_model = (MODEL_ROOT / "qwen3_moe/lite/model.py").read_text()

    assert "enable_ep_chunk_overlap: bool = False" in qwen_protocol
    assert "validate_ep_chunk_overlap_config(" in qwen_protocol
    assert "EPChunkForwardOp(" in qwen_model
    assert "EPChunkBackwardOp(" in qwen_model
    assert "EPChunkFusedForwardBackwardOp(" in qwen_model
    assert "get_ep_chunk_workspace(" in qwen_model
    assert "max_input_rows=ep_chunk_max_token_rows_per_rank" in qwen_model
    assert "materialize_ep_chunk_workspaces" in qwen_model
    assert "workspace.materialize(device=device)" in qwen_model
    assert "workspace.warmup(device=" not in qwen_model
    assert "ep_chunk_full_recompute" in qwen_model
    assert (
        "if torch.is_grad_enabled():\n                return self.ep_chunk_fused"
        not in qwen_model
    )

    for model_name, implementation in (
        ("qwen3_5", "model.py"),
        ("kimi_k2", "model.py"),
        ("glm5", "model.py"),
        ("deepseek_v4", "moe.py"),
    ):
        protocol = (MODEL_ROOT / model_name / "lite/protocol.py").read_text()
        model = (MODEL_ROOT / model_name / "lite" / implementation).read_text()
        assert "enable_ep_chunk_overlap" not in protocol
        assert "EPChunkForwardOp" not in model
        assert "EPChunkBackwardOp" not in model
        assert "EPChunkFusedForwardBackwardOp" not in model


def test_qwen3_layer_is_only_a_lightweight_three_op_composition():
    model = (MODEL_ROOT / "qwen3_moe/lite/model.py").read_text()

    assert "_full_recompute_fused_backward" not in model
    assert "submit_deepep_dispatch" not in model
    assert "MEGATRON_LITE_EP_CHUNK" not in model
    assert "layer_idx %" not in model
    assert "buffer_slot=" not in model


def test_dynamic_chunk_configuration_is_absent_from_product_code():
    product = LITE_ROOT / "megatron/lite"
    forbidden = (
        "ep_chunk_num_chunks",
        "ep_chunk_bwd_num_chunks",
        "MEGATRON_LITE_EP_CHUNK_WEIGHTS",
    )

    matches = {
        token: [path for path in product.rglob("*.py") if token in path.read_text()]
        for token in forbidden
    }

    assert matches == {token: [] for token in forbidden}


def test_chunk_policy_lives_with_the_moe_module_primitive():
    policy = (
        LITE_ROOT / "megatron/lite/primitive/modules/moe_ep_chunk_overlap_policy.py"
    )
    old_policy = LITE_ROOT / "megatron/lite/primitive/moe_ep_chunk_overlap_policy.py"

    assert policy.is_file()
    assert not old_policy.exists()


def test_moe_and_bench_skills_publish_chunked_ep_usage_contract():
    moe_skill = (SKILL_ROOT / "primitive/module/moe.md").read_text()
    bench_skill = (SKILL_ROOT / "application/bench.md").read_text()
    protocol = (MODEL_ROOT / "qwen3_moe/lite/protocol.py").read_text()

    for key in (
        "enable_ep_chunk_overlap",
        "ep_chunk_max_token_rows_per_rank",
        "ep_chunk_full_recompute",
    ):
        assert key in moe_skill
        assert key in protocol
    assert "unsupported_combinations" in moe_skill
    assert "top_k>expert_parallel_size" in moe_skill
    assert "experimental/lite/examples/bench/bench.py" in bench_skill
    assert "--impl-cfg-json" in bench_skill
