# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.mlite


def test_qwen3_moe_removes_the_legacy_model_threaded_fp8_path():
    lite_root = Path(__file__).resolve().parents[3]
    model_source = (
        lite_root / "megatron" / "lite" / "model" / "qwen3_moe" / "lite" / "model.py"
    ).read_text(encoding="utf-8")
    protocol_source = (
        lite_root / "megatron" / "lite" / "model" / "qwen3_moe" / "lite" / "protocol.py"
    ).read_text(encoding="utf-8")

    assert "build_fp8_recipe" not in model_source
    assert "fp8_autocast" not in model_source
    assert "fp8: bool" not in model_source
    assert "fp8=False" not in protocol_source


def test_qwen3_moe_impl_config_accepts_only_runtime_owned_precision_injection(
    transformer_engine_import_stub,
):
    transformer_engine_import_stub()
    from megatron.lite.model.qwen3_moe.lite.protocol import ImplConfig

    declared = {field.name: field.default for field in fields(ImplConfig)}
    assert declared["precision_coverage"] is None
    assert declared["precision_implementation"] is None
    assert declared["precision_parameter_contract"] is None


def test_qwen3_moe_rejects_partial_or_unvalidated_precision_composition(
    transformer_engine_import_stub,
):
    transformer_engine_import_stub()
    from megatron.lite.model.qwen3_moe.lite import protocol
    from megatron.lite.primitive.precision import PrecisionCoverage, resolve_precision

    with pytest.raises(ValueError, match="all three typed fields"):
        protocol.build_model(
            SimpleNamespace(),
            impl_cfg=protocol.ImplConfig(precision_coverage=object()),
        )

    implementation = resolve_precision("hopper_blockwise_bf16_weight")
    assert implementation is not None
    with pytest.raises(ValueError, match="do not cover.*mtp"):
        protocol.build_model(
            SimpleNamespace(),
            impl_cfg=protocol.ImplConfig(
                mtp_enable=True,
                precision_coverage=PrecisionCoverage(implementation),
                precision_implementation=implementation,
                precision_parameter_contract=implementation.parameter_contract,
            ),
        )


def test_qwen3_moe_composition_declares_every_selected_and_fixed_bf16_site(
    transformer_engine_import_stub,
):
    transformer_engine_import_stub()
    from megatron.lite.model.qwen3_moe.lite.model import (
        _declare_model_precision_requirements,
    )
    from megatron.lite.primitive.precision import (
        PrimitiveCapability,
        SemanticSite,
    )

    layer = SimpleNamespace(
        attn=SimpleNamespace(
            qkv=object(),
            proj=object(),
            core_attn=object(),
            q_norm=object(),
            k_norm=object(),
        ),
        mlp_norm=object(),
        moe=SimpleNamespace(
            router=object(),
            experts=SimpleNamespace(fc1=object(), fc2=object()),
        ),
    )
    model = SimpleNamespace(
        layers=[layer],
        embed=object(),
        norm=object(),
        head=object(),
    )
    requirements = []

    class RecordingCoverage:
        def require(self, owner, site, capabilities=frozenset(), *, diagnostic=""):
            requirements.append((owner, site, capabilities, diagnostic))

    _declare_model_precision_requirements(model, RecordingCoverage())

    selected = [item for item in requirements if item[2]]
    fixed_bf16 = [item for item in requirements if not item[2]]
    assert [(item[0], item[1], item[2]) for item in selected] == [
        (
            layer.attn.qkv,
            SemanticSite.ATTENTION_PROJECTION,
            frozenset({PrimitiveCapability.TE_LAYERNORM_LINEAR}),
        ),
        (
            layer.attn.proj,
            SemanticSite.ATTENTION_PROJECTION,
            frozenset({PrimitiveCapability.TE_LINEAR}),
        ),
        (
            layer.moe.experts.fc1,
            SemanticSite.MOE_EXPERT,
            frozenset({PrimitiveCapability.TE_GROUPED_LINEAR}),
        ),
        (
            layer.moe.experts.fc2,
            SemanticSite.MOE_EXPERT,
            frozenset({PrimitiveCapability.TE_GROUPED_LINEAR}),
        ),
    ]
    assert [(item[0], item[1]) for item in fixed_bf16] == [
        (layer.attn.qkv, SemanticSite.NORM),
        (layer.attn.core_attn, SemanticSite.ATTENTION_CORE),
        (layer.attn.q_norm, SemanticSite.NORM),
        (layer.attn.k_norm, SemanticSite.NORM),
        (layer.mlp_norm, SemanticSite.NORM),
        (layer.moe.router, SemanticSite.ROUTER),
        (model.embed, SemanticSite.EMBEDDING),
        (model.norm, SemanticSite.NORM),
        (model.head, SemanticSite.LM_HEAD),
    ]


def test_qwen3_moe_seals_precision_coverage_before_optimizer_construction():
    lite_root = Path(__file__).resolve().parents[3]
    source = (
        lite_root / "megatron" / "lite" / "model" / "qwen3_moe" / "lite" / "protocol.py"
    ).read_text(encoding="utf-8")

    assert source.index("precision_coverage.seal()") < source.index(
        'if impl_cfg.optimizer == "dist_opt"'
    )
