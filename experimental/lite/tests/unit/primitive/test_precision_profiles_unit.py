# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from __future__ import annotations

import sys
import types
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.mlite


def test_precision_resolver_exposes_only_the_closed_profiles():
    from megatron.lite.primitive.precision import (
        AuthoritativeSource,
        MasterOwner,
        PRECISION_NAMES,
        PrecisionDType,
        SemanticSite,
        WeightStorage,
        resolve_precision,
    )

    assert PRECISION_NAMES == (
        "bf16",
        "hopper_blockwise_bf16_weight",
        "hopper_blockwise_fp8_weight",
    )
    assert resolve_precision("bf16") is None

    bf16_weight = resolve_precision("hopper_blockwise_bf16_weight")
    fp8_weight = resolve_precision("hopper_blockwise_fp8_weight")
    assert bf16_weight is not None
    assert fp8_weight is not None
    assert bf16_weight.recipe_factory is fp8_weight.recipe_factory
    assert bf16_weight.fp8_sites == frozenset(
        {
            SemanticSite.ATTENTION_PROJECTION,
            SemanticSite.DENSE_MLP,
            SemanticSite.MOE_EXPERT,
        }
    )
    assert bf16_weight.bf16_sites == frozenset(
        {
            SemanticSite.ATTENTION_CORE,
            SemanticSite.ROUTER,
            SemanticSite.NORM,
            SemanticSite.EMBEDDING,
            SemanticSite.LM_HEAD,
        }
    )
    assert bf16_weight.parameter_contract.compute_weight is WeightStorage.BF16
    assert (
        fp8_weight.parameter_contract.compute_weight is WeightStorage.FP8_BLOCKWISE_E4M3
    )
    for implementation in (bf16_weight, fp8_weight):
        contract = implementation.parameter_contract
        assert contract.authoritative_load_source is AuthoritativeSource.HIGH_PRECISION
        assert contract.master_parameter is PrecisionDType.FP32
        assert contract.main_gradient is PrecisionDType.FP32
        assert contract.optimizer_state is PrecisionDType.FP32
        assert contract.parameter_all_gather is PrecisionDType.BF16
        assert contract.master_owner is MasterOwner.MLITE_OPTIMIZER
        with pytest.raises(FrozenInstanceError):
            contract.master_parameter = PrecisionDType.BF16

    with pytest.raises(ValueError, match="bf16.*hopper_blockwise_bf16_weight"):
        resolve_precision("blockwise")
    with pytest.raises(TypeError, match="string"):
        resolve_precision(None)


def test_hopper_recipe_matches_the_frozen_transformer_engine_mapping(monkeypatch):
    from megatron.lite.primitive.precision import build_hopper_blockwise_recipe
    from megatron.lite.primitive.precision import hopper_blockwise

    class FakeFormat:
        E4M3 = "E4M3"

    class FakeMMParams:
        def __init__(self, *, use_split_accumulator):
            self.use_split_accumulator = use_split_accumulator

    class FakeFloat8BlockScaling:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
            self.fp8_quant_fwd_inp = SimpleNamespace(power_2_scale=True)
            self.fp8_quant_fwd_weight = SimpleNamespace(power_2_scale=True)
            self.fp8_quant_bwd_grad = SimpleNamespace(power_2_scale=True)

    te = types.ModuleType("transformer_engine")
    common = types.ModuleType("transformer_engine.common")
    recipe = types.ModuleType("transformer_engine.common.recipe")
    recipe.Float8BlockScaling = FakeFloat8BlockScaling
    recipe.Format = FakeFormat
    recipe.MMParams = FakeMMParams
    te.common = common
    common.recipe = recipe
    monkeypatch.setitem(sys.modules, "transformer_engine", te)
    monkeypatch.setitem(sys.modules, "transformer_engine.common", common)
    monkeypatch.setitem(sys.modules, "transformer_engine.common.recipe", recipe)
    monkeypatch.delenv("NVTE_FP8_BLOCK_SCALING_FP32_SCALES", raising=False)
    monkeypatch.delenv("NVTE_BACKWARD_OVERRIDE", raising=False)

    hopper_blockwise._build_hopper_blockwise_recipe.cache_clear()
    built = build_hopper_blockwise_recipe()
    try:
        assert built.fp8_format == "E4M3"
        assert built.use_f32_scales is False
        assert built.x_block_scaling_dim == 1
        assert built.w_block_scaling_dim == 2
        assert built.grad_block_scaling_dim == 1
        assert built.fp8_gemm_fprop.use_split_accumulator is True
        assert built.fp8_gemm_dgrad.use_split_accumulator is True
        assert built.fp8_gemm_wgrad.use_split_accumulator is True
        assert built.fp8_dpa is False
        assert built.fp8_mha is False
        assert built.backward_override is None
        assert build_hopper_blockwise_recipe() is built
        built.fp8_mha = True
        with pytest.raises(RuntimeError, match="frozen contract"):
            build_hopper_blockwise_recipe()
    finally:
        hopper_blockwise._build_hopper_blockwise_recipe.cache_clear()


def _valid_hopper_environment(**overrides):
    values = {
        "compute_capability": (9, 0),
        "transformer_engine_version": "2.15.0+42b84005",
        "cuda_version": (12, 9),
        "cublas_version": 130400,
        "block_scaling_supported": True,
        "unsupported_reason": "",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"compute_capability": (8, 9)}, "SM90"),
        ({"compute_capability": (10, 0)}, "SM90"),
        ({"transformer_engine_version": "2.17.0"}, "2.15.0"),
        ({"transformer_engine_version": "2.18.0.dev0+8b99682"}, "2.15.0"),
        ({"cuda_version": (12, 8)}, "CUDA 12.9"),
        ({"cublas_version": 130300}, "cuBLAS 13.4"),
        (
            {
                "block_scaling_supported": False,
                "unsupported_reason": "kernel unavailable",
            },
            "kernel unavailable",
        ),
    ],
)
def test_hopper_environment_fails_loud_on_reference_mismatch(
    monkeypatch, overrides, message
):
    from megatron.lite.primitive.precision import hopper_blockwise

    monkeypatch.setattr(
        hopper_blockwise,
        "_probe_hopper_environment",
        lambda: _valid_hopper_environment(**overrides),
    )
    monkeypatch.delenv("NVTE_FP8_BLOCK_SCALING_FP32_SCALES", raising=False)
    monkeypatch.delenv("NVTE_BACKWARD_OVERRIDE", raising=False)

    with pytest.raises(RuntimeError, match=message):
        hopper_blockwise.validate_hopper_environment()


@pytest.mark.parametrize(
    "version",
    [
        # The released version is the contract; any build tag on it is accepted
        # so FP8 needs no bespoke build -- it runs on the canonical BF16 image.
        "2.15.0",
        "2.15.0+42b84005",
        "2.15.0+deadbeef",
    ],
)
def test_hopper_environment_accepts_the_canonical_reference(monkeypatch, version):
    from megatron.lite.primitive.precision import hopper_blockwise

    environment = _valid_hopper_environment(transformer_engine_version=version)
    monkeypatch.setattr(
        hopper_blockwise, "_probe_hopper_environment", lambda: environment
    )
    monkeypatch.delenv("NVTE_FP8_BLOCK_SCALING_FP32_SCALES", raising=False)
    monkeypatch.delenv("NVTE_BACKWARD_OVERRIDE", raising=False)

    assert hopper_blockwise.validate_hopper_environment() is environment


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("NVTE_FP8_BLOCK_SCALING_FP32_SCALES", "1"),
        ("NVTE_BACKWARD_OVERRIDE", "high_precision"),
    ],
)
def test_hopper_environment_rejects_recipe_overrides(monkeypatch, name, value):
    from megatron.lite.primitive.precision import hopper_blockwise

    monkeypatch.setattr(
        hopper_blockwise, "_probe_hopper_environment", _valid_hopper_environment
    )
    monkeypatch.setenv(name, value)

    with pytest.raises(RuntimeError, match=name):
        hopper_blockwise.validate_hopper_environment()


def _coverage_types():
    from megatron.lite.primitive.precision import (
        PrecisionCoverage,
        PrimitiveCapability,
        SemanticSite,
        resolve_precision,
    )

    implementation = resolve_precision("hopper_blockwise_bf16_weight")
    assert implementation is not None
    return implementation, PrecisionCoverage, PrimitiveCapability, SemanticSite


def _install_te_witnesses(transformer_engine_import_stub, monkeypatch):
    """Install stub TE classes and return real instances usable as claim witnesses.

    ``PrecisionCoverage.claim`` binds each capability to a genuine TE primitive
    instance, so coverage-algebra tests must present real TE-typed witnesses
    rather than bare ``object()`` claims.
    """

    transformer_engine_import_stub()
    import transformer_engine.pytorch as te

    class _FakeTEModule:
        def __init__(self, *args, **kwargs):
            pass

    linear_type = type("FakeTELinear", (_FakeTEModule,), {})
    layernorm_type = type("FakeTELayerNormLinear", (_FakeTEModule,), {})
    grouped_type = type("FakeTEGroupedLinear", (_FakeTEModule,), {})
    monkeypatch.setattr(te, "Linear", linear_type, raising=False)
    monkeypatch.setattr(te, "LayerNormLinear", layernorm_type, raising=False)
    monkeypatch.setattr(te, "GroupedLinear", grouped_type, raising=False)
    return SimpleNamespace(
        linear=linear_type(),
        layernorm_linear=layernorm_type(),
        grouped=grouped_type(),
    )


def test_typed_coverage_seals_exact_selected_sites_and_bf16_exclusions(
    transformer_engine_import_stub, monkeypatch
):
    from megatron.lite.primitive.precision import precision_model_init_context

    witnesses = _install_te_witnesses(transformer_engine_import_stub, monkeypatch)
    implementation, PrecisionCoverage, Capability, Site = _coverage_types()
    attention_projection = object()
    attention_core = object()
    router = object()
    coverage = PrecisionCoverage(implementation)

    with precision_model_init_context(implementation):
        coverage.require(
            attention_projection,
            Site.ATTENTION_PROJECTION,
            frozenset({Capability.TE_LINEAR, Capability.TE_LAYERNORM_LINEAR}),
            diagnostic="attention projection 0",
        )
        coverage.claim(
            attention_projection,
            Site.ATTENTION_PROJECTION,
            Capability.TE_LINEAR,
            witnesses.linear,
            diagnostic="TE column parallel linear",
        )
        coverage.require(
            attention_core, Site.ATTENTION_CORE, diagnostic="attention core 0"
        )
        coverage.require(router, Site.ROUTER, diagnostic="router 0")
        manifest = coverage.seal()

    assert manifest.implementation_name == implementation.name
    assert [entry.site for entry in manifest.entries] == [
        Site.ATTENTION_PROJECTION,
        Site.ATTENTION_CORE,
        Site.ROUTER,
    ]
    assert manifest.entries[0].capability is Capability.TE_LINEAR
    assert manifest.entries[1].capability is None
    assert coverage.manifest is manifest

    with pytest.raises(RuntimeError, match="sealed"):
        coverage.require(object(), Site.NORM, diagnostic="late norm")


def test_typed_coverage_requires_the_runtime_model_init_context():
    implementation, PrecisionCoverage, _, Site = _coverage_types()
    coverage = PrecisionCoverage(implementation)

    with pytest.raises(RuntimeError, match="model-init"):
        coverage.require(object(), Site.ATTENTION_CORE)


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("missing", "missing"),
        ("duplicate_requirement", "duplicate requirement"),
        ("duplicate_claim", "duplicate claim"),
        ("incompatible", "incompatible"),
        ("unconsumed", "unconsumed"),
        ("bf16_claim", "fixed BF16"),
    ],
)
def test_typed_coverage_fails_loud_for_incomplete_or_ambiguous_binding(
    case, message, transformer_engine_import_stub, monkeypatch
):
    from megatron.lite.primitive.precision import precision_model_init_context

    witnesses = _install_te_witnesses(transformer_engine_import_stub, monkeypatch)
    implementation, PrecisionCoverage, Capability, Site = _coverage_types()
    covered = object()
    coverage = PrecisionCoverage(implementation)

    with precision_model_init_context(implementation):
        coverage.require(
            covered,
            Site.DENSE_MLP,
            frozenset({Capability.TE_LINEAR}),
            diagnostic="dense mlp 0",
        )
        if case == "duplicate_requirement":
            coverage.require(
                covered,
                Site.DENSE_MLP,
                frozenset({Capability.TE_LINEAR}),
                diagnostic="dense mlp duplicate",
            )
        elif case == "duplicate_claim":
            coverage.claim(covered, Site.DENSE_MLP, Capability.TE_LINEAR, witnesses.linear)
            coverage.claim(covered, Site.DENSE_MLP, Capability.TE_LINEAR, witnesses.linear)
        elif case == "incompatible":
            coverage.claim(
                covered, Site.DENSE_MLP, Capability.TE_GROUPED_LINEAR, witnesses.grouped
            )
        elif case == "unconsumed":
            coverage.claim(covered, Site.DENSE_MLP, Capability.TE_LINEAR, witnesses.linear)
            coverage.claim(
                object(), Site.MOE_EXPERT, Capability.TE_GROUPED_LINEAR, witnesses.grouped
            )
        elif case == "bf16_claim":
            coverage.claim(covered, Site.DENSE_MLP, Capability.TE_LINEAR, witnesses.linear)
            core = object()
            coverage.require(core, Site.ATTENTION_CORE)
            coverage.claim(core, Site.ATTENTION_CORE, Capability.TE_LINEAR, witnesses.linear)

        with pytest.raises(ValueError, match=message):
            coverage.seal()


def test_precision_package_stays_narrow_and_model_agnostic():
    root = Path(__file__).resolve().parents[3]
    precision_root = root / "megatron" / "lite" / "primitive" / "precision"
    assert {path.name for path in precision_root.glob("*.py")} == {
        "__init__.py",
        "contract.py",
        "coverage.py",
        "hopper_blockwise.py",
    }
    source = "\n".join(
        path.read_text(encoding="utf-8") for path in sorted(precision_root.glob("*.py"))
    ).lower()
    for forbidden in (
        "megatron.lite.model",
        "qwen",
        "deepseek",
        "glm",
        "kimi",
        "fnmatch",
        "glob(",
        "re.compile",
    ):
        assert forbidden not in source
    assert list((root / "megatron" / "lite" / "model").glob("*/fp8.py")) == []
