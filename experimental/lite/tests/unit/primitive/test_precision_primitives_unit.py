# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

pytestmark = pytest.mark.mlite


class _FakeLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, **kwargs):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.kwargs = kwargs
        self.weight = nn.Parameter(torch.empty(out_features, in_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros(*x.shape[:-1], self.out_features, dtype=x.dtype)


def _parallel_state(**overrides):
    values = {
        "tp_size": 1,
        "tp_rank": 0,
        "tp_group": None,
        "ep_size": 1,
        "etp_size": 1,
        "etp_group": None,
        "cp_size": 1,
        "cp_group": None,
        "cp_global_ranks": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_precision_site_context_opens_only_the_selected_te_scope(
    transformer_engine_import_stub, monkeypatch
):
    transformer_engine_import_stub()
    import transformer_engine.pytorch as te
    from megatron.lite.primitive.precision import (
        SemanticSite,
        precision_forward_context,
        precision_site_forward_context,
        resolve_precision,
    )

    recipe = object()
    base = resolve_precision("hopper_blockwise_bf16_weight")
    assert base is not None
    implementation = replace(base, recipe_factory=lambda: recipe)
    calls = []

    @contextmanager
    def fake_fp8_autocast(**kwargs):
        calls.append(("enter", kwargs))
        yield
        calls.append(("exit", kwargs))

    monkeypatch.setattr(te, "fp8_autocast", fake_fp8_autocast, raising=False)

    with precision_forward_context(implementation):
        with precision_site_forward_context(
            implementation, SemanticSite.ATTENTION_PROJECTION
        ):
            calls.append(("body", None))

    assert calls == [
        ("enter", {"enabled": True, "fp8_recipe": recipe, "fp8_group": None}),
        ("body", None),
        ("exit", {"enabled": True, "fp8_recipe": recipe, "fp8_group": None}),
    ]
    with pytest.raises(ValueError, match="fixed BF16"):
        with precision_forward_context(implementation):
            precision_site_forward_context(implementation, SemanticSite.ATTENTION_CORE)
    with pytest.raises(RuntimeError, match="forward context"):
        precision_site_forward_context(implementation, SemanticSite.DENSE_MLP)


def test_tp_te_linear_primitives_claim_capability_and_enter_scoped_context(
    transformer_engine_import_stub, monkeypatch
):
    transformer_engine_import_stub()
    from megatron.lite.primitive.parallel import linear
    from megatron.lite.primitive.precision import (
        PrecisionCoverage,
        PrimitiveCapability,
        SemanticSite,
        precision_forward_context,
        precision_model_init_context,
        resolve_precision,
    )

    monkeypatch.setattr(linear.te, "Linear", _FakeLinear)
    monkeypatch.setattr(linear.te, "LayerNormLinear", _FakeLinear)
    implementation = resolve_precision("hopper_blockwise_bf16_weight")
    assert implementation is not None
    coverage = PrecisionCoverage(implementation)

    with precision_model_init_context(implementation):
        qkv = linear.ColumnParallelLinear(
            128,
            128,
            _parallel_state(),
            normalization="RMSNorm",
            precision_coverage=coverage,
            precision_site=SemanticSite.ATTENTION_PROJECTION,
        )
        proj = linear.RowParallelLinear(
            128,
            128,
            _parallel_state(),
            precision_coverage=coverage,
            precision_site=SemanticSite.ATTENTION_PROJECTION,
        )
        coverage.require(
            qkv,
            SemanticSite.ATTENTION_PROJECTION,
            frozenset({PrimitiveCapability.TE_LAYERNORM_LINEAR}),
        )
        coverage.require(
            proj,
            SemanticSite.ATTENTION_PROJECTION,
            frozenset({PrimitiveCapability.TE_LINEAR}),
        )
        manifest = coverage.seal()

    assert [entry.capability for entry in manifest.entries] == [
        PrimitiveCapability.TE_LAYERNORM_LINEAR,
        PrimitiveCapability.TE_LINEAR,
    ]
    scopes = []

    @contextmanager
    def record_scope(bound_implementation, site):
        scopes.append((bound_implementation, site))
        yield

    monkeypatch.setattr(linear, "precision_site_forward_context", record_scope)
    x = torch.zeros(128, 128)
    with precision_forward_context(implementation):
        qkv(x)
        proj(x)

    assert scopes == [
        (implementation, SemanticSite.ATTENTION_PROJECTION),
        (implementation, SemanticSite.ATTENTION_PROJECTION),
    ]


def test_bf16_weight_linear_binding_fails_loud_for_shape_or_fp8_weight_profile(
    transformer_engine_import_stub, monkeypatch
):
    transformer_engine_import_stub()
    from megatron.lite.primitive.parallel import linear
    from megatron.lite.primitive.precision import (
        PrecisionCoverage,
        SemanticSite,
        precision_model_init_context,
        resolve_precision,
    )

    monkeypatch.setattr(linear.te, "Linear", _FakeLinear)
    bf16_weight = resolve_precision("hopper_blockwise_bf16_weight")
    fp8_weight = resolve_precision("hopper_blockwise_fp8_weight")
    assert bf16_weight is not None and fp8_weight is not None

    with precision_model_init_context(bf16_weight):
        with pytest.raises(ValueError, match="divisible by 128"):
            linear.ColumnParallelLinear(
                64,
                128,
                _parallel_state(),
                precision_coverage=PrecisionCoverage(bf16_weight),
                precision_site=SemanticSite.ATTENTION_PROJECTION,
            )
    with precision_model_init_context(fp8_weight):
        with pytest.raises(NotImplementedError, match="FP8-weight"):
            linear.ColumnParallelLinear(
                128,
                128,
                _parallel_state(),
                precision_coverage=PrecisionCoverage(fp8_weight),
                precision_site=SemanticSite.ATTENTION_PROJECTION,
            )


def test_unbound_te_linear_preserves_bf16_path_without_precision_context(
    transformer_engine_import_stub, monkeypatch
):
    transformer_engine_import_stub()
    from megatron.lite.primitive.parallel import linear

    monkeypatch.setattr(linear.te, "Linear", _FakeLinear)

    def unexpected_context(*args, **kwargs):
        raise AssertionError("the default BF16 path must not enter FP8 autocast")

    monkeypatch.setattr(linear, "precision_site_forward_context", unexpected_context)
    module = linear.ColumnParallelLinear(128, 128, _parallel_state())

    assert module(torch.zeros(4, 128)).shape == (4, 128)


def test_non_te_dense_requirement_fails_coverage_instead_of_falling_back():
    from megatron.lite.primitive.precision import (
        PrecisionCoverage,
        PrimitiveCapability,
        SemanticSite,
        precision_model_init_context,
        resolve_precision,
    )

    implementation = resolve_precision("hopper_blockwise_bf16_weight")
    assert implementation is not None
    coverage = PrecisionCoverage(implementation)
    inline_torch_linear = nn.Linear(128, 128, bias=False)
    with precision_model_init_context(implementation):
        coverage.require(
            inline_torch_linear,
            SemanticSite.DENSE_MLP,
            frozenset({PrimitiveCapability.TE_LINEAR}),
            diagnostic="inline torch linear",
        )
        with pytest.raises(ValueError, match="missing claim.*inline torch linear"):
            coverage.seal()


def test_precision_coverage_rejects_forged_capability_from_non_te_owner(
    transformer_engine_import_stub, monkeypatch
):
    """A non-TE module cannot masquerade as an FP8-capable GEMM primitive.

    Regression for the claim-forgery gap: a bare ``nn.Linear`` could previously
    ``require`` an FP8 site and ``claim`` a TE capability, and ``seal`` returned
    a manifest instead of failing loud. The claim now binds the capability to a
    genuine TE primitive instance and rejects any other owner/witness.
    """

    transformer_engine_import_stub()
    import transformer_engine.pytorch as te

    # A distinct, constructible TE Linear so the *legitimate* path still seals,
    # proving the guard rejects only the non-TE owner, not every claim.
    te_linear_type = type("FakeTELinear", (object,), {"__init__": lambda self: None})
    monkeypatch.setattr(te, "Linear", te_linear_type, raising=False)

    from megatron.lite.primitive.precision import (
        PrecisionCoverage,
        PrimitiveCapability,
        SemanticSite,
        precision_model_init_context,
        resolve_precision,
    )

    implementation = resolve_precision("hopper_blockwise_bf16_weight")
    assert implementation is not None
    coverage = PrecisionCoverage(implementation)
    forged_owner = nn.Linear(128, 128, bias=False)

    with precision_model_init_context(implementation):
        coverage.require(
            forged_owner,
            SemanticSite.DENSE_MLP,
            frozenset({PrimitiveCapability.TE_LINEAR}),
            diagnostic="forged non-te linear",
        )
        # Owner-as-witness (default) is rejected: the owner is a raw nn.Linear.
        with pytest.raises(TypeError, match="must be backed by a real"):
            coverage.claim(
                forged_owner, SemanticSite.DENSE_MLP, PrimitiveCapability.TE_LINEAR
            )
        # An explicit non-TE witness is rejected as well.
        with pytest.raises(TypeError, match="must be backed by a real"):
            coverage.claim(
                forged_owner,
                SemanticSite.DENSE_MLP,
                PrimitiveCapability.TE_LINEAR,
                forged_owner,
            )
        # The genuine TE primitive witness is accepted for the same site.
        coverage.claim(
            forged_owner,
            SemanticSite.DENSE_MLP,
            PrimitiveCapability.TE_LINEAR,
            te_linear_type(),
        )
        manifest = coverage.seal()

    assert [entry.capability for entry in manifest.entries] == [
        PrimitiveCapability.TE_LINEAR
    ]


def test_gqa_binds_both_projection_linears_to_attention_projection(
    transformer_engine_import_stub, monkeypatch
):
    transformer_engine_import_stub()
    from megatron.lite.primitive.modules import gqa
    from megatron.lite.primitive.precision import SemanticSite

    calls = []

    class FakeProjection(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            calls.append(kwargs)
            self.local_in = args[0]
            self.local_out = args[1]
            self.use_sp = False

    class FakeModule(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

    monkeypatch.setattr(gqa, "ColumnParallelLinear", FakeProjection)
    monkeypatch.setattr(gqa, "RowParallelLinear", FakeProjection)
    monkeypatch.setattr(gqa.te, "RMSNorm", FakeModule)
    monkeypatch.setattr(gqa.te, "DotProductAttention", FakeModule)
    monkeypatch.setattr(gqa, "RotaryEmbedding", FakeModule)
    coverage = object()

    gqa.GQAttention(
        hidden_size=128,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=64,
        ps=_parallel_state(),
        precision_coverage=coverage,
    )

    assert len(calls) == 2
    for kwargs in calls:
        assert kwargs["precision_coverage"] is coverage
        assert kwargs["precision_site"] is SemanticSite.ATTENTION_PROJECTION


def test_dense_mlp_uses_two_precision_aware_te_linears(
    transformer_engine_import_stub, monkeypatch
):
    transformer_engine_import_stub()
    from megatron.lite.primitive.modules import mlp
    from megatron.lite.primitive.precision import (
        PrecisionCoverage,
        PrimitiveCapability,
        SemanticSite,
        precision_forward_context,
        precision_model_init_context,
        resolve_precision,
    )

    monkeypatch.setattr(mlp.te, "Linear", _FakeLinear)
    monkeypatch.setattr(
        mlp,
        "swiglu_with_probs",
        lambda y, probs, swiglu_limit=0.0: y[..., : y.shape[-1] // 2],
    )
    implementation = resolve_precision("hopper_blockwise_bf16_weight")
    assert implementation is not None
    coverage = PrecisionCoverage(implementation)
    with precision_model_init_context(implementation):
        module = mlp.SwiGLUMLP(128, 128, precision_coverage=coverage)
        for owner in (module.gate_up, module.down):
            coverage.require(
                owner,
                SemanticSite.DENSE_MLP,
                frozenset({PrimitiveCapability.TE_LINEAR}),
            )
        manifest = coverage.seal()

    assert isinstance(module.gate_up, _FakeLinear)
    assert isinstance(module.down, _FakeLinear)
    assert not hasattr(module.gate_up, "linear")
    assert set(module.state_dict()) == {"gate_up.weight", "down.weight"}
    assert len(manifest.entries) == 2
    scopes = []

    @contextmanager
    def record_scope(bound_implementation, site):
        scopes.append((bound_implementation, site))
        yield

    monkeypatch.setattr(mlp, "precision_site_forward_context", record_scope)
    with precision_forward_context(implementation):
        output = module(torch.zeros(128, 128))
    assert output.shape == (128, 128)
    assert scopes == [
        (implementation, SemanticSite.DENSE_MLP),
        (implementation, SemanticSite.DENSE_MLP),
    ]


def test_moe_experts_claim_grouped_linears_and_use_blockwise_padding(
    transformer_engine_import_stub, monkeypatch
):
    transformer_engine_import_stub()
    from megatron.lite.primitive.modules import experts
    from megatron.lite.primitive.precision import (
        PrecisionCoverage,
        PrimitiveCapability,
        SemanticSite,
        precision_forward_context,
        precision_model_init_context,
        resolve_precision,
    )

    class FakeGroupedLinear(nn.Module):
        def __init__(self, groups, in_features, out_features, **kwargs):
            super().__init__()
            self.out_features = out_features

        def forward(self, x, m_splits):
            return torch.zeros(x.shape[0], self.out_features, dtype=x.dtype)

    monkeypatch.setattr(experts.te, "GroupedLinear", FakeGroupedLinear, raising=False)
    monkeypatch.setattr(
        experts,
        "swiglu_with_probs",
        lambda y, probs, swiglu_limit=0.0: y[..., : y.shape[-1] // 2],
    )
    implementation = resolve_precision("hopper_blockwise_bf16_weight")
    assert implementation is not None
    coverage = PrecisionCoverage(implementation)
    config = SimpleNamespace(
        num_experts=2,
        hidden_size=128,
        moe_intermediate_size=128,
        swiglu_limit=0.0,
    )

    with precision_model_init_context(implementation):
        module = experts.Experts(
            config,
            _parallel_state(),
            precision_coverage=coverage,
        )
        for owner in (module.fc1, module.fc2):
            coverage.require(
                owner,
                SemanticSite.MOE_EXPERT,
                frozenset({PrimitiveCapability.TE_GROUPED_LINEAR}),
            )
        coverage.seal()

    scopes = []

    @contextmanager
    def record_scope(bound_implementation, site):
        scopes.append((bound_implementation, site))
        yield

    monkeypatch.setattr(experts, "precision_site_forward_context", record_scope)
    x = torch.ones(2, 128)
    with precision_forward_context(implementation):
        output = module(x, torch.tensor([1, 1]))

    assert output.shape == (2, 128)
    assert scopes == [
        (implementation, SemanticSite.MOE_EXPERT),
        (implementation, SemanticSite.MOE_EXPERT),
    ]
